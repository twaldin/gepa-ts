import { describe, expect, it } from 'vitest';
import { ExperimentTracker, Logger, create_experiment_tracker, log_detailed_metrics_after_discovering_new_program } from '../../src/logging/index.js';

describe('ExperimentTracker', () => {
  it('preserves upstream-shaped constructor flags and factory', () => {
    const tracker = create_experiment_tracker({
      use_wandb: true,
      wandb_api_key: 'key',
      wandb_init_kwargs: { project: 'gepa' },
      wandb_attach_existing: true,
      wandb_step_metric: 'iteration',
      use_mlflow: true,
      mlflow_tracking_uri: 'file:///tmp/mlflow',
      mlflow_experiment_name: 'exp',
      mlflow_attach_existing: true,
      key_prefix: 'gepa/',
    });

    expect(tracker).toBeInstanceOf(ExperimentTracker);
    expect(tracker.use_wandb).toBe(true);
    expect(tracker.wandb_api_key).toBe('key');
    expect(tracker.wandb_init_kwargs).toEqual({ project: 'gepa' });
    expect(tracker.wandb_attach_existing).toBe(true);
    expect(tracker.wandb_step_metric).toBe('iteration');
    expect(tracker.use_mlflow).toBe(true);
    expect(tracker.mlflow_tracking_uri).toBe('file:///tmp/mlflow');
    expect(tracker.mlflow_experiment_name).toBe('exp');
    expect(tracker.mlflow_attach_existing).toBe(true);
  });

  it('normalizes config, numeric metrics, tables, and summaries with key_prefix', () => {
    const tracker = new ExperimentTracker({ key_prefix: 'gepa/' });

    tracker.log_config({ seed: 0, nested: { a: 1 } });
    tracker.log_metrics({ score: 0.5, label: 'skip', nan: Number.NaN }, 3);
    tracker.log_table('valset_scores', ['candidate', 'score'], [[0, 0.5]]);
    tracker.log_summary({ best_idx: 0 });
    tracker.log_html('<html/>', 'candidate_tree');

    expect(tracker.logged_configs).toEqual({ 'gepa/seed': 0, 'gepa/nested': '[object Object]' });
    expect(tracker.logged_metrics).toEqual([{ metrics: { 'gepa/score': 0.5 }, step: 3 }]);
    expect(tracker.logged_tables).toEqual([
      { key: 'gepa/valset_scores', columns: ['candidate', 'score'], data: [[0, 0.5]] },
    ]);
    expect(tracker.logged_summaries).toEqual([
      { key: 'gepa/best_idx', value: 0 },
      { key: 'gepa/candidate_tree', value: '<html/>' },
    ]);
  });

  it('tracks context-style run lifecycle without suppressing errors', () => {
    const tracker = new ExperimentTracker();

    expect(tracker.enter()).toBe(tracker);
    expect(tracker.initialized).toBe(true);
    expect(tracker.run_started).toBe(true);
    expect(tracker.exit()).toBe(false);
    expect(tracker.run_ended).toBe(true);
  });

  it('forwards upstream-shaped W&B calls to an injected zero-dependency client', () => {
    const calls: Array<{ method: string; args: unknown[] }> = [];
    const run = { summary: {} as Record<string, unknown> };
    const wandb_client = {
      run,
      config: {
        update: (config: Record<string, unknown>, options?: Record<string, unknown>) => {
          calls.push({ method: 'config.update', args: [config, options] });
        },
      },
      login: (options?: Record<string, unknown>) => calls.push({ method: 'login', args: [options] }),
      init: (options?: Record<string, unknown>) => {
        calls.push({ method: 'init', args: [options] });
        return run;
      },
      finish: () => calls.push({ method: 'finish', args: [] }),
      define_metric: (name: string, options?: Record<string, unknown>) => {
        calls.push({ method: 'define_metric', args: [name, options] });
      },
      Table: (options: { columns: string[]; data: unknown[][] }) => ({ table: options }),
      Html: (html: string) => ({ html }),
      log: (data: Record<string, unknown>, options?: Record<string, unknown>) => {
        calls.push({ method: 'log', args: [data, options] });
      },
    };

    const tracker = new ExperimentTracker({
      use_wandb: true,
      wandb_api_key: 'key',
      wandb_init_kwargs: { project: 'gepa' },
      wandb_step_metric: 'gepa_step',
      key_prefix: 'gepa/',
      wandb_client,
    });

    tracker.enter();
    tracker.log_config({ seed: 1, nested: { a: 1 } });
    tracker.log_metrics({ score: 0.75, label: 'skip' }, 4);
    tracker.log_table('scores', ['candidate', 'score'], [[0, 0.75]]);
    tracker.log_summary({ best: 0.75 });
    tracker.log_html('<b>tree</b>', 'tree');
    tracker.exit();

    expect(calls.map((call) => call.method)).toEqual([
      'login',
      'init',
      'config.update',
      'define_metric',
      'define_metric',
      'log',
      'log',
      'log',
      'finish',
    ]);
    expect(calls[0]?.args[0]).toEqual({ key: 'key', verify: true });
    expect(calls[2]?.args[0]).toEqual({ 'gepa/seed': 1, 'gepa/nested': '[object Object]' });
    expect(calls[5]?.args[0]).toEqual({ 'gepa/score': 0.75, gepa_step: 4 });
    expect(run.summary['gepa/best']).toBe(0.75);
    expect(run.summary['gepa/tree']).toEqual({ html: '<b>tree</b>' });
  });

  it('forwards upstream-shaped MLflow calls to an injected zero-dependency client', () => {
    const calls: Array<{ method: string; args: unknown[] }> = [];
    const active_run = { info: { run_id: 'run-1' } };
    let current_run: typeof active_run | null = null;
    const mlflow_client = {
      set_tracking_uri: (uri: string) => calls.push({ method: 'set_tracking_uri', args: [uri] }),
      set_experiment: (name: string) => calls.push({ method: 'set_experiment', args: [name] }),
      active_run: () => current_run,
      start_run: () => {
        calls.push({ method: 'start_run', args: [] });
        current_run = active_run;
        return active_run;
      },
      end_run: () => {
        calls.push({ method: 'end_run', args: [] });
        current_run = null;
      },
      log_param: (run_id: string, key: string, value: string) => calls.push({ method: 'log_param', args: [run_id, key, value] }),
      log_metric: (run_id: string, key: string, value: number, step?: number) => calls.push({ method: 'log_metric', args: [run_id, key, value, step] }),
      log_table: (options: { data: Record<string, unknown[]>; artifact_file: string }) => calls.push({ method: 'log_table', args: [options] }),
      log_artifact: (path: string, artifact_path?: string) => calls.push({ method: 'log_artifact', args: [path, artifact_path] }),
    };

    const tracker = new ExperimentTracker({
      use_mlflow: true,
      mlflow_tracking_uri: 'file:///tmp/mlruns',
      mlflow_experiment_name: 'exp',
      key_prefix: 'gepa/',
      mlflow_client,
    });

    tracker.enter();
    tracker.log_config({ seed: 2 });
    tracker.log_metrics({ score: 0.5 }, 3);
    tracker.log_summary({ best: 0.5, label: 'done' });
    tracker.log_table('scores', ['candidate', 'score'], [[0, 0.5]]);
    tracker.log_html('<html/>', 'tree');
    tracker.exit();

    expect(calls.map((call) => call.method)).toEqual([
      'set_tracking_uri',
      'set_experiment',
      'start_run',
      'log_param',
      'log_metric',
      'log_metric',
      'log_param',
      'log_table',
      'log_artifact',
      'end_run',
    ]);
    expect(calls[3]?.args).toEqual(['run-1', 'gepa/seed', '2']);
    expect(calls[4]?.args).toEqual(['run-1', 'gepa/score', 0.5, 3]);
    expect(calls[6]?.args).toEqual(['run-1', 'summary/gepa/label', 'done']);
  });
});

describe('log_detailed_metrics_after_discovering_new_program', () => {
  it('logs upstream-shaped metrics and tables for a discovered candidate', async () => {
    const tracker = new ExperimentTracker();
    const logger = new Logger();
    const state = {
      i: 0,
      total_num_evals: 9,
      program_full_scores_val_set: [0.25, 0.75],
      pareto_front_valset: new Map<string, number>([
        ['a', 0.5],
        ['b', 1],
      ]),
      program_at_pareto_front_valset: new Map<string, Set<number>>([
        ['a', new Set([0, 1])],
        ['b', new Set([1])],
      ]),
      objective_pareto_front: new Map<string, number>([['quality', 0.9]]),
      program_at_pareto_front_objectives: new Map<string, Set<number>>([['quality', new Set([1])]]),
      prog_candidate_val_subscores: [
        new Map<string, number>([['a', 0.25]]),
        new Map<string, number>([
          ['a', 0.5],
          ['b', 1],
        ]),
      ],
      prog_candidate_objective_scores: [{}, { quality: 0.9 }],
      parent_program_for_candidate: [[null], [0]],
    };
    const policy = {
      get_eval_batch: async () => ['a', 'b'],
      get_best_program: async () => 1,
      get_valset_score: async (program_idx: number) => (program_idx === 1 ? 0.75 : 0.25),
    };

    await log_detailed_metrics_after_discovering_new_program({
      logger,
      gepa_state: state,
      new_program_idx: 1,
      valset_evaluation: {
        scores_by_val_id: new Map<string, number>([
          ['a', 0.5],
          ['b', 1],
        ]),
      },
      objective_scores: { quality: 0.9 },
      experiment_tracker: tracker,
      linear_pareto_front_program_idx: 1,
      valset_size: 2,
      val_evaluation_policy: policy,
    });

    expect(logger.messages.some((message) => message.includes('Valset score for new program: 0.75'))).toBe(true);
    expect(tracker.logged_metrics[0]?.metrics).toMatchObject({
      iteration: 1,
      new_program_idx: 1,
      val_program_average: 0.75,
      'objective/quality': 0.9,
    });
    expect(tracker.logged_tables.map((table) => table.key)).toEqual([
      'valset_scores',
      'valset_pareto_front',
      'objective_scores',
      'objective_pareto_front',
    ]);
  });
});
