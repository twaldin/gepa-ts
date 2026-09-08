export type LoggedTable = {
  key: string;
  columns: string[];
  data: unknown[][];
};

export type LoggedSummary = {
  key: string;
  value: unknown;
};

export type WandbRunLike = {
  summary?: Record<string, unknown>;
};

export type WandbClientLike = {
  run?: WandbRunLike | null;
  config?: {
    update(config: Record<string, unknown>, options?: Record<string, unknown>): void;
  };
  login?(options?: Record<string, unknown>): void;
  init?(options?: Record<string, unknown>): WandbRunLike | void;
  finish?(): void;
  log?(data: Record<string, unknown>, options?: Record<string, unknown>): void;
  define_metric?(name: string, options?: Record<string, unknown>): void;
  Table?(options: { columns: string[]; data: unknown[][] }): unknown;
  Html?(html: string): unknown;
};

export type MlflowRunLike = {
  info?: {
    run_id?: string;
  };
};

export type MlflowClientLike = {
  set_tracking_uri?(uri: string): void;
  get_tracking_uri?(): string | null;
  set_experiment?(name: string): void;
  active_run?(): MlflowRunLike | null;
  start_run?(): MlflowRunLike | void;
  end_run?(): void;
  log_params?(params: Record<string, string>): void;
  log_metrics?(metrics: Record<string, number>, step?: number): void;
  log_param?(run_id: string, key: string, value: string): void;
  log_metric?(run_id: string, key: string, value: number, step?: number): void;
  log_table?(options: { data: Record<string, unknown[]>; artifact_file: string }): void;
  log_artifact?(path: string, artifact_path?: string): void;
};

export interface ExperimentTrackerOptions {
  use_wandb?: boolean;
  wandb_api_key?: string | null;
  wandb_init_kwargs?: Record<string, unknown> | null;
  wandb_attach_existing?: boolean;
  wandb_step_metric?: string | null;
  use_mlflow?: boolean;
  mlflow_tracking_uri?: string | null;
  mlflow_experiment_name?: string | null;
  mlflow_attach_existing?: boolean;
  key_prefix?: string;
  wandb_client?: WandbClientLike | null;
  mlflow_client?: MlflowClientLike | null;
}

function is_scalar_config_value(value: unknown): value is boolean | number | string | null {
  return value === null || typeof value === 'boolean' || typeof value === 'number' || typeof value === 'string';
}

function is_numeric_value(value: unknown): value is number {
  return typeof value === 'number' && Number.isFinite(value);
}

export class ExperimentTracker {
  readonly use_wandb: boolean;
  readonly wandb_api_key: string | null;
  readonly wandb_init_kwargs: Record<string, unknown>;
  readonly wandb_attach_existing: boolean;
  readonly wandb_step_metric: string | null;
  readonly use_mlflow: boolean;
  readonly mlflow_tracking_uri: string | null;
  readonly mlflow_experiment_name: string | null;
  readonly mlflow_attach_existing: boolean;
  readonly key_prefix: string;
  readonly wandb_client: WandbClientLike | null;
  readonly mlflow_client: MlflowClientLike | null;

  initialized = false;
  run_started = false;
  run_ended = false;
  private created_mlflow_run = false;
  private mlflow_run_id: string | null = null;
  private wandb_step_metric_defined = false;
  private readonly wandb_table_rows = new Map<string, { columns: string[]; data: unknown[][] }>();
  readonly logged_configs: Record<string, boolean | number | string | null> = {};
  readonly logged_metrics: Array<{ metrics: Record<string, number>; step?: number }> = [];
  readonly logged_tables: LoggedTable[] = [];
  readonly logged_summaries: LoggedSummary[] = [];

  constructor(opts: ExperimentTrackerOptions = {}) {
    this.use_wandb = opts.use_wandb ?? false;
    this.wandb_api_key = opts.wandb_api_key ?? null;
    this.wandb_init_kwargs = opts.wandb_init_kwargs ?? {};
    this.wandb_attach_existing = opts.wandb_attach_existing ?? false;
    this.wandb_step_metric = opts.wandb_step_metric ?? null;
    this.use_mlflow = opts.use_mlflow ?? false;
    this.mlflow_tracking_uri = opts.mlflow_tracking_uri ?? null;
    this.mlflow_experiment_name = opts.mlflow_experiment_name ?? null;
    this.mlflow_attach_existing = opts.mlflow_attach_existing ?? false;
    this.key_prefix = opts.key_prefix ?? '';
    this.wandb_client = opts.wandb_client ?? null;
    this.mlflow_client = opts.mlflow_client ?? null;
  }

  private prefixed(key: string): string {
    return this.key_prefix === '' ? key : `${this.key_prefix}${key}`;
  }

  initialize(): void {
    if (this.use_wandb && this.wandb_client?.login !== undefined) {
      this.wandb_client.login(
        this.wandb_api_key === null ? undefined : { key: this.wandb_api_key, verify: true },
      );
    }
    if (this.use_mlflow && this.mlflow_client !== null) {
      if (this.mlflow_tracking_uri !== null) {
        this.mlflow_client.set_tracking_uri?.(this.mlflow_tracking_uri);
      }
      if (this.mlflow_experiment_name !== null) {
        this.mlflow_client.set_experiment?.(this.mlflow_experiment_name);
      }
    }
    this.initialized = true;
  }

  start_run(): void {
    if (this.use_wandb && !this.wandb_attach_existing) {
      const run = this.wandb_client?.init?.(this.wandb_init_kwargs);
      if (run !== undefined && this.wandb_client !== null) {
        this.wandb_client.run = run;
      }
    }
    if (this.use_mlflow && this.mlflow_client !== null) {
      if (this.mlflow_attach_existing) {
        this.created_mlflow_run = false;
      } else if (this.mlflow_client.active_run?.() === null) {
        this.mlflow_client.start_run?.();
        this.created_mlflow_run = true;
      }
      const active = this.mlflow_client.active_run?.() ?? null;
      this.mlflow_run_id = typeof active?.info?.run_id === 'string' ? active.info.run_id : null;
    }
    this.run_started = true;
    this.run_ended = false;
  }

  end_run(): void {
    if (this.use_wandb && !this.wandb_attach_existing) {
      this.wandb_client?.finish?.();
    }
    if (this.use_mlflow && this.created_mlflow_run) {
      this.mlflow_client?.end_run?.();
      this.created_mlflow_run = false;
    }
    this.run_ended = true;
    this.run_started = false;
  }

  enter(): this {
    this.initialize();
    this.start_run();
    return this;
  }

  exit(): false {
    this.end_run();
    return false;
  }

  log_config(config: Record<string, unknown>): void {
    const safe_config: Record<string, boolean | number | string | null> = {};
    for (const [key, value] of Object.entries(config)) {
      safe_config[key] = is_scalar_config_value(value) ? value : String(value);
      this.logged_configs[this.prefixed(key)] = safe_config[key];
    }
    const prefixed_config = Object.fromEntries(
      Object.entries(safe_config).map(([key, value]) => [this.prefixed(key), value]),
    );
    if (this.use_wandb) {
      this.wandb_client?.config?.update(prefixed_config, { allow_val_change: true });
    }
    if (this.use_mlflow) {
      const params = Object.fromEntries(
        Object.entries(prefixed_config).map(([key, value]) => [key, String(value)]),
      );
      if (this.mlflow_client !== null && this.mlflow_run_id !== null && this.mlflow_client.log_param !== undefined) {
        for (const [key, value] of Object.entries(params)) {
          this.mlflow_client.log_param(this.mlflow_run_id, key, value);
        }
      } else {
        this.mlflow_client?.log_params?.(params);
      }
    }
  }

  log_metrics(metrics: Record<string, unknown>, step?: number): void {
    const numeric_metrics: Record<string, number> = {};
    for (const [key, value] of Object.entries(metrics)) {
      if (is_numeric_value(value)) {
        numeric_metrics[this.prefixed(key)] = value;
      }
    }
    if (Object.keys(numeric_metrics).length > 0) {
      this.logged_metrics.push(step === undefined ? { metrics: numeric_metrics } : { metrics: numeric_metrics, step });
      if (this.use_wandb) {
        this.define_wandb_step_metric();
        if (this.wandb_step_metric !== null && step !== undefined) {
          this.wandb_client?.log?.({ ...numeric_metrics, [this.wandb_step_metric]: step });
        } else {
          this.wandb_client?.log?.(numeric_metrics, step === undefined ? undefined : { step });
        }
      }
      if (this.use_mlflow) {
        if (this.mlflow_client !== null && this.mlflow_run_id !== null && this.mlflow_client.log_metric !== undefined) {
          for (const [key, value] of Object.entries(numeric_metrics)) {
            this.mlflow_client.log_metric(this.mlflow_run_id, key, value, step ?? 0);
          }
        } else {
          this.mlflow_client?.log_metrics?.(numeric_metrics, step);
        }
      }
    }
  }

  log_table(key: string, columns: string[], data: unknown[][]): void {
    const prefixed_key = this.prefixed(key);
    this.logged_tables.push({
      key: prefixed_key,
      columns: [...columns],
      data: data.map((row) => [...row]),
    });
    if (this.use_wandb && this.wandb_client?.log !== undefined) {
      const existing = this.wandb_table_rows.get(prefixed_key);
      if (existing === undefined) {
        this.wandb_table_rows.set(prefixed_key, { columns: [...columns], data: data.map((row) => [...row]) });
      } else {
        existing.data.push(...data.map((row) => [...row]));
      }
      const table = this.wandb_table_rows.get(prefixed_key);
      if (table !== undefined) {
        const table_payload = this.wandb_client.Table?.({ columns: table.columns, data: table.data }) ?? {
          columns: table.columns,
          data: table.data,
        };
        this.wandb_client.log({ [prefixed_key]: table_payload }, { commit: false });
      }
    }
    if (this.use_mlflow) {
      const table_dict: Record<string, unknown[]> = {};
      columns.forEach((column, idx) => {
        table_dict[column] = data.map((row) => row[idx]);
      });
      this.mlflow_client?.log_table?.({ data: table_dict, artifact_file: `${prefixed_key}.json` });
    }
  }

  log_summary(summary: Record<string, unknown>): void;
  log_summary(key: string, value: unknown): void;
  log_summary(key_or_summary: string | Record<string, unknown>, value?: unknown): void {
    if (typeof key_or_summary === 'string') {
      const key = this.prefixed(key_or_summary);
      this.logged_summaries.push({ key, value });
      this.log_summary_to_backends({ [key]: value });
      return;
    }
    const prefixed_summary: Record<string, unknown> = {};
    for (const [key, entry_value] of Object.entries(key_or_summary)) {
      const prefixed_key = this.prefixed(key);
      this.logged_summaries.push({ key: prefixed_key, value: entry_value });
      prefixed_summary[prefixed_key] = entry_value;
    }
    this.log_summary_to_backends(prefixed_summary);
  }

  log_artifact(key: string, value: unknown): void {
    this.log_summary(key, value);
  }

  log_html(html: string, key = 'html'): void {
    const prefixed_key = this.prefixed(key);
    this.logged_summaries.push({ key: prefixed_key, value: html });
    const html_payload = this.wandb_client?.Html?.(html) ?? html;
    if (this.use_wandb) {
      this.wandb_client?.log?.({ [prefixed_key]: html_payload }, { commit: false });
      if (this.wandb_client?.run?.summary !== undefined) {
        this.wandb_client.run.summary[prefixed_key] = html_payload;
      }
    }
    if (this.use_mlflow) {
      this.mlflow_client?.log_artifact?.(html, prefixed_key);
    }
  }

  private define_wandb_step_metric(): void {
    if (this.wandb_step_metric_defined || this.wandb_step_metric === null || !this.use_wandb) return;
    this.wandb_client?.define_metric?.(this.wandb_step_metric, { hidden: false });
    this.wandb_client?.define_metric?.(this.key_prefix === '' ? '*' : `${this.key_prefix}*`, {
      step_metric: this.wandb_step_metric,
    });
    this.wandb_step_metric_defined = true;
  }

  private log_summary_to_backends(summary: Record<string, unknown>): void {
    if (this.use_wandb && this.wandb_client?.run?.summary !== undefined) {
      for (const [key, value] of Object.entries(summary)) {
        this.wandb_client.run.summary[key] = value;
      }
    }
    if (this.use_mlflow) {
      const numeric: Record<string, number> = {};
      const text: Record<string, string> = {};
      for (const [key, value] of Object.entries(summary)) {
        if (is_numeric_value(value)) numeric[key] = value;
        else if (typeof value === 'string') text[`summary/${key}`] = value;
      }
      if (this.mlflow_client !== null && this.mlflow_run_id !== null && this.mlflow_client.log_metric !== undefined) {
        for (const [key, value] of Object.entries(numeric)) {
          this.mlflow_client.log_metric(this.mlflow_run_id, key, value);
        }
        for (const [key, value] of Object.entries(text)) {
          this.mlflow_client.log_param?.(this.mlflow_run_id, key, value);
        }
      } else {
        if (Object.keys(numeric).length > 0) this.mlflow_client?.log_metrics?.(numeric);
        if (Object.keys(text).length > 0) this.mlflow_client?.log_params?.(text);
      }
    }
  }
}

export function create_experiment_tracker(opts: ExperimentTrackerOptions = {}): ExperimentTracker {
  return new ExperimentTracker(opts);
}
