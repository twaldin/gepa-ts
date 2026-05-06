export class SeededRandom {
  private state: number;

  constructor(seed: number) {
    this.state = seed >>> 0;
  }

  random(): number {
    this.state = (this.state + 0x6d2b79f5) >>> 0;
    let t = this.state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  }

  randint(min: number, max: number): number {
    if (max < min) {
      throw new Error(`Invalid randint bounds: min=${min}, max=${max}`);
    }
    return Math.floor(this.random() * (max - min + 1)) + min;
  }

  choice<T>(arr: T[]): T {
    if (arr.length === 0) {
      throw new Error("Cannot choose from an empty array.");
    }
    const idx = this.randint(0, arr.length - 1);
    return arr[idx]!;
  }

  shuffle<T>(arr: T[]): void {
    for (let i = arr.length - 1; i > 0; i -= 1) {
      const j = this.randint(0, i);
      const temp = arr[i]!;
      arr[i] = arr[j]!;
      arr[j] = temp;
    }
  }
}

export function idxmax(arr: number[]): number {
  const max_val = Math.max(...arr);
  return arr.indexOf(max_val);
}

export function is_dominated(
  y: number,
  programs: Set<number>,
  program_at_pareto_front: Record<string | number, Set<number>>,
): boolean {
  const y_fronts = Object.values(program_at_pareto_front).filter((front) => front.has(y));
  for (const front of y_fronts) {
    let found_dominator_in_front = false;
    for (const other_prog of front) {
      if (programs.has(other_prog)) {
        found_dominator_in_front = true;
        break;
      }
    }
    if (!found_dominator_in_front) {
      return false;
    }
  }

  return true;
}

export function remove_dominated_programs(
  program_at_pareto_front: Record<string | number, Set<number>>,
  scores?: Record<number, number> | number[],
): Record<string | number, Set<number>> {
  const freq: Record<number, number> = {};
  for (const front of Object.values(program_at_pareto_front)) {
    for (const p of front) {
      freq[p] = (freq[p] ?? 0) + 1;
    }
  }

  const dominated = new Set<number>();
  let programs = Object.keys(freq).map((p) => Number(p));

  let score_map: Record<number, number>;
  if (scores === undefined) {
    score_map = Object.fromEntries(programs.map((p) => [p, 1]));
  } else if (Array.isArray(scores)) {
    score_map = Object.fromEntries(programs.map((p) => [p, scores[p] ?? 0]));
  } else {
    score_map = scores;
  }

  programs = [...programs].sort((a, b) => (score_map[a] ?? 0) - (score_map[b] ?? 0));

  let found_to_remove = true;
  while (found_to_remove) {
    found_to_remove = false;
    for (const y of programs) {
      if (dominated.has(y)) {
        continue;
      }
      const others = new Set(programs.filter((p) => p !== y && !dominated.has(p)));
      if (is_dominated(y, others, program_at_pareto_front)) {
        dominated.add(y);
        found_to_remove = true;
        break;
      }
    }
  }

  const dominators = programs.filter((p) => !dominated.has(p));
  for (const front of Object.values(program_at_pareto_front)) {
    if (front.size === 0) {
      continue;
    }
    const valid = dominators.some((p) => front.has(p));
    if (!valid) {
      throw new Error("Invariant failed: expected at least one dominator in every non-empty front.");
    }
  }

  const new_program_at_pareto_front = Object.fromEntries(
    Object.entries(program_at_pareto_front).map(([val_id, front]) => [
      val_id,
      new Set([...front].filter((prog_idx) => dominators.includes(prog_idx))),
    ]),
  ) as Record<string | number, Set<number>>;

  for (const [val_id, front_new] of Object.entries(new_program_at_pareto_front)) {
    const old_front = program_at_pareto_front[val_id];
    if (!old_front) {
      throw new Error(`Invariant failed: missing original front for ${val_id}.`);
    }
    for (const prog of front_new) {
      if (!old_front.has(prog)) {
        throw new Error("Invariant failed: new front must be subset of old front.");
      }
    }
  }

  return new_program_at_pareto_front;
}

export function find_dominator_programs(
  pareto_front_programs: Record<string | number, Set<number>>,
  agg_scores: number[],
): number[] {
  const new_program_at_pareto_front = remove_dominated_programs(pareto_front_programs, agg_scores);
  const uniq_progs = new Set<number>();
  for (const front of Object.values(new_program_at_pareto_front)) {
    for (const prog of front) {
      uniq_progs.add(prog);
    }
  }
  return [...uniq_progs];
}

// NOT bit-exact with Python random.Random; chosen pytest target asserts result presence only, not specific candidates.
export function select_program_candidate_from_pareto_front(
  pareto_front_programs: Record<string | number, Set<number>>,
  agg_scores: number[],
  rng: SeededRandom,
): number {
  const new_program_at_pareto_front = remove_dominated_programs(pareto_front_programs, agg_scores);
  const frequency: Record<number, number> = {};
  for (const testcase_pareto_front of Object.values(new_program_at_pareto_front)) {
    for (const prog_idx of testcase_pareto_front) {
      frequency[prog_idx] = (frequency[prog_idx] ?? 0) + 1;
    }
  }

  const sampling_list: number[] = [];
  for (const [prog_idx, freq] of Object.entries(frequency)) {
    for (let i = 0; i < freq; i += 1) {
      sampling_list.push(Number(prog_idx));
    }
  }

  if (sampling_list.length <= 0) {
    throw new Error("Invariant failed: sampling_list must be non-empty.");
  }

  return rng.choice(sampling_list);
}
