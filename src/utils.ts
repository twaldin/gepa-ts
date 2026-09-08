export class SeededRandom {
  private static readonly N = 624;
  private static readonly M = 397;
  private static readonly MATRIX_A = 0x9908b0df;
  private static readonly UPPER_MASK = 0x80000000;
  private static readonly LOWER_MASK = 0x7fffffff;
  private readonly mt: number[];
  private mti: number;

  constructor(seed: number) {
    this.mt = new Array<number>(SeededRandom.N).fill(0);
    this.mti = SeededRandom.N + 1;
    this.init_by_array([seed >>> 0]);
  }

  private init_genrand(seed: number): void {
    this.mt[0] = seed >>> 0;
    for (this.mti = 1; this.mti < SeededRandom.N; this.mti += 1) {
      const prev = this.mt[this.mti - 1]!;
      const mixed = prev ^ (prev >>> 30);
      this.mt[this.mti] = (Math.imul(1812433253, mixed) + this.mti) >>> 0;
    }
  }

  private init_by_array(init_key: number[]): void {
    this.init_genrand(19650218);
    let i = 1;
    let j = 0;
    let k = Math.max(SeededRandom.N, init_key.length);

    for (; k > 0; k -= 1) {
      const prev = this.mt[i - 1]!;
      const mixed = prev ^ (prev >>> 30);
      this.mt[i] = ((this.mt[i]! ^ Math.imul(mixed, 1664525)) + init_key[j]! + j) >>> 0;
      i += 1;
      j += 1;
      if (i >= SeededRandom.N) {
        this.mt[0] = this.mt[SeededRandom.N - 1]!;
        i = 1;
      }
      if (j >= init_key.length) {
        j = 0;
      }
    }

    for (k = SeededRandom.N - 1; k > 0; k -= 1) {
      const prev = this.mt[i - 1]!;
      const mixed = prev ^ (prev >>> 30);
      this.mt[i] = ((this.mt[i]! ^ Math.imul(mixed, 1566083941)) - i) >>> 0;
      i += 1;
      if (i >= SeededRandom.N) {
        this.mt[0] = this.mt[SeededRandom.N - 1]!;
        i = 1;
      }
    }

    this.mt[0] = 0x80000000;
  }

  private genrand_int32(): number {
    let y: number;
    const mag01 = [0, SeededRandom.MATRIX_A] as const;

    if (this.mti >= SeededRandom.N) {
      let kk = 0;
      for (; kk < SeededRandom.N - SeededRandom.M; kk += 1) {
        y = (this.mt[kk]! & SeededRandom.UPPER_MASK) | (this.mt[kk + 1]! & SeededRandom.LOWER_MASK);
        this.mt[kk] = (this.mt[kk + SeededRandom.M]! ^ (y >>> 1) ^ mag01[y & 1]!) >>> 0;
      }
      for (; kk < SeededRandom.N - 1; kk += 1) {
        y = (this.mt[kk]! & SeededRandom.UPPER_MASK) | (this.mt[kk + 1]! & SeededRandom.LOWER_MASK);
        this.mt[kk] = (this.mt[kk + (SeededRandom.M - SeededRandom.N)]! ^ (y >>> 1) ^ mag01[y & 1]!) >>> 0;
      }
      y = (this.mt[SeededRandom.N - 1]! & SeededRandom.UPPER_MASK) | (this.mt[0]! & SeededRandom.LOWER_MASK);
      this.mt[SeededRandom.N - 1] = (this.mt[SeededRandom.M - 1]! ^ (y >>> 1) ^ mag01[y & 1]!) >>> 0;
      this.mti = 0;
    }

    y = this.mt[this.mti]!;
    this.mti += 1;
    y ^= y >>> 11;
    y ^= (y << 7) & 0x9d2c5680;
    y ^= (y << 15) & 0xefc60000;
    y ^= y >>> 18;
    return y >>> 0;
  }

  private getrandbits(k: number): number {
    if (!Number.isInteger(k) || k <= 0 || k > 32) {
      throw new Error(`getrandbits only supports 1..32 bits, got ${k}.`);
    }
    return this.genrand_int32() >>> (32 - k);
  }

  private randbelow(n: number): number {
    if (!Number.isInteger(n) || n <= 0) {
      throw new Error(`randbelow requires a positive integer, got ${n}.`);
    }
    const k = n.toString(2).length;
    let r = this.getrandbits(k);
    while (r >= n) {
      r = this.getrandbits(k);
    }
    return r;
  }

  random(): number {
    const a = this.genrand_int32() >>> 5;
    const b = this.genrand_int32() >>> 6;
    return (a * 67108864 + b) / 9007199254740992;
  }

  randint(min: number, max: number): number {
    if (max < min) {
      throw new Error(`Invalid randint bounds: min=${min}, max=${max}`);
    }
    return min + this.randbelow(max - min + 1);
  }

  choice<T>(arr: T[]): T {
    if (arr.length === 0) {
      throw new Error("Cannot choose from an empty array.");
    }
    const idx = this.randint(0, arr.length - 1);
    return arr[idx]!;
  }

  sample<T>(arr: T[], k: number): T[] {
    if (!Number.isInteger(k) || k < 0) {
      throw new Error(`sample requires a non-negative integer k, got ${k}.`);
    }
    if (k > arr.length) {
      throw new Error("Sample larger than population or is negative.");
    }
    const result: T[] = [];

    const n = arr.length;
    let setsize = 21;
    if (k > 5) {
      setsize += 4 ** Math.ceil(Math.log(k * 3) / Math.log(4));
    }

    if (n <= setsize) {
      const pool = [...arr];
      for (let i = 0; i < k; i += 1) {
        const j = this.randbelow(n - i);
        const picked = pool[j];
        const replacement = pool[n - i - 1];
        if (picked === undefined || replacement === undefined) {
          throw new Error("Invariant failed: sampled index missing.");
        }
        result.push(picked);
        pool[j] = replacement;
      }
      return result;
    }

    const selected = new Set<number>();
    for (let i = 0; i < k; i += 1) {
      let j = this.randbelow(n);
      while (selected.has(j)) {
        j = this.randbelow(n);
      }
      selected.add(j);
      const picked = arr[j];
      if (picked === undefined) {
        throw new Error("Invariant failed: sampled index missing.");
      }
      result.push(picked);
    }
    return result;
  }

  choices<T>(arr: T[], opts: { k: number; weights?: number[] }): T[] {
    const { k, weights } = opts;
    if (!Number.isInteger(k) || k < 0) {
      throw new Error(`choices requires a non-negative integer k, got ${k}.`);
    }
    if (arr.length === 0) {
      throw new Error("Cannot choose from an empty array.");
    }
    if (weights !== undefined && weights.length !== arr.length) {
      throw new Error("The number of weights does not match the population.");
    }

    const result: T[] = [];
    if (weights === undefined) {
      for (let i = 0; i < k; i += 1) {
        result.push(this.choice(arr));
      }
      return result;
    }

    const cum_weights: number[] = [];
    let total = 0;
    for (const weight of weights) {
      if (weight < 0) {
        throw new Error("Weights must be non-negative.");
      }
      total += weight;
      cum_weights.push(total);
    }
    if (total <= 0) {
      throw new Error("Total of weights must be greater than zero.");
    }

    for (let i = 0; i < k; i += 1) {
      const value = this.random() * total;
      const selected_idx = cum_weights.findIndex((cum_weight) => value < cum_weight);
      const selected = arr[selected_idx === -1 ? arr.length - 1 : selected_idx];
      if (selected === undefined) {
        throw new Error("Invariant failed: weighted choice index missing.");
      }
      result.push(selected);
    }
    return result;
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

export function json_default(value: unknown): unknown {
  if (value !== null && typeof value === "object") {
    if (value instanceof Map) {
      return Object.fromEntries(value.entries());
    }
    if (value instanceof Set) {
      return [...value];
    }
    try {
      return { ...(value as object) };
    } catch {
      return String(value);
    }
  }
  return String(value);
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
