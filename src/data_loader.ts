import type { DataId, DataLoader } from "./types";

export class ListDataLoader<T> implements DataLoader<number, T> {
  protected readonly items: T[];

  constructor(items: T[]) {
    this.items = [...items];
  }

  all_ids(): number[] {
    return Array.from({ length: this.items.length }, (_, i) => i);
  }

  fetch(ids: number[]): T[] {
    return ids.map((data_id) => this.items[data_id] as T);
  }

  add_items(items: T[]): void {
    this.items.push(...items);
  }

  get length(): number {
    return this.items.length;
  }
}

export class StagedDataLoader<T> extends ListDataLoader<T> {
  private readonly stages: Array<[number, T[]]>;
  private next_stage_idx = 0;
  private batches_served_count = 0;
  num_unlocked_stages = 1;

  constructor(initial_items: T[], staged_items: Array<[number, T[]]>) {
    super(initial_items);
    this.stages = staged_items
      .map(([threshold, items]) => [Math.max(0, threshold), [...items]] as [number, T[]])
      .sort(([a], [b]) => a - b);
    this.unlock_if_due();
  }

  get batches_served(): number {
    return this.batches_served_count;
  }

  override fetch(ids: number[]): T[] {
    const batch = super.fetch(ids);
    this.batches_served_count += 1;
    this.unlock_if_due();
    return batch;
  }

  unlock_next_stage(): boolean {
    if (this.next_stage_idx >= this.stages.length) {
      return false;
    }
    const [, items] = this.stages[this.next_stage_idx]!;
    this.add_items(items);
    this.next_stage_idx += 1;
    this.num_unlocked_stages += 1;
    return true;
  }

  private unlock_if_due(): void {
    while (this.next_stage_idx < this.stages.length) {
      const [threshold] = this.stages[this.next_stage_idx]!;
      if (this.batches_served_count < threshold) {
        break;
      }
      this.unlock_next_stage();
    }
  }
}

function is_data_loader<TDataId extends DataId, T>(
  data: T[] | DataLoader<TDataId, T>,
): data is DataLoader<TDataId, T> {
  if (typeof data !== "object" || data === null) {
    return false;
  }
  return "all_ids" in data && typeof data.all_ids === "function" && "fetch" in data && typeof data.fetch === "function";
}

export function ensure_loader<TDataId extends DataId, T>(data: T[] | DataLoader<TDataId, T>): DataLoader<TDataId, T> | DataLoader<number, T> {
  if (is_data_loader(data)) {
    return data;
  }
  return new ListDataLoader(data);
}

export async function refresh_loader(loader: DataLoader): Promise<void> {
  if (typeof loader.refresh === 'function') {
    await loader.refresh();
  }
}

export async function fetch_loader<TDataId extends DataId, TDataInst>(
  loader: DataLoader<TDataId, TDataInst>,
  ids: TDataId[],
): Promise<TDataInst[]> {
  if (typeof loader.fetch_async === 'function') {
    return loader.fetch_async(ids);
  }
  return loader.fetch(ids);
}

export type { DataId };
