import type { DataId, DataLoader } from "./types";

export class ListDataLoader<T> implements DataLoader<number, T> {
  private readonly items: T[];

  constructor(items: T[]) {
    this.items = [...items];
  }

  all_ids(): number[] {
    return Array.from({ length: this.items.length }, (_, i) => i);
  }

  fetch(ids: number[]): T[] {
    return ids.map((data_id) => this.items[data_id] as T);
  }

  get length(): number {
    return this.items.length;
  }
}

function is_data_loader<T>(
  data: T[] | DataLoader<number, T>,
): data is DataLoader<number, T> {
  if (typeof data !== "object" || data === null) {
    return false;
  }
  return "all_ids" in data && typeof data.all_ids === "function" && "fetch" in data && typeof data.fetch === "function";
}

export function ensure_loader<T>(data: T[] | DataLoader<number, T>): DataLoader<number, T> {
  if (is_data_loader(data)) {
    return data;
  }
  return new ListDataLoader(data);
}

export type { DataId };
