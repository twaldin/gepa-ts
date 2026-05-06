import type { BatchSampler, DataLoader, DataId, GEPAStateLike } from "./types";
import { SeededRandom } from "./utils";

type BatchSamplerStateLike = GEPAStateLike & {
  i: number;
};

function get_least_frequent_id<TDataId extends DataId>(counts: Map<TDataId, number>): TDataId {
  const sorted = [...counts.entries()].sort((a, b) => a[1] - b[1]);
  const least_frequent_id = sorted[0]?.[0];
  if (least_frequent_id === undefined) {
    throw new Error("Cannot pad shuffled ids without available ids.");
  }
  return least_frequent_id;
}

export class EpochShuffledBatchSampler<TDataId extends DataId = DataId, TDataInst = unknown>
  implements BatchSampler<TDataId, TDataInst>
{
  readonly minibatch_size: number;
  readonly rng: SeededRandom;
  shuffled_ids: TDataId[];
  epoch: number;
  id_freqs: Map<TDataId, number>;
  last_trainset_size: number;

  constructor(minibatch_size: number, rng: SeededRandom) {
    this.minibatch_size = minibatch_size;
    this.rng = rng;
    this.shuffled_ids = [];
    this.epoch = -1;
    this.id_freqs = new Map();
    this.last_trainset_size = 0;
  }

  private update_shuffled(loader: DataLoader<TDataId, TDataInst>): void {
    const all_ids = loader.all_ids();
    const trainset_size = loader.length;
    this.last_trainset_size = trainset_size;

    if (trainset_size === 0) {
      this.shuffled_ids = [];
      this.id_freqs = new Map();
      return;
    }

    this.shuffled_ids = [...all_ids];
    this.rng.shuffle(this.shuffled_ids);

    this.id_freqs = new Map();
    for (const id of this.shuffled_ids) {
      this.id_freqs.set(id, (this.id_freqs.get(id) ?? 0) + 1);
    }

    const mod = trainset_size % this.minibatch_size;
    const num_to_pad = mod !== 0 ? this.minibatch_size - mod : 0;
    for (let i = 0; i < num_to_pad; i += 1) {
      // Python parity for batch_sampler.py line with Counter.most_common()[::-1][0]: this picks least frequent id.
      const selected_id = get_least_frequent_id(this.id_freqs);
      this.shuffled_ids.push(selected_id);
      this.id_freqs.set(selected_id, (this.id_freqs.get(selected_id) ?? 0) + 1);
    }
  }

  next_minibatch_ids(loader: DataLoader<TDataId, TDataInst>, state: GEPAStateLike): TDataId[] {
    const trainset_size = loader.length;
    if (trainset_size === 0) {
      throw new Error("Cannot sample a minibatch from an empty loader.");
    }

    const typed_state = state as BatchSamplerStateLike;
    const base_idx = typed_state.i * this.minibatch_size;
    const curr_epoch = this.epoch === -1 ? 0 : Math.floor(base_idx / Math.max(this.shuffled_ids.length, 1));
    const needs_refresh =
      this.shuffled_ids.length === 0 || trainset_size !== this.last_trainset_size || curr_epoch > this.epoch;

    if (needs_refresh) {
      this.epoch = curr_epoch;
      this.update_shuffled(loader);
    }

    const start = base_idx % this.shuffled_ids.length;
    const end = start + this.minibatch_size;
    return this.shuffled_ids.slice(start, end);
  }
}
