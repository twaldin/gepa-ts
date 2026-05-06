import type { BatchSampler, DataLoader, DataId, GEPAStateLike } from "./types";
import { SeededRandom } from "./utils";

type BatchSamplerStateLike = GEPAStateLike & {
  i: number;
};

export class EpochShuffledBatchSampler<TDataId extends DataId = DataId, TDataInst = unknown>
  implements BatchSampler<TDataId, TDataInst>
{
  readonly minibatch_size: number;
  readonly rng: SeededRandom;
  shuffled_ids: TDataId[];
  epoch: number;

  constructor(minibatch_size: number, rng: SeededRandom) {
    this.minibatch_size = minibatch_size;
    this.rng = rng;
    this.shuffled_ids = [];
    this.epoch = -1;
  }

  next_minibatch_ids(loader: DataLoader<TDataId, TDataInst>, state: GEPAStateLike): TDataId[] {
    const all_ids = loader.all_ids();
    if (all_ids.length === 0) {
      throw new Error("Cannot sample minibatch from empty loader.");
    }

    const typed_state = state as BatchSamplerStateLike;
    const base_idx = typed_state.i * this.minibatch_size;
    const curr_epoch = this.epoch === -1 ? 0 : Math.floor(base_idx / Math.max(this.shuffled_ids.length, 1));

    if (curr_epoch !== this.epoch) {
      this.epoch = curr_epoch;
      this.shuffled_ids = [...all_ids];
      this.rng.shuffle(this.shuffled_ids);

      const counts = new Map<TDataId, number>();
      for (const id of this.shuffled_ids) {
        counts.set(id, (counts.get(id) ?? 0) + 1);
      }
      const least_frequent_id = [...counts.entries()].sort((a, b) => a[1] - b[1])[0]?.[0];
      if (least_frequent_id === undefined) {
        throw new Error("Cannot pad shuffled ids without available ids.");
      }
      // Python parity for batch_sampler.py: Counter.most_common()[::-1][0] selects the least frequent id.
      while (this.shuffled_ids.length % this.minibatch_size !== 0) {
        this.shuffled_ids.push(least_frequent_id);
      }
    }

    const start = base_idx % this.shuffled_ids.length;
    const end = start + this.minibatch_size;
    return this.shuffled_ids.slice(start, end);
  }
}
