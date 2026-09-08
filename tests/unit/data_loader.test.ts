import { describe, expect, it } from 'vitest';
import { ListDataLoader, StagedDataLoader, ensure_loader } from '../../src/data_loader.js';

describe('data loaders', () => {
  it('supports the upstream ListDataLoader contract', () => {
    const loader = new ListDataLoader(['a', 'b']);

    expect(loader.all_ids()).toEqual([0, 1]);
    expect(loader.fetch([1, 0])).toEqual(['b', 'a']);

    loader.add_items(['c']);
    expect(loader.all_ids()).toEqual([0, 1, 2]);
    expect(loader.fetch([2])).toEqual(['c']);
    expect(loader.length).toBe(3);
  });

  it('unlocks staged examples after fetch thresholds', () => {
    const loader = new StagedDataLoader(['base0', 'base1'], [
      [1, ['stage1_item']],
      [3, ['stage2_item']],
    ]);

    expect(loader.all_ids()).toEqual([0, 1]);
    expect(loader.num_unlocked_stages).toBe(1);
    expect(loader.batches_served).toBe(0);

    loader.fetch([0]);
    expect(loader.batches_served).toBe(1);
    expect(loader.num_unlocked_stages).toBe(2);
    expect(loader.all_ids()).toEqual([0, 1, 2]);

    loader.fetch([1]);
    expect(loader.batches_served).toBe(2);
    expect(loader.num_unlocked_stages).toBe(2);

    loader.fetch([2]);
    expect(loader.batches_served).toBe(3);
    expect(loader.num_unlocked_stages).toBe(3);
    expect(loader.all_ids()).toEqual([0, 1, 2, 3]);
  });

  it('supports manual staged unlocks', () => {
    const loader = new StagedDataLoader(['base'], [[5, ['late']]]);

    expect(loader.all_ids()).toEqual([0]);
    expect(loader.num_unlocked_stages).toBe(1);
    expect(loader.unlock_next_stage()).toBe(true);
    expect(loader.num_unlocked_stages).toBe(2);
    expect(loader.all_ids()).toEqual([0, 1]);
    expect(loader.unlock_next_stage()).toBe(false);
  });

  it('preserves existing loader instances through ensure_loader', () => {
    const loader = new StagedDataLoader(['base'], []);

    expect(ensure_loader(loader)).toBe(loader);
    expect(ensure_loader(['x']).fetch([0])).toEqual(['x']);
  });
});
