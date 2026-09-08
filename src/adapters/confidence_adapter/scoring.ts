export interface ScoringStrategy {
  score(is_correct: boolean, logprob_score: number | null): number;
  describe(): string;
}

export class LinearBlendScoring implements ScoringStrategy {
  readonly low_confidence_threshold: number;
  readonly min_score_on_correct: number;

  constructor(low_confidence_threshold: number = 0.5, min_score_on_correct: number = 0.3) {
    if (!(low_confidence_threshold > 0.0 && low_confidence_threshold <= 1.0)) {
      throw new Error('low_confidence_threshold must be in (0, 1]');
    }
    if (!(min_score_on_correct >= 0.0 && min_score_on_correct < 1.0)) {
      throw new Error('min_score_on_correct must be in [0, 1)');
    }
    this.low_confidence_threshold = low_confidence_threshold;
    this.min_score_on_correct = min_score_on_correct;
  }

  score(is_correct: boolean, logprob_score: number | null): number {
    if (!is_correct) return 0.0;
    if (logprob_score === null) return 1.0;
    const probability = Math.exp(logprob_score);
    if (probability >= this.low_confidence_threshold) return 1.0;
    const t = probability / this.low_confidence_threshold;
    return this.min_score_on_correct + (1.0 - this.min_score_on_correct) * t;
  }

  describe(): string {
    return `LinearBlendScoring(threshold=${this.low_confidence_threshold}, min_score=${this.min_score_on_correct})`;
  }
}

export class ThresholdScoring implements ScoringStrategy {
  readonly threshold: number;

  constructor(threshold: number = 0.7) {
    if (!(threshold > 0.0 && threshold <= 1.0)) {
      throw new Error('threshold must be in (0, 1]');
    }
    this.threshold = threshold;
  }

  score(is_correct: boolean, logprob_score: number | null): number {
    if (!is_correct) return 0.0;
    if (logprob_score === null) return 1.0;
    return Math.exp(logprob_score) >= this.threshold ? 1.0 : 0.0;
  }

  describe(): string {
    return `ThresholdScoring(threshold=${this.threshold})`;
  }
}

export class SigmoidScoring implements ScoringStrategy {
  readonly midpoint: number;
  readonly steepness: number;

  constructor(midpoint: number = 0.5, steepness: number = 10.0) {
    if (!(midpoint > 0.0 && midpoint < 1.0)) {
      throw new Error('midpoint must be in (0, 1)');
    }
    if (steepness <= 0) {
      throw new Error('steepness must be positive');
    }
    this.midpoint = midpoint;
    this.steepness = steepness;
  }

  score(is_correct: boolean, logprob_score: number | null): number {
    if (!is_correct) return 0.0;
    if (logprob_score === null) return 1.0;
    const probability = Math.exp(logprob_score);
    return 1.0 / (1.0 + Math.exp(-this.steepness * (probability - this.midpoint)));
  }

  describe(): string {
    return `SigmoidScoring(midpoint=${this.midpoint}, steepness=${this.steepness})`;
  }
}
