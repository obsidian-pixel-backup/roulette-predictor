// Ensemble utilities: Monte Carlo Markov simulation, DQN weighting, HMM regime detection, auto-tuner
// History: array<int 0..3>

// ---- Monte Carlo using Markov transition counts ----
export function monteCarloMarkov(
  counts,
  lastClass,
  { numSims = 200, horizon = 1 }
) {
  if (lastClass == null) return { probs: [0.25, 0.25, 0.25, 0.25], numSims };
  // Derive transition probabilities row-normalized from counts matrix (4x4)
  const row = counts[lastClass] || [1, 1, 1, 1];
  const sum = row.reduce((a, b) => a + b, 0) || 1;
  const trans = row.map((v) => v / sum);
  const tallies = [0, 0, 0, 0];
  for (let s = 0; s < numSims; s++) {
    let cls = lastClass;
    for (let h = 0; h < horizon; h++) {
      const r = Math.random();
      let acc = 0;
      for (let k = 0; k < 4; k++) {
        acc += k === 0 ? (cls === lastClass ? trans[k] : trans[k]) : trans[k];
        if (r <= acc) {
          cls = k;
          break;
        }
      }
    }
    tallies[cls]++; // final class after horizon
  }
  const tSum = tallies.reduce((a, b) => a + b, 0) || 1;
  const probs = tallies.map((v) => v / tSum);
  return { probs, numSims };
}

// ---- Simple HMM with 2 hidden states (regime detection) ----
// We estimate two regime emission distributions via k-means (k=2) over recent class frequency vectors
export function hmmProbs(history, { window = 200 } = {}) {
  if (history.length === 0)
    return {
      probs: [0.25, 0.25, 0.25, 0.25],
      states: [],
      emissions: [],
      trans: [
        [0.9, 0.1],
        [0.1, 0.9],
      ],
    };
  const recent = history.slice(-window);
  const segLen = Math.max(10, Math.floor(recent.length / 10));
  const freqVecs = [];
  for (let i = 0; i + segLen <= recent.length; i += segLen) {
    const seg = recent.slice(i, i + segLen);
    const counts = [0, 0, 0, 0];
    seg.forEach((c) => counts[c]++);
    const total = seg.length || 1;
    freqVecs.push(counts.map((c) => c / total));
  }
  if (freqVecs.length < 2)
    return {
      probs: [0.25, 0.25, 0.25, 0.25],
      states: [],
      emissions: [],
      trans: [
        [0.9, 0.1],
        [0.1, 0.9],
      ],
    };
  // k-means k=2
  let centroids = [freqVecs[0], freqVecs[freqVecs.length - 1]];
  for (let iter = 0; iter < 5; iter++) {
    const groups = [[], []];
    freqVecs.forEach((v) => {
      const d0 = v.reduce((a, b, i) => a + Math.pow(b - centroids[0][i], 2), 0);
      const d1 = v.reduce((a, b, i) => a + Math.pow(b - centroids[1][i], 2), 0);
      groups[d0 <= d1 ? 0 : 1].push(v);
    });
    for (let k = 0; k < 2; k++)
      if (groups[k].length) {
        const mean = [0, 0, 0, 0];
        groups[k].forEach((g) => g.forEach((val, i) => (mean[i] += val)));
        for (let i = 0; i < 4; i++) mean[i] /= groups[k].length;
        centroids[k] = mean;
      }
  }
  // Simple transition matrix biased towards staying
  const trans = [
    [0.9, 0.1],
    [0.1, 0.9],
  ];
  // Current regime guess: pick centroid closer to latest frequency vector of last segLen spins
  const lastSeg = recent.slice(-segLen);
  const lc = [0, 0, 0, 0];
  lastSeg.forEach((c) => lc[c]++);
  const lt = lastSeg.length || 1;
  const lastFreq = lc.map((c) => c / lt);
  const d0 = lastFreq.reduce(
    (a, b, i) => a + Math.pow(b - centroids[0][i], 2),
    0
  );
  const d1 = lastFreq.reduce(
    (a, b, i) => a + Math.pow(b - centroids[1][i], 2),
    0
  );
  const stateProbs =
    d0 + d1 === 0 ? [0.5, 0.5] : [d1 / (d0 + d1), d0 / (d0 + d1)]; // inverse distance weighting
  // Predicted next emission distribution mixture
  const emission = [0, 0, 0, 0];
  for (let s = 0; s < 2; s++)
    emission.forEach(
      (_, i) => (emission[i] += stateProbs[s] * centroids[s][i])
    );
  // Normalize
  const esum = emission.reduce((a, b) => a + b, 0) || 1;
  const probs = emission.map((e) => e / esum);
  return { probs, emissions: centroids, trans, stateProbs };
}

// ---- DQN for dynamic weight adaptation ----
export class DQNWeights {
  constructor({ nSources, learningRate = 0.01, epsilon = 0.2 }) {
    this.nSources = nSources;
    this.learningRate = learningRate;
    this.epsilon = epsilon;
    this.weights = Array(nSources).fill(1 / nSources);
  }
  chooseAction(state) {
    // Actions: for each source increase or decrease weight slightly
    const actions = [];
    for (let i = 0; i < this.nSources; i++)
      actions.push({ type: "inc", idx: i });
    for (let i = 0; i < this.nSources; i++)
      actions.push({ type: "dec", idx: i });
    // epsilon-greedy random (no Q-network yet - heuristic bandit style)
    if (Math.random() < this.epsilon)
      return actions[Math.floor(Math.random() * actions.length)];
    // Simple heuristic: prefer increasing sources with higher state average prob of previous correct class (not accessible here), fallback random
    return actions[Math.floor(Math.random() * actions.length)];
  }
  applyAction(action) {
    // Smaller step to avoid large sudden weight shifts which can cause
    // longer stuck runs when a single source dominates.
    const delta = 0.02;
    if (action.type === "inc") this.weights[action.idx] += delta;
    else this.weights[action.idx] -= delta;
    // Clamp & renormalize
    for (let i = 0; i < this.weights.length; i++)
      this.weights[i] = Math.max(0.01, this.weights[i]);
    const sum = this.weights.reduce((a, b) => a + b, 0) || 1;
    this.weights = this.weights.map((w) => w / sum);
  }
  updateReward(reward) {
    // Adjust epsilon & learningRate heuristically
    if (reward <= 0) {
      this.epsilon = Math.min(0.5, this.epsilon + 0.01);
    } else {
      this.epsilon = Math.max(0.05, this.epsilon - 0.005);
    }
  }
  // Apply externally computed performance scores (not necessarily bounded).
  // perfScores: array of non-negative scores per source. alpha: smoothing factor [0..1]
  applyPerformanceWeights(perfScores = [], alpha = 0.7) {
    if (!Array.isArray(perfScores) || perfScores.length !== this.nSources)
      return;
    // normalize perfScores
    const clipped = perfScores.map((s) => (isFinite(s) && s > 0 ? s : 0));
    const sum = clipped.reduce((a, b) => a + b, 0) || 1;
    const norm = clipped.map((s) => s / sum);
    // blend with existing weights
    const blended = this.weights.map(
      (w, i) => alpha * norm[i] + (1 - alpha) * w
    );
    // clamp and renormalize
    for (let i = 0; i < blended.length; i++)
      blended[i] = Math.max(0.01, blended[i]);
    const s2 = blended.reduce((a, b) => a + b, 0) || 1;
    this.weights = blended.map((w) => w / s2);
  }
  toJSON() {
    return {
      nSources: this.nSources,
      learningRate: this.learningRate,
      epsilon: this.epsilon,
      weights: this.weights,
    };
  }
  static fromJSON(obj) {
    const d = new DQNWeights({
      nSources: obj.nSources || obj.weights.length,
      learningRate: obj.learningRate,
      epsilon: obj.epsilon,
    });
    d.weights = obj.weights;
    return d;
  }
}

// ---- Log-space blending ----
export function blendLogSpace(
  sourceProbArrays,
  weights,
  { temperature = 1.0, clampMax = 0.92 } = {}
) {
  const nSources = sourceProbArrays.length;
  if (!nSources) return [0.25, 0.25, 0.25, 0.25];
  const wSum = weights.reduce((a, b) => a + b, 0) || 1;
  const normW = weights.map((w) => w / wSum);
  const logBlend = [0, 0, 0, 0];
  sourceProbArrays.forEach((p, si) => {
    p.forEach((v, i) => {
      const vv = Math.min(0.999, Math.max(1e-6, v));
      logBlend[i] += normW[si] * Math.log(vv);
    });
  });
  // Temperature scaling
  const computeExp = (temp) =>
    logBlend.map((l) => Math.exp(l / Math.max(1e-6, temp)));
  let expVals = computeExp(temperature);
  const es = expVals.reduce((a, b) => a + b, 0) || 1;
  let probs = expVals.map((e) => e / es);
  // Entropy-based fallback: if resulting distribution is too low-entropy
  // (overconfident), increase temperature slightly to flatten probabilities.
  const entropy = probs.reduce((s, p) => (p > 0 ? s - p * Math.log2(p) : s), 0);
  const entropyThresh = 1.05; // target minimal entropy for 4 classes (~2.0 is max)
  if (entropy < entropyThresh) {
    const tempAdj = Math.min(
      2.0,
      temperature + (entropyThresh - entropy) * 0.6
    );
    expVals = computeExp(tempAdj);
    const es2 = expVals.reduce((a, b) => a + b, 0) || 1;
    probs = expVals.map((e) => e / es2);
  }
  // Overconfidence clamp
  const maxP = Math.max(...probs);
  if (maxP > clampMax) {
    probs = probs.map((p) => p * (clampMax / maxP));
    const sum2 = probs.reduce((a, b) => a + b, 0) || 1;
    probs = probs.map((p) => p / sum2);
  }
  return probs;
}

// ---- Auto Tuner ----
export function autoTuner({
  predictionRecords,
  history,
  hyperparams,
  dqn,
  evaluationWindow = 100,
}) {
  const n = history.length;
  if (n < 20) return hyperparams;
  const start = Math.max(0, n - evaluationWindow);
  let correct = 0;
  let brierSum = 0;
  for (let i = start; i < Math.min(predictionRecords.length, n); i++) {
    const rec = predictionRecords[i];
    if (!rec) continue;
    const truth = history[i];
    if (rec.predicted === truth) correct++;
    const probs = rec.probs || [0.25, 0.25, 0.25, 0.25];
    for (let k = 0; k < 4; k++) {
      const y = truth === k ? 1 : 0;
      brierSum += Math.pow(probs[k] - y, 2);
    }
  }
  const evalCount = Math.min(predictionRecords.length, n) - start || 1;
  const accuracy = correct / evalCount;
  const brier = brierSum / (evalCount * 4);
  const newHypers = { ...hyperparams };
  // Heuristic adjustments
  if (brier > 0.2) {
    newHypers.temperature = Math.min(1.5, (newHypers.temperature || 1) + 0.05);
    newHypers.clampMax = Math.min(0.98, (newHypers.clampMax || 0.92) + 0.01);
  } else if (brier < 0.12) {
    newHypers.temperature = Math.max(0.7, (newHypers.temperature || 1) - 0.03);
  }
  if (accuracy < 0.3) {
    newHypers.mcMarkovSims = Math.min(
      1000,
      (newHypers.mcMarkovSims || 300) + 50
    );
    newHypers.mcDropoutSamples = Math.min(
      15,
      (newHypers.mcDropoutSamples || 5) + 1
    );
    dqn.epsilon = Math.min(0.5, dqn.epsilon + 0.02);
    // If accuracy is very low, increase stuck-run penalty aggressiveness
    newHypers.stuckPenalty = Math.min(
      0.3,
      (newHypers.stuckPenalty || 0.12) + 0.05
    );
    newHypers.stuckRunThreshold = Math.max(
      6,
      (newHypers.stuckRunThreshold || 12) - 3
    );
  } else if (accuracy > 0.45) {
    newHypers.mcMarkovSims = Math.max(
      100,
      (newHypers.mcMarkovSims || 300) - 20
    );
    newHypers.mcDropoutSamples = Math.max(
      3,
      (newHypers.mcDropoutSamples || 5) - 1
    );
    dqn.epsilon = Math.max(0.05, dqn.epsilon - 0.01);
  }
  // EWMA lambda adjustments based on volatility (approx by brier)
  if (brier > 0.25)
    newHypers.ewmaLambda = Math.min(0.2, (newHypers.ewmaLambda || 0.12) + 0.01);
  else if (brier < 0.1)
    newHypers.ewmaLambda = Math.max(
      0.05,
      (newHypers.ewmaLambda || 0.12) - 0.005
    );
  return { ...newHypers, lastEval: { accuracy, brier, n } };
}
