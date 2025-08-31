// Clean, single implementation for ensemble helpers.
// Exports: monteCarloMarkov, hmmProbs, DQNWeights, blendLogSpace, chiSquaredStat,
// detectFreqBias, shiftProbsTowardObserved, autoTuner

// Monte Carlo Markov simulation from transition counts.
export function monteCarloMarkov(
  counts,
  lastClass,
  { numSims = 200, horizon = 1 } = {}
) {
  if (lastClass == null) return { probs: [0.25, 0.25, 0.25, 0.25], numSims };
  const row = counts && counts[lastClass] ? counts[lastClass] : [1, 1, 1, 1];
  const sum = row.reduce((a, b) => a + b, 0) || 1;
  const trans = row.map((v) => v / sum);
  const tallies = [0, 0, 0, 0];
  for (let s = 0; s < numSims; s++) {
    let cls = lastClass;
    for (let h = 0; h < horizon; h++) {
      const r = Math.random();
      let acc = 0;
      for (let k = 0; k < 4; k++) {
        acc += trans[k];
        if (r <= acc) {
          cls = k;
          break;
        }
      }
    }
    tallies[cls]++;
  }
  const tSum = tallies.reduce((a, b) => a + b, 0) || 1;
  return { probs: tallies.map((v) => v / tSum), numSims };
}

// Small HMM-like regime detector that returns an emission distribution.
export function hmmProbs(history, { window = 200 } = {}) {
  if (!history || history.length === 0)
    return { probs: [0.25, 0.25, 0.25, 0.25] };
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
  if (freqVecs.length < 2) return { probs: [0.25, 0.25, 0.25, 0.25] };
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
    d0 + d1 === 0 ? [0.5, 0.5] : [d1 / (d0 + d1), d0 / (d0 + d1)];
  const emission = [0, 0, 0, 0];
  for (let s = 0; s < 2; s++)
    emission.forEach(
      (_, i) => (emission[i] += stateProbs[s] * centroids[s][i])
    );
  const esum = emission.reduce((a, b) => a + b, 0) || 1;
  return {
    probs: emission.map((e) => e / esum),
    emissions: centroids,
    stateProbs,
  };
}

// Lightweight DQN-like weight container for ensemble sources.
export class DQNWeights {
  constructor({ nSources, learningRate = 0.01, epsilon = 0.2 }) {
    this.nSources = nSources;
    this.learningRate = learningRate;
    this.epsilon = epsilon;
    this.weights = Array(nSources).fill(1 / nSources);
    this._lastReward = 0;
    this._lastAction = null;
  }
  applyBiasAdjustment(observedFreq = [0.25, 0.25, 0.25, 0.25], options = {}) {
    const {
      boostFactor = 1.5,
      bayesIdx = 0,
      ewmaIdx = 3,
      alpha = 0.6,
    } = options;
    if (!Array.isArray(observedFreq) || observedFreq.length !== 4) return;
    const adj = this.weights.slice();
    if (bayesIdx >= 0 && bayesIdx < adj.length) adj[bayesIdx] *= boostFactor;
    if (ewmaIdx >= 0 && ewmaIdx < adj.length) adj[ewmaIdx] *= boostFactor;
    const blended = adj.map(
      (v, i) => alpha * v + (1 - alpha) * this.weights[i]
    );
    for (let i = 0; i < blended.length; i++)
      blended[i] = Math.max(0.01, blended[i]);
    const s = blended.reduce((a, b) => a + b, 0) || 1;
    this.weights = blended.map((w) => w / s);
  }
  applyPerformanceWeights(perfScores = [], alpha = 0.7) {
    if (!Array.isArray(perfScores) || perfScores.length !== this.nSources)
      return;
    const clipped = perfScores.map((s) => (isFinite(s) && s > 0 ? s : 0));
    const sum = clipped.reduce((a, b) => a + b, 0) || 1;
    const norm = clipped.map((s) => s / sum);
    const blended = this.weights.map(
      (w, i) => alpha * norm[i] + (1 - alpha) * w
    );
    for (let i = 0; i < blended.length; i++)
      blended[i] = Math.max(0.01, blended[i]);
    const s2 = blended.reduce((a, b) => a + b, 0) || 1;
    this.weights = blended.map((w) => w / s2);
  }

  // Update internal exploration parameter based on reward signal
  updateReward(reward) {
    if (!isFinite(reward)) return;
    this._lastReward = reward;
    // Simple heuristic: reward > 0 -> reduce epsilon (more exploitation), otherwise increase
    if (reward > 0) {
      this.epsilon = Math.max(0.01, this.epsilon - 0.01);
    } else {
      this.epsilon = Math.min(0.9, this.epsilon + 0.02);
    }
  }

  // Choose an action (source index). Uses epsilon-greedy over current weights.
  chooseAction(_info) {
    // _info may be passed (e.g., probs), but we select based on internal weights
    if (Math.random() < (this.epsilon || 0)) {
      const a = Math.floor(Math.random() * Math.max(1, this.nSources));
      this._lastAction = a;
      return a;
    }
    let best = 0;
    for (let i = 1; i < this.weights.length; i++)
      if (this.weights[i] > this.weights[best]) best = i;
    this._lastAction = best;
    return best;
  }

  // Apply chosen action: reinforce or punish based on last reward and learningRate
  applyAction(action) {
    if (action == null || action < 0 || action >= this.nSources) return;
    const r = isFinite(this._lastReward) ? this._lastReward : 0;
    // simple additive update scaled by learningRate
    const delta = (this.learningRate || 0.01) * r;
    this.weights[action] = Math.max(0.001, (this.weights[action] || 0) + delta);
    // optionally decay others slightly
    for (let i = 0; i < this.weights.length; i++)
      if (i !== action)
        this.weights[i] = Math.max(0.001, this.weights[i] * 0.995);
    const s = this.weights.reduce((a, b) => a + b, 0) || 1;
    this.weights = this.weights.map((w) => w / s);
  }

  // Return a JSON-serializable representation of the DQNWeights state
  toJSON() {
    return {
      nSources: this.nSources,
      learningRate: this.learningRate,
      epsilon: this.epsilon,
      weights: Array.isArray(this.weights) ? this.weights.slice() : [],
      lastReward: this._lastReward,
      lastAction: this._lastAction,
    };
  }

  // Reconstruct a DQNWeights instance from a serialized object
  static fromJSON(obj = {}) {
    const inst = new DQNWeights({
      nSources: obj.nSources || (Array.isArray(obj.weights) ? obj.weights.length : 0),
      learningRate: typeof obj.learningRate === 'number' ? obj.learningRate : 0.01,
      epsilon: typeof obj.epsilon === 'number' ? obj.epsilon : 0.2,
    });
    if (Array.isArray(obj.weights) && obj.weights.length === inst.nSources) {
      inst.weights = obj.weights.slice();
    }
    inst._lastReward = typeof obj.lastReward === 'number' ? obj.lastReward : 0;
    inst._lastAction = obj.lastAction ?? null;
    return inst;
  }
}

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
  sourceProbArrays.forEach((p, si) =>
    p.forEach((v, i) => {
      const vv = Math.min(0.999, Math.max(1e-6, v));
      logBlend[i] += normW[si] * Math.log(vv);
    })
  );
  const computeExp = (temp) =>
    logBlend.map((l) => Math.exp(l / Math.max(1e-6, temp)));
  let expVals = computeExp(temperature);
  const es = expVals.reduce((a, b) => a + b, 0) || 1;
  let probs = expVals.map((e) => e / es);
  const entropy = probs.reduce((s, p) => (p > 0 ? s - p * Math.log2(p) : s), 0);
  const entropyThresh = 1.05;
  if (entropy < entropyThresh) {
    const tempAdj = Math.min(
      2.0,
      temperature + (entropyThresh - entropy) * 0.6
    );
    expVals = computeExp(tempAdj);
    const es2 = expVals.reduce((a, b) => a + b, 0) || 1;
    probs = expVals.map((e) => e / es2);
  }
  const maxP = Math.max(...probs);
  if (maxP > clampMax) {
    probs = probs.map((p) => p * (clampMax / maxP));
    const sum2 = probs.reduce((a, b) => a + b, 0) || 1;
    probs = probs.map((p) => p / sum2);
  }
  return probs;
}

export function chiSquaredStat(observedCounts, expectedFreq) {
  if (!Array.isArray(observedCounts) || !Array.isArray(expectedFreq))
    return NaN;
  const n = observedCounts.reduce((a, b) => a + b, 0);
  if (!n || n <= 0) return NaN;
  let stat = 0;
  for (let i = 0; i < expectedFreq.length; i++) {
    const exp = Math.max(1e-8, expectedFreq[i] * n);
    const obs = observedCounts[i] || 0;
    const d = obs - exp;
    stat += (d * d) / exp;
  }
  return stat;
}

export function detectFreqBias(observedCounts, expectedFreq, options = {}) {
  const { alpha = 0.05, minSamples = 50 } = options;
  const n = observedCounts.reduce((a, b) => a + b, 0);
  if (n < minSamples) return { biased: false, pValue: 1, chi2: 0 };
  const chi2 = chiSquaredStat(observedCounts, expectedFreq);
  const critical = options.critical ?? 7.815; // df=3 alpha=0.05
  return {
    biased: chi2 > critical,
    pValue: chi2 > critical ? 0.049 : 0.95,
    chi2,
  };
}

export function shiftProbsTowardObserved(probs, observedFreq, factor = 0.2) {
  if (!probs || !observedFreq) return probs;
  const out = probs.map((p, i) => p * (1 - factor) + observedFreq[i] * factor);
  const s = out.reduce((a, b) => a + b, 0) || 1;
  return out.map((v) => Math.max(1e-8, v / s));
}

export function autoTuner({
  predictionRecords,
  history,
  hyperparams,
  dqn,
  evaluationWindow = 100,
  trials = 20,
  biasWindow = 500,
} = {}) {
  const n = history.length;
  if (n < 20) return hyperparams;
  const start = Math.max(0, n - evaluationWindow);

  // baseline eval on bet spins
  let baseCorrect = 0,
    baseBrier = 0,
    baseCount = 0;
  for (let i = start; i < Math.min(predictionRecords.length, n); i++) {
    const rec = predictionRecords[i];
    if (!rec) continue;
    const truth = history[i];
    const probs = rec.probs || [0.25, 0.25, 0.25, 0.25];
    const maxP = Math.max(...probs);
    const thresh = hyperparams.predictionConfidenceThreshold ?? 0.3;
    if (maxP < thresh) continue;
    const predicted =
      rec.predicted == null ? probs.indexOf(maxP) : rec.predicted;
    if (predicted === truth) baseCorrect++;
    for (let k = 0; k < 4; k++) {
      const y = truth === k ? 1 : 0;
      baseBrier += Math.pow(probs[k] - y, 2);
    }
    baseCount++;
  }
  const baseEvalCount = Math.max(1, baseCount);
  const baseAccuracy = baseCorrect / baseEvalCount;
  const baseBrierScore = baseBrier / (baseEvalCount * 4);

  const bw = Math.min(biasWindow, history.length);
  const obsStart = Math.max(0, history.length - bw);
  const observedCounts = [0, 0, 0, 0];
  for (let i = obsStart; i < history.length; i++) {
    const c = history[i];
    if (typeof c === "number" && c >= 0 && c < 4) observedCounts[c]++;
  }
  const expected = [1 / 37, 12 / 37, 12 / 37, 12 / 37];
  const biasRes = detectFreqBias(observedCounts, expected, { minSamples: 40 });
  const biasDetected = biasRes?.biased === true;

  const randBetween = (a, b) => a + Math.random() * (b - a);
  const makeCandidate = () => {
    const cand = { ...hyperparams };
    cand.temperature = parseFloat(randBetween(0.6, 1.6).toFixed(3));
    cand.clampMax = parseFloat(randBetween(0.85, 0.99).toFixed(3));
    cand.mcMarkovSims = Math.round(randBetween(100, 1000));
    cand.mcDropoutSamples = Math.round(randBetween(3, 15));
    cand.ewmaLambda = parseFloat(
      randBetween(
        biasDetected ? 0.08 : 0.02,
        biasDetected ? 0.35 : 0.18
      ).toFixed(3)
    );
    cand.stuckPenalty = parseFloat(randBetween(0.05, 0.35).toFixed(3));
    cand.stuckRunThreshold = Math.round(randBetween(6, 20));
    cand.predictionConfidenceThreshold = parseFloat(
      randBetween(biasDetected ? 0.22 : 0.2, biasDetected ? 0.5 : 0.6).toFixed(
        3
      )
    );
    return cand;
  };

  let best = { score: -Infinity, hypers: hyperparams };
  // determine number of ensemble sources from predictionRecords (if available)
  const sampleRec =
    predictionRecords &&
    predictionRecords.find((r) => r && Array.isArray(r.sourceProbs));
  const nSources = sampleRec ? sampleRec.sourceProbs.length : 0;
  const corrPenaltyWeight =
    hyperparams && typeof hyperparams.corrPenalty === "number"
      ? hyperparams.corrPenalty
      : 0.2;

  for (let t = 0; t < Math.max(5, trials); t++) {
    const cand = makeCandidate();
    let cCorrect = 0,
      cBrier = 0,
      cCount = 0;
    // prepare per-source flattened probability traces to compute correlation
    const sourceTraces = Array.from(
      { length: Math.max(0, nSources) },
      () => []
    );
    for (let i = start; i < Math.min(predictionRecords.length, n); i++) {
      const rec = predictionRecords[i];
      if (!rec) continue;
      const truth = history[i];
      const probs = rec.probs || [0.25, 0.25, 0.25, 0.25];
      const maxP = Math.max(...probs);
      if (maxP < (cand.predictionConfidenceThreshold ?? 0.3)) continue;
      const predicted =
        rec.predicted == null ? probs.indexOf(maxP) : rec.predicted;
      if (predicted === truth) cCorrect++;
      for (let k = 0; k < 4; k++) {
        const y = truth === k ? 1 : 0;
        cBrier += Math.pow(probs[k] - y, 2);
      }
      cCount++;
      // collect source probs if present
      if (
        Array.isArray(rec.sourceProbs) &&
        rec.sourceProbs.length === nSources
      ) {
        for (let s = 0; s < nSources; s++) {
          const sp = rec.sourceProbs[s];
          if (!Array.isArray(sp) || sp.length !== 4) {
            // push a neutral vector flattened (0.25 repeated) to keep alignment
            sourceTraces[s].push(0.25, 0.25, 0.25, 0.25);
          } else {
            sourceTraces[s].push(sp[0], sp[1], sp[2], sp[3]);
          }
        }
      }
    }
    const cEvalCount = Math.max(1, cCount);
    const cAcc = cCorrect / cEvalCount;
    const cB = cBrier / (cEvalCount * 4);
    let score = cAcc - 0.5 * cB;
    if (biasDetected && biasRes && biasRes.chi2) {
      const biasStrength = Math.min(1, biasRes.chi2 / (biasRes.chi2 + 10));
      score += biasStrength * (cand.ewmaLambda || 0) * 0.6;
    }
    // penalize expensive candidates slightly
    score -= (cand.mcMarkovSims || 0) / 20000;

    // compute average absolute Pearson correlation between sources (if we have traces)
    if (sourceTraces.length > 1) {
      const m = sourceTraces[0].length;
      if (m > 0) {
        const pairCorrs = [];
        for (let a = 0; a < sourceTraces.length; a++) {
          for (let b = a + 1; b < sourceTraces.length; b++) {
            const A = sourceTraces[a];
            const B = sourceTraces[b];
            // compute Pearson r between A and B
            let sumA = 0,
              sumB = 0;
            for (let i = 0; i < m; i++) {
              sumA += A[i];
              sumB += B[i];
            }
            const meanA = sumA / m,
              meanB = sumB / m;
            let num = 0,
              denA = 0,
              denB = 0;
            for (let i = 0; i < m; i++) {
              const da = A[i] - meanA;
              const db = B[i] - meanB;
              num += da * db;
              denA += da * da;
              denB += db * db;
            }
            const denom = Math.sqrt(Math.max(1e-12, denA * denB));
            const r = denom > 0 ? num / denom : 0;
            pairCorrs.push(Math.abs(r));
          }
        }
        if (pairCorrs.length) {
          const avgAbsCorr =
            pairCorrs.reduce((a, b) => a + b, 0) / pairCorrs.length;
          // penalize the candidate score proportional to avgAbsCorr and corrPenaltyWeight
          score -= (corrPenaltyWeight || 0) * avgAbsCorr;
        }
      }
    }

    if (score > best.score)
      best = { score, hypers: cand, acc: cAcc, brier: cB, count: cEvalCount };
  }

  let newHypers = { ...hyperparams, ...best.hypers };
  // Overfitting check: use rollingMetrics if available to compute train/val brier
  let divergenceDetected = false;
  try {
    const metrics =
      typeof exports !== "undefined" && exports.rollingMetrics
        ? exports.rollingMetrics(history, predictionRecords, 0, {
            confidenceThreshold: newHypers.predictionConfidenceThreshold,
            valWindow: Math.floor(evaluationWindow / 2),
          })
        : null;
    if (metrics && metrics.train && metrics.val) {
      const trainB =
        metrics.train.brierArr && metrics.train.brierArr.length
          ? metrics.train.brierArr.reduce((a, b) => a + b, 0) /
            metrics.train.brierArr.length
          : null;
      const valB =
        metrics.val.brierArr && metrics.val.brierArr.length
          ? metrics.val.brierArr.reduce((a, b) => a + b, 0) /
            metrics.val.brierArr.length
          : null;
      if (trainB != null && valB != null && valB > trainB + 0.05) {
        divergenceDetected = true;
        // increase dropout and reduce epochs in the chosen hyperparams
        const curDrop =
          typeof newHypers.dropout === "number" ? newHypers.dropout : 0.25;
        newHypers.dropout = Math.min(0.6, curDrop + 0.1);
        const curEpochs =
          typeof newHypers.epochs === "number"
            ? Math.max(1, newHypers.epochs)
            : 3;
        newHypers.epochs = Math.max(1, Math.floor(curEpochs / 2));
      }
    }
  } catch (err) {
    // ignore
  }

  if (best.acc != null) {
    if (best.acc < 0.3) dqn.epsilon = Math.min(0.5, dqn.epsilon + 0.02);
    else if (best.acc > 0.45) dqn.epsilon = Math.max(0.05, dqn.epsilon - 0.01);
  }

  return {
    ...newHypers,
    lastEval: {
      accuracy: baseAccuracy,
      brier: baseBrierScore,
      n,
      biasDetected,
      biasChi2: biasRes?.chi2 ?? 0,
      chosen: { acc: best.acc, brier: best.brier, score: best.score },
      divergenceDetected,
    },
  };
}
