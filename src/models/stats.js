// Statistical methods implementation (Step 3)
// All methods operate on history array of class indices 0..3

// Bayesian Updating (Dirichlet-multinomial)
export function bayesianProbs(history, priorAlpha = [1, 1, 1, 1]) {
  const alpha = [...priorAlpha];
  history.forEach((c) => {
    if (alpha[c] != null) alpha[c] += 1;
  });
  const sum = alpha.reduce((a, b) => a + b, 0) || 1;
  const probs = alpha.map((a) => a / sum);
  return { probs, alphaPosterior: alpha };
}

// Markov chain 4x4 transition matrix with Laplace smoothing (1)
export function markovProbs(history) {
  const counts = Array.from({ length: 4 }, () => Array(4).fill(1)); // smoothing
  for (let i = 1; i < history.length; i++) {
    const prev = history[i - 1];
    const curr = history[i];
    counts[prev][curr] += 1;
  }
  let probs = [0.25, 0.25, 0.25, 0.25];
  if (history.length > 0) {
    const last = history[history.length - 1];
    const row = counts[last];
    const sumRow = row.reduce((a, b) => a + b, 0) || 1;
    probs = row.map((v) => v / sumRow);
  }
  return { probs, counts };
}

// Streak / run-length analysis
// Simple heuristic: probability to continue grows with streak length but capped.
export function streakProbs(history) {
  if (!history.length)
    return { probs: [0.25, 0.25, 0.25, 0.25], streak: { len: 0, class: null } };
  let len = 1;
  const last = history[history.length - 1];
  for (let i = history.length - 2; i >= 0; i--) {
    if (history[i] === last) len++;
    else break;
  }
  // Continuation probability: base 0.5 increasing 0.05 per additional element, capped 0.85
  const pContinue = Math.min(0.85, 0.5 + (len - 1) * 0.05);
  const pOthers = (1 - pContinue) / 3;
  const probs = [0, 0, 0, 0].map((_, k) => (k === last ? pContinue : pOthers));
  return { probs, streak: { len, class: last, pContinue } };
}

// EWMA recent frequency estimator with dynamic lambda
export function ewmaProbs(history, prevState) {
  if (!history.length)
    return {
      probs: [0.25, 0.25, 0.25, 0.25],
      lambda: 0.15,
      values: [0.25, 0.25, 0.25, 0.25],
    };
  // Auto-tune lambda by history size (placeholder heuristic – future: optimization loop)
  const n = history.length;
  let lambda = 0.15;
  if (n > 100) lambda = 0.12;
  if (n > 200) lambda = 0.1;
  if (n > 300) lambda = 0.08;
  if (n > 500) lambda = 0.07;

  let values = prevState?.values || [0.25, 0.25, 0.25, 0.25];
  let processed = prevState?.processed || 0;

  // Only process new spins since last state for efficiency
  for (let i = processed; i < history.length; i++) {
    const cls = history[i];
    const oneHot = [0, 0, 0, 0];
    oneHot[cls] = 1;
    values = values.map((v, idx) => (1 - lambda) * v + lambda * oneHot[idx]);
  }
  const sum = values.reduce((a, b) => a + b, 0) || 1;
  const probs = values.map((v) => v / sum);
  return { probs, lambda, values, processed: history.length };
}

// Aggregate method (optional future ensemble) placeholder
export function aggregateStatsMethods(methods) {
  // Simple average blend for now (not yet used for main prediction in Step 3)
  const arrs = methods.map((m) => m.probs);
  if (!arrs.length) return [0.25, 0.25, 0.25, 0.25];
  const sum = [0, 0, 0, 0];
  arrs.forEach((p) => p.forEach((v, i) => (sum[i] += v)));
  return sum.map((v) => v / arrs.length);
}
