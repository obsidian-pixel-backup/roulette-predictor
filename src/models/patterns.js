// Simple pattern recognition utilities.
// Scans recent history for repeating sequences and estimates next-class probabilities
// based on observed continuations. Lightweight and heuristic — meant to augment
// statistical sources in the ensemble.

export function patternProbs(
  history,
  { window = 300, maxPattern = 8, minCount = 2, ewmaLambda = 0.08 } = {}
) {
  if (!history || history.length === 0)
    return { probs: [0.25, 0.25, 0.25, 0.25] };
  const recent = history.slice(-window);

  // Build pattern continuation counts for lengths 2..maxPattern
  const patterns = {}; // patterns[L] = Map key->Map(next->count)
  for (let L = 2; L <= Math.min(maxPattern, recent.length - 1); L++) {
    patterns[L] = Object.create(null);
    for (let i = 0; i + L < recent.length; i++) {
      const key = recent.slice(i, i + L).join(",");
      const next = recent[i + L];
      const map = patterns[L][key] || (patterns[L][key] = [0, 0, 0, 0]);
      map[next] = (map[next] || 0) + 1;
    }
  }

  // Try longest-to-shortest match for the suffix of recent
  for (let L = Math.min(maxPattern, recent.length); L >= 2; L--) {
    const suffix = recent.slice(-L).join(",");
    const bucket = patterns[L] && patterns[L][suffix];
    if (bucket) {
      const sum = bucket.reduce((a, b) => a + (b || 0), 0) || 0;
      if (sum >= minCount) {
        const probs = bucket.map((c) => (c || 0) / sum);
        return { probs };
      }
    }
  }

  // Detect simple periodic/alternating patterns by scanning for small periods
  // Check periods 2..6 and if suffix matches repeating pattern, predict next accordingly
  for (let p = 2; p <= Math.min(6, recent.length); p++) {
    const pattern = recent.slice(-p);
    if (pattern.length < p) continue;
    // check if last few elements correspond to repeating pattern
    let repeats = 0;
    for (let i = recent.length - 1; i >= 0; i--) {
      const expected = pattern[(recent.length - 1 - i) % p];
      if (recent[i] === expected) repeats++;
      else break;
    }
    if (repeats >= p) {
      // next predicted element is pattern[ repeats % p ]
      const next = pattern[repeats % p];
      const probs = [0, 0, 0, 0];
      probs[next] = 1;
      return { probs };
    }
  }

  // Fallback: no pattern found. Use an EWMA of recent classes so the
  // fallback reflects short-term bias instead of a fixed, overconfident vector.
  // ewmaLambda controls how quickly recent spins dominate.
  const ewma = [0, 0, 0, 0];
  for (let i = 0; i < recent.length; i++) {
    const cls = recent[i];
    for (let k = 0; k < 4; k++) {
      ewma[k] = ewma[k] * (1 - ewmaLambda) + (cls === k ? ewmaLambda : 0);
    }
  }
  const s = ewma.reduce((a, b) => a + b, 0) || 1;
  const probs = ewma.map((v) => Math.max(1e-8, v / s));
  return { probs };
}

export default { patternProbs };
