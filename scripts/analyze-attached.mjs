import fs from "fs";
const path = "c:/Users/corne/Downloads/roulette_state (6).json";
const raw = fs.readFileSync(path, "utf8");
const data = JSON.parse(raw);
const history = data.history || [];
const pr = data.predictionRecords || [];
function counts(arr) {
  const c = [0, 0, 0, 0];
  arr.forEach((v) => {
    if (v >= 0 && v < 4) c[v]++;
  });
  return c;
}
console.log("history length", history.length);
console.log("class counts", counts(history));
let matches = 0,
  cnt = 0,
  brier = 0,
  skipped = 0;
const maxProbDist = [];
const classConfusion = [
  [0, 0, 0, 0],
  [0, 0, 0, 0],
  [0, 0, 0, 0],
  [0, 0, 0, 0],
];
for (let i = 0; i < Math.min(history.length, pr.length); i++) {
  const r = pr[i];
  const truth = history[i];
  if (!r) continue;
  const probs =
    Array.isArray(r.probs) && r.probs.length === 4
      ? r.probs
      : r.sourceProbs && r.sourceProbs[0]
      ? r.sourceProbs[0]
      : [0.25, 0.25, 0.25, 0.25];
  const maxP = Math.max(...probs);
  maxProbDist.push(maxP);
  if (r.skipped) skipped++;
  if (typeof r.predicted === "number") {
    cnt++;
    if (r.predicted === truth) matches++;
    classConfusion[truth][r.predicted]++;
  }
  for (let k = 0; k < 4; k++) {
    const y = truth === k ? 1 : 0;
    brier += Math.pow((probs[k] || 0) - y, 2);
  }
}
console.log("predictionRecords length", pr.length);
console.log("predicted count (non-null)", cnt);
console.log(
  "skipped count",
  skipped,
  "skip rate",
  ((skipped / (pr.length || 1)) * 100).toFixed(2) + "%"
);
console.log(
  "raw accuracy (on predicted)",
  cnt ? (matches / cnt).toFixed(4) : "n/a"
);
console.log(
  "avg brier (on all aligned)",
  (brier / ((Math.min(history.length, pr.length) || 1) * 4)).toFixed(4)
);
// moving window accuracy last 50/100/500
function windowAcc(w) {
  let m = 0,
    c = 0;
  for (let i = Math.max(0, pr.length - w); i < pr.length; i++) {
    const r = pr[i];
    if (!r) continue;
    const truth = history[i];
    if (typeof r.predicted === "number") {
      c++;
      if (r.predicted === truth) m++;
    }
  }
  return c ? m / c : NaN;
}
[50, 100, 250, 500].forEach((w) => console.log("acc last", w, windowAcc(w)));
// maxProb histogram
const buckets = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
maxProbDist.forEach((p) => {
  const idx = Math.min(9, Math.floor(p * 10));
  buckets[idx]++;
});
console.log(
  "maxProb histogram (0-0.1,...0.9-1.0):",
  buckets.map((x) => x)
);
console.log("class confusion matrix (rows=truth,cols=pred):");
console.table(classConfusion);
// per-source performance if present
const nSources =
  pr[0] && Array.isArray(pr[0].sourceProbs) ? pr[0].sourceProbs.length : 0;
if (nSources) {
  const stats = Array.from({ length: nSources }, () => ({
    correct: 0,
    brier: 0,
    cnt: 0,
  }));
  for (let i = 0; i < Math.min(history.length, pr.length); i++) {
    const r = pr[i];
    if (!r || !Array.isArray(r.sourceProbs)) continue;
    const truth = history[i];
    for (let s = 0; s < r.sourceProbs.length; s++) {
      const sp = r.sourceProbs[s];
      if (!Array.isArray(sp) || sp.length !== 4) continue;
      const pred = sp.indexOf(Math.max(...sp));
      if (pred === truth) stats[s].correct++;
      for (let k = 0; k < 4; k++) {
        const y = truth === k ? 1 : 0;
        stats[s].brier += Math.pow(sp[k] - y, 2);
      }
      stats[s].cnt++;
    }
  }
  console.log("per-source performance:");
  stats.forEach((s, i) => {
    const acc = s.cnt ? s.correct / s.cnt : 0;
    const avgB = s.cnt ? s.brier / (s.cnt * 4) : 0;
    console.log(i, "acc", acc.toFixed(4), "brier", avgB.toFixed(4), "n", s.cnt);
  });
}
console.log("done");
