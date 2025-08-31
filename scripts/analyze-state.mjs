import fs from "fs";
const path = "c:/Users/corne/Downloads/roulette_state (5).json";
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
// basic accuracy on aligned records
let matches = 0,
  cnt = 0,
  brier = 0;
for (let i = 0; i < Math.min(history.length, pr.length); i++) {
  const r = pr[i];
  if (!r) continue;
  const truth = history[i];
  if (typeof r.predicted === "number") {
    cnt++;
    if (r.predicted === truth) matches++;
  }
  const probs =
    Array.isArray(r.probs) && r.probs.length === 4
      ? r.probs
      : r.sourceProbs && r.sourceProbs[0]
      ? r.sourceProbs[0]
      : [0.25, 0.25, 0.25, 0.25];
  for (let k = 0; k < 4; k++) {
    const y = truth === k ? 1 : 0;
    brier += (probs[k] - y) * (probs[k] - y);
  }
}
console.log("predictionRecords length", pr.length);
console.log("aligned records used for metrics", cnt);
console.log("raw accuracy", cnt ? (matches / cnt).toFixed(4) : "n/a");
console.log("avg brier", cnt ? (brier / (cnt * 4)).toFixed(4) : "n/a");
// per-source perf
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
        stats[s].brier += (sp[k] - y) * (sp[k] - y);
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
// streak analysis
let maxRun = 0,
  curRun = 1;
for (let i = 1; i < history.length; i++) {
  if (history[i] === history[i - 1]) {
    curRun++;
  } else {
    if (curRun > maxRun) maxRun = curRun;
    curRun = 1;
  }
}
if (curRun > maxRun) maxRun = curRun;
console.log("max run length in history", maxRun);
// recent drift: distribution over last 500
const tail = history.slice(-500);
console.log("tail counts", counts(tail));
console.log("done");
