import fs from "fs";
const path = "c:/Users/corne/Downloads/roulette_state (6).json";
const raw = fs.readFileSync(path, "utf8");
const data = JSON.parse(raw);
const history = data.history || [];
const pr = data.predictionRecords || [];
// Collect sample records where sourceProbs exist and length>0
const samples = [];
for (let i = 0; i < Math.min(history.length, pr.length); i++) {
  const r = pr[i];
  if (!r) continue;
  if (!Array.isArray(r.sourceProbs) || r.sourceProbs.length === 0) continue;
  // flatten each source prob into an array of 4
  samples.push({ idx: i, truth: history[i], sourceProbs: r.sourceProbs });
}
console.log("usable samples", samples.length);
if (samples.length === 0) {
  console.log("no sourceProbs found, abort");
  process.exit(0);
}
const nSources = samples[0].sourceProbs.length;
console.log("nSources", nSources);
// helper to blend in log-space
function blendLog(sourceProbs, weights) {
  // weights length nSources
  const wsum = weights.reduce((a, b) => a + b, 0) || 1;
  const normW = weights.map((w) => w / wsum);
  const logBlend = [0, 0, 0, 0];
  for (let s = 0; s < sourceProbs.length; s++) {
    const sp = sourceProbs[s];
    for (let k = 0; k < 4; k++) {
      const v = Math.min(0.999999, Math.max(1e-12, sp[k] || 1e-12));
      logBlend[k] += normW[s] * Math.log(v);
    }
  }
  // softmax
  const maxL = Math.max(...logBlend);
  const exps = logBlend.map((l) => Math.exp(l - maxL));
  const s = exps.reduce((a, b) => a + b, 0) || 1;
  return exps.map((e) => e / s);
}

function scoreWeights(weights) {
  let correct = 0,
    total = 0;
  for (const samp of samples) {
    const probs = blendLog(samp.sourceProbs, weights);
    const pred = probs.indexOf(Math.max(...probs));
    total++;
    if (pred === samp.truth) correct++;
  }
  return correct / total;
}
// random search
let best = { score: -1, weights: null };
const trials = 4000;
for (let t = 0; t < trials; t++) {
  const w = Array.from({ length: nSources }, () => Math.random());
  const sc = scoreWeights(w);
  if (sc > best.score) {
    best = { score: sc, weights: w };
  }
}
console.log(
  "best score",
  best.score,
  "weights",
  best.weights.map((x) => x.toFixed(3))
);
// also evaluate uniform
const uniform = Array(nSources).fill(1);
console.log("uniform score", scoreWeights(uniform));

// try grid around best with gaussian perturbations
let curBest = best;
for (let round = 0; round < 5; round++) {
  for (let k = 0; k < 2000; k++) {
    const cand = curBest.weights.map((w) =>
      Math.max(1e-6, w * (1 + (Math.random() - 0.5) * 0.3))
    );
    const sc = scoreWeights(cand);
    if (sc > curBest.score) curBest = { score: sc, weights: cand };
  }
}
console.log(
  "refined best",
  curBest.score,
  curBest.weights.map((x) => x.toFixed(3))
);
// show confusion matrix for best
const bestW = curBest.weights;
const conf = [
  [0, 0, 0, 0],
  [0, 0, 0, 0],
  [0, 0, 0, 0],
  [0, 0, 0, 0],
];
let correct = 0;
for (const samp of samples) {
  const probs = blendLog(samp.sourceProbs, bestW);
  const pred = probs.indexOf(Math.max(...probs));
  conf[samp.truth][pred]++;
  if (pred === samp.truth) correct++;
}
console.log(
  "best accuracy on blended sources",
  (correct / samples.length).toFixed(4)
);
console.table(conf);
fs.writeFileSync(
  "c:/DEVELOPER-PROJECTS/roulette-predictor/scripts/optimize-ensemble-result.json",
  JSON.stringify({ best: curBest, accuracy: correct / samples.length }, null, 2)
);
console.log("done");
