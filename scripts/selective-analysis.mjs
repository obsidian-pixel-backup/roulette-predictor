import fs from "fs";
const path = "c:/Users/corne/Downloads/roulette_state (6).json";
const raw = fs.readFileSync(path, "utf8");
const data = JSON.parse(raw);
const history = data.history || [];
const pr = data.predictionRecords || [];
function evalAt(thresh) {
  let correct = 0,
    count = 0;
  for (let i = 0; i < Math.min(history.length, pr.length); i++) {
    const r = pr[i];
    if (!r || !r.probs) continue;
    const p = r.probs;
    const maxP = Math.max(...p);
    if (maxP < thresh) continue;
    const pred = p.indexOf(maxP);
    if (pred === history[i]) correct++;
    count++;
  }
  return {
    thresh,
    acc: count ? correct / count : NaN,
    coverage: count / pr.length,
  };
}
const out = [];
for (let t = 0.05; t <= 0.95; t += 0.01) {
  out.push(evalAt(parseFloat(t.toFixed(2))));
}
out.sort((a, b) => (b.acc || 0) - (a.acc || 0));
console.log("top 10 thresholds by accuracy:");
console.table(out.slice(0, 10));
console.log("best overall (by accuracy)");
console.log(out[0]);
console.log("done");
