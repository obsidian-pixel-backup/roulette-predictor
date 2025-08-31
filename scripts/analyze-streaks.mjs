import fs from "fs";
const path =
  process.argv[2] || "c:/Users/corne/Downloads/roulette_state (10).json";
let raw;
try {
  raw = fs.readFileSync(path, "utf8");
} catch (e) {
  console.error("READ_ERR", e.message);
  process.exit(2);
}
const data = JSON.parse(raw);
const history = data.history || [];
const pr = data.predictionRecords || [];
function counts(arr) {
  const c = [0, 0, 0, 0];
  arr.forEach((v) => {
    if (typeof v === "number" && v >= 0 && v < 4) c[v]++;
  });
  return c;
}
console.log("path", path);
console.log("history length", history.length);
console.log("class counts", counts(history));
let matches = 0,
  cnt = 0,
  skipped = 0,
  brier = 0;
const maxProbDist = [];
const correctSeq = [];
for (let i = 0; i < Math.min(history.length, pr.length); i++) {
  const r = pr[i];
  const truth = history[i];
  if (!r) {
    correctSeq.push(null);
    continue;
  }
  const probs =
    Array.isArray(r.probs) && r.probs.length === 4
      ? r.probs
      : r.sourceProbs && r.sourceProbs[0]
      ? r.sourceProbs[0]
      : [0.25, 0.25, 0.25, 0.25];
  const maxP = Math.max(...probs);
  maxProbDist.push(maxP);
  if (r.skipped) skipped++;
  const predicted =
    typeof r.predicted === "number" ? r.predicted : probs.indexOf(maxP);
  if (typeof predicted === "number") {
    cnt++;
    const ok = predicted === truth;
    if (ok) matches++;
    correctSeq.push(ok ? 1 : 0);
  } else correctSeq.push(null);
  for (let k = 0; k < 4; k++) {
    const y = truth === k ? 1 : 0;
    brier += Math.pow((probs[k] || 0) - y, 2);
  }
}
const accuracy = cnt ? matches / cnt : NaN;
console.log(
  "predicted count",
  cnt,
  "skipped",
  skipped,
  "skip rate",
  ((skipped / (pr.length || 1)) * 100).toFixed(3) + "%"
);
console.log("raw accuracy (on predicted)", accuracy.toFixed(4));
console.log(
  "avg brier (on aligned)",
  (brier / ((Math.min(history.length, pr.length) || 1) * 4)).toFixed(4)
);
// losing streaks: contiguous false values in correctSeq ignoring nulls
let longest = 0,
  cur = 0,
  totalStreaks = 0,
  sumLengths = 0;
const streakCounts = {};
for (let i = 0; i < correctSeq.length; i++) {
  const v = correctSeq[i];
  if (v === 0) {
    cur++;
  } else {
    if (cur > 0) {
      totalStreaks++;
      sumLengths += cur;
      streakCounts[cur] = (streakCounts[cur] || 0) + 1;
      if (cur > longest) longest = cur;
    }
    cur = 0;
  }
}
if (cur > 0) {
  totalStreaks++;
  sumLengths += cur;
  streakCounts[cur] = (streakCounts[cur] || 0) + 1;
  if (cur > longest) longest = cur;
}
const avgStreak = totalStreaks ? sumLengths / totalStreaks : 0;
console.log(
  "losing streaks total",
  totalStreaks,
  "longest",
  longest,
  "avg length",
  avgStreak.toFixed(2)
);
// show distribution for small lengths
const smallDist = [];
for (let l = 1; l <= 20; l++) smallDist.push(streakCounts[l] || 0);
console.log("streak length counts (1..20):", smallDist);
// examine confidence (maxP) in losing vs winning
let winMaxSum = 0,
  winN = 0,
  loseMaxSum = 0,
  loseN = 0;
for (let i = 0; i < correctSeq.length; i++) {
  const v = correctSeq[i];
  const p = maxProbDist[i] || 0;
  if (v === 1) {
    winMaxSum += p;
    winN++;
  } else if (v === 0) {
    loseMaxSum += p;
    loseN++;
  }
}
console.log(
  "avg maxProb on wins",
  winN ? (winMaxSum / winN).toFixed(3) : "n/a",
  "avg maxProb on losses",
  loseN ? (loseMaxSum / loseN).toFixed(3) : "n/a"
);
// identify long losing streak occurrences (>= threshold)
const longThresh = 8;
let longOccur = [];
cur = 0;
let startIdx = null;
for (let i = 0; i < correctSeq.length; i++) {
  const v = correctSeq[i];
  if (v === 0) {
    if (cur === 0) startIdx = i;
    cur++;
  } else {
    if (cur >= longThresh)
      longOccur.push({ start: startIdx, end: i - 1, len: cur });
    cur = 0;
    startIdx = null;
  }
}
if (cur >= longThresh)
  longOccur.push({ start: startIdx, end: correctSeq.length - 1, len: cur });
console.log(
  "long losing streak instances (len>=" + longThresh + "):",
  longOccur.length
);
if (longOccur.length) console.log("examples (first 5):", longOccur.slice(0, 5));
// correlation: are long streaks associated with low entropy or low maxProb?
// compute avg maxProb for windows preceding streak starts
const lookBack = 5;
const preMaxAve = [];
for (const s of longOccur.slice(0, 20)) {
  const idx = Math.max(0, s.start - 1);
  let sum = 0,
    n = 0;
  for (let j = Math.max(0, s.start - lookBack); j < s.start; j++) {
    if (maxProbDist[j] != null) {
      sum += maxProbDist[j];
      n++;
    }
  }
  preMaxAve.push(n ? sum / n : null);
}
console.log("avg pre-streak maxProb (examples):", preMaxAve.slice(0, 10));
// quick per-class prediction distribution during long streaks
const classPredDuringLong = { 0: 0, 1: 0, 2: 0, 3: 0 };
let classPredN = 0;
for (const s of longOccur) {
  for (let i = s.start; i <= s.end; i++) {
    const r = pr[i];
    if (r && Array.isArray(r.probs)) {
      const pred =
        typeof r.predicted === "number"
          ? r.predicted
          : r.probs.indexOf(Math.max(...r.probs));
      classPredDuringLong[pred]++;
      classPredN++;
    }
  }
}
console.log(
  "per-class predicted counts during long streaks:",
  classPredDuringLong,
  "n",
  classPredN
);
console.log("done");
