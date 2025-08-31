import {
  buildModel,
  trainIncremental,
  predictWithMC,
} from "../src/models/ml.js";

function makeSyntheticHistory(n) {
  const history = [];
  for (let i = 0; i < n; i++) {
    const r = Math.random();
    // create a changing distribution with occasional streaks
    if (i % 300 < 40) {
      // period with more zeros
      if (r < 0.15) history.push(0);
      else if (r < 0.5) history.push(1);
      else if (r < 0.85) history.push(2);
      else history.push(3);
    } else if (i % 200 < 30) {
      // short 2-heavy period
      if (r < 0.02) history.push(0);
      else if (r < 0.2) history.push(1);
      else if (r < 0.9) history.push(2);
      else history.push(3);
    } else {
      // baseline
      if (r < 0.05) history.push(0);
      else if (r < 0.5) history.push(1);
      else if (r < 0.85) history.push(2);
      else history.push(3);
    }
  }
  return history;
}

function makePredictionRecords(history) {
  // synthetic predictions: sometimes correct, sometimes wrong, with local streaks
  const recs = [];
  let currentBias = 1; // biased predicted class
  let streakLen = 0;
  for (let i = 0; i < history.length; i++) {
    const truth = history[i];
    // occasionally switch bias to create long-run repeating predictions
    if (Math.random() < 0.01) {
      currentBias = Math.floor(Math.random() * 4);
      streakLen = 0;
    }
    // create predicted class biased toward currentBias but sometimes correct
    let predicted;
    if (Math.random() < 0.25) predicted = truth; // some are correct
    else if (Math.random() < 0.6) predicted = currentBias;
    else predicted = Math.floor(Math.random() * 4);
    streakLen++;
    // synthetic sourceProbs array: array of sources; for simplicity include 6 sources
    const mk = () => {
      const base = Array.from({ length: 4 }, () => 0.25);
      base[predicted] = 0.6;
      // add small noise
      for (let j = 0; j < 4; j++)
        base[j] = Math.max(0, base[j] + (Math.random() - 0.5) * 0.1);
      const s = base.reduce((a, b) => a + b, 0) || 1;
      return base.map((v) => v / s);
    };
    const sourceProbs = [mk(), mk(), mk(), mk(), mk(), mk(), mk(), mk()];
    recs.push({ probs: mk(), predicted, sourceProbs });
  }
  return recs;
}

async function run() {
  console.log("Building model (seqLen=32)");
  const seqLen = 32;
  const model = buildModel({ seqLen, dropout: 0.25 });

  console.log("Generating synthetic history (2000)...");
  const history = makeSyntheticHistory(2000);
  console.log("Generating synthetic predictionRecords aligned to history...");
  const predictionRecords = makePredictionRecords(history);

  // print class distribution
  const counts = [0, 0, 0, 0];
  for (const v of history) counts[v]++;
  console.log("Class counts:", counts);

  console.log(
    "Running trainIncremental with MC filtering, validation and streak handling..."
  );
  try {
    const res = await trainIncremental(model, history, {
      seqLen,
      epochs: 4,
      batchSize: 64,
      maxWindow: 2000,
      decayLambda: 0.01,
      mcSamples: 5,
      mcUncertaintyThreshold: 0.06,
      validationSplit: 0.12,
      earlyStoppingPatience: 3,
      predictionRecords,
      successWeight: 2.0,
      mistakeWeight: 1.6,
      streakWindow: 12,
      streakThreshold: 6,
      streakBoost: 1.6,
    });
    console.log("Train finished. Keys:", Object.keys(res || {}));
    console.log("Sample metrics snapshot:", res);
  } catch (e) {
    console.error("Train failed:", e);
    process.exit(2);
  }

  console.log("Running predictWithMC(mcSamples=10)...");
  try {
    const p = await predictWithMC(model, history, { seqLen, mcSamples: 10 });
    console.log("Predict result:", p);
  } catch (e) {
    console.error("Predict failed:", e);
    process.exit(3);
  }

  console.log("Smoke test (2000) complete.");
}

run();
