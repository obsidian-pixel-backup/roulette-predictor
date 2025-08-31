import fs from "fs";
import path from "path";
import {
  buildModel,
  trainIncremental,
  predictWithMC,
} from "../src/models/ml.js";

const statePath = "c:/Users/corne/Downloads/roulette_state (5).json";
async function run() {
  console.log("Loading state from", statePath);
  const raw = fs.readFileSync(statePath, "utf8");
  const data = JSON.parse(raw);
  const history = data.history || [];
  const predictionRecords = data.predictionRecords || [];
  console.log("History length:", history.length);
  console.log("PredictionRecords length:", predictionRecords.length);

  const seqLen = 32;
  const model = buildModel({ seqLen, dropout: 0.25 });

  const opts = {
    seqLen,
    epochs: 3,
    batchSize: 64,
    maxWindow: 4000, // train on recent window to keep runtime reasonable for smoke test
    decayLambda: 0.01,
    mcSamples: 5,
    mcUncertaintyThreshold: 0.06,
    validationSplit: 0.12,
    earlyStoppingPatience: 3,
    predictionRecords,
    successWeight: 2.0,
    mistakeWeight: 1.6,
    replayBoost: 3.0,
  };

  try {
    console.log(
      "Starting trainIncremental on recent window (maxWindow=" +
        opts.maxWindow +
        ")"
    );
    const res = await trainIncremental(model, history, opts);
    console.log("Train finished. Keys:", Object.keys(res || {}));
    console.log("Sample metrics snapshot:", res);
  } catch (e) {
    console.error("Train failed:", e);
    process.exit(2);
  }

  try {
    console.log("Running predictWithMC(mcSamples=10)");
    const p = await predictWithMC(model, history, {
      seqLen,
      mcSamples: 10,
      noiseStd: 0.04,
    });
    console.log("Predict result:", p);
  } catch (e) {
    console.error("Predict failed:", e);
    process.exit(3);
  }

  console.log("Smoke test (large state) complete.");
}

run();
