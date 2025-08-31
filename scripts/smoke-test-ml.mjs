import {
  buildModel,
  trainIncremental,
  predictWithMC,
} from "../src/models/ml.js";

async function run() {
  console.log("Smoke test: build model");
  const model = buildModel({ seqLen: 8, dropout: 0.2 });
  // synthetic history: cycle classes to create some data
  const history = [];
  for (let i = 0; i < 200; i++) {
    history.push(i % 4);
  }

  console.log("starting trainIncremental...");
  try {
    const res = await trainIncremental(model, history, {
      seqLen: 8,
      epochs: 1,
      batchSize: 16,
      maxWindow: 200,
    });
    console.log("train result keys:", Object.keys(res || {}));
  } catch (e) {
    console.error("train failed", e);
    process.exit(2);
  }

  console.log("running predictWithMC...");
  try {
    const p = await predictWithMC(model, history, { seqLen: 8, mcSamples: 3 });
    console.log("predict", p);
  } catch (e) {
    console.error("predict failed", e);
    process.exit(3);
  }
  console.log("Smoke test done");
}

run();
