import {
  buildModel,
  trainIncremental,
  predictWithMC,
} from "../src/models/ml.js";

async function run() {
  console.log("Long smoke test: build model");
  const seqLen = 32;
  const model = buildModel({ seqLen, dropout: 0.2 });

  // synthetic history: skewed distribution to test resampling
  const history = [];
  for (let i = 0; i < 1200; i++) {
    // make class 0 rare
    const r = Math.random();
    if (r < 0.05) history.push(0);
    else if (r < 0.5) history.push(1);
    else if (r < 0.85) history.push(2);
    else history.push(3);
  }

  console.log("history length", history.length);
  console.log("starting trainIncremental (longer)...");
  try {
    const res = await trainIncremental(model, history, {
      seqLen,
      epochs: 3,
      batchSize: 32,
      maxWindow: 1200,
    });
    console.log("train result keys:", Object.keys(res || {}));
    console.log("sample train metrics snapshot:", res);
  } catch (e) {
    console.error("train failed", e);
    process.exit(2);
  }

  console.log("running predictWithMC (mcSamples=10)...");
  try {
    const p = await predictWithMC(model, history, { seqLen, mcSamples: 10 });
    console.log("predict", p);
  } catch (e) {
    console.error("predict failed", e);
    process.exit(3);
  }
  console.log("Long smoke test done");
}

run();
