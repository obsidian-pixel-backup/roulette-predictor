import * as tf from "@tensorflow/tfjs";

// Clean ML helpers for the roulette predictor project.
// Exports: buildDataset, buildModel, trainIncremental (with class-balanced sample weights),
// prepareLastSequence, predictWithMC (MC-dropout style).

export function buildDataset(history, seqLen = 32, maxWindow = 1000) {
  const recent = history.slice(-maxWindow);
  if (!recent || recent.length <= seqLen) return null;
  const xs = [];
  const ys = [];
  for (let i = 0; i + seqLen < recent.length; i++) {
    const window = recent.slice(i, i + seqLen);
    const target = recent[i + seqLen];
    const oneHotSeq = window.map((c) => [
      c === 0 ? 1 : 0,
      c === 1 ? 1 : 0,
      c === 2 ? 1 : 0,
      c === 3 ? 1 : 0,
    ]);
    xs.push(oneHotSeq);
    ys.push([
      target === 0 ? 1 : 0,
      target === 1 ? 1 : 0,
      target === 2 ? 1 : 0,
      target === 3 ? 1 : 0,
    ]);
  }
  return { xTensor: tf.tensor(xs), yTensor: tf.tensor(ys) };
}

function attentionBlock(x, units = 64) {
  const scoreH = tf.layers.dense({ units, activation: "tanh" }).apply(x);
  const score = tf.layers
    .dense({ units: 1, activation: "linear" })
    .apply(scoreH);
  const attn = tf.layers.activation({ activation: "softmax" }).apply(score);
  const attnT = tf.layers.permute({ dims: [2, 1] }).apply(attn);
  const context = tf.layers.dot({ axes: [2, 1] }).apply([attnT, x]);
  return tf.layers.reshape({ targetShape: [x.shape[2]] }).apply(context);
}

export function buildModel({ seqLen = 32, dropout = 0.25 } = {}) {
  const input = tf.input({ shape: [seqLen, 4] });
  let x = tf.layers
    .conv1d({ filters: 24, kernelSize: 3, activation: "relu", padding: "same" })
    .apply(input);
  x = tf.layers
    .conv1d({ filters: 40, kernelSize: 3, activation: "relu", padding: "same" })
    .apply(x);
  x = tf.layers.lstm({ units: 48, returnSequences: true }).apply(x);
  x = tf.layers.dropout({ rate: dropout }).apply(x);
  x = tf.layers.lstm({ units: 48, returnSequences: true }).apply(x);
  x = tf.layers.dropout({ rate: dropout }).apply(x);
  x = tf.layers.lstm({ units: 24, returnSequences: true }).apply(x);
  const context = attentionBlock(x, 48);
  let z = tf.layers.dense({ units: 48, activation: "relu" }).apply(context);
  z = tf.layers.dropout({ rate: dropout }).apply(z);
  const output = tf.layers.dense({ units: 4, activation: "softmax" }).apply(z);
  const model = tf.model({ inputs: input, outputs: output });
  model.compile({
    optimizer: tf.train.adam(0.001),
    loss: "categoricalCrossentropy",
    metrics: ["accuracy"],
  });
  return model;
}

export async function trainIncremental(
  model,
  history,
  {
    seqLen = 32,
    maxWindow = 1000,
    epochs = 3,
    batchSize = 32,
    decayLambda = 0.0,
    // MC-based confidence filtering: skip samples with mean std > mcUncertaintyThreshold
    mcSamples = 5,
    mcUncertaintyThreshold = 0.05,
    noiseStd = 0.02,
    // validation + early stopping
    validationSplit = 0.1,
    earlyStoppingPatience = 3,
    // learning rate scheduling (reduce-on-plateau)
    lrReduceFactor = 0.5,
    lrReducePatience = 2,
    minLR = 1e-6,
    // runtime environment checks
    requireGPU = false,
    workerRecommended = true,
    // Prefer validated results: pass predictionRecords to prioritize learning from
    // correct predictions and also learn from mistakes. These weights tune how
    // much to boost successes vs mistakes and how to respond to streaks.
    predictionRecords = null,
    successWeight = 2.0,
    mistakeWeight = 1.5,
    streakWindow = 10,
    streakThreshold = 5,
    streakBoost = 1.5,
    // replay buffer boost: preferentially upweight validated high-confidence matches
    replayBoost = 3.0,
    // bias augmentation: oversample sequences whose target class is favored by EWMA
    biasAugment = false,
    augmentFactor = 1.5,
    ewmaLambda = 0.12,
  } = {}
) {
  // Runtime environment checks: ensure GPU backend when requested and warn/fallback
  try {
    const backend =
      typeof tf.getBackend === "function"
        ? tf.getBackend()
        : tf && tf.env && typeof tf.env === "function"
        ? tf.env().get("BACKEND")
        : "unknown";
    const gpuBackends = ["webgl", "webgpu", "tensorflow"];
    const hasGPU = gpuBackends.includes(backend);
    let workerAvailable = false;
    try {
      if (typeof window !== "undefined" && typeof Worker === "function")
        workerAvailable = true;
    } catch (e) {
      workerAvailable = false;
    }
    if (!hasGPU) {
      const msg = `tfjs backend is '${backend}' — GPU backend not detected. Training will run on CPU; consider switching to a GPU-capable backend for faster training.`;
      if (requireGPU) {
        console.warn(
          msg +
            " requireGPU=true was requested; proceeding on CPU but performance will be degraded."
        );
      } else {
        console.warn(msg);
      }
      // attempt to set cpu backend if possible
      try {
        if (typeof tf.setBackend === "function") {
          // only change if not already cpu
          if (backend !== "cpu") {
            const ok = tf.setBackend("cpu");
            if (ok && typeof ok.then === "function") await ok;
            if (typeof tf.ready === "function") await tf.ready();
          }
        }
      } catch (e) {
        // non-fatal
        console.warn(
          "Failed to switch TF backend to cpu (non-fatal):",
          e && e.message ? e.message : e
        );
      }
    }
    if (workerRecommended && !workerAvailable) {
      console.warn(
        "Web Worker API not available - for heavy training consider using a Worker to avoid blocking the UI."
      );
    }
  } catch (e) {
    console.warn(
      "Runtime backend check failed:",
      e && e.message ? e.message : e
    );
  }

  const ds = buildDataset(history, seqLen, maxWindow);
  if (!ds) return null;
  const { xTensor, yTensor } = ds;

  // LR helpers (shared): tries multiple optimizer shapes used by tfjs
  const getLR = () => {
    try {
      const cfg = model?.optimizer?.getConfig?.();
      const lr = cfg?.learningRate ?? model?.optimizer?.learningRate;
      return typeof lr === "number" ? lr : (lr && lr.val) || 0.001;
    } catch (err) {
      return 0.001;
    }
  };
  const setLR = (v) => {
    try {
      if (model?.optimizer?.setLearningRate) model.optimizer.setLearningRate(v);
      else if (model?.optimizer?.learningRate != null)
        model.optimizer.learningRate = v;
    } catch (err) {
      // ignore
    }
  };

  // TFJS in the browser currently doesn't support `sampleWeight` in model.fit
  // (error: "sample weight is not supported yet"). Implement weighted training
  // by resampling with replacement according to per-sample class-balanced weights.
  try {
    const xsArr = await xTensor.array();
    const ysArr = await yTensor.array();

    // compute per-class inverse-frequency weights
    const counts = [0, 0, 0, 0];
    ysArr.forEach((y) => {
      const idx = y.indexOf(Math.max(...y));
      if (idx >= 0 && idx < 4) counts[idx]++;
    });
    const total = counts.reduce((a, b) => a + b, 0) || 1;
    const classWeights = counts.map((c) => total / (4 * Math.max(1, c)));

    // If bias augmentation requested, compute simple EWMA over recent history to get favored class probs
    let augmentMultiplier = [1, 1, 1, 1];
    try {
      if (biasAugment) {
        const recent = history.slice(-Math.max(ysArr.length + seqLen, 1));
        // compute EWMA of one-hot class indicators
        let ewma = [0.25, 0.25, 0.25, 0.25];
        const lambda = typeof ewmaLambda === "number" ? ewmaLambda : 0.12;
        for (
          let i = Math.max(0, recent.length - 1000);
          i < recent.length;
          i++
        ) {
          const c = recent[i];
          const one = [0, 0, 0, 0];
          if (c >= 0 && c < 4) one[c] = 1;
          for (let k = 0; k < 4; k++)
            ewma[k] = lambda * one[k] + (1 - lambda) * ewma[k];
        }
        // compute multipliers: boost classes with ewma > 0.25 proportionally
        for (let k = 0; k < 4; k++) {
          const excess = Math.max(0, ewma[k] - 0.25);
          augmentMultiplier[k] = 1 + augmentFactor * excess; // e.g., augmentFactor=1.5
        }
      }
    } catch (err) {
      console.debug("trainIncremental: ewma augment failed", err);
      augmentMultiplier = [1, 1, 1, 1];
    }

    // apply optional recency decay to prioritize recent samples (decayLambda==0 disables)
    const N = ysArr.length;
    const sampleWeights = ysArr.map((y, i) => {
      const idx = y.indexOf(Math.max(...y));
      const base = classWeights[idx] || 1;
      if (!decayLambda || decayLambda <= 0) return base;
      // recency weight: more recent samples (higher i) get larger weight
      const recency = Math.exp(-decayLambda * (N - 1 - i));
      return base * recency * (augmentMultiplier[idx] || 1);
    });

    // If predictionRecords were provided, boost weights for validated positives
    // (predicted == truth) and also boost mistakes so the model learns corrective
    // patterns. We map local sample indices back to global history indices.
    try {
      if (
        Array.isArray(predictionRecords) &&
        predictionRecords.length &&
        history &&
        history.length
      ) {
        const recentOffset = Math.max(
          0,
          history.length - Math.min(maxWindow, history.length)
        );
        // compute current streak info from the tail of predictionRecords
        let recentStreakLen = 0;
        let recentStreakIsWin = null;
        for (
          let i = predictionRecords.length - 1;
          i >= Math.max(0, predictionRecords.length - streakWindow);
          i--
        ) {
          const rec = predictionRecords[i];
          if (
            !rec ||
            typeof rec.predicted !== "number" ||
            typeof history[i] !== "number"
          )
            break;
          const correct = rec.predicted === history[i];
          if (recentStreakIsWin == null) {
            recentStreakIsWin = !!correct;
            recentStreakLen = 1;
          } else if (recentStreakIsWin === !!correct) {
            recentStreakLen++;
          } else break;
        }

        for (let localIdx = 0; localIdx < ysArr.length; localIdx++) {
          const globalIdx = recentOffset + localIdx + seqLen; // target index in full history
          const rec = predictionRecords[globalIdx];
          const truth = ysArr[localIdx].indexOf(Math.max(...ysArr[localIdx]));
          if (!rec || typeof rec.predicted !== "number") continue;
          if (rec.predicted === truth) {
            sampleWeights[localIdx] =
              (sampleWeights[localIdx] || 1) * (successWeight || 1);
            // If the prediction was high-confidence (not skipped), boost strongly
            if (!rec.skipped && replayBoost && replayBoost > 1) {
              sampleWeights[localIdx] *= replayBoost;
            }
          } else {
            // mistakes get a smaller boost by default but still emphasized
            sampleWeights[localIdx] =
              (sampleWeights[localIdx] || 1) * (mistakeWeight || 1);
          }
          // If there's a recent long streak, further boost samples that are part
          // of that streak to encourage correction or reinforcement.
          if (recentStreakLen >= streakThreshold) {
            // check if this sample is part of the recent streak (global index within tail)
            if (globalIdx >= predictionRecords.length - recentStreakLen) {
              const wasCorrect = rec.predicted === history[globalIdx];
              // For a winning streak, reinforce correct samples; for losing, focus on mistakes
              if (recentStreakIsWin && wasCorrect) {
                sampleWeights[localIdx] *= streakBoost;
              } else if (!recentStreakIsWin && !wasCorrect) {
                sampleWeights[localIdx] *= streakBoost;
              }
            }
          }
        }
      }
    } catch (err) {
      console.warn("trainIncremental: predictionRecords weighting failed", err);
    }

    // If MC-based filtering is requested, compute per-sample MC uncertainty and zero-out
    // weights for samples considered too uncertain. This is expensive, but helps avoid
    // training on noisy labels. We compute mean(std) across classes for each sample.
    if (mcUncertaintyThreshold && mcUncertaintyThreshold > 0 && model) {
      try {
        for (let si = 0; si < xsArr.length; si++) {
          // skip if already weight==0
          if (!sampleWeights[si]) continue;
          const seq = xsArr[si];
          // build a tensor for this one sequence
          const seqTensor = tf.tensor([seq]);
          const preds = [];
          for (let m = 0; m < mcSamples; m++) {
            const noisy = tf.tidy(() =>
              seqTensor.add(tf.randomNormal(seqTensor.shape, 0, noiseStd))
            );
            const p = model.predict(noisy);
            const data = await p.data();
            if (typeof p.dispose === "function") p.dispose();
            if (typeof noisy.dispose === "function") noisy.dispose();
            preds.push(Array.from(data));
          }
          seqTensor.dispose();
          // compute per-class std, then mean across classes
          const mean = [0, 0, 0, 0];
          preds.forEach((a) => a.forEach((v, i) => (mean[i] += v)));
          for (let i = 0; i < 4; i++) mean[i] /= preds.length;
          const variance = [0, 0, 0, 0];
          preds.forEach((a) =>
            a.forEach((v, i) => (variance[i] += Math.pow(v - mean[i], 2)))
          );
          for (let i = 0; i < 4; i++) variance[i] /= preds.length;
          const std = variance.map(Math.sqrt);
          const meanStd = std.reduce((a, b) => a + b, 0) / 4;
          if (meanStd > mcUncertaintyThreshold) {
            sampleWeights[si] = 0; // skip this sample
          }
        }
      } catch (err) {
        // If uncertainty estimation fails, fall back to no filtering
        console.warn(
          "trainIncremental: MC filtering failed, continuing without it",
          err
        );
      }
    }

    // normalize to probabilities for resampling (filter out zero-weight samples first)
    const filteredIdx = [];
    const filteredWeights = [];
    for (let i = 0; i < sampleWeights.length; i++) {
      if (
        sampleWeights[i] &&
        isFinite(sampleWeights[i]) &&
        sampleWeights[i] > 0
      ) {
        filteredIdx.push(i);
        filteredWeights.push(sampleWeights[i]);
      }
    }
    if (filteredIdx.length === 0) {
      // nothing to train on after filtering; fallback to original dataset
      const fitOpts = { epochs, batchSize, shuffle: true, verbose: 0 };
      const hist = await model.fit(xTensor, yTensor, fitOpts);
      xTensor.dispose();
      yTensor.dispose();
      return hist.history;
    }

    const sumW = filteredWeights.reduce((a, b) => a + b, 0) || 1;
    const probs = filteredWeights.map((w) => w / sumW);
    const cdf = [];
    probs.reduce((acc, p, i) => ((cdf[i] = acc + p), cdf[i]), 0);

    const targetSize = xsArr.length; // keep dataset size stable
    const newXs = [];
    const newYs = [];
    for (let i = 0; i < targetSize; i++) {
      const r = Math.random();
      // binary search on cdf
      let lo = 0,
        hi = cdf.length - 1;
      while (lo < hi) {
        const mid = Math.floor((lo + hi) / 2);
        if (r <= cdf[mid]) hi = mid;
        else lo = mid + 1;
      }
      const idx = lo;
      const srcIdx = filteredIdx[idx];
      newXs.push(xsArr[srcIdx]);
      newYs.push(ysArr[srcIdx]);
    }

    const newX = tf.tensor(newXs);
    const newY = tf.tensor(newYs);

    // build fit options with validation split and early stopping
    const fitOpts = {
      epochs,
      batchSize,
      shuffle: true,
      verbose: 0,
      validationSplit,
    };
    const callbacks = [];
    if (
      typeof tf.callbacks?.earlyStopping === "function" &&
      earlyStoppingPatience > 0
    ) {
      callbacks.push(
        tf.callbacks.earlyStopping({
          patience: earlyStoppingPatience,
          monitor: "val_loss",
        })
      );
    }
    if (callbacks.length) fitOpts.callbacks = callbacks;

    let finalHistory = null;
    if (
      (!fitOpts.callbacks || fitOpts.callbacks.length === 0) &&
      earlyStoppingPatience > 0 &&
      validationSplit > 0
    ) {
      // Manual early-stopping loop: fit one epoch at a time and monitor val_loss
      const maxEpochs = epochs;
      finalHistory = { loss: [], acc: [], val_loss: [], val_acc: [] };
      // learning rate scheduling state
      let bestVal = Infinity;
      let stale = 0;
      let lrStale = 0; // tracks epochs since last lr reduction

      for (let e = 0; e < maxEpochs; e++) {
        const h = await model.fit(newX, newY, {
          epochs: 1,
          batchSize,
          shuffle: true,
          verbose: 0,
          validationSplit,
        });
        // collect
        Object.keys(h.history || {}).forEach((k) => {
          finalHistory[k] = (finalHistory[k] || []).concat(h.history[k]);
        });
        const valLossArr = h.history?.val_loss || [];
        const valLoss = valLossArr[valLossArr.length - 1];
        if (valLoss != null && isFinite(valLoss)) {
          if (valLoss + 1e-8 < bestVal) {
            bestVal = valLoss;
            stale = 0;
            lrStale = 0;
          } else {
            stale++;
            lrStale++;
          }
        }
        // reduce LR on plateau
        if (lrStale >= lrReducePatience) {
          const cur = getLR();
          const next = Math.max(minLR, cur * (lrReduceFactor || 0.5));
          if (next < cur - 1e-12) {
            setLR(next);
          }
          lrStale = 0;
        }
        if (stale >= earlyStoppingPatience) break;
      }
    } else {
      const h = await model.fit(newX, newY, fitOpts);
      finalHistory = h.history;
    }

    // cleanup
    xTensor.dispose();
    yTensor.dispose();
    newX.dispose();
    newY.dispose();
    return finalHistory;
  } catch (e) {
    console.warn(
      "trainIncremental: weighted resampling failed, falling back to plain fit",
      e
    );
    try {
      const fitOpts = {
        epochs,
        batchSize,
        shuffle: true,
        verbose: 0,
        validationSplit,
      };
      const callbacks = [];
      if (
        typeof tf.callbacks?.earlyStopping === "function" &&
        earlyStoppingPatience > 0
      ) {
        callbacks.push(
          tf.callbacks.earlyStopping({
            patience: earlyStoppingPatience,
            monitor: "val_loss",
          })
        );
      }
      if (callbacks.length) fitOpts.callbacks = callbacks;

      let finalHistory = null;
      if (
        (!fitOpts.callbacks || fitOpts.callbacks.length === 0) &&
        earlyStoppingPatience > 0 &&
        validationSplit > 0
      ) {
        // Manual early-stopping loop for fallback path
        let bestVal = Infinity;
        let stale = 0;
        finalHistory = { loss: [], acc: [], val_loss: [], val_acc: [] };
        // fallback manual loop with lr scheduling
        let bestVal2 = Infinity;
        let stale2 = 0;
        let lrStale2 = 0;
        const getLR2 = getLR;
        const setLR2 = setLR;
        for (let e = 0; e < epochs; e++) {
          const h = await model.fit(xTensor, yTensor, {
            epochs: 1,
            batchSize,
            shuffle: true,
            verbose: 0,
            validationSplit,
          });
          Object.keys(h.history || {}).forEach((k) => {
            finalHistory[k] = (finalHistory[k] || []).concat(h.history[k]);
          });
          const valLossArr = h.history?.val_loss || [];
          const valLoss = valLossArr[valLossArr.length - 1];
          if (valLoss != null && isFinite(valLoss)) {
            if (valLoss + 1e-8 < bestVal2) {
              bestVal2 = valLoss;
              stale2 = 0;
              lrStale2 = 0;
            } else {
              stale2++;
              lrStale2++;
            }
          }
          if (lrStale2 >= lrReducePatience) {
            const cur = getLR2();
            const next = Math.max(minLR, cur * (lrReduceFactor || 0.5));
            if (next < cur - 1e-12) setLR2(next);
            lrStale2 = 0;
          }
          if (stale2 >= earlyStoppingPatience) break;
        }
      } else {
        const h = await model.fit(xTensor, yTensor, fitOpts);
        finalHistory = h.history;
      }
      xTensor.dispose();
      yTensor.dispose();
      return finalHistory;
    } catch (err) {
      xTensor.dispose();
      yTensor.dispose();
      throw err;
    }
  }
}

export function prepareLastSequence(history, seqLen = 32) {
  if (!history || history.length < seqLen) return null;
  const window = history.slice(-seqLen);
  const oneHotSeq = window.map((c) => [
    c === 0 ? 1 : 0,
    c === 1 ? 1 : 0,
    c === 2 ? 1 : 0,
    c === 3 ? 1 : 0,
  ]);
  return tf.tensor([oneHotSeq]);
}

export async function predictWithMC(
  model,
  history,
  { seqLen = 32, mcSamples = 5, noiseStd = 0.02 } = {}
) {
  const seqTensor = prepareLastSequence(history, seqLen);
  if (!seqTensor)
    return { probs: [0.25, 0.25, 0.25, 0.25], uncertainty: [0, 0, 0, 0] };
  const preds = [];
  for (let i = 0; i < mcSamples; i++) {
    // Avoid returning Promises from inside tf.tidy: create noisy tensor and sync-read prediction
    const noisy = tf.tidy(() =>
      seqTensor.add(tf.randomNormal(seqTensor.shape, 0, noiseStd))
    );
    const p = model.predict(noisy);
    // read data outside tidy to avoid async inside tidy
    const data = await p.data();
    if (typeof p.dispose === "function") p.dispose();
    if (typeof noisy.dispose === "function") noisy.dispose();
    preds.push(Array.from(data));
  }
  seqTensor.dispose();

  const mean = [0, 0, 0, 0];
  preds.forEach((a) => a.forEach((v, i) => (mean[i] += v)));
  for (let i = 0; i < 4; i++) mean[i] /= preds.length;
  const variance = [0, 0, 0, 0];
  preds.forEach((a) =>
    a.forEach((v, i) => (variance[i] += Math.pow(v - mean[i], 2)))
  );
  for (let i = 0; i < 4; i++) variance[i] /= preds.length;
  const std = variance.map(Math.sqrt);
  const s = mean.reduce((a, b) => a + b, 0) || 1;
  const probs = mean.map((m) => m / s);
  return { probs, uncertainty: std };
}

// Export a serializable JSON representation of a tfjs LayersModel.
// Returns the topology/weights manifest object (suitable for later import via tf.models.modelFromJSON).
export function exportModelJSON(model) {
  if (!model || typeof model.toJSON !== "function") return null;
  try {
    // model.toJSON returns a JS object describing the model topology and weightsManifest reference.
    return model.toJSON(null);
  } catch (e) {
    // Fallback: attempt synchronous string serialization
    try {
      return JSON.parse(model.toJSON());
    } catch (e2) {
      console.warn("exportModelJSON: could not serialize model", e, e2);
      return null;
    }
  }
}

// Reconstruct a tfjs model from JSON produced by exportModelJSON.
// Accepts either a JSON object or JSON string. Returns a Promise resolving to a LayersModel.
export async function importModelFromJSON(json) {
  if (!json) return null;
  try {
    // tf.models.modelFromJSON accepts either object or string
    const model = await tf.models.modelFromJSON(json);
    return model;
  } catch (e) {
    console.error("importModelFromJSON: failed to load model from JSON", e);
    throw e;
  }
}
