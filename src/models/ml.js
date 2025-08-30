import * as tf from "@tensorflow/tfjs";

// Sequence preparation: use recent maxWindow spins (default 1000). One-hot encode.
export function buildDataset(history, seqLen = 32, maxWindow = 1000) {
  const recent = history.slice(-maxWindow);
  if (recent.length <= seqLen) return null;
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
  const xTensor = tf.tensor(xs); // [samples, seqLen, 4]
  const yTensor = tf.tensor(ys); // [samples, 4]
  return { xTensor, yTensor };
}

function attentionBlock(x, units = 64) {
  // x shape: [batch, time, features]
  const timeSteps = x.shape[1];
  // Scores per timestep
  const scoreH = tf.layers.dense({ units, activation: "tanh" }).apply(x); // [b,t,u]
  const score = tf.layers
    .dense({ units: 1, activation: "linear" })
    .apply(scoreH); // [b,t,1]
  const attn = tf.layers.activation({ activation: "softmax" }).apply(score); // [b,t,1]
  // context = sum_t attn_t * x_t -> implemented via permutation & dot
  const attnT = tf.layers.permute({ dims: [2, 1] }).apply(attn); // [b,1,t]
  const context = tf.layers.dot({ axes: [2, 1] }).apply([attnT, x]); // [b,1,features]
  return tf.layers.reshape({ targetShape: [x.shape[2]] }).apply(context);
}

export function buildModel({ seqLen = 32, dropout = 0.25 } = {}) {
  const input = tf.input({ shape: [seqLen, 4] });
  // Replace unsupported 'causal' padding with 'same'; keep temporal order via no future lookahead operations.
  let x = tf.layers
    .conv1d({
      filters: 24,
      kernelSize: 3,
      activation: "relu",
      padding: "same",
      kernelInitializer: "glorotUniform",
    })
    .apply(input);
  x = tf.layers
    .conv1d({
      filters: 40,
      kernelSize: 3,
      activation: "relu",
      padding: "same",
      kernelInitializer: "glorotUniform",
    })
    .apply(x);
  // LSTM stack (reduced units to mitigate large orthogonal matrices + faster incremental training)
  x = tf.layers
    .lstm({
      units: 48,
      returnSequences: true,
      kernelInitializer: "glorotUniform",
    })
    .apply(x);
  x = tf.layers.dropout({ rate: dropout }).apply(x);
  x = tf.layers
    .lstm({
      units: 48,
      returnSequences: true,
      kernelInitializer: "glorotUniform",
    })
    .apply(x);
  x = tf.layers.dropout({ rate: dropout }).apply(x);
  x = tf.layers
    .lstm({
      units: 24,
      returnSequences: true,
      kernelInitializer: "glorotUniform",
    })
    .apply(x);
  // Attention
  const context = attentionBlock(x, 48);
  let z = tf.layers
    .dense({
      units: 48,
      activation: "relu",
      kernelInitializer: "glorotUniform",
    })
    .apply(context);
  z = tf.layers.dropout({ rate: dropout }).apply(z);
  const output = tf.layers
    .dense({
      units: 4,
      activation: "softmax",
      kernelInitializer: "glorotUniform",
    })
    .apply(z);
  const model = tf.model({
    inputs: input,
    outputs: output,
    name: "cnn_lstm_attn_v2",
  });
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
  { seqLen = 32, maxWindow = 1000, epochs = 3, batchSize = 32 } = {}
) {
  const ds = buildDataset(history, seqLen, maxWindow);
  if (!ds) return null;
  const { xTensor, yTensor } = ds;
  const hist = await model.fit(xTensor, yTensor, {
    epochs,
    batchSize,
    shuffle: true,
    verbose: 0,
  });
  xTensor.dispose();
  yTensor.dispose();
  return hist.history;
}

export function prepareLastSequence(history, seqLen = 32) {
  if (history.length < seqLen) return null;
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
    const noisy = tf.tidy(() =>
      seqTensor.add(tf.randomNormal(seqTensor.shape, 0, noiseStd))
    );
    const p = tf.tidy(() => model.predict(noisy));
    const arr = await p.data();
    preds.push(Array.from(arr));
    p.dispose();
    noisy.dispose();
  }
  seqTensor.dispose();
  // Average & std-dev per class
  const mean = [0, 0, 0, 0];
  preds.forEach((a) => a.forEach((v, i) => (mean[i] += v)));
  for (let i = 0; i < 4; i++) mean[i] /= preds.length;
  const variance = [0, 0, 0, 0];
  preds.forEach((a) =>
    a.forEach((v, i) => {
      variance[i] += Math.pow(v - mean[i], 2);
    })
  );
  for (let i = 0; i < 4; i++) variance[i] /= preds.length;
  const std = variance.map(Math.sqrt);
  // Normalize mean (should already sum 1 but ensure numerical stability)
  const sum = mean.reduce((a, b) => a + b, 0) || 1;
  const probs = mean.map((m) => m / sum);
  return { probs, uncertainty: std };
}

export async function exportModelJSON(model) {
  if (!model) return null;
  const json = await model.toJSON();
  return json; // structured object
}

export async function importModelFromJSON(json) {
  if (!json) return null;
  const model = await tf.models.modelFromJSON(json);
  model.compile({
    optimizer: tf.train.adam(0.001),
    loss: "categoricalCrossentropy",
    metrics: ["accuracy"],
  });
  return model;
}
