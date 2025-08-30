// Calibration utilities: multi-class Brier, temperature scaling, shrinkage, dynamic clamping

export function brierScore(probs, truthIdx) {
  let sum = 0;
  for (let k = 0; k < 4; k++) {
    const y = k === truthIdx ? 1 : 0;
    const p = probs[k] ?? 0.25;
    sum += Math.pow(p - y, 2);
  }
  return sum / 4;
}

export function rollingMetrics(history, predictionRecords, start = 0) {
  const accArr = [];
  const brierArr = [];
  for (
    let i = start;
    i < Math.min(history.length, predictionRecords.length);
    i++
  ) {
    const rec = predictionRecords[i];
    if (!rec) continue;
    const truth = history[i];
    accArr.push(rec.predicted === truth ? 1 : 0);
    brierArr.push(brierScore(rec.probs || [0.25, 0.25, 0.25, 0.25], truth));
  }
  return { accArr, brierArr };
}

export function calibrateProbs(rawProbs, calibrationState, perf) {
  const { avgBrier = 0.2, avgAcc = 0.4 } = perf || {};
  let {
    dynamicTemperature = 1,
    dynamicClamp = 0.92,
    shrinkage = 0,
  } = calibrationState || {};
  // Adjust temperature based on Brier
  if (avgBrier > 0.25)
    dynamicTemperature = Math.min(1.4, dynamicTemperature + 0.05);
  else if (avgBrier < 0.12)
    dynamicTemperature = Math.max(0.7, dynamicTemperature - 0.03);
  // Adjust clamp if overconfidence suspected
  if (avgBrier > 0.3) dynamicClamp = Math.min(0.97, dynamicClamp + 0.01);
  // Shrinkage towards uniform if accuracy low & Brier high
  if (avgAcc < 0.3 && avgBrier > 0.25)
    shrinkage = Math.min(0.4, shrinkage + 0.05);
  else if (avgAcc > 0.45 && shrinkage > 0)
    shrinkage = Math.max(0, shrinkage - 0.02);

  // Apply calibration: temp scaling
  const tAdj = rawProbs.map((p) =>
    Math.pow(Math.max(1e-6, p), 1 / Math.max(1e-6, dynamicTemperature))
  );
  const tSum = tAdj.reduce((a, b) => a + b, 0) || 1;
  let probs = tAdj.map((v) => v / tSum);
  // Overconfidence clamp
  const maxP = Math.max(...probs);
  if (maxP > dynamicClamp) {
    probs = probs.map((p) => p * (dynamicClamp / maxP));
    const s2 = probs.reduce((a, b) => a + b, 0) || 1;
    probs = probs.map((p) => p / s2);
  }
  // Shrinkage toward uniform
  if (shrinkage > 0) {
    probs = probs.map((p) => (1 - shrinkage) * p + shrinkage * 0.25);
    const s3 = probs.reduce((a, b) => a + b, 0) || 1;
    probs = probs.map((p) => p / s3);
  }

  return {
    probs,
    calibrationState: { dynamicTemperature, dynamicClamp, shrinkage },
  };
}
