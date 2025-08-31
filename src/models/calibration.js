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

export function rollingMetrics(
  history,
  predictionRecords,
  start = 0,
  options = {}
) {
  // options: { confidenceThreshold }
  const {
    confidenceThreshold = 0,
    splitIndex = null,
    valWindow = null,
  } = options || {};
  const accArr = [];
  const brierArr = [];
  const trainAcc = [];
  const trainBrier = [];
  const valAcc = [];
  const valBrier = [];
  let computedSplit = splitIndex;
  if (computedSplit == null && typeof valWindow === "number" && valWindow > 0) {
    computedSplit = Math.max(start, history.length - valWindow);
  }
  for (
    let i = start;
    i < Math.min(history.length, predictionRecords.length);
    i++
  ) {
    const rec = predictionRecords[i];
    if (!rec) continue;
    const truth = history[i];
    const probs = rec.probs || [0.25, 0.25, 0.25, 0.25];
    const maxP = Math.max(...probs);
    // Only count spins where the model would place a bet (confident enough)
    if (maxP < confidenceThreshold) continue;
    const predicted =
      rec.predicted == null ? probs.indexOf(maxP) : rec.predicted;
    const acc = predicted === truth ? 1 : 0;
    const br = brierScore(probs, truth);
    accArr.push(acc);
    brierArr.push(br);
    if (computedSplit != null) {
      if (i >= computedSplit) {
        valAcc.push(acc);
        valBrier.push(br);
      } else {
        trainAcc.push(acc);
        trainBrier.push(br);
      }
    }
  }
  return {
    accArr,
    brierArr,
    train: { accArr: trainAcc, brierArr: trainBrier },
    val: { accArr: valAcc, brierArr: valBrier },
  };
}

export function calibrateProbs(rawProbs, calibrationState, perf, options = {}) {
  // options: { confidenceThreshold }
  const { confidenceThreshold: optThreshold } = options || {};
  const { avgBrier = 0.2, avgAcc = 0.4 } = perf || {};
  let {
    dynamicTemperature = 1,
    dynamicClamp = 0.92,
    shrinkage = 0,
    predictionConfidenceThreshold = 0.3,
  } = calibrationState || {};
  // If caller passed an explicit option, prefer it
  const threshold =
    typeof optThreshold === "number"
      ? optThreshold
      : predictionConfidenceThreshold;
  // Adjust temperature based on Brier
  if (avgBrier > 0.25)
    dynamicTemperature = Math.min(1.4, dynamicTemperature + 0.05);
  else if (avgBrier < 0.12)
    dynamicTemperature = Math.max(0.7, dynamicTemperature - 0.03);
  // Adjust clamp if overconfidence suspected
  if (avgBrier > 0.3) dynamicClamp = Math.min(0.97, dynamicClamp + 0.01);
  // If accuracy is low, aggressively reduce the clamp (limit max probability)
  // so the model cannot be extremely confident when it's performing poorly.
  if (avgAcc < 0.32) {
    // move dynamicClamp toward 0.5 but don't go below 0.5
    dynamicClamp = Math.max(0.5, dynamicClamp - 0.08);
  }
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
  let maxP = Math.max(...probs);
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

  // Determine predicted class or no-bet marker (-1) when below threshold
  // allow threshold to adapt based on recent performance: if recent bet accuracy
  // is high, loosen threshold (allow more bets); if accuracy is low, tighten it.
  let dynamicThreshold = threshold;
  try {
    if (perf && typeof perf.avgAcc === "number") {
      if (perf.avgAcc > 0.55) {
        // lower threshold modestly to increase bet volume
        dynamicThreshold = Math.max(0.05, dynamicThreshold - 0.05);
      } else if (perf.avgAcc < 0.5) {
        // raise threshold to be more selective
        dynamicThreshold = Math.min(0.9, dynamicThreshold + 0.05);
      }
    }
  } catch (e) {
    /* ignore and use base threshold */
  }

  maxP = Math.max(...probs);
  const predicted = probs.indexOf(maxP);
  const skipped = maxP < dynamicThreshold;

  return {
    probs,
    predicted,
    skipped,
    calibrationState: {
      dynamicTemperature,
      dynamicClamp,
      shrinkage,
      predictionConfidenceThreshold: dynamicThreshold,
    },
  };
}
