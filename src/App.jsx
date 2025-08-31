import React, {
  useState,
  useMemo,
  useEffect,
  useCallback,
  useRef,
} from "react";
import debounce from "lodash/debounce";
import {
  exportHistoryCSV,
  exportStateJSON,
  importFromFiles,
} from "./utils/file";
import { simulateSpins } from "./utils/simulate";
import {
  bayesianProbs,
  markovProbs,
  streakProbs,
  ewmaProbs,
  aggregateStatsMethods,
} from "./models/stats";
import {
  buildModel,
  trainIncremental,
  predictWithMC,
  exportModelJSON,
  importModelFromJSON,
} from "./models/ml";
import {
  monteCarloMarkov,
  DQNWeights,
  blendLogSpace,
  autoTuner,
  hmmProbs,
} from "./models/ensemble";
import { patternProbs } from "./models/patterns";
import { calibrateProbs } from "./models/calibration";
import DiagnosticsChart from "./components/DiagnosticsChart";
import PredictionCard from "./components/PredictionCard";
import HistoryTable from "./components/HistoryTable";
import MetricsPanel from "./components/MetricsPanel";
import SpinInput from "./components/SpinInput";

// Basic probability helper converting class counts to probs
function countsToProbs(counts) {
  const total = counts.reduce((a, b) => a + b, 0) || 1;
  return counts.map((c) => c / total);
}

// Sanitize a probability array of length 4: replace non-finite/negative, renormalize, fallback to uniform.
function sanitizeProbs(arr) {
  if (!Array.isArray(arr) || arr.length !== 4) return [0.25, 0.25, 0.25, 0.25];
  let anyBad = false;
  const cleaned = arr.map((v) => {
    if (!isFinite(v) || v < 0) {
      anyBad = true;
      return 0;
    }
    return v;
  });
  let sum = cleaned.reduce((a, b) => a + b, 0);
  if (!isFinite(sum) || sum <= 0) {
    return [0.25, 0.25, 0.25, 0.25];
  }
  const norm = cleaned.map((v) => v / sum);
  // Guard against rounding creating NaN
  for (let i = 0; i < 4; i++)
    if (!isFinite(norm[i])) return [0.25, 0.25, 0.25, 0.25];
  return norm;
}

export default function App() {
  // history: array of class indices (0-3). predictionRecords: array aligned with history containing predicted class & probs at that time.
  const [history, setHistory] = useState([]);
  const [predictionRecords, setPredictionRecords] = useState([]); // { probs:[p0..p3], predicted:int }
  const [hyperparams, setHyperparams] = useState({ placeholder: true }); // future automatic tuning store
  const [bayesState, setBayesState] = useState({
    alpha: [1, 1, 1, 1],
    probs: [0.25, 0.25, 0.25, 0.25],
  });
  const [markovState, setMarkovState] = useState({
    counts: Array.from({ length: 4 }, () => Array(4).fill(1)),
    probs: [0.25, 0.25, 0.25, 0.25],
  });
  const [streakState, setStreakState] = useState({
    streak: { len: 0, class: null },
    probs: [0.25, 0.25, 0.25, 0.25],
  });
  const [ewmaState, setEwmaState] = useState({
    probs: [0.25, 0.25, 0.25, 0.25],
    lambda: 0.15,
    values: [0.25, 0.25, 0.25, 0.25],
    processed: 0,
  });
  const [ensembleProbs, setEnsembleProbs] = useState([0.25, 0.25, 0.25, 0.25]);
  const ensembleRef = useRef([0.25, 0.25, 0.25, 0.25]);
  const [dqn, setDqn] = useState(() => new DQNWeights({ nSources: 6 })); // sources: bayes, markov, streak, ewma, ml, hmmMarkovMC
  const [hmmState, setHmmState] = useState({ probs: [0.25, 0.25, 0.25, 0.25] });
  const [mcMarkovState, setMcMarkovState] = useState({
    probs: [0.25, 0.25, 0.25, 0.25],
  });
  const [calibrationState, setCalibrationState] = useState({
    dynamicTemperature: 1,
    dynamicClamp: 0.92,
    shrinkage: 0,
  });
  // Track when repeat penalty parameters were last changed to compute pre/post accuracy
  const [penaltyBaselineIndex, setPenaltyBaselineIndex] = useState(null);
  const [diagnosticsSummary, setDiagnosticsSummary] = useState({});
  const [mlModel, setMlModel] = useState(null);
  const [mlProbs, setMlProbs] = useState(null);
  const [mlUncertainty, setMlUncertainty] = useState(null);
  const seqLen = 32; // base sequence length
  const [windowSize, setWindowSize] = useState(50);
  const [viewWindow, setViewWindow] = useState("50");
  // Alerts / notifications
  const [alerts, setAlerts] = useState([]); // {id,msg,type}
  const alertIdRef = useRef(0);
  // Simulation + training control
  const [isSimulating, setIsSimulating] = useState(false);
  const simStopRef = useRef(false);
  const simQueueRef = useRef([]); // queued spins
  const trainingRef = useRef(false); // lock for fit()
  const [pendingQueue, setPendingQueue] = useState(0); // remaining spins in simulation queue
  const [isTraining, setIsTraining] = useState(false); // UI flag for active training
  const [simCountInput, setSimCountInput] = useState(100);
  const pushAlert = useCallback((msg, type = "info", ttl = 6000) => {
    const id = ++alertIdRef.current;
    setAlerts((a) => [...a, { id, msg, type }]);
    if (ttl)
      setTimeout(() => setAlerts((a) => a.filter((x) => x.id !== id)), ttl);
  }, []);

  // Current probs derived from naive Laplace counts on full history
  // Combine ensemble stats with ML probabilities if available (simple average placeholder)
  const currentProbs = useMemo(
    () => sanitizeProbs(ensembleProbs),
    [ensembleProbs]
  );

  const currentPrediction = useMemo(
    () => currentProbs.indexOf(Math.max(...currentProbs)),
    [currentProbs]
  );

  // Training queue and batching logic
  const trainingQueueRef = useRef([]);
  const TRAIN_BATCH_SIZE = 5;
  const pushSpin = useCallback(
    (cls) => {
      // Capture prediction BEFORE adding the new spin using latest ensembleRef
      setPredictionRecords((pr) => {
        const probsSafe = sanitizeProbs(
          ensembleRef.current || [0.25, 0.25, 0.25, 0.25]
        );
        const predicted = probsSafe.indexOf(Math.max(...probsSafe));
        const rec = { probs: probsSafe, predicted };
        console.debug("pushSpin: recording prediction", {
          probsSafe,
          predicted,
          cls,
        });
        return [...pr, rec];
      });
      setHistory((prev) => {
        const newHist = [...prev, cls];
        // Queue for training
        trainingQueueRef.current.push(cls);
        if (newHist.length === 1)
          pushAlert("First spin recorded", "info", 4000);
        return newHist;
      });
    },
    [pushAlert]
  );

  const handleManualClassAdd = (cls) => {
    if (cls == null) return;
    pushSpin(cls);
  };

  // Simulation queue utilities
  const enqueueSimulation = (count) => {
    // push generated classes to queue
    for (let i = 0; i < count; i++) {
      const r = Math.random();
      const pZero = 1 / 37,
        pDozen = 12 / 37;
      let cls;
      if (r < pZero) cls = 0;
      else if (r < pZero + pDozen) cls = 1;
      else if (r < pZero + 2 * pDozen) cls = 2;
      else cls = 3;
      simQueueRef.current.push(cls);
    }
    setPendingQueue(simQueueRef.current.length);
  };
  // Simulation queue: process spins one at a time, waiting for prediction update
  const processSimQueue = useCallback(() => {
    if (simStopRef.current) {
      setIsSimulating(false);
      return;
    }
    if (!simQueueRef.current.length) {
      setIsSimulating(false);
      setPendingQueue(0);
      return;
    }
    const nextSpin = simQueueRef.current.shift();
    pushSpin(nextSpin);
    setPendingQueue(simQueueRef.current.length);
    // Force prediction pipeline to run immediately after spin is pushed
    if (recomputeRef.current) {
      if (typeof recomputeRef.current.flush === "function") {
        try {
          recomputeRef.current.flush();
        } catch (_) {}
      } else if (typeof recomputeRef.current === "function") {
        try {
          recomputeRef.current();
        } catch (_) {}
      }
    }
    // Schedule next spin after a short delay so recompute can run and update ensembleRef
    setTimeout(() => {
      if (simStopRef.current) {
        setIsSimulating(false);
        return;
      }
      // proceed to next spin
      processSimQueue();
    }, 50);
  }, [pushSpin]);

  // The simulation progression is paced inside processSimQueue to avoid racing
  // with React state updates and the recompute pipeline. Removed the previous
  // effect-based progression which could process the next spin before
  // recompute had a chance to update ensemble probabilities.

  const runSimulation = (count = 10) => {
    enqueueSimulation(count);
    if (!isSimulating) {
      simStopRef.current = false;
      // Ensure ensembleRef is fresh before recording predictions
      if (recomputeRef.current) {
        try {
          if (typeof recomputeRef.current.flush === "function")
            recomputeRef.current.flush();
          else if (typeof recomputeRef.current === "function")
            recomputeRef.current();
        } catch (_) {}
      }
      setIsSimulating(true);
      // Start first spin
      processSimQueue();
    }
  };
  const runSimulationBatched = (total = 500) => {
    enqueueSimulation(total);
    if (!isSimulating) {
      simStopRef.current = false;
      // Ensure ensembleRef is fresh before recording predictions
      if (recomputeRef.current) {
        try {
          if (typeof recomputeRef.current.flush === "function")
            recomputeRef.current.flush();
          else if (typeof recomputeRef.current === "function")
            recomputeRef.current();
        } catch (_) {}
      }
      setIsSimulating(true);
      // Start first spin
      processSimQueue();
    }
  };
  const stopSimulation = () => {
    simStopRef.current = true;
  };

  const resetAll = () => {
    setHistory([]);
    setPredictionRecords([]);
    setHyperparams({ placeholder: true });
  };

  const clearState = () => {
    if (confirm("This will clear all current state. Continue?")) resetAll();
  };

  const importHandler = async (files) => {
    let imported;
    try {
      imported = await importFromFiles(files);
    } catch (e) {
      console.error("Import parse error", e);
      pushAlert("Import parse error", "error", 10000);
      return;
    }
    if (imported.history) {
      // Expect history as original spins (0-36) or already class indices 0-3; normalize
      const arr = imported.history.map((h) => {
        const s = typeof h.spin === "number" ? h.spin : h.spin?.spin; // support nested
        if (s === 0) return 0;
        if (s >= 1 && s <= 12) return 1;
        if (s >= 13 && s <= 24) return 2;
        if (s >= 25 && s <= 36) return 3;
        return 0;
      });
      setHistory(arr);
      setPredictionRecords([]); // rebuild progressively
      pushAlert(`Imported ${arr.length} spins`, "info");
    }
    // Sanitize imported predictionRecords if present
    if (
      imported.predictionRecords &&
      Array.isArray(imported.predictionRecords)
    ) {
      try {
        const sanitized = imported.predictionRecords.map((rec) => {
          const probs = sanitizeProbs(rec.probs);
          const predicted =
            typeof rec.predicted === "number"
              ? rec.predicted
              : probs.indexOf(Math.max(...probs));
          // If sourceProbs were exported, sanitize each source array too
          const sourceProbs = Array.isArray(rec.sourceProbs)
            ? rec.sourceProbs.map((s) => sanitizeProbs(s))
            : undefined;
          return { ...rec, probs, predicted, sourceProbs };
        });
        setPredictionRecords(sanitized);
        pushAlert(`Imported ${sanitized.length} prediction records`, "info");
      } catch (e) {
        console.warn("Imported predictionRecords malformed, skipping", e);
      }
    }
    if (imported.bayesState) setBayesState(imported.bayesState);
    if (imported.markovState) setMarkovState(imported.markovState);
    if (imported.streakState) setStreakState(imported.streakState);
    if (imported.ewmaState) setEwmaState(imported.ewmaState);
    if (imported.mlModelJSON) {
      try {
        importModelFromJSON(imported.mlModelJSON).then((m) => setMlModel(m));
      } catch (e) {
        console.error("ML model import failed", e);
        pushAlert("Model import failed", "error");
      }
    }
    if (imported.dqnState) {
      try {
        setDqn(DQNWeights.fromJSON(imported.dqnState));
      } catch (e) {
        console.error("DQN import failed", e);
        pushAlert("DQN import failed", "error");
      }
    }
    if (imported.hyperparams)
      setHyperparams((prev) => ({ ...prev, ...imported.hyperparams }));
    if (imported.calibrationState)
      setCalibrationState(imported.calibrationState);
    if (imported.diagnosticsSummary)
      setDiagnosticsSummary(imported.diagnosticsSummary);
  };

  useEffect(() => {
    // future initialization (e.g., warm start from cached localStorage) left intentionally empty per file-based persistence requirement
  }, []);

  const metricsWindowOptions = ["10", "25", "50", "100", "250", "500", "all"];

  // Sync slider and button selection
  useEffect(() => {
    if (viewWindow !== "all") {
      const w = parseInt(viewWindow, 10);
      setWindowSize(w);
    }
  }, [viewWindow]);

  const displayedHistory = useMemo(() => {
    if (viewWindow === "all") return history;
    const w = parseInt(viewWindow, 10);
    return history.slice(-w);
  }, [history, viewWindow]);

  // Map current probs into object for PredictionCard compatibility
  const probsObj = useMemo(
    () => ({
      zero: currentProbs[0],
      first: currentProbs[1],
      second: currentProbs[2],
      third: currentProbs[3],
    }),
    [currentProbs]
  );

  // Debounced recompute pipeline
  const recomputeRef = useRef();
  useEffect(() => {
    const fn = async () => {
      if (!history.length) return;
      try {
        const bayes = bayesianProbs(history, bayesState.alpha || [1, 1, 1, 1]);
        const mk = markovProbs(history);
        const st = streakProbs(history);
        const ew = ewmaProbs(history, ewmaState);
        setBayesState({ alpha: bayes.alphaPosterior, probs: bayes.probs });
        setMarkovState({ counts: mk.counts, probs: mk.probs });
        setStreakState(st);
        setEwmaState(ew);
        const hmm = hmmProbs(history, {});
        setHmmState(hmm);
        const mcSims = hyperparams.mcMarkovSims || 300;
        const mc = monteCarloMarkov(mk.counts, history[history.length - 1], {
          numSims: mcSims,
          horizon: 1,
        });
        setMcMarkovState(mc);
        const statBlend = aggregateStatsMethods([bayes, mk, st, ew]);
        const pat = patternProbs(history, {
          window: 300,
          maxPattern: 8,
          minCount: 2,
        });
        const sources = [
          bayes.probs,
          mk.probs,
          st.probs,
          ew.probs,
          mlProbs || statBlend,
          hmm.probs,
          mc.probs,
          pat.probs,
        ];
        if (dqn.nSources !== sources.length)
          setDqn(new DQNWeights({ nSources: sources.length }));
        const weights = dqn.weights.slice(0, sources.length);
        // Sanitize each source before blending to avoid propagating NaNs
        const safeSources = sources.map((s) => sanitizeProbs(s));

        // Compute per-source rolling performance (accuracy / (brier + eps)) and update DQN weights
        try {
          const nSources = safeSources.length;
          const perfWindow = Math.min(
            predictionRecords.length,
            hyperparams.perfWindow || 100
          );
          const startIdx = Math.max(0, predictionRecords.length - perfWindow);
          const stats = Array.from({ length: nSources }, () => ({
            correct: 0,
            brier: 0,
            cnt: 0,
          }));
          for (let i = startIdx; i < predictionRecords.length; i++) {
            const rec = predictionRecords[i];
            if (!rec || !Array.isArray(rec.sourceProbs)) continue;
            const truth = history[i];
            for (let si = 0; si < nSources; si++) {
              const sp = sanitizeProbs(rec.sourceProbs[si]);
              const predIdx = sp.indexOf(Math.max(...sp));
              if (predIdx === truth) stats[si].correct++;
              let bsum = 0;
              for (let k = 0; k < 4; k++) {
                const y = truth === k ? 1 : 0;
                bsum += (sp[k] - y) * (sp[k] - y);
              }
              stats[si].brier += bsum;
              stats[si].cnt++;
            }
          }
          const eps = 1e-6;
          let perfScores = stats.map((s) => {
            const cnt = s.cnt || 0;
            if (!cnt) return 0;
            const acc = s.correct / cnt;
            const avgBrier = s.brier / (cnt * 4);
            return (acc + eps) / (avgBrier + eps);
          });
          const totalScore = perfScores.reduce((a, b) => a + b, 0);
          if (!totalScore || !isFinite(totalScore)) {
            perfScores = dqn.weights.slice(0, nSources);
          }
          // Apply to DQN with moderate smoothing (alpha)
          if (typeof dqn.applyPerformanceWeights === "function") {
            dqn.applyPerformanceWeights(perfScores, 0.6);
          } else if (totalScore) {
            // fallback: normalize and assign locally
            const normalized = perfScores.map((s) => Math.max(0.01, s));
            const ssum = normalized.reduce((a, b) => a + b, 0) || 1;
            const nw = normalized.map((v) => v / ssum);
            for (let i = 0; i < nw.length; i++) weights[i] = nw[i];
          }
        } catch (e) {
          // non-fatal
          console.debug("perf weight calc failed", e);
        }
        const blendedRaw = blendLogSpace(safeSources, weights, {
          temperature: hyperparams.temperature || 1.0,
          clampMax: hyperparams.clampMax || 0.92,
        });
        let blendedSafe = sanitizeProbs(blendedRaw);
        // Optional repeat penalty to avoid always parroting the last class when alternatives are close.
        const lastClass = history[history.length - 1];
        let penalized = blendedSafe;
        if (lastClass != null) {
          // Use hyperparams from UI directly, not fallback defaults
          const penalty = Number(hyperparams.repeatPenalty);
          const minGap = Number(hyperparams.repeatMinGap);
          const sorted = [...blendedSafe].sort((a, b) => b - a);
          const top = sorted[0];
          const second = sorted[1] ?? 0;
          if (
            penalty > 0 &&
            blendedSafe[lastClass] === top &&
            top - second < minGap
          ) {
            const reduction = Math.min(
              penalty,
              blendedSafe[lastClass] - second * 0.5
            ); // don't over-penalize
            const redistribute = reduction / 3;
            penalized = blendedSafe.map((p, i) =>
              i === lastClass ? p - reduction : p + redistribute
            );
            penalized = sanitizeProbs(penalized);
          }
        }
        // Stuck-run mitigation: more aggressive and add randomization
        if (predictionRecords.length > 15) {
          const lastPred =
            predictionRecords[predictionRecords.length - 1]?.predicted;
          if (lastPred != null) {
            let runLen = 0;
            for (let i = predictionRecords.length - 1; i >= 0; i--) {
              if (predictionRecords[i]?.predicted === lastPred) runLen++;
              else break;
            }
            const stuckRunThreshold = hyperparams.stuckRunThreshold ?? 12; // lower threshold
            // Hard prediction limit: if a class is predicted repeatedly and failing,
            // force a different class after `predictionLimit` repeats.
            const predictionLimit = hyperparams.predictionLimit ?? 4;
            if (runLen > predictionLimit) {
              // compute recent accuracy for this run
              let correctInRunHard = 0;
              for (
                let i = predictionRecords.length - runLen;
                i < predictionRecords.length;
                i++
              ) {
                if (predictionRecords[i]?.predicted === history[i])
                  correctInRunHard++;
              }
              const runAccHard = correctInRunHard / runLen;
              // If run is failing (<50% accuracy), apply a strong forced penalty
              if (runAccHard < 0.5) {
                const forceDecay = Math.min(
                  hyperparams.stuckPenaltyForced ?? 0.35,
                  blendedSafe[lastPred] * 0.8
                );
                const redistributeF = forceDecay / 3;
                let forced = blendedSafe.map((p, i) =>
                  i === lastPred ? p - forceDecay : p + redistributeF
                );
                penalized = sanitizeProbs(forced);
              }
            }
            if (runLen >= stuckRunThreshold) {
              let correctInRun = 0;
              for (
                let i = predictionRecords.length - runLen;
                i < predictionRecords.length;
                i++
              ) {
                if (predictionRecords[i]?.predicted === history[i])
                  correctInRun++;
              }
              const runAcc = correctInRun / runLen;
              const entropy = blendedSafe.reduce(
                (s, p) => (p > 0 ? s + -p * Math.log2(p) : s),
                0
              );
              const minAcc = hyperparams.stuckMinAcc ?? 0.55; // require higher accuracy to avoid penalty
              const entropyThresh = hyperparams.stuckEntropyThresh ?? 1.35; // require higher entropy
              if (runAcc < minAcc && entropy < entropyThresh) {
                // Stronger decay
                const decay = Math.min(
                  hyperparams.stuckPenalty ?? 0.12,
                  blendedSafe[lastPred] * 0.7
                );
                const redistribute = decay / 3;
                let noisyPenalized = blendedSafe.map((p, i) =>
                  i === lastPred ? p - decay : p + redistribute
                );
                // Add small randomization to all classes
                const noiseLevel = 0.03;
                noisyPenalized = noisyPenalized.map(
                  (p) => p + (Math.random() - 0.5) * noiseLevel
                );
                penalized = sanitizeProbs(noisyPenalized);
              }
            }
          }
        }
        // Evaluate recent performance
        const evalWindow = 50;
        let correct = 0,
          brierSum = 0,
          cnt = 0;
        for (
          let i = Math.max(0, predictionRecords.length - evalWindow);
          i < predictionRecords.length;
          i++
        ) {
          const rec = predictionRecords[i];
          if (!rec) continue;
          const truth = history[i];
          if (rec.predicted === truth) correct++;
          const probs = rec.probs || [0.25, 0.25, 0.25, 0.25];
          for (let k = 0; k < 4; k++) {
            const y = truth === k ? 1 : 0;
            brierSum += (probs[k] - y) * (probs[k] - y);
          }
          cnt++;
        }
        const avgAcc = cnt ? correct / cnt : 0;
        const avgBrier = cnt ? brierSum / (cnt * 4) : 0.25;
        const cal = calibrateProbs(penalized, calibrationState, {
          avgAcc,
          avgBrier,
        });
        const calSafe = sanitizeProbs(cal.probs);
        // Debug: if ensemble ends up uniform, log source details to help trace
        const isUniform = calSafe.every((p) => Math.abs(p - 0.25) < 1e-6);
        if (isUniform && history.length) {
          try {
            console.debug(
              "recompute: ensemble uniform => dumping sources/weights",
              {
                historyLen: history.length,
                sources: safeSources,
                weights,
                blendedRaw,
                penalized,
                calSafe,
              }
            );
          } catch (_) {}
        }
        setCalibrationState(cal.calibrationState);
        setEnsembleProbs(calSafe);
        // keep ref in sync for synchronous access during pushSpin (esp. simulation)
        ensembleRef.current = calSafe;
        // Compute pre/post penalty change accuracy if baseline defined
        let penaltyComparison = null;
        if (penaltyBaselineIndex != null && predictionRecords.length > 5) {
          let preCorrect = 0,
            preCount = 0,
            postCorrect = 0,
            postCount = 0;
          for (
            let i = 0;
            i < predictionRecords.length && i < history.length;
            i++
          ) {
            const rec = predictionRecords[i];
            if (!rec) continue;
            const truth = history[i];
            if (i < penaltyBaselineIndex) {
              preCount++;
              if (rec.predicted === truth) preCorrect++;
            } else {
              postCount++;
              if (rec.predicted === truth) postCorrect++;
            }
          }
          const preAcc = preCount ? preCorrect / preCount : null;
          const postAcc = postCount ? postCorrect / postCount : null;
          penaltyComparison = {
            preAcc,
            postAcc,
            preN: preCount,
            postN: postCount,
            baselineIdx: penaltyBaselineIndex,
          };
        }
        setDiagnosticsSummary((prev) => ({
          ...prev,
          lastEval: { avgAcc, avgBrier, n: history.length },
          penaltyComparison,
        }));
        if (history.length && history.length % 100 === 0) {
          pushAlert(
            `Eval @${history.length}: Acc ${(avgAcc * 100).toFixed(
              1
            )}% Brier ${avgBrier.toFixed(3)}`,
            "metric",
            8000
          );
        }
      } catch (e) {
        console.error("Recompute error", e);
        pushAlert("Prediction pipeline error", "error", 10000);
      }
    };
    const debounced = debounce(fn, 60, { maxWait: 90 });
    recomputeRef.current = debounced;
    return () => debounced.cancel();
  }, [
    history,
    mlProbs,
    predictionRecords,
    hyperparams,
    dqn,
    calibrationState,
    bayesState.alpha,
    ewmaState.lambda,
    pushAlert,
  ]);

  useEffect(() => {
    if (recomputeRef.current) recomputeRef.current();
  }, [history, predictionRecords, mlProbs]);

  // DQN reward update after each new actual outcome (compare previous prediction)
  useEffect(() => {
    if (!history.length || predictionRecords.length < history.length) return;
    const idx = history.length - 1;
    const rec = predictionRecords[idx];
    if (!rec) return;
    const reward = rec.predicted === history[idx] ? 1 : -0.5;
    dqn.updateReward(reward);
    const action = dqn.chooseAction(rec.probs);
    dqn.applyAction(action);
    // DQN instance is updated in-place by its methods above. Avoid recreating
    // the DQN object here because that would change `dqn` state every render
    // and trigger other effects that depend on `dqn`, causing render loops.
  }, [history, predictionRecords]);

  // Auto-tuner every 50 spins
  useEffect(() => {
    if (history.length && history.length % 50 === 0) {
      const tuned = autoTuner({ predictionRecords, history, hyperparams, dqn });
      setHyperparams(tuned);
      setDiagnosticsSummary((prev) => ({ ...prev, tuner: tuned.lastEval }));
    }
  }, [history]);

  // Initialize model lazily
  useEffect(() => {
    if (!mlModel) {
      const m = buildModel({ seqLen, dropout: 0.25 });
      setMlModel(m);
    }
  }, [mlModel]);

  // Incremental training trigger (debounced by history length changes)
  // Incremental training every TRAIN_BATCH_SIZE spins
  useEffect(() => {
    if (!mlModel) return;
    if (history.length < seqLen + 10) return;
    let canceled = false;
    let retried = false;
    // Only train if enough new spins have been queued and not already training
    if (
      trainingQueueRef.current.length >= TRAIN_BATCH_SIZE &&
      !trainingRef.current
    ) {
      (async () => {
        try {
          trainingRef.current = true;
          setIsTraining(true);
          await trainIncremental(mlModel, history, {
            seqLen,
            epochs: 2,
            batchSize: 32,
            decayLambda: Number(hyperparams.decayLambda) || 0.0,
            mcSamples: Number(hyperparams.mcSamples) || 5,
            mcUncertaintyThreshold:
              Number(hyperparams.mcUncertaintyThreshold) || 0.05,
            validationSplit: Number(hyperparams.validationSplit) || 0.1,
            earlyStoppingPatience:
              Number(hyperparams.earlyStoppingPatience) || 3,
            predictionRecords,
            successWeight: Number(hyperparams.successWeight) || 2.0,
            mistakeWeight: Number(hyperparams.mistakeWeight) || 1.5,
            streakWindow: Number(hyperparams.streakWindow) || 10,
            streakThreshold: Number(hyperparams.streakThreshold) || 5,
            streakBoost: Number(hyperparams.streakBoost) || 1.5,
          });
          trainingQueueRef.current = [];
          if (canceled) return;
          const { probs, uncertainty } = await predictWithMC(mlModel, history, {
            seqLen,
            mcSamples: 5,
          });
          setMlProbs(probs);
          setMlUncertainty(uncertainty);
        } catch (e) {
          console.error("ML train/predict error", e);
          const msg = String(e?.message || e || "");
          if (!retried) {
            if (msg.includes("CAUSAL") || msg.includes("NotImplemented")) {
              retried = true;
              try {
                const rebuilt = buildModel({ seqLen, dropout: 0.25 });
                setMlModel(rebuilt);
                if (!trainingRef.current) {
                  trainingRef.current = true;
                  setIsTraining(true);
                  await trainIncremental(rebuilt, history, {
                    seqLen,
                    epochs: 1,
                    batchSize: 32,
                    decayLambda: Number(hyperparams.decayLambda) || 0.0,
                    mcSamples: Number(hyperparams.mcSamples) || 5,
                    mcUncertaintyThreshold:
                      Number(hyperparams.mcUncertaintyThreshold) || 0.05,
                    validationSplit: Number(hyperparams.validationSplit) || 0.1,
                    earlyStoppingPatience:
                      Number(hyperparams.earlyStoppingPatience) || 3,
                    predictionRecords,
                    successWeight: Number(hyperparams.successWeight) || 2.0,
                    mistakeWeight: Number(hyperparams.mistakeWeight) || 1.5,
                    streakWindow: Number(hyperparams.streakWindow) || 10,
                    streakThreshold: Number(hyperparams.streakThreshold) || 5,
                    streakBoost: Number(hyperparams.streakBoost) || 1.5,
                  });
                }
                trainingQueueRef.current = [];
                if (!canceled) {
                  const { probs, uncertainty } = await predictWithMC(
                    rebuilt,
                    history,
                    { seqLen, mcSamples: 5 }
                  );
                  setMlProbs(probs);
                  setMlUncertainty(uncertainty);
                  pushAlert("Recovered model after rebuild", "info", 6000);
                }
                return;
              } catch (re) {
                console.error("Model rebuild failed", re);
                pushAlert("Model rebuild failed", "error", 10000);
              }
            }
          }
          pushAlert("ML train/predict error", "error", 9000);
        } finally {
          trainingRef.current = false;
          setIsTraining(false);
        }
      })();
    }
    return () => {
      canceled = true;
    };
  }, [history, mlModel]);

  // Export state package builder

  // Export state package builder
  const exportState = async () => {
    let mlModelJSON = null;
    try {
      mlModelJSON = mlModel ? await mlModel.toJSON() : null;
    } catch (e) {
      console.warn("Model export failed", e);
    }
    exportStateJSON({
      history,
      predictionRecords,
      hyperparams,
      bayesState,
      markovState,
      streakState,
      ewmaState,
      mlModelJSON,
      dqnState: dqn.toJSON(),
      hmmState,
      mcMarkovState,
      calibrationState,
      diagnosticsSummary,
    });
  };

  return (
    <div className="app-container">
      <header
        className="app-header"
        style={{ display: "flex", alignItems: "center", gap: "1rem" }}
      >
        <h1 style={{ margin: 0 }}>Roulette Dozen Predictor</h1>
        {(() => {
          const lastEval = diagnosticsSummary.lastEval || {}; // {avgAcc, avgBrier}
          const acc50 =
            lastEval.avgAcc != null
              ? (lastEval.avgAcc * 100).toFixed(1) + "%"
              : "—";
          const brier50 =
            lastEval.avgBrier != null ? lastEval.avgBrier.toFixed(3) : "—";
          const confidence = (() => {
            if (!currentProbs) return "—";
            const m = Math.max(...currentProbs);
            return (m * 100).toFixed(1) + "%";
          })();
          const badgeStyle = {
            background: "#1e2835",
            padding: "4px 10px",
            borderRadius: 6,
            display: "flex",
            flexDirection: "column",
            alignItems: "flex-start",
            lineHeight: 1.1,
            border: "1px solid #2d3f55",
            minWidth: 70,
          };
          const valStyle = { fontWeight: 600, fontSize: "0.8rem" };
          const labelStyle = {
            fontSize: "0.55rem",
            opacity: 0.65,
            letterSpacing: 0.5,
          };
          return (
            <div
              style={{
                marginLeft: "auto",
                display: "flex",
                gap: "0.5rem",
                alignItems: "stretch",
                fontSize: "0.75rem",
                flexWrap: "wrap",
              }}
            >
              <div style={badgeStyle} title="Total spins recorded">
                <span style={labelStyle}>SPINS</span>
                <span style={valStyle}>{history.length}</span>
              </div>
              <div
                style={badgeStyle}
                title="Accuracy over last 50 predictions (rolling eval window)"
              >
                <span style={labelStyle}>ACC (50)</span>
                <span style={valStyle}>{acc50}</span>
              </div>
              <div
                style={badgeStyle}
                title="Brier score (lower better) over last 50 predictions"
              >
                <span style={labelStyle}>BRIER (50)</span>
                <span style={valStyle}>{brier50}</span>
              </div>
              <div
                style={badgeStyle}
                title="Current max probability among classes (model confidence)"
              >
                <span style={labelStyle}>CONF</span>
                <span style={valStyle}>{confidence}</span>
              </div>
              <div
                style={{
                  ...badgeStyle,
                  borderColor: isTraining ? "#6b4f1f" : "#2d3f55",
                  background: isTraining ? "#3a2f16" : "#1e2835",
                }}
                title="Incremental training status"
              >
                <span style={labelStyle}>TRAIN</span>
                <span
                  style={{
                    ...valStyle,
                    color: isTraining ? "#fbbf24" : "#4ade80",
                  }}
                >
                  {isTraining ? "Running" : "Idle"}
                </span>
              </div>
              <div
                style={{
                  ...badgeStyle,
                  borderColor: isSimulating ? "#1e4f6b" : "#2d3f55",
                  background: isSimulating ? "#143040" : "#1e2835",
                }}
                title="Queued spins waiting to process"
              >
                <span style={labelStyle}>QUEUE</span>
                <span
                  style={{
                    ...valStyle,
                    color: isSimulating ? "#60a5fa" : "#aaa",
                  }}
                >
                  {pendingQueue}
                </span>
              </div>
            </div>
          );
        })()}
        <div style={{ marginLeft: 12, position: "relative" }}>
          <div className="toast-anchor">
            {alerts.map((a) => (
              <div
                key={a.id}
                className={`alert-toast alert-${a.type}`}
                role={a.type === "error" ? "alert" : "status"}
                aria-live={a.type === "error" ? "assertive" : "polite"}
              >
                {a.msg}
              </div>
            ))}
          </div>
        </div>
      </header>
      <section className="controls">
        <SpinInput onAdd={handleManualClassAdd} />
        <div className="btn-row">
          <button onClick={() => runSimulation(20)}>Sim 20</button>
          <button onClick={() => runSimulation(100)}>Sim 100</button>
          <button onClick={() => runSimulationBatched(500)}>Sim 500</button>
          {/* Manual simulation count control */}
          <div style={{ display: "inline-flex", alignItems: "center", gap: 8 }}>
            <input
              type="number"
              min={1}
              value={simCountInput ?? 100}
              onChange={(e) => setSimCountInput(Number(e.target.value))}
              style={{
                width: 90,
                padding: "6px 8px",
                borderRadius: 6,
                border: "1px solid #2d3f55",
                background: "var(--card)",
                color: "var(--text)",
              }}
              aria-label="Number of simulations"
            />
            <button
              onClick={() => runSimulationBatched(Number(simCountInput || 100))}
            >
              Run
            </button>
          </div>
          <button
            onClick={() => {
              if (
                confirm("Run 8000 spin stress test? This may take some time.")
              )
                runSimulationBatched(8000);
            }}
          >
            Stress 8k
          </button>
          <button onClick={stopSimulation} disabled={!isSimulating}>
            Stop Sim
          </button>
          <button onClick={clearState}>Clear State</button>
          <button
            onClick={() =>
              exportHistoryCSV(
                history.map((c, i) => ({
                  spin: c,
                  ts: Date.now() - (history.length - i) * 1000,
                }))
              )
            }
          >
            Export History CSV
          </button>
          <button onClick={exportState}>Export JSON</button>
          <label className="import-btn" title="Import JSON / CSV state">
            <span className="import-visual">📥 Import</span>
            <input
              type="file"
              multiple
              onChange={(e) => importHandler(e.target.files)}
              aria-label="Import state files"
            />
          </label>
        </div>
        <div className="window-controls">
          <div className="window-toggles">
            {metricsWindowOptions.map((opt) => (
              <button
                key={opt}
                className={opt === viewWindow ? "active" : ""}
                onClick={() => {
                  setViewWindow(opt);
                  if (opt !== "all") setWindowSize(parseInt(opt, 10));
                }}
              >
                {opt}
              </button>
            ))}
          </div>
          <div className="slider-box compact">
            <label>Custom Window: {windowSize}</label>
            <input
              type="range"
              min="10"
              max={history.length || 10}
              value={Math.min(windowSize, history.length || 10)}
              onChange={(e) => {
                const val = parseInt(e.target.value, 10);
                setWindowSize(val);
                setViewWindow(val.toString());
              }}
            />
          </div>
        </div>
        <fieldset
          style={{
            border: "1px solid #2d3f55",
            padding: "0.75rem 1rem",
            borderRadius: 8,
            marginTop: "0.75rem",
            background: "#1a2430",
            display: "flex",
            flexWrap: "wrap",
            gap: "1.5rem",
          }}
        >
          <legend style={{ padding: "0 6px", fontSize: "0.75rem" }}>
            Repeat Penalty Tuning
          </legend>
          <div style={{ minWidth: 180 }}>
            <label style={{ fontSize: "0.65rem", opacity: 0.75 }}>
              Penalty (reduce top prob if repeats)
            </label>
            <input
              type="range"
              min={0}
              max={0.2}
              step={0.005}
              value={hyperparams.repeatPenalty ?? 0.08}
              onChange={(e) => {
                const val = parseFloat(e.target.value);
                setHyperparams((h) => ({ ...h, repeatPenalty: val }));
                setPenaltyBaselineIndex((idx) => idx ?? history.length);
              }}
              style={{ width: "100%" }}
            />
            <div style={{ fontSize: "0.65rem" }}>
              {(hyperparams.repeatPenalty ?? 0.08).toFixed(3)}
            </div>
          </div>
          <div style={{ minWidth: 220 }}>
            <label style={{ fontSize: "0.65rem", opacity: 0.75 }}>
              Decay Lambda (recency weighting for training)
            </label>
            <input
              type="range"
              min={0}
              max={0.1}
              step={0.001}
              value={hyperparams.decayLambda ?? 0}
              onChange={(e) => {
                const val = parseFloat(e.target.value);
                setHyperparams((h) => ({ ...h, decayLambda: val }));
              }}
              style={{ width: "100%" }}
            />
            <div style={{ fontSize: "0.65rem" }}>
              {(hyperparams.decayLambda ?? 0).toFixed(3)}
            </div>
          </div>
          <div style={{ minWidth: 220 }}>
            <label style={{ fontSize: "0.65rem", opacity: 0.75 }}>
              MC Samples
            </label>
            <input
              type="number"
              min={1}
              max={50}
              value={hyperparams.mcSamples ?? 5}
              onChange={(e) =>
                setHyperparams((h) => ({
                  ...h,
                  mcSamples: parseInt(e.target.value || "5", 10),
                }))
              }
              style={{ width: "100%" }}
            />
            <div style={{ fontSize: "0.65rem" }}>
              {hyperparams.mcSamples ?? 5}
            </div>
          </div>
          <div style={{ minWidth: 180 }}>
            <label style={{ fontSize: "0.65rem", opacity: 0.75 }}>
              MC Uncertainty Thresh
            </label>
            <input
              type="range"
              min={0}
              max={0.2}
              step={0.001}
              value={hyperparams.mcUncertaintyThreshold ?? 0.05}
              onChange={(e) =>
                setHyperparams((h) => ({
                  ...h,
                  mcUncertaintyThreshold: parseFloat(e.target.value),
                }))
              }
              style={{ width: "100%" }}
            />
            <div style={{ fontSize: "0.65rem" }}>
              {(hyperparams.mcUncertaintyThreshold ?? 0.05).toFixed(3)}
            </div>
          </div>
          <div style={{ minWidth: 180 }}>
            <label style={{ fontSize: "0.65rem", opacity: 0.75 }}>
              Val Split
            </label>
            <input
              type="range"
              min={0}
              max={0.5}
              step={0.01}
              value={hyperparams.validationSplit ?? 0.1}
              onChange={(e) =>
                setHyperparams((h) => ({
                  ...h,
                  validationSplit: parseFloat(e.target.value),
                }))
              }
              style={{ width: "100%" }}
            />
            <div style={{ fontSize: "0.65rem" }}>
              {(hyperparams.validationSplit ?? 0.1).toFixed(2)}
            </div>
          </div>
          <div style={{ minWidth: 140 }}>
            <label style={{ fontSize: "0.65rem", opacity: 0.75 }}>
              Early Stop Patience
            </label>
            <input
              type="number"
              min={0}
              max={10}
              value={hyperparams.earlyStoppingPatience ?? 3}
              onChange={(e) =>
                setHyperparams((h) => ({
                  ...h,
                  earlyStoppingPatience: parseInt(e.target.value || "3", 10),
                }))
              }
              style={{ width: "100%" }}
            />
            <div style={{ fontSize: "0.65rem" }}>
              {hyperparams.earlyStoppingPatience ?? 3}
            </div>
          </div>
          <div style={{ minWidth: 180 }}>
            <label style={{ fontSize: "0.65rem", opacity: 0.75 }}>
              Min Gap to Trigger (confidence gap)
            </label>
            <input
              type="range"
              min={0}
              max={0.2}
              step={0.005}
              value={hyperparams.repeatMinGap ?? 0.07}
              onChange={(e) => {
                const val = parseFloat(e.target.value);
                setHyperparams((h) => ({ ...h, repeatMinGap: val }));
                setPenaltyBaselineIndex((idx) => idx ?? history.length);
              }}
              style={{ width: "100%" }}
            />
            <div style={{ fontSize: "0.65rem" }}>
              {(hyperparams.repeatMinGap ?? 0.07).toFixed(3)}
            </div>
          </div>
          <div style={{ fontSize: "0.6rem", maxWidth: 340, lineHeight: 1.3 }}>
            Baseline set at spin index {penaltyBaselineIndex ?? "—"}. Pre / post
            accuracy computed relative to that point.
          </div>
          <details className="glossary">
            <summary style={{ cursor: "pointer" }}>
              Glossary — how to tune controls
            </summary>
            <div
              style={{ paddingTop: 8, fontSize: "0.85rem", lineHeight: 1.35 }}
            >
              <dl>
                <dt style={{ fontWeight: 600 }}>Penalty</dt>
                <dd style={{ margin: "0 0 0.6rem 0" }}>
                  Reduces the top predicted probability when the same class is
                  repeatedly predicted. Increase slightly if the system gets
                  stuck repeating a class incorrectly; decrease if it becomes
                  overly conservative.
                </dd>

                <dt style={{ fontWeight: 600 }}>Min Gap to Trigger</dt>
                <dd style={{ margin: "0 0 0.6rem 0" }}>
                  Minimum confidence gap between top and second prediction
                  required to apply the repeat penalty. Raise to avoid
                  penalizing when the model is clearly confident; lower to be
                  more aggressive.
                </dd>

                <dt style={{ fontWeight: 600 }}>Decay Lambda</dt>
                <dd style={{ margin: "0 0 0.6rem 0" }}>
                  Recency weighting applied during training resampling. Small
                  positive values prioritize recent spins to adapt to drift; set
                  near 0 to treat history uniformly.
                </dd>

                <dt style={{ fontWeight: 600 }}>MC Samples</dt>
                <dd style={{ margin: "0 0 0.6rem 0" }}>
                  Number of stochastic forward passes used to estimate model
                  uncertainty. More samples give better uncertainty estimates
                  but increase compute time.
                </dd>

                <dt style={{ fontWeight: 600 }}>MC Uncertainty Thresh</dt>
                <dd style={{ margin: "0 0 0.6rem 0" }}>
                  Threshold on mean per-class std from MC sampling. Samples with
                  uncertainty above this are skipped during training to avoid
                  learning from noisy labels. Lower to be stricter.
                </dd>

                <dt style={{ fontWeight: 600 }}>Val Split</dt>
                <dd style={{ margin: "0 0 0.6rem 0" }}>
                  Fraction of training data held out for validation used for
                  early stopping. Typical values 0.05–0.2. Increase to get
                  better early stopping signals, but leave enough data to train
                  on.
                </dd>

                <dt style={{ fontWeight: 600 }}>Early Stop Patience</dt>
                <dd style={{ margin: "0 0 0.6rem 0" }}>
                  Number of validation epochs with no improvement before
                  training halts. Set small (1–3) for quick stops, higher to
                  allow slow improvements.
                </dd>
              </dl>
            </div>
          </details>
        </fieldset>
      </section>
      <div className="layout-grid">
        <PredictionCard
          probs={probsObj}
          prediction={["zero", "first", "second", "third"][currentPrediction]}
          uncertainty={
            mlUncertainty
              ? {
                  0: mlUncertainty[0] || 0,
                  1: mlUncertainty[1] || 0,
                  2: mlUncertainty[2] || 0,
                  3: mlUncertainty[3] || 0,
                }
              : null
          }
        />
        <HistoryTable history={history} predictionRecords={predictionRecords} />
        <MetricsPanel
          history={history.map((c) => ({
            spin: c === 0 ? 0 : c === 1 ? 5 : c === 2 ? 17 : 30,
            ts: Date.now(),
          }))}
          windowSize={windowSize}
          viewWindow={viewWindow}
        />
        <DiagnosticsChart
          history={history}
          predictionRecords={predictionRecords}
          windowSize={windowSize}
        />
      </div>
    </div>
  );
}
