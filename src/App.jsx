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

  const pushSpin = useCallback(
    (cls) => {
      setHistory((prev) => {
        const newHist = [...prev, cls];
        setPredictionRecords((pr) => {
          const probsSafe = sanitizeProbs(currentProbs);
          const predicted = probsSafe.indexOf(Math.max(...probsSafe));
          const rec = { probs: probsSafe, predicted };
          return [...pr, rec];
        });
        // (Removed export milestone prompt per user request)
        if (newHist.length === 1)
          pushAlert("First spin recorded", "info", 4000);
        return newHist;
      });
    },
    [currentProbs, currentPrediction]
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
    const burst = simQueueRef.current.splice(0, 25); // 25 per tick
    burst.forEach((c) => pushSpin(c));
    setPendingQueue(simQueueRef.current.length);
    setTimeout(processSimQueue, 40); // pacing delay
  }, [pushSpin]);
  const runSimulation = (count = 10) => {
    enqueueSimulation(count);
    if (!isSimulating) {
      simStopRef.current = false;
      setIsSimulating(true);
      processSimQueue();
    }
  };
  const runSimulationBatched = (total = 500) => {
    enqueueSimulation(total);
    if (!isSimulating) {
      simStopRef.current = false;
      setIsSimulating(true);
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
        const sources = [
          bayes.probs,
          mk.probs,
          st.probs,
          ew.probs,
          mlProbs || statBlend,
          hmm.probs,
          mc.probs,
        ];
        if (dqn.nSources !== sources.length)
          setDqn(new DQNWeights({ nSources: sources.length }));
        const weights = dqn.weights.slice(0, sources.length);
        // Sanitize each source before blending to avoid propagating NaNs
        const safeSources = sources.map((s) => sanitizeProbs(s));
        const blendedRaw = blendLogSpace(safeSources, weights, {
          temperature: hyperparams.temperature || 1.0,
          clampMax: hyperparams.clampMax || 0.92,
        });
        const blendedSafe = sanitizeProbs(blendedRaw);
        // Optional repeat penalty to avoid always parroting the last class when alternatives are close.
        const lastClass = history[history.length - 1];
        let penalized = blendedSafe;
        if (lastClass != null) {
          const penalty = hyperparams.repeatPenalty ?? 0.08; // reduce up to 8 percentage points (relative mass) if close
          const minGap = hyperparams.repeatMinGap ?? 0.07; // only penalize if advantage over 2nd best < 7%
          const sorted = [...blendedSafe].sort((a, b) => b - a);
          const top = sorted[0];
          const second = sorted[1] ?? 0;
          if (blendedSafe[lastClass] === top && top - second < minGap) {
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
        setCalibrationState(cal.calibrationState);
        setEnsembleProbs(calSafe);
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
    const debounced = debounce(fn, 120);
    recomputeRef.current = debounced;
    return () => debounced.cancel();
  }, [
    history,
    mlProbs,
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
    setDqn(
      new DQNWeights({
        nSources: dqn.nSources,
        learningRate: dqn.learningRate,
        epsilon: dqn.epsilon,
      })
    );
    // copy weights
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
  useEffect(() => {
    if (!mlModel) return;
    if (history.length < seqLen + 10) return; // need minimal samples
    let canceled = false;
    let retried = false;
    (async () => {
      try {
        if (trainingRef.current) return; // prevent overlapping fit calls
        trainingRef.current = true;
        setIsTraining(true);
        await trainIncremental(mlModel, history, {
          seqLen,
          epochs: 2,
          batchSize: 32,
        });
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
                });
              }
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
      </header>
      {alerts.length > 0 && (
        <div style={{ padding: "0.5rem 2rem", width: "100%" }}>
          {alerts.map((a) => (
            <div
              key={a.id}
              className="alert-banner"
              style={{
                background:
                  a.type === "error"
                    ? "#3d1f1f"
                    : a.type === "metric"
                    ? "#1e2f1e"
                    : "#1e2835",
                borderColor:
                  a.type === "error"
                    ? "#6b2d2d"
                    : a.type === "metric"
                    ? "#2f6b3a"
                    : "#2d3f55",
              }}
            >
              {a.msg}
            </div>
          ))}
        </div>
      )}
      <section className="controls">
        <SpinInput onAdd={handleManualClassAdd} />
        <div className="btn-row">
          <button onClick={() => runSimulation(20)}>Sim 20</button>
          <button onClick={() => runSimulation(100)}>Sim 100</button>
          <button onClick={() => runSimulationBatched(500)}>Sim 500</button>
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
                onClick={() => setViewWindow(opt)}
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
              onChange={(e) => setWindowSize(parseInt(e.target.value, 10))}
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
