# Roulette Dozen Predictor (Steps 1-2)

Foundational project scaffolding created with Vite + React + TensorFlow.js.

## Available Scripts

- Development: `npm run dev`
- Build: `npm run build`
- Preview build: `npm run preview`

## Features Implemented (Steps 1-6)

1. Project scaffold (Vite + React + TF.js) & dark fintech theme.
2. Class-based data model: spins stored as 0 (zero) / 1 / 2 / 3 for dozens.
3. Spin input via dropdown component (`SpinInput`).
4. Simulation utility (`simulateSpins`) producing uniform class distribution aggregated from 37 pockets.
5. History & prediction tracking arrays (`history`, `predictionRecords`).
6. Naive Laplace-smoothed probability estimation baseline.
7. Updated `HistoryTable` shows actual vs predicted with probability columns and match indicator (limited to last 500 for performance).
8. Rolling distribution diagnostics chart with adjustable window slider & preset window toggles.
9. Statistical methods (Bayesian Dirichlet posterior, Markov transitions, streak continuation heuristic, EWMA adaptive smoothing) with ensemble averaging baseline.
10. CNN-LSTM hybrid with attention + Monte Carlo dropout inference (incremental training on recent history) integrated into ensemble.
11. Monte Carlo Markov simulation (adaptive sample count via tuner) and HMM regime detection as additional probability sources.
12. DQN-inspired adaptive weighting (bandit heuristic) blending all sources in log-space with temperature & clamp.
13. Auto-tuner (every 50 spins) adjusts hyperparameters (temperature, clampMax, MC sims, MC dropout samples, EWMA lambda, exploration epsilon) based on recent accuracy & Brier.
14. Calibration layer (temperature scaling, dynamic clamp, shrinkage) using rolling Brier & accuracy; probabilities adjusted post-blend.
15. Diagnostics: rolling Accuracy & Brier charts, window slider/presets; diagnostics summary persisted.
16. File utilities: export CSV (history), export JSON (history + predictions + hyperparams + statistical states + model JSON + DQN/HMM/MC + calibration + diagnostics), import normalization and reconstruction.
17. TensorFlow.js WebGL backend init (fallback CPU).

Upcoming steps: integrate statistical modules (Bayesian, Markov, streak, EWMA), build ensemble blending, CNN-LSTM + DQN components, autonomous hyperparameter tuning, advanced diagnostics & calibration, persistence of model weights, and performance optimization (workers, batching, adaptive sampling).

## Run Locally

```
npm install
npm run dev
```

Open the shown local URL (usually http://localhost:5173).
