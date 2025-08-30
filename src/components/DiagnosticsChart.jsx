import React, { useMemo } from "react";
import { Line } from "react-chartjs-2";
import {
  Chart as ChartJS,
  LineElement,
  CategoryScale,
  LinearScale,
  PointElement,
  Tooltip,
  Legend,
} from "chart.js";

ChartJS.register(
  LineElement,
  CategoryScale,
  LinearScale,
  PointElement,
  Tooltip,
  Legend
);

export default function DiagnosticsChart({
  history,
  predictionRecords,
  windowSize,
}) {
  const metrics = useMemo(() => {
    const acc = [];
    const brier = [];
    for (
      let i = 0;
      i < Math.min(history.length, predictionRecords.length);
      i++
    ) {
      const rec = predictionRecords[i];
      if (!rec) continue;
      const truth = history[i];
      acc.push(rec.predicted === truth ? 1 : 0);
      const probs = rec.probs || [0.25, 0.25, 0.25, 0.25];
      let bs = 0;
      for (let k = 0; k < 4; k++) {
        const y = truth === k ? 1 : 0;
        bs += Math.pow(probs[k] - y, 2);
      }
      bs /= 4;
      brier.push(bs);
    }
    const labels = acc.map((_, i) => i + 1);
    return { labels, acc, brier };
  }, [history, predictionRecords]);

  const sliced = useMemo(() => {
    const { labels, acc, brier } = metrics;
    const len = labels.length;
    const start = Math.max(0, len - windowSize);
    return {
      labels: labels.slice(start),
      acc: acc.slice(start),
      brier: brier.slice(start),
    };
  }, [metrics, windowSize]);

  const roll = (arr, w) => {
    const out = [];
    let sum = 0;
    const q = [];
    for (let i = 0; i < arr.length; i++) {
      sum += arr[i];
      q.push(arr[i]);
      if (q.length > w) sum -= q.shift();
      out.push(sum / Math.min(q.length, w));
    }
    return out;
  };
  const accRoll = roll(sliced.acc, Math.min(20, sliced.acc.length));
  const brierRoll = roll(sliced.brier, Math.min(20, sliced.brier.length));

  const chartData = {
    labels: sliced.labels,
    datasets: [
      {
        label: "Accuracy (rolling)",
        data: accRoll,
        borderColor: "#4ade80",
        yAxisID: "y1",
        tension: 0.2,
      },
      {
        label: "Brier (rolling)",
        data: brierRoll,
        borderColor: "#f87171",
        yAxisID: "y2",
        tension: 0.2,
      },
    ],
  };

  return (
    <div className="card metrics-card">
      <h2>Performance Diagnostics</h2>
      <Line
        data={chartData}
        options={{
          animation: false,
          interaction: { mode: "index", intersect: false },
          stacked: false,
          plugins: { legend: { labels: { color: "#ddd" } } },
          scales: {
            x: { ticks: { color: "#888" }, grid: { color: "#222" } },
            y1: {
              type: "linear",
              position: "left",
              min: 0,
              max: 1,
              ticks: { color: "#4ade80" },
              grid: { color: "#222" },
            },
            y2: {
              type: "linear",
              position: "right",
              min: 0,
              max: 0.6,
              ticks: { color: "#f87171" },
              grid: { drawOnChartArea: false },
            },
          },
        }}
      />
    </div>
  );
}
