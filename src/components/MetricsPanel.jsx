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

export default function MetricsPanel({ history, windowSize, viewWindow }) {
  // Placeholder metrics: distribution of dozens over rolling window
  const data = useMemo(() => {
    const counts = [];
    const spins = history.map((h) => h.spin);
    for (let i = 0; i < spins.length; i++) {
      const start = Math.max(0, i - windowSize + 1);
      const slice = spins.slice(start, i + 1);
      const c = { first: 0, second: 0, third: 0, zero: 0 };
      slice.forEach((s) => {
        if (s === 0) c.zero++;
        else if (s <= 12) c.first++;
        else if (s <= 24) c.second++;
        else c.third++;
      });
      const denom = slice.length || 1;
      counts.push({
        i: i + 1,
        first: c.first / denom,
        second: c.second / denom,
        third: c.third / denom,
        zero: c.zero / denom,
      });
    }
    return counts;
  }, [history, windowSize]);

  const chartData = {
    labels: data.map((d) => d.i),
    datasets: [
      {
        label: "First",
        data: data.map((d) => d.first),
        borderColor: "#4ade80",
        tension: 0.2,
      },
      {
        label: "Second",
        data: data.map((d) => d.second),
        borderColor: "#60a5fa",
        tension: 0.2,
      },
      {
        label: "Third",
        data: data.map((d) => d.third),
        borderColor: "#fbbf24",
        tension: 0.2,
      },
      {
        label: "Zero",
        data: data.map((d) => d.zero),
        borderColor: "#f87171",
        tension: 0.2,
      },
    ],
  };

  return (
    <div className="card metrics-card">
      <h2>Diagnostics (Rolling Distribution)</h2>
      <Line
        data={chartData}
        options={{
          animation: false,
          plugins: {
            legend: { labels: { color: "#ddd" } },
            tooltip: { callbacks: {} },
          },
          scales: {
            x: { ticks: { color: "#888" }, grid: { color: "#222" } },
            y: {
              min: 0,
              max: 1,
              ticks: { color: "#888" },
              grid: { color: "#222" },
            },
          },
        }}
      />
    </div>
  );
}
