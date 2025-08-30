import React from "react";

export default function PredictionCard({ probs, prediction, uncertainty }) {
  const ordered = [
    { key: "zero", label: "Zero" },
    { key: "first", label: "1st Dozen" },
    { key: "second", label: "2nd Dozen" },
    { key: "third", label: "3rd Dozen" },
  ];
  const suggestion = (() => {
    if (!prediction) return "—";
    if (prediction === "zero") return "Caution: Zero spike risk";
    return `Bet on ${
      prediction === "first"
        ? "1st Dozen"
        : prediction === "second"
        ? "2nd Dozen"
        : "3rd Dozen"
    }`;
  })();
  const max = Math.max(...ordered.map((o) => probs?.[o.key] || 0));
  return (
    <div className="card prediction-card">
      <h2>Prediction</h2>
      <div className="probs">
        {ordered.map(({ key, label }) => {
          const p = probs?.[key] || 0;
          const u = uncertainty
            ? uncertainty[
                key === "zero"
                  ? 0
                  : key === "first"
                  ? 1
                  : key === "second"
                  ? 2
                  : 3
              ]
            : null;
          return (
            <div
              key={key}
              className={"prob-row " + (prediction === key ? "highlight" : "")}
            >
              <span className="label" style={{ width: 90 }}>
                {label}
              </span>
              <div
                style={{
                  flex: 1,
                  margin: "0 8px",
                  background: "#0d1822",
                  borderRadius: 4,
                  position: "relative",
                  height: 10,
                }}
              >
                <div
                  style={{
                    position: "absolute",
                    inset: 0,
                    background: "#1b2a36",
                    borderRadius: 4,
                  }}
                />
                <div
                  style={{
                    position: "absolute",
                    left: 0,
                    top: 0,
                    bottom: 0,
                    width: `${(p / max) * 100}%`,
                    background: "linear-gradient(90deg,#2563eb,#9333ea)",
                    borderRadius: 4,
                  }}
                />
              </div>
              <span
                className="val"
                style={{ minWidth: 60, textAlign: "right" }}
              >
                {(p * 100).toFixed(1)}%
              </span>
              {u != null && (
                <span
                  className="val"
                  style={{ minWidth: 52, fontSize: "0.65rem", opacity: 0.7 }}
                >
                  ±{(u * 100).toFixed(1)}%
                </span>
              )}
            </div>
          );
        })}
      </div>
      <div className="suggested">{suggestion}</div>
    </div>
  );
}
