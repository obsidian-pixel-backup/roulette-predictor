import React, { useMemo } from "react";

// history: array of class indices (0=zero,1=first,2=second,3=third)
// predictionRecords: array of { probs: [p0..p3], predicted:int }
export default function HistoryTable({ history, predictionRecords }) {
  const maxDisplay = 500;
  const trimmed = useMemo(() => history.slice(-maxDisplay), [history]);
  const offset = history.length - trimmed.length;
  return (
    <div className="card history-card">
      <h2>History ({history.length})</h2>
      <div className="table-wrapper">
        <table>
          <thead>
            <tr>
              <th>#</th>
              <th>Actual</th>
              <th>Predicted</th>
              <th>Match</th>
              <th>P0</th>
              <th>P1</th>
              <th>P2</th>
              <th>P3</th>
            </tr>
          </thead>
          <tbody>
            {trimmed
              .map((cls, i) => {
                const idx = offset + i; // zero-based index
                const rec = predictionRecords[idx] || {};
                const predicted = rec.predicted;
                const probs = rec.probs || [];
                const match = predicted === cls ? "✓" : "";
                return (
                  <tr key={idx} className={match ? "row-match" : ""}>
                    <td>{idx + 1}</td>
                    <td>{cls}</td>
                    <td>{predicted != null ? predicted : "—"}</td>
                    <td>{match}</td>
                    {[0, 1, 2, 3].map((k) => (
                      <td key={k}>
                        {probs[k] != null ? (probs[k] * 100).toFixed(1) : "—"}
                      </td>
                    ))}
                  </tr>
                );
              })
              .reverse()}
          </tbody>
        </table>
      </div>
    </div>
  );
}
