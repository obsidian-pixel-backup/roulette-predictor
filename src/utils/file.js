import Papa from "papaparse";

export function exportHistoryCSV(history) {
  if (!history || !history.length) return;
  const rows = history.map((h, i) => ({
    index: i + 1,
    spin: h.spin,
    timestamp: new Date(h.ts).toISOString(),
  }));
  const csv = Papa.unparse(rows);
  const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "spin_history.csv";
  a.click();
  URL.revokeObjectURL(a.href);
}

export function exportStateJSON(state) {
  const blob = new Blob([JSON.stringify(state, null, 2)], {
    type: "application/json",
  });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "roulette_state.json";
  a.click();
  URL.revokeObjectURL(a.href);
}

export async function importFromFiles(fileList) {
  const result = {};
  const files = Array.from(fileList || []);
  for (const file of files) {
    if (file.name.endsWith(".json")) {
      const text = await file.text();
      try {
        Object.assign(result, JSON.parse(text));
      } catch (e) {
        console.error("JSON parse error", e);
      }
    } else if (file.name.endsWith(".csv")) {
      const text = await file.text();
      const parsed = Papa.parse(text, { header: true });
      if (parsed.data) {
        result.history = parsed.data
          .filter(
            (r) => r.spin !== undefined && r.spin !== null && r.spin !== ""
          )
          .map((r) => ({
            spin: Number(r.spin),
            ts: Date.parse(r.timestamp) || Date.now(),
          }));
      }
    }
  }
  return result;
}
