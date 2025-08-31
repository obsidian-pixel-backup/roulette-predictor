const fs = require("fs");
const s = fs.readFileSync(
  "c:/Users/corne/Downloads/roulette_state (6).json",
  "utf8"
);
const idx = s.indexOf("predictionRecords");
console.log("idx", idx);
if (idx > 0) {
  console.log(s.slice(Math.max(0, idx - 200), idx + 400));
}
