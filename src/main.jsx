import React from "react";
import { createRoot } from "react-dom/client";
import App from "./App.jsx";
import "./index.css";
import * as tf from "@tensorflow/tfjs";

(async () => {
  try {
    await tf.setBackend("webgl");
  } catch (e) {
    console.warn("Falling back to cpu backend", e);
    await tf.setBackend("cpu");
  }
  await tf.ready();
  console.log("TF Backend:", tf.getBackend());
  const root = createRoot(document.getElementById("root"));
  root.render(<App />);
})();
