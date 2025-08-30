import React from "react";

// Four quick-action buttons for class index input
export default function SpinInput({ onAdd }) {
  const add = (cls) => () => onAdd(cls);
  return (
    <div className="manual-form" style={{ gap: "0.4rem", flexWrap: "wrap" }}>
      <button type="button" onClick={add(0)} title="Zero">
        0
      </button>
      <button type="button" onClick={add(1)} title="1st Dozen 1-12">
        1st
      </button>
      <button type="button" onClick={add(2)} title="2nd Dozen 13-24">
        2nd
      </button>
      <button type="button" onClick={add(3)} title="3rd Dozen 25-36">
        3rd
      </button>
    </div>
  );
}
