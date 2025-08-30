// Simulation utility for generating spin class indices (0-3)
// 0 = zero, 1 = first dozen (1-12), 2 = second (13-24), 3 = third (25-36)
// Placeholder: uniform distribution. Future: bias detection + adaptive distribution based on model diagnostics.
export function simulateSpins(count, pushSpin) {
  for (let i = 0; i < count; i++) {
    const r = Math.random();
    // Uniform over 37 pockets aggregated into 4 classes:
    // P(zero)=1/37, each dozen = 12/37
    const pZero = 1 / 37;
    const pDozen = 12 / 37; // cumulative ranges
    let cls;
    if (r < pZero) cls = 0;
    else if (r < pZero + pDozen) cls = 1;
    else if (r < pZero + 2 * pDozen) cls = 2;
    else cls = 3;
    pushSpin(cls);
  }
}
