# Measure Theory and Integration (amari-measure)

*Added in v0.10.0*

Lebesgue integration, probability measures, and measure-theoretic foundations.

## Quick Start

```typescript
import { WasmLebesgueMeasure, WasmProbabilityMeasure, integrate } from '@justinelliottcobb/amari-wasm';

// Lebesgue measure - compute volumes
const measure = new WasmLebesgueMeasure(3); // 3D space
const volume = measure.measureBox([2.0, 3.0, 4.0]); // 2x3x4 box = 24

// Numerical integration
const f = (x) => x * x;
const result = integrate(f, 0, 2, 1000, WasmIntegrationMethod.Riemann);
console.log(`int_0^2 x^2 dx = ${result}`); // ~2.667

// Probability measures
const prob = WasmProbabilityMeasure.uniform(0, 1);
const p = prob.probabilityInterval(0.25, 0.75, 0, 1); // P(0.25 <= X <= 0.75) = 0.5
```

## Use Cases

- **Scientific Computing**: Numerical integration
- **Probability & Statistics**: Measure-theoretic probability
- **Monte Carlo Methods**: Numerical estimation of integrals
