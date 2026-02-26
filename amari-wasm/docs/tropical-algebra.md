# Tropical Algebra (amari-tropical)

Tropical algebra replaces addition with max and multiplication with addition, providing a framework for optimization and shortest-path algorithms.

## Quick Start

```typescript
import { tropical_add, tropical_multiply } from '@justinelliottcobb/amari-wasm';

// Tropical operations: add = max, multiply = add
const a = 5.0, b = 3.0;
const trop_sum = tropical_add(a, b);  // max(5, 3) = 5
const trop_prod = tropical_multiply(a, b);  // 5 + 3 = 8
```

## API Reference

### Tropical Number Operations

- `WasmTropicalNumber(value)`: Create tropical number
- `WasmTropicalNumber.tropicalAdd(other)`: Tropical addition (max)
- `WasmTropicalNumber.tropicalMultiply(other)`: Tropical multiplication (classical +)
- `WasmTropicalNumber.tropicalPower(exp)`: Tropical exponentiation

### Tropical Viterbi

- `WasmTropicalViterbi(states, observations, transitions, emissions, initial)`: Create HMM decoder
- `WasmTropicalViterbi.decode(observations)`: Run Viterbi decoding

### Tropical Polynomial

- `WasmTropicalPolynomial(coefficients)`: Create tropical polynomial
- `WasmTropicalPolynomial.evaluate(x)`: Evaluate at point
- `WasmTropicalPolynomial.add(other)`: Add polynomials
- `WasmTropicalPolynomial.multiply(other)`: Multiply polynomials

### Batch & ML Operations

- `TropicalBatch.max(a, b)`: Batch tropical max
- `TropicalBatch.add(a, b)`: Batch tropical add
- `TropicalMLOps.convexCombination(a, b, t)`: Tropical convex combination
- `TropicalMLOps.matrixMultiply(a, b, m, n, k)`: Tropical matrix multiply
- `TropicalMLOps.shortestPaths(matrix, n)`: All-pairs shortest paths (Floyd-Warshall)

## Use Cases

- **Optimization**: Shortest path and scheduling problems
- **Machine Learning**: Tropical neural networks
- **HMM Decoding**: Viterbi algorithm via tropical semiring
