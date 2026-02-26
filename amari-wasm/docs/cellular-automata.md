# Cellular Automata (amari-automata)

Geometric cellular automata with multivector states, inverse design, and self-assembly.

## Quick Start

```typescript
import { WasmGeometricCA, WasmMultivector } from '@justinelliottcobb/amari-wasm';

const ca = WasmGeometricCA.new(100, 100);

// Set initial configuration
ca.set_cell(50, 50, WasmMultivector.basis_vector(0));

// Evolve the system
for (let i = 0; i < 100; i++) {
  ca.step();
}

console.log(`Generation: ${ca.generation()}`);
```

## Use Cases

- **Pattern Formation**: Emergent patterns from geometric algebra rules
- **Self-Assembly**: Design target configurations via inverse optimization
- **Scientific Simulation**: Spatially extended dynamical systems
