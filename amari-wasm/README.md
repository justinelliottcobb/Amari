# @justinelliottcobb/amari-wasm v0.13.0

**Unified Mathematical Computing Library with High-Precision WebAssembly Support**

[![npm version](https://badge.fury.io/js/%40justinelliottcobb%2Famari-wasm.svg)](https://www.npmjs.com/package/@justinelliottcobb/amari-wasm)
[![CI](https://github.com/justinelliottcobb/Amari/actions/workflows/ci.yml/badge.svg)](https://github.com/justinelliottcobb/Amari/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Amari is a comprehensive mathematical computing library that brings advanced algebraic systems to JavaScript/TypeScript through WebAssembly. Features high-precision arithmetic for spacecraft orbital mechanics and relativistic physics calculations with pure Rust implementation and no native dependencies for universal deployment.

## Features

### Core Mathematical Systems

- **Geometric Algebra (Clifford Algebra)**: Multivectors, rotors, and geometric products for 3D rotations and spatial transformations
- **Tropical Algebra**: Max-plus semiring operations for optimization and neural network applications
- **Automatic Differentiation**: Forward-mode AD with dual numbers for exact derivatives
- **Measure Theory** *(v0.10.0)*: Lebesgue integration, probability measures, and measure-theoretic foundations
- **Holographic Memory** *(v0.12.3)*: Vector Symbolic Architecture for associative memory with binding and bundling operations
- **Optical Field Operations** *(v0.15.1)*: GA-native Lee hologram encoding for DMD displays and VSA-based optical processing
- **Probability Theory** *(v0.13.0)*: Distributions on multivector spaces, MCMC sampling, and Monte Carlo estimation
- **Relativistic Physics**: Spacetime algebra (Cl(1,3)) with WebAssembly-compatible precision
- **Spacecraft Orbital Mechanics**: Full-precision trajectory calculations in browsers
- **Cellular Automata**: Geometric cellular automata with multivector states
- **WebGPU Acceleration**: Optional GPU acceleration for large-scale operations
- **Pure Rust Implementation**: Memory-safe, high-performance core with WASM bindings
- **TypeScript Support**: Full TypeScript definitions included

### High-Precision Arithmetic

- **Pure Rust Backend**: dashu-powered arithmetic with no native dependencies
- **Universal Deployment**: Same precision guarantees across desktop, web, and edge environments
- **Orbital-Grade Tolerance**: Configurable precision for critical trajectory calculations
- **WebAssembly 3.0 Ready**: Leverages latest WASM features for enhanced mathematical computing

## Installation

```bash
npm install @justinelliottcobb/amari-wasm
```

Or with yarn:

```bash
yarn add @justinelliottcobb/amari-wasm
```

## Quick Start

```typescript
import init, { WasmMultivector, WasmRotor } from '@justinelliottcobb/amari-wasm';

async function main() {
  // Initialize the WASM module
  await init();

  // Create basis vectors
  const e1 = WasmMultivector.basis_vector(0);
  const e2 = WasmMultivector.basis_vector(1);

  // Compute geometric product
  const product = e1.geometric_product(e2);
  console.log(product.to_string()); // e12 (bivector)

  // Create a rotor for 90-degree rotation
  const rotor = WasmRotor.from_axis_angle(
    WasmMultivector.basis_vector(2), // z-axis
    Math.PI / 2
  );

  // Rotate a vector
  const vector = WasmMultivector.from_coefficients(
    new Float64Array([1, 0, 0, 0, 0, 0, 0, 0])
  );
  const rotated = rotor.rotate_vector(vector);

  // Clean up WASM memory
  e1.free();
  e2.free();
  product.free();
  rotor.free();
  vector.free();
  rotated.free();
}

main();
```

## High-Precision Orbital Mechanics

```typescript
import init, {
  WasmSpacetimeVector,
  WasmFourVelocity,
  WasmRelativisticParticle,
  WasmSchwarzschildMetric
} from '@justinelliottcobb/amari-wasm';

async function spacecraftSimulation() {
  await init();

  // Create Earth's gravitational field
  const earth = WasmSchwarzschildMetric.earth();

  // Spacecraft at 400km altitude (ISS orbit)
  const altitude = 400e3; // 400 km
  const earthRadius = 6.371e6; // Earth radius in meters
  const position = new Float64Array([earthRadius + altitude, 0.0, 0.0]);
  const velocity = new Float64Array([0.0, 7.67e3, 0.0]); // ~7.67 km/s orbital velocity

  // Create spacecraft with high-precision arithmetic
  const spacecraft = WasmRelativisticParticle.new(
    position,
    velocity,
    0.0,    // No charge
    1000.0, // 1000 kg spacecraft
    0.0     // No magnetic charge
  );

  // Propagate orbit for one complete period with high precision
  const orbitalPeriod = 5580.0; // ~93 minutes
  const timeStep = 60.0; // 1-minute time steps

  // High-precision geodesic integration using dashu backend
  const trajectory = spacecraft.propagate_trajectory(
    earth,
    orbitalPeriod,
    timeStep
  );

  console.log(`Orbital trajectory computed with ${trajectory.length} points`);
  console.log(`Final position deviation: ${spacecraft.position_error()} meters`);

  // WebAssembly precision matches native accuracy

  // Clean up WASM memory
  earth.free();
  spacecraft.free();
  trajectory.forEach(point => point.free());
}

spacecraftSimulation();
```

## Core Concepts

### Geometric Algebra

Geometric algebra extends linear algebra with the geometric product, enabling intuitive representation of rotations, reflections, and other transformations:

```typescript
// Multivector operations
const v1 = WasmMultivector.from_coefficients(coeffs);
const v2 = WasmMultivector.random();

const sum = v1.add(v2);
const product = v1.geometric_product(v2);
const wedge = v1.wedge_product(v2);
const inner = v1.inner_product(v2);
```

### Tropical Algebra

Tropical algebra replaces addition with max and multiplication with addition, useful for optimization:

```typescript
import { tropical_add, tropical_multiply } from '@justinelliottcobb/amari-wasm';

// Tropical operations: add = max, multiply = add
const a = 5.0, b = 3.0;
const trop_sum = tropical_add(a, b); // max(5, 3) = 5
const trop_prod = tropical_multiply(a, b); // 5 + 3 = 8
```

### Cellular Automata

Create and evolve cellular automata with geometric algebra states:

```typescript
const ca = WasmGeometricCA.new(100, 100);

// Set initial configuration
ca.set_cell(50, 50, WasmMultivector.basis_vector(0));

// Evolve the system
for (let i = 0; i < 100; i++) {
  ca.step();
}

console.log(`Generation: ${ca.generation()}`);
```

### Measure Theory and Integration *(v0.10.0)*

Perform numerical integration and work with probability measures:

```typescript
import { WasmLebesgueMeasure, WasmProbabilityMeasure, integrate } from '@justinelliottcobb/amari-wasm';

// Lebesgue measure - compute volumes
const measure = new WasmLebesgueMeasure(3); // 3D space
const volume = measure.measureBox([2.0, 3.0, 4.0]); // 2×3×4 box = 24

// Numerical integration
const f = (x) => x * x; // Function to integrate
const result = integrate(f, 0, 2, 1000, WasmIntegrationMethod.Riemann);
console.log(`∫₀² x² dx ≈ ${result}`); // ≈ 2.667

// Probability measures
const prob = WasmProbabilityMeasure.uniform(0, 1);
const p = prob.probabilityInterval(0.25, 0.75, 0, 1); // P(0.25 ≤ X ≤ 0.75) = 0.5
```

### Holographic Memory *(v0.12.2)*

Store and retrieve key-value associations using Vector Symbolic Architecture:

```typescript
import init, {
  WasmTropicalDualClifford,
  WasmHolographicMemory,
  WasmResonator
} from '@justinelliottcobb/amari-wasm';

async function holographicDemo() {
  await init();

  // Create random vectors for keys and values
  const key1 = WasmTropicalDualClifford.randomVector();
  const value1 = WasmTropicalDualClifford.randomVector();
  const key2 = WasmTropicalDualClifford.randomVector();
  const value2 = WasmTropicalDualClifford.randomVector();

  // Create holographic memory
  const memory = new WasmHolographicMemory();

  // Store associations
  memory.store(key1, value1);
  memory.store(key2, value2);

  // Retrieve with a key
  const result = memory.retrieve(key1);
  console.log(`Confidence: ${result.confidence()}`);
  console.log(`Similarity to original: ${result.value().similarity(value1)}`);

  // Check capacity
  const info = memory.capacityInfo();
  console.log(`Items stored: ${info.itemCount}`);
  console.log(`Estimated SNR: ${info.estimatedSnr}`);

  // Binding operations (key ⊛ value)
  const bound = key1.bind(value1);
  const unbound = bound.unbind(key1); // Recovers value1

  // Similarity computation
  const sim = key1.similarity(key2);
  console.log(`Key similarity: ${sim}`);

  // Resonator cleanup for noisy inputs
  const codebook = [key1, key2, value1, value2];
  const resonator = WasmResonator.new(codebook);
  const noisyInput = key1; // Add noise in practice
  const cleaned = resonator.cleanup(noisyInput);
  console.log(`Best match index: ${cleaned.bestMatchIndex()}`);

  // Clean up WASM memory
  key1.free();
  value1.free();
  key2.free();
  value2.free();
  memory.free();
  bound.free();
  unbound.free();
  resonator.free();
}

holographicDemo();
```

### Probability on Geometric Algebra *(v0.13.0)*

Sample from distributions on multivector spaces and perform Monte Carlo estimation:

```typescript
import init, {
  WasmGaussianMultivector,
  WasmUniformMultivector,
  WasmMonteCarloEstimator
} from '@justinelliottcobb/amari-wasm';

async function probabilisticDemo() {
  await init();

  // Create a standard Gaussian distribution on Cl(3,0,0)
  const gaussian = WasmGaussianMultivector.standard();

  // Draw samples
  const samples = [];
  for (let i = 0; i < 1000; i++) {
    samples.push(gaussian.sample());
  }
  console.log(`Drew ${samples.length} Gaussian samples`);

  // Compute log probability
  const sample = gaussian.sample();
  const logProb = gaussian.logProb(sample);
  console.log(`Log probability: ${logProb}`);

  // Grade-concentrated distribution (e.g., only on bivectors)
  const bivectorDist = WasmGaussianMultivector.gradeConcentrated(2, 1.0);
  const bivectorSample = bivectorDist.sample();

  // Uniform distribution on unit multivectors
  const uniform = WasmUniformMultivector.unitSphere();
  const unitSample = uniform.sample();

  // Monte Carlo estimation
  const estimator = new WasmMonteCarloEstimator();

  // Estimate expectation of a function
  const estimate = estimator.estimate((mv) => mv.norm(), gaussian, 10000);
  console.log(`Expected norm: ${estimate.mean} ± ${estimate.stdError}`);

  // Clean up WASM memory
  gaussian.free();
  bivectorDist.free();
  uniform.free();
  sample.free();
  bivectorSample.free();
  unitSample.free();
  estimator.free();
  samples.forEach(s => s.free());
}

probabilisticDemo();
```

### Optical Field Operations *(v0.15.1)*

Encode optical fields as binary holograms for DMD displays using geometric algebra:

```typescript
import init, {
  WasmOpticalRotorField,
  WasmBinaryHologram,
  WasmGeometricLeeEncoder,
  WasmOpticalFieldAlgebra,
  WasmOpticalCodebook,
  WasmTropicalOpticalAlgebra
} from '@justinelliottcobb/amari-wasm';

async function opticalDemo() {
  await init();

  // Create optical rotor fields (phase + amplitude on a grid)
  const field1 = WasmOpticalRotorField.random(256, 256, 12345n);
  const field2 = WasmOpticalRotorField.random(256, 256, 67890n);
  const uniform = WasmOpticalRotorField.uniform(0.0, 0.5, 256, 256);

  console.log(`Field dimensions: ${field1.width} x ${field1.height}`);
  console.log(`Total energy: ${field1.totalEnergy()}`);

  // Lee hologram encoding for DMD display
  const encoder = WasmGeometricLeeEncoder.withFrequency(256, 256, 0.25);
  const hologram = encoder.encode(uniform);

  console.log(`Hologram fill factor: ${hologram.fillFactor()}`);
  console.log(`Theoretical efficiency: ${encoder.theoreticalEfficiency(uniform)}`);

  // Get binary data for hardware interface
  const binaryData = hologram.asBytes();
  console.log(`Packed binary size: ${binaryData.length} bytes`);

  // VSA operations on optical fields
  const algebra = new WasmOpticalFieldAlgebra(256, 256);

  // Bind two fields (rotor multiplication = phase addition)
  const bound = algebra.bind(field1, field2);

  // Compute similarity between fields
  const similarity = algebra.similarity(field1, field1); // Self-similarity = 1.0
  console.log(`Self-similarity: ${similarity}`);

  // Unbind to retrieve original field
  const retrieved = algebra.unbind(field1, bound);
  const retrievalSim = algebra.similarity(retrieved, field2);
  console.log(`Retrieval similarity: ${retrievalSim}`);

  // Seed-based symbol codebook for VSA
  const codebook = new WasmOpticalCodebook(64, 64, 42n);
  codebook.register("AGENT");
  codebook.register("ACTION");
  codebook.register("TARGET");

  const agentField = codebook.get("AGENT");
  const actionField = codebook.get("ACTION");
  console.log(`Registered symbols: ${codebook.symbols()}`);

  // Tropical operations for attractor dynamics
  const tropical = new WasmTropicalOpticalAlgebra(64, 64);

  // Tropical add: pointwise minimum of phase magnitudes
  const tropicalSum = tropical.tropicalAdd(field1, field2);

  // Soft tropical add with temperature parameter
  const softSum = tropical.softTropicalAdd(field1, field2, 10.0);

  // Phase distance between fields
  const distance = tropical.phaseDistance(field1, field2);
  console.log(`Phase distance: ${distance}`);

  // Clean up WASM memory
  field1.free();
  field2.free();
  uniform.free();
  hologram.free();
  bound.free();
  retrieved.free();
  agentField.free();
  actionField.free();
  tropicalSum.free();
  softSum.free();
}

opticalDemo();
```

#### Optical Field API

**WasmOpticalRotorField:**
- `random(width, height, seed)`: Create random phase field
- `uniform(phase, amplitude, width, height)`: Uniform field
- `identity(width, height)`: Identity field (phase = 0)
- `fromPhase(phases, width, height)`: Create from phase array
- `phaseAt(x, y)`: Get phase at point (radians)
- `amplitudeAt(x, y)`: Get amplitude at point
- `totalEnergy()`: Sum of squared amplitudes
- `normalized()`: Normalized copy (energy = 1)

**WasmGeometricLeeEncoder:**
- `withFrequency(width, height, frequency)`: Create horizontal carrier encoder
- `new(width, height, frequency, angle)`: Create with angled carrier
- `encode(field)`: Encode to binary hologram
- `modulate(field)`: Get modulated field before thresholding
- `theoreticalEfficiency(field)`: Compute diffraction efficiency

**WasmBinaryHologram:**
- `get(x, y)`: Get pixel value
- `set(x, y, value)`: Set pixel value
- `fillFactor()`: Fraction of "on" pixels
- `hammingDistance(other)`: Compute Hamming distance
- `asBytes()`: Get packed binary data
- `inverted()`: Create inverted copy

**WasmOpticalFieldAlgebra:**
- `bind(a, b)`: Rotor multiplication (phase addition)
- `unbind(key, bound)`: Retrieve associated field
- `bundle(fields, weights)`: Weighted superposition
- `bundleUniform(fields)`: Equal-weight bundle
- `similarity(a, b)`: Normalized inner product
- `inverse(field)`: Phase negation
- `scale(field, factor)`: Amplitude scaling
- `addPhase(field, phase)`: Add constant phase

**WasmOpticalCodebook:**
- `new(width, height, baseSeed)`: Create codebook
- `register(symbol)`: Register symbol with auto-seed
- `get(symbol)`: Get or generate field for symbol
- `contains(symbol)`: Check if symbol is registered
- `symbols()`: Get all registered symbol names

**WasmTropicalOpticalAlgebra:**
- `tropicalAdd(a, b)`: Pointwise minimum phase magnitude
- `tropicalMax(a, b)`: Pointwise maximum phase magnitude
- `tropicalMul(a, b)`: Binding (phase addition)
- `softTropicalAdd(a, b, beta)`: Soft minimum with temperature
- `phaseDistance(a, b)`: Sum of absolute phase differences
- `attractorConverge(initial, attractors, maxIter, tol)`: Attractor dynamics

#### Probability API

**WasmGaussianMultivector:**
- `standard()`: Create standard Gaussian on full multivector space
- `new(mean, covariance)`: Create with specified mean and covariance
- `gradeConcentrated(grade, scale)`: Gaussian concentrated on specific grade
- `sample()`: Draw a random sample
- `logProb(sample)`: Compute log probability density

**WasmUniformMultivector:**
- `unitSphere()`: Uniform distribution on unit multivectors
- `gradeSimplex(grade)`: Uniform on grade components summing to 1
- `sample()`: Draw a random sample

**WasmMonteCarloEstimator:**
- `estimate(fn, distribution, nSamples)`: Estimate expectation
- `estimateVariance(fn, distribution, nSamples)`: Estimate variance

#### Holographic Memory API

**TropicalDualClifford Operations:**
- `bind(other)`: Binding operation using geometric product
- `unbind(other)`: Inverse binding for retrieval
- `bundle(other, beta)`: Bundling operation for superposition
- `similarity(other)`: Compute normalized similarity
- `bindingIdentity()`: Get the identity element for binding
- `bindingInverse()`: Compute approximate inverse
- `randomVector()`: Create a random unit vector
- `normalizeToUnit()`: Normalize to unit magnitude

**HolographicMemory:**
- `store(key, value)`: Store a key-value association
- `storeBatch(pairs)`: Store multiple associations efficiently
- `retrieve(key)`: Retrieve value associated with key
- `capacityInfo()`: Get storage statistics (item count, SNR, capacity)
- `clear()`: Clear all stored associations

**Resonator:**
- `new(codebook)`: Create resonator with clean reference vectors
- `cleanup(input)`: Clean up noisy input to nearest codebook entry
- `cleanupWithIterations(input, maxIter)`: Iterative cleanup

## Use Cases

- **Computer Graphics**: 3D rotations and transformations using rotors
- **Physics Simulations**: Geometric algebra for electromagnetic fields and relativistic calculations
- **Machine Learning**: Tropical neural networks and automatic differentiation
- **Optimization**: Tropical algebra for shortest path and scheduling problems
- **Scientific Computing**: High-performance mathematical operations with orbital-grade precision
- **Probability & Statistics**: Measure theory and numerical integration for statistical computations
- **Bayesian Inference**: Probabilistic modeling on geometric algebra spaces
- **Uncertainty Quantification**: Monte Carlo methods for error propagation
- **Game Development**: Efficient spatial transformations and physics
- **Spacecraft Trajectory Planning**: High-precision orbital mechanics in web applications
- **Symbolic AI**: Holographic memory for associative reasoning and concept binding
- **Cognitive Architectures**: Brain-inspired memory systems for AI agents
- **Embedding Retrieval**: Content-addressable semantic search in vector databases
- **Holographic Displays**: Lee hologram encoding for DMD and SLM devices
- **Optical Computing**: Phase-encoded VSA operations for optical neural networks

## API Reference

### Multivector Operations

- `WasmMultivector.basis_vector(index)`: Create a basis vector
- `WasmMultivector.scalar(value)`: Create a scalar multivector
- `WasmMultivector.from_coefficients(array)`: Create from coefficients
- `geometric_product(a, b)`: Compute geometric product
- `wedge_product(a, b)`: Compute wedge (outer) product
- `inner_product(a, b)`: Compute inner product

### Rotor Operations

- `WasmRotor.from_axis_angle(axis, angle)`: Create rotation rotor
- `WasmRotor.from_bivector(bivector, angle)`: Create from bivector
- `rotate_vector(vector)`: Apply rotation to vector
- `compose(other)`: Compose rotations

### Tropical Operations

- `tropical_add(a, b)`: Tropical addition (max)
- `tropical_multiply(a, b)`: Tropical multiplication (addition)
- `tropical_power(base, exp)`: Tropical exponentiation

### Holographic Memory Operations

- `WasmTropicalDualClifford.bind(other)`: Binding via geometric product
- `WasmTropicalDualClifford.unbind(other)`: Inverse binding
- `WasmTropicalDualClifford.bundle(other, beta)`: Superposition bundling
- `WasmTropicalDualClifford.similarity(other)`: Normalized similarity
- `WasmTropicalDualClifford.randomVector()`: Create random unit vector
- `WasmHolographicMemory.store(key, value)`: Store association
- `WasmHolographicMemory.retrieve(key)`: Retrieve by key
- `WasmResonator.cleanup(input)`: Clean up noisy input

### Probabilistic Operations

- `WasmGaussianMultivector.standard()`: Standard Gaussian distribution
- `WasmGaussianMultivector.gradeConcentrated(grade, scale)`: Grade-specific Gaussian
- `WasmGaussianMultivector.sample()`: Draw random sample
- `WasmGaussianMultivector.logProb(sample)`: Log probability density
- `WasmUniformMultivector.unitSphere()`: Uniform on unit sphere
- `WasmUniformMultivector.sample()`: Draw random sample
- `WasmMonteCarloEstimator.estimate(fn, dist, n)`: Monte Carlo expectation

### Optical Field Operations

- `WasmOpticalRotorField.random(width, height, seed)`: Create random phase field
- `WasmOpticalRotorField.uniform(phase, amplitude, width, height)`: Uniform field
- `WasmGeometricLeeEncoder.withFrequency(width, height, freq)`: Create Lee encoder
- `WasmGeometricLeeEncoder.encode(field)`: Encode to binary hologram
- `WasmBinaryHologram.fillFactor()`: Fraction of "on" pixels
- `WasmBinaryHologram.asBytes()`: Get packed binary data for hardware
- `WasmOpticalFieldAlgebra.bind(a, b)`: Rotor product (phase addition)
- `WasmOpticalFieldAlgebra.unbind(key, bound)`: Retrieve associated field
- `WasmOpticalFieldAlgebra.similarity(a, b)`: Normalized inner product
- `WasmOpticalCodebook.register(symbol)`: Register symbol with auto-seed
- `WasmOpticalCodebook.get(symbol)`: Get field for symbol
- `WasmTropicalOpticalAlgebra.tropicalAdd(a, b)`: Pointwise minimum phase

## Examples

Check out the [examples directory](https://github.com/justinelliottcobb/Amari/tree/master/examples) for more detailed usage:

- [Basic Geometric Algebra](https://github.com/justinelliottcobb/Amari/blob/master/examples/typescript/geometric.ts)
- [3D Rotations with Rotors](https://github.com/justinelliottcobb/Amari/blob/master/examples/typescript/rotations.ts)
- [Tropical Neural Networks](https://github.com/justinelliottcobb/Amari/blob/master/examples/typescript/tropical.ts)
- [Cellular Automata](https://github.com/justinelliottcobb/Amari/blob/master/examples/typescript/cellular.ts)

## Building from Source

```bash
# Clone the repository
git clone https://github.com/justinelliottcobb/Amari.git
cd Amari/amari-wasm

# Install dependencies
npm install

# Build WASM module
npm run build

# Run tests
npm test
```

## Contributing

Contributions are welcome. Please see [CONTRIBUTING.md](https://github.com/justinelliottcobb/Amari/blob/master/CONTRIBUTING.md) for details.

## License

MIT License - see [LICENSE](https://github.com/justinelliottcobb/Amari/blob/master/LICENSE) for details.

## Acknowledgments

- Built with Rust and wasm-bindgen
- Inspired by geometric algebra libraries like GAViewer and Ganja.js
- Tropical algebra concepts from discrete mathematics

## Contact

- GitHub: [@justinelliottcobb](https://github.com/justinelliottcobb)
- NPM: [@justinelliottcobb/amari-wasm](https://www.npmjs.com/package/@justinelliottcobb/amari-wasm)

---

Made with Rust by the Amari team
