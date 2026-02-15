# @justinelliottcobb/amari-wasm v0.18.0

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
- **Functional Analysis** *(v0.15.0)*: Hilbert spaces, linear operators, spectral decomposition, and Sobolev spaces
- **Optical Field Operations** *(v0.15.1)*: GA-native Lee hologram encoding for DMD displays and VSA-based optical processing
- **Computational Topology** *(v0.16.0)*: Simplicial complexes, homology computation, persistent homology, and Morse theory
- **Dynamical Systems** *(v0.18.0)*: ODE solvers, stability analysis, bifurcation diagrams, Lyapunov exponents, and phase portraits
- **Enumerative Geometry** *(v0.18.1)*: WDVV curve counting, equivariant localization, matroids, CSM classes, operadic composition, and stability conditions
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

### Functional Analysis *(v0.15.0)*

Work with Hilbert spaces, linear operators, and spectral decomposition on multivector spaces:

```typescript
import init, {
  WasmHilbertSpace,
  WasmMatrixOperator,
  WasmSpectralDecomposition,
  WasmSobolevSpace,
  powerMethod,
  computeEigenvalues
} from '@justinelliottcobb/amari-wasm';

async function functionalDemo() {
  await init();

  // Create a Hilbert space Cl(2,0,0) ≅ ℝ⁴
  const hilbert = new WasmHilbertSpace();
  console.log(`Dimension: ${hilbert.dimension()}`); // 4
  console.log(`Signature: ${hilbert.signature()}`); // [2, 0, 0]

  // Create multivectors from coefficients [scalar, e1, e2, e12]
  const x = hilbert.fromCoefficients([1.0, 2.0, 3.0, 4.0]);
  const y = hilbert.fromCoefficients([0.5, 1.5, 2.5, 3.5]);

  // Inner product and norm
  const inner = hilbert.innerProduct(x, y);
  console.log(`⟨x, y⟩ = ${inner}`);

  const norm = hilbert.norm(x);
  console.log(`‖x‖ = ${norm}`);

  // Orthogonal projection
  const proj = hilbert.project(x, y);
  console.log(`proj_y(x) = ${proj}`);

  // Create a matrix operator (4x4 matrix in row-major order)
  const A = new WasmMatrixOperator([
    4, 1, 0, 0,
    1, 3, 1, 0,
    0, 1, 2, 1,
    0, 0, 1, 1
  ]);

  // Apply operator to a vector
  const Ax = A.apply(x);
  console.log(`Ax = ${Ax}`);

  // Operator properties
  console.log(`‖A‖ = ${A.operatorNorm()}`);
  console.log(`tr(A) = ${A.trace()}`);
  console.log(`Symmetric: ${A.isSymmetric(1e-10)}`);

  // Spectral decomposition for symmetric matrices
  const decomp = WasmSpectralDecomposition.compute(A, 100, 1e-10);
  const eigenvalues = decomp.eigenvalues();
  console.log(`Eigenvalues: ${eigenvalues}`);
  console.log(`Spectral radius: ${decomp.spectralRadius()}`);
  console.log(`Condition number: ${decomp.conditionNumber()}`);

  // Functional calculus: apply f(A) = exp(A)
  const expAx = decomp.applyFunction((lambda) => Math.exp(lambda), x);
  console.log(`exp(A)x = ${expAx}`);

  // Power method for dominant eigenvalue
  const dominant = powerMethod(A, null, 100, 1e-10);
  console.log(`Dominant eigenvalue: ${dominant[0]}`);

  // Sobolev spaces for PDE analysis
  const h1 = new WasmSobolevSpace(1, 0.0, 1.0); // H^1([0,1])
  console.log(`Poincaré constant: ${h1.poincareConstant()}`);

  // Compute H^1 norm of sin(πx)
  const f = (x: number) => Math.sin(Math.PI * x);
  const df = (x: number) => Math.PI * Math.cos(Math.PI * x);
  const h1Norm = h1.h1Norm(f, df);
  console.log(`‖sin(πx)‖_{H^1} = ${h1Norm}`);

  // Clean up WASM memory
  hilbert.free();
  A.free();
  decomp.free();
  h1.free();
}

functionalDemo();
```

#### Functional Analysis API

**WasmHilbertSpace:**
- `new()`: Create Hilbert space Cl(2,0,0) ≅ ℝ⁴
- `dimension()`: Get space dimension
- `signature()`: Get Clifford algebra signature [p, q, r]
- `fromCoefficients(coeffs)`: Create multivector from coefficients
- `innerProduct(x, y)`: Compute ⟨x, y⟩
- `norm(x)`: Compute ‖x‖
- `distance(x, y)`: Compute d(x, y) = ‖x - y‖
- `normalize(x)`: Normalize to unit length
- `project(x, y)`: Orthogonal projection of x onto y
- `isOrthogonal(x, y, tol)`: Check orthogonality

**WasmMatrixOperator:**
- `new(entries)`: Create from 16 entries (4×4 row-major)
- `identity()`: Create identity operator
- `zero()`: Create zero operator
- `diagonal(entries)`: Create diagonal matrix
- `scaling(lambda)`: Create λI
- `apply(x)`: Apply operator T(x)
- `operatorNorm()`: Compute ‖T‖
- `isSymmetric(tol)`: Check symmetry
- `add(other)`: Add operators
- `compose(other)`: Compose operators (matrix multiply)
- `scale(lambda)`: Scale by scalar
- `transpose()`: Compute transpose
- `trace()`: Compute trace

**WasmSpectralDecomposition:**
- `compute(matrix, maxIter, tol)`: Compute eigenvalue decomposition
- `eigenvalues()`: Get eigenvalues
- `eigenvectors()`: Get eigenvectors (flattened)
- `isComplete()`: Check if decomposition is complete
- `spectralRadius()`: Get largest |eigenvalue|
- `conditionNumber()`: Get condition number
- `isPositiveDefinite()`: Check positive definiteness
- `apply(x)`: Apply reconstructed operator
- `applyFunction(f, x)`: Functional calculus f(T)x

**WasmSobolevSpace:**
- `new(order, lower, upper)`: Create H^k([a,b])
- `h1UnitInterval()`: Create H^1([0,1])
- `h2UnitInterval()`: Create H^2([0,1])
- `poincareConstant()`: Get Poincaré constant
- `h1Norm(f, df)`: Compute H^1 norm
- `h1Seminorm(df)`: Compute H^1 seminorm |f|_{H^1}
- `l2Norm(f)`: Compute L^2 norm
- `l2InnerProduct(f, g)`: Compute L^2 inner product

**Standalone Functions:**
- `powerMethod(matrix, initial, maxIter, tol)`: Dominant eigenvalue
- `inverseIteration(matrix, shift, initial, maxIter, tol)`: Eigenvalue near shift
- `computeEigenvalues(matrix, maxIter, tol)`: All eigenvalues

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

### Computational Topology *(v0.16.0)*

Compute homology groups, persistent homology, and analyze topological features of data:

```typescript
import init, {
  WasmSimplex,
  WasmSimplicialComplex,
  WasmFiltration,
  WasmPersistentHomology,
  ripsFromDistances,
  findCriticalPoints2D,
  WasmMorseComplex
} from '@justinelliottcobb/amari-wasm';

async function topologyDemo() {
  await init();

  // ========================================
  // Simplicial Complexes and Homology
  // ========================================

  // Create a triangle (2-simplex with all faces)
  const complex = new WasmSimplicialComplex();
  complex.addSimplex([0, 1, 2]); // Triangle

  // Closure property: edges and vertices automatically added
  console.log(`Vertices: ${complex.vertexCount()}`);   // 3
  console.log(`Edges: ${complex.edgeCount()}`);        // 3
  console.log(`Triangles: ${complex.simplexCount(2)}`); // 1

  // Compute Betti numbers (topological invariants)
  const betti = complex.bettiNumbers();
  console.log(`β₀ = ${betti[0]}`); // 1 (one connected component)
  console.log(`β₁ = ${betti[1]}`); // 0 (no holes - filled triangle)

  // Euler characteristic: χ = V - E + F
  console.log(`χ = ${complex.eulerCharacteristic()}`); // 1

  // ========================================
  // Circle (unfilled triangle boundary)
  // ========================================

  const circle = new WasmSimplicialComplex();
  circle.addSimplex([0, 1]); // Edge 0-1
  circle.addSimplex([1, 2]); // Edge 1-2
  circle.addSimplex([2, 0]); // Edge 2-0

  const circleBetti = circle.bettiNumbers();
  console.log(`Circle β₀ = ${circleBetti[0]}`); // 1 (connected)
  console.log(`Circle β₁ = ${circleBetti[1]}`); // 1 (one hole!)

  // ========================================
  // Persistent Homology
  // ========================================

  // Build a filtration (simplices appearing over time)
  const filt = new WasmFiltration();
  filt.add(0.0, [0]);       // Point 0 at t=0
  filt.add(0.0, [1]);       // Point 1 at t=0
  filt.add(0.0, [2]);       // Point 2 at t=0
  filt.add(1.0, [0, 1]);    // Edge 0-1 at t=1
  filt.add(2.0, [1, 2]);    // Edge 1-2 at t=2
  filt.add(3.0, [0, 2]);    // Edge 0-2 at t=3 (creates loop)

  // Compute persistent homology
  const ph = WasmPersistentHomology.compute(filt);

  // Get persistence diagram as [dim, birth, death] triples
  const diagram = ph.getDiagram();
  for (let i = 0; i < diagram.length; i += 3) {
    const dim = diagram[i];
    const birth = diagram[i + 1];
    const death = diagram[i + 2];
    console.log(`H${dim}: born at ${birth}, dies at ${death === Infinity ? '∞' : death}`);
  }

  // Betti numbers at different times
  console.log(`Betti at t=0.5: ${ph.bettiAt(0.5)}`); // [3, 0] - 3 components
  console.log(`Betti at t=3.5: ${ph.bettiAt(3.5)}`); // [1, 1] - 1 component, 1 loop

  // ========================================
  // Vietoris-Rips Filtration from Point Cloud
  // ========================================

  // 3 points forming an equilateral triangle
  // Distances: d(0,1)=1, d(1,2)=1, d(0,2)=1
  const numPoints = 3;
  const maxDim = 2;
  // Upper triangular pairwise distances: [d(0,1), d(0,2), d(1,2)]
  const distances = [1.0, 1.0, 1.0];

  const ripsFilt = ripsFromDistances(numPoints, maxDim, distances);
  const ripsPH = WasmPersistentHomology.compute(ripsFilt);
  console.log(`Rips intervals in H0: ${ripsPH.intervalCount(0)}`);
  console.log(`Rips intervals in H1: ${ripsPH.intervalCount(1)}`);

  // ========================================
  // Morse Theory (Critical Points)
  // ========================================

  // Precompute f(x,y) = x² + y² on a grid
  const resolution = 20;
  const values = [];
  for (let i = 0; i <= resolution; i++) {
    const x = -1.0 + (2.0 * i) / resolution;
    for (let j = 0; j <= resolution; j++) {
      const y = -1.0 + (2.0 * j) / resolution;
      values.push(x * x + y * y);
    }
  }

  const criticalPoints = findCriticalPoints2D(
    resolution,
    -1.0, 1.0, // x range
    -1.0, 1.0, // y range
    0.1,       // tolerance
    values
  );

  console.log(`Found ${criticalPoints.length} critical points`);
  for (const cp of criticalPoints) {
    console.log(`  ${cp.criticalType} at (${cp.position[0].toFixed(2)}, ${cp.position[1].toFixed(2)}), value=${cp.value.toFixed(2)}`);
  }

  // Morse complex analysis
  const morse = new WasmMorseComplex(criticalPoints);
  const counts = morse.countsByIndex();
  console.log(`Critical points by index: ${counts}`);

  // Verify Morse inequalities: c_k >= β_k
  const complexBetti = complex.bettiNumbers();
  console.log(`Weak Morse inequalities hold: ${morse.checkWeakMorseInequalities(complexBetti)}`);

  // Clean up WASM memory
  complex.free();
  circle.free();
  filt.free();
  ph.free();
  ripsFilt.free();
  ripsPH.free();
  morse.free();
  criticalPoints.forEach(cp => cp.free());
}

topologyDemo();
```

### Dynamical Systems *(v0.18.0)*

Analyze chaotic systems, compute bifurcation diagrams, and explore phase space:

```typescript
import init, {
  WasmLorenzSystem,
  WasmVanDerPolOscillator,
  WasmDuffingOscillator,
  WasmRungeKutta4,
  WasmLyapunovSpectrum,
  WasmBifurcationDiagram,
  WasmPhasePortrait,
  WasmStabilityAnalysis,
  computeLyapunovExponents,
  findFixedPoints
} from '@justinelliottcobb/amari-wasm';

async function dynamicsDemo() {
  await init();

  // ========================================
  // Lorenz Attractor
  // ========================================

  // Create classic Lorenz system (sigma=10, rho=28, beta=8/3)
  const lorenz = WasmLorenzSystem.classic();
  console.log(`Lorenz parameters: σ=${lorenz.sigma}, ρ=${lorenz.rho}, β=${lorenz.beta}`);

  // Create RK4 solver
  const solver = new WasmRungeKutta4();

  // Integrate trajectory from initial condition
  const initial = [1.0, 1.0, 1.0];
  const trajectory = solver.solve(lorenz, initial, 0.0, 50.0, 5000);

  console.log(`Trajectory has ${trajectory.length} points`);

  // Get final state
  const finalState = trajectory[trajectory.length - 1];
  console.log(`Final state: (${finalState[0].toFixed(3)}, ${finalState[1].toFixed(3)}, ${finalState[2].toFixed(3)})`);

  // ========================================
  // Van der Pol Limit Cycle
  // ========================================

  // Create Van der Pol oscillator with mu = 1.0
  const vdp = WasmVanDerPolOscillator.new(1.0);

  // Small initial displacement
  const vdpTrajectory = solver.solve(vdp, [0.1, 0.0], 0.0, 50.0, 5000);

  // Check limit cycle amplitude
  const vdpFinal = vdpTrajectory[vdpTrajectory.length - 1];
  console.log(`Van der Pol final amplitude: ${Math.abs(vdpFinal[0]).toFixed(3)}`);

  // ========================================
  // Lyapunov Exponents
  // ========================================

  // Compute Lyapunov spectrum for Lorenz system
  const lyapunov = computeLyapunovExponents(lorenz, initial, 10000, 0.01);

  console.log(`Lyapunov exponents: [${lyapunov.exponents.map(e => e.toFixed(4)).join(', ')}]`);
  console.log(`Sum: ${lyapunov.sum().toFixed(4)} (negative = dissipative)`);

  if (lyapunov.exponents[0] > 0) {
    console.log('System is chaotic!');
  }

  // Kaplan-Yorke dimension
  console.log(`Kaplan-Yorke dimension: ${lyapunov.kaplanYorkeDimension().toFixed(3)}`);

  // ========================================
  // Bifurcation Diagram
  // ========================================

  // Compute bifurcation diagram for logistic map
  const bifurcation = WasmBifurcationDiagram.compute(
    'logistic',
    2.5,   // r_min
    4.0,   // r_max
    1000,  // num_parameters
    500,   // transient iterations
    100    // sample points per parameter
  );

  console.log(`Bifurcation diagram: ${bifurcation.parameterCount()} parameter values`);

  // Get attractor points at specific parameter
  const attractorAt3_5 = bifurcation.attractorPoints(3.5);
  console.log(`Attractor at r=3.5: ${attractorAt3_5.length} points`);

  // ========================================
  // Stability Analysis
  // ========================================

  // Find fixed points of Van der Pol oscillator
  const fixedPoints = findFixedPoints(vdp, [[0.0, 0.0]], 1e-10);

  for (const fp of fixedPoints) {
    console.log(`Fixed point: (${fp.point[0].toFixed(6)}, ${fp.point[1].toFixed(6)})`);

    // Analyze stability
    const stability = WasmStabilityAnalysis.analyze(vdp, fp.point);
    console.log(`  Stability: ${stability.stabilityType}`);
    console.log(`  Eigenvalues: ${stability.eigenvalues.map(e => `${e.real.toFixed(4)}+${e.imag.toFixed(4)}i`).join(', ')}`);
  }

  // ========================================
  // Phase Portrait
  // ========================================

  // Generate phase portrait for Duffing oscillator
  const duffing = WasmDuffingOscillator.new(1.0, -1.0, 0.2, 0.3, 1.2);
  const portrait = WasmPhasePortrait.generate(
    duffing,
    [-2.0, 2.0],  // x range
    [-2.0, 2.0],  // y range
    20,           // grid resolution
    5.0,          // integration time
    0.01          // dt
  );

  console.log(`Phase portrait: ${portrait.trajectoryCount()} trajectories`);

  // Get nullclines
  const nullclines = portrait.nullclines();
  console.log(`x-nullcline: ${nullclines.x.length} points`);
  console.log(`y-nullcline: ${nullclines.y.length} points`);

  // Clean up WASM memory
  lorenz.free();
  vdp.free();
  duffing.free();
  solver.free();
  bifurcation.free();
  portrait.free();
  fixedPoints.forEach(fp => fp.free());
}

dynamicsDemo();
```

#### Dynamics API

**WasmLorenzSystem:**
- `classic()`: Create with σ=10, ρ=28, β=8/3
- `new(sigma, rho, beta)`: Create with custom parameters
- `sigma`, `rho`, `beta`: Parameter getters
- `vectorField(state)`: Evaluate dx/dt at state

**WasmVanDerPolOscillator:**
- `new(mu)`: Create with damping parameter μ
- `mu`: Parameter getter
- `vectorField(state)`: Evaluate dx/dt at state

**WasmDuffingOscillator:**
- `new(alpha, beta, delta, gamma, omega)`: Create driven Duffing oscillator
- `vectorField(state, t)`: Evaluate dx/dt at state and time t

**WasmRosslerSystem:**
- `new(a, b, c)`: Create Rossler attractor
- `classic()`: Create with a=0.2, b=0.2, c=5.7

**WasmHenonMap:**
- `new(a, b)`: Create Henon map
- `classic()`: Create with a=1.4, b=0.3
- `iterate(state)`: Apply one map iteration

**WasmRungeKutta4:**
- `new()`: Create RK4 solver
- `solve(system, initial, t0, t1, steps)`: Integrate trajectory
- `step(system, state, t, dt)`: Single integration step

**WasmAdaptiveSolver:**
- `rkf45()`: Create RKF45 adaptive solver
- `dormandPrince()`: Create Dormand-Prince solver
- `solve(system, initial, t0, t1, tolerance)`: Adaptive integration

**Lyapunov Functions:**
- `computeLyapunovExponents(system, initial, steps, dt)`: Compute spectrum
- Returns: `{ exponents, sum(), kaplanYorkeDimension(), isChaotic() }`

**WasmBifurcationDiagram:**
- `compute(systemType, paramMin, paramMax, numParams, transient, samples)`: Generate diagram
- `parameterCount()`: Number of parameter values
- `attractorPoints(param)`: Get attractor at specific parameter
- `branches()`: Get all (parameter, points) pairs

**WasmStabilityAnalysis:**
- `analyze(system, point)`: Analyze stability at point
- `stabilityType`: 'stable_node', 'stable_spiral', 'unstable_node', 'unstable_spiral', 'saddle', 'center'
- `eigenvalues`: Array of {real, imag} pairs
- `isStable()`: True if asymptotically stable

**findFixedPoints:**
- `findFixedPoints(system, initialGuesses, tolerance)`: Find fixed points via Newton's method
- Returns array of `{ point, converged, iterations }`

**WasmPhasePortrait:**
- `generate(system, xRange, yRange, resolution, tMax, dt)`: Generate portrait
- `trajectoryCount()`: Number of trajectories
- `trajectories()`: Get all trajectory arrays
- `nullclines()`: Get {x, y} nullcline point arrays

#### Topology API

**WasmSimplex:**
- `new(vertices)`: Create simplex from vertex indices
- `dimension()`: Get dimension (vertices - 1)
- `getVertices()`: Get sorted vertex array
- `orientation()`: Get orientation sign (+1 or -1)
- `containsVertex(v)`: Check if vertex is in simplex
- `faces(k)`: Get all k-dimensional faces
- `boundaryFaces()`: Get boundary faces with signs

**WasmSimplicialComplex:**
- `new()`: Create empty complex
- `addSimplex(vertices)`: Add simplex and all its faces
- `contains(vertices)`: Check if simplex exists
- `dimension()`: Get maximum simplex dimension
- `simplexCount(dim)`: Count simplices in dimension
- `totalSimplexCount()`: Total simplex count
- `vertexCount()`: Count 0-simplices
- `edgeCount()`: Count 1-simplices
- `bettiNumbers()`: Compute [β₀, β₁, β₂, ...]
- `eulerCharacteristic()`: Compute χ = Σ(-1)^k f_k
- `fVector()`: Get face counts [f₀, f₁, f₂, ...]
- `isConnected()`: Check if complex is connected
- `connectedComponents()`: Count components

**WasmFiltration:**
- `new()`: Create empty filtration
- `add(time, vertices)`: Add simplex at filtration time
- `isEmpty()`: Check if filtration is empty
- `complexAt(time)`: Get complex at given time
- `bettiAt(time)`: Get Betti numbers at time

**WasmPersistentHomology:**
- `compute(filtration)`: Compute persistent homology
- `getDiagram()`: Get [dim, birth, death, ...] triples
- `bettiAt(time)`: Get Betti numbers at time
- `intervalCount(dim)`: Count intervals in dimension

**Standalone Functions:**
- `ripsFromDistances(numPoints, maxDim, distances)`: Create Rips filtration
- `findCriticalPoints2D(resolution, xMin, xMax, yMin, yMax, tolerance, values)`: Find critical points

**WasmMorseComplex:**
- `new(criticalPoints)`: Create from critical points
- `countsByIndex()`: Get counts by Morse index
- `checkWeakMorseInequalities(betti)`: Verify c_k >= β_k

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
- **Topological Data Analysis**: Persistent homology for shape and feature detection
- **Computational Biology**: Protein structure analysis via simplicial complexes
- **Sensor Networks**: Coverage analysis using homology
- **Chaos Theory**: Lorenz attractors, bifurcation diagrams, Lyapunov exponents
- **Control Systems**: Stability analysis and phase portraits for dynamical systems
- **Climate Modeling**: Sensitivity analysis via Lyapunov spectrum computation
- **Algebraic Geometry**: Rational curve counting, Schubert calculus, Gromov-Witten invariants
- **Combinatorics**: Matroid operations, Littlewood-Richardson coefficients
- **Access Control**: Geometric namespace/capability systems for secure multi-agent coordination

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

### Functional Analysis Operations

- `WasmHilbertSpace.new()`: Create Hilbert space Cl(2,0,0)
- `WasmHilbertSpace.innerProduct(x, y)`: Compute inner product
- `WasmHilbertSpace.norm(x)`: Compute norm
- `WasmHilbertSpace.project(x, y)`: Orthogonal projection
- `WasmMatrixOperator.new(entries)`: Create matrix operator
- `WasmMatrixOperator.apply(x)`: Apply operator to vector
- `WasmMatrixOperator.operatorNorm()`: Compute operator norm
- `WasmSpectralDecomposition.compute(matrix, maxIter, tol)`: Eigenvalue decomposition
- `WasmSpectralDecomposition.eigenvalues()`: Get eigenvalues
- `WasmSpectralDecomposition.applyFunction(f, x)`: Functional calculus
- `WasmSobolevSpace.new(order, lower, upper)`: Create Sobolev space
- `WasmSobolevSpace.h1Norm(f, df)`: Compute H^1 norm
- `powerMethod(matrix, initial, maxIter, tol)`: Dominant eigenvalue
- `computeEigenvalues(matrix, maxIter, tol)`: All eigenvalues

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

### Topology Operations

- `WasmSimplex.new(vertices)`: Create simplex from vertex array
- `WasmSimplex.dimension()`: Get simplex dimension
- `WasmSimplex.faces(k)`: Get k-dimensional faces
- `WasmSimplicialComplex.new()`: Create empty complex
- `WasmSimplicialComplex.addSimplex(vertices)`: Add simplex with closure
- `WasmSimplicialComplex.bettiNumbers()`: Compute Betti numbers
- `WasmSimplicialComplex.eulerCharacteristic()`: Compute Euler characteristic
- `WasmFiltration.add(time, vertices)`: Add simplex at filtration time
- `WasmPersistentHomology.compute(filtration)`: Compute persistence
- `WasmPersistentHomology.getDiagram()`: Get persistence diagram
- `ripsFromDistances(n, dim, distances)`: Build Rips filtration
- `findCriticalPoints2D(...)`: Find Morse critical points

### Dynamics Operations

- `WasmLorenzSystem.classic()`: Create classic Lorenz attractor
- `WasmVanDerPolOscillator.new(mu)`: Create Van der Pol oscillator
- `WasmDuffingOscillator.new(alpha, beta, delta, gamma, omega)`: Create Duffing oscillator
- `WasmRosslerSystem.classic()`: Create Rossler attractor
- `WasmHenonMap.classic()`: Create Henon map
- `WasmRungeKutta4.solve(system, initial, t0, t1, steps)`: Integrate trajectory
- `WasmAdaptiveSolver.rkf45()`: Create adaptive RKF45 solver
- `computeLyapunovExponents(system, initial, steps, dt)`: Compute Lyapunov spectrum
- `WasmBifurcationDiagram.compute(type, paramMin, paramMax, n, transient, samples)`: Generate bifurcation diagram
- `WasmStabilityAnalysis.analyze(system, point)`: Analyze fixed point stability
- `findFixedPoints(system, guesses, tolerance)`: Find fixed points
- `WasmPhasePortrait.generate(system, xRange, yRange, res, tMax, dt)`: Generate phase portrait

### Enumerative Geometry Operations

- `WasmWDVVEngine.new()`: Create WDVV engine for rational curve counting
- `WasmWDVVEngine.rationalCurveCount(degree)`: Compute N_d (rational curves through 3d-1 points)
- `WasmWDVVEngine.requiredPointCount(degree, genus)`: Required point count (3d+g-1)
- `WasmWDVVEngine.getTable()`: Get table of computed curve counts
- `WasmWDVVEngine.p1xp1Count(a, b)`: Rational curves on P^1 x P^1 of bidegree (a,b)
- `WasmWDVVEngine.p3Count(degree)`: Rational curves in P^3
- `WasmEquivariantLocalizer.new(k, n)`: Create localizer for Gr(k,n)
- `WasmEquivariantLocalizer.fixedPointCount()`: Count T-fixed points (= C(n,k))
- `WasmEquivariantLocalizer.localizedIntersection(classes)`: Intersection via localization
- `WasmMatroid.uniform(k, n)`: Create uniform matroid U_{k,n}
- `WasmMatroid.getRank()`: Get matroid rank
- `WasmMatroid.getNumBases()`: Get number of bases
- `WasmMatroid.dual()`: Compute dual matroid
- `WasmMatroid.deleteElement(e)`: Delete element
- `WasmMatroid.contractElement(e)`: Contract element
- `WasmCSMClass.ofSchubertCell(partition, k, n)`: CSM class of Schubert cell
- `WasmCSMClass.ofSchubertVariety(partition, k, n)`: CSM class of Schubert variety
- `WasmCSMClass.eulerCharacteristic()`: Get Euler characteristic
- `WasmStabilityCondition.new(k, n, trust)`: Create stability condition
- `WasmStabilityCondition.phase(class)`: Compute phase of a class
- `WasmStabilityCondition.stableCount(namespace)`: Count stable objects
- `WasmWallCrossingEngine.new(k, n)`: Create wall-crossing engine
- `WasmWallCrossingEngine.computeWalls(namespace)`: Find walls
- `WasmWallCrossingEngine.stableCountAt(namespace, trust)`: Stable count at trust level
- `WasmWallCrossingEngine.phaseDiagram(namespace)`: Generate phase diagram
- `WasmComposableNamespace.new(namespace)`: Create composable namespace
- `WasmComposableNamespace.markOutput(capId)`: Mark output interface
- `WasmComposableNamespace.markInput(capId)`: Mark input interface
- `composeNamespaces(outer, inner)`: Compose along matching interfaces
- `compositionMultiplicity(outer, inner)`: Intersection number of interfaces
- `interfacesCompatible(outer, inner)`: Check interface compatibility

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
