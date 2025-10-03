# Amari v0.4.0

**Unified Mathematical Computing Platform for JavaScript/TypeScript**

A comprehensive mathematical computing library that brings five advanced algebraic systems to JavaScript/TypeScript through WebAssembly: geometric algebra, tropical algebra, automatic differentiation, fusion systems, and information geometry.

[![Rust](https://img.shields.io/badge/Rust-1.75+-orange.svg)](https://www.rust-lang.org/)
[![WebAssembly](https://img.shields.io/badge/WebAssembly-Ready-blue.svg)](https://webassembly.org/)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.0+-blue.svg)](https://www.typescriptlang.org/)
[![License](https://img.shields.io/badge/License-MIT%20OR%20Apache--2.0-green.svg)](LICENSE)

## Features

- **Geometric Algebra (Clifford Algebra)**: Multivectors, rotors, and geometric products for 3D rotations and spatial transformations
- **Tropical Algebra**: Max-plus semiring operations for optimization and neural network applications
- **Automatic Differentiation**: Forward-mode AD with dual numbers for exact derivatives
- **Fusion Systems**: Tropical-dual-Clifford fusion combining three algebraic systems
- **Information Geometry**: Statistical manifolds, KL/JS divergences, and Fisher information
- **WebAssembly**: High-performance WASM bindings for JavaScript/TypeScript
- **Pure Rust Implementation**: Memory-safe, high-performance core with WASM bindings
- **TypeScript Support**: Full TypeScript definitions included
- **Real-World Applications**: Physics simulations, ML optimization, statistical analysis

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
import init, { WasmMultivector, WasmTropicalNumber, WasmDualNumber } from '@justinelliottcobb/amari-wasm';

async function main() {
  // Initialize the WASM module
  await init();

  // Geometric Algebra: Create and rotate vectors
  const e1 = WasmMultivector.basisVector(0);
  const e2 = WasmMultivector.basisVector(1);
  const bivector = e1.geometricProduct(e2);
  console.log('Geometric product:', bivector.toString());

  // Tropical Algebra: Neural network operations
  const trop1 = new WasmTropicalNumber(3.0);
  const trop2 = new WasmTropicalNumber(5.0);
  const sum = trop1.tropicalAdd(trop2); // max(3, 5) = 5
  const product = trop1.tropicalMul(trop2); // 3 + 5 = 8
  console.log('Tropical operations:', sum.getValue(), product.getValue());

  // Automatic Differentiation: Compute derivatives
  const x = new WasmDualNumber(2.0, 1.0);
  const xSquared = x.mul(x); // f(x) = x², f'(x) = 2x
  console.log('f(2) =', xSquared.getReal(), "f'(2) =", xSquared.getDual());

  // Clean up WASM memory
  e1.free(); e2.free(); bivector.free();
  trop1.free(); trop2.free(); sum.free(); product.free();
  x.free(); xSquared.free();
}

main();
```

## Use Cases

- **Computer Graphics**: 3D rotations and transformations using rotors
- **Physics Simulations**: Geometric algebra for electromagnetic fields
- **Machine Learning**: Tropical neural networks and automatic differentiation
- **Optimization**: Tropical algebra for shortest path and scheduling problems
- **Scientific Computing**: High-performance mathematical operations
- **Game Development**: Efficient spatial transformations and physics

## Architecture

### Crates

- `amari-core`: Core Clifford algebra types and CPU implementations
- `amari-tropical`: Tropical (max-plus) algebra for neural networks
- `amari-dual`: Dual numbers for automatic differentiation
- `amari-fusion`: Unified Tropical-Dual-Clifford system
- `amari-info-geom`: Information geometry and statistical manifolds
- `amari-wasm`: WASM bindings for TypeScript/JavaScript (**New in v0.4.0**)
- `amari-gpu`: Optional GPU acceleration via WebGPU/wgpu
- `amari-automata`: Cellular automata with geometric algebra
- `amari-enumerative`: Enumerative geometry and algebraic curves

### Key Types

```rust
// Multivector in Clifford algebra Cl(P,Q,R)
Multivector<const P: usize, const Q: usize, const R: usize>

// Tropical-Dual-Clifford unified system
TropicalDualClifford<T: Float, const DIM: usize>

// Dual numbers for automatic differentiation
DualNumber<T: Float>

// Tropical numbers for max-plus algebra
TropicalNumber<T: Float>

// Common algebras
type Cl3 = Multivector<3, 0, 0>;  // 3D Euclidean
type Spacetime = Multivector<1, 3, 0>;  // Minkowski spacetime
```

## Quick Start

### Tropical-Dual-Clifford System

```rust
use amari_fusion::TropicalDualClifford;
use amari_dual::DualNumber;
use amari_tropical::TropicalNumber;

// Create from logits (common in ML applications)
let logits = vec![1.5, 2.0, 0.8, 1.2];
let tdc = TropicalDualClifford::<f64, 4>::from_logits(&logits);

// Evaluate using all three algebras simultaneously
let other = TropicalDualClifford::from_logits(&[2.0, 1.5, 1.0, 0.9]);
let evaluation = tdc.evaluate(&other);

// Extract features from each algebra
let tropical_features = tdc.extract_tropical_features(); // Fast path-finding
let dual_features = tdc.extract_dual_features();         // Automatic gradients
let clifford_geom = tdc.clifford;                        // Geometric relationships

// Perform sensitivity analysis
let sensitivity = tdc.sensitivity_analysis();
let most_sensitive = sensitivity.most_sensitive(2);

println!("Combined score: {}", evaluation.combined_score);
println!("Most sensitive components: {:?}", most_sensitive);
```

### Rust

```rust
use amari_core::{Multivector, basis::Basis, rotor::Rotor};

// 3D Euclidean Clifford algebra
type Cl3 = Multivector<3, 0, 0>;

// Create basis vectors
let e1: Cl3 = Basis::e1();
let e2: Cl3 = Basis::e2();

// Geometric product: e1 * e2 = e1 ∧ e2 (bivector)
let e12 = e1.geometric_product(&e2);

// Create rotor for 90° rotation in xy-plane
let rotor = Rotor::from_bivector(&e12, std::f64::consts::PI / 2.0);

// Apply rotation: e1 → e2
let rotated = rotor.apply(&e1);
```

### TypeScript/JavaScript

```typescript
import { initAmari, GA, Rotor } from 'amari';

await initAmari();

// Create basis vectors
const e1 = GA.e1();
const e2 = GA.e2();

// Geometric product
const e12 = e1.geometricProduct(e2);

// Create and apply rotor
const rotor = Rotor.fromBivector(e12, Math.PI / 2);
const rotated = rotor.apply(e1); // e1 → e2
```

## Building

### Prerequisites

- Rust 1.75+ with `cargo`
- Node.js 16+ (for TypeScript bindings)
- `wasm-pack` (installed automatically by build script)

### Build Everything

```bash
./build.sh
```

### Build Options

```bash
./build.sh --clean      # Clean all artifacts
./build.sh --bench      # Run benchmarks
./build.sh --examples   # Run examples
./build.sh --help       # Show options
```

### Manual Build

```bash
# Rust workspace
cargo build --workspace --release

# WASM package
cd amari-wasm && wasm-pack build --target web

# TypeScript
cd typescript && npm install && npm run build
```

## Performance

The library is optimized for high-performance applications:

- SIMD: Vectorized operations where supported
- Cache Alignment: 64-byte aligned data structures
- Const Generics: Zero-cost abstractions for dimensions
- GPU Fallback: Automatic CPU/GPU dispatch based on workload size
- Batch Operations: Efficient batch processing for large datasets

### Benchmarks

Run benchmarks to see performance on your system:

```bash
./build.sh --bench
```

## Mathematical Foundation

### Tropical-Dual-Clifford System

The revolutionary fusion of three algebraic systems:

#### Tropical Algebra (Max-Plus)
```
a ⊕ b = max(a, b)    // Tropical addition
a ⊙ b = a + b        // Tropical multiplication
```
- **Applications**: Path optimization, sequence decoding, dynamic programming
- **Benefits**: Converts exponential operations to linear max operations

#### Dual Numbers
```
a + εb where ε² = 0
(a + εb) + (c + εd) = (a + c) + ε(b + d)
(a + εb) × (c + εd) = ac + ε(ad + bc)
```
- **Applications**: Automatic differentiation, gradient computation
- **Benefits**: Exact derivatives without finite differences or computational graphs

#### Clifford Algebra
```
ab = a·b + a∧b      // Geometric product
```
- **Applications**: Rotations, reflections, geometric transformations
- **Benefits**: Unified treatment of scalars, vectors, bivectors, trivectors

### Unified TDC Operations

The fusion system enables simultaneous computation across all three algebras:

1. **Tropical Phase**: Fast approximation using max-plus operations
2. **Dual Phase**: Exact computation with automatic gradients
3. **Clifford Phase**: Geometric refinement and spatial reasoning

### Geometric Product

The fundamental operation combining inner and outer products:

```
ab = a·b + a∧b
```

### Clifford Algebra Cl(P,Q,R)

- **P**: Basis vectors with e²ᵢ = +1
- **Q**: Basis vectors with e²ᵢ = -1  
- **R**: Basis vectors with e²ᵢ = 0

### Information Geometry

- **Fisher Information Metric**: Riemannian metric on statistical manifolds
- **α-Connections**: Generalized connections parameterized by α ∈ [-1,1]
- **Dually Flat Manifolds**: Manifolds with e-connection (α=+1) and m-connection (α=-1)
- **Bregman Divergences**: Information-geometric divergences
- **Amari-Chentsov Tensor**: Fundamental tensor structure

## Examples

### Tropical-Dual-Clifford System

```rust
use amari_fusion::{TropicalDualClifford, optimizer::TDCOptimizer};

// Optimization using all three algebras
let initial_params = vec![0.1, 0.5, -0.2, 0.8];
let tdc = TropicalDualClifford::<f64, 4>::from_logits(&initial_params);

let optimizer = TDCOptimizer::new()
    .with_tropical_warmup(5)     // Fast tropical approximation
    .with_dual_refinement(10)    // Exact dual gradients
    .with_clifford_projection(); // Geometric constraints

let result = optimizer.optimize(&tdc, &target_function)?;
```

### Automatic Differentiation

```rust
use amari_dual::{DualNumber, functions::softmax};

// Forward-mode autodiff
let inputs: Vec<DualNumber<f64>> = vec![
    DualNumber::variable(1.0),
    DualNumber::variable(2.0),
    DualNumber::variable(0.5),
];

let output = softmax(&inputs);
// output[i].real contains the value
// output[i].dual contains the gradient
```

### Tropical Sequence Decoding

```rust
use amari_tropical::viterbi::ViterbiDecoder;

// Efficient Viterbi algorithm using tropical algebra
let transitions = create_transition_matrix();
let emissions = create_emission_matrix();
let observations = vec![0, 1, 2, 1, 0];

let decoder = ViterbiDecoder::new(&transitions, &emissions);
let best_path = decoder.decode(&observations);
```

### 3D Rotations

```rust
cargo run --example basic
```

See `amari-core/examples/basic.rs` for a comprehensive rotation example.

### Information Geometry

```rust
use amari_info_geom::{bregman_divergence, kl_divergence};

// Bregman divergence with quadratic potential
let phi = |mv: &Multivector<3,0,0>| mv.norm_squared();
let divergence = bregman_divergence(phi, &p, &q)?;
```

## Testing

```bash
# Run all tests
cargo test --workspace

# Run with features
cargo test --workspace --features "parallel"

# Property-based tests
cargo test --workspace --features "proptest"
```

## Documentation

```bash
# Generate and open docs
cargo doc --workspace --open

# API documentation will be available at:
# target/doc/amari_core/index.html
```

## Use Cases

- Computer Graphics: Rotations, reflections, and transformations
- Robotics: Orientation representation and interpolation
- Physics: Spacetime calculations and electromagnetic field theory
- Machine Learning: Statistical manifold operations and natural gradients
- Computer Vision: Multi-view geometry and camera calibration
- Mathematical Optimization: Hybrid tropical-dual-Clifford optimization
- Sequence Analysis: Efficient decoding using tropical Viterbi
- Automatic Differentiation: Exact gradients for scientific computing

## Research Applications

- Information Geometry: Statistical manifold computations
- Geometric Deep Learning: Operations on non-Euclidean data
- Quantum Computing: Clifford group operations
- Crystallography: Symmetry group calculations
- Tropical Geometry: Max-plus linear algebra and optimization
- Computational Algebra: Multi-algebraic system integration
- Neural Architecture Search: Gradient-based optimization with geometric constraints

## Contributing

Contributions are welcome! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup

```bash
git clone https://github.com/your-username/amari.git
cd amari
./build.sh --examples
```

## License

This project is licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT License ([LICENSE-MIT](LICENSE-MIT))

at your option.

## Acknowledgments

- Inspired by the geometric algebra community and research in Information Geometry
- Built with modern Rust performance idioms and WebAssembly best practices
- Named after Shun-ichi Amari's contributions to Information Geometry

## Support

- **Issues**: [GitHub Issues](https://github.com/your-username/amari/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-username/amari/discussions)
- **Documentation**: [API Docs](https://docs.rs/amari)

---

*"Geometry is the art of correct reasoning from incorrectly drawn figures."* - Henri Poincaré