# Amari v0.14.0

**Comprehensive Mathematical Computing Platform with Geometric Algebra, Differential Calculus, Measure Theory, Probability Theory, and Vector Symbolic Architectures**

A unified mathematical computing library featuring geometric algebra, differential calculus, measure theory, probability theory on geometric spaces, relativistic physics, tropical algebra, automatic differentiation, holographic associative memory (Vector Symbolic Architectures), and information geometry. The library provides multi-GPU infrastructure with intelligent workload distribution and complete WebAssembly support for browser deployment.

[![Rust](https://img.shields.io/badge/Rust-1.75+-orange.svg)](https://www.rust-lang.org/)
[![WebAssembly](https://img.shields.io/badge/WebAssembly-Ready-blue.svg)](https://webassembly.org/)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.0+-blue.svg)](https://www.typescriptlang.org/)
[![License](https://img.shields.io/badge/License-MIT%20OR%20Apache--2.0-green.svg)](LICENSE)

**[Live Examples Suite](https://amari-math.netlify.app)** | **[API Documentation](https://docs.rs/amari)**

## Features

### Core Mathematical Systems

- **Geometric Algebra (Clifford Algebra)**: Multivectors, rotors, and geometric products for 3D rotations and spatial transformations
- **Differential Calculus**: Unified geometric calculus with scalar/vector fields, gradients, divergence, curl, and Lie derivatives
- **Measure Theory**: Sigma-algebras, measurable functions, integration on geometric spaces, and probability measures
- **Probability Theory**: Distributions on multivector spaces, stochastic processes, MCMC sampling, and Bayesian inference
- **Vector Symbolic Architectures**: Holographic Reduced Representations (HRR), binding algebras, and associative memory
- **Relativistic Physics**: Complete spacetime algebra (Cl(1,3)) with Minkowski signature for relativistic calculations
- **Tropical Algebra**: Max-plus semiring operations for optimization and neural network applications
- **Automatic Differentiation**: Forward-mode AD with dual numbers for exact derivatives
- **Fusion Systems**: Tropical-dual-Clifford fusion combining three algebraic systems
- **Information Geometry**: Statistical manifolds, KL/JS divergences, and Fisher information
- **Optimization**: Gradient-based optimization with geometric constraints
- **Network Analysis**: Geometric network analysis and graph neural networks
- **Cellular Automata**: Geometric automata with configurable rules
- **Enumerative Geometry**: Algebraic curves and enumerative computations

### Multi-GPU Infrastructure

- **Multi-GPU Architecture**: Infrastructure supporting up to 8 GPUs with intelligent workload distribution
- **Advanced Load Balancing**: Five strategies including Balanced, CapabilityAware, MemoryAware, LatencyOptimized, and Adaptive
- **Performance Profiling**: Timeline analysis with microsecond precision and automatic bottleneck detection
- **Comprehensive Benchmarking**: Production-ready validation across all mathematical domains
- **Graceful Degradation**: Automatic fallback to single GPU or CPU when multi-GPU unavailable

### Platform Support

- **Native Rust**: High-performance execution with rug (GMP/MPFR) backend for high-precision arithmetic
- **WebAssembly**: Full-featured WASM bindings with dashu backend for browser compatibility
- **GPU Acceleration**: WebGPU support for large-scale parallel computations
- **TypeScript Support**: Complete TypeScript definitions included
- **Cross-Platform**: Linux, macOS, Windows, browsers, Node.js, and edge computing environments

## Installation

### Rust Crates

Add to your `Cargo.toml`:

```toml
[dependencies]
# Complete library with all features
amari = "0.14.0"

# Or individual crates:

# Core geometric algebra and mathematical foundations
amari-core = "0.14.0"

# Differential calculus with geometric algebra
amari-calculus = "0.14.0"

# Measure theory and integration
amari-measure = "0.14.0"

# Probability theory on geometric algebra spaces
amari-probabilistic = "0.14.0"

# Vector Symbolic Architectures and holographic memory
amari-holographic = "0.14.0"

# High-precision relativistic physics
amari-relativistic = { version = "0.14.0", features = ["high-precision"] }

# GPU acceleration
amari-gpu = "0.14.0"

# Optimization algorithms
amari-optimization = "0.14.0"

# Additional mathematical systems
amari-tropical = "0.14.0"
amari-dual = "0.14.0"
amari-info-geom = "0.14.0"
amari-automata = "0.14.0"
amari-fusion = "0.14.0"
amari-network = "0.14.0"
amari-enumerative = "0.14.0"
```

### JavaScript/TypeScript (WebAssembly)

```bash
npm install @justinelliottcobb/amari-wasm
```

Or with yarn:

```bash
yarn add @justinelliottcobb/amari-wasm
```

## Quick Start

### Rust: Geometric Algebra

```rust
use amari_core::{Multivector, basis::Basis, rotor::Rotor};

// 3D Euclidean Clifford algebra Cl(3,0,0)
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

### Rust: Differential Calculus

```rust
use amari_calculus::{ScalarField, VectorField, VectorDerivative};
use amari_core::Multivector;

// Define a scalar field f(x,y,z) = x² + y² + z²
let field = ScalarField::<3, 0, 0>::new(|pos| {
    pos[0] * pos[0] + pos[1] * pos[1] + pos[2] * pos[2]
});

// Compute gradient at a point
let point = [1.0, 2.0, 3.0];
let gradient = field.gradient(&point, 1e-6);

// Define a vector field
let vector_field = VectorField::<3, 0, 0>::new(|pos| {
    [pos[1] * pos[2], pos[0] * pos[2], pos[0] * pos[1]]
});

// Compute divergence and curl
let derivative = VectorDerivative::<3, 0, 0>::new(1e-6);
let div = derivative.divergence(&vector_field, &point);
let curl = derivative.curl(&vector_field, &point);
```

### Rust: Tropical Algebra

```rust
use amari_tropical::TropicalNumber;

// Create tropical numbers using the constructor
let a = TropicalNumber::new(3.0);
let b = TropicalNumber::new(5.0);

// Tropical operations
let sum = a.tropical_add(&b);     // max(3, 5) = 5
let product = a.tropical_mul(&b); // 3 + 5 = 8

// Access values
println!("Tropical sum: {}", sum.value());
println!("Tropical product: {}", product.value());

// Tropical identities
let zero = TropicalNumber::<f64>::tropical_zero(); // -∞ (additive identity)
let one = TropicalNumber::<f64>::tropical_one();   // 0 (multiplicative identity)
```

### Rust: Automatic Differentiation

```rust
use amari_dual::DualNumber;

// Create dual number for differentiation
// f(x) = x² at x = 3, with seed derivative 1
let x = DualNumber::new(3.0, 1.0);

// Compute f(x) = x²
let result = x * x;

// Extract value and derivative
println!("f(3) = {}", result.value());       // 9.0
println!("f'(3) = {}", result.derivative()); // 6.0

// For constants (derivative = 0)
let c = DualNumber::constant(2.0);
let scaled = x * c; // 2x, derivative = 2
```

### Rust: Tropical-Dual-Clifford Fusion

```rust
use amari_fusion::TropicalDualClifford;

// Create from logits (common in ML applications)
let logits = vec![1.5, 2.0, 0.8, 1.2];
let tdc = TropicalDualClifford::<f64, 4>::from_logits(&logits);

// Evaluate using all three algebras simultaneously
let other = TropicalDualClifford::from_logits(&[2.0, 1.5, 1.0, 0.9]);
let evaluation = tdc.evaluate(&other);

// Extract features from each algebra
let tropical_features = tdc.extract_tropical_features(); // Fast path-finding
let dual_features = tdc.extract_dual_features();         // Automatic gradients

// Perform sensitivity analysis
let sensitivity = tdc.sensitivity_analysis();
let most_sensitive = sensitivity.most_sensitive(2);

println!("Combined score: {}", evaluation.combined_score);
println!("Most sensitive components: {:?}", most_sensitive);
```

### Rust: Vector Symbolic Architectures (Holographic Memory)

```rust
use amari_holographic::{HolographicMemory, ProductCl3x32, BindingAlgebra};

// Create holographic memory with 256-dimensional vectors
let mut memory: HolographicMemory<ProductCl3x32> = HolographicMemory::new();

// Generate random keys and values
let key1 = ProductCl3x32::random();
let value1 = ProductCl3x32::random();

// Store key-value associations via superposition
memory.store(&key1, &value1);

let key2 = ProductCl3x32::random();
let value2 = ProductCl3x32::random();
memory.store(&key2, &value2);

// Retrieve with a query key
let retrieved = memory.retrieve(&key1);
println!("Similarity to original: {:.3}", retrieved.similarity(&value1));

// Binding operations for role-filler structures
let bound = key1.bind(&value1);        // key ⊛ value
let recovered = bound.unbind(&key1);    // Approximately recovers value1
println!("Recovery similarity: {:.3}", recovered.similarity(&value1));

// Resonator network for cleanup/factorization
use amari_holographic::Resonator;
let codebook = vec![value1.clone(), value2.clone()];
let resonator = Resonator::new(codebook.clone());
let noisy = retrieved.clone(); // Noisy retrieval
let cleaned = resonator.cleanup(&noisy, 10); // 10 iterations
```

### Rust: Probability on Geometric Algebra

```rust
use amari_probabilistic::distribution::{GaussianMultivector, Distribution, MultivectorDistribution};
use amari_probabilistic::stochastic::{GeometricBrownianMotion, StochasticProcess};

// Gaussian distribution on Cl(3,0,0) - 8-dimensional multivector space
let gaussian = GaussianMultivector::<3, 0, 0>::standard();

// Draw samples
let mut rng = rand::thread_rng();
let sample = gaussian.sample(&mut rng);
println!("Sample: {:?}", sample.to_vec());

// Evaluate log-probability
let log_p = gaussian.log_prob(&sample).unwrap();
println!("Log probability: {:.3}", log_p);

// Grade-concentrated distribution (e.g., only on bivectors)
let bivector_dist = GaussianMultivector::<3, 0, 0>::grade_concentrated(2, 1.0).unwrap();

// Geometric Brownian Motion on multivector space
let gbm = GeometricBrownianMotion::<3, 0, 0>::new(0.1, 0.2); // drift=0.1, volatility=0.2
let path = gbm.sample_path(0.0, 1.0, 100, &mut rng).unwrap();
println!("Path has {} points", path.len());

// MCMC sampling with Metropolis-Hastings
use amari_probabilistic::sampling::MetropolisHastings;
let sampler = MetropolisHastings::new(gaussian.clone(), 0.5); // step_size=0.5
let samples = sampler.sample_n(&mut rng, 1000);
println!("Drew {} MCMC samples", samples.len());
```

### JavaScript/TypeScript: Mathematical Computing

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
  const trop1 = WasmTropicalNumber.new(3.0);
  const trop2 = WasmTropicalNumber.new(5.0);
  const sum = trop1.tropicalAdd(trop2); // max(3, 5) = 5
  const product = trop1.tropicalMul(trop2); // 3 + 5 = 8
  console.log('Tropical operations:', sum.getValue(), product.getValue());

  // Automatic Differentiation: Compute derivatives
  const x = WasmDualNumber.new(2.0, 1.0);
  const xSquared = x.mul(x); // f(x) = x², f'(x) = 2x
  console.log('f(2) =', xSquared.getValue(), "f'(2) =", xSquared.getDerivative());

  // Clean up WASM memory
  e1.free(); e2.free(); bivector.free();
  trop1.free(); trop2.free(); sum.free(); product.free();
  x.free(); xSquared.free();
}

main();
```

## Architecture

### Crate Hierarchy

**Domain Crates** (provide mathematical APIs):
- `amari-core`: Core Clifford algebra types and CPU implementations
- `amari-measure`: Measure theory, sigma-algebras, and integration
- `amari-calculus`: Differential calculus with geometric algebra
- `amari-probabilistic`: Probability distributions on multivector spaces, stochastic processes, MCMC
- `amari-holographic`: Vector Symbolic Architectures (VSA), binding algebras, holographic memory
- `amari-tropical`: Tropical (max-plus) algebra for optimization
- `amari-dual`: Dual numbers for automatic differentiation
- `amari-fusion`: Unified Tropical-Dual-Clifford system
- `amari-info-geom`: Information geometry and statistical manifolds
- `amari-automata`: Cellular automata with geometric algebra
- `amari-network`: Graph neural networks and network analysis
- `amari-relativistic`: Spacetime algebra and relativistic physics
- `amari-enumerative`: Enumerative geometry and algebraic curves
- `amari-optimization`: Gradient-based optimization algorithms
- `amari-flynn`: Probabilistic verification contracts

**Integration Crates** (consume domain APIs):
- `amari-gpu`: Multi-GPU acceleration with WebGPU
- `amari-wasm`: WebAssembly bindings for TypeScript/JavaScript
- `amari`: Umbrella crate re-exporting all features

### Key Types

```rust
// Multivector in Clifford algebra Cl(P,Q,R)
// P: positive signature, Q: negative signature, R: zero signature
Multivector<const P: usize, const Q: usize, const R: usize>

// Tropical numbers (max-plus semiring)
TropicalNumber<T: Float>  // Use TropicalNumber::new(value)

// Dual numbers for automatic differentiation
DualNumber<T: Float>      // Use DualNumber::new(value, derivative)

// Multi-variable dual numbers
MultiDualNumber<T: Float> // Use MultiDualNumber::new(value, gradients)

// Tropical-Dual-Clifford unified system
TropicalDualClifford<T: Float, const DIM: usize>

// Common algebras
type Cl3 = Multivector<3, 0, 0>;      // 3D Euclidean
type Spacetime = Multivector<1, 3, 0>; // Minkowski spacetime
type PGA3D = Multivector<3, 0, 1>;    // Projective Geometric Algebra
```

## Mathematical Foundation

### Clifford Algebra Cl(P,Q,R)

- **P**: Basis vectors with e²ᵢ = +1 (positive signature)
- **Q**: Basis vectors with e²ᵢ = -1 (negative signature)
- **R**: Basis vectors with e²ᵢ = 0 (degenerate)

```
ab = a·b + a∧b      // Geometric product = inner + outer product
```

### Tropical Algebra (Max-Plus)

```
a ⊕ b = max(a, b)    // Tropical addition
a ⊙ b = a + b        // Tropical multiplication
```

Applications: Path optimization, sequence decoding, dynamic programming

### Dual Numbers

```
a + εb where ε² = 0
(a + εb) + (c + εd) = (a + c) + ε(b + d)
(a + εb) × (c + εd) = ac + ε(ad + bc)
```

Applications: Automatic differentiation, gradient computation

### Information Geometry

- **Fisher Information Metric**: Riemannian metric on statistical manifolds
- **α-Connections**: Generalized connections parameterized by α ∈ [-1,1]
- **Dually Flat Manifolds**: Manifolds with e-connection (α=+1) and m-connection (α=-1)
- **Bregman Divergences**: Information-geometric divergences

## v0.12.0 Breaking Changes

Version 0.12.0 introduced significant API improvements for better encapsulation:

### TropicalNumber

```rust
// Before (v0.11.x)
let a = TropicalNumber(3.0);
let value = a.0;
let sum = a.tropical_add(b);

// After (v0.12.0+)
let a = TropicalNumber::new(3.0);
let value = a.value();
let sum = a.tropical_add(&b);  // Now takes reference
```

### DualNumber

```rust
// Before (v0.11.x)
let x = DualNumber { real: 3.0, dual: 1.0 };
let value = x.real;

// After (v0.12.0+)
let x = DualNumber::new(3.0, 1.0);
let value = x.value();
let deriv = x.derivative();
```

See [MIGRATION_v0.12.0.md](MIGRATION_v0.12.0.md) for complete migration guide.

## Interactive Examples Suite

The **[Amari Examples Suite](https://amari-math.netlify.app)** provides comprehensive interactive documentation:

- **Live Visualizations**: 7 interactive visualizations demonstrating mathematical concepts
  - Multivector coefficient manipulation in Cl(3,0,0)
  - Tropical algebra operations with convergence animation
  - Dual numbers with real-time derivative curves
  - Rotor-based 3D rotations
  - Fisher information on probability simplex
  - MCMC sampling visualization
  - Interactive geometric networks

- **Comprehensive API Reference**: 77 classes with 300+ methods fully documented
  - Geometric Algebra (Multivector, Rotor, Bivector)
  - Tropical Algebra (TropicalNumber, TropicalMatrix)
  - Automatic Differentiation (DualNumber, MultiDualNumber)
  - Probability (GaussianMultivector, MCMC samplers)
  - And 12 more categories

- **Interactive Playground**: Write and run JavaScript code with live WASM execution

## GPU Module Status (v0.14.0)

| Module | Status | Feature Flag |
|--------|--------|--------------|
| Core GA | ✅ Enabled | default |
| Info Geometry | ✅ Enabled | default |
| Relativistic | ✅ Enabled | default |
| Network | ✅ Enabled | default |
| Measure | ✅ Enabled | `measure` |
| Calculus | ✅ Enabled | `calculus` |
| Dual | ✅ Enabled | `dual` |
| Enumerative | ✅ Enabled | `enumerative` |
| Automata | ✅ Enabled | `automata` |
| Fusion | ✅ Enabled | `fusion` |
| Holographic | ✅ Enabled | `holographic` |
| Probabilistic | ✅ Enabled | `probabilistic` |
| Tropical | ❌ Disabled | - |

Note: Tropical GPU module temporarily disabled due to Rust orphan impl rules. Use CPU implementations from domain crates.

## Building

### Prerequisites

- Rust 1.75+ with `cargo`
- Node.js 16+ (for TypeScript bindings)
- `wasm-pack` (for WASM builds)

### Build and Test

```bash
# Run all tests
cargo test --workspace

# Run with all features
cargo test --workspace --all-features

# Build documentation
cargo doc --workspace --open
```

### WebAssembly Build

```bash
cd amari-wasm
wasm-pack build --target web
```

## Performance

The library is optimized for high-performance applications:

- **SIMD**: Vectorized operations where supported
- **Cache Alignment**: 64-byte aligned data structures
- **Const Generics**: Zero-cost abstractions for dimensions
- **GPU Fallback**: Automatic CPU/GPU dispatch based on workload size
- **Batch Operations**: Efficient batch processing for large datasets

## Documentation

- **[Migration Guide](MIGRATION_v0.12.0.md)**: Migrating from v0.11.x to v0.12.0+
- **[Changelog](CHANGELOG.md)**: Version history and changes
- **[API Documentation](https://docs.rs/amari)**: Complete API reference

## Contributing

Contributions are welcome. For development setup:

```bash
git clone https://github.com/justinelliottcobb/Amari.git
cd Amari
cargo test --workspace
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

- Issues: [GitHub Issues](https://github.com/justinelliottcobb/Amari/issues)
- Discussions: [GitHub Discussions](https://github.com/justinelliottcobb/Amari/discussions)
- Documentation: [API Docs](https://docs.rs/amari)

---

*"Geometry is the art of correct reasoning from incorrectly drawn figures."* - Henri Poincaré
