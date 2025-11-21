# Amari v0.9.10

**Multi-GPU Mathematical Computing Platform with Intelligent Load Balancing**

A comprehensive mathematical computing library featuring geometric algebra, relativistic physics, tropical algebra, automatic differentiation, and information geometry. The library provides complete multi-GPU infrastructure with intelligent workload distribution, advanced profiling systems, and optimization algorithms across all mathematical domains.

[![Rust](https://img.shields.io/badge/Rust-1.75+-orange.svg)](https://www.rust-lang.org/)
[![WebAssembly](https://img.shields.io/badge/WebAssembly-Ready-blue.svg)](https://webassembly.org/)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.0+-blue.svg)](https://www.typescriptlang.org/)
[![License](https://img.shields.io/badge/License-MIT%20OR%20Apache--2.0-green.svg)](LICENSE)

## Features

### Core Mathematical Systems

- **Geometric Algebra (Clifford Algebra)**: Multivectors, rotors, and geometric products for 3D rotations and spatial transformations
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
- **Scaling Efficiency**: Realistic performance modeling with 90%/80%/70% efficiency for 2/4/8 GPUs
- **Graceful Degradation**: Automatic fallback to single GPU or CPU when multi-GPU unavailable

### High-Precision Arithmetic

- **Spacecraft Orbital Mechanics**: High-precision arithmetic for critical trajectory calculations with configurable tolerance
- **Geodesic Integration**: Velocity Verlet method for curved spacetime particle trajectories
- **Schwarzschild Metric**: Spherically symmetric gravitational fields for astrophysics applications
- **Phantom Types**: Compile-time verification of relativistic invariants and spacetime signatures

### Platform Support

- **Native Rust**: High-performance execution with rug (GMP/MPFR) backend for high-precision arithmetic
- **WebAssembly**: Full-featured WASM bindings with dashu backend for browser compatibility
- **Universal Precision**: Consistent API and mathematical accuracy across all platforms
- **GPU Acceleration**: WebGPU support for large-scale parallel computations
- **TypeScript Support**: Complete TypeScript definitions included
- **Deployment Freedom**: Pure Rust WASM builds deploy anywhere without system dependencies
- **Cross-Platform**: Linux, macOS, Windows, browsers, Node.js, and edge computing environments

## Installation

### Rust Crates

Add to your `Cargo.toml`:

```toml
[dependencies]
# Core geometric algebra and mathematical foundations
amari-core = "0.9.10"

# High-precision relativistic physics with multi-backend support
amari-relativistic = { version = "0.9.10", features = ["high-precision"] }

# For native applications (uses rug/GMP backend)
amari-relativistic = { version = "0.9.10", features = ["native-precision"] }

# For WebAssembly targets (uses dashu backend)
amari-relativistic = { version = "0.9.10", features = ["wasm-precision"] }

# Multi-GPU acceleration and intelligent load balancing
amari-gpu = "0.9.10"

# Optimization algorithms
amari-optimization = "0.9.10"

# Additional mathematical systems
amari-tropical = "0.9.10"
amari-dual = "0.9.10"
amari-info-geom = "0.9.10"
amari-automata = "0.9.10"
amari-fusion = "0.9.10"
amari-network = "0.9.10"
amari-enumerative = "0.9.10"
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

### Rust: Spacecraft Orbital Mechanics

```rust
use amari_relativistic::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create gravitational field from massive object (Earth)
    let earth = schwarzschild::SchwarzschildMetric::earth();
    let mut integrator = geodesic::GeodesicIntegrator::with_metric(Box::new(earth));

    // Create spacecraft at 400 km altitude
    let altitude = 400e3; // 400 km
    let earth_radius = 6.371e6; // Earth radius
    let position = Vector3::new(earth_radius + altitude, 0.0, 0.0);
    let orbital_velocity = Vector3::new(0.0, 7.67e3, 0.0); // ~7.67 km/s orbital velocity

    // Create spacecraft particle
    let mut spacecraft = particle::RelativisticParticle::new(
        position,
        orbital_velocity,
        0.0, // Uncharged
        1000.0, // 1000 kg spacecraft
        0.0, // No charge
    )?;

    // Propagate orbit for one period with high precision
    let orbital_period = 5580.0; // ~93 minutes
    let trajectory = particle::propagate_relativistic(
        &mut spacecraft,
        &mut integrator,
        orbital_period,
        60.0, // 1-minute time steps
    )?;

    println!("Spacecraft orbital trajectory computed with {} points", trajectory.len());
    println!("Final position: {:?}", spacecraft.position_3d());

    Ok(())
}
```

### Rust: Multi-GPU Load Balancing

```rust
use amari_gpu::{
    SharedGpuContext, IntelligentLoadBalancer, LoadBalancingStrategy,
    Workload, ComputeIntensity, BenchmarkRunner
};

async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize multi-GPU context
    let context = SharedGpuContext::with_multi_gpu().await?;
    println!("Initialized {} GPU devices", context.device_count().await);

    // Create intelligent load balancer
    let load_balancer = IntelligentLoadBalancer::new(LoadBalancingStrategy::CapabilityAware);

    // Define computational workload
    let workload = Workload {
        operation_type: "geometric_product".to_string(),
        data_size: 100000,
        memory_requirement_mb: 256.0,
        compute_intensity: ComputeIntensity::Heavy,
        parallelizable: true,
        synchronization_required: true,
    };

    // Distribute workload across available GPUs
    let assignments = load_balancer.distribute_workload(&workload).await?;
    println!("Workload distributed across {} devices", assignments.len());

    for assignment in &assignments {
        println!(
            "Device {}: {:.1}% workload, estimated completion: {:.2}ms",
            assignment.device_id.0,
            assignment.workload_fraction * 100.0,
            assignment.estimated_completion_ms
        );
    }

    // Run comprehensive benchmarks
    let benchmark_results = BenchmarkRunner::run_quick_validation().await?;
    println!("Benchmark completed: {} tests, average scaling efficiency: {:.2}%",
        benchmark_results.performance_summary.total_tests,
        benchmark_results.performance_summary.average_scaling_efficiency * 100.0
    );

    Ok(())
}
```

### Rust: Geometric Algebra

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

### Rust: Tropical-Dual-Clifford System

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

## Architecture

### Crates

- `amari-core`: Core Clifford algebra types and CPU implementations
- `amari-tropical`: Tropical (max-plus) algebra for neural networks
- `amari-dual`: Dual numbers for automatic differentiation
- `amari-fusion`: Unified Tropical-Dual-Clifford system
- `amari-info-geom`: Information geometry and statistical manifolds
- `amari-wasm`: WASM bindings for TypeScript/JavaScript
- `amari-gpu`: Multi-GPU acceleration with intelligent load balancing
- `amari-automata`: Cellular automata with geometric algebra
- `amari-network`: Graph neural networks and network analysis
- `amari-relativistic`: Spacetime algebra and relativistic physics
- `amari-enumerative`: Enumerative geometry and algebraic curves
- `amari-optimization`: Gradient-based optimization algorithms

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

## Multi-Backend Precision Architecture

### Automatic Backend Selection

Amari provides intelligent backend selection for high-precision arithmetic:

```rust
// Same API, different backends automatically selected:

// For native builds (optimal performance)
cargo build --features native-precision  // Uses rug (GMP/MPFR)

// For WASM builds (maximum compatibility)
cargo build --target wasm32-unknown-unknown --features wasm-precision  // Uses dashu

// Auto-selection (recommended)
cargo build --features high-precision  // Chooses best backend for target
```

### Backend Characteristics

| Backend | Platform | Performance | Dependencies | Use Case |
|---------|----------|-------------|--------------|----------|
| **rug** | Native | Ultimate | GMP/MPFR (C libraries) | High-performance computing, research |
| **dashu** | WASM | Excellent | Pure Rust | Web apps, edge computing, universality |

### Mathematical Consistency

Both backends provide:
- **Identical API**: Same function signatures across platforms
- **Numerical Accuracy**: Configurable precision with orbital-grade tolerance
- **Mathematical Correctness**: All relativistic calculations preserve physical invariants
- **Feature Parity**: Full support for spacecraft orbital mechanics in both environments

### WebAssembly Deployment Example

```rust
// Compile for WASM with high-precision arithmetic
#[cfg(target_arch = "wasm32")]
use amari_relativistic::precision::StandardFloat; // Uses dashu backend

// Native compilation automatically uses rug
#[cfg(not(target_arch = "wasm32"))]
use amari_relativistic::precision::StandardFloat; // Uses rug backend

// Same code works everywhere!
let spacecraft_trajectory = propagate_orbital_mechanics(
    initial_conditions,
    StandardFloat::orbital_tolerance(), // 1e-12 precision
)?;
```

## Mathematical Foundation

### Tropical-Dual-Clifford System

The fusion of three algebraic systems:

#### Tropical Algebra (Max-Plus)
```
a ⊕ b = max(a, b)    // Tropical addition
a ⊙ b = a + b        // Tropical multiplication
```
- Applications: Path optimization, sequence decoding, dynamic programming
- Benefits: Converts exponential operations to linear max operations

#### Dual Numbers
```
a + εb where ε² = 0
(a + εb) + (c + εd) = (a + c) + ε(b + d)
(a + εb) × (c + εd) = ac + ε(ad + bc)
```
- Applications: Automatic differentiation, gradient computation
- Benefits: Exact derivatives without finite differences or computational graphs

#### Clifford Algebra
```
ab = a·b + a∧b      // Geometric product
```
- Applications: Rotations, reflections, geometric transformations
- Benefits: Unified treatment of scalars, vectors, bivectors, trivectors

### Unified TDC Operations

The fusion system enables simultaneous computation across all three algebras:

1. **Tropical Phase**: Fast approximation using max-plus operations
2. **Dual Phase**: Exact computation with automatic gradients
3. **Clifford Phase**: Geometric refinement and spatial reasoning

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

## Use Cases

### Scientific Computing
- Computer Graphics: 3D rotations and transformations using rotors
- Physics Simulations: Geometric algebra for electromagnetic fields
- Relativistic Physics: Spacetime calculations and orbital mechanics
- Astrophysics: Schwarzschild metric and geodesic integration

### Machine Learning
- Neural Networks: Tropical algebra for efficient network operations
- Automatic Differentiation: Forward-mode AD with dual numbers
- Statistical Manifolds: Information geometry and natural gradients
- Geometric Deep Learning: Operations on non-Euclidean data

### Optimization
- Path Optimization: Tropical algebra for shortest path problems
- Gradient-Based Optimization: Hybrid tropical-dual-Clifford optimization
- Sequence Analysis: Efficient decoding using tropical Viterbi
- Mathematical Optimization: Multi-algebraic system integration

### Research Applications
- Information Geometry: Statistical manifold computations
- Quantum Computing: Clifford group operations
- Crystallography: Symmetry group calculations
- Tropical Geometry: Max-plus linear algebra and optimization
- Computational Algebra: Multi-algebraic system integration
- Neural Architecture Search: Gradient-based optimization with geometric constraints

## Examples

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

### Information Geometry

```rust
use amari_info_geom::{bregman_divergence, kl_divergence};

// Bregman divergence with quadratic potential
let phi = |mv: &Multivector<3,0,0>| mv.norm_squared();
let divergence = bregman_divergence(phi, &p, &q)?;
```

### Optimization

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

## Building

### Prerequisites

- Rust 1.75+ with `cargo`
- Node.js 16+ (for TypeScript bindings)
- `wasm-pack` (installed automatically by build script)

### Build All Tests

```bash
cargo test --workspace
```

### Documentation

```bash
cargo doc --workspace --open
```

API documentation will be available at `target/doc/amari_core/index.html`

## Performance

The library is optimized for high-performance applications:

- SIMD: Vectorized operations where supported
- Cache Alignment: 64-byte aligned data structures
- Const Generics: Zero-cost abstractions for dimensions
- GPU Fallback: Automatic CPU/GPU dispatch based on workload size
- Batch Operations: Efficient batch processing for large datasets

## Integration Status

Amari provides three deployment targets with varying levels of integration across crates:

| Target | Description | Coverage |
|--------|-------------|----------|
| **Rust** | Native library | All crates fully integrated |
| **WASM** | Web browsers/Node.js | Complete coverage for all domain crates |
| **GPU** | Hardware acceleration | Core + network + info-geom + relativistic modules |

For detailed architecture documentation, see [docs/architecture/v0.9.6-multi-gpu-design.md](docs/architecture/v0.9.6-multi-gpu-design.md).

## Documentation

Comprehensive documentation is available in the `docs/` directory:

- **[Release Process](docs/releases/RELEASE_PROCESS_v0.9.8.md)**: Current release procedures
- **[Multi-GPU Design](docs/architecture/v0.9.6-multi-gpu-design.md)**: Multi-GPU architecture details
- **[Publishing Guide](docs/development/PUBLISHING.md)**: Publishing to npm and crates.io
- **[Technical Documentation](docs/technical/)**: Mathematical rigor, phantom types, formal verification

See [docs/README.md](docs/README.md) for the complete documentation index.

## Contributing

Contributions are welcome. For development setup:

```bash
git clone https://github.com/justinelliottcobb/Amari.git
cd amari
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
