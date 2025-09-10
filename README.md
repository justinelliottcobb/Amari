# Amari 🌟

A high-performance Geometric Algebra/Clifford Algebra library with Information Geometry operations, designed for TypeScript interop via WASM and optional GPU acceleration.

[![Rust](https://img.shields.io/badge/Rust-1.75+-orange.svg)](https://www.rust-lang.org/)
[![WebAssembly](https://img.shields.io/badge/WebAssembly-Ready-blue.svg)](https://webassembly.org/)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.0+-blue.svg)](https://www.typescriptlang.org/)
[![License](https://img.shields.io/badge/License-MIT%20OR%20Apache--2.0-green.svg)](LICENSE)

## ✨ Features

- **High-Performance Core**: Optimized Rust implementation with SIMD support and cache-aligned data structures
- **WebAssembly Bindings**: Zero-copy TypeScript/JavaScript bindings for web applications
- **GPU Acceleration**: Optional WebGPU compute shaders for batch operations
- **Information Geometry**: Fisher metrics, α-connections, Bregman divergences, and Amari-Chentsov tensors
- **Type Safety**: Const generics eliminate runtime dimension checks
- **Flexible Signatures**: Support for arbitrary metric signatures Cl(P,Q,R)

## 🏗️ Architecture

### Crates

- **`amari-core`**: Core Clifford algebra types and CPU implementations
- **`amari-wasm`**: WASM bindings for TypeScript/JavaScript
- **`amari-gpu`**: Optional GPU acceleration via WebGPU/wgpu
- **`amari-info-geom`**: Information geometry operations

### Key Types

```rust
// Multivector in Clifford algebra Cl(P,Q,R)
Multivector<const P: usize, const Q: usize, const R: usize>

// Common algebras
type Cl3 = Multivector<3, 0, 0>;  // 3D Euclidean
type Spacetime = Multivector<1, 3, 0>;  // Minkowski spacetime
```

## 🚀 Quick Start

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

## 🔧 Building

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

## 📊 Performance

The library is optimized for high-performance applications:

- **SIMD**: Vectorized operations where supported
- **Cache Alignment**: 64-byte aligned data structures
- **Const Generics**: Zero-cost abstractions for dimensions
- **GPU Fallback**: Automatic CPU/GPU dispatch based on workload size
- **Batch Operations**: Efficient batch processing for large datasets

### Benchmarks

Run benchmarks to see performance on your system:

```bash
./build.sh --bench
```

## 🧮 Mathematical Foundation

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

## 📚 Examples

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

## 🧪 Testing

```bash
# Run all tests
cargo test --workspace

# Run with features
cargo test --workspace --features "parallel"

# Property-based tests
cargo test --workspace --features "proptest"
```

## 📖 Documentation

```bash
# Generate and open docs
cargo doc --workspace --open

# API documentation will be available at:
# target/doc/amari_core/index.html
```

## 🎯 Use Cases

- **Computer Graphics**: Rotations, reflections, and transformations
- **Robotics**: Orientation representation and interpolation  
- **Physics**: Spacetime calculations and electromagnetic field theory
- **Machine Learning**: Statistical manifold operations and natural gradients
- **Computer Vision**: Multi-view geometry and camera calibration

## 🔬 Research Applications

- **Information Geometry**: Statistical manifold computations
- **Geometric Deep Learning**: Operations on non-Euclidean data
- **Quantum Computing**: Clifford group operations
- **Crystallography**: Symmetry group calculations

## 🤝 Contributing

Contributions are welcome! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup

```bash
git clone https://github.com/your-username/amari.git
cd amari
./build.sh --examples
```

## 📄 License

This project is licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT License ([LICENSE-MIT](LICENSE-MIT))

at your option.

## 🙏 Acknowledgments

- Inspired by the geometric algebra community and research in Information Geometry
- Built with modern Rust performance idioms and WebAssembly best practices
- Named after Shun-ichi Amari's contributions to Information Geometry

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/your-username/amari/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-username/amari/discussions)
- **Documentation**: [API Docs](https://docs.rs/amari)

---

*"Geometry is the art of correct reasoning from incorrectly drawn figures."* - Henri Poincaré