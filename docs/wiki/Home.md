# Amari - Geometric Algebra Library for Rust

Welcome to the Amari wiki! Amari is a comprehensive geometric algebra library for Rust, providing tools for Clifford algebras, differential geometry, tropical algebra, and more.

## Quick Links

- [[Getting Started]]
- [[v1.0.0 Release Notes]]
- [[Architecture Overview]]
- [[API Reference]](https://docs.rs/amari)

## Crate Overview

| Crate | Description |
|-------|-------------|
| `amari-core` | Core Clifford algebra types and operations |
| `amari-dual` | Automatic differentiation via dual numbers |
| `amari-tropical` | Tropical (max-plus) algebra |
| `amari-calculus` | Geometric calculus and differential operators |
| `amari-topology` | Simplicial complexes and homology |
| `amari-functional` | Functional analysis on Clifford spaces |
| `amari-info-geom` | Information geometry and Fisher metrics |
| `amari-optimization` | Optimization on geometric manifolds |
| `amari-network` | Graph algorithms with geometric structure |
| `amari-probabilistic` | Probabilistic methods and SDEs |
| `amari-measure` | Measure theory and integration |
| `amari-relativistic` | Special and general relativity |
| `amari-holographic` | Holographic principle and AdS/CFT |
| `amari-enumerative` | Enumerative geometry (Gromov-Witten, etc.) |
| `amari-automata` | Cellular automata with geometric algebra |
| `amari-fusion` | Tropical-Dual-Clifford fusion for ML |
| `amari-dynamics` | Dynamical systems analysis |
| `amari-gpu` | GPU acceleration via WebGPU/wgpu |
| `amari-wasm` | WebAssembly bindings |

## Features

- **No-std compatible**: Core crates work in embedded environments
- **Compile-time verification**: Phantom types ensure dimensional consistency
- **GPU acceleration**: WebGPU support for parallel computation
- **WebAssembly ready**: Full WASM support for web deployment
- **Comprehensive testing**: Extensive test suites with property-based testing

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
amari = "1.0"
```

Or use individual crates:

```toml
[dependencies]
amari-core = "1.0"
amari-dual = "1.0"
```

## License

MIT OR Apache-2.0
