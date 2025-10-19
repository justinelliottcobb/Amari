# Amari Optimization

Advanced optimization algorithms and techniques for mathematical computing.

## Overview

The `amari-optimization` crate provides a comprehensive suite of optimization algorithms designed for integration with the Amari mathematical computing ecosystem. This module offers both classical and modern optimization techniques with support for GPU acceleration and geometric algebra spaces.

## Features

- **Linear Programming**: Simplex method, interior-point methods
- **Nonlinear Optimization**: Gradient descent, Newton's method, quasi-Newton methods
- **Constrained Optimization**: Penalty methods, barrier methods, Lagrange multipliers
- **Metaheuristics**: Genetic algorithms, simulated annealing, particle swarm optimization
- **Convex Optimization**: Specialized algorithms for convex problems
- **Multi-objective Optimization**: Pareto optimization, NSGA-II
- **GPU Acceleration**: WGPU-based parallel optimization for large-scale problems
- **Geometric Algebra Integration**: Optimization in geometric algebra spaces
- **Tropical Optimization**: Optimization in tropical semirings and max-plus algebras

## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
amari-optimization = "0.9.7"
```

For GPU support:

```toml
[dependencies]
amari-optimization = { version = "0.9.7", features = ["gpu"] }
```

## Usage

```rust
use amari_optimization::prelude::*;

// Basic error handling
let result: OptimizationResult<f64> = Ok(42.0);
match result {
    Ok(value) => println!("Optimization result: {}", value),
    Err(e) => println!("Optimization failed: {}", e),
}
```

## Modules

- `core`: Core optimization framework and trait definitions
- `linear`: Linear programming solvers
- `nonlinear`: Nonlinear optimization algorithms
- `constrained`: Constrained optimization methods
- `metaheuristics`: Population-based optimization algorithms
- `convex`: Specialized convex optimization algorithms
- `multiobjective`: Multi-objective optimization techniques
- `gpu`: GPU-accelerated optimization (with `gpu` feature)
- `geometric`: Geometric algebra optimization
- `tropical`: Tropical semiring optimization
- `utils`: Common utilities and helpers

## Integration

This crate integrates seamlessly with other Amari components:

- **amari-core**: Geometric algebra operations and multivectors
- **amari-dual**: Automatic differentiation for gradient computation
- **amari-tropical**: Optimization in tropical semirings
- **amari-gpu**: GPU acceleration for large-scale optimization

## Development Status

This module is part of Amari v0.9.7 and is currently in active development. The foundational types and error handling are implemented, with optimization algorithms to be added in subsequent releases.

## License

Licensed under either of

 * Apache License, Version 2.0, ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
 * MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.