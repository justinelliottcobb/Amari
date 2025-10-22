# Amari Optimization

Advanced optimization algorithms for geometric computing with information geometry, tropical algebra, and multi-objective optimization.

## Overview

Amari Optimization provides state-of-the-art optimization algorithms designed for mathematical and scientific computing. The library emphasizes type safety, performance, and mathematical rigor with compile-time verification through phantom types.

## Features

### 🎯 **Natural Gradient Optimization**
- Information geometric optimization on statistical manifolds
- Fisher information matrix computation
- Riemannian optimization with proper metric tensors
- Ideal for machine learning and statistical parameter estimation

### 🌴 **Tropical Optimization**
- Max-plus algebra optimization algorithms
- Task scheduling and resource allocation
- Shortest path problems in tropical semirings
- Critical path analysis and timing optimization

### 🎪 **Multi-Objective Optimization**
- NSGA-II algorithm for Pareto optimization
- Hypervolume calculation and metrics
- Crowding distance for diversity preservation
- Engineering design trade-offs and decision making

### ⚖️ **Constrained Optimization**
- Penalty methods (exterior/interior)
- Barrier methods with logarithmic barriers
- Augmented Lagrangian methods
- KKT condition verification

### 🔧 **Type-Safe Design**
- Phantom types for compile-time verification
- Dimension checking at compile time
- Optimization state tracking
- Zero-cost abstractions

## Quick Start

Add to your `Cargo.toml`:

```toml
[dependencies]
amari-optimization = "0.9.7-1"
```

### Basic Usage

```rust
use amari_optimization::prelude::*;

// Natural gradient optimization
let config = NaturalGradientConfig::default();
let optimizer = NaturalGradientOptimizer::new(config);

// Multi-objective optimization
let nsga2_config = MultiObjectiveConfig::default();
let nsga2 = NsgaII::new(nsga2_config);

// Tropical optimization
let tropical_optimizer = TropicalOptimizer::with_default_config();
```

## Examples

### Natural Gradient Optimization

```rust
use amari_optimization::prelude::*;

struct GaussianMLE {
    data: Vec<f64>,
}

impl ObjectiveWithFisher<f64> for GaussianMLE {
    fn evaluate(&self, params: &[f64]) -> f64 {
        // Negative log-likelihood
        let mu = params[0];
        let log_sigma = params[1];
        // ... implementation
    }

    fn gradient(&self, params: &[f64]) -> Vec<f64> {
        // Gradient computation
        // ... implementation
    }

    fn fisher_information(&self, params: &[f64]) -> Vec<Vec<f64>> {
        // Fisher information matrix
        // ... implementation
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let data = vec![1.0, 2.0, 3.0, 2.5, 1.8]; // Your data
    let objective = GaussianMLE { data };

    let config = NaturalGradientConfig::default();
    let optimizer = NaturalGradientOptimizer::new(config);

    // Create type-safe problem specification
    use amari_optimization::phantom::{Statistical, NonConvex, SingleObjective, Unconstrained};
    let problem: OptimizationProblem<2, Unconstrained, SingleObjective, NonConvex, Statistical> =
        OptimizationProblem::new();

    let initial_params = vec![0.0, 0.0]; // mu, log_sigma
    let result = optimizer.optimize_statistical(&problem, &objective, initial_params)?;

    println!("Converged: {}", result.converged);
    println!("Final parameters: {:?}", result.parameters);
    Ok(())
}
```

### Multi-Objective Optimization

```rust
use amari_optimization::prelude::*;

struct EngineeringDesign;

impl MultiObjectiveFunction<f64> for EngineeringDesign {
    fn num_objectives(&self) -> usize { 3 }
    fn num_variables(&self) -> usize { 3 }

    fn evaluate(&self, variables: &[f64]) -> Vec<f64> {
        let thickness = variables[0];
        let width = variables[1];
        let height = variables[2];

        let volume = thickness * width * height;
        let weight = volume * 2.7; // kg
        let cost = weight * 3.5;   // $
        let strength = -(thickness.sqrt() * width * height).powf(0.3); // Negative for minimization

        vec![weight, cost, strength]
    }

    fn variable_bounds(&self) -> Vec<(f64, f64)> {
        vec![(0.001, 0.1), (0.1, 2.0), (0.1, 2.0)]
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let problem_fn = EngineeringDesign;
    let config = MultiObjectiveConfig::default();
    let nsga2 = NsgaII::new(config);

    use amari_optimization::phantom::{Euclidean, NonConvex, Unconstrained};
    let problem: OptimizationProblem<3, Unconstrained, MultiObjective, NonConvex, Euclidean> =
        OptimizationProblem::new();

    let result = nsga2.optimize(&problem, &problem_fn)?;

    println!("Pareto front size: {}", result.pareto_front.solutions.len());
    if let Some(hv) = result.pareto_front.hypervolume {
        println!("Hypervolume: {:.6}", hv);
    }
    Ok(())
}
```

### Tropical Optimization

```rust
use amari_optimization::prelude::*;
use amari_tropical::{TropicalMatrix, TropicalNumber};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Task scheduling with dependencies
    let dependency_data = vec![
        vec![0.0, 2.0, f64::NEG_INFINITY, 1.0],
        vec![f64::NEG_INFINITY, 0.0, 1.0, 2.0],
        vec![3.0, f64::NEG_INFINITY, 0.0, 1.0],
        vec![f64::NEG_INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY, 0.0],
    ];

    let dependency_matrix = TropicalMatrix::from_log_probs(&dependency_data);
    let optimizer = TropicalOptimizer::with_default_config();

    // Solve for critical cycle time
    let eigen_result = optimizer.solve_tropical_eigenvalue(&dependency_matrix)?;
    if let Some(critical_time) = eigen_result.eigenvalue {
        println!("Critical cycle time: {:.2} hours", critical_time);
    }

    Ok(())
}
```

## Detailed Examples

The `examples/` directory contains comprehensive demonstrations:

- **`natural_gradient_example.rs`**: Complete Gaussian parameter estimation with Fisher information
- **`tropical_optimization_example.rs`**: Task scheduling, resource allocation, and shortest path problems
- **`multi_objective_example.rs`**: Engineering design trade-offs and environmental optimization

Run examples with:
```bash
cargo run --example natural_gradient_example
cargo run --example tropical_optimization_example
cargo run --example multi_objective_example
```

## API Reference

### Core Traits

#### `ObjectiveWithFisher<T>`
For natural gradient optimization on statistical manifolds:
```rust
trait ObjectiveWithFisher<T: Float> {
    fn evaluate(&self, params: &[T]) -> T;
    fn gradient(&self, params: &[T]) -> Vec<T>;
    fn fisher_information(&self, params: &[T]) -> Vec<Vec<T>>;
}
```

#### `MultiObjectiveFunction<T>`
For multi-objective Pareto optimization:
```rust
trait MultiObjectiveFunction<T: Float> {
    fn num_objectives(&self) -> usize;
    fn num_variables(&self) -> usize;
    fn evaluate(&self, variables: &[T]) -> Vec<T>;
    fn variable_bounds(&self) -> Vec<(T, T)>;
    fn evaluate_constraints(&self, variables: &[T]) -> Vec<T> { vec![] }
    fn ideal_point(&self) -> Option<Vec<T>> { None }
}
```

#### `ConstrainedObjective<T>`
For constrained optimization problems:
```rust
trait ConstrainedObjective<T: Float> {
    fn evaluate(&self, x: &[T]) -> T;
    fn gradient(&self, x: &[T]) -> Vec<T>;
    fn evaluate_constraints(&self, x: &[T]) -> Vec<T>;
    fn constraint_gradients(&self, x: &[T]) -> Vec<Vec<T>>;
    fn variable_bounds(&self) -> Vec<(T, T)>;
}
```

### Optimization Algorithms

#### `NaturalGradientOptimizer<T>`
Information geometric optimization:
```rust
impl<T: Float> NaturalGradientOptimizer<T> {
    pub fn new(config: NaturalGradientConfig<T>) -> Self;
    pub fn optimize_statistical<const DIM: usize, O: ObjectiveWithFisher<T>>(
        &self,
        problem: &OptimizationProblem<DIM, Unconstrained, SingleObjective, NonConvex, Statistical>,
        objective: &O,
        initial_params: Vec<T>,
    ) -> Result<NaturalGradientResult<T>, OptimizationError>;
}
```

#### `NsgaII<T>`
Multi-objective evolutionary optimization:
```rust
impl<T: Float> NsgaII<T> {
    pub fn new(config: MultiObjectiveConfig<T>) -> Self;
    pub fn optimize<const DIM: usize, O: MultiObjectiveFunction<T>>(
        &self,
        problem: &OptimizationProblem<DIM, Unconstrained, MultiObjective, NonConvex, Euclidean>,
        objective: &O,
    ) -> Result<MultiObjectiveResult<T>, OptimizationError>;
}
```

#### `TropicalOptimizer`
Max-plus algebra optimization:
```rust
impl TropicalOptimizer {
    pub fn with_default_config() -> Self;
    pub fn solve_tropical_eigenvalue(&self, matrix: &TropicalMatrix) -> Result<TropicalEigenResult, OptimizationError>;
    pub fn solve_tropical_linear_program(&self, objective: &[TropicalNumber], constraints: &TropicalMatrix, rhs: &[TropicalNumber]) -> Result<TropicalLPResult, OptimizationError>;
    pub fn solve_shortest_path(&self, matrix: &TropicalMatrix, start: usize, end: usize) -> Result<ShortestPathResult, OptimizationError>;
}
```

### Phantom Types

The library uses phantom types for compile-time verification:

```rust
// Constraint states
struct Constrained;
struct Unconstrained;

// Objective states
struct SingleObjective;
struct MultiObjective;

// Convexity states
struct Convex;
struct NonConvex;

// Manifold types
struct Euclidean;
struct Statistical;

// Problem specification
struct OptimizationProblem<const DIM: usize, C, O, V, M> {
    // Zero-sized phantom fields
}
```

## Configuration

### Natural Gradient Configuration
```rust
pub struct NaturalGradientConfig<T: Float> {
    pub learning_rate: T,
    pub max_iterations: usize,
    pub gradient_tolerance: T,
    pub parameter_tolerance: T,
    pub fisher_regularization: T,
    pub use_line_search: bool,
    pub line_search_beta: T,
    pub line_search_alpha: T,
}
```

### Multi-Objective Configuration
```rust
pub struct MultiObjectiveConfig<T: Float> {
    pub population_size: usize,
    pub max_generations: usize,
    pub crossover_probability: T,
    pub mutation_probability: T,
    pub mutation_strength: T,
    pub elite_ratio: T,
    pub convergence_tolerance: T,
    pub reference_point: Option<Vec<T>>,
    pub preserve_diversity: bool,
}
```

### Constrained Optimization Configuration
```rust
pub struct ConstrainedConfig<T: Float> {
    pub method: ConstrainedMethod,
    pub penalty_parameter: T,
    pub penalty_increase_factor: T,
    pub barrier_parameter: T,
    pub max_outer_iterations: usize,
    pub max_inner_iterations: usize,
    pub tolerance: T,
    pub gradient_tolerance: T,
}
```

## Performance

The library is optimized for performance with:

- **SIMD operations** where available
- **GPU acceleration** support (optional feature)
- **Parallel computation** for population-based algorithms
- **Memory-efficient** implementations
- **Zero-cost abstractions** through phantom types

Run benchmarks with:
```bash
cargo bench
```

## Features

Enable optional features in `Cargo.toml`:

```toml
[dependencies]
amari-optimization = { version = "0.9.7", features = ["gpu", "serde", "parallel"] }
```

Available features:
- **`gpu`**: GPU acceleration support
- **`serde`**: Serialization support for configurations and results
- **`parallel`**: Parallel processing for multi-objective algorithms
- **`linalg`**: Enhanced linear algebra operations

## Mathematical Background

### Information Geometry
Natural gradient optimization uses the Fisher information metric to define gradients on statistical manifolds, providing parameter-invariant optimization that converges faster than standard gradient descent for statistical problems.

### Tropical Algebra
Tropical optimization uses the max-plus semiring where addition becomes max and multiplication becomes addition. This algebra is naturally suited for scheduling, timing analysis, and discrete optimization problems.

### Multi-Objective Optimization
NSGA-II uses evolutionary principles with Pareto dominance ranking and crowding distance to maintain diversity while converging to the Pareto front of optimal trade-off solutions.

### Constrained Optimization
The library implements classical constrained optimization methods:
- **Penalty methods**: Transform constraints into penalty terms
- **Barrier methods**: Use logarithmic barriers to enforce constraints
- **Augmented Lagrangian**: Combine Lagrange multipliers with penalties

## Testing

The library includes comprehensive testing:

```bash
# Unit tests
cargo test

# Integration tests
cargo test --test integration_tests

# Property-based tests
cargo test --test property_tests

# Documentation tests
cargo test --doc
```

## Contributing

Contributions are welcome! Please ensure:
1. All tests pass: `cargo test`
2. Code is formatted: `cargo fmt`
3. No clippy warnings: `cargo clippy`
4. Benchmarks show no regressions: `cargo bench`

## License

Licensed under either of

 * Apache License, Version 2.0, ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
 * MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

## References

1. Amari, S. (1998). *Natural Gradient Works Efficiently in Learning*
2. Deb, K. et al. (2002). *A Fast and Elitist Multiobjective Genetic Algorithm: NSGA-II*
3. Maclagan, D. & Sturmfels, B. (2015). *Introduction to Tropical Geometry*
4. Nocedal, J. & Wright, S. (2006). *Numerical Optimization*

---

For more information, visit the [Amari project documentation](https://github.com/amari-team/amari).