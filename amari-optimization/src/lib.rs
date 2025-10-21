//! # Amari Optimization
//!
//! Advanced optimization algorithms and techniques for mathematical computing.
//!
//! This crate provides a comprehensive suite of optimization algorithms designed for
//! integration with the Amari mathematical computing ecosystem. It supports:
//!
//! ## Features
//!
//! - **Linear Programming**: Simplex method, interior-point methods
//! - **Nonlinear Optimization**: Gradient descent, Newton's method, quasi-Newton methods
//! - **Constrained Optimization**: Penalty methods, barrier methods, Lagrange multipliers
//! - **Metaheuristics**: Genetic algorithms, simulated annealing, particle swarm optimization
//! - **Convex Optimization**: Specialized algorithms for convex problems
//! - **Multi-objective Optimization**: Pareto optimization, NSGA-II
//! - **GPU Acceleration**: WGPU-based parallel optimization for large-scale problems
//! - **Geometric Algebra Integration**: Optimization in geometric algebra spaces
//!
//! ## Usage
//!
//! ```rust,no_run
//! use amari_optimization::prelude::*;
//!
//! // Define an objective function
//! let objective = |x: &[f64]| x[0].powi(2) + x[1].powi(2);
//!
//! // Set up optimization problem (to be implemented)
//! // let problem = OptimizationProblem::new(objective)
//! //     .with_bounds(vec![(-10.0, 10.0), (-10.0, 10.0)])
//! //     .with_initial_guess(vec![1.0, 1.0]);
//! //
//! // // Solve using gradient descent (to be implemented)
//! // let result = GradientDescent::default().solve(problem)?;
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! ## Integration with Amari Ecosystem
//!
//! This crate is designed to work seamlessly with other Amari components:
//!
//! - [`amari-core`]: Geometric algebra operations and multivectors
//! - [`amari-dual`]: Automatic differentiation for gradient computation
//! - [`amari-tropical`]: Optimization in tropical semirings
//! - [`amari-gpu`]: GPU acceleration for large-scale optimization
//!
//! [`amari-core`]: https://docs.rs/amari-core
//! [`amari-dual`]: https://docs.rs/amari-dual
//! [`amari-tropical`]: https://docs.rs/amari-tropical
//! [`amari-gpu`]: https://docs.rs/amari-gpu

#![cfg_attr(not(feature = "std"), no_std)]
#![warn(missing_docs)]
#![warn(clippy::all)]

#[cfg(feature = "std")]
extern crate std;

#[cfg(not(feature = "std"))]
extern crate alloc;

#[cfg(not(feature = "std"))]
use alloc::string::String;

use thiserror::Error;

/// Main error type for optimization operations
#[derive(Error, Debug)]
pub enum OptimizationError {
    /// Convergence failure
    #[error("Optimization failed to converge after {iterations} iterations")]
    ConvergenceFailure {
        /// Number of iterations attempted
        iterations: usize,
    },
    /// Invalid problem specification
    #[error("Invalid optimization problem: {message}")]
    InvalidProblem {
        /// Error message
        message: String,
    },
    /// Numerical error during computation
    #[error("Numerical error: {message}")]
    NumericalError {
        /// Error message
        message: String,
    },
    /// GPU computation error
    #[cfg(feature = "gpu")]
    #[error("GPU optimization error: {0}")]
    GpuError(#[from] amari_gpu::UnifiedGpuError),
}

/// Result type for optimization operations
pub type OptimizationResult<T> = Result<T, OptimizationError>;

/// Phantom types for compile-time optimization state verification
pub mod phantom;

/// Natural gradient optimization on statistical manifolds
pub mod natural_gradient;

/// Core optimization traits and types
pub mod core {}

/// Linear programming algorithms
pub mod linear {}

/// Nonlinear optimization algorithms
pub mod nonlinear {}

/// Constrained optimization methods
pub mod constrained;

/// Metaheuristic optimization algorithms
pub mod metaheuristics {}

/// Convex optimization specializations
pub mod convex {}

/// Multi-objective optimization
pub mod multiobjective;

/// GPU-accelerated optimization
#[cfg(feature = "gpu")]
pub mod gpu {}

/// Geometric algebra optimization
pub mod geometric {}

/// Tropical optimization
pub mod tropical;

/// Utility functions and helpers
pub mod utils {}

/// Convenient re-exports for common usage
pub mod prelude {
    pub use crate::{OptimizationError, OptimizationResult};

    // Phantom types for compile-time safety
    pub use crate::phantom::{
        Constrained, ConstraintState, Convex, ConvexityState, Euclidean, HandlesConstrained,
        HandlesMultiObjective, HandlesStatistical, HandlesUnconstrained, ManifoldState,
        MultiObjective, NonConvex, ObjectiveState, OptimizationProblem, RequiresConvex, Riemannian,
        SingleObjective, Statistical, Unconstrained,
    };

    // Natural gradient optimization
    pub use crate::natural_gradient::{
        NaturalGradientConfig, NaturalGradientOptimizer, NaturalGradientResult, ObjectiveWithFisher,
    };

    // Tropical optimization
    pub use crate::tropical::{
        scheduling::TropicalScheduler, TropicalConfig, TropicalConstraint, TropicalObjective,
        TropicalOptimizer, TropicalResult,
    };

    // Multi-objective optimization
    pub use crate::multiobjective::{
        Individual, MultiObjectiveConfig, MultiObjectiveFunction, MultiObjectiveResult, NsgaII,
        ParetoFront,
    };

    // Constrained optimization
    pub use crate::constrained::{
        ConstrainedConfig, ConstrainedObjective, ConstrainedOptimizer, ConstrainedResult,
        PenaltyMethod,
    };
}

// Error types are already defined above and don't need re-export

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_creation() {
        let error = OptimizationError::ConvergenceFailure { iterations: 1000 };
        assert!(error.to_string().contains("1000"));

        let error = OptimizationError::InvalidProblem {
            message: "test".to_string(),
        };
        assert!(error.to_string().contains("test"));
    }
}
