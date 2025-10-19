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

/// Core optimization traits and types
pub mod core {
    //! Core optimization framework and trait definitions
}

/// Linear programming algorithms
pub mod linear {
    //! Linear programming solvers including simplex and interior-point methods
}

/// Nonlinear optimization algorithms
pub mod nonlinear {
    //! Nonlinear optimization algorithms for unconstrained and constrained problems
}

/// Constrained optimization methods
pub mod constrained {
    //! Algorithms for optimization with equality and inequality constraints
}

/// Metaheuristic optimization algorithms
pub mod metaheuristics {
    //! Population-based and nature-inspired optimization algorithms
}

/// Convex optimization specializations
pub mod convex {
    //! Specialized algorithms for convex optimization problems
}

/// Multi-objective optimization
pub mod multiobjective {
    //! Algorithms for problems with multiple conflicting objectives
}

/// GPU-accelerated optimization
#[cfg(feature = "gpu")]
pub mod gpu {
    //! GPU-accelerated optimization algorithms using WGPU
}

/// Geometric algebra optimization
pub mod geometric {
    //! Optimization algorithms operating in geometric algebra spaces
}

/// Tropical optimization
pub mod tropical {
    //! Optimization in tropical semirings and max-plus algebras
}

/// Utility functions and helpers
pub mod utils {
    //! Common utilities for optimization algorithms
}

/// Convenient re-exports for common usage
pub mod prelude {
    //! Commonly used types and traits for optimization

    pub use crate::{OptimizationError, OptimizationResult};
    // Additional re-exports will be added as modules are implemented
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