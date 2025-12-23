//! Probability theory on geometric algebra spaces
//!
//! This crate provides probability distributions, stochastic processes, and
//! Bayesian inference for multivector-valued random variables in Clifford algebras.
//!
//! # Overview
//!
//! `amari-probabilistic` extends probability theory to geometric algebra, enabling:
//!
//! - **Distributions** over multivector spaces Cl(P,Q,R)
//! - **Stochastic differential equations** with geometric structure
//! - **MCMC sampling** using geometric product operations
//! - **Bayesian inference** on statistical manifolds
//!
//! # Quick Start
//!
//! ## 1. Sampling from Multivector Distributions
//!
//! ```
//! use amari_probabilistic::distribution::{Distribution, GaussianMultivector};
//! use amari_core::Multivector;
//!
//! // Standard Gaussian on Cl(3,0,0) - 8-dimensional multivector space
//! let gaussian = GaussianMultivector::<3, 0, 0>::standard();
//!
//! // Draw samples
//! let mut rng = rand::thread_rng();
//! let sample: Multivector<3, 0, 0> = gaussian.sample(&mut rng);
//!
//! // Evaluate log-probability
//! let log_p = gaussian.log_prob(&sample).unwrap();
//! ```
//!
//! ## 2. Grade-Specific Distributions
//!
//! ```
//! use amari_probabilistic::distribution::GaussianMultivector;
//!
//! // Distribution concentrated on bivectors (grade 2)
//! let bivector_dist = GaussianMultivector::<3, 0, 0>::grade_concentrated(2, 1.0).unwrap();
//!
//! // Grade marginal from full distribution
//! use amari_probabilistic::distribution::MultivectorDistribution;
//! let full = GaussianMultivector::<3, 0, 0>::standard();
//! let vector_marginal = full.grade_marginal(1);
//! ```
//!
//! # Mathematical Foundation
//!
//! ## Probability on Cl(P,Q,R)
//!
//! A Clifford algebra Cl(P,Q,R) with signature (P,Q,R) has dimension 2^(P+Q+R).
//! Probability distributions on this space assign measures to subsets of the
//! multivector space, respecting the geometric structure.
//!
//! Key considerations:
//!
//! - **Grade structure**: Distributions may concentrate on specific grades
//! - **Geometric products**: Random multivectors can be multiplied geometrically
//! - **Rotor subgroups**: Special distributions on rotation representations
//!
//! ## Integration with Amari Ecosystem
//!
//! - **amari-measure**: Provides the measure-theoretic foundation
//! - **amari-info-geom**: Fisher geometry for natural gradients
//! - **amari-dual**: Automatic differentiation for gradient computation
//!
//! # Feature Flags
//!
//! - `std` (default): Use standard library
//! - `parallel`: Enable parallel sampling with rayon
//! - `gpu`: GPU-accelerated sampling via amari-gpu

#![cfg_attr(not(feature = "std"), no_std)]
#![warn(missing_docs)]
#![warn(clippy::all)]

// Re-export core types
pub use amari_core::Multivector;

// Error types
mod error;
pub use error::{ProbabilisticError, Result};

// Phantom type system for distribution properties
mod phantom;
pub use phantom::{
    // Support properties
    Bounded,
    Continuous,
    Discrete,
    DistributionProperty,
    // Moment properties
    FiniteMoments,
    GeometricProperty,
    // Geometric properties
    GradeHeterogeneous,
    GradeHomogeneous,
    HeavyTailed,
    LightTailed,
    MomentProperty,
    PropertyMarker,
    RotorValued,
    SupportProperty,
    Unbounded,
    VersorValued,
};

// Distribution trait and implementations
pub mod distribution;
pub use distribution::{
    Distribution, GaussianMultivector, GradeProjectedDistribution, HasMoments, HasSupport,
    MultivectorDistribution, UniformMultivector,
};

// Random variable traits (to be implemented)
pub mod random;

// Sampling algorithms (to be implemented)
pub mod sampling;

// Stochastic processes (to be implemented)
pub mod stochastic;

// Bayesian inference (to be implemented)
pub mod bayesian;

// Monte Carlo methods (to be implemented)
pub mod monte_carlo;

// Uncertainty propagation (to be implemented)
pub mod uncertainty;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_crate_compiles() {
        // Basic smoke test
        let gaussian = GaussianMultivector::<2, 0, 0>::standard();
        let mut rng = rand::thread_rng();
        let _ = gaussian.sample(&mut rng);
    }

    #[test]
    fn test_phantom_types_exported() {
        // Verify phantom types are accessible
        fn _check_property<T: DistributionProperty>() {}
        _check_property::<Bounded>();
        _check_property::<Unbounded>();
        _check_property::<LightTailed>();
    }
}
