//! Functional analysis on multivector spaces.
//!
//! This crate provides the mathematical foundations of functional analysis
//! applied to Clifford algebra-valued function spaces, including:
//!
//! - **Hilbert spaces**: Complete inner product spaces over multivectors
//! - **Linear operators**: Bounded and unbounded operators on Hilbert spaces
//! - **Spectral theory**: Eigenvalues, eigenvectors, and spectral decomposition
//! - **Compact operators**: Fredholm theory and index theorems
//! - **Sobolev spaces**: Function spaces with weak derivatives
//!
//! # Space Hierarchy
//!
//! The module implements the standard functional analysis hierarchy:
//!
//! ```text
//! VectorSpace
//!     ↓
//! NormedSpace (adds ||·||)
//!     ↓
//! BanachSpace (complete normed space)
//!     ↓
//! InnerProductSpace (adds ⟨·,·⟩)
//!     ↓
//! HilbertSpace (complete inner product space)
//! ```
//!
//! # Quick Start
//!
//! ## Finite-Dimensional Hilbert Space
//!
//! ```
//! use amari_functional::space::{MultivectorHilbertSpace, HilbertSpace, InnerProductSpace};
//!
//! // Create the Hilbert space Cl(2,0,0) ≅ ℝ⁴
//! let space: MultivectorHilbertSpace<2, 0, 0> = MultivectorHilbertSpace::new();
//!
//! // Create elements
//! let x = space.from_coefficients(&[1.0, 2.0, 0.0, 0.0]).unwrap();
//! let y = space.from_coefficients(&[0.0, 0.0, 3.0, 4.0]).unwrap();
//!
//! // Compute inner product
//! let ip = space.inner_product(&x, &y);
//! assert!(ip.abs() < 1e-10); // x and y are orthogonal
//! ```
//!
//! ## L² Space of Multivector Functions
//!
//! ```
//! use amari_functional::space::{MultivectorL2, L2Function, NormedSpace};
//! use amari_core::Multivector;
//!
//! // Create L²([0,1], Cl(2,0,0))
//! let l2: MultivectorL2<2, 0, 0> = MultivectorL2::unit_interval();
//!
//! // Define a function f(x) = x * e₀ (scalar coefficient is x)
//! let f = L2Function::new(|x| {
//!     Multivector::<2, 0, 0>::scalar(x[0])
//! });
//!
//! // Compute L² norm: ||f||² = ∫₀¹ x² dx = 1/3
//! let norm_sq = l2.norm(&f).powi(2);
//! assert!((norm_sq - 1.0/3.0).abs() < 0.1);
//! ```
//!
//! # Mathematical Background
//!
//! This crate is built on the following mathematical concepts:
//!
//! - **Clifford algebras Cl(P,Q,R)**: 2^(P+Q+R)-dimensional real algebras
//! - **Hilbert spaces**: Complete inner product spaces with orthogonal decomposition
//! - **Bounded operators**: Continuous linear maps between Hilbert spaces
//! - **Spectral theorem**: Self-adjoint operators have orthonormal eigenbases
//! - **Compact operators**: Operators mapping bounded sets to precompact sets
//!
//! # Features
//!
//! - `std` (default): Enable standard library features
//! - `parallel`: Enable Rayon parallelism for large-scale computations
//! - `formal-verification`: Enable Creusot contracts for formal verification

#![cfg_attr(not(feature = "std"), no_std)]
#![warn(missing_docs)]
#![warn(clippy::all)]
#![allow(dead_code)] // SpaceWithCompleteness reserved for future use
#![allow(clippy::type_complexity)] // Complex function types are intentional for flexibility
#![allow(clippy::needless_range_loop)] // Matrix indexing is clearer with explicit ranges

// Note: formal-verification feature is reserved for future Creusot integration
// #[cfg(feature = "formal-verification")]
// extern crate creusot_contracts;

// Re-export core types
pub use amari_core::{Multivector, Scalar};

// Error types
mod error;
pub use error::{FunctionalError, Result};

// Phantom types for compile-time property verification
mod phantom;
pub use phantom::{
    // Boundedness
    Bounded,
    BoundednessProperty,
    // Compactness
    Compact,
    // Type aliases
    CompactSelfAdjointOperator,
    CompactnessProperty,
    // Completeness
    Complete,
    CompletenessProperty,
    // Spectral
    ContinuousSpectrum,
    DiscreteSpectrum,
    // Fredholm
    Fredholm,
    FredholmProperty,
    // Symmetry
    General,
    // Regularity
    H1Regularity,
    H1SpaceProperties,
    H2Regularity,
    HilbertSchmidtOperator,
    HkRegularity,
    L2Regularity,
    L2SpaceProperties,
    MixedSpectrum,
    NonCompact,
    Normal,
    NotFredholm,
    PreHilbert,
    // Wrapper
    Properties,
    PurePointSpectrum,
    RegularityProperty,
    SelfAdjoint,
    SemiFredholm,
    SpectralProperty,
    SymmetryProperty,
    Unbounded,
    Unitary,
    UnitaryOperator,
};

// Function spaces
pub mod space;
pub use space::{
    BanachSpace, Domain, HilbertSpace, InnerProductSpace, L2Function, MultivectorHilbertSpace,
    MultivectorL2, NormedSpace, VectorSpace,
};

// Linear operators
pub mod operator;
pub use operator::{
    AdjointableOperator, BoundedOperator, CompactMatrixOperator, CompactOperator,
    CompositeOperator, FiniteRankOperator, FredholmMatrixOperator, FredholmOperator,
    IdentityOperator, LinearOperator, MatrixOperator, OperatorNorm, ProjectionOperator,
    ScalingOperator, SelfAdjointOperator, ZeroOperator,
};

// Spectral theory
pub mod spectral;
pub use spectral::{
    compute_eigenvalues, inverse_iteration, power_method, spectral_decompose, Eigenpair,
    Eigenvalue, SpectralDecomposition,
};

// Sobolev spaces
pub mod sobolev;
pub use sobolev::{poincare_constant_estimate, H1Space, H2Space, SobolevFunction, SobolevSpace};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_crate_compiles() {
        // Basic smoke test to ensure the crate structure is valid
        let space: MultivectorHilbertSpace<2, 0, 0> = MultivectorHilbertSpace::new();
        assert_eq!(space.signature(), (2, 0, 0));
    }

    #[test]
    fn test_l2_space_compiles() {
        let l2: MultivectorL2<2, 0, 0> = MultivectorL2::unit_interval();
        assert_eq!(l2.signature(), (2, 0, 0));
    }
}
