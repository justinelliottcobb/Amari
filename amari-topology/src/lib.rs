//! # amari-topology
//!
//! Topological tools for geometric structures built on Clifford algebra.
//!
//! This crate provides computational topology primitives integrated with
//! geometric algebra, enabling rigorous analysis of geometric structures
//! through algebraic topology methods.
//!
//! ## Core Capabilities
//!
//! - **Simplicial Complexes**: Abstract simplicial complexes with chain groups
//! - **Homology**: Compute homology groups H_k of simplicial complexes
//! - **Persistent Homology**: Track topological features across filtrations
//! - **Morse Theory**: Critical point analysis of scalar fields
//! - **Manifold Boundaries**: Detect and characterize manifold boundaries
//! - **Fiber Bundles**: Geometric structures over multivector spaces
//!
//! ## Features
//!
//! - `std` (default): Enable standard library support
//! - `parallel`: Enable Rayon-based parallel computation
//!
//! ## Phantom Types
//!
//! The crate uses phantom types for compile-time verification of:
//! - Orientation tracking (Oriented/Unoriented)
//! - Coefficient rings (ℤ, ℤ/2ℤ, ℝ)
//! - Complex properties (Closed/WithBoundary)
//! - Filtration validation state
//!
//! ## Example
//!
//! ```rust
//! use amari_topology::{SimplicialComplex, Simplex};
//!
//! // Create a triangle (2-simplex with its faces)
//! let mut complex = SimplicialComplex::new();
//! complex.add_simplex(Simplex::new(vec![0, 1, 2])); // Triangle
//!
//! // Compute Betti numbers
//! let betti = complex.betti_numbers();
//! assert_eq!(betti[0], 1); // One connected component
//! assert_eq!(betti[1], 0); // No holes (filled triangle)
//! ```
//!
//! ## Parallel Computation
//!
//! Enable the `parallel` feature for Rayon-based parallelism:
//!
//! ```toml
//! amari-topology = { version = "0.16", features = ["parallel"] }
//! ```
//!
//! ```rust,ignore
//! use amari_topology::parallel::*;
//!
//! // Parallel Betti number computation
//! let betti = parallel_betti_numbers(&complex);
//!
//! // Parallel Rips filtration construction
//! let filt = parallel_rips_filtration(points, max_dim, distance);
//! ```
//!
//! ## Mathematical Background
//!
//! The crate implements algebraic topology over geometric algebra spaces,
//! combining classical methods with the computational advantages of Clifford
//! algebra representations.
//!
//! Key mathematical structures:
//! - **Chain Complex**: C_n → C_{n-1} → ... → C_0 with ∂∂ = 0
//! - **Homology Groups**: H_k = ker(∂_k) / im(∂_{k+1})
//! - **Betti Numbers**: β_k = dim(H_k)
//! - **Euler Characteristic**: χ = Σ (-1)^k β_k

#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(not(feature = "std"))]
extern crate alloc;

#[cfg(not(feature = "std"))]
use alloc::{vec, vec::Vec};

// Core modules
pub mod boundary;
pub mod bundle;
pub mod chain;
pub mod complex;
pub mod homology;
pub mod morse;
pub mod persistence;
pub mod simplex;

// Enhanced modules
pub mod phantom;
pub mod verified_contracts;

// Parallel computation (feature-gated)
#[cfg(feature = "parallel")]
pub mod parallel;

// Re-export parallel module items when feature is enabled
#[cfg(feature = "parallel")]
pub use parallel::*;

// Re-exports - Core types
pub use boundary::{
    is_manifold, is_orientable, BoundaryComponent, ManifoldBoundary, OrientedBoundary,
};
pub use bundle::{Connection, FiberBundle, PrincipalBundle, Section, VectorBundle};
pub use chain::{BoundaryMap, Chain, ChainGroup};
pub use complex::SimplicialComplex;
pub use homology::{compute_homology, BettiNumbers, HomologyGroup};
pub use morse::{
    classify_critical_point, find_critical_points_grid, CriticalPoint, CriticalType, MorseComplex,
    MorseFunction,
};
pub use persistence::{
    rips_filtration, BarcodeInterval, Filtration, PersistenceDiagram, PersistentHomology,
};
pub use simplex::Simplex;

// Re-exports - Phantom types
pub use phantom::{
    // Complex properties
    BoundaryProperty,
    Closed,
    // Coefficient rings
    CoefficientRing,
    Connected,
    ConnectivityProperty,
    Disconnected,
    IntegerCoefficients,
    Mod2Coefficients,
    OrientationProperty,
    // Orientation
    Oriented,
    RealCoefficients,
    // Typed wrappers
    TypedChain,
    TypedFiltration,
    TypedSimplex,
    Unoriented,
    // Filtration validation
    Unvalidated,
    Validated,
    ValidationProperty,
    WithBoundary,
};

// Re-exports - Verified contracts
pub use verified_contracts::{
    verify_beta_0_counts_components, verify_betti_nonnegative, verify_boundary_squared_zero,
    verify_euler_poincare, verify_strong_morse_inequality, verify_weak_morse_inequalities,
    VerifiedBoundaryMap, VerifiedChain, VerifiedSimplex,
};

use thiserror::Error;

/// Errors that can occur in topological computations
#[derive(Error, Debug, Clone, PartialEq)]
pub enum TopologyError {
    /// Invalid simplex (e.g., duplicate vertices)
    #[error("Invalid simplex: {0}")]
    InvalidSimplex(String),

    /// Dimension mismatch in chain operations
    #[error("Dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },

    /// Empty complex has no valid homology
    #[error("Empty complex")]
    EmptyComplex,

    /// Invalid filtration (non-monotonic)
    #[error("Invalid filtration: {0}")]
    InvalidFiltration(String),

    /// Numerical computation failed
    #[error("Numerical error: {0}")]
    NumericalError(String),

    /// Manifold structure invalid
    #[error("Invalid manifold: {0}")]
    InvalidManifold(String),
}

/// Result type for topology operations
pub type Result<T> = core::result::Result<T, TopologyError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_simplex() {
        let simplex = Simplex::new(vec![0, 1, 2]);
        assert_eq!(simplex.dimension(), 2);
        assert_eq!(simplex.vertices().len(), 3);
    }

    #[test]
    fn test_triangle_complex() {
        let mut complex = SimplicialComplex::new();
        complex.add_simplex(Simplex::new(vec![0, 1, 2]));

        // Should have 1 2-simplex, 3 1-simplices, 3 0-simplices
        assert_eq!(complex.simplex_count(2), 1);
        assert_eq!(complex.simplex_count(1), 3);
        assert_eq!(complex.simplex_count(0), 3);
    }

    #[test]
    fn test_betti_numbers_triangle() {
        let mut complex = SimplicialComplex::new();
        complex.add_simplex(Simplex::new(vec![0, 1, 2]));

        let betti = complex.betti_numbers();
        assert_eq!(betti[0], 1); // One connected component
        assert_eq!(betti[1], 0); // No holes (filled)
    }

    #[test]
    fn test_betti_numbers_circle() {
        // Circle = 3 vertices, 3 edges, no face
        let mut complex = SimplicialComplex::new();
        complex.add_simplex(Simplex::new(vec![0, 1]));
        complex.add_simplex(Simplex::new(vec![1, 2]));
        complex.add_simplex(Simplex::new(vec![2, 0]));

        let betti = complex.betti_numbers();
        assert_eq!(betti[0], 1); // One connected component
        assert_eq!(betti[1], 1); // One hole
    }
}
