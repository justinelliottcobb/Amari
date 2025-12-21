//! # Amari Enumerative Geometry
//!
//! This crate provides enumerative geometry capabilities for the Amari mathematical library.
//! It implements intersection theory, Schubert calculus, and tools for counting geometric
//! configurations such as curves, surfaces, and higher-dimensional varieties.
//!
//! ## Features
//!
//! - **Intersection Theory**: Chow rings, intersection multiplicities, and Bézout's theorem
//! - **Schubert Calculus**: Computations on Grassmannians and flag varieties
//! - **Gromov-Witten Theory**: Curve counting and quantum cohomology
//! - **Tropical Geometry**: Tropical curve counting and correspondence theorems
//! - **Moduli Spaces**: Computations on moduli spaces of curves and surfaces
//!
//! ## Usage
//!
//! ```rust
//! use amari_enumerative::{ProjectiveSpace, ChowClass, IntersectionRing};
//!
//! // Create projective 2-space
//! let p2 = ProjectiveSpace::new(2);
//!
//! // Define two curves
//! let cubic = ChowClass::hypersurface(3);
//! let quartic = ChowClass::hypersurface(4);
//!
//! // Compute intersection number (Bézout's theorem)
//! let intersection = p2.intersect(&cubic, &quartic);
//! assert_eq!(intersection.multiplicity(), 12); // 3 * 4 = 12
//! ```

#[cfg(test)]
pub mod comprehensive_tests;
pub mod verified_contracts;

pub mod geometric_algebra;
pub mod gromov_witten;
pub mod higher_genus;
pub mod intersection;
pub mod moduli_space;
pub mod performance;
pub mod schubert;
pub mod tropical_curves;

// Re-export core types
pub use geometric_algebra::{
    quantum_k_theory, signatures, GeometricProjectiveSpace, GeometricSchubertClass,
    GeometricVariety,
};
pub use gromov_witten::{CurveClass as GWCurveClass, GromovWittenInvariant, QuantumCohomology};
pub use higher_genus::{
    AdvancedCurveCounting, DTInvariant, HigherGenusCurve, JacobianData, PTInvariant,
};
pub use intersection::{
    AlgebraicVariety, ChowClass, Constraint, Grassmannian, IntersectionNumber, IntersectionPoint,
    IntersectionRing, MockMultivector, ProjectiveSpace, QuantumProduct,
};
pub use moduli_space::{CurveClass, ModuliSpace, TautologicalClass};
pub use performance::{
    CurveBatchProcessor, FastIntersectionComputer, MemoryPool, SparseSchubertMatrix,
    WasmPerformanceConfig,
};
pub use schubert::{FlagVariety, SchubertCalculus, SchubertClass};
pub use tropical_curves::{
    TropicalCurve, TropicalEdge, TropicalIntersection, TropicalModuliSpace, TropicalPoint,
};

use thiserror::Error;

/// Error types for enumerative geometry computations
#[derive(Error, Debug, Clone, PartialEq)]
pub enum EnumerativeError {
    /// Invalid dimension for the ambient space
    #[error("Invalid dimension: {0}")]
    InvalidDimension(String),

    /// Intersection computation failed
    #[error("Intersection error: {0}")]
    IntersectionError(String),

    /// Schubert calculus error
    #[error("Schubert calculus error: {0}")]
    SchubertError(String),

    /// Gromov-Witten invariant computation error
    #[error("Gromov-Witten error: {0}")]
    GromovWittenError(String),

    /// General computational error
    #[error("Computation error: {0}")]
    ComputationError(String),
}

/// Result type for enumerative geometry computations
pub type EnumerativeResult<T> = Result<T, EnumerativeError>;

// GPU acceleration exports
