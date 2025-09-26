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
//! use amari_enumerative::{ProjectiveSpace, ChowClass};
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

pub mod intersection;
pub mod schubert;
pub mod gromov_witten;
pub mod tropical_curves;
pub mod moduli_space;

// Re-export core types
pub use intersection::{
    IntersectionRing, ChowClass, IntersectionNumber, ProjectiveSpace,
    Grassmannian, Constraint, AlgebraicVariety, IntersectionPoint, MockMultivector,
    QuantumProduct
};
pub use schubert::{SchubertClass, SchubertCalculus, FlagVariety};
pub use gromov_witten::{GromovWittenInvariant, QuantumCohomology, CurveClass as GWCurveClass};
pub use tropical_curves::{TropicalCurve, TropicalIntersection, TropicalPoint, TropicalEdge, TropicalModuliSpace};
pub use moduli_space::{ModuliSpace, CurveClass, TautologicalClass};

/// Error types for enumerative geometry computations
#[derive(Debug, Clone, PartialEq)]
pub enum EnumerativeError {
    /// Invalid dimension for the ambient space
    InvalidDimension(String),
    /// Intersection computation failed
    IntersectionError(String),
    /// Schubert calculus error
    SchubertError(String),
    /// Gromov-Witten invariant computation error
    GromovWittenError(String),
    /// General computational error
    ComputationError(String),
}

impl std::fmt::Display for EnumerativeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EnumerativeError::InvalidDimension(msg) => write!(f, "Invalid dimension: {}", msg),
            EnumerativeError::IntersectionError(msg) => write!(f, "Intersection error: {}", msg),
            EnumerativeError::SchubertError(msg) => write!(f, "Schubert calculus error: {}", msg),
            EnumerativeError::GromovWittenError(msg) => write!(f, "Gromov-Witten error: {}", msg),
            EnumerativeError::ComputationError(msg) => write!(f, "Computation error: {}", msg),
        }
    }
}

impl std::error::Error for EnumerativeError {}

/// Result type for enumerative geometry computations
pub type EnumerativeResult<T> = Result<T, EnumerativeError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_library_compiles() {
        // Basic smoke test to ensure the library compiles
        assert!(true);
    }
}