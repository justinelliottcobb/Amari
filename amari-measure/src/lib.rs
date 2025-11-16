//! Measure-theoretic foundations for geometric algebra
//!
//! This crate provides measure theory, integration, and probability theory
//! for multivector spaces in Clifford algebras.
//!
//! # Core Concepts
//!
//! - **σ-algebras**: Collections of measurable sets closed under complements and countable unions
//! - **Measures**: Functions assigning sizes to measurable sets (countably additive)
//! - **Geometric Measures**: Multivector-valued measures extending real measures to Cl(p,q,r)
//! - **Lebesgue Integration**: Integration of measurable functions
//! - **Radon-Nikodym Derivatives**: Densities dν/dμ for absolutely continuous measures
//! - **Phantom Types**: Compile-time verification of measure properties
//!
//! # Examples
//!
//! ```
//! use amari_measure::{Measure, LebesgueMeasure};
//!
//! // Create Lebesgue measure on ℝ³
//! let mu = LebesgueMeasure::new(3);
//!
//! // Measure the volume of a cube [0,1]³
//! // let volume = mu.measure(&Cube::unit(3));
//! // assert_eq!(volume, 1.0);
//! ```

#![cfg_attr(not(feature = "std"), no_std)]
#![warn(missing_docs)]
#![warn(clippy::all)]

#[cfg(feature = "formal-verification")]
extern crate creusot_contracts;

// Re-export core types
pub use amari_core::{Multivector, Scalar};

// Error types
mod error;
pub use error::{MeasureError, Result};

// Phantom type system for compile-time measure properties
mod phantom;
pub use phantom::{
    // Completeness markers
    Complete,
    Complex,
    // Finiteness markers
    Finite,
    Incomplete,
    Infinite,
    // Property marker trait
    MeasureProperty,
    SigmaFinite,
    Signed,
    // Sign markers
    Unsigned,
};

// σ-algebra trait and implementations
mod sigma_algebra;
pub use sigma_algebra::{BorelSigma, LebesgueSigma, PowerSet, SigmaAlgebra, TrivialSigma};

// Measure trait and basic measures
mod measure;
pub use measure::{CountingMeasure, DiracMeasure, LebesgueMeasure, Measure, ProbabilityMeasure};

// Geometric measures (multivector-valued)
mod geometric_measure;
pub use geometric_measure::{
    geometric_lebesgue_measure, GeometricDensity, GeometricLebesgueMeasure, GeometricMeasure,
};

// Lebesgue integration
mod integration;
pub use integration::{
    integrate, integrate_simple, Integrable, Integrator, MeasurableFunction, SimpleFunction,
};

// Radon-Nikodym derivatives and densities
mod density;
pub use density::{absolutely_continuous, singular, Density, LebesgueDecomposition, RadonNikodym};

// TODO: Implement remaining modules
// These modules are planned but not yet implemented

// Pushforward and pullback of measures
mod pushforward;
pub use pushforward::{change_of_variables, pullback, pushforward, Pullback, Pushforward};
// Note: pushforward::MeasurableFunction is available via module path
// (different trait from integration::MeasurableFunction)

// Product measures and Fubini's theorem
mod product;
pub use product::{fubini, FubiniIntegrator, ProductMeasure, ProductSigma};

// Convergence theorems
mod convergence;
pub use convergence::{
    dominated_convergence, fatou_lemma, monotone_convergence, DominatedConvergenceResult,
    FatouResult, MonotoneConvergenceResult,
};

// // Signed and complex measures
// mod signed_measure;
// pub use signed_measure::{
//     SignedMeasure,
//     ComplexMeasure,
//     TotalVariation,
//     jordan_decomposition,
//     hahn_decomposition,
// };

#[cfg(test)]
mod tests {
    #[test]
    fn test_crate_compiles() {
        // Basic smoke test to ensure the crate structure is valid
        // Compile successfully means this test passes
    }
}
