//! Measure-theoretic foundations for geometric algebra
//!
//! This crate provides measure theory, integration, and probability theory
//! for multivector spaces in Clifford algebras, with advanced features including:
//!
//! - **Parametric densities** with automatic differentiation
//! - **Numerical integration** algorithms (Monte Carlo, adaptive quadrature, etc.)
//! - **Geometric measures** on multivector spaces
//! - **Fisher-Riemannian geometry** for statistical manifolds
//! - **Radon-Nikodym derivatives** with dual number differentiation
//! - **Tropical measures** for extreme value statistics
//! - **Type-safe convergence theorems** with compile-time verification
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
//! # Quick Start Examples
//!
//! ## 1. Parametric Densities with Automatic Differentiation
//!
//! ```
//! use amari_measure::parametric::families;
//!
//! // Create a Gaussian density family
//! let gaussian = families::gaussian();
//!
//! // Evaluate density and gradient at x=1.5, params=[μ=1.0, σ=2.0]
//! let (value, gradient) = gaussian.evaluate_with_gradient(1.5, &[1.0, 2.0]).unwrap();
//!
//! // Compute Fisher information matrix from data
//! let data = vec![1.2, 1.5, 1.8];
//! let fisher = gaussian.fisher_information(&data, &[1.0, 2.0]).unwrap();
//! ```
//!
//! ## 2. Numerical Integration
//!
//! ```
//! use amari_measure::{monte_carlo_integrate, simpson_integrate};
//!
//! // Monte Carlo integration of f(x) = x² over [0, 1]
//! let mc_result = monte_carlo_integrate(&|x| x * x, 0.0, 1.0, 10000).unwrap();
//! // Result ≈ 1/3
//!
//! // Simpson's rule (higher accuracy)
//! let simpson_result = simpson_integrate(&|x| x * x, 0.0, 1.0, 100).unwrap();
//! ```
//!
//! ## 3. Geometric Measures on Multivector Spaces
//!
//! ```
//! use amari_measure::multivector_measure::GradeDecomposedMeasure;
//! use amari_core::Multivector;
//!
//! // Create a 3D geometric measure (signature (3,0,0))
//! let measure = GradeDecomposedMeasure::<3, 0, 0>::new();
//!
//! // Set from a multivector
//! let mv = Multivector::<3, 0, 0>::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
//! let measure = GradeDecomposedMeasure::<3, 0, 0>::from_multivector(&mv);
//! ```
//!
//! ## 4. Fisher-Riemannian Geometry
//!
//! ```
//! use amari_measure::fisher_measure::FisherMeasure;
//! use amari_measure::parametric::families;
//!
//! // Create Fisher measure from Gaussian density
//! let gaussian = families::gaussian();
//! let fisher_measure = FisherMeasure::from_density(gaussian);
//!
//! // Compute volume element at parameter point
//! let params = vec![0.0, 1.0]; // μ=0, σ=1
//! let data = vec![-1.0, 0.0, 1.0];
//! let volume = fisher_measure.volume_element(&params, &data).unwrap();
//! ```
//!
//! ## 5. Radon-Nikodym Derivatives with Dual Numbers
//!
//! ```
//! use amari_measure::radon_nikodym_dual::DualRadonNikodym;
//!
//! // Define a Gaussian density as Radon-Nikodym derivative
//! let rn = DualRadonNikodym::new(|x: f64, theta: f64| {
//!     let diff = x - theta;
//!     (-0.5 * diff * diff).exp()
//! });
//!
//! // Compute score function: ∂/∂θ log p(x|θ)
//! let score = rn.score(1.0, 0.0).unwrap();
//!
//! // Compute Fisher information from data
//! let data = vec![-1.0, 0.0, 1.0];
//! let fisher = rn.fisher_information(&data, 0.0).unwrap();
//! ```
//!
//! ## 6. Tropical Measures for Extreme Values
//!
//! ```
//! use amari_measure::tropical_measure::{MaxPlusMeasure, tropical_supremum_integrate};
//!
//! // Create max-plus measure for finding suprema (need type annotation)
//! let max_measure: MaxPlusMeasure = MaxPlusMeasure::new();
//!
//! // Find supremum via tropical integration
//! let f = |x: f64| x * x;
//! let sample_points = vec![1.0, 2.0, 3.0, 4.0, 5.0];
//! let supremum = tropical_supremum_integrate(&f, &(), &sample_points).unwrap();
//! // supremum.value() == 25.0 (at x=5)
//! ```
//!
//! ## 7. Type-Safe Convergence Theorems
//!
//! ```
//! use amari_measure::type_safe_convergence::{
//!     FunctionSequence, MonotoneIncreasing, NonNegative,
//!     apply_monotone_convergence_theorem
//! };
//!
//! // Create a monotone increasing, non-negative sequence
//! let seq: FunctionSequence<f64, (MonotoneIncreasing, NonNegative)> =
//!     FunctionSequence::from_monotone_nonnegative_closures(vec![
//!         |x: f64| x,
//!         |x: f64| x + 1.0,
//!         |x: f64| x + 2.0,
//!     ]);
//!
//! // Apply MCT - compiles because types are correct
//! let result = apply_monotone_convergence_theorem(&seq).unwrap();
//!
//! // Would NOT compile with wrong property type:
//! // let wrong_seq: FunctionSequence<f64, NonNegative> = ...
//! // apply_monotone_convergence_theorem(&wrong_seq); // ❌ compile error
//! ```
//!
//! # Integration Example: Statistical Inference
//!
//! Here's how to combine multiple features for statistical inference:
//!
//! ```
//! use amari_measure::parametric::families;
//! use amari_measure::fisher_measure::FisherMeasure;
//! use amari_measure::monte_carlo_integrate;
//!
//! // 1. Define parametric density family
//! let gaussian = families::gaussian();
//!
//! // 2. Collect data
//! let data = vec![0.9, 1.1, 1.2, 0.8, 1.0];
//!
//! // 3. Compute Fisher information (for Cramér-Rao bound)
//! let params = vec![1.0, 0.5]; // μ=1.0, σ=0.5
//! let fisher_info = gaussian.fisher_information(&data, &params).unwrap();
//!
//! // 4. Create Fisher measure for geometric inference
//! let fisher_measure = FisherMeasure::from_density(gaussian);
//!
//! // 5. Compute volume element (for Jeffreys prior)
//! let volume = fisher_measure.volume_element(&params, &data).unwrap();
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

// Grade-decomposed measures with actual Multivector operations
pub mod multivector_measure;
pub use multivector_measure::{
    geometric_product_measures, inner_product_measures, wedge_product_measures,
    GradeDecomposedMeasure,
};

// Lebesgue integration
mod integration;
pub use integration::{
    integrate, integrate_simple, Integrable, Integrator, MeasurableFunction, SimpleFunction,
};

// Radon-Nikodym derivatives and densities
mod density;
pub use density::{absolutely_continuous, singular, Density, LebesgueDecomposition, RadonNikodym};

// Radon-Nikodym with automatic differentiation using dual numbers
pub mod radon_nikodym_dual;
pub use radon_nikodym_dual::{kl_divergence, DualRadonNikodym};

// Parametric density families with automatic differentiation
pub mod parametric;
pub use parametric::ParametricDensity;

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

// Type-safe convergence theorems with compile-time verification
pub mod type_safe_convergence;
pub use type_safe_convergence::{
    apply_dominated_convergence_theorem, apply_fatou_lemma, apply_monotone_convergence_theorem,
    Dominated, FunctionSequence, MonotoneDecreasing, MonotoneIncreasing, NonNegative,
    PointwiseConvergent, TypeSafeDominatedConvergenceResult, TypeSafeFatouResult,
    TypeSafeMonotoneConvergenceResult,
};

// Tropical measures for extreme value statistics
pub mod tropical_measure;
pub use tropical_measure::{
    tropical_infimum_integrate, tropical_supremum_integrate, ExtremeValueMeasure, MaxPlusMeasure,
    MinPlusMeasure,
};

// Fisher-Riemannian measures on statistical manifolds
pub mod fisher_measure;
pub use fisher_measure::{FisherMeasure, FisherStatisticalManifold};

// Numerical integration algorithms
pub mod numerical_integration;
pub use numerical_integration::{
    adaptive_quadrature, monte_carlo_integrate, multidim_monte_carlo, simpson_integrate,
    trapezoidal_integrate, IntegrationResult,
};

// Signed and complex measures
mod signed_measure;
pub use signed_measure::{
    hahn_decomposition, jordan_decomposition, ComplexMeasure, HahnDecomposition,
    JordanDecomposition, SignedMeasure, TotalVariation,
};

#[cfg(test)]
mod tests {
    #[test]
    fn test_crate_compiles() {
        // Basic smoke test to ensure the crate structure is valid
        // Compile successfully means this test passes
    }
}
