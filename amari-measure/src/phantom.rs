//! Phantom type system for compile-time measure properties
//!
//! This module provides zero-cost phantom types that encode measure properties
//! at compile time, enabling the type system to enforce mathematical invariants
//! and prevent invalid operations.
//!
//! # Design Philosophy
//!
//! Following the Amari phantom types methodology:
//! - Zero runtime cost (all markers are zero-sized types)
//! - Compile-time verification of measure properties
//! - Type-level encoding of mathematical structure
//! - Prevention of invalid measure operations
//!
//! # Property Categories
//!
//! ## Finiteness Properties
//!
//! - `Finite`: μ(X) < ∞ (total measure is finite)
//! - `SigmaFinite`: X = ⋃ Xₙ where μ(Xₙ) < ∞ (countable union of finite sets)
//! - `Infinite`: μ(X) = ∞ (infinite total measure)
//!
//! ## Sign Properties
//!
//! - `Unsigned`: μ(A) ≥ 0 for all measurable A (standard measures)
//! - `Signed`: μ(A) ∈ ℝ (Jordan decomposition μ = μ⁺ - μ⁻)
//! - `Complex`: μ(A) ∈ ℂ (complex-valued measures)
//!
//! ## Completeness Properties
//!
//! - `Complete`: All subsets of null sets are measurable
//! - `Incomplete`: May have non-measurable subsets of null sets
//!
//! # Examples
//!
//! ```
//! use amari_measure::{Measure, LebesgueMeasure, Finite, Unsigned, Complete};
//!
//! // Lebesgue measure on [0,1] is finite, unsigned, and complete
//! // let mu: Measure<Finite, Unsigned, Complete> = LebesgueMeasure::on_interval(0.0, 1.0);
//!
//! // Counting measure on ℕ is sigma-finite, unsigned, complete
//! // let nu: Measure<SigmaFinite, Unsigned, Complete> = CountingMeasure::natural_numbers();
//! ```

/// Marker trait for all measure property types
///
/// This trait is sealed and cannot be implemented outside this module,
/// ensuring type safety of the phantom type system.
pub trait MeasureProperty: private::Sealed {}

// Sealed trait pattern to prevent external implementations
mod private {
    pub trait Sealed {}
}

// ============================================================================
// Finiteness Properties
// ============================================================================

/// Marker type for finite measures: μ(X) < ∞
///
/// A measure μ is finite if the measure of the entire space is finite.
///
/// # Examples
///
/// - Lebesgue measure on bounded intervals [a,b] ⊂ ℝ
/// - Probability measures (with μ(X) = 1)
/// - Dirac measures δₓ
/// - Finite counting measures on finite sets
///
/// # Properties
///
/// - All finite measures are σ-finite
/// - Finite measures allow simpler integration theory
/// - Many convergence theorems simplify for finite measures
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Finite;

impl MeasureProperty for Finite {}
impl private::Sealed for Finite {}

/// Marker type for σ-finite measures: X = ⋃ Xₙ where μ(Xₙ) < ∞
///
/// A measure μ is σ-finite if the space X can be written as a countable union
/// of measurable sets with finite measure.
///
/// # Examples
///
/// - Lebesgue measure on ℝⁿ (write ℝⁿ = ⋃ Bₙ where Bₙ are balls)
/// - Counting measure on countable sets
/// - Product measures of σ-finite measures
///
/// # Properties
///
/// - Crucial for Radon-Nikodym theorem
/// - Necessary for Fubini's theorem
/// - Allows construction of densities
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SigmaFinite;

impl MeasureProperty for SigmaFinite {}
impl private::Sealed for SigmaFinite {}

/// Marker type for infinite measures: μ(X) = ∞
///
/// A measure μ is infinite if the measure of the entire space is infinite.
///
/// # Examples
///
/// - Lebesgue measure on ℝⁿ
/// - Counting measure on uncountable sets
/// - Hausdorff measure of dimension d on ℝⁿ when d < n
///
/// # Properties
///
/// - May still be σ-finite (e.g., Lebesgue measure)
/// - Requires care in integration theory
/// - Some theorems require finiteness assumptions
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Infinite;

impl MeasureProperty for Infinite {}
impl private::Sealed for Infinite {}

// ============================================================================
// Sign Properties
// ============================================================================

/// Marker type for unsigned (non-negative) measures: μ(A) ≥ 0
///
/// Standard measures that assign non-negative values to all measurable sets.
///
/// # Examples
///
/// - Lebesgue measure
/// - Probability measures
/// - Counting measure
/// - All geometric measures with non-negative scalars
///
/// # Properties
///
/// - Satisfies countable additivity
/// - Monotonicity: A ⊆ B ⟹ μ(A) ≤ μ(B)
/// - Subadditivity: μ(⋃ Aₙ) ≤ ∑ μ(Aₙ)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Unsigned;

impl MeasureProperty for Unsigned {}
impl private::Sealed for Unsigned {}

/// Marker type for signed measures: μ(A) ∈ ℝ
///
/// Measures that can take both positive and negative values, defined via
/// Jordan decomposition μ = μ⁺ - μ⁻.
///
/// # Examples
///
/// - Charge distributions in physics (positive and negative charges)
/// - Difference of two unsigned measures
/// - Radon-Nikodym derivatives of signed measures
///
/// # Properties
///
/// - Has Jordan decomposition μ = μ⁺ - μ⁻ (unique minimal)
/// - Has Hahn decomposition X = P ⊔ N where μ(A∩P) ≥ 0, μ(A∩N) ≤ 0
/// - Total variation |μ| = μ⁺ + μ⁻ is an unsigned measure
/// - Radon-Nikodym theorem extends to signed measures
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Signed;

impl MeasureProperty for Signed {}
impl private::Sealed for Signed {}

/// Marker type for complex measures: μ(A) ∈ ℂ
///
/// Measures taking complex values, important in quantum mechanics and
/// functional analysis.
///
/// # Examples
///
/// - Complex-valued probability amplitudes (quantum mechanics)
/// - Fourier transforms of signed/complex measures
/// - Spectral measures in operator theory
///
/// # Properties
///
/// - Can decompose as μ = μ₁ + iμ₂ where μ₁, μ₂ are signed measures
/// - Total variation |μ| defined via supremum over partitions
/// - Radon-Nikodym theorem extends to complex measures
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Complex;

impl MeasureProperty for Complex {}
impl private::Sealed for Complex {}

// ============================================================================
// Completeness Properties
// ============================================================================

/// Marker type for complete measures
///
/// A measure μ on σ-algebra Σ is complete if every subset of a null set
/// is measurable (and has measure zero).
///
/// # Completion Process
///
/// Given (X, Σ, μ), the completion Σ̄ consists of sets of the form A ∪ N where:
/// - A ∈ Σ
/// - N ⊆ B for some B ∈ Σ with μ(B) = 0
///
/// # Examples
///
/// - Lebesgue measure on ℝⁿ (completion of Borel measure)
/// - Completed probability spaces
/// - Haar measure on locally compact groups (completed)
///
/// # Properties
///
/// - Ensures all "negligible" sets are measurable
/// - Essential for Lebesgue integration
/// - Simplifies almost-everywhere statements
/// - Completion preserves σ-finiteness
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Complete;

impl MeasureProperty for Complete {}
impl private::Sealed for Complete {}

/// Marker type for incomplete measures
///
/// Measures where some subsets of null sets may not be measurable.
///
/// # Examples
///
/// - Borel measure on ℝⁿ (before completion to Lebesgue)
/// - Counting measure restricted to Borel sets
/// - Product σ-algebras (may not be complete even if factors are)
///
/// # Properties
///
/// - Can always be completed to obtain a complete measure
/// - Completion is minimal: adds only subsets of null sets
/// - Original and completed measures agree on original σ-algebra
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Incomplete;

impl MeasureProperty for Incomplete {}
impl private::Sealed for Incomplete {}

// ============================================================================
// Property Combinations
// ============================================================================

/// Type alias for probability measures
///
/// Probability measures are finite, unsigned, and typically complete.
#[allow(dead_code)]
pub type ProbabilityProperty = (Finite, Unsigned, Complete);

/// Type alias for Lebesgue measure properties
///
/// Lebesgue measure on ℝⁿ is σ-finite (infinite if n≥1), unsigned, complete.
#[allow(dead_code)]
pub type LebesgueProperty = (SigmaFinite, Unsigned, Complete);

/// Type alias for Borel measure properties
///
/// Borel measures are typically σ-finite, unsigned, but incomplete.
#[allow(dead_code)]
pub type BorelProperty = (SigmaFinite, Unsigned, Incomplete);

/// Type alias for counting measure properties
///
/// Counting measures are σ-finite, unsigned, and complete.
#[allow(dead_code)]
pub type CountingProperty = (SigmaFinite, Unsigned, Complete);

#[cfg(test)]
mod tests {
    use super::*;
    use core::marker::PhantomData;

    #[test]
    fn test_phantom_types_are_zero_sized() {
        use core::mem::size_of;

        assert_eq!(size_of::<Finite>(), 0);
        assert_eq!(size_of::<SigmaFinite>(), 0);
        assert_eq!(size_of::<Infinite>(), 0);
        assert_eq!(size_of::<Unsigned>(), 0);
        assert_eq!(size_of::<Signed>(), 0);
        assert_eq!(size_of::<Complex>(), 0);
        assert_eq!(size_of::<Complete>(), 0);
        assert_eq!(size_of::<Incomplete>(), 0);
    }

    #[test]
    fn test_phantom_types_implement_traits() {
        // Finiteness markers
        fn is_measure_property<T: MeasureProperty>() {}

        is_measure_property::<Finite>();
        is_measure_property::<SigmaFinite>();
        is_measure_property::<Infinite>();

        // Sign markers
        is_measure_property::<Unsigned>();
        is_measure_property::<Signed>();
        is_measure_property::<Complex>();

        // Completeness markers
        is_measure_property::<Complete>();
        is_measure_property::<Incomplete>();
    }

    #[test]
    fn test_phantom_data_integration() {
        // Verify PhantomData works with our phantom types
        struct TestMeasure<F, S, C> {
            _finiteness: PhantomData<F>,
            _sign: PhantomData<S>,
            _completeness: PhantomData<C>,
        }

        let _: TestMeasure<Finite, Unsigned, Complete> = TestMeasure {
            _finiteness: PhantomData,
            _sign: PhantomData,
            _completeness: PhantomData,
        };

        let _: TestMeasure<SigmaFinite, Signed, Incomplete> = TestMeasure {
            _finiteness: PhantomData,
            _sign: PhantomData,
            _completeness: PhantomData,
        };
    }
}
