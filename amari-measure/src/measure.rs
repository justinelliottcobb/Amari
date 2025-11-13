//! Measures and measure theory
//!
//! A measure μ on a σ-algebra Σ is a function μ: Σ → [0, ∞] satisfying:
//! 1. μ(∅) = 0 (null empty set)
//! 2. For disjoint sets {Aₙ}ₙ₌₁^∞ ⊂ Σ: μ(⋃ₙ Aₙ) = ∑ₙ μ(Aₙ) (countable additivity)
//!
//! # Properties
//!
//! Measures satisfy:
//! - **Monotonicity**: A ⊆ B ⟹ μ(A) ≤ μ(B)
//! - **Subadditivity**: μ(⋃ₙ Aₙ) ≤ ∑ₙ μ(Aₙ) (not necessarily disjoint)
//! - **Continuity from below**: Aₙ ↑ A ⟹ μ(Aₙ) → μ(A)
//! - **Continuity from above**: Aₙ ↓ A and μ(A₁) < ∞ ⟹ μ(Aₙ) → μ(A)
//!
//! # Examples
//!
//! ```
//! use amari_measure::{LebesgueMeasure, CountingMeasure, DiracMeasure};
//!
//! // Lebesgue measure on ℝ³
//! let lebesgue = LebesgueMeasure::new(3);
//!
//! // Counting measure on a discrete space
//! let counting = CountingMeasure::new();
//!
//! // Dirac measure at a point
//! let dirac = DiracMeasure::new(0.0);
//! ```

use crate::error::{MeasureError, Result};
use crate::phantom::*;
use crate::sigma_algebra::SigmaAlgebra;
use core::marker::PhantomData;
use num_traits::Float;

/// Trait for measures on a measurable space
///
/// A measure assigns a non-negative extended real number to each measurable set,
/// satisfying countable additivity.
///
/// # Type Parameters
///
/// - `Σ`: The σ-algebra of measurable sets
/// - `F`: Finiteness property (Finite, SigmaFinite, or Infinite)
/// - `S`: Sign property (Unsigned, Signed, or Complex)
/// - `C`: Completeness property (Complete or Incomplete)
///
/// # Laws
///
/// Implementations must satisfy:
/// 1. `measure(∅) = 0`
/// 2. For disjoint sets {Aₙ}: `measure(⋃ Aₙ) = ∑ measure(Aₙ)`
/// 3. Monotonicity: A ⊆ B ⟹ measure(A) ≤ measure(B)
///
/// # Safety
///
/// The implementation must ensure mathematical correctness of the measure
/// axioms. Violations can lead to undefined measure-theoretic behavior.
pub trait Measure<Σ: SigmaAlgebra, F, S, C>
where
    F: MeasureProperty + 'static,
    S: MeasureProperty + 'static,
    C: MeasureProperty + 'static,
{
    /// The value type of the measure (typically f64 or extended reals)
    type Value;

    /// Compute the measure of a set
    ///
    /// # Arguments
    ///
    /// * `set` - A measurable set from the σ-algebra
    ///
    /// # Returns
    ///
    /// The measure μ(A) of the set, or an error if the set is not measurable.
    ///
    /// # Errors
    ///
    /// Returns `MeasureError::NotMeasurable` if the set is not in the σ-algebra.
    fn measure(&self, set: &Σ::Set) -> Result<Self::Value>;

    /// Get the underlying σ-algebra
    fn sigma_algebra(&self) -> &Σ;

    /// Check if this is a probability measure (μ(X) = 1)
    ///
    /// Default implementation returns false. Probability measures should override.
    fn is_probability_measure(&self) -> bool {
        false
    }

    /// Check if this is a finite measure (μ(X) < ∞)
    ///
    /// Default implementation checks the finiteness phantom type.
    fn is_finite(&self) -> bool {
        // This is determined at compile time by the phantom type F
        core::any::TypeId::of::<F>() == core::any::TypeId::of::<Finite>()
    }

    /// Check if this is a σ-finite measure
    ///
    /// Default implementation checks the phantom types.
    fn is_sigma_finite(&self) -> bool {
        core::any::TypeId::of::<F>() == core::any::TypeId::of::<SigmaFinite>()
            || core::any::TypeId::of::<F>() == core::any::TypeId::of::<Finite>()
    }
}

// ============================================================================
// Counting Measure
// ============================================================================

/// Counting measure on discrete sets
///
/// The counting measure assigns to each set the number of elements it contains:
/// - μ({x₁, ..., xₙ}) = n for finite sets
/// - μ(A) = ∞ for infinite sets
///
/// # Properties
///
/// - **Finiteness**: σ-finite (write ℕ as ⋃ₙ {0, ..., n})
/// - **Sign**: Unsigned (counts are non-negative)
/// - **Completeness**: Complete (power set is complete)
///
/// # Examples
///
/// ```
/// use amari_measure::CountingMeasure;
///
/// let counting = CountingMeasure::new();
/// // On finite sets: μ({1,2,3}) = 3
/// // On infinite sets: μ(ℕ) = ∞
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CountingMeasure {
    _private: (),
}

impl CountingMeasure {
    /// Create a new counting measure
    ///
    /// # Examples
    ///
    /// ```
    /// use amari_measure::CountingMeasure;
    ///
    /// let mu = CountingMeasure::new();
    /// ```
    pub fn new() -> Self {
        Self { _private: () }
    }
}

impl Default for CountingMeasure {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Dirac Measure (Point Mass)
// ============================================================================

/// Dirac measure (point mass) concentrated at a single point
///
/// The Dirac measure δₓ is defined by:
/// - δₓ(A) = 1 if x ∈ A
/// - δₓ(A) = 0 if x ∉ A
///
/// This is the fundamental example of a probability measure.
///
/// # Properties
///
/// - **Finiteness**: Finite (δₓ(X) = 1)
/// - **Sign**: Unsigned
/// - **Completeness**: Complete
/// - **Probability**: Yes (δₓ(X) = 1)
///
/// # Examples
///
/// ```
/// use amari_measure::DiracMeasure;
///
/// // Dirac measure at x = 2.0
/// let delta = DiracMeasure::new(2.0);
/// ```
///
/// # Applications
///
/// - Quantum mechanics: Pure states |ψ⟩⟨ψ|
/// - Statistics: Empirical distributions
/// - Analysis: Test functions and distributions
#[derive(Debug, Clone, PartialEq)]
pub struct DiracMeasure<T> {
    /// The point where the measure is concentrated
    point: T,
}

impl<T> DiracMeasure<T> {
    /// Create a new Dirac measure at a point
    ///
    /// # Arguments
    ///
    /// * `point` - The point x where δₓ is concentrated
    ///
    /// # Examples
    ///
    /// ```
    /// use amari_measure::DiracMeasure;
    ///
    /// let delta = DiracMeasure::new(0.0);
    /// ```
    pub fn new(point: T) -> Self {
        Self { point }
    }

    /// Get the point where this Dirac measure is concentrated
    pub fn point(&self) -> &T {
        &self.point
    }
}

// ============================================================================
// Lebesgue Measure
// ============================================================================

/// Lebesgue measure on ℝⁿ
///
/// The Lebesgue measure is the standard measure of volume in Euclidean space:
/// - On ℝ: λ([a,b]) = b - a (length)
/// - On ℝ²: λ([a,b] × [c,d]) = (b-a)(d-c) (area)
/// - On ℝⁿ: λ generalizes to n-dimensional volume
///
/// # Construction
///
/// Lebesgue measure is constructed as the completion of Borel measure,
/// starting from the pre-measure on rectangles:
///
/// μ([a₁,b₁] × ... × [aₙ,bₙ]) = ∏ᵢ (bᵢ - aᵢ)
///
/// # Properties
///
/// - **Finiteness**: σ-finite but infinite (λ(ℝⁿ) = ∞)
/// - **Sign**: Unsigned
/// - **Completeness**: Complete (all subsets of null sets are measurable)
/// - **Translation Invariant**: λ(A + x) = λ(A) for all x ∈ ℝⁿ
/// - **Rotation Invariant**: λ(RA) = λ(A) for all rotations R
///
/// # Examples
///
/// ```
/// use amari_measure::LebesgueMeasure;
///
/// // Lebesgue measure on ℝ (length)
/// let lambda_1d = LebesgueMeasure::new(1);
///
/// // Lebesgue measure on ℝ³ (volume)
/// let lambda_3d = LebesgueMeasure::new(3);
/// ```
///
/// # Applications
///
/// - Integration theory (Lebesgue integral)
/// - Probability theory (continuous distributions)
/// - Geometric measure theory
/// - Harmonic analysis and Fourier theory
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LebesgueMeasure {
    /// Dimension of the space ℝⁿ
    dimension: usize,
}

impl LebesgueMeasure {
    /// Create Lebesgue measure on ℝⁿ
    ///
    /// # Arguments
    ///
    /// * `dimension` - The dimension n of the space ℝⁿ
    ///
    /// # Examples
    ///
    /// ```
    /// use amari_measure::LebesgueMeasure;
    ///
    /// let lambda = LebesgueMeasure::new(3);
    /// assert_eq!(lambda.dimension(), 3);
    /// ```
    pub fn new(dimension: usize) -> Self {
        Self { dimension }
    }

    /// Get the dimension of the space
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Check if a dimension matches this measure's dimension
    pub fn check_dimension(&self, dim: usize) -> Result<()> {
        if self.dimension == dim {
            Ok(())
        } else {
            Err(MeasureError::dimension_mismatch(self.dimension, dim))
        }
    }
}

// ============================================================================
// Probability Measure
// ============================================================================

/// Probability measure (normalized measure with μ(X) = 1)
///
/// A probability measure is a finite measure μ satisfying μ(X) = 1.
/// This is the foundation of probability theory.
///
/// # Axioms (Kolmogorov)
///
/// 1. **Non-negativity**: μ(A) ≥ 0 for all measurable A
/// 2. **Normalization**: μ(X) = 1
/// 3. **Countable additivity**: For disjoint {Aₙ}: μ(⋃ Aₙ) = ∑ μ(Aₙ)
///
/// # Properties
///
/// - **Finiteness**: Finite (μ(X) = 1)
/// - **Sign**: Unsigned
/// - **Completeness**: Typically complete (completion ensures a.e. statements work)
///
/// # Examples
///
/// ```
/// use amari_measure::ProbabilityMeasure;
///
/// // Uniform distribution on [0,1]
/// // let uniform = ProbabilityMeasure::uniform_interval(0.0, 1.0);
/// ```
///
/// # Applications
///
/// - Probability theory and statistics
/// - Quantum mechanics (Born rule)
/// - Information theory
/// - Stochastic processes
#[derive(Debug, Clone, PartialEq)]
pub struct ProbabilityMeasure<T: Float> {
    /// Total measure of the space (must be 1.0)
    total: T,
    /// Phantom data for probability property
    _phantom: PhantomData<(Finite, Unsigned, Complete)>,
}

impl<T: Float> ProbabilityMeasure<T> {
    /// Create a new probability measure
    ///
    /// # Arguments
    ///
    /// * `total` - Total measure of the space (should be 1.0)
    ///
    /// # Returns
    ///
    /// A probability measure, or an error if total ≠ 1.0
    ///
    /// # Errors
    ///
    /// Returns `MeasureError::InvalidProbabilityMeasure` if total ≠ 1.0 within tolerance.
    ///
    /// # Examples
    ///
    /// ```
    /// use amari_measure::ProbabilityMeasure;
    ///
    /// let prob = ProbabilityMeasure::new(1.0).unwrap();
    /// ```
    pub fn new(total: T) -> Result<Self> {
        let one = T::one();
        let epsilon = T::from(1e-10).unwrap();

        if (total - one).abs() > epsilon {
            return Err(MeasureError::invalid_probability(total.to_f64().unwrap()));
        }

        Ok(Self {
            total,
            _phantom: PhantomData,
        })
    }

    /// Get the total measure (should always be 1.0)
    pub fn total(&self) -> T {
        self.total
    }

    /// Normalize a finite measure to a probability measure
    ///
    /// Given a finite measure μ with μ(X) = c > 0, returns (1/c)μ.
    ///
    /// # Arguments
    ///
    /// * `total_measure` - The total measure of the space
    ///
    /// # Errors
    ///
    /// Returns error if total_measure ≤ 0.
    pub fn normalize(total_measure: T) -> Result<T> {
        if total_measure <= T::zero() {
            return Err(MeasureError::computation(
                "Cannot normalize: total measure must be positive",
            ));
        }
        Ok(T::one() / total_measure)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_counting_measure_creation() {
        let counting = CountingMeasure::new();
        let counting_default = CountingMeasure::default();
        assert_eq!(counting, counting_default);
    }

    #[test]
    fn test_dirac_measure_creation() {
        let delta = DiracMeasure::new(std::f64::consts::PI);
        assert_eq!(delta.point(), &std::f64::consts::PI);

        let delta_int = DiracMeasure::new(42);
        assert_eq!(delta_int.point(), &42);
    }

    #[test]
    fn test_lebesgue_measure_creation() {
        let lambda_1d = LebesgueMeasure::new(1);
        assert_eq!(lambda_1d.dimension(), 1);

        let lambda_3d = LebesgueMeasure::new(3);
        assert_eq!(lambda_3d.dimension(), 3);
    }

    #[test]
    fn test_lebesgue_dimension_check() {
        let lambda = LebesgueMeasure::new(3);

        assert!(lambda.check_dimension(3).is_ok());

        let err = lambda.check_dimension(2).unwrap_err();
        assert!(matches!(err, MeasureError::DimensionMismatch { .. }));
    }

    #[test]
    fn test_probability_measure_creation() {
        // Valid probability measure
        let prob = ProbabilityMeasure::new(1.0);
        assert!(prob.is_ok());
        assert_eq!(prob.unwrap().total(), 1.0);

        // Invalid: total ≠ 1
        let invalid = ProbabilityMeasure::new(0.5);
        assert!(invalid.is_err());
        assert!(matches!(
            invalid.unwrap_err(),
            MeasureError::InvalidProbabilityMeasure { .. }
        ));
    }

    #[test]
    fn test_probability_measure_normalization() {
        // Normalize a measure with total = 2.0
        let scale = ProbabilityMeasure::normalize(2.0).unwrap();
        assert_eq!(scale, 0.5);

        // Normalize a measure with total = 0.25
        let scale2 = ProbabilityMeasure::normalize(0.25).unwrap();
        assert_eq!(scale2, 4.0);

        // Cannot normalize zero measure
        let err = ProbabilityMeasure::<f64>::normalize(0.0);
        assert!(err.is_err());
    }

    #[test]
    fn test_measure_types_are_small() {
        use core::mem::size_of;

        // Counting measure is zero-sized
        assert_eq!(size_of::<CountingMeasure>(), 0);

        // Dirac measure stores only the point
        assert_eq!(size_of::<DiracMeasure<f64>>(), size_of::<f64>());

        // Lebesgue measure stores only dimension
        assert_eq!(size_of::<LebesgueMeasure>(), size_of::<usize>());

        // Probability measure stores total + PhantomData (just the float)
        assert_eq!(size_of::<ProbabilityMeasure<f64>>(), size_of::<f64>());
    }

    #[test]
    fn test_probability_measure_tolerance() {
        // Should accept values very close to 1.0
        let almost_one = 1.0 + 1e-12;
        let prob = ProbabilityMeasure::new(almost_one);
        assert!(prob.is_ok());

        // Should reject values noticeably different from 1.0
        let not_one = 1.0 + 1e-8;
        let prob2 = ProbabilityMeasure::new(not_one);
        assert!(prob2.is_err());
    }

    #[test]
    fn test_dirac_measure_clone() {
        let delta = DiracMeasure::new(2.5);
        let delta_clone = delta.clone();
        assert_eq!(delta, delta_clone);
    }

    #[test]
    fn test_lebesgue_measure_equality() {
        let lambda1 = LebesgueMeasure::new(2);
        let lambda2 = LebesgueMeasure::new(2);
        let lambda3 = LebesgueMeasure::new(3);

        assert_eq!(lambda1, lambda2);
        assert_ne!(lambda1, lambda3);
    }
}
