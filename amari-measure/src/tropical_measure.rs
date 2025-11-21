//! Tropical measures for extreme value statistics
//!
//! This module provides measures using tropical semiring operations (max-plus, min-plus)
//! instead of standard addition. These are useful for:
//!
//! - **Extreme value statistics**: Finding maxima/minima over sets
//! - **Optimization on measure spaces**: Computing suprema and infima
//! - **Path optimization**: Shortest/longest path problems
//! - **Tropical geometry**: Geometry over tropical semirings
//!
//! # Tropical Semirings
//!
//! ## Max-Plus Semiring (ℝ ∪ {-∞}, ⊕, ⊗)
//!
//! - Addition: a ⊕ b = max(a, b)
//! - Multiplication: a ⊗ b = a + b
//! - Zero: -∞
//! - One: 0
//!
//! ## Min-Plus Semiring (ℝ ∪ {+∞}, ⊕, ⊗)
//!
//! - Addition: a ⊕ b = min(a, b)
//! - Multiplication: a ⊗ b = a + b
//! - Zero: +∞
//! - One: 0
//!
//! # Examples
//!
//! ```
//! use amari_measure::tropical_measure::{MaxPlusMeasure, MinPlusMeasure};
//!
//! // Max-plus measure for finding suprema (need type annotation)
//! let max_measure: MaxPlusMeasure = MaxPlusMeasure::new();
//!
//! // Min-plus measure for finding infima (need type annotation)
//! let min_measure: MinPlusMeasure = MinPlusMeasure::new();
//! ```

use crate::error::Result;
use crate::sigma_algebra::SigmaAlgebra;
use amari_tropical::TropicalNumber;

/// Tropical measure using max-plus semiring
///
/// A max-plus measure μ: Σ → ℝ ∪ {-∞} satisfies:
/// - μ(A ∪ B) = max(μ(A), μ(B)) for disjoint A, B
/// - μ(∅) = -∞
///
/// This is useful for computing suprema over measurable sets.
///
/// # Examples
///
/// ```
/// use amari_measure::tropical_measure::MaxPlusMeasure;
///
/// let measure: MaxPlusMeasure = MaxPlusMeasure::new();
/// ```
pub struct MaxPlusMeasure<Σ = crate::sigma_algebra::LebesgueSigma>
where
    Σ: SigmaAlgebra,
{
    sigma_algebra: Σ,
    /// Function assigning tropical values to sets
    #[allow(clippy::type_complexity)]
    measure_fn: Option<Box<dyn Fn(&Σ::Set) -> TropicalNumber<f64>>>,
}

impl<Σ> MaxPlusMeasure<Σ>
where
    Σ: SigmaAlgebra,
{
    /// Create a new max-plus measure
    pub fn new() -> Self
    where
        Σ: Default,
    {
        Self {
            sigma_algebra: Σ::default(),
            measure_fn: None,
        }
    }

    /// Create a max-plus measure with a custom function
    pub fn with_function<F>(f: F) -> Self
    where
        F: Fn(&Σ::Set) -> TropicalNumber<f64> + 'static,
        Σ: Default,
    {
        Self {
            sigma_algebra: Σ::default(),
            measure_fn: Some(Box::new(f)),
        }
    }

    /// Measure a set, returning the supremum
    ///
    /// Returns μ(A) in the max-plus semiring.
    pub fn measure(&self, set: &Σ::Set) -> Result<TropicalNumber<f64>> {
        if let Some(ref f) = self.measure_fn {
            Ok(f(set))
        } else {
            // Default: return zero element (-∞)
            Ok(TropicalNumber::tropical_zero())
        }
    }

    /// Measure the union of two sets (max operation)
    ///
    /// For disjoint sets: μ(A ∪ B) = max(μ(A), μ(B))
    pub fn measure_union(&self, a: &Σ::Set, b: &Σ::Set) -> Result<TropicalNumber<f64>> {
        let mu_a = self.measure(a)?;
        let mu_b = self.measure(b)?;
        Ok(mu_a + mu_b) // In max-plus, + means max
    }

    /// Get the σ-algebra
    pub fn sigma_algebra(&self) -> &Σ {
        &self.sigma_algebra
    }
}

impl<Σ> Default for MaxPlusMeasure<Σ>
where
    Σ: SigmaAlgebra + Default,
{
    fn default() -> Self {
        Self::new()
    }
}

/// Min-plus number (wrapper around regular number with min operation)
///
/// This wraps f64 values and provides min-plus semiring operations.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct MinPlusNumber(f64);

impl MinPlusNumber {
    /// Create a new min-plus number
    pub fn new(value: f64) -> Self {
        Self(value)
    }

    /// Get the underlying value
    pub fn value(&self) -> f64 {
        self.0
    }

    /// Create the zero element (+∞)
    pub fn zero() -> Self {
        Self(f64::INFINITY)
    }

    /// Create the one element (0)
    pub fn one() -> Self {
        Self(0.0)
    }

    /// Check if this is zero (+∞)
    pub fn is_zero(&self) -> bool {
        self.0.is_infinite() && self.0.is_sign_positive()
    }

    /// Check if this is one (0)
    pub fn is_one(&self) -> bool {
        self.0 == 0.0
    }

    /// Min-plus addition (min operation)
    #[allow(clippy::should_implement_trait)]
    pub fn add(self, other: Self) -> Self {
        Self(self.0.min(other.0))
    }

    /// Min-plus multiplication (addition)
    #[allow(clippy::should_implement_trait)]
    pub fn mul(self, other: Self) -> Self {
        Self(self.0 + other.0)
    }
}

/// Tropical measure using min-plus semiring
///
/// A min-plus measure μ: Σ → ℝ ∪ {+∞} satisfies:
/// - μ(A ∪ B) = min(μ(A), μ(B)) for disjoint A, B
/// - μ(∅) = +∞
///
/// This is useful for computing infima over measurable sets.
///
/// # Examples
///
/// ```
/// use amari_measure::tropical_measure::MinPlusMeasure;
///
/// let measure: MinPlusMeasure = MinPlusMeasure::new();
/// ```
pub struct MinPlusMeasure<Σ = crate::sigma_algebra::LebesgueSigma>
where
    Σ: SigmaAlgebra,
{
    sigma_algebra: Σ,
    /// Function assigning min-plus values to sets
    #[allow(clippy::type_complexity)]
    measure_fn: Option<Box<dyn Fn(&Σ::Set) -> MinPlusNumber>>,
}

impl<Σ> MinPlusMeasure<Σ>
where
    Σ: SigmaAlgebra,
{
    /// Create a new min-plus measure
    pub fn new() -> Self
    where
        Σ: Default,
    {
        Self {
            sigma_algebra: Σ::default(),
            measure_fn: None,
        }
    }

    /// Create a min-plus measure with a custom function
    pub fn with_function<F>(f: F) -> Self
    where
        F: Fn(&Σ::Set) -> MinPlusNumber + 'static,
        Σ: Default,
    {
        Self {
            sigma_algebra: Σ::default(),
            measure_fn: Some(Box::new(f)),
        }
    }

    /// Measure a set, returning the infimum
    ///
    /// Returns μ(A) in the min-plus semiring.
    pub fn measure(&self, set: &Σ::Set) -> Result<MinPlusNumber> {
        if let Some(ref f) = self.measure_fn {
            Ok(f(set))
        } else {
            // Default: return zero element (+∞)
            Ok(MinPlusNumber::zero())
        }
    }

    /// Measure the union of two sets (min operation)
    ///
    /// For disjoint sets: μ(A ∪ B) = min(μ(A), μ(B))
    pub fn measure_union(&self, a: &Σ::Set, b: &Σ::Set) -> Result<MinPlusNumber> {
        let mu_a = self.measure(a)?;
        let mu_b = self.measure(b)?;
        Ok(mu_a.add(mu_b)) // min operation
    }

    /// Get the σ-algebra
    pub fn sigma_algebra(&self) -> &Σ {
        &self.sigma_algebra
    }
}

impl<Σ> Default for MinPlusMeasure<Σ>
where
    Σ: SigmaAlgebra + Default,
{
    fn default() -> Self {
        Self::new()
    }
}

/// Tropical integration: compute supremum of a function over a set
///
/// For a function f: X → ℝ and measurable set A ⊆ X, computes:
/// ∫ᵗʳᵒᵖ f dμ = sup_{x ∈ A} f(x)
///
/// This is the max-plus analog of integration.
///
/// # Arguments
///
/// * `f` - The function to integrate
/// * `set` - The measurable set
/// * `sample_points` - Points to sample (for approximation)
///
/// # Returns
///
/// The supremum of f over the set (in max-plus semiring)
pub fn tropical_supremum_integrate<X, F>(
    f: &F,
    _set: &(),
    sample_points: &[X],
) -> Result<TropicalNumber<f64>>
where
    X: Clone,
    F: Fn(X) -> f64,
{
    if sample_points.is_empty() {
        return Ok(TropicalNumber::tropical_zero()); // -∞
    }

    let mut supremum = TropicalNumber::tropical_zero();
    for point in sample_points {
        let value = f(point.clone());
        let tropical_value = TropicalNumber::new(value);
        supremum = supremum + tropical_value; // max operation
    }

    Ok(supremum)
}

/// Tropical integration: compute infimum of a function over a set
///
/// For a function f: X → ℝ and measurable set A ⊆ X, computes:
/// ∫ᵗʳᵒᵖ f dμ = inf_{x ∈ A} f(x)
///
/// This is the min-plus analog of integration.
///
/// # Arguments
///
/// * `f` - The function to integrate
/// * `set` - The measurable set
/// * `sample_points` - Points to sample (for approximation)
///
/// # Returns
///
/// The infimum of f over the set (in min-plus semiring)
pub fn tropical_infimum_integrate<X, F>(
    f: &F,
    _set: &(),
    sample_points: &[X],
) -> Result<MinPlusNumber>
where
    X: Clone,
    F: Fn(X) -> f64,
{
    if sample_points.is_empty() {
        return Ok(MinPlusNumber::zero()); // +∞
    }

    let mut infimum = MinPlusNumber::zero();
    for point in sample_points {
        let value = f(point.clone());
        let minplus_value = MinPlusNumber::new(value);
        infimum = infimum.add(minplus_value); // min operation
    }

    Ok(infimum)
}

/// Extreme value measure combining max-plus and min-plus
///
/// Tracks both supremum and infimum of measurable sets.
///
/// # Examples
///
/// ```
/// use amari_measure::tropical_measure::ExtremeValueMeasure;
///
/// let measure: ExtremeValueMeasure = ExtremeValueMeasure::new();
/// ```
pub struct ExtremeValueMeasure<Σ = crate::sigma_algebra::LebesgueSigma>
where
    Σ: SigmaAlgebra,
{
    max_measure: MaxPlusMeasure<Σ>,
    min_measure: MinPlusMeasure<Σ>,
}

impl<Σ> ExtremeValueMeasure<Σ>
where
    Σ: SigmaAlgebra + Default,
{
    /// Create a new extreme value measure
    pub fn new() -> Self {
        Self {
            max_measure: MaxPlusMeasure::new(),
            min_measure: MinPlusMeasure::new(),
        }
    }

    /// Create with custom max and min functions
    pub fn with_functions<FMax, FMin>(f_max: FMax, f_min: FMin) -> Self
    where
        FMax: Fn(&Σ::Set) -> TropicalNumber<f64> + 'static,
        FMin: Fn(&Σ::Set) -> MinPlusNumber + 'static,
    {
        Self {
            max_measure: MaxPlusMeasure::with_function(f_max),
            min_measure: MinPlusMeasure::with_function(f_min),
        }
    }

    /// Measure both supremum and infimum of a set
    ///
    /// Returns (max, min) values for the set.
    pub fn measure_extremes(&self, set: &Σ::Set) -> Result<(TropicalNumber<f64>, MinPlusNumber)> {
        let max_val = self.max_measure.measure(set)?;
        let min_val = self.min_measure.measure(set)?;
        Ok((max_val, min_val))
    }

    /// Get the range (max - min) of a set
    ///
    /// Computes the difference between supremum and infimum.
    pub fn measure_range(&self, set: &Σ::Set) -> Result<f64> {
        let (max_val, min_val) = self.measure_extremes(set)?;

        let max_real = max_val.value();
        let min_real = min_val.value();

        // Check for infinities
        if max_real.is_infinite() || min_real.is_infinite() {
            return Ok(f64::INFINITY);
        }

        Ok(max_real - min_real)
    }

    /// Get the max-plus measure
    pub fn max_measure(&self) -> &MaxPlusMeasure<Σ> {
        &self.max_measure
    }

    /// Get the min-plus measure
    pub fn min_measure(&self) -> &MinPlusMeasure<Σ> {
        &self.min_measure
    }
}

impl<Σ> Default for ExtremeValueMeasure<Σ>
where
    Σ: SigmaAlgebra + Default,
{
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_max_plus_measure_creation() {
        let measure: MaxPlusMeasure = MaxPlusMeasure::new();
        let dummy_set = ();

        let result = measure.measure(&dummy_set).unwrap();
        assert!(result.is_zero()); // Should be -∞
    }

    #[test]
    fn test_min_plus_measure_creation() {
        let measure: MinPlusMeasure = MinPlusMeasure::new();
        let dummy_set = ();

        let result = measure.measure(&dummy_set).unwrap();
        assert!(result.is_zero()); // Should be +∞
    }

    #[test]
    fn test_max_plus_with_function() {
        let measure: MaxPlusMeasure =
            MaxPlusMeasure::with_function(|_: &()| TropicalNumber::new(5.0));

        let result = measure.measure(&()).unwrap();
        assert_eq!(result.value(), 5.0);
    }

    #[test]
    fn test_min_plus_with_function() {
        let measure: MinPlusMeasure =
            MinPlusMeasure::with_function(|_: &()| MinPlusNumber::new(3.0));

        let result = measure.measure(&()).unwrap();
        assert_eq!(result.value(), 3.0);
    }

    #[test]
    fn test_max_plus_union() {
        let measure: MaxPlusMeasure =
            MaxPlusMeasure::with_function(|_: &()| TropicalNumber::new(5.0));

        // Union should take max
        let result = measure.measure_union(&(), &()).unwrap();
        assert_eq!(result.value(), 5.0);
    }

    #[test]
    fn test_min_plus_union() {
        let measure: MinPlusMeasure =
            MinPlusMeasure::with_function(|_: &()| MinPlusNumber::new(3.0));

        // Union should take min
        let result = measure.measure_union(&(), &()).unwrap();
        assert_eq!(result.value(), 3.0);
    }

    #[test]
    fn test_tropical_supremum_integration() {
        let f = |x: f64| x * x;
        let sample_points = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        let supremum = tropical_supremum_integrate(&f, &(), &sample_points).unwrap();

        // Supremum should be at x=5: 5² = 25
        assert_eq!(supremum.value(), 25.0);
    }

    #[test]
    fn test_tropical_infimum_integration() {
        let f = |x: f64| x * x;
        let sample_points = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        let infimum = tropical_infimum_integrate(&f, &(), &sample_points).unwrap();

        // Infimum should be at x=1: 1² = 1
        assert_eq!(infimum.value(), 1.0);
    }

    #[test]
    fn test_extreme_value_measure() {
        let f_max = |_: &()| TropicalNumber::new(10.0);
        let f_min = |_: &()| MinPlusNumber::new(2.0);

        let measure: ExtremeValueMeasure = ExtremeValueMeasure::with_functions(f_max, f_min);

        let (max_val, min_val) = measure.measure_extremes(&()).unwrap();
        assert_eq!(max_val.value(), 10.0);
        assert_eq!(min_val.value(), 2.0);

        let range = measure.measure_range(&()).unwrap();
        assert_eq!(range, 8.0); // 10.0 - 2.0
    }

    #[test]
    fn test_extreme_value_measure_default() {
        let measure: ExtremeValueMeasure = ExtremeValueMeasure::new();

        let (max_val, min_val) = measure.measure_extremes(&()).unwrap();
        assert!(max_val.is_zero()); // -∞
        assert!(min_val.is_zero()); // +∞

        let range = measure.measure_range(&()).unwrap();
        assert!(range.is_infinite());
    }

    #[test]
    fn test_minplus_number_operations() {
        let a = MinPlusNumber::new(5.0);
        let b = MinPlusNumber::new(3.0);

        // Min operation
        let sum = a.add(b);
        assert_eq!(sum.value(), 3.0); // min(5, 3) = 3

        // Addition (for multiplication)
        let product = a.mul(b);
        assert_eq!(product.value(), 8.0); // 5 + 3 = 8
    }
}
