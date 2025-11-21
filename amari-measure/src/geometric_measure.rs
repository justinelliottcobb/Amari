//! Geometric measures - multivector-valued measures on Clifford algebras
//!
//! This module extends classical measure theory to geometric algebra by defining
//! measures that assign multivectors (elements of Cl(p,q,r)) to measurable sets.
//!
//! # Core Concept
//!
//! A geometric measure μ: Σ → Cl(p,q,r) assigns a multivector to each measurable set:
//!
//! μ(A) = μ₀(A) + μ₁(A)e₁ + μ₂(A)e₂ + ... + μ₁₂(A)e₁e₂ + ...
//!
//! where each coefficient μᵢ(A) is a real-valued measure.
//!
//! # Properties
//!
//! - **Grade decomposition**: μ = μ₀ + μ₁ + μ₂ + ... where μₖ measures grade-k components
//! - **Countable additivity**: μ(⋃ Aₙ) = ∑ μ(Aₙ) for disjoint sets (as multivector sum)
//! - **Non-negativity**: Only defined for specific components (scalar part usually)
//!
//! # Examples
//!
//! ```
//! // TODO: Add examples once integration with amari-core Multivector is complete
//! ```

use crate::measure::LebesgueMeasure;
use crate::phantom::*;
use crate::sigma_algebra::SigmaAlgebra;
use core::marker::PhantomData;

/// Geometric measure assigning multivectors to measurable sets
///
/// A geometric measure μ: Σ → Cl(p,q,r) where:
///  - Σ is a σ-algebra of measurable sets
///  - Cl(p,q,r) is a Clifford algebra with signature (p,q,r)
///  - μ satisfies countable additivity in the multivector space
///
/// # Type Parameters
///
/// - `Σ`: The σ-algebra type
/// - `F`: Finiteness property (Finite, SigmaFinite, Infinite)
/// - `S`: Sign property (Unsigned, Signed, Complex)
/// - `C`: Completeness property (Complete, Incomplete)
///
/// # Mathematical Foundation
///
/// The geometric measure decomposes by grade:
/// μ(A) = μ⟨0⟩(A) + μ⟨1⟩(A) + μ⟨2⟩(A) + ...
///
/// where μ⟨k⟩(A) measures the grade-k component.
///
/// # Implementation Note
///
/// This foundational implementation establishes the structure for geometric measures.
/// Full integration with amari-core's Multivector type will be completed as
/// the amari-core API stabilizes.
#[derive(Clone)]
pub struct GeometricMeasure<Σ, F, S, C>
where
    Σ: SigmaAlgebra,
    F: MeasureProperty + 'static,
    S: MeasureProperty + 'static,
    C: MeasureProperty + 'static,
{
    /// Dimension of the underlying space
    dimension: usize,

    /// Reference σ-algebra
    sigma_algebra: Σ,

    /// Phantom type markers for measure properties
    _phantom: PhantomData<(F, S, C)>,
}

impl<Σ, F, S, C> GeometricMeasure<Σ, F, S, C>
where
    Σ: SigmaAlgebra,
    F: MeasureProperty + 'static,
    S: MeasureProperty + 'static,
    C: MeasureProperty + 'static,
{
    /// Create a new geometric measure
    ///
    /// # Arguments
    ///
    /// * `sigma_algebra` - The σ-algebra of measurable sets
    /// * `dimension` - The dimension of the underlying space
    ///
    /// # Examples
    ///
    /// ```
    /// use amari_measure::{GeometricMeasure, BorelSigma, SigmaFinite, Unsigned, Complete};
    ///
    /// let borel = BorelSigma::new(3);
    /// let mu: GeometricMeasure<_, SigmaFinite, Unsigned, Complete> =
    ///     GeometricMeasure::new(borel, 3);
    /// ```
    pub fn new(sigma_algebra: Σ, dimension: usize) -> Self {
        Self {
            dimension,
            sigma_algebra,
            _phantom: PhantomData,
        }
    }

    /// Get the dimension of the underlying space
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Get a reference to the underlying σ-algebra
    pub fn sigma_algebra(&self) -> &Σ {
        &self.sigma_algebra
    }

    /// Number of grades in the geometric algebra
    ///
    /// For a Clifford algebra of dimension n, there are n+1 grades (0 to n).
    pub fn num_grades(&self) -> usize {
        self.dimension + 1
    }

    /// Total number of basis elements (multivector components)
    ///
    /// For dimension n, this is 2^n.
    pub fn num_components(&self) -> usize {
        2_usize.pow(self.dimension as u32)
    }
}

/// Geometric Lebesgue measure on ℝⁿ
///
/// This is a geometric measure where each grade component uses the standard
/// Lebesgue measure. It extends n-dimensional volume to multivector-valued measure.
///
/// # Properties
///
/// - Each grade inherits σ-finiteness from Lebesgue measure
/// - Translation invariant (like scalar Lebesgue measure)
/// - Rotation invariant in the appropriate sense
///
/// # Examples
///
/// ```
/// use amari_measure::geometric_lebesgue_measure;
///
/// // 3D geometric Lebesgue measure
/// let mu = geometric_lebesgue_measure(3);
/// assert_eq!(mu.dimension(), 3);
/// assert_eq!(mu.num_grades(), 4);  // grades 0,1,2,3
/// assert_eq!(mu.num_components(), 8);  // 2^3 = 8 basis elements
/// ```
pub type GeometricLebesgueMeasure =
    GeometricMeasure<crate::sigma_algebra::LebesgueSigma, SigmaFinite, Unsigned, Complete>;

/// Create a geometric Lebesgue measure on ℝⁿ
///
/// Convenience function for creating a geometric Lebesgue measure without
/// explicitly constructing the σ-algebra.
///
/// # Arguments
///
/// * `dimension` - The dimension n of the space ℝⁿ
///
/// # Examples
///
/// ```
/// use amari_measure::geometric_lebesgue_measure;
///
/// let mu = geometric_lebesgue_measure(2);  // 2D plane
/// assert_eq!(mu.dimension(), 2);
/// ```
pub fn geometric_lebesgue_measure(dimension: usize) -> GeometricLebesgueMeasure {
    let lebesgue_sigma = crate::sigma_algebra::LebesgueSigma::new(dimension);
    GeometricMeasure::new(lebesgue_sigma, dimension)
}

/// Geometric density function ρ: ℝⁿ → Cl(p,q,r)
///
/// A multivector-valued density function that defines a geometric measure via:
/// μ(A) = ∫_A ρ(x) dλ(x)
///
/// where λ is a reference measure (typically Lebesgue).
///
/// # Mathematical Properties
///
/// - ρ is measurable (each component is measurable)
/// - Integration is performed component-wise
/// - The resulting measure inherits properties from ρ and λ
///
/// # Type Parameters
///
/// This is a placeholder type that will be fully implemented when
/// integrated with amari-core's Multivector type.
pub struct GeometricDensity {
    /// Dimension of the domain space
    dimension: usize,

    /// Reference measure (usually Lebesgue)
    reference: LebesgueMeasure,
}

impl GeometricDensity {
    /// Create a geometric density
    ///
    /// # Arguments
    ///
    /// * `dimension` - The dimension of the domain space
    ///
    /// # Examples
    ///
    /// ```
    /// use amari_measure::GeometricDensity;
    ///
    /// let density = GeometricDensity::new(3);
    /// assert_eq!(density.dimension(), 3);
    /// ```
    pub fn new(dimension: usize) -> Self {
        Self {
            dimension,
            reference: LebesgueMeasure::new(dimension),
        }
    }

    /// Get the dimension of the domain space
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Get a reference to the underlying reference measure
    pub fn reference_measure(&self) -> &LebesgueMeasure {
        &self.reference
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sigma_algebra::{BorelSigma, LebesgueSigma};

    #[test]
    fn test_geometric_measure_creation() {
        let borel = BorelSigma::new(3);
        let mu: GeometricMeasure<_, SigmaFinite, Unsigned, Complete> =
            GeometricMeasure::new(borel, 3);

        assert_eq!(mu.dimension(), 3);
        assert_eq!(mu.num_grades(), 4); // 0, 1, 2, 3
        assert_eq!(mu.num_components(), 8); // 2^3
    }

    #[test]
    fn test_geometric_lebesgue_measure() {
        let mu = geometric_lebesgue_measure(2);
        assert_eq!(mu.dimension(), 2);
        assert_eq!(mu.num_grades(), 3); // 0, 1, 2
        assert_eq!(mu.num_components(), 4); // 2^2
    }

    #[test]
    fn test_geometric_density_creation() {
        let density = GeometricDensity::new(3);
        assert_eq!(density.dimension(), 3);
        assert_eq!(density.reference_measure().dimension(), 3);
    }

    #[test]
    fn test_num_components_formula() {
        // Verify 2^n formula for various dimensions
        for dim in 0..=5 {
            let mu = geometric_lebesgue_measure(dim);
            assert_eq!(mu.num_components(), 1 << dim); // 2^dim
        }
    }

    #[test]
    fn test_num_grades_formula() {
        // Verify n+1 formula for grades
        for dim in 0..=5 {
            let mu = geometric_lebesgue_measure(dim);
            assert_eq!(mu.num_grades(), dim + 1);
        }
    }

    #[test]
    fn test_geometric_measure_with_different_algebras() {
        // Test with Borel σ-algebra
        let borel = BorelSigma::new(2);
        let _mu_borel: GeometricMeasure<_, SigmaFinite, Unsigned, Incomplete> =
            GeometricMeasure::new(borel, 2);

        // Test with Lebesgue σ-algebra
        let lebesgue = LebesgueSigma::new(2);
        let _mu_lebesgue: GeometricMeasure<_, SigmaFinite, Unsigned, Complete> =
            GeometricMeasure::new(lebesgue, 2);
    }
}
