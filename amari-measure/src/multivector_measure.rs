//! Grade-decomposed geometric measures with actual Multivector storage
//!
//! This module provides geometric measures that store and operate on actual
//! `Multivector` instances from amari-core, enabling full geometric algebra
//! operations on measure spaces.
//!
//! # Overview
//!
//! A grade-decomposed measure assigns a multivector value to each measurable set,
//! where each basis component represents an independent measure:
//!
//! μ(A) = Σᵢ μᵢ(A) eᵢ
//!
//! where eᵢ are the 2^n basis elements of Cl(p,q,r).
//!
//! # Examples
//!
//! ```
//! use amari_measure::multivector_measure::GradeDecomposedMeasure;
//! use amari_core::Multivector;
//!
//! // Create a 3D measure (signature (3,0,0))
//! let measure = GradeDecomposedMeasure::<3, 0, 0>::new();
//!
//! // Measure has 2^3 = 8 basis components
//! assert_eq!(measure.num_components(), 8);
//! ```

use crate::error::{MeasureError, Result};
use crate::phantom::*;
use crate::sigma_algebra::SigmaAlgebra;
use amari_core::Multivector;
use core::marker::PhantomData;

/// Grade-decomposed geometric measure with Multivector values
///
/// This measure assigns a `Multivector<P, Q, R>` to each measurable set,
/// where each component is an independent real measure.
///
/// # Type Parameters
///
/// - `P`: Number of basis vectors with +1 signature
/// - `Q`: Number of basis vectors with -1 signature
/// - `R`: Number of basis vectors with 0 signature
/// - `Σ`: σ-algebra type
/// - `F`: Finiteness property
/// - `S`: Sign property
/// - `C`: Completeness property
///
/// # Mathematical Structure
///
/// The measure μ: Σ → Cl(p,q,r) satisfies:
/// - Countable additivity: μ(⋃ Aₙ) = Σ μ(Aₙ) (multivector sum)
/// - Each coefficient μᵢ is a real-valued measure
/// - Grade-k projection: μ⟨k⟩(A) extracts grade-k components
#[derive(Clone)]
pub struct GradeDecomposedMeasure<
    const P: usize,
    const Q: usize,
    const R: usize,
    Σ = crate::sigma_algebra::LebesgueSigma,
    F = SigmaFinite,
    S = Unsigned,
    C = Complete,
> where
    Σ: SigmaAlgebra,
    F: MeasureProperty + 'static,
    S: MeasureProperty + 'static,
    C: MeasureProperty + 'static,
{
    /// The σ-algebra of measurable sets
    sigma_algebra: Σ,

    /// Measure coefficients for each basis element
    ///
    /// coefficients[i] stores the measure value for basis element eᵢ
    coefficients: Vec<f64>,

    _phantom: PhantomData<(F, S, C)>,
}

impl<const P: usize, const Q: usize, const R: usize, Σ, F, S, C>
    GradeDecomposedMeasure<P, Q, R, Σ, F, S, C>
where
    Σ: SigmaAlgebra,
    F: MeasureProperty + 'static,
    S: MeasureProperty + 'static,
    C: MeasureProperty + 'static,
{
    /// Create a new grade-decomposed measure with zero coefficients
    pub fn new() -> Self
    where
        Σ: Default,
    {
        let dimension = P + Q + R;
        let num_components = 2_usize.pow(dimension as u32);
        Self {
            sigma_algebra: Σ::default(),
            coefficients: vec![0.0; num_components],
            _phantom: PhantomData,
        }
    }

    /// Create measure from a multivector template
    ///
    /// Sets the measure coefficients to match the given multivector.
    ///
    /// # Arguments
    ///
    /// * `mv` - The multivector defining the measure values
    pub fn from_multivector(mv: &Multivector<P, Q, R>) -> Self
    where
        Σ: Default,
    {
        Self {
            sigma_algebra: Σ::default(),
            coefficients: mv.to_vec(),
            _phantom: PhantomData,
        }
    }

    /// Get the σ-algebra
    pub fn sigma_algebra(&self) -> &Σ {
        &self.sigma_algebra
    }

    /// Get the dimension of the underlying space
    pub const fn dimension(&self) -> usize {
        P + Q + R
    }

    /// Get the number of basis components (2^n)
    pub const fn num_components(&self) -> usize {
        1 << (P + Q + R)
    }

    /// Get the number of grades (n+1)
    pub const fn num_grades(&self) -> usize {
        P + Q + R + 1
    }

    /// Measure a set, returning the multivector value
    ///
    /// Returns μ(A) as a multivector in Cl(p,q,r).
    ///
    /// # Arguments
    ///
    /// * `_set` - The measurable set (currently unused in placeholder)
    pub fn measure(&self, _set: &Σ::Set) -> Result<Multivector<P, Q, R>> {
        // TODO: Implement actual set measurement
        // For now, return the stored coefficients
        Ok(Multivector::from_slice(&self.coefficients))
    }

    /// Extract the grade-k component of the measure
    ///
    /// Returns μ⟨k⟩(A), which measures only the grade-k part.
    ///
    /// # Arguments
    ///
    /// * `grade` - The grade to extract (0 to n)
    /// * `_set` - The measurable set
    pub fn measure_grade(&self, grade: usize, _set: &Σ::Set) -> Result<Multivector<P, Q, R>> {
        if grade > self.num_grades() {
            return Err(MeasureError::computation(format!(
                "Grade {} exceeds maximum grade {}",
                grade,
                self.num_grades()
            )));
        }

        // Extract grade-k components
        let mv = Multivector::from_slice(&self.coefficients);
        Ok(mv.grade_project(grade))
    }

    /// Get the scalar (grade-0) component
    ///
    /// Returns μ⟨0⟩(A), the scalar part of the measure.
    pub fn measure_scalar(&self, set: &Σ::Set) -> Result<f64> {
        let mv = self.measure_grade(0, set)?;
        Ok(mv.scalar_part())
    }

    /// Set measure coefficients from a multivector
    pub fn set_coefficients(&mut self, mv: &Multivector<P, Q, R>) {
        self.coefficients = mv.to_vec();
    }

    /// Get measure as a multivector (constant value for all sets)
    pub fn as_multivector(&self) -> Multivector<P, Q, R> {
        Multivector::from_slice(&self.coefficients)
    }
}

impl<const P: usize, const Q: usize, const R: usize, Σ, F, S, C> Default
    for GradeDecomposedMeasure<P, Q, R, Σ, F, S, C>
where
    Σ: SigmaAlgebra + Default,
    F: MeasureProperty + 'static,
    S: MeasureProperty + 'static,
    C: MeasureProperty + 'static,
{
    fn default() -> Self {
        Self::new()
    }
}

/// Geometric product of two measures
///
/// Computes (μ · ν)(A) = μ(A) · ν(A) using the geometric product.
///
/// # Arguments
///
/// * `mu` - First measure
/// * `nu` - Second measure
/// * `set` - The measurable set
///
/// # Returns
///
/// The geometric product μ(A) · ν(A) as a multivector
pub fn geometric_product_measures<const P: usize, const Q: usize, const R: usize, Σ, F, S, C>(
    mu: &GradeDecomposedMeasure<P, Q, R, Σ, F, S, C>,
    nu: &GradeDecomposedMeasure<P, Q, R, Σ, F, S, C>,
    set: &Σ::Set,
) -> Result<Multivector<P, Q, R>>
where
    Σ: SigmaAlgebra,
    F: MeasureProperty + 'static,
    S: MeasureProperty + 'static,
    C: MeasureProperty + 'static,
{
    let mu_val = mu.measure(set)?;
    let nu_val = nu.measure(set)?;
    Ok(mu_val.geometric_product(&nu_val))
}

/// Wedge (outer) product of two measures
///
/// Computes (μ ∧ ν)(A) = μ(A) ∧ ν(A) using the wedge product.
pub fn wedge_product_measures<const P: usize, const Q: usize, const R: usize, Σ, F, S, C>(
    mu: &GradeDecomposedMeasure<P, Q, R, Σ, F, S, C>,
    nu: &GradeDecomposedMeasure<P, Q, R, Σ, F, S, C>,
    set: &Σ::Set,
) -> Result<Multivector<P, Q, R>>
where
    Σ: SigmaAlgebra,
    F: MeasureProperty + 'static,
    S: MeasureProperty + 'static,
    C: MeasureProperty + 'static,
{
    let mu_val = mu.measure(set)?;
    let nu_val = nu.measure(set)?;
    Ok(mu_val.wedge(&nu_val))
}

/// Inner (contraction) product of two measures
pub fn inner_product_measures<const P: usize, const Q: usize, const R: usize, Σ, F, S, C>(
    mu: &GradeDecomposedMeasure<P, Q, R, Σ, F, S, C>,
    nu: &GradeDecomposedMeasure<P, Q, R, Σ, F, S, C>,
    set: &Σ::Set,
) -> Result<Multivector<P, Q, R>>
where
    Σ: SigmaAlgebra,
    F: MeasureProperty + 'static,
    S: MeasureProperty + 'static,
    C: MeasureProperty + 'static,
{
    let mu_val = mu.measure(set)?;
    let nu_val = nu.measure(set)?;
    Ok(mu_val.dot(&nu_val))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_grade_decomposed_measure_creation() {
        // Create a 3D measure (signature (3,0,0))
        let measure = GradeDecomposedMeasure::<3, 0, 0>::new();

        assert_eq!(measure.dimension(), 3);
        assert_eq!(measure.num_components(), 8); // 2^3
        assert_eq!(measure.num_grades(), 4); // 0,1,2,3
    }

    #[test]
    fn test_from_multivector() {
        // Create a multivector with known values
        let mv = Multivector::<2, 0, 0>::from_slice(&[1.0, 2.0, 3.0, 4.0]);
        let measure = GradeDecomposedMeasure::<2, 0, 0>::from_multivector(&mv);

        let result = measure.as_multivector();
        assert_abs_diff_eq!(result.scalar_part(), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_measure_grade_projection() {
        let mv = Multivector::<2, 0, 0>::from_slice(&[1.0, 2.0, 3.0, 4.0]);
        let measure = GradeDecomposedMeasure::<2, 0, 0>::from_multivector(&mv);

        // Use unit type () as the dummy set
        let dummy_set = ();

        // Extract grade-0 (scalar)
        let grade0 = measure.measure_grade(0, &dummy_set).unwrap();
        assert_abs_diff_eq!(grade0.scalar_part(), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_geometric_product_measures() {
        // Create two measures
        let mv1 = Multivector::<2, 0, 0>::from_slice(&[2.0, 0.0, 0.0, 0.0]);
        let mv2 = Multivector::<2, 0, 0>::from_slice(&[3.0, 0.0, 0.0, 0.0]);

        let mu = GradeDecomposedMeasure::<2, 0, 0>::from_multivector(&mv1);
        let nu = GradeDecomposedMeasure::<2, 0, 0>::from_multivector(&mv2);

        let dummy_set = ();

        // Geometric product of scalars: 2 * 3 = 6
        let product = geometric_product_measures(&mu, &nu, &dummy_set).unwrap();
        assert_abs_diff_eq!(product.scalar_part(), 6.0, epsilon = 1e-10);
    }

    #[test]
    fn test_different_signatures() {
        // Test with Minkowski signature (1,3,0) for spacetime
        let measure_spacetime = GradeDecomposedMeasure::<1, 3, 0>::new();
        assert_eq!(measure_spacetime.dimension(), 4);
        assert_eq!(measure_spacetime.num_components(), 16); // 2^4

        // Test with Euclidean signature (3,0,0)
        let measure_euclidean = GradeDecomposedMeasure::<3, 0, 0>::new();
        assert_eq!(measure_euclidean.dimension(), 3);
        assert_eq!(measure_euclidean.num_components(), 8); // 2^3
    }

    #[test]
    fn test_measure_scalar_extraction() {
        let mv = Multivector::<2, 0, 0>::from_slice(&[5.0, 1.0, 2.0, 3.0]);
        let measure = GradeDecomposedMeasure::<2, 0, 0>::from_multivector(&mv);

        let dummy_set = ();

        let scalar_part = measure.measure_scalar(&dummy_set).unwrap();
        assert_abs_diff_eq!(scalar_part, 5.0, epsilon = 1e-10);
    }
}
