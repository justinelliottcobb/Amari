//! Probability distributions over multivector spaces
//!
//! This module provides probability distributions over Clifford algebras Cl(P,Q,R),
//! including Gaussian, uniform, and grade-projected distributions.
//!
//! # Mathematical Background
//!
//! A multivector in Cl(P,Q,R) has 2^(P+Q+R) components. Distributions over this
//! space can be:
//!
//! - **Component-wise**: Independent distributions on each coefficient
//! - **Grade-structured**: Different distributions per grade
//! - **Geometrically constrained**: Distributions on submanifolds (e.g., rotors)
//!
//! # Examples
//!
//! ```ignore
//! use amari_probabilistic::distribution::{GaussianMultivector, MultivectorDistribution};
//! use amari_core::Multivector;
//!
//! // Gaussian distribution with zero mean and unit variance per component
//! let gaussian = GaussianMultivector::<3, 0, 0>::standard();
//!
//! // Sample and evaluate
//! let mut rng = rand::thread_rng();
//! let sample = gaussian.sample(&mut rng);
//! let log_p = gaussian.log_prob(&sample)?;
//! ```

use crate::error::{ProbabilisticError, Result};
use crate::Distribution;
use amari_core::Multivector;
use rand::Rng;
use rand_distr::{Distribution as RandDist, Normal, Uniform};
use std::f64::consts::PI;
use std::marker::PhantomData;

// ============================================================================
// MultivectorDistribution Trait
// ============================================================================

/// Trait for distributions specifically over multivector spaces
///
/// Extends the base `Distribution` trait with geometric algebra operations.
pub trait MultivectorDistribution<const P: usize, const Q: usize, const R: usize>:
    Distribution<Multivector<P, Q, R>>
{
    /// Dimension of the multivector space (2^(P+Q+R))
    const DIM: usize = 1 << (P + Q + R);

    /// Get the mean multivector
    fn mean(&self) -> Multivector<P, Q, R>;

    /// Get component-wise variances
    fn variances(&self) -> Vec<f64>;

    /// Get the covariance matrix (if available)
    ///
    /// Returns a flattened DIM×DIM covariance matrix, or None if not tractable.
    fn covariance_matrix(&self) -> Option<Vec<f64>> {
        None
    }

    /// Project distribution onto a specific grade
    ///
    /// Returns a distribution concentrated on the given grade.
    fn grade_marginal(&self, grade: usize) -> GradeProjectedDistribution<P, Q, R>;
}

// ============================================================================
// Gaussian Distribution on Multivector Space
// ============================================================================

/// Gaussian (normal) distribution on multivector space
///
/// A multivariate Gaussian with independent components. Each coefficient
/// of the multivector is drawn from a 1D Gaussian.
///
/// # Mathematical Form
///
/// For multivector x with mean μ and diagonal covariance Σ = diag(σ²):
///
/// p(x) = (2π)^(-n/2) |Σ|^(-1/2) exp(-½(x-μ)ᵀΣ⁻¹(x-μ))
///
/// With diagonal covariance, this simplifies to:
///
/// p(x) = ∏ᵢ (2πσᵢ²)^(-1/2) exp(-(xᵢ-μᵢ)²/(2σᵢ²))
#[derive(Debug, Clone)]
pub struct GaussianMultivector<const P: usize, const Q: usize, const R: usize> {
    /// Mean multivector
    mean: Multivector<P, Q, R>,
    /// Standard deviations per component
    std_devs: Vec<f64>,
    /// Log normalization constant
    log_norm: f64,
}

impl<const P: usize, const Q: usize, const R: usize> GaussianMultivector<P, Q, R> {
    /// Dimension of the multivector space
    const DIM: usize = 1 << (P + Q + R);

    /// Create a Gaussian with given mean and per-component standard deviations
    ///
    /// # Arguments
    ///
    /// * `mean` - Mean multivector
    /// * `std_devs` - Standard deviation for each component
    ///
    /// # Errors
    ///
    /// Returns error if std_devs length doesn't match dimension or has non-positive values.
    pub fn new(mean: Multivector<P, Q, R>, std_devs: Vec<f64>) -> Result<Self> {
        if std_devs.len() != Self::DIM {
            return Err(ProbabilisticError::dimension_mismatch(
                Self::DIM,
                std_devs.len(),
            ));
        }

        for (i, &s) in std_devs.iter().enumerate() {
            if s <= 0.0 {
                return Err(ProbabilisticError::invalid_parameters(format!(
                    "Standard deviation must be positive, got {} at index {}",
                    s, i
                )));
            }
        }

        // Log normalization: -n/2 * log(2π) - sum(log(σᵢ))
        let n = Self::DIM as f64;
        let log_norm = -0.5 * n * (2.0 * PI).ln() - std_devs.iter().map(|s| s.ln()).sum::<f64>();

        Ok(Self {
            mean,
            std_devs,
            log_norm,
        })
    }

    /// Create a standard Gaussian (zero mean, unit variance)
    pub fn standard() -> Self {
        let mean = Multivector::zero();
        let std_devs = vec![1.0; Self::DIM];
        let n = Self::DIM as f64;
        let log_norm = -0.5 * n * (2.0 * PI).ln();

        Self {
            mean,
            std_devs,
            log_norm,
        }
    }

    /// Create an isotropic Gaussian with given variance
    ///
    /// # Arguments
    ///
    /// * `mean` - Mean multivector
    /// * `variance` - Variance (same for all components)
    pub fn isotropic(mean: Multivector<P, Q, R>, variance: f64) -> Result<Self> {
        if variance <= 0.0 {
            return Err(ProbabilisticError::invalid_parameters(format!(
                "Variance must be positive, got {}",
                variance
            )));
        }

        let std_dev = variance.sqrt();
        let std_devs = vec![std_dev; Self::DIM];
        Self::new(mean, std_devs)
    }

    /// Create a Gaussian concentrated on a specific grade
    ///
    /// Non-specified grades have very small variance (effectively zero).
    ///
    /// # Arguments
    ///
    /// * `grade` - Grade to concentrate on
    /// * `variance` - Variance for components in the grade
    pub fn grade_concentrated(grade: usize, variance: f64) -> Result<Self> {
        if variance <= 0.0 {
            return Err(ProbabilisticError::invalid_parameters(format!(
                "Variance must be positive, got {}",
                variance
            )));
        }

        let std_dev = variance.sqrt();
        let tiny = 1e-10; // Very small variance for other grades
        let mean = Multivector::zero();

        // Determine which components belong to the target grade
        let dim = P + Q + R;
        let mut std_devs = vec![tiny; Self::DIM];

        for (i, std_dev_val) in std_devs.iter_mut().enumerate() {
            if i.count_ones() as usize == grade {
                *std_dev_val = std_dev;
            }
        }

        // Check that grade is valid
        if grade > dim {
            return Err(ProbabilisticError::invalid_parameters(format!(
                "Grade {} exceeds maximum grade {} for Cl({},{},{})",
                grade, dim, P, Q, R
            )));
        }

        Self::new(mean, std_devs)
    }

    /// Get the mean
    pub fn get_mean(&self) -> &Multivector<P, Q, R> {
        &self.mean
    }

    /// Get the standard deviations
    pub fn get_std_devs(&self) -> &[f64] {
        &self.std_devs
    }
}

impl<const P: usize, const Q: usize, const R: usize> Distribution<Multivector<P, Q, R>>
    for GaussianMultivector<P, Q, R>
{
    fn sample<R_: Rng>(&self, rng: &mut R_) -> Multivector<P, Q, R> {
        let mut coeffs = Vec::with_capacity(Self::DIM);

        for i in 0..Self::DIM {
            let normal = Normal::new(self.mean.get(i), self.std_devs[i]).unwrap();
            coeffs.push(normal.sample(rng));
        }

        Multivector::from_coefficients(coeffs)
    }

    fn log_prob(&self, x: &Multivector<P, Q, R>) -> Result<f64> {
        let mut log_p = self.log_norm;

        for i in 0..Self::DIM {
            let diff = x.get(i) - self.mean.get(i);
            let sigma = self.std_devs[i];
            log_p -= 0.5 * (diff * diff) / (sigma * sigma);
        }

        Ok(log_p)
    }
}

impl<const P: usize, const Q: usize, const R: usize> MultivectorDistribution<P, Q, R>
    for GaussianMultivector<P, Q, R>
{
    fn mean(&self) -> Multivector<P, Q, R> {
        self.mean.clone()
    }

    fn variances(&self) -> Vec<f64> {
        self.std_devs.iter().map(|s| s * s).collect()
    }

    fn covariance_matrix(&self) -> Option<Vec<f64>> {
        // Return diagonal covariance matrix in flattened form
        let n = Self::DIM;
        let mut cov = vec![0.0; n * n];
        for i in 0..n {
            cov[i * n + i] = self.std_devs[i] * self.std_devs[i];
        }
        Some(cov)
    }

    fn grade_marginal(&self, grade: usize) -> GradeProjectedDistribution<P, Q, R> {
        GradeProjectedDistribution::from_gaussian(self, grade)
    }
}

// ============================================================================
// Uniform Distribution on Multivector Hypercube
// ============================================================================

/// Uniform distribution on a multivector hypercube
///
/// Each component is independently uniformly distributed on [min, max].
#[derive(Debug, Clone)]
pub struct UniformMultivector<const P: usize, const Q: usize, const R: usize> {
    /// Lower bounds per component
    mins: Vec<f64>,
    /// Upper bounds per component
    maxs: Vec<f64>,
    /// Log probability (constant)
    log_prob: f64,
}

impl<const P: usize, const Q: usize, const R: usize> UniformMultivector<P, Q, R> {
    /// Dimension of the multivector space
    const DIM: usize = 1 << (P + Q + R);

    /// Create a uniform distribution on a hypercube [min, max]^n
    pub fn hypercube(min: f64, max: f64) -> Result<Self> {
        if min >= max {
            return Err(ProbabilisticError::invalid_parameters(format!(
                "min ({}) must be less than max ({})",
                min, max
            )));
        }

        let mins = vec![min; Self::DIM];
        let maxs = vec![max; Self::DIM];
        let volume = (max - min).powi(Self::DIM as i32);
        let log_prob = -volume.ln();

        Ok(Self {
            mins,
            maxs,
            log_prob,
        })
    }

    /// Create a uniform distribution with per-component bounds
    pub fn new(mins: Vec<f64>, maxs: Vec<f64>) -> Result<Self> {
        if mins.len() != Self::DIM {
            return Err(ProbabilisticError::dimension_mismatch(
                Self::DIM,
                mins.len(),
            ));
        }
        if maxs.len() != Self::DIM {
            return Err(ProbabilisticError::dimension_mismatch(
                Self::DIM,
                maxs.len(),
            ));
        }

        let mut log_volume = 0.0;
        for i in 0..Self::DIM {
            if mins[i] >= maxs[i] {
                return Err(ProbabilisticError::invalid_parameters(format!(
                    "min[{}] ({}) must be less than max[{}] ({})",
                    i, mins[i], i, maxs[i]
                )));
            }
            log_volume += (maxs[i] - mins[i]).ln();
        }

        Ok(Self {
            mins,
            maxs,
            log_prob: -log_volume,
        })
    }
}

impl<const P: usize, const Q: usize, const R: usize> Distribution<Multivector<P, Q, R>>
    for UniformMultivector<P, Q, R>
{
    fn sample<R_: Rng>(&self, rng: &mut R_) -> Multivector<P, Q, R> {
        let mut coeffs = Vec::with_capacity(Self::DIM);

        for i in 0..Self::DIM {
            let uniform = Uniform::new(self.mins[i], self.maxs[i]);
            coeffs.push(uniform.sample(rng));
        }

        Multivector::from_coefficients(coeffs)
    }

    fn log_prob(&self, x: &Multivector<P, Q, R>) -> Result<f64> {
        // Check if in support
        for i in 0..Self::DIM {
            let xi = x.get(i);
            if xi < self.mins[i] || xi > self.maxs[i] {
                return Err(ProbabilisticError::out_of_support(format!(
                    "Component {} = {} outside bounds [{}, {}]",
                    i, xi, self.mins[i], self.maxs[i]
                )));
            }
        }

        Ok(self.log_prob)
    }
}

impl<const P: usize, const Q: usize, const R: usize> MultivectorDistribution<P, Q, R>
    for UniformMultivector<P, Q, R>
{
    fn mean(&self) -> Multivector<P, Q, R> {
        let coeffs: Vec<f64> = self
            .mins
            .iter()
            .zip(&self.maxs)
            .map(|(&a, &b)| 0.5 * (a + b))
            .collect();
        Multivector::from_coefficients(coeffs)
    }

    fn variances(&self) -> Vec<f64> {
        self.mins
            .iter()
            .zip(&self.maxs)
            .map(|(&a, &b)| {
                let range = b - a;
                range * range / 12.0 // Variance of uniform on [a,b]
            })
            .collect()
    }

    fn grade_marginal(&self, grade: usize) -> GradeProjectedDistribution<P, Q, R> {
        GradeProjectedDistribution::from_uniform(self, grade)
    }
}

// ============================================================================
// Grade-Projected Distribution
// ============================================================================

/// Distribution projected onto a specific grade
///
/// Represents the marginal distribution of a multivector distribution
/// restricted to components of a given grade.
#[derive(Debug, Clone)]
pub struct GradeProjectedDistribution<const P: usize, const Q: usize, const R: usize> {
    /// Grade this distribution is projected onto
    grade: usize,
    /// Indices of components in this grade
    component_indices: Vec<usize>,
    /// Mean values for grade components
    means: Vec<f64>,
    /// Standard deviations for grade components (Gaussian assumption)
    std_devs: Vec<f64>,
    _phantom: PhantomData<((), (), ())>,
}

impl<const P: usize, const Q: usize, const R: usize> GradeProjectedDistribution<P, Q, R> {
    /// Dimension of the multivector space
    const DIM: usize = 1 << (P + Q + R);

    /// Create from a Gaussian distribution
    pub fn from_gaussian(gaussian: &GaussianMultivector<P, Q, R>, grade: usize) -> Self {
        let mut component_indices = Vec::new();
        let mut means = Vec::new();
        let mut std_devs = Vec::new();

        for i in 0..Self::DIM {
            if i.count_ones() as usize == grade {
                component_indices.push(i);
                means.push(gaussian.mean.get(i));
                std_devs.push(gaussian.std_devs[i]);
            }
        }

        Self {
            grade,
            component_indices,
            means,
            std_devs,
            _phantom: PhantomData,
        }
    }

    /// Create from a uniform distribution
    pub fn from_uniform(uniform: &UniformMultivector<P, Q, R>, grade: usize) -> Self {
        let mut component_indices = Vec::new();
        let mut means = Vec::new();
        let mut std_devs = Vec::new();

        for i in 0..Self::DIM {
            if i.count_ones() as usize == grade {
                component_indices.push(i);
                let a = uniform.mins[i];
                let b = uniform.maxs[i];
                means.push(0.5 * (a + b));
                // Use Gaussian approximation with matched variance
                std_devs.push((b - a) / (12.0_f64).sqrt());
            }
        }

        Self {
            grade,
            component_indices,
            means,
            std_devs,
            _phantom: PhantomData,
        }
    }

    /// Get the grade this distribution is projected onto
    pub fn grade(&self) -> usize {
        self.grade
    }

    /// Number of components in this grade
    pub fn num_components(&self) -> usize {
        self.component_indices.len()
    }

    /// Sample grade-projected values
    pub fn sample_components<R_: Rng>(&self, rng: &mut R_) -> Vec<f64> {
        self.means
            .iter()
            .zip(&self.std_devs)
            .map(|(&m, &s)| {
                let normal = Normal::new(m, s).unwrap();
                normal.sample(rng)
            })
            .collect()
    }
}

impl<const P: usize, const Q: usize, const R: usize> Distribution<Multivector<P, Q, R>>
    for GradeProjectedDistribution<P, Q, R>
{
    fn sample<R_: Rng>(&self, rng: &mut R_) -> Multivector<P, Q, R> {
        let mut mv = Multivector::zero();
        let values = self.sample_components(rng);

        for (&idx, &val) in self.component_indices.iter().zip(&values) {
            mv.set(idx, val);
        }

        mv
    }

    fn log_prob(&self, x: &Multivector<P, Q, R>) -> Result<f64> {
        // Only evaluate probability for the projected grade
        let n = self.means.len() as f64;
        let mut log_p = -0.5 * n * (2.0 * PI).ln();

        for (i, &idx) in self.component_indices.iter().enumerate() {
            let diff = x.get(idx) - self.means[i];
            let sigma = self.std_devs[i];
            log_p -= sigma.ln();
            log_p -= 0.5 * (diff * diff) / (sigma * sigma);
        }

        Ok(log_p)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_gaussian_standard() {
        let gaussian = GaussianMultivector::<2, 0, 0>::standard();
        let mut rng = rand::thread_rng();

        // Sample and check dimensions
        let sample = gaussian.sample(&mut rng);
        assert_eq!(sample.to_vec().len(), 4); // 2^2 = 4 components

        // Log probability should be finite
        let log_p = gaussian.log_prob(&sample).unwrap();
        assert!(log_p.is_finite());
    }

    #[test]
    fn test_gaussian_isotropic() {
        let mean = Multivector::<2, 0, 0>::scalar(1.0);
        let gaussian = GaussianMultivector::isotropic(mean.clone(), 0.5).unwrap();

        // Mean should be preserved
        assert_relative_eq!(gaussian.mean().get(0), 1.0, epsilon = 1e-10);

        // All variances should be 0.5
        for v in gaussian.variances() {
            assert_relative_eq!(v, 0.5, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_gaussian_log_prob_at_mean() {
        let mean = Multivector::<2, 0, 0>::zero();
        let gaussian = GaussianMultivector::isotropic(mean.clone(), 1.0).unwrap();

        // Log prob at mean should be the normalization constant
        let log_p = gaussian.log_prob(&mean).unwrap();
        let expected = -0.5 * 4.0 * (2.0 * PI).ln(); // n=4, σ=1
        assert_relative_eq!(log_p, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_uniform_hypercube() {
        let uniform = UniformMultivector::<1, 0, 0>::hypercube(-1.0, 1.0).unwrap();
        let mut rng = rand::thread_rng();

        // Sample multiple times and check bounds
        for _ in 0..100 {
            let sample = uniform.sample(&mut rng);
            for i in 0..2 {
                assert!(sample.get(i) >= -1.0);
                assert!(sample.get(i) <= 1.0);
            }
        }
    }

    #[test]
    fn test_uniform_log_prob() {
        let uniform = UniformMultivector::<1, 0, 0>::hypercube(0.0, 1.0).unwrap();

        // Inside support
        let inside = Multivector::from_coefficients(vec![0.5, 0.5]);
        let log_p = uniform.log_prob(&inside).unwrap();
        assert_relative_eq!(log_p, 0.0, epsilon = 1e-10); // log(1) for unit hypercube

        // Outside support
        let outside = Multivector::from_coefficients(vec![1.5, 0.5]);
        assert!(uniform.log_prob(&outside).is_err());
    }

    #[test]
    fn test_grade_projected() {
        let gaussian = GaussianMultivector::<2, 0, 0>::standard();

        // Grade 0 (scalar) should have 1 component
        let grade0 = gaussian.grade_marginal(0);
        assert_eq!(grade0.num_components(), 1);

        // Grade 1 (vectors) should have 2 components for Cl(2,0,0)
        let grade1 = gaussian.grade_marginal(1);
        assert_eq!(grade1.num_components(), 2);

        // Grade 2 (bivectors) should have 1 component
        let grade2 = gaussian.grade_marginal(2);
        assert_eq!(grade2.num_components(), 1);
    }

    #[test]
    fn test_sample_n() {
        let gaussian = GaussianMultivector::<2, 0, 0>::standard();
        let mut rng = rand::thread_rng();

        let samples = gaussian.sample_n(&mut rng, 50);
        assert_eq!(samples.len(), 50);
    }

    #[test]
    fn test_invalid_parameters() {
        // Negative variance
        let mean = Multivector::<2, 0, 0>::zero();
        let result = GaussianMultivector::isotropic(mean, -1.0);
        assert!(result.is_err());

        // Zero variance
        let mean2 = Multivector::<2, 0, 0>::zero();
        let result2 = GaussianMultivector::isotropic(mean2, 0.0);
        assert!(result2.is_err());

        // Invalid uniform bounds
        let result3 = UniformMultivector::<2, 0, 0>::hypercube(1.0, 0.0);
        assert!(result3.is_err());
    }
}
