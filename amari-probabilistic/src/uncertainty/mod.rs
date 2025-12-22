//! Uncertainty propagation through geometric algebra operations
//!
//! This module provides methods for propagating uncertainty through
//! operations on multivector-valued random variables.
//!
//! # Methods
//!
//! - **Linear Propagation**: First-order Taylor expansion (fast, approximate)
//! - **Unscented Transform**: Sigma-point propagation (accurate for nonlinear)
//! - **Monte Carlo**: Sample-based propagation (exact but slow)
//!
//! # Example
//!
//! ```ignore
//! use amari_probabilistic::uncertainty::{UncertainMultivector, LinearPropagation};
//!
//! let x = UncertainMultivector::gaussian(mean, covariance);
//! let y = UncertainMultivector::gaussian(other_mean, other_cov);
//!
//! // Propagate through geometric product
//! let propagator = LinearPropagation::new();
//! let z = propagator.geometric_product(&x, &y)?;
//! ```

use crate::error::{ProbabilisticError, Result};
use crate::random::CovarianceMatrix;
use amari_core::Multivector;

/// Multivector with associated uncertainty (mean and covariance)
#[derive(Debug, Clone)]
pub struct UncertainMultivector<const P: usize, const Q: usize, const R: usize> {
    /// Mean value
    mean: Multivector<P, Q, R>,
    /// Covariance matrix
    covariance: CovarianceMatrix<P, Q, R>,
}

impl<const P: usize, const Q: usize, const R: usize> UncertainMultivector<P, Q, R> {
    /// Dimension of the multivector space
    #[allow(dead_code)]
    const DIM: usize = 1 << (P + Q + R);

    /// Create with given mean and covariance
    pub fn new(mean: Multivector<P, Q, R>, covariance: CovarianceMatrix<P, Q, R>) -> Self {
        Self { mean, covariance }
    }

    /// Create with diagonal covariance
    pub fn diagonal(mean: Multivector<P, Q, R>, variances: &[f64]) -> Result<Self> {
        let covariance = CovarianceMatrix::diagonal(variances)?;
        Ok(Self { mean, covariance })
    }

    /// Create a deterministic (zero-variance) uncertain multivector
    pub fn deterministic(value: Multivector<P, Q, R>) -> Self {
        Self {
            mean: value,
            covariance: CovarianceMatrix::zero(),
        }
    }

    /// Get the mean
    pub fn mean(&self) -> &Multivector<P, Q, R> {
        &self.mean
    }

    /// Get the covariance
    pub fn covariance(&self) -> &CovarianceMatrix<P, Q, R> {
        &self.covariance
    }

    /// Get variances (diagonal of covariance)
    pub fn variances(&self) -> Vec<f64> {
        self.covariance.get_diagonal()
    }

    /// Compute total uncertainty (trace of covariance)
    pub fn total_variance(&self) -> f64 {
        self.covariance.trace()
    }

    /// Compute standard deviations
    pub fn std_devs(&self) -> Vec<f64> {
        self.variances().iter().map(|&v| v.sqrt()).collect()
    }
}

/// Linear (first-order) uncertainty propagation
///
/// Uses Jacobian-based propagation: Cov(f(X)) ≈ J Cov(X) Jᵀ
#[derive(Debug, Clone, Default)]
pub struct LinearPropagation;

impl LinearPropagation {
    /// Create a new linear propagation method
    pub fn new() -> Self {
        Self
    }

    /// Propagate through a linear transformation
    ///
    /// For Y = AX + b, we have:
    /// - mean(Y) = A·mean(X) + b
    /// - Cov(Y) = A·Cov(X)·Aᵀ
    pub fn linear_transform<const P: usize, const Q: usize, const R: usize>(
        &self,
        x: &UncertainMultivector<P, Q, R>,
        a: &[f64], // Flattened transformation matrix
        b: &Multivector<P, Q, R>,
    ) -> Result<UncertainMultivector<P, Q, R>> {
        let dim = 1 << (P + Q + R);

        if a.len() != dim * dim {
            return Err(ProbabilisticError::dimension_mismatch(dim * dim, a.len()));
        }

        // Transform mean: y = Ax + b
        let x_mean = x.mean.to_vec();
        let mut y_mean_coeffs = b.to_vec();

        for i in 0..dim {
            for j in 0..dim {
                y_mean_coeffs[i] += a[i * dim + j] * x_mean[j];
            }
        }

        // Transform covariance: Σ_y = A Σ_x Aᵀ
        let sigma_x = x.covariance.as_slice();
        let mut sigma_y = vec![0.0; dim * dim];

        // A Σ_x
        let mut temp = vec![0.0; dim * dim];
        for i in 0..dim {
            for j in 0..dim {
                for k in 0..dim {
                    temp[i * dim + j] += a[i * dim + k] * sigma_x[k * dim + j];
                }
            }
        }

        // (A Σ_x) Aᵀ
        for i in 0..dim {
            for j in 0..dim {
                for k in 0..dim {
                    sigma_y[i * dim + j] += temp[i * dim + k] * a[j * dim + k];
                }
            }
        }

        Ok(UncertainMultivector::new(
            Multivector::from_coefficients(y_mean_coeffs),
            CovarianceMatrix::from_data(sigma_y)?,
        ))
    }

    /// Propagate through scalar multiplication
    pub fn scalar_multiply<const P: usize, const Q: usize, const R: usize>(
        &self,
        x: &UncertainMultivector<P, Q, R>,
        scalar: f64,
    ) -> UncertainMultivector<P, Q, R> {
        let _dim = 1 << (P + Q + R);

        // Mean scales linearly
        let mean_coeffs: Vec<f64> = x.mean.to_vec().iter().map(|&c| c * scalar).collect();

        // Covariance scales by scalar²
        let cov_data: Vec<f64> = x
            .covariance
            .as_slice()
            .iter()
            .map(|&c| c * scalar * scalar)
            .collect();

        UncertainMultivector::new(
            Multivector::from_coefficients(mean_coeffs),
            CovarianceMatrix::from_data(cov_data).unwrap(),
        )
    }

    /// Propagate through addition of two uncertain multivectors
    pub fn add<const P: usize, const Q: usize, const R: usize>(
        &self,
        x: &UncertainMultivector<P, Q, R>,
        y: &UncertainMultivector<P, Q, R>,
    ) -> UncertainMultivector<P, Q, R> {
        let _dim = 1 << (P + Q + R);

        // Means add
        let mean = x.mean.add(&y.mean);

        // Covariances add (assuming independence)
        let cov_data: Vec<f64> = x
            .covariance
            .as_slice()
            .iter()
            .zip(y.covariance.as_slice().iter())
            .map(|(&a, &b)| a + b)
            .collect();

        UncertainMultivector::new(mean, CovarianceMatrix::from_data(cov_data).unwrap())
    }
}

/// Unscented transform for nonlinear propagation
///
/// Uses sigma points to capture mean and covariance through nonlinear transformations.
#[derive(Debug, Clone)]
pub struct UnscentedTransform {
    /// Alpha parameter (spread of sigma points)
    alpha: f64,
    /// Beta parameter (distribution knowledge, 2 optimal for Gaussian)
    beta: f64,
    /// Kappa parameter (secondary scaling)
    kappa: f64,
}

impl UnscentedTransform {
    /// Create with default parameters optimized for Gaussian
    pub fn new() -> Self {
        Self {
            alpha: 1e-3,
            beta: 2.0,
            kappa: 0.0,
        }
    }

    /// Create with custom parameters
    pub fn with_params(alpha: f64, beta: f64, kappa: f64) -> Self {
        Self { alpha, beta, kappa }
    }

    /// Generate sigma points
    pub fn sigma_points<const P: usize, const Q: usize, const R: usize>(
        &self,
        x: &UncertainMultivector<P, Q, R>,
    ) -> Vec<Multivector<P, Q, R>> {
        let dim = 1 << (P + Q + R);
        let n = dim as f64;

        let lambda = self.alpha * self.alpha * (n + self.kappa) - n;
        let c = n + lambda;

        let mean = x.mean.to_vec();
        let cov = x.covariance.as_slice();

        // Compute sqrt of scaled covariance using Cholesky-like decomposition
        // For simplicity, we use diagonal approximation
        let sqrt_cov: Vec<f64> = (0..dim).map(|i| (c * cov[i * dim + i]).sqrt()).collect();

        let mut sigma_points = Vec::with_capacity(2 * dim + 1);

        // Mean point
        sigma_points.push(x.mean.clone());

        // Points at +/- sqrt(c * Σ)
        for i in 0..dim {
            let mut plus = mean.clone();
            let mut minus = mean.clone();

            plus[i] += sqrt_cov[i];
            minus[i] -= sqrt_cov[i];

            sigma_points.push(Multivector::from_coefficients(plus));
            sigma_points.push(Multivector::from_coefficients(minus));
        }

        sigma_points
    }

    /// Compute weights for sigma points
    pub fn weights<const P: usize, const Q: usize, const R: usize>(&self) -> (Vec<f64>, Vec<f64>) {
        let dim = 1 << (P + Q + R);
        let n = dim as f64;

        let lambda = self.alpha * self.alpha * (n + self.kappa) - n;

        // Weights for mean computation
        let w0_m = lambda / (n + lambda);
        let wi_m = 1.0 / (2.0 * (n + lambda));

        // Weights for covariance computation
        let w0_c = w0_m + (1.0 - self.alpha * self.alpha + self.beta);
        let wi_c = wi_m;

        let mut w_m = vec![wi_m; 2 * dim + 1];
        let mut w_c = vec![wi_c; 2 * dim + 1];
        w_m[0] = w0_m;
        w_c[0] = w0_c;

        (w_m, w_c)
    }

    /// Propagate through a general nonlinear function
    pub fn propagate<const P: usize, const Q: usize, const R: usize, F>(
        &self,
        x: &UncertainMultivector<P, Q, R>,
        f: F,
    ) -> Result<UncertainMultivector<P, Q, R>>
    where
        F: Fn(&Multivector<P, Q, R>) -> Multivector<P, Q, R>,
    {
        let dim = 1 << (P + Q + R);

        // Generate sigma points
        let sigma_points = self.sigma_points(x);
        let (w_m, w_c) = self.weights::<P, Q, R>();

        // Transform sigma points
        let transformed: Vec<Multivector<P, Q, R>> = sigma_points.iter().map(f).collect();

        // Compute transformed mean
        let mut mean = vec![0.0; dim];
        for (i, t) in transformed.iter().enumerate() {
            let t_vec = t.to_vec();
            for j in 0..dim {
                mean[j] += w_m[i] * t_vec[j];
            }
        }

        // Compute transformed covariance
        let mut cov = vec![0.0; dim * dim];
        for (i, t) in transformed.iter().enumerate() {
            let t_vec = t.to_vec();
            for j in 0..dim {
                for k in 0..dim {
                    let diff_j = t_vec[j] - mean[j];
                    let diff_k = t_vec[k] - mean[k];
                    cov[j * dim + k] += w_c[i] * diff_j * diff_k;
                }
            }
        }

        Ok(UncertainMultivector::new(
            Multivector::from_coefficients(mean),
            CovarianceMatrix::from_data(cov)?,
        ))
    }
}

impl Default for UnscentedTransform {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_uncertain_multivector() {
        let mean = Multivector::<2, 0, 0>::scalar(1.0);
        let variances = vec![0.1, 0.2, 0.3, 0.4];
        let um = UncertainMultivector::diagonal(mean, &variances).unwrap();

        assert_relative_eq!(um.total_variance(), 1.0, epsilon = 1e-10);
        assert_eq!(um.variances().len(), 4);
    }

    #[test]
    fn test_linear_scalar_multiply() {
        let mean = Multivector::<2, 0, 0>::scalar(2.0);
        let variances = vec![1.0, 1.0, 1.0, 1.0];
        let um = UncertainMultivector::diagonal(mean, &variances).unwrap();

        let prop = LinearPropagation::new();
        let result = prop.scalar_multiply(&um, 3.0);

        // Mean should be scaled by 3
        assert_relative_eq!(result.mean().get(0), 6.0, epsilon = 1e-10);

        // Variance should be scaled by 9
        for v in result.variances() {
            assert_relative_eq!(v, 9.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_linear_add() {
        let mean1 = Multivector::<2, 0, 0>::scalar(1.0);
        let mean2 = Multivector::<2, 0, 0>::scalar(2.0);
        let var = vec![1.0, 1.0, 1.0, 1.0];

        let um1 = UncertainMultivector::diagonal(mean1, &var).unwrap();
        let um2 = UncertainMultivector::diagonal(mean2, &var).unwrap();

        let prop = LinearPropagation::new();
        let result = prop.add(&um1, &um2);

        // Mean should be sum
        assert_relative_eq!(result.mean().get(0), 3.0, epsilon = 1e-10);

        // Variances should add (assuming independence)
        for v in result.variances() {
            assert_relative_eq!(v, 2.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_unscented_sigma_points() {
        let mean = Multivector::<1, 0, 0>::scalar(0.0);
        let variances = vec![1.0, 1.0];
        let um = UncertainMultivector::diagonal(mean, &variances).unwrap();

        let ut = UnscentedTransform::new();
        let sigma_points = ut.sigma_points(&um);

        // Should have 2n + 1 sigma points (n=2 for Cl(1,0,0))
        assert_eq!(sigma_points.len(), 5);
    }

    #[test]
    fn test_unscented_identity() {
        let mean = Multivector::<2, 0, 0>::scalar(1.0);
        let variances = vec![0.1, 0.1, 0.1, 0.1];
        let um = UncertainMultivector::diagonal(mean.clone(), &variances).unwrap();

        let ut = UnscentedTransform::new();

        // Identity function should preserve mean and covariance
        let result = ut.propagate(&um, |x| x.clone()).unwrap();

        assert_relative_eq!(result.mean().get(0), 1.0, epsilon = 1e-6);
        for (&v1, &v2) in result.variances().iter().zip(variances.iter()) {
            assert_relative_eq!(v1, v2, epsilon = 1e-6);
        }
    }
}
