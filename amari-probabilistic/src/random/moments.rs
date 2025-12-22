//! Moment computation for multivector-valued random variables
//!
//! This module provides tools for computing moments (mean, variance, covariance,
//! higher-order) of distributions over geometric algebra spaces.

use crate::distribution::Distribution;
use crate::error::Result;
use amari_core::Multivector;

/// Trait for random variables with computable moments
///
/// Provides methods for computing expectations, covariances, and higher moments
/// of multivector-valued random variables.
pub trait GeometricRandomVariable<const P: usize, const Q: usize, const R: usize>:
    Distribution<Multivector<P, Q, R>>
{
    /// Dimension of the multivector space
    const DIM: usize = 1 << (P + Q + R);

    /// Compute the expectation (mean) of the random variable
    fn expectation(&self) -> Multivector<P, Q, R>;

    /// Compute the covariance matrix
    ///
    /// Returns a DIM×DIM covariance matrix in row-major order.
    fn covariance(&self) -> CovarianceMatrix<P, Q, R>;

    /// Compute the k-th raw moment
    ///
    /// The raw moment E[X^k] where X is the multivector and power is via
    /// repeated geometric product.
    fn raw_moment(&self, k: usize) -> Result<Multivector<P, Q, R>>;

    /// Compute the k-th central moment
    ///
    /// The central moment E[(X - μ)^k].
    fn central_moment(&self, k: usize) -> Result<Multivector<P, Q, R>>;

    /// Compute the characteristic function at a given point
    ///
    /// φ_X(t) = E[exp(i⟨t,X⟩)] where ⟨·,·⟩ is the scalar product.
    fn characteristic_function(&self, t: &Multivector<P, Q, R>) -> Result<(f64, f64)>;
}

/// Covariance matrix for multivector-valued random variables
///
/// Stores the DIM×DIM covariance matrix in row-major order.
#[derive(Debug, Clone)]
pub struct CovarianceMatrix<const P: usize, const Q: usize, const R: usize> {
    /// Flattened covariance matrix (row-major)
    data: Vec<f64>,
}

impl<const P: usize, const Q: usize, const R: usize> CovarianceMatrix<P, Q, R> {
    /// Dimension of the multivector space
    const DIM: usize = 1 << (P + Q + R);

    /// Create a zero covariance matrix
    pub fn zero() -> Self {
        Self {
            data: vec![0.0; Self::DIM * Self::DIM],
        }
    }

    /// Create a diagonal covariance matrix
    pub fn diagonal(variances: &[f64]) -> Result<Self> {
        if variances.len() != Self::DIM {
            return Err(crate::error::ProbabilisticError::dimension_mismatch(
                Self::DIM,
                variances.len(),
            ));
        }

        let mut data = vec![0.0; Self::DIM * Self::DIM];
        for i in 0..Self::DIM {
            data[i * Self::DIM + i] = variances[i];
        }

        Ok(Self { data })
    }

    /// Create from a full matrix (row-major)
    pub fn from_data(data: Vec<f64>) -> Result<Self> {
        if data.len() != Self::DIM * Self::DIM {
            return Err(crate::error::ProbabilisticError::dimension_mismatch(
                Self::DIM * Self::DIM,
                data.len(),
            ));
        }
        Ok(Self { data })
    }

    /// Get element at (i, j)
    pub fn get(&self, i: usize, j: usize) -> f64 {
        self.data[i * Self::DIM + j]
    }

    /// Set element at (i, j)
    pub fn set(&mut self, i: usize, j: usize, value: f64) {
        self.data[i * Self::DIM + j] = value;
    }

    /// Get the diagonal (variances)
    pub fn get_diagonal(&self) -> Vec<f64> {
        (0..Self::DIM).map(|i| self.get(i, i)).collect()
    }

    /// Compute trace (sum of variances)
    pub fn trace(&self) -> f64 {
        (0..Self::DIM).map(|i| self.get(i, i)).sum()
    }

    /// Get the underlying data
    pub fn as_slice(&self) -> &[f64] {
        &self.data
    }
}

/// Helper for computing moments via Monte Carlo sampling
pub struct MomentComputer<const P: usize, const Q: usize, const R: usize> {
    /// Number of samples to use
    pub num_samples: usize,
}

impl<const P: usize, const Q: usize, const R: usize> MomentComputer<P, Q, R> {
    /// Dimension of the multivector space
    const DIM: usize = 1 << (P + Q + R);

    /// Create with default number of samples (10000)
    pub fn new() -> Self {
        Self { num_samples: 10000 }
    }

    /// Create with specified number of samples
    pub fn with_samples(num_samples: usize) -> Self {
        Self { num_samples }
    }

    /// Estimate mean via Monte Carlo
    pub fn estimate_mean<D>(&self, dist: &D) -> Multivector<P, Q, R>
    where
        D: Distribution<Multivector<P, Q, R>>,
    {
        let mut rng = rand::thread_rng();
        let samples = dist.sample_n(&mut rng, self.num_samples);

        let mut sum = Multivector::zero();
        for sample in &samples {
            sum = sum.add(sample);
        }

        // Scale by 1/n
        let scale = 1.0 / (self.num_samples as f64);
        let coeffs: Vec<f64> = sum.to_vec().iter().map(|&x| x * scale).collect();
        Multivector::from_coefficients(coeffs)
    }

    /// Estimate covariance via Monte Carlo
    pub fn estimate_covariance<D>(&self, dist: &D) -> CovarianceMatrix<P, Q, R>
    where
        D: Distribution<Multivector<P, Q, R>>,
    {
        let mut rng = rand::thread_rng();
        let samples = dist.sample_n(&mut rng, self.num_samples);

        // Compute mean first
        let mean = self.estimate_mean(dist);
        let mean_vec = mean.to_vec();

        // Accumulate outer products
        let mut cov_sum = vec![0.0; Self::DIM * Self::DIM];

        for sample in &samples {
            let sample_vec = sample.to_vec();

            for i in 0..Self::DIM {
                for j in 0..Self::DIM {
                    let diff_i = sample_vec[i] - mean_vec[i];
                    let diff_j = sample_vec[j] - mean_vec[j];
                    cov_sum[i * Self::DIM + j] += diff_i * diff_j;
                }
            }
        }

        // Normalize
        let scale = 1.0 / (self.num_samples as f64 - 1.0);
        let cov_data: Vec<f64> = cov_sum.iter().map(|&x| x * scale).collect();

        CovarianceMatrix::from_data(cov_data).unwrap()
    }
}

impl<const P: usize, const Q: usize, const R: usize> Default for MomentComputer<P, Q, R> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distribution::GaussianMultivector;
    use approx::assert_relative_eq;

    #[test]
    fn test_covariance_matrix_diagonal() {
        let variances = vec![1.0, 2.0, 3.0, 4.0];
        let cov = CovarianceMatrix::<2, 0, 0>::diagonal(&variances).unwrap();

        // Check diagonal elements
        for (i, &var) in variances.iter().enumerate() {
            assert_relative_eq!(cov.get(i, i), var, epsilon = 1e-10);
        }

        // Check off-diagonal elements are zero
        for i in 0..4 {
            for j in 0..4 {
                if i != j {
                    assert_relative_eq!(cov.get(i, j), 0.0, epsilon = 1e-10);
                }
            }
        }
    }

    #[test]
    fn test_covariance_matrix_trace() {
        let variances = vec![1.0, 2.0, 3.0, 4.0];
        let cov = CovarianceMatrix::<2, 0, 0>::diagonal(&variances).unwrap();

        assert_relative_eq!(cov.trace(), 10.0, epsilon = 1e-10);
    }

    #[test]
    fn test_moment_computer_mean_estimate() {
        let gaussian = GaussianMultivector::<2, 0, 0>::standard();
        let computer = MomentComputer::<2, 0, 0>::with_samples(10000);

        let mean = computer.estimate_mean(&gaussian);

        // Mean should be close to zero for standard Gaussian
        for i in 0..4 {
            assert!(mean.get(i).abs() < 0.1); // Generous tolerance for MC
        }
    }
}
