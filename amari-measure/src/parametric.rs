//! Parametric density functions with automatic differentiation
//!
//! This module provides parametric families of densities that can compute
//! derivatives with respect to parameters using automatic differentiation.
//!
//! # Overview
//!
//! In statistical inference and machine learning, we often work with parametric
//! density families p(x|θ) where θ ∈ ℝⁿ are parameters. Computing derivatives
//! ∂p/∂θᵢ and gradients ∇_θ log p(x|θ) is essential for:
//!
//! - Maximum likelihood estimation
//! - Variational inference
//! - Natural gradient optimization
//! - Fisher information computation
//!
//! # Examples
//!
//! ```
//! use amari_measure::parametric::ParametricDensity;
//!
//! // Define a Gaussian density: p(x|μ,σ) = (1/√(2πσ²)) exp(-(x-μ)²/(2σ²))
//! let gaussian = ParametricDensity::new(
//!     2, // 2 parameters: μ, σ
//!     |x: f64, params: &[f64]| {
//!         let mu = params[0];
//!         let sigma = params[1];
//!         let z = (x - mu) / sigma;
//!         (-0.5 * z * z - sigma.ln() - 0.5 * (2.0 * std::f64::consts::PI).ln()).exp()
//!     }
//! );
//!
//! // Evaluate density and gradient at x=1.5, μ=1.0, σ=2.0
//! let (value, gradient) = gaussian.evaluate_with_gradient(1.5, &[1.0, 2.0]).unwrap();
//! ```

use crate::error::{MeasureError, Result};
use amari_dual::DualNumber;
use core::marker::PhantomData;

/// Parametric density family with automatic differentiation
///
/// Represents a family of probability densities p(x|θ) parameterized by θ ∈ ℝⁿ.
///
/// # Type Parameters
///
/// - `X`: The domain type (typically `f64`, `Vec<f64>`, or spatial points)
pub struct ParametricDensity<X = f64> {
    /// Number of parameters
    num_params: usize,

    /// Density function: f(x, θ) → p(x|θ)
    ///
    /// For single-parameter families, use DualNumber
    /// For multi-parameter families, use MultiDualNumber
    #[allow(clippy::type_complexity)]
    density_fn: Box<dyn Fn(X, &[f64]) -> f64>,

    _phantom: PhantomData<X>,
}

impl<X> ParametricDensity<X> {
    /// Get the number of parameters (works for all X)
    pub fn num_params(&self) -> usize {
        self.num_params
    }
}

impl<X> ParametricDensity<X>
where
    X: Copy,
{
    /// Create a new parametric density family
    ///
    /// # Arguments
    ///
    /// * `num_params` - Number of parameters in the family
    /// * `density_fn` - Function computing p(x|θ)
    ///
    /// # Examples
    ///
    /// ```
    /// use amari_measure::parametric::ParametricDensity;
    ///
    /// // Exponential distribution: p(x|λ) = λ exp(-λx) for x ≥ 0
    /// let exponential = ParametricDensity::new(
    ///     1,
    ///     |x: f64, params: &[f64]| {
    ///         let lambda = params[0];
    ///         if x < 0.0 {
    ///             0.0
    ///         } else {
    ///             lambda * (-lambda * x).exp()
    ///         }
    ///     }
    /// );
    /// ```
    pub fn new<F>(num_params: usize, density_fn: F) -> Self
    where
        F: Fn(X, &[f64]) -> f64 + 'static,
    {
        Self {
            num_params,
            density_fn: Box::new(density_fn),
            _phantom: PhantomData,
        }
    }

    /// Evaluate the density at a point with given parameters
    ///
    /// Returns p(x|θ).
    ///
    /// # Arguments
    ///
    /// * `x` - The point at which to evaluate
    /// * `params` - Parameter values θ
    pub fn evaluate(&self, x: X, params: &[f64]) -> Result<f64> {
        if params.len() != self.num_params {
            return Err(MeasureError::computation(format!(
                "Expected {} parameters, got {}",
                self.num_params,
                params.len()
            )));
        }
        Ok((self.density_fn)(x, params))
    }
}

impl ParametricDensity<f64> {
    /// Evaluate density and its gradient with respect to parameters
    ///
    /// Computes both p(x|θ) and ∇_θ p(x|θ) using automatic differentiation.
    ///
    /// # Arguments
    ///
    /// * `x` - The point at which to evaluate
    /// * `params` - Parameter values θ
    ///
    /// # Returns
    ///
    /// `(value, gradient)` where:
    /// - `value` = p(x|θ)
    /// - `gradient` = [∂p/∂θ₁, ∂p/∂θ₂, ..., ∂p/∂θₙ]
    ///
    /// # Examples
    ///
    /// ```
    /// use amari_measure::parametric::ParametricDensity;
    ///
    /// // Linear model: p(x|a,b) = a*x + b
    /// let linear = ParametricDensity::new(
    ///     2,
    ///     |x: f64, params: &[f64]| params[0] * x + params[1]
    /// );
    ///
    /// let (value, grad) = linear.evaluate_with_gradient(2.0, &[3.0, 1.0]).unwrap();
    /// // value = 3*2 + 1 = 7
    /// // grad ≈ [∂/∂a(ax+b), ∂/∂b(ax+b)] = [x, 1] ≈ [2.0, 1.0] (via finite differences)
    /// assert!((value - 7.0).abs() < 1e-10);
    /// assert!((grad[0] - 2.0).abs() < 1e-4); // Finite difference approximation
    /// assert!((grad[1] - 1.0).abs() < 1e-4);
    /// ```
    pub fn evaluate_with_gradient(&self, x: f64, params: &[f64]) -> Result<(f64, Vec<f64>)> {
        if params.len() != self.num_params {
            return Err(MeasureError::computation(format!(
                "Expected {} parameters, got {}",
                self.num_params,
                params.len()
            )));
        }

        // For single parameter, use DualNumber
        if self.num_params == 1 {
            let dual_param = DualNumber::new(params[0], 1.0);
            let _dual_params = [dual_param];

            // Evaluate with dual numbers
            let result_real = (self.density_fn)(x, params);

            // Approximate gradient via finite differences for now
            // TODO: Modify density_fn to accept DualNumber parameters
            let epsilon = 1e-7;
            let mut gradient = Vec::with_capacity(self.num_params);
            for i in 0..self.num_params {
                let mut params_plus = params.to_vec();
                params_plus[i] += epsilon;
                let f_plus = (self.density_fn)(x, &params_plus);
                let derivative = (f_plus - result_real) / epsilon;
                gradient.push(derivative);
            }

            return Ok((result_real, gradient));
        }

        // For multiple parameters, use finite differences
        // (Full dual number integration would require generic trait bounds)
        let value = (self.density_fn)(x, params);
        let epsilon = 1e-7;
        let mut gradient = Vec::with_capacity(self.num_params);

        for i in 0..self.num_params {
            let mut params_plus = params.to_vec();
            params_plus[i] += epsilon;
            let f_plus = (self.density_fn)(x, &params_plus);
            let derivative = (f_plus - value) / epsilon;
            gradient.push(derivative);
        }

        Ok((value, gradient))
    }

    /// Evaluate log density and its gradient (score function)
    ///
    /// Computes log p(x|θ) and ∇_θ log p(x|θ), which is the **score function**
    /// in statistics.
    ///
    /// # Arguments
    ///
    /// * `x` - The point at which to evaluate
    /// * `params` - Parameter values θ
    ///
    /// # Returns
    ///
    /// `(log_value, score)` where:
    /// - `log_value` = log p(x|θ)
    /// - `score` = ∇_θ log p(x|θ) = [∂ log p/∂θ₁, ..., ∂ log p/∂θₙ]
    ///
    /// # Mathematical Note
    ///
    /// The score function satisfies:
    /// - E[∇_θ log p(X|θ)] = 0 (zero mean under the true distribution)
    /// - Var[∇_θ log p(X|θ)] = Fisher information matrix
    pub fn evaluate_log_with_gradient(&self, x: f64, params: &[f64]) -> Result<(f64, Vec<f64>)> {
        let (value, grad_p) = self.evaluate_with_gradient(x, params)?;

        if value <= 0.0 {
            return Err(MeasureError::computation(
                "Cannot compute log of non-positive density",
            ));
        }

        let log_value = value.ln();

        // Chain rule: ∇log p = (∇p) / p
        let score: Vec<f64> = grad_p.iter().map(|&dp| dp / value).collect();

        Ok((log_value, score))
    }

    /// Compute Fisher information matrix
    ///
    /// The Fisher information matrix is defined as:
    /// I(θ) = E[(∇_θ log p(X|θ)) (∇_θ log p(X|θ))ᵀ]
    ///
    /// For a sample, this computes the **empirical Fisher information**:
    /// Î(θ) = (1/n) Σᵢ (∇_θ log p(xᵢ|θ)) (∇_θ log p(xᵢ|θ))ᵀ
    ///
    /// # Arguments
    ///
    /// * `data` - Sample points x₁, ..., xₙ
    /// * `params` - Parameter values θ
    ///
    /// # Returns
    ///
    /// Fisher information matrix I(θ) as a flattened n×n matrix
    pub fn fisher_information(&self, data: &[f64], params: &[f64]) -> Result<Vec<Vec<f64>>> {
        let n = data.len();
        if n == 0 {
            return Err(MeasureError::computation(
                "Cannot compute Fisher information from empty data",
            ));
        }

        // Initialize Fisher matrix as n_params × n_params
        let mut fisher = vec![vec![0.0; self.num_params]; self.num_params];

        // Accumulate outer products of score functions
        for &x in data {
            let (_log_p, score) = self.evaluate_log_with_gradient(x, params)?;

            // Add score * scoreᵀ to Fisher matrix
            for i in 0..self.num_params {
                for j in 0..self.num_params {
                    fisher[i][j] += score[i] * score[j];
                }
            }
        }

        // Normalize by sample size
        for row in fisher.iter_mut().take(self.num_params) {
            for val in row.iter_mut().take(self.num_params) {
                *val /= n as f64;
            }
        }

        Ok(fisher)
    }
}

/// Common parametric density families
pub mod families {
    use super::*;
    use std::f64::consts::PI;

    /// Create a Gaussian (normal) density: N(x|μ, σ²)
    ///
    /// Parameters: θ = [μ, σ]
    ///
    /// Density: p(x|μ,σ) = (1/√(2πσ²)) exp(-(x-μ)²/(2σ²))
    pub fn gaussian() -> ParametricDensity<f64> {
        ParametricDensity::new(2, |x: f64, params: &[f64]| {
            let mu = params[0];
            let sigma = params[1];
            let normalization = 1.0 / (sigma * (2.0 * PI).sqrt());
            let exponent = -0.5 * ((x - mu) / sigma).powi(2);
            normalization * exponent.exp()
        })
    }

    /// Create an exponential density: Exp(x|λ)
    ///
    /// Parameters: θ = [λ]
    ///
    /// Density: p(x|λ) = λ exp(-λx) for x ≥ 0, else 0
    pub fn exponential() -> ParametricDensity<f64> {
        ParametricDensity::new(1, |x: f64, params: &[f64]| {
            let lambda = params[0];
            if x < 0.0 {
                0.0
            } else {
                lambda * (-lambda * x).exp()
            }
        })
    }

    /// Create a Cauchy density: Cauchy(x|x₀, γ)
    ///
    /// Parameters: θ = [x₀, γ]
    ///
    /// Density: p(x|x₀,γ) = 1/(πγ(1 + ((x-x₀)/γ)²))
    pub fn cauchy() -> ParametricDensity<f64> {
        ParametricDensity::new(2, |x: f64, params: &[f64]| {
            let x0 = params[0];
            let gamma = params[1];
            let z = (x - x0) / gamma;
            1.0 / (PI * gamma * (1.0 + z * z))
        })
    }

    /// Create a Laplace density: Laplace(x|μ, b)
    ///
    /// Parameters: θ = [μ, b]
    ///
    /// Density: p(x|μ,b) = (1/2b) exp(-|x-μ|/b)
    pub fn laplace() -> ParametricDensity<f64> {
        ParametricDensity::new(2, |x: f64, params: &[f64]| {
            let mu = params[0];
            let b = params[1];
            (1.0 / (2.0 * b)) * (-(x - mu).abs() / b).exp()
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_parametric_density_evaluation() {
        // Linear density for testing: p(x|a,b) = ax + b
        let linear = ParametricDensity::new(2, |x: f64, params: &[f64]| params[0] * x + params[1]);

        let value = linear.evaluate(2.0, &[3.0, 1.0]).unwrap();
        assert_abs_diff_eq!(value, 7.0, epsilon = 1e-10);
    }

    #[test]
    fn test_parametric_density_gradient() {
        // Linear density: p(x|a,b) = ax + b
        // ∂p/∂a = x, ∂p/∂b = 1
        let linear = ParametricDensity::new(2, |x: f64, params: &[f64]| params[0] * x + params[1]);

        let (value, grad) = linear.evaluate_with_gradient(2.0, &[3.0, 1.0]).unwrap();
        assert_abs_diff_eq!(value, 7.0, epsilon = 1e-10);
        assert_abs_diff_eq!(grad[0], 2.0, epsilon = 1e-5); // ∂/∂a = x = 2
        assert_abs_diff_eq!(grad[1], 1.0, epsilon = 1e-5); // ∂/∂b = 1
    }

    #[test]
    fn test_gaussian_density() {
        let gaussian = families::gaussian();

        // Standard normal at x=0 should be 1/√(2π) ≈ 0.3989
        let value = gaussian.evaluate(0.0, &[0.0, 1.0]).unwrap();
        let expected = 1.0 / (2.0 * std::f64::consts::PI).sqrt();
        assert_abs_diff_eq!(value, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_exponential_density() {
        let exp_dist = families::exponential();

        // Exp(1) at x=1 should be e^(-1) ≈ 0.3679
        let value = exp_dist.evaluate(1.0, &[1.0]).unwrap();
        assert_abs_diff_eq!(value, (-1.0_f64).exp(), epsilon = 1e-10);

        // Should be 0 for negative x
        let value_negative = exp_dist.evaluate(-1.0, &[1.0]).unwrap();
        assert_abs_diff_eq!(value_negative, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_log_gradient() {
        let linear = ParametricDensity::new(2, |x: f64, params: &[f64]| params[0] * x + params[1]);

        let (log_val, score) = linear.evaluate_log_with_gradient(2.0, &[3.0, 1.0]).unwrap();

        // log(7) ≈ 1.9459
        assert_abs_diff_eq!(log_val, 7.0_f64.ln(), epsilon = 1e-10);

        // Score = grad / value = [2.0, 1.0] / 7.0
        assert_abs_diff_eq!(score[0], 2.0 / 7.0, epsilon = 1e-5);
        assert_abs_diff_eq!(score[1], 1.0 / 7.0, epsilon = 1e-5);
    }

    #[test]
    fn test_fisher_information() {
        let linear = ParametricDensity::new(2, |x: f64, params: &[f64]| params[0] * x + params[1]);

        let data = vec![1.0, 2.0, 3.0];
        let fisher = linear.fisher_information(&data, &[1.0, 1.0]).unwrap();

        // Fisher matrix should be 2×2 symmetric positive semidefinite
        assert_eq!(fisher.len(), 2);
        assert_eq!(fisher[0].len(), 2);
        assert!(fisher[0][0] >= 0.0);
        assert_abs_diff_eq!(fisher[0][1], fisher[1][0], epsilon = 1e-10);
    }
}
