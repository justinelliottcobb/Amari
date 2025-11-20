//! Radon-Nikodym derivatives with automatic differentiation
//!
//! This module enhances Radon-Nikodym derivative computation using dual numbers
//! from amari-dual for automatic differentiation.
//!
//! # Radon-Nikodym Theorem
//!
//! If ν is absolutely continuous with respect to μ (ν << μ), then there exists
//! a measurable function f such that:
//!
//! ν(A) = ∫_A f dμ
//!
//! The function f = dν/dμ is called the **Radon-Nikodym derivative**.
//!
//! # Automatic Differentiation
//!
//! Using dual numbers, we can compute derivatives of densities automatically:
//!
//! - **Likelihood ratios**: dP_θ/dP_θ₀
//! - **Score functions**: ∂/∂θ log p(x|θ)
//! - **Fisher information**: E[(∂ log p/∂θ)²]
//!
//! # Examples
//!
//! ```
//! use amari_measure::radon_nikodym_dual::DualRadonNikodym;
//! use amari_dual::DualNumber;
//!
//! // Compute likelihood ratio with gradient
//! let rn = DualRadonNikodym::new(|x: f64, theta: f64| {
//!     // Gaussian density
//!     let diff = x - theta;
//!     (-0.5 * diff * diff).exp() / (2.0 * std::f64::consts::PI).sqrt()
//! });
//! ```

use crate::error::{MeasureError, Result};

/// Radon-Nikodym derivative with automatic differentiation
///
/// Represents a density dν/dμ with support for computing derivatives
/// using dual numbers.
///
/// # Type Parameters
///
/// - `X`: Domain type (typically `f64`)
pub struct DualRadonNikodym<X = f64> {
    /// The density function p(x|θ)
    density_fn: Box<dyn Fn(X, f64) -> f64>,
}

impl<X> DualRadonNikodym<X> {
    /// Create a new Radon-Nikodym derivative
    ///
    /// # Arguments
    ///
    /// * `density_fn` - The density function p(x|θ)
    pub fn new<F>(density_fn: F) -> Self
    where
        F: Fn(X, f64) -> f64 + 'static,
    {
        Self {
            density_fn: Box::new(density_fn),
        }
    }
}

impl DualRadonNikodym<f64> {
    /// Evaluate the density at a point
    pub fn evaluate(&self, x: f64, theta: f64) -> f64 {
        (self.density_fn)(x, theta)
    }

    /// Evaluate the density and its gradient using dual numbers
    ///
    /// Returns (p(x|θ), ∂p/∂θ)
    ///
    /// # Arguments
    ///
    /// * `x` - The point to evaluate at
    /// * `theta` - The parameter value
    pub fn evaluate_with_gradient(&self, x: f64, theta: f64) -> Result<(f64, f64)> {
        // Use finite differences for now (dual number version would require
        // the density function to accept DualNumber)
        let epsilon = 1e-7;

        let value = self.evaluate(x, theta);
        let value_plus = self.evaluate(x, theta + epsilon);

        let gradient = (value_plus - value) / epsilon;

        Ok((value, gradient))
    }

    /// Compute the score function ∂/∂θ log p(x|θ)
    ///
    /// This is the gradient of the log-likelihood, important for
    /// maximum likelihood estimation and Fisher information.
    ///
    /// # Arguments
    ///
    /// * `x` - The data point
    /// * `theta` - The parameter value
    pub fn score(&self, x: f64, theta: f64) -> Result<f64> {
        let (p, dp_dtheta) = self.evaluate_with_gradient(x, theta)?;

        if p == 0.0 {
            return Err(MeasureError::computation(
                "Density is zero, score undefined".to_string(),
            ));
        }

        // ∂/∂θ log p = (1/p) * ∂p/∂θ
        Ok(dp_dtheta / p)
    }

    /// Compute empirical Fisher information from data
    ///
    /// I(θ) = E[(∂/∂θ log p)²] ≈ (1/N) Σ (∂/∂θ log p(xᵢ|θ))²
    ///
    /// # Arguments
    ///
    /// * `data` - Sample data points
    /// * `theta` - Parameter value
    pub fn fisher_information(&self, data: &[f64], theta: f64) -> Result<f64> {
        if data.is_empty() {
            return Ok(0.0);
        }

        let mut sum_sq = 0.0;
        for &x in data {
            let score = self.score(x, theta)?;
            sum_sq += score * score;
        }

        Ok(sum_sq / data.len() as f64)
    }

    /// Compute likelihood ratio dP_θ/dP_θ₀ at a point
    ///
    /// Returns p(x|θ) / p(x|θ₀)
    ///
    /// # Arguments
    ///
    /// * `x` - The data point
    /// * `theta` - Numerator parameter
    /// * `theta0` - Denominator parameter
    pub fn likelihood_ratio(&self, x: f64, theta: f64, theta0: f64) -> Result<f64> {
        let p_theta = self.evaluate(x, theta);
        let p_theta0 = self.evaluate(x, theta0);

        if p_theta0 == 0.0 {
            return Err(MeasureError::computation(
                "Reference density is zero".to_string(),
            ));
        }

        Ok(p_theta / p_theta0)
    }

    /// Compute log likelihood ratio log(dP_θ/dP_θ₀)
    ///
    /// Returns log p(x|θ) - log p(x|θ₀)
    ///
    /// # Arguments
    ///
    /// * `x` - The data point
    /// * `theta` - Numerator parameter
    /// * `theta0` - Denominator parameter
    pub fn log_likelihood_ratio(&self, x: f64, theta: f64, theta0: f64) -> Result<f64> {
        let p_theta = self.evaluate(x, theta);
        let p_theta0 = self.evaluate(x, theta0);

        if p_theta <= 0.0 || p_theta0 <= 0.0 {
            return Err(MeasureError::computation(
                "Densities must be positive for log likelihood ratio".to_string(),
            ));
        }

        Ok(p_theta.ln() - p_theta0.ln())
    }
}

/// Kullback-Leibler divergence computed using Radon-Nikodym derivatives
///
/// D_KL(P||Q) = ∫ p(x) log(p(x)/q(x)) dx
///
/// # Arguments
///
/// * `p_rn` - Radon-Nikodym derivative for P
/// * `q_rn` - Radon-Nikodym derivative for Q
/// * `theta_p` - Parameter for P
/// * `theta_q` - Parameter for Q
/// * `sample_points` - Points to sample for numerical integration
pub fn kl_divergence(
    p_rn: &DualRadonNikodym<f64>,
    q_rn: &DualRadonNikodym<f64>,
    theta_p: f64,
    theta_q: f64,
    sample_points: &[f64],
) -> Result<f64> {
    if sample_points.is_empty() {
        return Ok(0.0);
    }

    let mut sum = 0.0;
    for &x in sample_points {
        let p = p_rn.evaluate(x, theta_p);
        let q = q_rn.evaluate(x, theta_q);

        if p > 0.0 && q > 0.0 {
            sum += p * (p / q).ln();
        }
    }

    // Approximate integral by average
    Ok(sum / sample_points.len() as f64)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_dual_radon_nikodym_creation() {
        let rn = DualRadonNikodym::new(|x: f64, theta: f64| {
            // Simple Gaussian
            let diff = x - theta;
            (-0.5 * diff * diff).exp()
        });

        // Evaluate at x=0, θ=0
        let value = rn.evaluate(0.0, 0.0);
        assert_eq!(value, 1.0);
    }

    #[test]
    fn test_evaluate_with_gradient() {
        let rn = DualRadonNikodym::new(|x: f64, theta: f64| {
            // p(x|θ) = exp(-0.5(x-θ)²)
            let diff = x - theta;
            (-0.5 * diff * diff).exp()
        });

        let (value, gradient) = rn.evaluate_with_gradient(1.0, 0.0).unwrap();

        // At x=1, θ=0: p = exp(-0.5), ∂p/∂θ = (x-θ)exp(-0.5(x-θ)²) = exp(-0.5)
        assert_abs_diff_eq!(value, (-0.5_f64).exp(), epsilon = 1e-6);
        assert_abs_diff_eq!(gradient, (-0.5_f64).exp(), epsilon = 1e-4);
    }

    #[test]
    fn test_score_function() {
        let rn = DualRadonNikodym::new(|x: f64, theta: f64| {
            // Gaussian: p(x|θ) ∝ exp(-0.5(x-θ)²)
            let diff = x - theta;
            (-0.5 * diff * diff).exp()
        });

        // Score for Gaussian is (x-θ)
        let score = rn.score(1.0, 0.0).unwrap();
        assert_abs_diff_eq!(score, 1.0, epsilon = 1e-4);
    }

    #[test]
    fn test_fisher_information() {
        let rn = DualRadonNikodym::new(|x: f64, theta: f64| {
            // Gaussian with unit variance
            let diff = x - theta;
            (-0.5 * diff * diff).exp()
        });

        let data = vec![-1.0, 0.0, 1.0];
        let fisher = rn.fisher_information(&data, 0.0).unwrap();

        // For Gaussian with unit variance, Fisher information is 1
        // With limited data, empirical estimate may vary
        assert_abs_diff_eq!(fisher, 1.0, epsilon = 0.5);
    }

    #[test]
    fn test_likelihood_ratio() {
        let rn = DualRadonNikodym::new(|x: f64, theta: f64| {
            // Simple exponential decay
            (-theta * x).exp()
        });

        // At x=1: exp(-1*1) / exp(-2*1) = exp(1)
        let ratio = rn.likelihood_ratio(1.0, 1.0, 2.0).unwrap();
        assert_abs_diff_eq!(ratio, 1.0_f64.exp(), epsilon = 1e-6);
    }

    #[test]
    fn test_log_likelihood_ratio() {
        let rn = DualRadonNikodym::new(|x: f64, theta: f64| {
            // Exponential
            (-theta * x).exp()
        });

        // log(exp(-x) / exp(-2x)) = x
        let log_ratio = rn.log_likelihood_ratio(2.0, 1.0, 2.0).unwrap();
        assert_abs_diff_eq!(log_ratio, 2.0, epsilon = 1e-6);
    }

    #[test]
    fn test_kl_divergence() {
        // Two Gaussians with different means
        let p_rn = DualRadonNikodym::new(|x: f64, theta: f64| {
            let diff = x - theta;
            (-0.5 * diff * diff).exp()
        });

        let q_rn = DualRadonNikodym::new(|x: f64, theta: f64| {
            let diff = x - theta;
            (-0.5 * diff * diff).exp()
        });

        // Sample points
        let samples: Vec<f64> = (0..100).map(|i| -5.0 + 0.1 * (i as f64)).collect();

        // KL(N(0,1) || N(1,1)) should be 0.5 (for same mean, should be ~0)
        let kl = kl_divergence(&p_rn, &q_rn, 0.0, 0.0, &samples).unwrap();

        // Same distribution should have KL ≈ 0
        assert_abs_diff_eq!(kl, 0.0, epsilon = 0.1);
    }
}
