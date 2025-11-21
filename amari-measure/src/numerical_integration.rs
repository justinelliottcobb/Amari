//! Numerical integration algorithms for measure theory
//!
//! This module provides concrete numerical methods for computing integrals
//! with respect to measures, complementing the abstract integration framework.
//!
//! # Algorithms
//!
//! - **Monte Carlo Integration**: ∫ f dμ ≈ (1/N) Σ f(xᵢ) for random samples xᵢ
//! - **Adaptive Quadrature**: Recursive subdivision for 1D integrals
//! - **Trapezoidal Rule**: Simple numerical quadrature
//! - **Simpson's Rule**: Higher-order quadrature method
//!
//! # Examples
//!
//! ```
//! use amari_measure::numerical_integration::monte_carlo_integrate;
//!
//! // Integrate f(x) = x² over [0, 1] with 10000 samples
//! let result = monte_carlo_integrate(
//!     &|x: f64| x * x,
//!     0.0,
//!     1.0,
//!     10000
//! );
//! // Result should be close to 1/3 ≈ 0.333...
//! ```

use crate::error::{MeasureError, Result};
use rand::Rng;

/// Monte Carlo integration over an interval
///
/// Approximates ∫_a^b f(x) dx using random sampling:
/// ∫ f dx ≈ (b-a) * (1/N) Σᵢ f(xᵢ)
///
/// # Arguments
///
/// * `f` - The function to integrate
/// * `a` - Lower bound
/// * `b` - Upper bound
/// * `num_samples` - Number of random samples
///
/// # Returns
///
/// Approximate integral value
pub fn monte_carlo_integrate<F>(f: &F, a: f64, b: f64, num_samples: usize) -> Result<f64>
where
    F: Fn(f64) -> f64,
{
    if a >= b {
        return Err(MeasureError::computation(format!(
            "Invalid interval: [{}, {}]",
            a, b
        )));
    }

    if num_samples == 0 {
        return Err(MeasureError::computation(
            "Number of samples must be positive".to_string(),
        ));
    }

    let mut rng = rand::thread_rng();
    let mut sum = 0.0;

    for _ in 0..num_samples {
        let x = rng.gen_range(a..b);
        sum += f(x);
    }

    let average = sum / (num_samples as f64);
    let integral = (b - a) * average;

    Ok(integral)
}

/// Trapezoidal rule for numerical integration
///
/// Approximates ∫_a^b f(x) dx using the trapezoidal rule:
/// ∫ f dx ≈ (b-a)/2N * Σᵢ (f(xᵢ) + f(xᵢ₊₁))
///
/// # Arguments
///
/// * `f` - The function to integrate
/// * `a` - Lower bound
/// * `b` - Upper bound
/// * `num_intervals` - Number of subdivisions
///
/// # Returns
///
/// Approximate integral value
pub fn trapezoidal_integrate<F>(f: &F, a: f64, b: f64, num_intervals: usize) -> Result<f64>
where
    F: Fn(f64) -> f64,
{
    if a >= b {
        return Err(MeasureError::computation(format!(
            "Invalid interval: [{}, {}]",
            a, b
        )));
    }

    if num_intervals == 0 {
        return Err(MeasureError::computation(
            "Number of intervals must be positive".to_string(),
        ));
    }

    let h = (b - a) / (num_intervals as f64);
    let mut sum = 0.5 * (f(a) + f(b));

    for i in 1..num_intervals {
        let x = a + (i as f64) * h;
        sum += f(x);
    }

    Ok(h * sum)
}

/// Simpson's rule for numerical integration
///
/// Approximates ∫_a^b f(x) dx using Simpson's rule (quadratic interpolation):
/// ∫ f dx ≈ (h/3) * [f(x₀) + 4Σf(x_odd) + 2Σf(x_even) + f(xₙ)]
///
/// # Arguments
///
/// * `f` - The function to integrate
/// * `a` - Lower bound
/// * `b` - Upper bound
/// * `num_intervals` - Number of subdivisions (must be even)
///
/// # Returns
///
/// Approximate integral value
pub fn simpson_integrate<F>(f: &F, a: f64, b: f64, num_intervals: usize) -> Result<f64>
where
    F: Fn(f64) -> f64,
{
    if a >= b {
        return Err(MeasureError::computation(format!(
            "Invalid interval: [{}, {}]",
            a, b
        )));
    }

    if num_intervals == 0 || num_intervals % 2 != 0 {
        return Err(MeasureError::computation(
            "Number of intervals must be positive and even".to_string(),
        ));
    }

    let h = (b - a) / (num_intervals as f64);
    let mut sum = f(a) + f(b);

    for i in 1..num_intervals {
        let x = a + (i as f64) * h;
        if i % 2 == 0 {
            sum += 2.0 * f(x);
        } else {
            sum += 4.0 * f(x);
        }
    }

    Ok((h / 3.0) * sum)
}

/// Adaptive quadrature with error estimation
///
/// Recursively subdivides intervals until desired tolerance is achieved.
///
/// # Arguments
///
/// * `f` - The function to integrate
/// * `a` - Lower bound
/// * `b` - Upper bound
/// * `tolerance` - Desired absolute error
/// * `max_depth` - Maximum recursion depth
///
/// # Returns
///
/// Approximate integral value
pub fn adaptive_quadrature<F>(
    f: &F,
    a: f64,
    b: f64,
    tolerance: f64,
    max_depth: usize,
) -> Result<f64>
where
    F: Fn(f64) -> f64,
{
    adaptive_simpson_recursive(f, a, b, tolerance, f(a), f((a + b) / 2.0), f(b), max_depth)
}

/// Recursive helper for adaptive Simpson's rule
#[allow(clippy::too_many_arguments)]
fn adaptive_simpson_recursive<F>(
    f: &F,
    a: f64,
    b: f64,
    tolerance: f64,
    fa: f64,
    fm: f64,
    fb: f64,
    depth: usize,
) -> Result<f64>
where
    F: Fn(f64) -> f64,
{
    let m = (a + b) / 2.0;
    let h = b - a;

    let left_mid = (a + m) / 2.0;
    let right_mid = (m + b) / 2.0;

    let flm = f(left_mid);
    let frm = f(right_mid);

    // Simpson's rule on whole interval
    let whole = (h / 6.0) * (fa + 4.0 * fm + fb);

    // Simpson's rule on left and right halves
    let left = (h / 12.0) * (fa + 4.0 * flm + fm);
    let right = (h / 12.0) * (fm + 4.0 * frm + fb);
    let halves = left + right;

    // Error estimate
    let error = (halves - whole).abs() / 15.0;

    if error < tolerance || depth == 0 {
        // Accept approximation with Richardson extrapolation
        Ok(halves + (halves - whole) / 15.0)
    } else {
        // Subdivide recursively
        let left_result =
            adaptive_simpson_recursive(f, a, m, tolerance / 2.0, fa, flm, fm, depth - 1)?;
        let right_result =
            adaptive_simpson_recursive(f, m, b, tolerance / 2.0, fm, frm, fb, depth - 1)?;
        Ok(left_result + right_result)
    }
}

/// Multidimensional Monte Carlo integration
///
/// Approximates ∫_R f(x) dx over a rectangular region R = [a₁,b₁] × ... × [aₙ,bₙ]
///
/// # Arguments
///
/// * `f` - The function to integrate (takes a slice of coordinates)
/// * `bounds` - Lower and upper bounds for each dimension: [(a₁, b₁), ..., (aₙ, bₙ)]
/// * `num_samples` - Number of random samples
///
/// # Returns
///
/// Approximate integral value
pub fn multidim_monte_carlo<F>(f: &F, bounds: &[(f64, f64)], num_samples: usize) -> Result<f64>
where
    F: Fn(&[f64]) -> f64,
{
    if bounds.is_empty() {
        return Err(MeasureError::computation(
            "Bounds cannot be empty".to_string(),
        ));
    }

    if num_samples == 0 {
        return Err(MeasureError::computation(
            "Number of samples must be positive".to_string(),
        ));
    }

    // Check bounds validity
    for (a, b) in bounds {
        if a >= b {
            return Err(MeasureError::computation(format!(
                "Invalid bound: [{}, {}]",
                a, b
            )));
        }
    }

    let dimension = bounds.len();
    let mut rng = rand::thread_rng();
    let mut sum = 0.0;

    // Compute volume of integration region
    let volume: f64 = bounds.iter().map(|(a, b)| b - a).product();

    // Monte Carlo sampling
    let mut point = vec![0.0; dimension];
    for _ in 0..num_samples {
        for (i, (a, b)) in bounds.iter().enumerate() {
            point[i] = rng.gen_range(*a..*b);
        }
        sum += f(&point);
    }

    let average = sum / (num_samples as f64);
    Ok(volume * average)
}

/// Integration result with error estimate
#[derive(Debug, Clone, Copy)]
pub struct IntegrationResult {
    /// Estimated integral value
    pub value: f64,

    /// Estimated absolute error
    pub error: f64,

    /// Number of function evaluations
    pub num_evaluations: usize,
}

impl IntegrationResult {
    /// Create a new integration result
    pub fn new(value: f64, error: f64, num_evaluations: usize) -> Self {
        Self {
            value,
            error,
            num_evaluations,
        }
    }

    /// Check if error is within tolerance
    pub fn is_accurate(&self, tolerance: f64) -> bool {
        self.error <= tolerance
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_monte_carlo_integrate() {
        // Integrate x² over [0, 1] should give 1/3
        let result = monte_carlo_integrate(&|x: f64| x * x, 0.0, 1.0, 100000).unwrap();

        assert_abs_diff_eq!(result, 1.0 / 3.0, epsilon = 0.01);
    }

    #[test]
    fn test_trapezoidal_integrate() {
        // Integrate x² over [0, 1] should give 1/3
        let result = trapezoidal_integrate(&|x: f64| x * x, 0.0, 1.0, 1000).unwrap();

        assert_abs_diff_eq!(result, 1.0 / 3.0, epsilon = 1e-4);
    }

    #[test]
    fn test_simpson_integrate() {
        // Integrate x³ over [0, 1] should give 1/4
        let result = simpson_integrate(&|x: f64| x * x * x, 0.0, 1.0, 100).unwrap();

        assert_abs_diff_eq!(result, 1.0 / 4.0, epsilon = 1e-6);
    }

    #[test]
    fn test_simpson_odd_intervals_error() {
        let result = simpson_integrate(&|x: f64| x * x, 0.0, 1.0, 99);
        assert!(result.is_err());
    }

    #[test]
    fn test_adaptive_quadrature() {
        // Integrate sin(x) over [0, π] should give 2
        let result =
            adaptive_quadrature(&|x: f64| x.sin(), 0.0, std::f64::consts::PI, 1e-6, 10).unwrap();

        assert_abs_diff_eq!(result, 2.0, epsilon = 1e-6);
    }

    #[test]
    fn test_multidim_monte_carlo_2d() {
        // Integrate f(x,y) = xy over [0,1] × [0,1] should give 1/4
        let f = |coords: &[f64]| coords[0] * coords[1];
        let bounds = vec![(0.0, 1.0), (0.0, 1.0)];

        let result = multidim_monte_carlo(&f, &bounds, 100000).unwrap();

        assert_abs_diff_eq!(result, 1.0 / 4.0, epsilon = 0.01);
    }

    #[test]
    fn test_multidim_monte_carlo_3d() {
        // Integrate f(x,y,z) = xyz over [0,1]³ should give 1/8
        let f = |coords: &[f64]| coords[0] * coords[1] * coords[2];
        let bounds = vec![(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)];

        let result = multidim_monte_carlo(&f, &bounds, 100000).unwrap();

        assert_abs_diff_eq!(result, 1.0 / 8.0, epsilon = 0.01);
    }

    #[test]
    fn test_integration_result() {
        let result = IntegrationResult::new(1.0, 0.001, 1000);

        assert_eq!(result.value, 1.0);
        assert_eq!(result.error, 0.001);
        assert_eq!(result.num_evaluations, 1000);
        assert!(result.is_accurate(0.01));
        assert!(!result.is_accurate(0.0001));
    }

    #[test]
    fn test_invalid_bounds() {
        let result = monte_carlo_integrate(&|x: f64| x, 1.0, 0.0, 1000);
        assert!(result.is_err());
    }

    #[test]
    fn test_zero_samples() {
        let result = monte_carlo_integrate(&|x: f64| x, 0.0, 1.0, 0);
        assert!(result.is_err());
    }
}
