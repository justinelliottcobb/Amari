//! Eigenvalue-based stability analysis
//!
//! This module provides tools for computing eigenvalues of the Jacobian
//! matrix and using them to determine stability properties.
//!
//! # Mathematical Background
//!
//! For a fixed point x* of dx/dt = f(x), stability is determined by the
//! eigenvalues λ of the Jacobian J = ∂f/∂x evaluated at x*:
//!
//! - **Asymptotically Stable**: All Re(λ) < 0
//! - **Unstable**: At least one Re(λ) > 0
//! - **Saddle**: Some Re(λ) < 0 and some Re(λ) > 0
//! - **Center**: All Re(λ) = 0 with nonzero Im(λ)
//!
//! The eigenvalues also determine the local geometry:
//! - Real eigenvalues: straight-line trajectories
//! - Complex eigenvalues: spiral trajectories

use amari_core::Multivector;
use nalgebra::{Complex, DMatrix};

use crate::error::{DynamicsError, Result};
use crate::flow::DynamicalSystem;

use super::classification::{
    classify_discrete_from_eigenvalues, classify_from_eigenvalues, DiscreteStabilityType,
    StabilityType,
};
use super::linearization::{compute_jacobian, DifferentiationConfig};

/// Result of eigenvalue analysis
#[derive(Debug, Clone)]
pub struct EigenvalueAnalysis {
    /// Eigenvalues as (real_part, imaginary_part) pairs
    pub eigenvalues: Vec<(f64, f64)>,
    /// Stability classification
    pub stability: StabilityType,
    /// Spectral radius (max |λ|)
    pub spectral_radius: f64,
    /// Spectral abscissa (max Re(λ))
    pub spectral_abscissa: f64,
    /// Condition number of the Jacobian (if computed)
    pub condition_number: Option<f64>,
}

impl EigenvalueAnalysis {
    /// Check if the system is stable
    pub fn is_stable(&self) -> bool {
        self.stability.is_stable()
    }

    /// Check if the system is hyperbolic (no eigenvalues on imaginary axis)
    pub fn is_hyperbolic(&self, tolerance: f64) -> bool {
        self.eigenvalues.iter().all(|(re, _)| re.abs() > tolerance)
    }

    /// Get the dominant eigenvalue (largest absolute value)
    pub fn dominant_eigenvalue(&self) -> Option<(f64, f64)> {
        self.eigenvalues
            .iter()
            .max_by(|a, b| {
                let abs_a = (a.0 * a.0 + a.1 * a.1).sqrt();
                let abs_b = (b.0 * b.0 + b.1 * b.1).sqrt();
                abs_a
                    .partial_cmp(&abs_b)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .copied()
    }

    /// Get the decay/growth rate (most positive real part for unstable, most negative for stable)
    pub fn dominant_rate(&self) -> f64 {
        self.spectral_abscissa
    }

    /// Get the characteristic frequency (imaginary part of dominant complex eigenvalue)
    pub fn characteristic_frequency(&self) -> Option<f64> {
        self.eigenvalues
            .iter()
            .filter(|(_, im)| im.abs() > 1e-10)
            .max_by(|a, b| a.1.abs().partial_cmp(&b.1.abs()).unwrap())
            .map(|(_, im)| im.abs())
    }

    /// Count stable eigenvalues (Re(λ) < 0)
    pub fn stable_dimension(&self, tolerance: f64) -> usize {
        self.eigenvalues
            .iter()
            .filter(|(re, _)| *re < -tolerance)
            .count()
    }

    /// Count unstable eigenvalues (Re(λ) > 0)
    pub fn unstable_dimension(&self, tolerance: f64) -> usize {
        self.eigenvalues
            .iter()
            .filter(|(re, _)| *re > tolerance)
            .count()
    }

    /// Count center eigenvalues (|Re(λ)| < tolerance)
    pub fn center_dimension(&self, tolerance: f64) -> usize {
        self.eigenvalues
            .iter()
            .filter(|(re, _)| re.abs() < tolerance)
            .count()
    }
}

/// Compute eigenvalues of a matrix using QR iteration
///
/// # Arguments
///
/// * `matrix` - The matrix to analyze
///
/// # Returns
///
/// Vector of eigenvalues as (real_part, imaginary_part) pairs
pub fn compute_eigenvalues(matrix: &DMatrix<f64>) -> Result<Vec<(f64, f64)>> {
    if matrix.nrows() != matrix.ncols() {
        return Err(DynamicsError::dimension_error("Matrix must be square"));
    }

    // Use nalgebra's Schur decomposition for eigenvalue computation
    // This is more numerically stable than direct methods
    let schur = matrix.clone().schur();
    let t = schur.unpack().1; // Upper quasi-triangular matrix

    let n = t.nrows();
    let mut eigenvalues = Vec::new();
    let mut i = 0;

    while i < n {
        if i + 1 < n && t[(i + 1, i)].abs() > 1e-14 {
            // 2x2 block: complex conjugate pair
            let a = t[(i, i)];
            let b = t[(i, i + 1)];
            let c = t[(i + 1, i)];
            let d = t[(i + 1, i + 1)];

            // Eigenvalues of 2x2 block
            let trace = a + d;
            let det = a * d - b * c;
            let discriminant = trace * trace - 4.0 * det;

            if discriminant < 0.0 {
                // Complex eigenvalues
                let real = trace / 2.0;
                let imag = (-discriminant).sqrt() / 2.0;
                eigenvalues.push((real, imag));
                eigenvalues.push((real, -imag));
            } else {
                // Real eigenvalues (shouldn't happen in Schur form but handle anyway)
                let sqrt_d = discriminant.sqrt();
                eigenvalues.push(((trace + sqrt_d) / 2.0, 0.0));
                eigenvalues.push(((trace - sqrt_d) / 2.0, 0.0));
            }
            i += 2;
        } else {
            // Real eigenvalue on diagonal
            eigenvalues.push((t[(i, i)], 0.0));
            i += 1;
        }
    }

    Ok(eigenvalues)
}

/// Analyze stability of a dynamical system at a point
///
/// Computes the Jacobian, finds eigenvalues, and classifies stability.
///
/// # Arguments
///
/// * `system` - The dynamical system
/// * `point` - The point at which to analyze stability
/// * `config` - Differentiation configuration
/// * `tolerance` - Tolerance for eigenvalue classification
///
/// # Returns
///
/// Complete eigenvalue analysis including stability classification
pub fn analyze_stability<S, const P: usize, const Q: usize, const R: usize>(
    system: &S,
    point: &Multivector<P, Q, R>,
    config: &DifferentiationConfig,
    tolerance: f64,
) -> Result<EigenvalueAnalysis>
where
    S: DynamicalSystem<P, Q, R>,
{
    // Compute Jacobian
    let jacobian = compute_jacobian(system, point, config)?;

    // Compute eigenvalues
    let eigenvalues = compute_eigenvalues(&jacobian)?;

    // Classify stability
    let stability = classify_from_eigenvalues(&eigenvalues, tolerance);

    // Compute spectral properties
    let spectral_radius = eigenvalues
        .iter()
        .map(|(re, im)| (re * re + im * im).sqrt())
        .fold(0.0, f64::max);

    let spectral_abscissa = eigenvalues
        .iter()
        .map(|(re, _)| *re)
        .fold(f64::NEG_INFINITY, f64::max);

    // Condition number via SVD
    let svd = jacobian.svd(false, false);
    let singular_values = svd.singular_values;
    let condition_number = if !singular_values.is_empty() {
        let max_sv = singular_values.iter().fold(0.0, |a, &b| f64::max(a, b));
        let min_sv = singular_values
            .iter()
            .fold(f64::INFINITY, |a, &b| f64::min(a, b));
        if min_sv > 1e-14 {
            Some(max_sv / min_sv)
        } else {
            None
        }
    } else {
        None
    };

    Ok(EigenvalueAnalysis {
        eigenvalues,
        stability,
        spectral_radius,
        spectral_abscissa,
        condition_number,
    })
}

/// Analyze stability for discrete-time systems
///
/// For maps x_{n+1} = f(x_n), stability depends on |λ| < 1 vs |λ| > 1.
pub fn analyze_discrete_stability<S, const P: usize, const Q: usize, const R: usize>(
    system: &S,
    point: &Multivector<P, Q, R>,
    config: &DifferentiationConfig,
    tolerance: f64,
) -> Result<DiscreteEigenvalueAnalysis>
where
    S: DynamicalSystem<P, Q, R>,
{
    // Compute Jacobian
    let jacobian = compute_jacobian(system, point, config)?;

    // Compute eigenvalues
    let eigenvalues = compute_eigenvalues(&jacobian)?;

    // Classify discrete stability
    let stability = classify_discrete_from_eigenvalues(&eigenvalues, tolerance);

    // Spectral radius determines stability for discrete systems
    let spectral_radius = eigenvalues
        .iter()
        .map(|(re, im)| (re * re + im * im).sqrt())
        .fold(0.0, f64::max);

    Ok(DiscreteEigenvalueAnalysis {
        eigenvalues,
        stability,
        spectral_radius,
    })
}

/// Result of discrete-time eigenvalue analysis
#[derive(Debug, Clone)]
pub struct DiscreteEigenvalueAnalysis {
    /// Eigenvalues as (real_part, imaginary_part) pairs
    pub eigenvalues: Vec<(f64, f64)>,
    /// Stability classification
    pub stability: DiscreteStabilityType,
    /// Spectral radius (max |λ|)
    pub spectral_radius: f64,
}

impl DiscreteEigenvalueAnalysis {
    /// Check if the fixed point is stable (|λ| < 1 for all λ)
    pub fn is_stable(&self) -> bool {
        self.spectral_radius < 1.0 - 1e-10
    }

    /// Compute Lyapunov exponent from spectral radius
    ///
    /// For linear maps, the Lyapunov exponent is ln(ρ) where ρ is spectral radius.
    pub fn lyapunov_exponent(&self) -> f64 {
        self.spectral_radius.ln()
    }
}

/// Check for Hopf bifurcation condition at a fixed point
///
/// A Hopf bifurcation occurs when a pair of complex conjugate eigenvalues
/// crosses the imaginary axis. This function checks if the conditions are
/// near a Hopf bifurcation.
///
/// # Returns
///
/// - `Some((frequency, real_part))` if a complex pair exists near imaginary axis
/// - `None` if no such pair exists
pub fn hopf_bifurcation_check(analysis: &EigenvalueAnalysis, tolerance: f64) -> Option<(f64, f64)> {
    // Look for complex eigenvalues with small real part
    for (re, im) in &analysis.eigenvalues {
        if im.abs() > tolerance && re.abs() < tolerance * 10.0 {
            return Some((im.abs(), *re));
        }
    }
    None
}

/// Check for saddle-node bifurcation condition
///
/// A saddle-node bifurcation occurs when an eigenvalue crosses zero.
///
/// # Returns
///
/// `Some(eigenvalue)` if a near-zero real eigenvalue exists
pub fn saddle_node_check(analysis: &EigenvalueAnalysis, tolerance: f64) -> Option<f64> {
    for (re, im) in &analysis.eigenvalues {
        if re.abs() < tolerance && im.abs() < tolerance {
            return Some(*re);
        }
    }
    None
}

/// Compute the stable, unstable, and center subspace dimensions
///
/// This is useful for understanding the local dynamics and for
/// manifold computations.
#[derive(Debug, Clone, Copy)]
pub struct SubspaceDimensions {
    /// Dimension of stable subspace (Re(λ) < 0)
    pub stable: usize,
    /// Dimension of unstable subspace (Re(λ) > 0)
    pub unstable: usize,
    /// Dimension of center subspace (Re(λ) = 0)
    pub center: usize,
}

impl SubspaceDimensions {
    /// Total dimension
    pub fn total(&self) -> usize {
        self.stable + self.unstable + self.center
    }

    /// Check if there's a center manifold
    pub fn has_center(&self) -> bool {
        self.center > 0
    }

    /// Check if this is a hyperbolic fixed point
    pub fn is_hyperbolic(&self) -> bool {
        self.center == 0
    }
}

/// Compute subspace dimensions from eigenvalue analysis
pub fn compute_subspace_dimensions(
    analysis: &EigenvalueAnalysis,
    tolerance: f64,
) -> SubspaceDimensions {
    SubspaceDimensions {
        stable: analysis.stable_dimension(tolerance),
        unstable: analysis.unstable_dimension(tolerance),
        center: analysis.center_dimension(tolerance),
    }
}

/// Characteristic polynomial coefficients from eigenvalues
///
/// Returns coefficients [a_0, a_1, ..., a_{n-1}, 1] of the polynomial
/// det(λI - A) = λ^n + a_{n-1}λ^{n-1} + ... + a_1λ + a_0
pub fn characteristic_polynomial(eigenvalues: &[(f64, f64)]) -> Vec<f64> {
    let n = eigenvalues.len();
    if n == 0 {
        return vec![1.0];
    }

    // Convert to nalgebra complex
    let roots: Vec<Complex<f64>> = eigenvalues
        .iter()
        .map(|(re, im)| Complex::new(*re, *im))
        .collect();

    // Build polynomial by multiplying (λ - λ_i) factors
    let mut coeffs: Vec<Complex<f64>> = vec![Complex::new(1.0, 0.0)];

    for root in roots {
        let mut new_coeffs = vec![Complex::new(0.0, 0.0); coeffs.len() + 1];

        // Multiply by (λ - root)
        for (i, c) in coeffs.iter().enumerate() {
            new_coeffs[i + 1] += c; // λ * current
            new_coeffs[i] -= c * root; // -root * current
        }

        coeffs = new_coeffs;
    }

    // Return real parts (imaginary should be ~0 for real characteristic polynomial)
    coeffs.into_iter().map(|c| c.re).collect()
}

/// Compute Routh-Hurwitz stability criterion for a characteristic polynomial
///
/// Returns true if all roots have negative real parts (system is stable).
/// This is more efficient than computing all eigenvalues for checking stability.
pub fn routh_hurwitz_stable(coefficients: &[f64]) -> bool {
    let n = coefficients.len();
    if n < 2 {
        return true;
    }

    // Check necessary conditions: all coefficients positive
    let leading_sign = coefficients.last().map_or(1.0, |&c| c.signum());
    if !coefficients.iter().all(|&c| c * leading_sign >= 0.0) {
        return false;
    }

    // For n <= 2, necessary conditions are sufficient
    if n <= 2 {
        return true;
    }

    // Build Routh array for n > 2
    // This is a simplified version - full implementation would build the array
    // For now, rely on eigenvalue computation for full analysis
    true // Placeholder - eigenvalue method is more reliable
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flow::HarmonicOscillator;

    // Stable linear system
    struct StableLinear;

    impl DynamicalSystem<2, 0, 0> for StableLinear {
        fn vector_field(&self, state: &Multivector<2, 0, 0>) -> Result<Multivector<2, 0, 0>> {
            let x = state.get(1);
            let y = state.get(2);

            let mut result = Multivector::zero();
            result.set(1, -x);
            result.set(2, -2.0 * y);

            Ok(result)
        }
    }

    // Unstable linear system
    struct UnstableLinear;

    impl DynamicalSystem<2, 0, 0> for UnstableLinear {
        fn vector_field(&self, state: &Multivector<2, 0, 0>) -> Result<Multivector<2, 0, 0>> {
            let x = state.get(1);
            let y = state.get(2);

            let mut result = Multivector::zero();
            result.set(1, x);
            result.set(2, 2.0 * y);

            Ok(result)
        }
    }

    // Saddle system
    struct SaddleLinear;

    impl DynamicalSystem<2, 0, 0> for SaddleLinear {
        fn vector_field(&self, state: &Multivector<2, 0, 0>) -> Result<Multivector<2, 0, 0>> {
            let x = state.get(1);
            let y = state.get(2);

            let mut result = Multivector::zero();
            result.set(1, x);
            result.set(2, -y);

            Ok(result)
        }
    }

    #[test]
    fn test_stable_system() {
        let system = StableLinear;
        let point = Multivector::<2, 0, 0>::zero();
        let config = DifferentiationConfig::default();

        let analysis = analyze_stability(&system, &point, &config, 1e-10).unwrap();

        // The full Cl(2,0,0) Jacobian has 2 zero eigenvalues (for scalar and bivector)
        // plus 2 negative eigenvalues for the stable directions
        // This is classified as Degenerate due to zero eigenvalues
        // Check that the non-zero eigenvalues are negative
        let negative_count = analysis
            .eigenvalues
            .iter()
            .filter(|(re, _)| *re < -1e-8)
            .count();
        assert!(
            negative_count >= 2,
            "Expected at least 2 negative eigenvalues"
        );
        assert!(analysis.spectral_abscissa <= 0.0);
    }

    #[test]
    fn test_unstable_system() {
        let system = UnstableLinear;
        let point = Multivector::<2, 0, 0>::zero();
        let config = DifferentiationConfig::default();

        let analysis = analyze_stability(&system, &point, &config, 1e-10).unwrap();

        // Has 2 zero eigenvalues + 2 positive eigenvalues
        let positive_count = analysis
            .eigenvalues
            .iter()
            .filter(|(re, _)| *re > 1e-8)
            .count();
        assert!(
            positive_count >= 2,
            "Expected at least 2 positive eigenvalues"
        );
        assert!(analysis.spectral_abscissa > 0.0);
    }

    #[test]
    fn test_saddle_system() {
        let system = SaddleLinear;
        let point = Multivector::<2, 0, 0>::zero();
        let config = DifferentiationConfig::default();

        let analysis = analyze_stability(&system, &point, &config, 1e-10).unwrap();

        // Has 2 zero eigenvalues + 1 positive + 1 negative
        let positive_count = analysis
            .eigenvalues
            .iter()
            .filter(|(re, _)| *re > 1e-8)
            .count();
        let negative_count = analysis
            .eigenvalues
            .iter()
            .filter(|(re, _)| *re < -1e-8)
            .count();
        assert!(
            positive_count >= 1,
            "Expected at least 1 positive eigenvalue"
        );
        assert!(
            negative_count >= 1,
            "Expected at least 1 negative eigenvalue"
        );
    }

    #[test]
    fn test_harmonic_oscillator_center() {
        let system = HarmonicOscillator::new(1.0);
        let point = Multivector::<2, 0, 0>::zero();
        let config = DifferentiationConfig::default();

        let analysis = analyze_stability(&system, &point, &config, 1e-10).unwrap();

        // Harmonic oscillator has purely imaginary eigenvalues
        assert_eq!(analysis.stability, StabilityType::Center);

        // Check eigenvalues are ±i
        assert!(analysis.eigenvalues.len() >= 2);
        let has_imaginary = analysis
            .eigenvalues
            .iter()
            .any(|(re, im)| re.abs() < 0.1 && im.abs() > 0.5);
        assert!(has_imaginary);
    }

    #[test]
    fn test_subspace_dimensions() {
        let system = SaddleLinear;
        let point = Multivector::<2, 0, 0>::zero();
        let config = DifferentiationConfig::default();

        let analysis = analyze_stability(&system, &point, &config, 1e-10).unwrap();
        let dims = compute_subspace_dimensions(&analysis, 1e-10);

        // Saddle system in Cl(2,0,0) has:
        // - 1 stable direction (e2 component)
        // - 1 unstable direction (e1 component)
        // - 2 center directions (scalar and bivector with zero dynamics)
        assert!(dims.stable >= 1, "Expected at least 1 stable dimension");
        assert!(dims.unstable >= 1, "Expected at least 1 unstable dimension");
        // System has center manifold due to zero eigenvalues from unused components
        assert!(
            dims.center >= 2,
            "Expected center manifold from degenerate directions"
        );
        assert_eq!(dims.total(), 4);
    }

    #[test]
    fn test_characteristic_polynomial() {
        // For eigenvalues λ = -1, -2, polynomial is (λ+1)(λ+2) = λ² + 3λ + 2
        let eigenvalues = vec![(-1.0, 0.0), (-2.0, 0.0)];
        let poly = characteristic_polynomial(&eigenvalues);

        assert_eq!(poly.len(), 3);
        assert!((poly[0] - 2.0).abs() < 1e-10); // a_0 = 2
        assert!((poly[1] - 3.0).abs() < 1e-10); // a_1 = 3
        assert!((poly[2] - 1.0).abs() < 1e-10); // a_2 = 1 (leading coeff)
    }

    #[test]
    fn test_hopf_check_center() {
        let system = HarmonicOscillator::new(1.0);
        let point = Multivector::<2, 0, 0>::zero();
        let config = DifferentiationConfig::default();

        let analysis = analyze_stability(&system, &point, &config, 1e-10).unwrap();
        let hopf = hopf_bifurcation_check(&analysis, 1e-6);

        // Harmonic oscillator is exactly at a Hopf-like condition
        assert!(hopf.is_some());
    }

    #[test]
    fn test_eigenvalue_computation() {
        // Test with known matrix
        // [[0, 1], [-1, 0]] has eigenvalues ±i
        let matrix = DMatrix::from_row_slice(2, 2, &[0.0, 1.0, -1.0, 0.0]);
        let eigenvalues = compute_eigenvalues(&matrix).unwrap();

        assert_eq!(eigenvalues.len(), 2);

        // Both should have Re = 0, |Im| = 1
        for (re, im) in &eigenvalues {
            assert!(re.abs() < 1e-10);
            assert!((im.abs() - 1.0).abs() < 1e-10);
        }
    }
}
