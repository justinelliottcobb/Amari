//! Lyapunov exponent computation
//!
//! This module provides tools for computing Lyapunov exponents,
//! which characterize the rate of separation of infinitesimally close
//! trajectories in a dynamical system.
//!
//! # Overview
//!
//! Lyapunov exponents are fundamental invariants of dynamical systems:
//!
//! - **Positive exponents**: Indicate chaos (exponential divergence)
//! - **Zero exponents**: Indicate marginal stability (periodic/quasiperiodic)
//! - **Negative exponents**: Indicate contraction (stability)
//!
//! # The Lyapunov Spectrum
//!
//! For an n-dimensional system, there are n Lyapunov exponents
//! λ₁ ≥ λ₂ ≥ ... ≥ λₙ, forming the spectrum.
//!
//! Key properties:
//! - **Largest exponent (λ₁)**: Determines chaotic/regular behavior
//! - **Sum of exponents**: Related to phase space volume change
//! - **Kaplan-Yorke dimension**: Fractal dimension from spectrum
//!
//! # Computation Methods
//!
//! - **QR method**: Standard algorithm for full spectrum (Benettin et al.)
//! - **Single vector**: Faster computation of largest exponent only
//! - **FTLE**: Finite-time Lyapunov exponents for flow visualization
//!
//! # Example
//!
//! ```ignore
//! use amari_dynamics::lyapunov::{
//!     compute_lyapunov_spectrum, compute_largest_lyapunov,
//!     is_chaotic, LyapunovConfig, LyapunovSpectrum,
//! };
//!
//! // Compute full spectrum
//! let config = LyapunovConfig::default();
//! let spectrum = compute_lyapunov_spectrum(&lorenz, &initial, &config)?;
//!
//! println!("Largest exponent: {:.4}", spectrum.largest());
//! println!("Is chaotic: {}", spectrum.is_chaotic());
//!
//! if let Some(d_ky) = spectrum.kaplan_yorke_dimension() {
//!     println!("Kaplan-Yorke dimension: {:.4}", d_ky);
//! }
//!
//! // Quick chaos check
//! if is_chaotic(&system, &initial, &LyapunovConfig::fast())? {
//!     println!("System is chaotic!");
//! }
//! ```
//!
//! # Geometric Algebra Context
//!
//! In Clifford algebra spaces, Lyapunov exponents characterize:
//!
//! - Grade-specific divergence rates
//! - Rotor dynamics stability
//! - Multivector field chaos
//!
//! The grade structure of Cl(P,Q,R) is preserved in the computation,
//! allowing analysis of grade-specific Lyapunov structure.

mod qr_method;
mod spectrum;

// Re-export main types and functions
pub use qr_method::{
    compute_ftle_field, compute_largest_lyapunov, compute_lyapunov_spectrum, is_chaotic,
};

#[cfg(feature = "parallel")]
pub use qr_method::compute_lyapunov_ensemble;

pub use spectrum::{LyapunovClassification, LyapunovConfig, LyapunovSpectrum};

/// Estimate the Kaplan-Yorke dimension from Lyapunov exponents
///
/// The Kaplan-Yorke (information) dimension is:
/// D_KY = j + (λ₁ + ... + λⱼ) / |λⱼ₊₁|
///
/// where j is the largest integer such that λ₁ + ... + λⱼ ≥ 0.
///
/// # Example
///
/// ```ignore
/// use amari_dynamics::lyapunov::kaplan_yorke_dimension;
///
/// // Lorenz-like spectrum
/// let exponents = [0.9, 0.0, -14.5];
/// let d_ky = kaplan_yorke_dimension(&exponents);
/// assert!(d_ky.unwrap() > 2.0);
/// ```
pub fn kaplan_yorke_dimension(exponents: &[f64]) -> Option<f64> {
    if exponents.is_empty() {
        return None;
    }

    // Sort descending
    let mut sorted = exponents.to_vec();
    sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

    // Find j
    let mut cumsum = 0.0;
    let mut j = 0;

    for (i, &lambda) in sorted.iter().enumerate() {
        cumsum += lambda;
        if cumsum >= 0.0 {
            j = i + 1;
        }
    }

    if j == 0 {
        return Some(0.0);
    }

    if j >= sorted.len() {
        return None;
    }

    let sum_j: f64 = sorted.iter().take(j).sum();
    let lambda_j_plus_1 = sorted[j];

    if lambda_j_plus_1.abs() < 1e-15 {
        return None;
    }

    Some(j as f64 + sum_j / lambda_j_plus_1.abs())
}

/// Estimate Kolmogorov-Sinai entropy from positive Lyapunov exponents
///
/// By Pesin's identity for systems with an SRB measure:
/// h_KS = Σ λᵢ for all λᵢ > 0
///
/// # Example
///
/// ```ignore
/// use amari_dynamics::lyapunov::ks_entropy;
///
/// let exponents = [0.9, 0.3, 0.0, -14.5];
/// let h = ks_entropy(&exponents);
/// assert!((h - 1.2).abs() < 1e-10);
/// ```
pub fn ks_entropy(exponents: &[f64]) -> f64 {
    exponents.iter().filter(|&&x| x > 0.0).sum()
}

/// Check if the system appears Hamiltonian from its Lyapunov spectrum
///
/// Hamiltonian systems have:
/// - Zero sum of Lyapunov exponents
/// - Exponents come in ± pairs (symplectic structure)
///
/// # Example
///
/// ```ignore
/// use amari_dynamics::lyapunov::is_hamiltonian;
///
/// let hamiltonian = [0.5, -0.5];
/// assert!(is_hamiltonian(&hamiltonian, 1e-6));
///
/// let dissipative = [0.9, 0.0, -14.5];
/// assert!(!is_hamiltonian(&dissipative, 1e-6));
/// ```
pub fn is_hamiltonian(exponents: &[f64], tolerance: f64) -> bool {
    exponents.iter().sum::<f64>().abs() < tolerance
}

/// Count positive, zero, and negative exponents
///
/// Returns (positive_count, zero_count, negative_count)
pub fn count_exponents(exponents: &[f64], zero_tolerance: f64) -> (usize, usize, usize) {
    let pos = exponents.iter().filter(|&&x| x > zero_tolerance).count();
    let zero = exponents
        .iter()
        .filter(|&&x| x.abs() <= zero_tolerance)
        .count();
    let neg = exponents.iter().filter(|&&x| x < -zero_tolerance).count();

    (pos, zero, neg)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kaplan_yorke() {
        // Lorenz-like: (0.9, 0, -14.5)
        let exponents = [0.9, 0.0, -14.5];
        let d_ky = kaplan_yorke_dimension(&exponents).unwrap();

        // D_KY = 2 + (0.9 + 0) / 14.5 ≈ 2.062
        assert!((d_ky - 2.062).abs() < 0.01);
    }

    #[test]
    fn test_ks_entropy() {
        let exponents = [0.9, 0.3, 0.0, -14.5];
        assert!((ks_entropy(&exponents) - 1.2).abs() < 1e-10);

        // No positive exponents
        let regular = [-1.0, -2.0, -3.0];
        assert_eq!(ks_entropy(&regular), 0.0);
    }

    #[test]
    fn test_hamiltonian_check() {
        let hamiltonian = [0.5, -0.5];
        assert!(is_hamiltonian(&hamiltonian, 1e-10));

        let dissipative = [0.9, 0.0, -14.5];
        assert!(!is_hamiltonian(&dissipative, 1e-6));
    }

    #[test]
    fn test_count_exponents() {
        let spectrum = [0.9, 0.0, -2.0, -14.5];
        let (pos, zero, neg) = count_exponents(&spectrum, 1e-6);

        assert_eq!(pos, 1);
        assert_eq!(zero, 1);
        assert_eq!(neg, 2);
    }
}
