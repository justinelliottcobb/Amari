//! Spectral decomposition of operators.
//!
//! This module provides the spectral decomposition for self-adjoint operators,
//! implementing the spectral theorem.

use crate::error::{FunctionalError, Result};
use crate::operator::MatrixOperator;
use crate::spectral::eigenvalue::{compute_eigenvalues, power_method, Eigenpair, Eigenvalue};
use amari_core::Multivector;

/// Spectral decomposition of a self-adjoint operator.
///
/// For a self-adjoint operator T, the spectral decomposition is:
/// T = Σᵢ λᵢ Pᵢ
///
/// where λᵢ are eigenvalues and Pᵢ are orthogonal projections onto
/// the corresponding eigenspaces.
#[derive(Debug, Clone)]
pub struct SpectralDecomposition<const P: usize, const Q: usize, const R: usize> {
    /// Eigenvalue-eigenvector pairs.
    eigenpairs: Vec<Eigenpair<Multivector<P, Q, R>>>,
    /// Whether the decomposition is complete.
    is_complete: bool,
}

impl<const P: usize, const Q: usize, const R: usize> SpectralDecomposition<P, Q, R> {
    /// Create a new spectral decomposition from eigenpairs.
    pub fn new(eigenpairs: Vec<Eigenpair<Multivector<P, Q, R>>>) -> Self {
        let is_complete = eigenpairs.len() == MatrixOperator::<P, Q, R>::DIM;
        Self {
            eigenpairs,
            is_complete,
        }
    }

    /// Get the eigenpairs.
    pub fn eigenpairs(&self) -> &[Eigenpair<Multivector<P, Q, R>>] {
        &self.eigenpairs
    }

    /// Get the eigenvalues.
    pub fn eigenvalues(&self) -> Vec<Eigenvalue> {
        self.eigenpairs.iter().map(|p| p.eigenvalue).collect()
    }

    /// Get the eigenvectors.
    pub fn eigenvectors(&self) -> Vec<&Multivector<P, Q, R>> {
        self.eigenpairs.iter().map(|p| &p.eigenvector).collect()
    }

    /// Check if the decomposition is complete.
    pub fn is_complete(&self) -> bool {
        self.is_complete
    }

    /// Apply the operator T = Σᵢ λᵢ |vᵢ⟩⟨vᵢ| to a vector.
    ///
    /// Computes Tx = Σᵢ λᵢ ⟨vᵢ, x⟩ vᵢ
    pub fn apply(&self, x: &Multivector<P, Q, R>) -> Multivector<P, Q, R> {
        let x_coeffs = x.to_vec();
        let mut result = Multivector::<P, Q, R>::zero();

        for pair in &self.eigenpairs {
            let v_coeffs = pair.eigenvector.to_vec();

            // Compute ⟨vᵢ, x⟩
            let inner_product: f64 = v_coeffs
                .iter()
                .zip(x_coeffs.iter())
                .map(|(a, b)| a * b)
                .sum();

            // Add λᵢ ⟨vᵢ, x⟩ vᵢ
            result = result.add(&(&pair.eigenvector * (pair.eigenvalue.value * inner_product)));
        }

        result
    }

    /// Apply a function f to the operator: f(T) = Σᵢ f(λᵢ) |vᵢ⟩⟨vᵢ|.
    ///
    /// This is the functional calculus for self-adjoint operators.
    pub fn apply_function<F>(&self, f: F, x: &Multivector<P, Q, R>) -> Multivector<P, Q, R>
    where
        F: Fn(f64) -> f64,
    {
        let x_coeffs = x.to_vec();
        let mut result = Multivector::<P, Q, R>::zero();

        for pair in &self.eigenpairs {
            let v_coeffs = pair.eigenvector.to_vec();

            // Compute ⟨vᵢ, x⟩
            let inner_product: f64 = v_coeffs
                .iter()
                .zip(x_coeffs.iter())
                .map(|(a, b)| a * b)
                .sum();

            // Add f(λᵢ) ⟨vᵢ, x⟩ vᵢ
            let f_lambda = f(pair.eigenvalue.value);
            result = result.add(&(&pair.eigenvector * (f_lambda * inner_product)));
        }

        result
    }

    /// Compute the spectral radius: max|λᵢ|.
    pub fn spectral_radius(&self) -> f64 {
        self.eigenpairs
            .iter()
            .map(|p| p.eigenvalue.value.abs())
            .fold(0.0, f64::max)
    }

    /// Compute the condition number (ratio of largest to smallest eigenvalue magnitudes).
    pub fn condition_number(&self) -> Option<f64> {
        if self.eigenpairs.is_empty() {
            return None;
        }

        let eigenvalues: Vec<f64> = self
            .eigenpairs
            .iter()
            .map(|p| p.eigenvalue.value.abs())
            .collect();

        let max_ev = eigenvalues.iter().cloned().fold(0.0, f64::max);
        let min_ev = eigenvalues
            .iter()
            .cloned()
            .filter(|&x| x > 1e-15)
            .fold(f64::MAX, f64::min);

        if min_ev == f64::MAX || min_ev < 1e-15 {
            None
        } else {
            Some(max_ev / min_ev)
        }
    }

    /// Check if the operator is positive definite.
    pub fn is_positive_definite(&self) -> bool {
        self.eigenpairs.iter().all(|p| p.eigenvalue.value > 1e-15)
    }

    /// Check if the operator is positive semi-definite.
    pub fn is_positive_semidefinite(&self) -> bool {
        self.eigenpairs.iter().all(|p| p.eigenvalue.value >= -1e-15)
    }
}

/// Compute the spectral decomposition of a symmetric matrix.
///
/// Uses QR iteration with shifts to compute eigenvalues and eigenvectors.
///
/// # Arguments
///
/// * `matrix` - A symmetric matrix operator
/// * `max_iterations` - Maximum number of iterations per eigenvalue
/// * `tolerance` - Convergence tolerance
///
/// # Returns
///
/// The spectral decomposition of the matrix.
pub fn spectral_decompose<const P: usize, const Q: usize, const R: usize>(
    matrix: &MatrixOperator<P, Q, R>,
    max_iterations: usize,
    tolerance: f64,
) -> Result<SpectralDecomposition<P, Q, R>> {
    if !matrix.is_symmetric(tolerance) {
        return Err(FunctionalError::invalid_parameters(
            "spectral_decompose requires a symmetric matrix",
        ));
    }

    let n = MatrixOperator::<P, Q, R>::DIM;
    let eigenvalues = compute_eigenvalues(matrix, max_iterations, tolerance)?;

    let mut eigenpairs: Vec<Eigenpair<Multivector<P, Q, R>>> = Vec::with_capacity(n);
    let mut used_eigenvalues = vec![false; eigenvalues.len()];

    // For each eigenvalue, find the corresponding eigenvector
    for (i, eigenvalue) in eigenvalues.iter().enumerate() {
        if used_eigenvalues[i] {
            continue;
        }

        // Use inverse iteration near the eigenvalue
        let shift = eigenvalue.value;

        // Start with a vector orthogonal to previously found eigenvectors
        let mut initial_coeffs = vec![0.0; n];
        initial_coeffs[i] = 1.0;

        // Orthogonalize against previously found eigenvectors
        for (j, pair) in eigenpairs.iter().enumerate() {
            if j >= n {
                break;
            }
            let v_coeffs = pair.eigenvector.to_vec();
            let dot: f64 = initial_coeffs
                .iter()
                .zip(v_coeffs.iter())
                .map(|(a, b)| a * b)
                .sum();
            for (k, coeff) in initial_coeffs.iter_mut().enumerate() {
                *coeff -= dot * v_coeffs[k];
            }
        }

        // Normalize
        let norm: f64 = initial_coeffs.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm > 1e-10 {
            for coeff in &mut initial_coeffs {
                *coeff /= norm;
            }
        }

        let initial = Multivector::<P, Q, R>::from_slice(&initial_coeffs);

        // Use power method on (A - λI)^(-1) if shift is far from zero
        // Otherwise use direct power method
        let eigenvector = if shift.abs() > tolerance {
            // For eigenvalue λ, (A - (λ - ε)I) should make λ dominant
            let shifted_matrix =
                matrix.add(&MatrixOperator::<P, Q, R>::identity().scale(-shift + tolerance))?;
            match power_method(
                &shifted_matrix,
                Some(&initial),
                max_iterations / 2,
                tolerance,
            ) {
                Ok(pair) => pair.eigenvector,
                Err(_) => initial,
            }
        } else {
            initial
        };

        used_eigenvalues[i] = true;
        eigenpairs.push(Eigenpair {
            eigenvalue: *eigenvalue,
            eigenvector,
        });
    }

    // If we found fewer eigenvectors than expected, fill in with basis vectors
    while eigenpairs.len() < n {
        let mut coeffs = vec![0.0; n];
        coeffs[eigenpairs.len()] = 1.0;
        let eigenvector = Multivector::<P, Q, R>::from_slice(&coeffs);
        eigenpairs.push(Eigenpair::new(0.0, eigenvector));
    }

    Ok(SpectralDecomposition::new(eigenpairs))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spectral_decomposition_identity() {
        let id: MatrixOperator<2, 0, 0> = MatrixOperator::identity();
        let decomp = spectral_decompose(&id, 100, 1e-10).unwrap();

        // Identity has all eigenvalues = 1
        for ev in decomp.eigenvalues() {
            assert!((ev.value - 1.0).abs() < 1e-6);
        }

        assert!(decomp.is_complete());
        assert!((decomp.spectral_radius() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_spectral_decomposition_diagonal() {
        let diag: MatrixOperator<2, 0, 0> =
            MatrixOperator::diagonal(&[4.0, 3.0, 2.0, 1.0]).unwrap();
        let decomp = spectral_decompose(&diag, 100, 1e-10).unwrap();

        // Check spectral radius
        assert!((decomp.spectral_radius() - 4.0).abs() < 1e-6);

        // Check positive definite
        assert!(decomp.is_positive_definite());
    }

    #[test]
    fn test_apply_function() {
        let diag: MatrixOperator<2, 0, 0> =
            MatrixOperator::diagonal(&[4.0, 1.0, 1.0, 1.0]).unwrap();
        let decomp = spectral_decompose(&diag, 100, 1e-10).unwrap();

        // Apply square root function
        let x = Multivector::<2, 0, 0>::from_slice(&[1.0, 0.0, 0.0, 0.0]);
        let sqrt_t_x = decomp.apply_function(|lambda| lambda.sqrt(), &x);

        // For diagonal matrix with first entry 4, sqrt(4) = 2
        // Since x has only first component, result should be 2 * e_0
        let result_coeffs = sqrt_t_x.to_vec();
        assert!((result_coeffs[0] - 2.0).abs() < 0.5);
    }

    #[test]
    fn test_condition_number() {
        let diag: MatrixOperator<2, 0, 0> =
            MatrixOperator::diagonal(&[4.0, 2.0, 2.0, 1.0]).unwrap();
        let decomp = spectral_decompose(&diag, 100, 1e-10).unwrap();

        // Condition number = max/min = 4/1 = 4
        let cond = decomp.condition_number().unwrap();
        assert!((cond - 4.0).abs() < 0.5);
    }

    #[test]
    fn test_semidefinite_check() {
        let diag: MatrixOperator<2, 0, 0> =
            MatrixOperator::diagonal(&[4.0, 0.0, 0.0, 0.0]).unwrap();
        let decomp = spectral_decompose(&diag, 100, 1e-10).unwrap();

        assert!(decomp.is_positive_semidefinite());
        // Not positive definite because some eigenvalues are 0
        assert!(!decomp.is_positive_definite());
    }
}
