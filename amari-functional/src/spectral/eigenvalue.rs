//! Eigenvalue computation algorithms.
//!
//! This module provides algorithms for computing eigenvalues and
//! eigenvectors of linear operators.

use crate::error::{FunctionalError, Result};
use crate::operator::{LinearOperator, MatrixOperator};
use amari_core::Multivector;

/// An eigenvalue with optional algebraic multiplicity.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Eigenvalue {
    /// The eigenvalue λ.
    pub value: f64,
    /// Algebraic multiplicity (if known).
    pub multiplicity: Option<usize>,
}

impl Eigenvalue {
    /// Create a new eigenvalue.
    pub fn new(value: f64) -> Self {
        Self {
            value,
            multiplicity: None,
        }
    }

    /// Create an eigenvalue with known multiplicity.
    pub fn with_multiplicity(value: f64, multiplicity: usize) -> Self {
        Self {
            value,
            multiplicity: Some(multiplicity),
        }
    }
}

/// An eigenvalue-eigenvector pair.
#[derive(Debug, Clone)]
pub struct Eigenpair<V> {
    /// The eigenvalue λ.
    pub eigenvalue: Eigenvalue,
    /// The eigenvector v satisfying Tv = λv.
    pub eigenvector: V,
}

impl<V> Eigenpair<V> {
    /// Create a new eigenpair.
    pub fn new(eigenvalue: f64, eigenvector: V) -> Self {
        Self {
            eigenvalue: Eigenvalue::new(eigenvalue),
            eigenvector,
        }
    }
}

/// Power method for computing the dominant eigenvalue.
///
/// Iteratively computes x_{k+1} = Ax_k / ||Ax_k|| to find the
/// eigenvector corresponding to the largest eigenvalue (in absolute value).
///
/// # Arguments
///
/// * `matrix` - The matrix operator
/// * `initial` - Initial guess for the eigenvector
/// * `max_iterations` - Maximum number of iterations
/// * `tolerance` - Convergence tolerance
///
/// # Returns
///
/// The dominant eigenpair (λ, v) where λ is the largest eigenvalue.
pub fn power_method<const P: usize, const Q: usize, const R: usize>(
    matrix: &MatrixOperator<P, Q, R>,
    initial: Option<&Multivector<P, Q, R>>,
    max_iterations: usize,
    tolerance: f64,
) -> Result<Eigenpair<Multivector<P, Q, R>>> {
    let n = MatrixOperator::<P, Q, R>::DIM;

    // Initialize with provided vector or default
    let mut v = if let Some(init) = initial {
        init.clone()
    } else {
        // Default: vector with all 1s normalized
        let coeffs: Vec<f64> = (0..n).map(|_| 1.0 / (n as f64).sqrt()).collect();
        Multivector::<P, Q, R>::from_slice(&coeffs)
    };

    let mut eigenvalue = 0.0;

    for iter in 0..max_iterations {
        // Compute Av
        let av = matrix.apply(&v)?;

        // Compute Rayleigh quotient: λ = v^T Av / v^T v
        let v_coeffs = v.to_vec();
        let av_coeffs = av.to_vec();
        let numerator: f64 = v_coeffs
            .iter()
            .zip(av_coeffs.iter())
            .map(|(a, b)| a * b)
            .sum();
        let denominator: f64 = v_coeffs.iter().map(|x| x * x).sum();

        if denominator.abs() < 1e-15 {
            return Err(FunctionalError::numerical_instability(
                "power method",
                "Vector norm became zero",
            ));
        }

        let new_eigenvalue = numerator / denominator;

        // Check convergence
        if (new_eigenvalue - eigenvalue).abs() < tolerance && iter > 0 {
            return Ok(Eigenpair::new(new_eigenvalue, v));
        }
        eigenvalue = new_eigenvalue;

        // Normalize Av for next iteration
        let norm: f64 = av_coeffs.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm < 1e-15 {
            return Err(FunctionalError::numerical_instability(
                "power method",
                "Zero vector encountered",
            ));
        }
        v = &av * (1.0 / norm);
    }

    Ok(Eigenpair::new(eigenvalue, v))
}

/// Inverse iteration for computing eigenvalue near a target.
///
/// Uses (A - σI)^{-1} power iteration to find the eigenvalue closest to σ.
///
/// # Arguments
///
/// * `matrix` - The matrix operator
/// * `shift` - The target value σ (shift)
/// * `initial` - Initial guess for the eigenvector
/// * `max_iterations` - Maximum number of iterations
/// * `tolerance` - Convergence tolerance
///
/// # Returns
///
/// The eigenpair (λ, v) where λ is closest to σ.
pub fn inverse_iteration<const P: usize, const Q: usize, const R: usize>(
    matrix: &MatrixOperator<P, Q, R>,
    shift: f64,
    initial: Option<&Multivector<P, Q, R>>,
    max_iterations: usize,
    tolerance: f64,
) -> Result<Eigenpair<Multivector<P, Q, R>>> {
    let n = MatrixOperator::<P, Q, R>::DIM;

    // Compute (A - σI)
    let identity = MatrixOperator::<P, Q, R>::identity();
    let shifted = matrix.add(&identity.scale(-shift))?;

    // For small matrices, use direct solve via LU decomposition
    // For now, approximate with a few Gauss-Seidel iterations

    let mut v = if let Some(init) = initial {
        init.clone()
    } else {
        let coeffs: Vec<f64> = (0..n).map(|_| 1.0 / (n as f64).sqrt()).collect();
        Multivector::<P, Q, R>::from_slice(&coeffs)
    };

    for _ in 0..max_iterations {
        // Solve (A - σI)w = v using iterative refinement
        let w = solve_linear_system(&shifted, &v, 50, tolerance)?;

        // Normalize
        let w_coeffs = w.to_vec();
        let norm: f64 = w_coeffs.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm < 1e-15 {
            break;
        }
        v = &w * (1.0 / norm);
    }

    // Compute the eigenvalue via Rayleigh quotient on original matrix
    let av = matrix.apply(&v)?;
    let v_coeffs = v.to_vec();
    let av_coeffs = av.to_vec();
    let numerator: f64 = v_coeffs
        .iter()
        .zip(av_coeffs.iter())
        .map(|(a, b)| a * b)
        .sum();
    let denominator: f64 = v_coeffs.iter().map(|x| x * x).sum();

    let eigenvalue = if denominator.abs() < 1e-15 {
        shift
    } else {
        numerator / denominator
    };

    Ok(Eigenpair::new(eigenvalue, v))
}

/// Compute all eigenvalues of a symmetric matrix.
///
/// Uses the Jacobi eigenvalue algorithm for symmetric matrices.
///
/// # Arguments
///
/// * `matrix` - A symmetric matrix operator
/// * `max_iterations` - Maximum number of Jacobi rotations
/// * `tolerance` - Convergence tolerance
///
/// # Returns
///
/// A vector of eigenvalues sorted by magnitude.
pub fn compute_eigenvalues<const P: usize, const Q: usize, const R: usize>(
    matrix: &MatrixOperator<P, Q, R>,
    max_iterations: usize,
    tolerance: f64,
) -> Result<Vec<Eigenvalue>> {
    if !matrix.is_symmetric(tolerance) {
        return Err(FunctionalError::invalid_parameters(
            "compute_eigenvalues requires a symmetric matrix",
        ));
    }

    let n = matrix.rows();
    let mut a = matrix.clone();

    // Jacobi eigenvalue algorithm
    for _ in 0..max_iterations {
        // Find largest off-diagonal element
        let mut max_val = 0.0;
        let mut p = 0;
        let mut q = 1;

        for i in 0..n {
            for j in (i + 1)..n {
                let val = a.get(i, j).abs();
                if val > max_val {
                    max_val = val;
                    p = i;
                    q = j;
                }
            }
        }

        // Check convergence
        if max_val < tolerance {
            break;
        }

        // Compute Jacobi rotation angle
        let app = a.get(p, p);
        let aqq = a.get(q, q);
        let apq = a.get(p, q);

        let theta = if (aqq - app).abs() < 1e-15 {
            std::f64::consts::FRAC_PI_4
        } else {
            0.5 * ((2.0 * apq) / (aqq - app)).atan()
        };

        let c = theta.cos();
        let s = theta.sin();

        // Apply Jacobi rotation
        a = apply_jacobi_rotation(&a, p, q, c, s)?;
    }

    // Extract eigenvalues from diagonal
    let mut eigenvalues: Vec<Eigenvalue> = (0..n).map(|i| Eigenvalue::new(a.get(i, i))).collect();

    // Sort by absolute value (descending)
    eigenvalues.sort_by(|a, b| {
        b.value
            .abs()
            .partial_cmp(&a.value.abs())
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    Ok(eigenvalues)
}

/// Apply a Jacobi rotation to a matrix.
fn apply_jacobi_rotation<const P: usize, const Q: usize, const R: usize>(
    a: &MatrixOperator<P, Q, R>,
    p: usize,
    q: usize,
    c: f64,
    s: f64,
) -> Result<MatrixOperator<P, Q, R>> {
    let n = a.rows();
    let mut result = a.clone();

    for i in 0..n {
        if i != p && i != q {
            let aip = a.get(i, p);
            let aiq = a.get(i, q);
            result.set(i, p, c * aip - s * aiq);
            result.set(p, i, c * aip - s * aiq);
            result.set(i, q, s * aip + c * aiq);
            result.set(q, i, s * aip + c * aiq);
        }
    }

    let app = a.get(p, p);
    let aqq = a.get(q, q);
    let apq = a.get(p, q);

    result.set(p, p, c * c * app - 2.0 * c * s * apq + s * s * aqq);
    result.set(q, q, s * s * app + 2.0 * c * s * apq + c * c * aqq);
    result.set(p, q, 0.0);
    result.set(q, p, 0.0);

    Ok(result)
}

/// Solve a linear system Ax = b using Gauss-Seidel iteration.
fn solve_linear_system<const P: usize, const Q: usize, const R: usize>(
    a: &MatrixOperator<P, Q, R>,
    b: &Multivector<P, Q, R>,
    max_iterations: usize,
    _tolerance: f64,
) -> Result<Multivector<P, Q, R>> {
    let n = a.rows();
    let b_coeffs = b.to_vec();
    let mut x = vec![0.0; n];

    for _ in 0..max_iterations {
        for i in 0..n {
            let mut sum = b_coeffs[i];
            for j in 0..n {
                if i != j {
                    sum -= a.get(i, j) * x[j];
                }
            }
            let aii = a.get(i, i);
            if aii.abs() > 1e-15 {
                x[i] = sum / aii;
            }
        }
    }

    Ok(Multivector::<P, Q, R>::from_slice(&x))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_eigenvalue_creation() {
        let e = Eigenvalue::new(3.0);
        assert!((e.value - 3.0).abs() < 1e-10);
        assert!(e.multiplicity.is_none());

        let e2 = Eigenvalue::with_multiplicity(2.0, 2);
        assert_eq!(e2.multiplicity, Some(2));
    }

    #[test]
    fn test_power_method_identity() {
        let id: MatrixOperator<2, 0, 0> = MatrixOperator::identity();
        let result = power_method(&id, None, 100, 1e-10).unwrap();

        // Eigenvalue of identity is 1
        assert!((result.eigenvalue.value - 1.0).abs() < 0.1);
    }

    #[test]
    fn test_power_method_diagonal() {
        // Create a diagonal matrix with eigenvalues 4, 3, 2, 1
        let diag: MatrixOperator<2, 0, 0> =
            MatrixOperator::diagonal(&[4.0, 3.0, 2.0, 1.0]).unwrap();
        let result = power_method(&diag, None, 100, 1e-10).unwrap();

        // Dominant eigenvalue should be 4
        assert!((result.eigenvalue.value - 4.0).abs() < 0.1);
    }

    #[test]
    fn test_jacobi_eigenvalues_identity() {
        let id: MatrixOperator<2, 0, 0> = MatrixOperator::identity();
        let eigenvalues = compute_eigenvalues(&id, 100, 1e-10).unwrap();

        // All eigenvalues of identity are 1
        for e in &eigenvalues {
            assert!((e.value - 1.0).abs() < 1e-8);
        }
    }

    #[test]
    fn test_jacobi_eigenvalues_diagonal() {
        let diag: MatrixOperator<2, 0, 0> =
            MatrixOperator::diagonal(&[4.0, 3.0, 2.0, 1.0]).unwrap();
        let eigenvalues = compute_eigenvalues(&diag, 100, 1e-10).unwrap();

        // Eigenvalues should be 4, 3, 2, 1 (sorted by magnitude descending)
        assert!((eigenvalues[0].value - 4.0).abs() < 1e-8);
        assert!((eigenvalues[1].value - 3.0).abs() < 1e-8);
        assert!((eigenvalues[2].value - 2.0).abs() < 1e-8);
        assert!((eigenvalues[3].value - 1.0).abs() < 1e-8);
    }

    #[test]
    fn test_eigenpair_creation() {
        let v = Multivector::<2, 0, 0>::from_slice(&[1.0, 0.0, 0.0, 0.0]);
        let pair = Eigenpair::new(2.0, v);
        assert!((pair.eigenvalue.value - 2.0).abs() < 1e-10);
    }
}
