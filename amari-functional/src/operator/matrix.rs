//! Matrix representation of linear operators.
//!
//! This module provides a matrix-based representation for linear operators
//! on finite-dimensional Hilbert spaces.

use crate::error::{FunctionalError, Result};
use crate::operator::traits::{AdjointableOperator, BoundedOperator, LinearOperator, OperatorNorm};
use crate::phantom::Bounded;
use amari_core::Multivector;

/// A linear operator represented as a matrix.
///
/// The matrix is stored in row-major order.
#[derive(Debug, Clone)]
pub struct MatrixOperator<const P: usize, const Q: usize, const R: usize> {
    /// Matrix entries in row-major order.
    entries: Vec<f64>,
    /// Number of rows.
    rows: usize,
    /// Number of columns.
    cols: usize,
}

impl<const P: usize, const Q: usize, const R: usize> MatrixOperator<P, Q, R> {
    /// The dimension of the Clifford algebra.
    pub const DIM: usize = 1 << (P + Q + R);

    /// Create a new matrix operator from entries.
    ///
    /// Entries should be in row-major order.
    pub fn new(entries: Vec<f64>, rows: usize, cols: usize) -> Result<Self> {
        if entries.len() != rows * cols {
            return Err(FunctionalError::dimension_mismatch(
                rows * cols,
                entries.len(),
            ));
        }
        if cols != Self::DIM {
            return Err(FunctionalError::dimension_mismatch(Self::DIM, cols));
        }
        if rows != Self::DIM {
            return Err(FunctionalError::dimension_mismatch(Self::DIM, rows));
        }
        Ok(Self {
            entries,
            rows,
            cols,
        })
    }

    /// Create an identity matrix.
    pub fn identity() -> Self {
        let n = Self::DIM;
        let mut entries = vec![0.0; n * n];
        for i in 0..n {
            entries[i * n + i] = 1.0;
        }
        Self {
            entries,
            rows: n,
            cols: n,
        }
    }

    /// Create a zero matrix.
    pub fn zeros() -> Self {
        let n = Self::DIM;
        Self {
            entries: vec![0.0; n * n],
            rows: n,
            cols: n,
        }
    }

    /// Create a diagonal matrix from diagonal entries.
    pub fn diagonal(diag: &[f64]) -> Result<Self> {
        let n = Self::DIM;
        if diag.len() != n {
            return Err(FunctionalError::dimension_mismatch(n, diag.len()));
        }
        let mut entries = vec![0.0; n * n];
        for i in 0..n {
            entries[i * n + i] = diag[i];
        }
        Ok(Self {
            entries,
            rows: n,
            cols: n,
        })
    }

    /// Get a matrix entry.
    pub fn get(&self, row: usize, col: usize) -> f64 {
        if row < self.rows && col < self.cols {
            self.entries[row * self.cols + col]
        } else {
            0.0
        }
    }

    /// Set a matrix entry.
    pub fn set(&mut self, row: usize, col: usize, value: f64) {
        if row < self.rows && col < self.cols {
            self.entries[row * self.cols + col] = value;
        }
    }

    /// Get the number of rows.
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Get the number of columns.
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// Compute the transpose of the matrix.
    pub fn transpose(&self) -> Self {
        let mut entries = vec![0.0; self.rows * self.cols];
        for i in 0..self.rows {
            for j in 0..self.cols {
                entries[j * self.rows + i] = self.get(i, j);
            }
        }
        Self {
            entries,
            rows: self.cols,
            cols: self.rows,
        }
    }

    /// Compute the trace of the matrix.
    pub fn trace(&self) -> f64 {
        let n = self.rows.min(self.cols);
        (0..n).map(|i| self.get(i, i)).sum()
    }

    /// Multiply this matrix by another.
    pub fn multiply(&self, other: &Self) -> Result<Self> {
        if self.cols != other.rows {
            return Err(FunctionalError::dimension_mismatch(self.cols, other.rows));
        }

        let mut entries = vec![0.0; self.rows * other.cols];
        for i in 0..self.rows {
            for j in 0..other.cols {
                let mut sum = 0.0;
                for k in 0..self.cols {
                    sum += self.get(i, k) * other.get(k, j);
                }
                entries[i * other.cols + j] = sum;
            }
        }

        Ok(Self {
            entries,
            rows: self.rows,
            cols: other.cols,
        })
    }

    /// Add another matrix.
    pub fn add(&self, other: &Self) -> Result<Self> {
        if self.rows != other.rows || self.cols != other.cols {
            return Err(FunctionalError::dimension_mismatch(
                self.rows * self.cols,
                other.rows * other.cols,
            ));
        }

        let entries: Vec<f64> = self
            .entries
            .iter()
            .zip(other.entries.iter())
            .map(|(a, b)| a + b)
            .collect();

        Ok(Self {
            entries,
            rows: self.rows,
            cols: self.cols,
        })
    }

    /// Scale the matrix by a scalar.
    pub fn scale(&self, scalar: f64) -> Self {
        let entries: Vec<f64> = self.entries.iter().map(|x| x * scalar).collect();
        Self {
            entries,
            rows: self.rows,
            cols: self.cols,
        }
    }

    /// Check if the matrix is symmetric.
    pub fn is_symmetric(&self, tolerance: f64) -> bool {
        if self.rows != self.cols {
            return false;
        }
        for i in 0..self.rows {
            for j in (i + 1)..self.cols {
                if (self.get(i, j) - self.get(j, i)).abs() > tolerance {
                    return false;
                }
            }
        }
        true
    }
}

impl<const P: usize, const Q: usize, const R: usize> LinearOperator<Multivector<P, Q, R>>
    for MatrixOperator<P, Q, R>
{
    fn apply(&self, x: &Multivector<P, Q, R>) -> Result<Multivector<P, Q, R>> {
        let x_coeffs = x.to_vec();
        if x_coeffs.len() != self.cols {
            return Err(FunctionalError::dimension_mismatch(
                self.cols,
                x_coeffs.len(),
            ));
        }

        let mut result = vec![0.0; self.rows];
        for i in 0..self.rows {
            for j in 0..self.cols {
                result[i] += self.get(i, j) * x_coeffs[j];
            }
        }

        Ok(Multivector::<P, Q, R>::from_slice(&result))
    }

    fn domain_dimension(&self) -> Option<usize> {
        Some(self.cols)
    }

    fn codomain_dimension(&self) -> Option<usize> {
        Some(self.rows)
    }
}

impl<const P: usize, const Q: usize, const R: usize> OperatorNorm for MatrixOperator<P, Q, R> {
    fn norm(&self) -> f64 {
        // Use power iteration to estimate spectral norm
        // This is an approximation for the 2-norm ||A||₂
        let ata = self.transpose().multiply(self).unwrap();
        power_iteration_spectral_radius(&ata, 100).sqrt()
    }

    fn frobenius_norm(&self) -> Option<f64> {
        let sum_sq: f64 = self.entries.iter().map(|x| x * x).sum();
        Some(sum_sq.sqrt())
    }
}

impl<const P: usize, const Q: usize, const R: usize>
    BoundedOperator<Multivector<P, Q, R>, Multivector<P, Q, R>, Bounded>
    for MatrixOperator<P, Q, R>
{
    fn operator_norm(&self) -> f64 {
        self.norm()
    }
}

impl<const P: usize, const Q: usize, const R: usize> AdjointableOperator<Multivector<P, Q, R>>
    for MatrixOperator<P, Q, R>
{
    type Adjoint = MatrixOperator<P, Q, R>;

    fn adjoint(&self) -> Self::Adjoint {
        // For real matrices, the adjoint is the transpose
        self.transpose()
    }

    fn is_self_adjoint(&self) -> bool {
        self.is_symmetric(1e-10)
    }

    fn is_normal(&self) -> bool {
        // Check if AA* = A*A
        let a_adj = self.transpose();
        let aa_adj = self.multiply(&a_adj).unwrap();
        let a_adj_a = a_adj.multiply(self).unwrap();

        for i in 0..self.rows {
            for j in 0..self.cols {
                if (aa_adj.get(i, j) - a_adj_a.get(i, j)).abs() > 1e-10 {
                    return false;
                }
            }
        }
        true
    }
}

/// Power iteration to estimate the spectral radius.
fn power_iteration_spectral_radius<const P: usize, const Q: usize, const R: usize>(
    matrix: &MatrixOperator<P, Q, R>,
    iterations: usize,
) -> f64 {
    let n = matrix.rows;
    if n == 0 {
        return 0.0;
    }

    // Start with a random-ish vector
    let mut v: Vec<f64> = (0..n).map(|i| 1.0 / ((i + 1) as f64)).collect();

    for _ in 0..iterations {
        // Compute Av
        let mut w = vec![0.0; n];
        for i in 0..n {
            for j in 0..n {
                w[i] += matrix.get(i, j) * v[j];
            }
        }

        // Normalize
        let norm: f64 = w.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm < 1e-15 {
            return 0.0;
        }
        for x in &mut w {
            *x /= norm;
        }

        v = w;
    }

    // Compute Rayleigh quotient
    let mut av = vec![0.0; n];
    for i in 0..n {
        for j in 0..n {
            av[i] += matrix.get(i, j) * v[j];
        }
    }

    let numerator: f64 = v.iter().zip(av.iter()).map(|(a, b)| a * b).sum();
    let denominator: f64 = v.iter().map(|x| x * x).sum();

    if denominator.abs() < 1e-15 {
        0.0
    } else {
        (numerator / denominator).abs()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identity_matrix() {
        let id: MatrixOperator<2, 0, 0> = MatrixOperator::identity();
        let x = Multivector::<2, 0, 0>::from_slice(&[1.0, 2.0, 3.0, 4.0]);
        let y = id.apply(&x).unwrap();
        assert_eq!(x.to_vec(), y.to_vec());
    }

    #[test]
    fn test_diagonal_matrix() {
        let diag: MatrixOperator<2, 0, 0> =
            MatrixOperator::diagonal(&[2.0, 3.0, 4.0, 5.0]).unwrap();
        let x = Multivector::<2, 0, 0>::from_slice(&[1.0, 1.0, 1.0, 1.0]);
        let y = diag.apply(&x).unwrap();
        assert_eq!(y.to_vec(), vec![2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn test_transpose() {
        let mut matrix: MatrixOperator<1, 0, 0> = MatrixOperator::zeros();
        matrix.set(0, 1, 1.0);
        let transposed = matrix.transpose();
        assert!((transposed.get(1, 0) - 1.0).abs() < 1e-10);
        assert!(transposed.get(0, 1).abs() < 1e-10);
    }

    #[test]
    fn test_trace() {
        let id: MatrixOperator<2, 0, 0> = MatrixOperator::identity();
        assert!((id.trace() - 4.0).abs() < 1e-10); // 4-dimensional space
    }

    #[test]
    fn test_matrix_multiplication() {
        let a: MatrixOperator<1, 0, 0> = MatrixOperator::identity();
        let b: MatrixOperator<1, 0, 0> = MatrixOperator::identity();
        let c = a.multiply(&b).unwrap();

        // I * I = I
        for i in 0..2 {
            for j in 0..2 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!((c.get(i, j) - expected).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_symmetric_check() {
        let id: MatrixOperator<2, 0, 0> = MatrixOperator::identity();
        assert!(id.is_symmetric(1e-10));
        assert!(id.is_self_adjoint());
    }

    #[test]
    fn test_frobenius_norm() {
        let id: MatrixOperator<2, 0, 0> = MatrixOperator::identity();
        // ||I||_F = sqrt(n) for n×n identity
        let expected = 2.0; // sqrt(4) = 2
        assert!((id.frobenius_norm().unwrap() - expected).abs() < 1e-10);
    }

    #[test]
    fn test_operator_norm_identity() {
        let id: MatrixOperator<2, 0, 0> = MatrixOperator::identity();
        // ||I||₂ = 1
        assert!((id.operator_norm() - 1.0).abs() < 0.1);
    }
}
