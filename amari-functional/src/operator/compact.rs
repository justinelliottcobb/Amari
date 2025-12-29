//! Compact and Fredholm operators.
//!
//! This module provides traits and types for compact operators and
//! Fredholm operators, which are central to many results in functional analysis.
//!
//! # Mathematical Background
//!
//! - **Compact operators** map bounded sets to precompact sets. On infinite-dimensional
//!   spaces, they are the closure of finite-rank operators.
//!
//! - **Fredholm operators** have finite-dimensional kernel and cokernel.
//!   The Fredholm index is ind(T) = dim(ker T) - dim(coker T).

use crate::error::{FunctionalError, Result};
use crate::operator::traits::{BoundedOperator, LinearOperator};
use crate::operator::MatrixOperator;
use crate::phantom::{Bounded, Compact, Fredholm as FredholmMarker};
use amari_core::Multivector;
use std::marker::PhantomData;

/// Trait for compact operators.
///
/// A compact operator maps bounded sets to precompact sets.
/// Equivalently, every bounded sequence has a subsequence whose
/// image converges.
pub trait CompactOperator<V, W = V>: BoundedOperator<V, W, Bounded> {
    /// Check if the operator has finite rank.
    fn is_finite_rank(&self) -> bool;

    /// Get the rank (dimension of range) if finite.
    fn rank(&self) -> Option<usize>;

    /// Get the singular values (if computable).
    ///
    /// Singular values are eigenvalues of √(T*T).
    fn singular_values(&self) -> Result<Vec<f64>>;
}

/// Trait for Fredholm operators.
///
/// A Fredholm operator has:
/// - Finite-dimensional kernel (null space)
/// - Closed range
/// - Finite-dimensional cokernel
pub trait FredholmOperator<V, W = V>: BoundedOperator<V, W, Bounded> {
    /// Compute the dimension of the kernel.
    fn kernel_dimension(&self) -> usize;

    /// Compute the dimension of the cokernel.
    fn cokernel_dimension(&self) -> usize;

    /// Compute the Fredholm index: dim(ker T) - dim(coker T).
    fn index(&self) -> i64 {
        self.kernel_dimension() as i64 - self.cokernel_dimension() as i64
    }

    /// Check if the operator is an isomorphism (index 0, trivial kernel).
    fn is_isomorphism(&self) -> bool {
        self.kernel_dimension() == 0 && self.cokernel_dimension() == 0
    }
}

/// A finite-rank operator represented as a sum of rank-1 operators.
///
/// T = Σᵢ |vᵢ⟩⟨wᵢ|
///
/// where |vᵢ⟩⟨wᵢ| maps x to ⟨wᵢ, x⟩vᵢ.
#[derive(Clone)]
pub struct FiniteRankOperator<const P: usize, const Q: usize, const R: usize> {
    /// The "left" vectors vᵢ (range vectors).
    left_vectors: Vec<Multivector<P, Q, R>>,
    /// The "right" vectors wᵢ (functionals).
    right_vectors: Vec<Multivector<P, Q, R>>,
}

impl<const P: usize, const Q: usize, const R: usize> std::fmt::Debug
    for FiniteRankOperator<P, Q, R>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FiniteRankOperator")
            .field("rank", &self.left_vectors.len())
            .field("signature", &(P, Q, R))
            .finish()
    }
}

impl<const P: usize, const Q: usize, const R: usize> FiniteRankOperator<P, Q, R> {
    /// Create a new finite-rank operator from left and right vectors.
    pub fn new(
        left_vectors: Vec<Multivector<P, Q, R>>,
        right_vectors: Vec<Multivector<P, Q, R>>,
    ) -> Result<Self> {
        if left_vectors.len() != right_vectors.len() {
            return Err(FunctionalError::dimension_mismatch(
                left_vectors.len(),
                right_vectors.len(),
            ));
        }
        Ok(Self {
            left_vectors,
            right_vectors,
        })
    }

    /// Create a rank-1 operator |v⟩⟨w|.
    pub fn rank_one(v: Multivector<P, Q, R>, w: Multivector<P, Q, R>) -> Self {
        Self {
            left_vectors: vec![v],
            right_vectors: vec![w],
        }
    }

    /// Get the rank of the operator.
    pub fn get_rank(&self) -> usize {
        self.left_vectors.len()
    }

    /// Convert to a matrix representation.
    pub fn to_matrix(&self) -> Result<MatrixOperator<P, Q, R>> {
        let n = MatrixOperator::<P, Q, R>::DIM;
        let mut entries = vec![0.0; n * n];

        for (v, w) in self.left_vectors.iter().zip(self.right_vectors.iter()) {
            let v_coeffs = v.to_vec();
            let w_coeffs = w.to_vec();

            for i in 0..n {
                for j in 0..n {
                    entries[i * n + j] += v_coeffs[i] * w_coeffs[j];
                }
            }
        }

        MatrixOperator::new(entries, n, n)
    }
}

impl<const P: usize, const Q: usize, const R: usize> LinearOperator<Multivector<P, Q, R>>
    for FiniteRankOperator<P, Q, R>
{
    fn apply(&self, x: &Multivector<P, Q, R>) -> Result<Multivector<P, Q, R>> {
        let x_coeffs = x.to_vec();
        let mut result = Multivector::<P, Q, R>::zero();

        for (v, w) in self.left_vectors.iter().zip(self.right_vectors.iter()) {
            let w_coeffs = w.to_vec();

            // Compute ⟨w, x⟩
            let inner_product: f64 = w_coeffs
                .iter()
                .zip(x_coeffs.iter())
                .map(|(a, b)| a * b)
                .sum();

            // Add ⟨w, x⟩ v
            result = result.add(&(v * inner_product));
        }

        Ok(result)
    }

    fn domain_dimension(&self) -> Option<usize> {
        Some(MatrixOperator::<P, Q, R>::DIM)
    }

    fn codomain_dimension(&self) -> Option<usize> {
        Some(MatrixOperator::<P, Q, R>::DIM)
    }
}

impl<const P: usize, const Q: usize, const R: usize>
    BoundedOperator<Multivector<P, Q, R>, Multivector<P, Q, R>, Bounded>
    for FiniteRankOperator<P, Q, R>
{
    fn operator_norm(&self) -> f64 {
        // Upper bound: sum of products of norms
        let mut sum = 0.0;
        for (v, w) in self.left_vectors.iter().zip(self.right_vectors.iter()) {
            let v_norm: f64 = v.to_vec().iter().map(|x| x * x).sum::<f64>().sqrt();
            let w_norm: f64 = w.to_vec().iter().map(|x| x * x).sum::<f64>().sqrt();
            sum += v_norm * w_norm;
        }
        sum
    }
}

impl<const P: usize, const Q: usize, const R: usize> CompactOperator<Multivector<P, Q, R>>
    for FiniteRankOperator<P, Q, R>
{
    fn is_finite_rank(&self) -> bool {
        true
    }

    fn rank(&self) -> Option<usize> {
        Some(self.left_vectors.len())
    }

    fn singular_values(&self) -> Result<Vec<f64>> {
        // Compute via T*T matrix eigenvalues
        let matrix = self.to_matrix()?;
        let t_star_t = matrix.transpose().multiply(&matrix)?;

        let eigenvalues = crate::spectral::compute_eigenvalues(&t_star_t, 100, 1e-10)?;

        // Singular values are square roots of eigenvalues
        let mut singular_values: Vec<f64> = eigenvalues
            .iter()
            .map(|e| e.value.abs().sqrt())
            .filter(|&s| s > 1e-10)
            .collect();

        singular_values.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        Ok(singular_values)
    }
}

impl<const P: usize, const Q: usize, const R: usize> FredholmOperator<Multivector<P, Q, R>>
    for FiniteRankOperator<P, Q, R>
{
    fn kernel_dimension(&self) -> usize {
        // Kernel has dimension n - rank
        let n = MatrixOperator::<P, Q, R>::DIM;
        n.saturating_sub(self.left_vectors.len())
    }

    fn cokernel_dimension(&self) -> usize {
        // For finite-rank operators, cokernel has dimension n - rank
        let n = MatrixOperator::<P, Q, R>::DIM;
        n.saturating_sub(self.left_vectors.len())
    }
}

/// Wrapper to mark a matrix operator as compact.
#[derive(Debug, Clone)]
pub struct CompactMatrixOperator<const P: usize, const Q: usize, const R: usize> {
    matrix: MatrixOperator<P, Q, R>,
    _marker: PhantomData<Compact>,
}

impl<const P: usize, const Q: usize, const R: usize> CompactMatrixOperator<P, Q, R> {
    /// Create a compact operator from a matrix.
    ///
    /// Note: In finite dimensions, all bounded operators are compact.
    pub fn new(matrix: MatrixOperator<P, Q, R>) -> Self {
        Self {
            matrix,
            _marker: PhantomData,
        }
    }

    /// Get the underlying matrix.
    pub fn matrix(&self) -> &MatrixOperator<P, Q, R> {
        &self.matrix
    }
}

impl<const P: usize, const Q: usize, const R: usize> LinearOperator<Multivector<P, Q, R>>
    for CompactMatrixOperator<P, Q, R>
{
    fn apply(&self, x: &Multivector<P, Q, R>) -> Result<Multivector<P, Q, R>> {
        self.matrix.apply(x)
    }

    fn domain_dimension(&self) -> Option<usize> {
        self.matrix.domain_dimension()
    }

    fn codomain_dimension(&self) -> Option<usize> {
        self.matrix.codomain_dimension()
    }
}

impl<const P: usize, const Q: usize, const R: usize>
    BoundedOperator<Multivector<P, Q, R>, Multivector<P, Q, R>, Bounded>
    for CompactMatrixOperator<P, Q, R>
{
    fn operator_norm(&self) -> f64 {
        self.matrix.operator_norm()
    }
}

impl<const P: usize, const Q: usize, const R: usize> CompactOperator<Multivector<P, Q, R>>
    for CompactMatrixOperator<P, Q, R>
{
    fn is_finite_rank(&self) -> bool {
        // All finite-dimensional operators have finite rank
        true
    }

    fn rank(&self) -> Option<usize> {
        // Compute rank via singular value decomposition
        let singular_values = self.singular_values().ok()?;
        Some(singular_values.len())
    }

    fn singular_values(&self) -> Result<Vec<f64>> {
        let t_star_t = self.matrix.transpose().multiply(&self.matrix)?;
        let eigenvalues = crate::spectral::compute_eigenvalues(&t_star_t, 100, 1e-10)?;

        let mut singular_values: Vec<f64> = eigenvalues
            .iter()
            .map(|e| e.value.abs().sqrt())
            .filter(|&s| s > 1e-10)
            .collect();

        singular_values.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        Ok(singular_values)
    }
}

/// Wrapper to mark a matrix operator as Fredholm.
#[derive(Debug, Clone)]
pub struct FredholmMatrixOperator<const P: usize, const Q: usize, const R: usize> {
    matrix: MatrixOperator<P, Q, R>,
    kernel_dim: usize,
    cokernel_dim: usize,
    _marker: PhantomData<FredholmMarker>,
}

impl<const P: usize, const Q: usize, const R: usize> FredholmMatrixOperator<P, Q, R> {
    /// Create a Fredholm operator from a matrix.
    ///
    /// Computes the kernel and cokernel dimensions.
    pub fn new(matrix: MatrixOperator<P, Q, R>) -> Result<Self> {
        // Compute kernel dimension by finding null space
        let eigenvalues = crate::spectral::compute_eigenvalues(&matrix, 100, 1e-10)?;
        let kernel_dim = eigenvalues.iter().filter(|e| e.value.abs() < 1e-10).count();

        // For square matrices, cokernel = kernel dimension
        let cokernel_dim = kernel_dim;

        Ok(Self {
            matrix,
            kernel_dim,
            cokernel_dim,
            _marker: PhantomData,
        })
    }

    /// Get the underlying matrix.
    pub fn matrix(&self) -> &MatrixOperator<P, Q, R> {
        &self.matrix
    }
}

impl<const P: usize, const Q: usize, const R: usize> LinearOperator<Multivector<P, Q, R>>
    for FredholmMatrixOperator<P, Q, R>
{
    fn apply(&self, x: &Multivector<P, Q, R>) -> Result<Multivector<P, Q, R>> {
        self.matrix.apply(x)
    }

    fn domain_dimension(&self) -> Option<usize> {
        self.matrix.domain_dimension()
    }

    fn codomain_dimension(&self) -> Option<usize> {
        self.matrix.codomain_dimension()
    }
}

impl<const P: usize, const Q: usize, const R: usize>
    BoundedOperator<Multivector<P, Q, R>, Multivector<P, Q, R>, Bounded>
    for FredholmMatrixOperator<P, Q, R>
{
    fn operator_norm(&self) -> f64 {
        self.matrix.operator_norm()
    }
}

impl<const P: usize, const Q: usize, const R: usize> FredholmOperator<Multivector<P, Q, R>>
    for FredholmMatrixOperator<P, Q, R>
{
    fn kernel_dimension(&self) -> usize {
        self.kernel_dim
    }

    fn cokernel_dimension(&self) -> usize {
        self.cokernel_dim
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rank_one_operator() {
        let v = Multivector::<2, 0, 0>::from_slice(&[1.0, 0.0, 0.0, 0.0]);
        let w = Multivector::<2, 0, 0>::from_slice(&[0.0, 1.0, 0.0, 0.0]);

        let op = FiniteRankOperator::rank_one(v, w);

        // Apply to x = (0, 1, 0, 0)
        // Result should be ⟨w, x⟩ v = 1 * (1, 0, 0, 0) = (1, 0, 0, 0)
        let x = Multivector::<2, 0, 0>::from_slice(&[0.0, 1.0, 0.0, 0.0]);
        let result = op.apply(&x).unwrap();

        assert!((result.to_vec()[0] - 1.0).abs() < 1e-10);
        assert!(result.to_vec()[1].abs() < 1e-10);
    }

    #[test]
    fn test_finite_rank_is_compact() {
        let v = Multivector::<2, 0, 0>::from_slice(&[1.0, 0.0, 0.0, 0.0]);
        let w = Multivector::<2, 0, 0>::from_slice(&[1.0, 0.0, 0.0, 0.0]);

        let op = FiniteRankOperator::rank_one(v, w);

        assert!(op.is_finite_rank());
        assert_eq!(op.rank(), Some(1));
    }

    #[test]
    fn test_finite_rank_to_matrix() {
        let v = Multivector::<2, 0, 0>::from_slice(&[1.0, 0.0, 0.0, 0.0]);
        let w = Multivector::<2, 0, 0>::from_slice(&[1.0, 0.0, 0.0, 0.0]);

        let op = FiniteRankOperator::rank_one(v, w);
        let matrix = op.to_matrix().unwrap();

        // The matrix should be a projection onto the first coordinate
        assert!((matrix.get(0, 0) - 1.0).abs() < 1e-10);
        assert!(matrix.get(0, 1).abs() < 1e-10);
        assert!(matrix.get(1, 0).abs() < 1e-10);
    }

    #[test]
    fn test_compact_matrix_operator() {
        let id: MatrixOperator<2, 0, 0> = MatrixOperator::identity();
        let compact = CompactMatrixOperator::new(id);

        assert!(compact.is_finite_rank());
        assert_eq!(compact.rank(), Some(4));
    }

    #[test]
    fn test_fredholm_identity() {
        let id: MatrixOperator<2, 0, 0> = MatrixOperator::identity();
        let fredholm = FredholmMatrixOperator::new(id).unwrap();

        // Identity has trivial kernel and cokernel
        assert_eq!(fredholm.kernel_dimension(), 0);
        assert_eq!(fredholm.cokernel_dimension(), 0);
        assert_eq!(fredholm.index(), 0);
        assert!(fredholm.is_isomorphism());
    }

    #[test]
    fn test_fredholm_singular() {
        // Create a singular matrix (projection onto first 2 dimensions)
        let proj: MatrixOperator<2, 0, 0> =
            MatrixOperator::diagonal(&[1.0, 1.0, 0.0, 0.0]).unwrap();
        let fredholm = FredholmMatrixOperator::new(proj).unwrap();

        // Should have 2-dimensional kernel
        assert_eq!(fredholm.kernel_dimension(), 2);
        assert_eq!(fredholm.index(), 0);
        assert!(!fredholm.is_isomorphism());
    }

    #[test]
    fn test_singular_values() {
        let id: MatrixOperator<2, 0, 0> = MatrixOperator::identity();
        let compact = CompactMatrixOperator::new(id);

        let sv = compact.singular_values().unwrap();

        // All singular values of identity are 1
        for s in &sv {
            assert!((s - 1.0).abs() < 0.1);
        }
    }
}
