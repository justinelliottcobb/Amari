//! Hilbert space structure for multivector-valued functions.
//!
//! This module provides the `MultivectorHilbertSpace` type, which represents
//! the Hilbert space of square-integrable multivector-valued functions.

use crate::error::{FunctionalError, Result};
use crate::phantom::Complete;
use crate::space::traits::{
    BanachSpace, HilbertSpace, InnerProductSpace, NormedSpace, VectorSpace,
};
use amari_core::Multivector;
use core::marker::PhantomData;

/// A Hilbert space of multivector elements.
///
/// This represents the finite-dimensional Hilbert space Cl(P,Q,R) with
/// the standard inner product inherited from the coefficient representation.
///
/// # Type Parameters
///
/// * `P` - Number of positive signature basis vectors
/// * `Q` - Number of negative signature basis vectors
/// * `R` - Number of zero signature basis vectors
///
/// # Mathematical Background
///
/// The Clifford algebra Cl(P,Q,R) is a 2^(P+Q+R)-dimensional real vector space.
/// We equip it with the standard L² inner product on the coefficients:
///
/// ⟨x, y⟩ = Σᵢ xᵢ yᵢ
///
/// This makes Cl(P,Q,R) into a finite-dimensional Hilbert space.
#[derive(Debug, Clone)]
pub struct MultivectorHilbertSpace<const P: usize, const Q: usize, const R: usize> {
    _phantom: PhantomData<(Multivector<P, Q, R>,)>,
}

impl<const P: usize, const Q: usize, const R: usize> Default for MultivectorHilbertSpace<P, Q, R> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const P: usize, const Q: usize, const R: usize> MultivectorHilbertSpace<P, Q, R> {
    /// The dimension of the Clifford algebra.
    pub const DIM: usize = 1 << (P + Q + R);

    /// Create a new multivector Hilbert space.
    pub fn new() -> Self {
        Self {
            _phantom: PhantomData,
        }
    }

    /// Get the signature (P, Q, R) of the algebra.
    pub fn signature(&self) -> (usize, usize, usize) {
        (P, Q, R)
    }

    /// Get the dimension of the algebra.
    pub fn algebra_dimension(&self) -> usize {
        Self::DIM
    }

    /// Create a multivector from coefficients.
    pub fn from_coefficients(&self, coefficients: &[f64]) -> Result<Multivector<P, Q, R>> {
        if coefficients.len() != Self::DIM {
            return Err(FunctionalError::dimension_mismatch(
                Self::DIM,
                coefficients.len(),
            ));
        }

        Ok(Multivector::<P, Q, R>::from_slice(coefficients))
    }

    /// Get the coefficients of a multivector.
    pub fn to_coefficients(&self, mv: &Multivector<P, Q, R>) -> Vec<f64> {
        mv.to_vec()
    }

    /// Create a basis vector (unit multivector in direction i).
    pub fn basis_vector(&self, index: usize) -> Result<Multivector<P, Q, R>> {
        if index >= Self::DIM {
            return Err(FunctionalError::invalid_parameters(format!(
                "Basis index {} out of range [0, {})",
                index,
                Self::DIM
            )));
        }

        let mut coeffs = vec![0.0; Self::DIM];
        coeffs[index] = 1.0;
        self.from_coefficients(&coeffs)
    }

    /// Get all basis vectors.
    pub fn basis(&self) -> Vec<Multivector<P, Q, R>> {
        (0..Self::DIM)
            .map(|i| self.basis_vector(i).unwrap())
            .collect()
    }
}

impl<const P: usize, const Q: usize, const R: usize> VectorSpace<Multivector<P, Q, R>, f64>
    for MultivectorHilbertSpace<P, Q, R>
{
    fn add(&self, x: &Multivector<P, Q, R>, y: &Multivector<P, Q, R>) -> Multivector<P, Q, R> {
        x.add(y)
    }

    fn sub(&self, x: &Multivector<P, Q, R>, y: &Multivector<P, Q, R>) -> Multivector<P, Q, R> {
        x - y
    }

    fn scale(&self, scalar: f64, x: &Multivector<P, Q, R>) -> Multivector<P, Q, R> {
        x * scalar
    }

    fn zero(&self) -> Multivector<P, Q, R> {
        Multivector::<P, Q, R>::zero()
    }

    fn dimension(&self) -> Option<usize> {
        Some(Self::DIM)
    }
}

impl<const P: usize, const Q: usize, const R: usize> NormedSpace<Multivector<P, Q, R>, f64>
    for MultivectorHilbertSpace<P, Q, R>
{
    fn norm(&self, x: &Multivector<P, Q, R>) -> f64 {
        // L² norm on coefficients
        let coeffs = x.to_vec();
        coeffs.iter().map(|c| c * c).sum::<f64>().sqrt()
    }

    fn normalize(&self, x: &Multivector<P, Q, R>) -> Option<Multivector<P, Q, R>> {
        let n = self.norm(x);
        if n < 1e-15 {
            None
        } else {
            Some(self.scale(1.0 / n, x))
        }
    }
}

impl<const P: usize, const Q: usize, const R: usize> InnerProductSpace<Multivector<P, Q, R>, f64>
    for MultivectorHilbertSpace<P, Q, R>
{
    fn inner_product(&self, x: &Multivector<P, Q, R>, y: &Multivector<P, Q, R>) -> f64 {
        let x_coeffs = x.to_vec();
        let y_coeffs = y.to_vec();
        x_coeffs
            .iter()
            .zip(y_coeffs.iter())
            .map(|(a, b)| a * b)
            .sum()
    }

    fn project(&self, x: &Multivector<P, Q, R>, y: &Multivector<P, Q, R>) -> Multivector<P, Q, R> {
        let ip_xy = self.inner_product(x, y);
        let ip_yy = self.inner_product(y, y);
        if ip_yy.abs() < 1e-15 {
            self.zero()
        } else {
            self.scale(ip_xy / ip_yy, y)
        }
    }

    fn gram_schmidt(&self, vectors: &[Multivector<P, Q, R>]) -> Vec<Multivector<P, Q, R>> {
        let mut orthonormal = Vec::new();
        for v in vectors {
            let mut u = v.clone();
            for q in &orthonormal {
                let proj = self.project(&u, q);
                u = self.sub(&u, &proj);
            }
            if let Some(normalized) = self.normalize(&u) {
                orthonormal.push(normalized);
            }
        }
        orthonormal
    }
}

impl<const P: usize, const Q: usize, const R: usize>
    BanachSpace<Multivector<P, Q, R>, f64, Complete> for MultivectorHilbertSpace<P, Q, R>
{
    fn is_cauchy_sequence(&self, sequence: &[Multivector<P, Q, R>], tolerance: f64) -> bool {
        if sequence.len() < 2 {
            return true;
        }

        // Check if the last few terms are getting closer
        let n = sequence.len();
        for i in (n.saturating_sub(5))..n {
            for j in (i + 1)..n {
                if self.distance(&sequence[i], &sequence[j]) > tolerance {
                    return false;
                }
            }
        }
        true
    }

    fn sequence_limit(
        &self,
        sequence: &[Multivector<P, Q, R>],
        tolerance: f64,
    ) -> Result<Multivector<P, Q, R>> {
        if sequence.is_empty() {
            return Err(FunctionalError::convergence_error(
                0,
                "Empty sequence has no limit",
            ));
        }

        if !self.is_cauchy_sequence(sequence, tolerance) {
            return Err(FunctionalError::convergence_error(
                sequence.len(),
                "Sequence is not Cauchy",
            ));
        }

        // For finite-dimensional spaces, Cauchy sequences converge
        // Return the last element as the limit approximation
        Ok(sequence.last().unwrap().clone())
    }
}

impl<const P: usize, const Q: usize, const R: usize>
    HilbertSpace<Multivector<P, Q, R>, f64, Complete> for MultivectorHilbertSpace<P, Q, R>
{
    fn riesz_representative<F>(&self, functional: F) -> Result<Multivector<P, Q, R>>
    where
        F: Fn(&Multivector<P, Q, R>) -> f64,
    {
        // In finite dimensions, the Riesz representative is just
        // the vector of functional values on the basis
        let mut coeffs = Vec::with_capacity(Self::DIM);
        for i in 0..Self::DIM {
            let basis_i = self.basis_vector(i)?;
            coeffs.push(functional(&basis_i));
        }
        self.from_coefficients(&coeffs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hilbert_space_creation() {
        let space: MultivectorHilbertSpace<3, 0, 0> = MultivectorHilbertSpace::new();
        assert_eq!(space.algebra_dimension(), 8);
        assert_eq!(space.signature(), (3, 0, 0));
    }

    #[test]
    fn test_basis_vectors() {
        let space: MultivectorHilbertSpace<2, 0, 0> = MultivectorHilbertSpace::new();
        let basis = space.basis();
        assert_eq!(basis.len(), 4); // 2^2 = 4

        // Verify orthonormality
        for (i, bi) in basis.iter().enumerate() {
            assert!((space.norm(bi) - 1.0).abs() < 1e-10);
            for (j, bj) in basis.iter().enumerate() {
                if i != j {
                    assert!(space.inner_product(bi, bj).abs() < 1e-10);
                }
            }
        }
    }

    #[test]
    fn test_vector_space_operations() {
        let space: MultivectorHilbertSpace<2, 0, 0> = MultivectorHilbertSpace::new();

        let x = space.from_coefficients(&[1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = space.from_coefficients(&[5.0, 6.0, 7.0, 8.0]).unwrap();

        let sum = space.add(&x, &y);
        let sum_coeffs = space.to_coefficients(&sum);
        assert_eq!(sum_coeffs, vec![6.0, 8.0, 10.0, 12.0]);

        let scaled = space.scale(2.0, &x);
        let scaled_coeffs = space.to_coefficients(&scaled);
        assert_eq!(scaled_coeffs, vec![2.0, 4.0, 6.0, 8.0]);
    }

    #[test]
    fn test_inner_product() {
        let space: MultivectorHilbertSpace<2, 0, 0> = MultivectorHilbertSpace::new();

        let x = space.from_coefficients(&[1.0, 0.0, 0.0, 0.0]).unwrap();
        let y = space.from_coefficients(&[0.0, 1.0, 0.0, 0.0]).unwrap();
        let z = space.from_coefficients(&[1.0, 1.0, 0.0, 0.0]).unwrap();

        // Orthogonality
        assert!(space.inner_product(&x, &y).abs() < 1e-10);
        assert!(!space.are_orthogonal(&x, &z, 1e-10));

        // Self inner product equals squared norm
        assert!((space.inner_product(&z, &z) - 2.0).abs() < 1e-10);
        assert!((space.norm(&z) - 2.0_f64.sqrt()).abs() < 1e-10);
    }

    #[test]
    fn test_gram_schmidt() {
        let space: MultivectorHilbertSpace<2, 0, 0> = MultivectorHilbertSpace::new();

        let v1 = space.from_coefficients(&[1.0, 1.0, 0.0, 0.0]).unwrap();
        let v2 = space.from_coefficients(&[1.0, 0.0, 1.0, 0.0]).unwrap();
        let v3 = space.from_coefficients(&[0.0, 1.0, 1.0, 0.0]).unwrap();

        let orthonormal = space.gram_schmidt(&[v1, v2, v3]);
        assert_eq!(orthonormal.len(), 3);

        // Verify orthonormality
        assert!(space.is_orthonormal(&orthonormal, 1e-10));
    }

    #[test]
    fn test_riesz_representation() {
        let space: MultivectorHilbertSpace<2, 0, 0> = MultivectorHilbertSpace::new();

        // Define a linear functional: f(x) = x₀ + 2x₁
        let functional = |x: &Multivector<2, 0, 0>| {
            let coeffs = x.to_vec();
            coeffs[0] + 2.0 * coeffs[1]
        };

        let repr = space.riesz_representative(functional).unwrap();
        let repr_coeffs = space.to_coefficients(&repr);

        // The Riesz representative should be [1, 2, 0, 0]
        assert!((repr_coeffs[0] - 1.0).abs() < 1e-10);
        assert!((repr_coeffs[1] - 2.0).abs() < 1e-10);
        assert!(repr_coeffs[2].abs() < 1e-10);
        assert!(repr_coeffs[3].abs() < 1e-10);
    }

    #[test]
    fn test_projection() {
        let space: MultivectorHilbertSpace<2, 0, 0> = MultivectorHilbertSpace::new();

        let x = space.from_coefficients(&[3.0, 4.0, 0.0, 0.0]).unwrap();
        let y = space.from_coefficients(&[1.0, 0.0, 0.0, 0.0]).unwrap();

        let proj = space.project(&x, &y);
        let proj_coeffs = space.to_coefficients(&proj);

        // Projection of (3,4,0,0) onto (1,0,0,0) should be (3,0,0,0)
        assert!((proj_coeffs[0] - 3.0).abs() < 1e-10);
        assert!(proj_coeffs[1].abs() < 1e-10);
    }

    #[test]
    fn test_best_approximation() {
        let space: MultivectorHilbertSpace<2, 0, 0> = MultivectorHilbertSpace::new();

        let x = space.from_coefficients(&[1.0, 2.0, 3.0, 4.0]).unwrap();

        // Project onto the subspace spanned by first two basis vectors
        let e0 = space.basis_vector(0).unwrap();
        let e1 = space.basis_vector(1).unwrap();

        let approx = space.best_approximation(&x, &[e0, e1]);
        let approx_coeffs = space.to_coefficients(&approx);

        // Best approximation should be (1, 2, 0, 0)
        assert!((approx_coeffs[0] - 1.0).abs() < 1e-10);
        assert!((approx_coeffs[1] - 2.0).abs() < 1e-10);
        assert!(approx_coeffs[2].abs() < 1e-10);
        assert!(approx_coeffs[3].abs() < 1e-10);
    }
}
