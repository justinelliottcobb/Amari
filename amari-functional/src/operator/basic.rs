//! Basic linear operators.
//!
//! This module provides fundamental operator types that serve as
//! building blocks for more complex operators.

use crate::error::Result;
use crate::operator::traits::{BoundedOperator, LinearOperator};
use crate::phantom::Bounded;
use amari_core::Multivector;
use core::marker::PhantomData;

/// The identity operator I: x ↦ x.
///
/// The identity operator is the simplest non-trivial operator.
/// It has operator norm 1 and is self-adjoint.
#[derive(Debug, Clone, Copy)]
pub struct IdentityOperator<V> {
    _phantom: PhantomData<V>,
}

impl<V> Default for IdentityOperator<V> {
    fn default() -> Self {
        Self::new()
    }
}

impl<V> IdentityOperator<V> {
    /// Create a new identity operator.
    pub fn new() -> Self {
        Self {
            _phantom: PhantomData,
        }
    }
}

impl<const P: usize, const Q: usize, const R: usize> LinearOperator<Multivector<P, Q, R>>
    for IdentityOperator<Multivector<P, Q, R>>
{
    fn apply(&self, x: &Multivector<P, Q, R>) -> Result<Multivector<P, Q, R>> {
        Ok(x.clone())
    }

    fn domain_dimension(&self) -> Option<usize> {
        Some(1 << (P + Q + R))
    }

    fn codomain_dimension(&self) -> Option<usize> {
        Some(1 << (P + Q + R))
    }
}

impl<const P: usize, const Q: usize, const R: usize>
    BoundedOperator<Multivector<P, Q, R>, Multivector<P, Q, R>, Bounded>
    for IdentityOperator<Multivector<P, Q, R>>
{
    fn operator_norm(&self) -> f64 {
        1.0
    }
}

/// The zero operator 0: x ↦ 0.
///
/// The zero operator maps everything to zero.
/// It has operator norm 0.
#[derive(Debug, Clone, Copy)]
pub struct ZeroOperator<V, W = V> {
    _phantom: PhantomData<(V, W)>,
}

impl<V, W> Default for ZeroOperator<V, W> {
    fn default() -> Self {
        Self::new()
    }
}

impl<V, W> ZeroOperator<V, W> {
    /// Create a new zero operator.
    pub fn new() -> Self {
        Self {
            _phantom: PhantomData,
        }
    }
}

impl<const P: usize, const Q: usize, const R: usize> LinearOperator<Multivector<P, Q, R>>
    for ZeroOperator<Multivector<P, Q, R>>
{
    fn apply(&self, _x: &Multivector<P, Q, R>) -> Result<Multivector<P, Q, R>> {
        Ok(Multivector::<P, Q, R>::zero())
    }

    fn domain_dimension(&self) -> Option<usize> {
        Some(1 << (P + Q + R))
    }

    fn codomain_dimension(&self) -> Option<usize> {
        Some(1 << (P + Q + R))
    }
}

impl<const P: usize, const Q: usize, const R: usize>
    BoundedOperator<Multivector<P, Q, R>, Multivector<P, Q, R>, Bounded>
    for ZeroOperator<Multivector<P, Q, R>>
{
    fn operator_norm(&self) -> f64 {
        0.0
    }
}

/// Scaling operator αI: x ↦ αx.
///
/// Scales all elements by a fixed scalar.
#[derive(Debug, Clone, Copy)]
pub struct ScalingOperator<V> {
    /// The scaling factor.
    scalar: f64,
    _phantom: PhantomData<V>,
}

impl<V> ScalingOperator<V> {
    /// Create a new scaling operator.
    pub fn new(scalar: f64) -> Self {
        Self {
            scalar,
            _phantom: PhantomData,
        }
    }

    /// Get the scaling factor.
    pub fn scalar(&self) -> f64 {
        self.scalar
    }
}

impl<const P: usize, const Q: usize, const R: usize> LinearOperator<Multivector<P, Q, R>>
    for ScalingOperator<Multivector<P, Q, R>>
{
    fn apply(&self, x: &Multivector<P, Q, R>) -> Result<Multivector<P, Q, R>> {
        Ok(x * self.scalar)
    }

    fn domain_dimension(&self) -> Option<usize> {
        Some(1 << (P + Q + R))
    }

    fn codomain_dimension(&self) -> Option<usize> {
        Some(1 << (P + Q + R))
    }
}

impl<const P: usize, const Q: usize, const R: usize>
    BoundedOperator<Multivector<P, Q, R>, Multivector<P, Q, R>, Bounded>
    for ScalingOperator<Multivector<P, Q, R>>
{
    fn operator_norm(&self) -> f64 {
        self.scalar.abs()
    }
}

/// Orthogonal projection operator onto a subspace.
///
/// Projects onto the span of a set of orthonormal basis vectors.
#[derive(Clone)]
pub struct ProjectionOperator<const P: usize, const Q: usize, const R: usize> {
    /// Orthonormal basis for the projection subspace.
    basis: Vec<Multivector<P, Q, R>>,
}

impl<const P: usize, const Q: usize, const R: usize> std::fmt::Debug
    for ProjectionOperator<P, Q, R>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ProjectionOperator")
            .field("basis_size", &self.basis.len())
            .field("signature", &(P, Q, R))
            .finish()
    }
}

impl<const P: usize, const Q: usize, const R: usize> ProjectionOperator<P, Q, R> {
    /// Create a projection operator from an orthonormal basis.
    ///
    /// The basis vectors should already be orthonormal.
    pub fn from_orthonormal_basis(basis: Vec<Multivector<P, Q, R>>) -> Self {
        Self { basis }
    }

    /// Create a projection onto a single normalized direction.
    pub fn onto_direction(direction: Multivector<P, Q, R>) -> Self {
        Self {
            basis: vec![direction],
        }
    }

    /// Get the dimension of the projection subspace.
    pub fn subspace_dimension(&self) -> usize {
        self.basis.len()
    }
}

impl<const P: usize, const Q: usize, const R: usize> LinearOperator<Multivector<P, Q, R>>
    for ProjectionOperator<P, Q, R>
{
    fn apply(&self, x: &Multivector<P, Q, R>) -> Result<Multivector<P, Q, R>> {
        let mut result = Multivector::<P, Q, R>::zero();

        for basis_vec in &self.basis {
            // Compute ⟨x, basis_vec⟩ * basis_vec
            let x_coeffs = x.to_vec();
            let b_coeffs = basis_vec.to_vec();
            let inner_product: f64 = x_coeffs
                .iter()
                .zip(b_coeffs.iter())
                .map(|(a, b)| a * b)
                .sum();
            result = result.add(&(basis_vec * inner_product));
        }

        Ok(result)
    }

    fn domain_dimension(&self) -> Option<usize> {
        Some(1 << (P + Q + R))
    }

    fn codomain_dimension(&self) -> Option<usize> {
        Some(1 << (P + Q + R))
    }
}

impl<const P: usize, const Q: usize, const R: usize>
    BoundedOperator<Multivector<P, Q, R>, Multivector<P, Q, R>, Bounded>
    for ProjectionOperator<P, Q, R>
{
    fn operator_norm(&self) -> f64 {
        if self.basis.is_empty() {
            0.0
        } else {
            1.0
        }
    }
}

/// Composition of two operators: (S ∘ T)(x) = S(T(x)).
#[derive(Clone)]
pub struct CompositeOperator<S, T, V, W, U> {
    /// The outer operator S: W → U.
    outer: S,
    /// The inner operator T: V → W.
    inner: T,
    _phantom: PhantomData<(V, W, U)>,
}

impl<S, T, V, W, U> std::fmt::Debug for CompositeOperator<S, T, V, W, U>
where
    S: std::fmt::Debug,
    T: std::fmt::Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CompositeOperator")
            .field("outer", &self.outer)
            .field("inner", &self.inner)
            .finish()
    }
}

impl<S, T, V, W, U> CompositeOperator<S, T, V, W, U>
where
    S: LinearOperator<W, U>,
    T: LinearOperator<V, W>,
{
    /// Create a composite operator S ∘ T.
    pub fn new(outer: S, inner: T) -> Self {
        Self {
            outer,
            inner,
            _phantom: PhantomData,
        }
    }
}

impl<S, T, V, W, U> LinearOperator<V, U> for CompositeOperator<S, T, V, W, U>
where
    S: LinearOperator<W, U>,
    T: LinearOperator<V, W>,
{
    fn apply(&self, x: &V) -> Result<U> {
        let intermediate = self.inner.apply(x)?;
        self.outer.apply(&intermediate)
    }

    fn domain_dimension(&self) -> Option<usize> {
        self.inner.domain_dimension()
    }

    fn codomain_dimension(&self) -> Option<usize> {
        self.outer.codomain_dimension()
    }
}

impl<S, T, V, W, U> BoundedOperator<V, U, Bounded> for CompositeOperator<S, T, V, W, U>
where
    S: BoundedOperator<W, U, Bounded>,
    T: BoundedOperator<V, W, Bounded>,
{
    fn operator_norm(&self) -> f64 {
        // ||ST|| ≤ ||S|| ||T||
        self.outer.operator_norm() * self.inner.operator_norm()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::space::MultivectorHilbertSpace;

    #[test]
    fn test_identity_operator() {
        let identity: IdentityOperator<Multivector<2, 0, 0>> = IdentityOperator::new();
        let x = Multivector::<2, 0, 0>::from_slice(&[1.0, 2.0, 3.0, 4.0]);
        let y = identity.apply(&x).unwrap();
        assert_eq!(x.to_vec(), y.to_vec());
        assert!((identity.operator_norm() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_zero_operator() {
        let zero: ZeroOperator<Multivector<2, 0, 0>> = ZeroOperator::new();
        let x = Multivector::<2, 0, 0>::from_slice(&[1.0, 2.0, 3.0, 4.0]);
        let y = zero.apply(&x).unwrap();
        assert!(y.to_vec().iter().all(|&c| c.abs() < 1e-10));
        assert!((zero.operator_norm() - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_scaling_operator() {
        let scale: ScalingOperator<Multivector<2, 0, 0>> = ScalingOperator::new(2.0);
        let x = Multivector::<2, 0, 0>::from_slice(&[1.0, 2.0, 3.0, 4.0]);
        let y = scale.apply(&x).unwrap();
        assert_eq!(y.to_vec(), vec![2.0, 4.0, 6.0, 8.0]);
        assert!((scale.operator_norm() - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_projection_operator() {
        let space: MultivectorHilbertSpace<2, 0, 0> = MultivectorHilbertSpace::new();

        // Create a projection onto the first basis vector (e₀)
        let e0 = space.basis_vector(0).unwrap();
        let proj = ProjectionOperator::onto_direction(e0);

        let x = Multivector::<2, 0, 0>::from_slice(&[3.0, 4.0, 0.0, 0.0]);
        let y = proj.apply(&x).unwrap();

        // Projection of (3, 4, 0, 0) onto (1, 0, 0, 0) should be (3, 0, 0, 0)
        let y_coeffs = y.to_vec();
        assert!((y_coeffs[0] - 3.0).abs() < 1e-10);
        assert!(y_coeffs[1].abs() < 1e-10);
    }

    #[test]
    fn test_composite_operator() {
        let scale2: ScalingOperator<Multivector<2, 0, 0>> = ScalingOperator::new(2.0);
        let scale3: ScalingOperator<Multivector<2, 0, 0>> = ScalingOperator::new(3.0);

        let composite = CompositeOperator::new(scale2, scale3);

        let x = Multivector::<2, 0, 0>::from_slice(&[1.0, 0.0, 0.0, 0.0]);
        let y = composite.apply(&x).unwrap();

        // 2 * (3 * x) = 6x
        assert_eq!(y.to_vec(), vec![6.0, 0.0, 0.0, 0.0]);

        // ||S ∘ T|| ≤ ||S|| ||T|| = 6
        assert!((composite.operator_norm() - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_projection_is_idempotent() {
        let space: MultivectorHilbertSpace<2, 0, 0> = MultivectorHilbertSpace::new();

        // Create orthonormal basis for a 2D subspace
        let e0 = space.basis_vector(0).unwrap();
        let e1 = space.basis_vector(1).unwrap();
        let proj = ProjectionOperator::from_orthonormal_basis(vec![e0, e1]);

        let x = Multivector::<2, 0, 0>::from_slice(&[1.0, 2.0, 3.0, 4.0]);
        let y = proj.apply(&x).unwrap();
        let z = proj.apply(&y).unwrap();

        // P² = P (idempotent)
        let y_coeffs = y.to_vec();
        let z_coeffs = z.to_vec();
        for (a, b) in y_coeffs.iter().zip(z_coeffs.iter()) {
            assert!((a - b).abs() < 1e-10);
        }
    }
}
