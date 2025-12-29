//! Core traits for function spaces.
//!
//! This module defines the fundamental traits that characterize
//! different types of function spaces in functional analysis.

use crate::error::Result;
use crate::phantom::{Complete, CompletenessProperty};
use core::marker::PhantomData;

/// A vector space over a scalar field.
///
/// This is the most fundamental algebraic structure, providing
/// addition and scalar multiplication.
///
/// # Type Parameters
///
/// * `V` - The type of vectors in the space
/// * `S` - The scalar field (typically `f64` or `Complex<f64>`)
pub trait VectorSpace<V, S = f64> {
    /// Add two vectors.
    fn add(&self, x: &V, y: &V) -> V;

    /// Subtract two vectors.
    fn sub(&self, x: &V, y: &V) -> V;

    /// Multiply a vector by a scalar.
    fn scale(&self, scalar: S, x: &V) -> V;

    /// Return the zero vector.
    fn zero(&self) -> V;

    /// Return the dimension of the space (may be infinite).
    fn dimension(&self) -> Option<usize>;
}

/// A normed vector space.
///
/// Adds a norm ||·|| to a vector space, satisfying:
/// - ||x|| ≥ 0 and ||x|| = 0 iff x = 0
/// - ||αx|| = |α| ||x||
/// - ||x + y|| ≤ ||x|| + ||y|| (triangle inequality)
///
/// # Type Parameters
///
/// * `V` - The type of vectors in the space
/// * `S` - The scalar field
pub trait NormedSpace<V, S = f64>: VectorSpace<V, S> {
    /// Compute the norm of a vector.
    fn norm(&self, x: &V) -> f64;

    /// Compute the distance between two vectors.
    fn distance(&self, x: &V, y: &V) -> f64 {
        self.norm(&self.sub(x, y))
    }

    /// Normalize a vector to unit length.
    ///
    /// Returns `None` if the vector is zero.
    fn normalize(&self, x: &V) -> Option<V>
    where
        S: From<f64>,
        V: Clone;
}

/// A complete normed space (Banach space).
///
/// A normed space where every Cauchy sequence converges.
/// This is essential for many existence theorems in analysis.
///
/// # Type Parameters
///
/// * `V` - The type of vectors in the space
/// * `S` - The scalar field
/// * `C` - Completeness phantom type (must be `Complete`)
pub trait BanachSpace<V, S = f64, C = Complete>: NormedSpace<V, S>
where
    C: CompletenessProperty,
{
    /// Check if a sequence appears to be Cauchy.
    ///
    /// This is a numerical approximation based on the last few terms.
    fn is_cauchy_sequence(&self, sequence: &[V], tolerance: f64) -> bool;

    /// Compute the limit of a Cauchy sequence if it exists.
    fn sequence_limit(&self, sequence: &[V], tolerance: f64) -> Result<V>;
}

/// An inner product space.
///
/// Adds an inner product ⟨·,·⟩ satisfying:
/// - ⟨x,y⟩ = conj(⟨y,x⟩) (conjugate symmetry)
/// - ⟨αx+βy,z⟩ = α⟨x,z⟩ + β⟨y,z⟩ (linearity)
/// - ⟨x,x⟩ ≥ 0 and ⟨x,x⟩ = 0 iff x = 0 (positive definiteness)
///
/// The norm is induced by ||x|| = √⟨x,x⟩.
///
/// # Type Parameters
///
/// * `V` - The type of vectors in the space
/// * `S` - The scalar field (inner product values)
pub trait InnerProductSpace<V, S = f64>: NormedSpace<V, S> {
    /// Compute the inner product of two vectors.
    fn inner_product(&self, x: &V, y: &V) -> S;

    /// Check if two vectors are orthogonal.
    fn are_orthogonal(&self, x: &V, y: &V, tolerance: f64) -> bool
    where
        S: Into<f64>,
    {
        let ip: f64 = self.inner_product(x, y).into();
        ip.abs() < tolerance
    }

    /// Project x onto y.
    ///
    /// Returns the component of x in the direction of y.
    fn project(&self, x: &V, y: &V) -> V
    where
        S: Into<f64> + From<f64>,
        V: Clone;

    /// Gram-Schmidt orthogonalization of a set of vectors.
    fn gram_schmidt(&self, vectors: &[V]) -> Vec<V>
    where
        S: Into<f64> + From<f64>,
        V: Clone;
}

/// A complete inner product space (Hilbert space).
///
/// The most important space in functional analysis, combining
/// the geometric structure of inner products with completeness.
///
/// # Type Parameters
///
/// * `V` - The type of vectors in the space
/// * `S` - The scalar field
/// * `C` - Completeness phantom type (must be `Complete`)
pub trait HilbertSpace<V, S = f64, C = Complete>:
    InnerProductSpace<V, S> + BanachSpace<V, S, C>
where
    C: CompletenessProperty,
{
    /// Apply the Riesz representation theorem.
    ///
    /// Given a continuous linear functional f, returns the unique
    /// vector v such that f(x) = ⟨x, v⟩ for all x.
    fn riesz_representative<F>(&self, functional: F) -> Result<V>
    where
        F: Fn(&V) -> S;

    /// Compute the orthogonal complement projection.
    ///
    /// Given a closed subspace (represented by an orthonormal basis),
    /// project onto the orthogonal complement.
    fn orthogonal_complement_projection(&self, x: &V, subspace_basis: &[V]) -> V
    where
        S: Into<f64> + From<f64>,
        V: Clone,
    {
        let mut result = x.clone();
        for basis_vec in subspace_basis {
            let proj = self.project(&result, basis_vec);
            result = self.sub(&result, &proj);
        }
        result
    }

    /// Compute the best approximation to x from a subspace.
    ///
    /// Returns the element of the subspace closest to x.
    fn best_approximation(&self, x: &V, subspace_basis: &[V]) -> V
    where
        S: Into<f64> + From<f64>,
        V: Clone,
    {
        let mut result = self.zero();
        for basis_vec in subspace_basis {
            let proj = self.project(x, basis_vec);
            result = self.add(&result, &proj);
        }
        result
    }

    /// Check if a set of vectors forms an orthonormal system.
    fn is_orthonormal(&self, vectors: &[V], tolerance: f64) -> bool
    where
        S: Into<f64>,
    {
        for (i, vi) in vectors.iter().enumerate() {
            // Check normalization
            let norm_i = self.norm(vi);
            if (norm_i - 1.0).abs() > tolerance {
                return false;
            }

            // Check orthogonality
            for vj in vectors.iter().skip(i + 1) {
                if !self.are_orthogonal(vi, vj, tolerance) {
                    return false;
                }
            }
        }
        true
    }
}

/// Marker struct for spaces with specific completeness properties.
#[derive(Debug, Clone, Copy)]
pub struct SpaceWithCompleteness<S, C: CompletenessProperty> {
    space: S,
    _completeness: PhantomData<C>,
}

impl<S, C: CompletenessProperty> SpaceWithCompleteness<S, C> {
    /// Create a new space with completeness marker.
    pub fn new(space: S) -> Self {
        Self {
            space,
            _completeness: PhantomData,
        }
    }

    /// Get a reference to the underlying space.
    pub fn space(&self) -> &S {
        &self.space
    }

    /// Get a mutable reference to the underlying space.
    pub fn space_mut(&mut self) -> &mut S {
        &mut self.space
    }

    /// Consume and return the underlying space.
    pub fn into_inner(self) -> S {
        self.space
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Simple finite-dimensional test space
    struct R3Space;

    impl VectorSpace<[f64; 3], f64> for R3Space {
        fn add(&self, x: &[f64; 3], y: &[f64; 3]) -> [f64; 3] {
            [x[0] + y[0], x[1] + y[1], x[2] + y[2]]
        }

        fn sub(&self, x: &[f64; 3], y: &[f64; 3]) -> [f64; 3] {
            [x[0] - y[0], x[1] - y[1], x[2] - y[2]]
        }

        fn scale(&self, scalar: f64, x: &[f64; 3]) -> [f64; 3] {
            [scalar * x[0], scalar * x[1], scalar * x[2]]
        }

        fn zero(&self) -> [f64; 3] {
            [0.0, 0.0, 0.0]
        }

        fn dimension(&self) -> Option<usize> {
            Some(3)
        }
    }

    impl NormedSpace<[f64; 3], f64> for R3Space {
        fn norm(&self, x: &[f64; 3]) -> f64 {
            (x[0] * x[0] + x[1] * x[1] + x[2] * x[2]).sqrt()
        }

        fn normalize(&self, x: &[f64; 3]) -> Option<[f64; 3]> {
            let n = self.norm(x);
            if n < 1e-15 {
                None
            } else {
                Some(self.scale(1.0 / n, x))
            }
        }
    }

    impl InnerProductSpace<[f64; 3], f64> for R3Space {
        fn inner_product(&self, x: &[f64; 3], y: &[f64; 3]) -> f64 {
            x[0] * y[0] + x[1] * y[1] + x[2] * y[2]
        }

        fn project(&self, x: &[f64; 3], y: &[f64; 3]) -> [f64; 3] {
            let ip_xy = self.inner_product(x, y);
            let ip_yy = self.inner_product(y, y);
            if ip_yy.abs() < 1e-15 {
                self.zero()
            } else {
                self.scale(ip_xy / ip_yy, y)
            }
        }

        fn gram_schmidt(&self, vectors: &[[f64; 3]]) -> Vec<[f64; 3]> {
            let mut orthonormal = Vec::new();
            for v in vectors {
                let mut u = *v;
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

    #[test]
    fn test_vector_space_operations() {
        let space = R3Space;

        let x = [1.0, 2.0, 3.0];
        let y = [4.0, 5.0, 6.0];

        let sum = space.add(&x, &y);
        assert_eq!(sum, [5.0, 7.0, 9.0]);

        let diff = space.sub(&x, &y);
        assert_eq!(diff, [-3.0, -3.0, -3.0]);

        let scaled = space.scale(2.0, &x);
        assert_eq!(scaled, [2.0, 4.0, 6.0]);

        assert_eq!(space.dimension(), Some(3));
    }

    #[test]
    fn test_normed_space_operations() {
        let space = R3Space;

        let x = [3.0, 4.0, 0.0];
        assert!((space.norm(&x) - 5.0).abs() < 1e-10);

        let normalized = space.normalize(&x).unwrap();
        assert!((space.norm(&normalized) - 1.0).abs() < 1e-10);

        let y = [1.0, 0.0, 0.0];
        let dist = space.distance(&x, &y);
        assert!((dist - (4.0_f64 + 16.0).sqrt()).abs() < 1e-10);
    }

    #[test]
    fn test_inner_product_space_operations() {
        let space = R3Space;

        let x = [1.0, 0.0, 0.0];
        let y = [0.0, 1.0, 0.0];
        let z = [1.0, 1.0, 0.0];

        // Orthogonality
        assert!(space.are_orthogonal(&x, &y, 1e-10));
        assert!(!space.are_orthogonal(&x, &z, 1e-10));

        // Projection
        let proj = space.project(&z, &x);
        assert!((proj[0] - 1.0).abs() < 1e-10);
        assert!(proj[1].abs() < 1e-10);
        assert!(proj[2].abs() < 1e-10);
    }

    #[test]
    fn test_gram_schmidt() {
        let space = R3Space;

        let vectors = [[1.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0]];
        let orthonormal = space.gram_schmidt(&vectors);

        assert_eq!(orthonormal.len(), 3);

        // Check orthonormality
        for (i, vi) in orthonormal.iter().enumerate() {
            // Check unit length
            assert!((space.norm(vi) - 1.0).abs() < 1e-10);

            // Check orthogonality
            for (j, vj) in orthonormal.iter().enumerate() {
                if i != j {
                    assert!(space.are_orthogonal(vi, vj, 1e-10));
                }
            }
        }
    }
}
