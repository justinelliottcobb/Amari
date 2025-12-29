//! Core traits for linear operators.
//!
//! This module defines the fundamental traits for linear operators
//! on Hilbert spaces.

use crate::error::Result;
use crate::phantom::{Bounded, BoundednessProperty, SelfAdjoint, SymmetryProperty};

/// A linear operator between vector spaces.
///
/// A linear operator T: V → W satisfies:
/// - T(x + y) = T(x) + T(y)
/// - T(αx) = αT(x)
///
/// # Type Parameters
///
/// * `V` - Domain space element type
/// * `W` - Codomain space element type
pub trait LinearOperator<V, W = V> {
    /// Apply the operator to an element.
    fn apply(&self, x: &V) -> Result<W>;

    /// Get the domain dimension (None if infinite).
    fn domain_dimension(&self) -> Option<usize>;

    /// Get the codomain dimension (None if infinite).
    fn codomain_dimension(&self) -> Option<usize>;
}

/// A bounded linear operator.
///
/// A linear operator T is bounded if there exists M > 0 such that
/// ||Tx|| ≤ M||x|| for all x.
///
/// The operator norm is ||T|| = sup{||Tx|| : ||x|| = 1}.
///
/// # Type Parameters
///
/// * `V` - Domain space element type
/// * `W` - Codomain space element type
/// * `B` - Boundedness phantom marker
pub trait BoundedOperator<V, W = V, B = Bounded>: LinearOperator<V, W>
where
    B: BoundednessProperty,
{
    /// Compute or estimate the operator norm ||T||.
    fn operator_norm(&self) -> f64;

    /// Check if the operator is bounded by a given constant.
    fn is_bounded_by(&self, bound: f64) -> bool {
        self.operator_norm() <= bound
    }
}

/// Trait for computing operator norms.
pub trait OperatorNorm {
    /// Compute the operator norm.
    fn norm(&self) -> f64;

    /// Compute the Frobenius norm (for matrix operators).
    fn frobenius_norm(&self) -> Option<f64> {
        None
    }
}

/// An operator that has an adjoint.
///
/// The adjoint T* of an operator T satisfies:
/// ⟨Tx, y⟩ = ⟨x, T*y⟩
///
/// # Type Parameters
///
/// * `V` - The Hilbert space element type
pub trait AdjointableOperator<V>: LinearOperator<V, V> {
    /// The type of the adjoint operator.
    type Adjoint: LinearOperator<V, V>;

    /// Compute the adjoint operator.
    fn adjoint(&self) -> Self::Adjoint;

    /// Check if this operator is self-adjoint (T = T*).
    fn is_self_adjoint(&self) -> bool;

    /// Check if this operator is normal (TT* = T*T).
    fn is_normal(&self) -> bool;
}

/// A self-adjoint operator (T = T*).
///
/// Self-adjoint operators have:
/// - Real spectrum
/// - Orthogonal eigenvectors
/// - Spectral decomposition
///
/// # Type Parameters
///
/// * `V` - The Hilbert space element type
/// * `S` - Symmetry phantom marker (should be SelfAdjoint)
pub trait SelfAdjointOperator<V, S = SelfAdjoint>: LinearOperator<V, V>
where
    S: SymmetryProperty,
{
    /// Check if all eigenvalues are real (always true for self-adjoint).
    fn has_real_spectrum(&self) -> bool {
        true
    }

    /// Compute the quadratic form ⟨Tx, x⟩.
    fn quadratic_form(&self, x: &V) -> Result<f64>;

    /// Check if the operator is positive (⟨Tx, x⟩ ≥ 0).
    fn is_positive(&self, x: &V) -> Result<bool> {
        Ok(self.quadratic_form(x)? >= 0.0)
    }

    /// Check if the operator is positive definite (⟨Tx, x⟩ > 0 for x ≠ 0).
    fn is_positive_definite(&self, x: &V, tolerance: f64) -> Result<bool> {
        Ok(self.quadratic_form(x)? > tolerance)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use amari_core::Multivector;

    // Simple test operator: identity
    struct TestIdentity;

    impl LinearOperator<Multivector<2, 0, 0>> for TestIdentity {
        fn apply(&self, x: &Multivector<2, 0, 0>) -> Result<Multivector<2, 0, 0>> {
            Ok(x.clone())
        }

        fn domain_dimension(&self) -> Option<usize> {
            Some(4) // 2^2 = 4
        }

        fn codomain_dimension(&self) -> Option<usize> {
            Some(4)
        }
    }

    impl BoundedOperator<Multivector<2, 0, 0>> for TestIdentity {
        fn operator_norm(&self) -> f64 {
            1.0
        }
    }

    #[test]
    fn test_linear_operator_apply() {
        let identity = TestIdentity;
        let x = Multivector::<2, 0, 0>::from_slice(&[1.0, 2.0, 3.0, 4.0]);
        let y = identity.apply(&x).unwrap();
        assert_eq!(x.to_vec(), y.to_vec());
    }

    #[test]
    fn test_bounded_operator_norm() {
        let identity = TestIdentity;
        assert!((identity.operator_norm() - 1.0).abs() < 1e-10);
        assert!(identity.is_bounded_by(1.0));
        assert!(identity.is_bounded_by(2.0));
        assert!(!identity.is_bounded_by(0.5));
    }

    #[test]
    fn test_operator_dimensions() {
        let identity = TestIdentity;
        assert_eq!(identity.domain_dimension(), Some(4));
        assert_eq!(identity.codomain_dimension(), Some(4));
    }
}
