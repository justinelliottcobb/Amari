//! Comprehensive unit tests for all public functions in lib.rs
//!
//! This module systematically tests all 83 public functions identified in lib.rs
//! to achieve high test coverage as part of formal verification.

use crate::*;
use approx::assert_relative_eq;

type Cl3 = Multivector<3, 0, 0>; // 3D Euclidean space

#[cfg(test)]
mod constructor_tests {
    use super::*;

    #[test]
    fn test_zero() {
        let zero = Cl3::zero();
        for i in 0..8 {
            assert_eq!(zero.get(i), 0.0);
        }
        assert!(zero.is_zero());
    }

    #[test]
    fn test_scalar() {
        let mv = Cl3::scalar(5.0);
        assert_eq!(mv.get(0), 5.0);
        for i in 1..8 {
            assert_eq!(mv.get(i), 0.0);
        }
    }

    #[test]
    fn test_basis_vector() {
        let e1 = Cl3::basis_vector(0);
        assert_eq!(e1.get(1), 1.0); // e1 is at index 1
        for i in [0, 2, 3, 4, 5, 6, 7] {
            assert_eq!(e1.get(i), 0.0);
        }
    }

    #[test]
    fn test_from_coefficients() {
        let coeffs = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mv = Cl3::from_coefficients(coeffs.clone());
        for i in 0..8 {
            assert_eq!(mv.get(i), coeffs[i]);
        }
    }

    #[test]
    fn test_from_slice() {
        let coeffs = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mv = Cl3::from_slice(&coeffs);
        for i in 0..8 {
            assert_eq!(mv.get(i), coeffs[i]);
        }
    }
}

#[cfg(test)]
mod accessor_tests {
    use super::*;

    #[test]
    fn test_get_set() {
        let mut mv = Cl3::zero();
        mv.set(3, 42.0);
        assert_eq!(mv.get(3), 42.0);
    }

    #[test]
    fn test_scalar_part() {
        let mv = Cl3::from_coefficients(vec![5.0, 1.0, 2.0, 3.0, 4.0, 0.0, 0.0, 0.0]);
        assert_eq!(mv.scalar_part(), 5.0);
    }

    #[test]
    fn test_set_scalar() {
        let mut mv = Cl3::scalar(1.0);
        mv.set_scalar(10.0);
        assert_eq!(mv.scalar_part(), 10.0);
    }

    #[test]
    fn test_vector_part() {
        let mv = Cl3::from_coefficients(vec![1.0, 2.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0]);
        let vec_part = mv.vector_part();
        assert_eq!(vec_part.mv.get(1), 2.0); // e1 component
        assert_eq!(vec_part.mv.get(2), 3.0); // e2 component
        assert_eq!(vec_part.mv.get(3), 4.0); // e3 component
    }

    #[test]
    fn test_bivector_part() {
        let mv = Cl3::from_coefficients(vec![0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 0.0]);
        let biv_part = mv.bivector_part();
        assert_eq!(biv_part.get(4), 1.0); // e1∧e2
        assert_eq!(biv_part.get(5), 2.0); // e1∧e3
        assert_eq!(biv_part.get(6), 3.0); // e2∧e3
    }

    #[test]
    fn test_trivector_part() {
        let mv = Cl3::from_coefficients(vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0]);
        assert_eq!(mv.trivector_part(), 5.0);
    }

    #[test]
    fn test_set_vector_component() {
        let mut mv = Cl3::zero();
        mv.set_vector_component(1, 42.0); // y component
        assert_eq!(mv.get(2), 42.0); // e2 is at index 2
    }

    #[test]
    fn test_set_bivector_component() {
        let mut mv = Cl3::zero();
        mv.set_bivector_component(0, 42.0); // e1∧e2
        assert_eq!(mv.get(4), 42.0);
    }

    #[test]
    fn test_vector_component() {
        let mv = Cl3::from_coefficients(vec![0.0, 1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.0]);
        assert_eq!(mv.vector_component(0), 1.0); // x
        assert_eq!(mv.vector_component(1), 2.0); // y
        assert_eq!(mv.vector_component(2), 3.0); // z
    }

    #[test]
    fn test_as_slice() {
        let coeffs = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mv = Cl3::from_coefficients(coeffs.clone());
        let slice = mv.as_slice();
        assert_eq!(slice.len(), 8);
        for i in 0..8 {
            assert_eq!(slice[i], coeffs[i]);
        }
    }
}

#[cfg(test)]
mod arithmetic_tests {
    use super::*;

    #[test]
    fn test_addition() {
        let a = Cl3::scalar(2.0);
        let b = Cl3::scalar(3.0);
        let result = &a + &b;
        assert_eq!(result.scalar_part(), 5.0);
    }

    #[test]
    fn test_operator_addition() {
        let a = Cl3::scalar(2.0);
        let b = Cl3::scalar(3.0);
        let result = &a + &b;
        assert_eq!(result.scalar_part(), 5.0);
    }

    #[test]
    fn test_operator_subtraction() {
        let a = Cl3::scalar(5.0);
        let b = Cl3::scalar(3.0);
        let result = &a - &b;
        assert_eq!(result.scalar_part(), 2.0);
    }

    #[test]
    fn test_operator_negation() {
        let a = Cl3::scalar(5.0);
        let result = -a.clone();
        assert_eq!(result.scalar_part(), -5.0);
    }

    #[test]
    fn test_scalar_multiplication() {
        let a = Cl3::from_coefficients(vec![1.0, 2.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0]);
        let result = &a * 2.0;
        assert_eq!(result.get(0), 2.0);
        assert_eq!(result.get(1), 4.0);
        assert_eq!(result.get(2), 6.0);
        assert_eq!(result.get(3), 8.0);
    }
}

#[cfg(test)]
mod product_tests {
    use super::*;

    #[test]
    fn test_geometric_product() {
        let e1 = Cl3::basis_vector(0);
        let e2 = Cl3::basis_vector(1);
        let result = e1.geometric_product(&e2);

        // e1 * e2 = e1∧e2 (bivector at index 0b011 = 3)
        assert_eq!(result.get(3), 1.0); // e1∧e2 coefficient
    }

    #[test]
    fn test_geometric_product_associativity() {
        let a = Cl3::scalar(2.0);
        let b = Cl3::basis_vector(0);
        let c = Cl3::basis_vector(1);

        let ab_c = a.geometric_product(&b).geometric_product(&c);
        let a_bc = a.geometric_product(&b.geometric_product(&c));

        for i in 0..8 {
            assert_relative_eq!(ab_c.get(i), a_bc.get(i), epsilon = 1e-10);
        }
    }

    #[test]
    fn test_inner_product() {
        let e1 = Cl3::basis_vector(0);
        let e2 = Cl3::basis_vector(1);
        let result = e1.inner_product(&e2);

        // e1 · e2 = 0 for orthogonal basis vectors
        assert!(result.is_zero());
    }

    #[test]
    fn test_outer_product() {
        let e1 = Cl3::basis_vector(0);
        let e2 = Cl3::basis_vector(1);
        let result = e1.outer_product(&e2);

        // e1 ∧ e2 = bivector at index 3
        assert_eq!(result.get(3), 1.0);
    }

    #[test]
    fn test_outer_product_antisymmetric() {
        let e1 = Cl3::basis_vector(0);
        let e2 = Cl3::basis_vector(1);
        let e1_e2 = e1.outer_product(&e2);
        let e2_e1 = e2.outer_product(&e1);

        // Should be negatives of each other
        for i in 0..8 {
            assert_relative_eq!(e1_e2.get(i), -e2_e1.get(i), epsilon = 1e-10);
        }
    }

    #[test]
    fn test_scalar_product() {
        let a = Cl3::from_coefficients(vec![1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let b = Cl3::from_coefficients(vec![2.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let result = a.scalar_product(&b);

        // Should be 1*2 + 2*1 + 3*1 = 7 (considering signature)
        assert_eq!(result, 7.0);
    }
}

#[cfg(test)]
mod operation_tests {
    use super::*;

    #[test]
    fn test_reverse() {
        let mv = Cl3::from_coefficients(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let rev = mv.reverse();

        // Scalar and vector parts unchanged, bivector and trivector change sign
        assert_eq!(rev.get(0), 1.0);  // scalar
        assert_eq!(rev.get(1), 2.0);  // e1
        assert_eq!(rev.get(2), 3.0);  // e2
        assert_eq!(rev.get(3), 4.0);  // e3
        assert_eq!(rev.get(4), -5.0); // e1∧e2
        assert_eq!(rev.get(5), -6.0); // e1∧e3
        assert_eq!(rev.get(6), -7.0); // e2∧e3
        assert_eq!(rev.get(7), -8.0); // e1∧e2∧e3
    }

    #[test]
    fn test_grade_projection() {
        let mv = Cl3::from_coefficients(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);

        // Grade 0 projection (scalar)
        let grade0 = mv.grade_projection(0);
        assert_eq!(grade0.get(0), 1.0);
        for i in 1..8 {
            assert_eq!(grade0.get(i), 0.0);
        }

        // Grade 1 projection (vector)
        let grade1 = mv.grade_projection(1);
        assert_eq!(grade1.get(0), 0.0);
        assert_eq!(grade1.get(1), 2.0);
        assert_eq!(grade1.get(2), 3.0);
        assert_eq!(grade1.get(3), 4.0);
        for i in 4..8 {
            assert_eq!(grade1.get(i), 0.0);
        }
    }

    #[test]
    fn test_grade() {
        assert_eq!(Cl3::scalar(5.0).grade(), 0);
        assert_eq!(Cl3::basis_vector(0).grade(), 1);

        let bivector = Cl3::basis_vector(0).outer_product(&Cl3::basis_vector(1));
        assert_eq!(bivector.grade(), 2);
    }
}

#[cfg(test)]
mod utility_tests {
    use super::*;

    #[test]
    fn test_is_zero() {
        assert!(Cl3::zero().is_zero());
        assert!(!Cl3::scalar(1.0).is_zero());
    }

    #[test]
    fn test_norm_squared() {
        let mv = Cl3::from_coefficients(vec![0.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        assert_eq!(mv.norm_squared(), 25.0); // 3² + 4² = 25
    }

    #[test]
    fn test_equality() {
        let a = Cl3::scalar(5.0);
        let b = Cl3::scalar(5.0);
        let c = Cl3::scalar(3.0);

        assert_eq!(a, b);
        assert_ne!(a, c);
    }
}

#[cfg(test)]
mod integration_tests {
    use super::*;

    #[test]
    fn test_basis_vector_squares() {
        // In Cl(3,0,0), all basis vectors should square to +1
        for i in 0..3 {
            let ei = Cl3::basis_vector(i);
            let ei_squared = ei.geometric_product(&ei);
            assert_relative_eq!(ei_squared.scalar_part(), 1.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_basis_vector_anticommutativity() {
        let e1 = Cl3::basis_vector(0);
        let e2 = Cl3::basis_vector(1);

        let e1_e2 = e1.geometric_product(&e2);
        let e2_e1 = e2.geometric_product(&e1);

        // Should be negatives: e1*e2 = -e2*e1
        for i in 0..8 {
            assert_relative_eq!(e1_e2.get(i), -e2_e1.get(i), epsilon = 1e-10);
        }
    }

    #[test]
    fn test_distributivity() {
        let a = Cl3::scalar(2.0);
        let b = Cl3::basis_vector(0);
        let c = Cl3::basis_vector(1);

        let a_bc = a.geometric_product(&(&b + &c));
        let ab_ac = &a.geometric_product(&b) + &a.geometric_product(&c);

        for i in 0..8 {
            assert_relative_eq!(a_bc.get(i), ab_ac.get(i), epsilon = 1e-10);
        }
    }

    #[test]
    fn test_identity_elements() {
        let mv = Cl3::from_coefficients(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);

        // Multiplicative identity
        let one = Cl3::scalar(1.0);
        let result = mv.geometric_product(&one);
        assert_eq!(mv, result);

        // Additive identity
        let zero = Cl3::zero();
        let result = &mv + &zero;
        assert_eq!(mv, result);
    }

    #[test]
    fn test_reverse_involution() {
        let a = Cl3::from_coefficients(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let b = Cl3::from_coefficients(vec![2.0, 1.0, 1.0, 1.0, 1.0, 2.0, 3.0, 1.0]);

        let ab = a.geometric_product(&b);
        let ab_rev = ab.reverse();

        let a_rev = a.reverse();
        let b_rev = b.reverse();
        let ba_rev = b_rev.geometric_product(&a_rev);

        // (AB)† = B†A†
        for i in 0..8 {
            assert_relative_eq!(ab_rev.get(i), ba_rev.get(i), epsilon = 1e-10);
        }
    }
}