use amari_dual::{DualMultivector, DualNumber};

mod dual_tests {
    use super::*;

    #[test]
    fn test_dual_number_addition() {
        let a = DualNumber::variable(3.0); // 3 + 1ε
        let b = DualNumber::variable(5.0); // 5 + 1ε

        let sum = a + b;
        assert_eq!(sum.value(), 8.0);
        assert_eq!(sum.derivative(), 2.0); // Derivative
    }

    #[test]
    fn test_dual_number_multiplication() {
        let a = DualNumber::variable(3.0); // 3 + 1ε
        let b = DualNumber::constant(5.0); // 5 + 0ε

        let product = a * b;
        assert_eq!(product.value(), 15.0);
        assert_eq!(product.derivative(), 5.0); // Derivative: d/dx(5x) = 5
    }

    #[test]
    fn test_dual_number_chain_rule() {
        // f(x) = (x + 2)²
        let x = DualNumber::variable(3.0);
        let x_plus_2 = x + DualNumber::constant(2.0);
        let squared = x_plus_2 * x_plus_2;

        assert_eq!(squared.value(), 25.0); // (3 + 2)² = 25
        assert_eq!(squared.derivative(), 10.0); // d/dx((x+2)²) at x=3 = 2(3+2) = 10
    }

    #[test]
    fn test_dual_multivector_geometric_product_derivative() {
        let a = DualMultivector::<f64, 3, 0, 0>::new_variable(&[1.0, 2.0, 3.0]);
        let b = DualMultivector::<f64, 3, 0, 0>::constant(&[4.0, 5.0, 6.0]);

        let product = a.geometric_product(&b);

        // Check that we get both value and derivative
        assert!(!product.value().as_slice().is_empty());
        assert!(!product.derivative().as_slice().is_empty());
    }
}
