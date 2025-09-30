//! Comprehensive test suite for dual number automatic differentiation
//!
//! This module provides extensive testing for mathematical properties,
//! edge cases, and automatic differentiation correctness.

#[cfg(test)]
mod tests {
    use crate::functions::*;
    use crate::multivector::DualMultivector;
    use crate::{DualNumber, MultiDualNumber};
    use num_traits::{Float, One, Zero};

    const EPSILON: f64 = 1e-10;

    // ========== Basic Dual Number Properties ==========

    #[test]
    fn test_dual_number_creation() {
        let d = DualNumber::new(3.0, 2.0);
        assert_eq!(d.real, 3.0);
        assert_eq!(d.dual, 2.0);

        let var = DualNumber::variable(5.0);
        assert_eq!(var.real, 5.0);
        assert_eq!(var.dual, 1.0);

        let constant = DualNumber::constant(7.0);
        assert_eq!(constant.real, 7.0);
        assert_eq!(constant.dual, 0.0);
    }

    #[test]
    fn test_dual_number_arithmetic_laws() {
        let a = DualNumber::new(2.0, 3.0);
        let b = DualNumber::new(4.0, 5.0);
        let c = DualNumber::new(1.0, 2.0);

        // Commutativity of addition
        assert_eq!(a + b, b + a);

        // Associativity of addition
        assert_eq!((a + b) + c, a + (b + c));

        // Commutativity of multiplication
        assert_eq!(a * b, b * a);

        // Associativity of multiplication
        let prod1 = (a * b) * c;
        let prod2 = a * (b * c);
        assert!((prod1.real - prod2.real).abs() < EPSILON);
        assert!((prod1.dual - prod2.dual).abs() < EPSILON);

        // Distributivity
        let dist1 = a * (b + c);
        let dist2 = a * b + a * c;
        assert!((dist1.real - dist2.real).abs() < EPSILON);
        assert!((dist1.dual - dist2.dual).abs() < EPSILON);
    }

    #[test]
    fn test_dual_number_identities() {
        let a = DualNumber::new(3.0, 2.0);
        let zero = DualNumber::zero();
        let one = DualNumber::one();

        // Additive identity
        assert_eq!(a + zero, a);
        assert_eq!(zero + a, a);

        // Multiplicative identity
        assert_eq!(a * one, a);
        assert_eq!(one * a, a);

        // Additive inverse
        let neg_a = -a;
        let sum = a + neg_a;
        assert!(sum.real.abs() < EPSILON);
        assert!(sum.dual.abs() < EPSILON);
    }

    #[test]
    fn test_chain_rule() {
        // f(g(x)) where f(u) = u² and g(x) = 3x + 2
        let x = DualNumber::variable(2.0);
        let g = x * 3.0 + DualNumber::constant(2.0); // g(2) = 8, g'(2) = 3
        let f = g * g; // f(g(2)) = 64, f'(g(2)) * g'(2) = 2*8*3 = 48

        assert!((f.real - 64.0).abs() < EPSILON);
        assert!((f.dual - 48.0).abs() < EPSILON);
    }

    #[test]
    fn test_product_rule() {
        // (f*g)' = f'*g + f*g'
        let x = DualNumber::variable(3.0);
        let f = x * 2.0; // f(x) = 2x, f'(x) = 2
        let g = x * x; // g(x) = x², g'(x) = 2x
        let product = f * g; // (2x)(x²) = 2x³

        // At x=3: f(3)=6, f'(3)=2, g(3)=9, g'(3)=6
        // (fg)'(3) = 2*9 + 6*6 = 18 + 36 = 54
        assert!((product.real - 54.0).abs() < EPSILON);
        assert!((product.dual - 54.0).abs() < EPSILON);
    }

    #[test]
    fn test_quotient_rule() {
        // (f/g)' = (f'*g - f*g')/g²
        let x = DualNumber::variable(2.0);
        let f = x * 3.0 + DualNumber::constant(1.0); // f(x) = 3x + 1, f'(x) = 3
        let g = x * x; // g(x) = x², g'(x) = 2x
        let quotient = f / g;

        // At x=2: f(2)=7, f'(2)=3, g(2)=4, g'(2)=4
        // (f/g)'(2) = (3*4 - 7*4)/16 = (12-28)/16 = -16/16 = -1
        assert!((quotient.real - 1.75).abs() < EPSILON); // 7/4 = 1.75
        assert!((quotient.dual - (-1.0)).abs() < EPSILON);
    }

    // ========== Transcendental Functions ==========

    #[test]
    fn test_exponential_derivative() {
        let x = DualNumber::variable(1.0);
        let exp_x = x.exp();

        // exp'(x) = exp(x)
        assert!((exp_x.real - 1.0_f64.exp()).abs() < EPSILON);
        assert!((exp_x.dual - 1.0_f64.exp()).abs() < EPSILON);
    }

    #[test]
    fn test_logarithm_derivative() {
        let x = DualNumber::variable(2.0);
        let ln_x = x.ln();

        // ln'(x) = 1/x
        assert!((ln_x.real - 2.0_f64.ln()).abs() < EPSILON);
        assert!((ln_x.dual - 0.5).abs() < EPSILON);
    }

    #[test]
    fn test_trigonometric_derivatives() {
        let x = DualNumber::variable(std::f64::consts::PI / 4.0);

        let sin_x = x.sin();
        let cos_x = x.cos();
        let tan_x = x.tan();

        // sin'(x) = cos(x)
        assert!((sin_x.dual - (std::f64::consts::PI / 4.0).cos()).abs() < EPSILON);

        // cos'(x) = -sin(x)
        assert!((cos_x.dual + (std::f64::consts::PI / 4.0).sin()).abs() < EPSILON);

        // tan'(x) = sec²(x) = 1/cos²(x)
        let sec_squared = 1.0 / ((std::f64::consts::PI / 4.0).cos().powi(2));
        assert!((tan_x.dual - sec_squared).abs() < EPSILON);
    }

    #[test]
    fn test_hyperbolic_derivatives() {
        let x = DualNumber::variable(1.0);

        let sinh_x = x.sinh();
        let cosh_x = x.cosh();
        let tanh_x = x.tanh();

        // sinh'(x) = cosh(x)
        assert!((sinh_x.dual - 1.0_f64.cosh()).abs() < EPSILON);

        // cosh'(x) = sinh(x)
        assert!((cosh_x.dual - 1.0_f64.sinh()).abs() < EPSILON);

        // tanh'(x) = sech²(x) = 1/cosh²(x)
        let sech_squared = 1.0 / (1.0_f64.cosh().powi(2));
        assert!((tanh_x.dual - sech_squared).abs() < EPSILON);
    }

    #[test]
    fn test_power_derivatives() {
        let x = DualNumber::variable(3.0);

        // x² derivative
        let x_squared = x.powi(2);
        assert!((x_squared.real - 9.0).abs() < EPSILON);
        assert!((x_squared.dual - 6.0).abs() < EPSILON); // 2x at x=3

        // x³ derivative
        let x_cubed = x.powi(3);
        assert!((x_cubed.real - 27.0).abs() < EPSILON);
        assert!((x_cubed.dual - 27.0).abs() < EPSILON); // 3x² at x=3

        // x^0.5 derivative
        let x_sqrt = x.sqrt();
        assert!((x_sqrt.real - 3.0_f64.sqrt()).abs() < EPSILON);
        assert!((x_sqrt.dual - 0.5 / 3.0_f64.sqrt()).abs() < EPSILON);

        // x^π derivative
        let x_pi = x.powf(std::f64::consts::PI);
        let expected_deriv = std::f64::consts::PI * 3.0_f64.powf(std::f64::consts::PI - 1.0);
        assert!((x_pi.dual - expected_deriv).abs() < 1e-8);
    }

    // ========== Multi-Variable Dual Numbers ==========

    #[test]
    fn test_multi_dual_creation() {
        let m = MultiDualNumber::new(2.0, vec![1.0, 2.0, 3.0]);
        assert_eq!(m.real, 2.0);
        assert_eq!(m.duals, vec![1.0, 2.0, 3.0]);

        let var = MultiDualNumber::variable(5.0, 3, 1);
        assert_eq!(var.real, 5.0);
        assert_eq!(var.duals, vec![0.0, 1.0, 0.0]);
    }

    #[test]
    fn test_multi_dual_gradient() {
        // f(x, y, z) = x²y + yz²
        let x = MultiDualNumber::variable(2.0, 3, 0);
        let y = MultiDualNumber::variable(3.0, 3, 1);
        let z = MultiDualNumber::variable(4.0, 3, 2);

        let f = &(&x * &x) * &y + &(&y * &(&z * &z));

        // At (2,3,4): f = 4*3 + 3*16 = 12 + 48 = 60
        assert!((f.real - 60.0).abs() < EPSILON);

        // ∂f/∂x = 2xy = 2*2*3 = 12
        assert!((f.duals[0] - 12.0).abs() < EPSILON);

        // ∂f/∂y = x² + z² = 4 + 16 = 20
        assert!((f.duals[1] - 20.0).abs() < EPSILON);

        // ∂f/∂z = 2yz = 2*3*4 = 24
        assert!((f.duals[2] - 24.0).abs() < EPSILON);
    }

    #[test]
    fn test_jacobian_computation() {
        // Vector function F(x,y) = [x² + y, xy]
        let x = MultiDualNumber::variable(2.0, 2, 0);
        let y = MultiDualNumber::variable(3.0, 2, 1);

        let f1 = &(&x * &x) + &y;
        let f2 = &x * &y;

        // F(2,3) = [7, 6]
        assert!((f1.real - 7.0).abs() < EPSILON);
        assert!((f2.real - 6.0).abs() < EPSILON);

        // Jacobian at (2,3):
        // [∂f1/∂x  ∂f1/∂y]   [2x  1]   [4  1]
        // [∂f2/∂x  ∂f2/∂y] = [y   x] = [3  2]

        assert!((f1.duals[0] - 4.0).abs() < EPSILON);
        assert!((f1.duals[1] - 1.0).abs() < EPSILON);
        assert!((f2.duals[0] - 3.0).abs() < EPSILON);
        assert!((f2.duals[1] - 2.0).abs() < EPSILON);
    }

    // ========== Edge Cases and Numerical Stability ==========

    #[test]
    fn test_division_by_zero_handling() {
        let x = DualNumber::variable(0.0);
        let one = DualNumber::constant(1.0);
        let result = one / x;

        assert!(result.real.is_infinite());
        assert!(result.dual.is_infinite());
    }

    #[test]
    fn test_logarithm_of_zero() {
        let x = DualNumber::variable(0.0);
        let ln_x = x.ln();

        assert!(ln_x.real.is_infinite() && ln_x.real.is_sign_negative());
        assert!(ln_x.dual.is_infinite());
    }

    #[test]
    fn test_sqrt_of_negative() {
        let x = DualNumber::variable(-1.0);
        let sqrt_x = x.sqrt();

        assert!(sqrt_x.real.is_nan());
        assert!(sqrt_x.dual.is_nan());
    }

    #[test]
    fn test_large_values() {
        let x = DualNumber::variable(1e100);
        let y = DualNumber::variable(1e100);
        let product = x * y;

        assert!(product.real.is_finite());
        assert!(product.dual.is_finite());
    }

    #[test]
    fn test_small_values() {
        let x = DualNumber::variable(1e-100);
        let y = DualNumber::variable(1e-100);
        let product = x * y;

        assert_eq!(product.real, 1e-200);
        assert!(product.dual.abs() < 1e-99);
    }

    // ========== Dual Multivector Tests ==========

    #[test]
    fn test_dual_multivector_creation() {
        let dm = DualMultivector::<f64, 3, 0, 0>::scalar(DualNumber::new(2.0, 1.0));
        assert_eq!(dm.grade(), 0);

        let basis_vec =
            DualMultivector::<f64, 3, 0, 0>::basis_vector(1).expect("Should create basis vector");
        assert_eq!(basis_vec.grade(), 1);
    }

    #[test]
    fn test_dual_multivector_geometric_product() {
        let a = DualMultivector::<f64, 2, 0, 0>::basis_vector(0).unwrap();
        let b = DualMultivector::<f64, 2, 0, 0>::basis_vector(1).unwrap();

        let product = a.geometric_product(&b);

        // e₁ * e₂ = e₁₂ (bivector)
        assert_eq!(product.grade(), 2);
    }

    #[test]
    fn test_dual_multivector_automatic_differentiation() {
        // Create a multivector function and compute its derivative
        let x = DualNumber::variable(2.0);
        let scalar = DualMultivector::<f64, 3, 0, 0>::scalar(x);

        // Square the multivector
        let squared = scalar.geometric_product(&scalar);

        // Check that we get the correct derivative
        let result = squared.get(0);
        assert!((result.real - 4.0).abs() < EPSILON);
        assert!((result.dual - 4.0).abs() < EPSILON); // d/dx(x²) = 2x at x=2
    }

    // ========== Integration with Other Functions ==========

    #[test]
    fn test_softmax_gradient() {
        let logits = vec![
            DualNumber::variable(1.0),
            DualNumber::constant(2.0),
            DualNumber::constant(3.0),
        ];

        let softmax_result = softmax(&logits);

        // Softmax derivative with respect to first input
        // ∂softmax_i/∂x_j = softmax_i(δ_ij - softmax_j)
        let s0 = softmax_result[0].real;
        let expected_grad = s0 * (1.0 - s0);

        assert!((softmax_result[0].dual - expected_grad).abs() < 1e-6);
    }

    #[test]
    fn test_cross_entropy_gradient() {
        let inputs = vec![
            DualNumber::variable(0.5),
            DualNumber::constant(0.3),
            DualNumber::constant(0.2),
        ];
        let targets = vec![1.0, 0.0, 0.0];

        let loss = cross_entropy_loss(&inputs, &targets);

        // Cross entropy gradient: ∂L/∂x_i = softmax_i - target_i
        // For inputs [0.5, 0.3, 0.2] and targets [1.0, 0.0, 0.0]
        // We need to compute softmax values manually
        use crate::functions::softmax;
        let softmax_probs = softmax(&inputs);
        let expected_grad = softmax_probs[0].real - targets[0];
        assert!((loss.dual - expected_grad).abs() < EPSILON);
    }

    // ========== Performance and Optimization Tests ==========

    #[test]
    fn test_nested_function_composition() {
        // f(g(h(x))) where h(x) = x², g(u) = exp(u), f(v) = sin(v)
        let x = DualNumber::variable(0.5);
        let h = x * x; // h(0.5) = 0.25, h'(0.5) = 1.0
        let g = h.exp(); // g(h(0.5)) = exp(0.25), g'(h)*h' = exp(0.25)*1.0
        let f = g.sin(); // f(g(h(0.5))) = sin(exp(0.25))

        // Chain rule: f'(g(h(x))) * g'(h(x)) * h'(x)
        // = cos(exp(x²)) * exp(x²) * 2x at x=0.5
        let expected_value = (0.25_f64.exp()).sin();
        let expected_deriv = (0.25_f64.exp()).cos() * 0.25_f64.exp() * 1.0;

        assert!((f.real - expected_value).abs() < EPSILON);
        assert!((f.dual - expected_deriv).abs() < EPSILON);
    }

    #[test]
    fn test_vector_norm_derivative() {
        // ||v|| = sqrt(x² + y² + z²)
        let x = MultiDualNumber::variable(3.0, 3, 0);
        let y = MultiDualNumber::variable(4.0, 3, 1);
        let z = MultiDualNumber::variable(0.0, 3, 2);

        let norm_squared = &(&x * &x) + &(&(&y * &y) + &(&z * &z));
        let norm = norm_squared.sqrt();

        // ||v|| = 5
        assert!((norm.real - 5.0).abs() < EPSILON);

        // ∂||v||/∂x = x/||v|| = 3/5
        assert!((norm.duals[0] - 0.6).abs() < EPSILON);

        // ∂||v||/∂y = y/||v|| = 4/5
        assert!((norm.duals[1] - 0.8).abs() < EPSILON);

        // ∂||v||/∂z = z/||v|| = 0/5
        assert!(norm.duals[2].abs() < EPSILON);
    }
}
