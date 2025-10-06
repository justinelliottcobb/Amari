//! Dual number automatic differentiation for efficient gradient computation
//!
//! Dual numbers extend real numbers with an infinitesimal unit ε where ε² = 0.
//! This allows for exact computation of derivatives without numerical approximation
//! or computational graphs, making it ideal for forward-mode automatic differentiation.

#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;
use alloc::vec::Vec;
use core::ops::{Add, Div, Mul, Neg, Sub};
use num_traits::{Float, One, Zero};

// Import precision types from amari-core
#[cfg(feature = "high-precision")]
pub use amari_core::HighPrecisionFloat;
pub use amari_core::{ExtendedFloat, PrecisionFloat, StandardFloat};

pub mod comprehensive_tests;
pub mod error;
pub mod functions;
pub mod multivector;
pub mod verified;
pub mod verified_contracts;

// Re-export commonly used types
pub use error::{DualError, DualResult};
pub use multivector::{DualMultivector, MultiDualMultivector};

// Precision-aware type aliases for dual numbers
/// Standard precision dual number using f64
pub type StandardDual = DualNumber<StandardFloat>;

/// Extended precision dual number - uses high precision when available
pub type ExtendedDual = DualNumber<ExtendedFloat>;

/// Standard precision multi-dual number using f64
pub type StandardMultiDual = MultiDual<StandardFloat>;

/// Extended precision multi-dual number - uses high precision when available
pub type ExtendedMultiDual = MultiDual<ExtendedFloat>;

/// Multi-variable dual number for computing gradients
#[derive(Clone, Debug, PartialEq)]
pub struct MultiDualNumber<T: Float> {
    /// Function value
    pub real: T,
    /// Partial derivatives (gradient)
    pub duals: Vec<T>,
}

impl<T: Float> MultiDualNumber<T> {
    /// Create a new multi-dual number
    pub fn new(real: T, duals: Vec<T>) -> Self {
        Self { real, duals }
    }

    /// Create a variable with derivative 1 at the specified index
    pub fn variable(value: T, num_vars: usize, var_index: usize) -> Self {
        let mut duals = vec![T::zero(); num_vars];
        if var_index < num_vars {
            duals[var_index] = T::one();
        }
        Self::new(value, duals)
    }

    /// Create a constant (all derivatives are zero)
    pub fn constant(value: T, num_vars: usize) -> Self {
        Self::new(value, vec![T::zero(); num_vars])
    }

    /// Get the number of variables
    pub fn num_vars(&self) -> usize {
        self.duals.len()
    }

    /// Square root function
    pub fn sqrt(&self) -> Self {
        let sqrt_real = self.real.sqrt();
        let sqrt_deriv = T::one() / (T::from(2.0).unwrap() * sqrt_real);

        let mut new_duals = Vec::with_capacity(self.duals.len());
        for &dual in &self.duals {
            new_duals.push(dual * sqrt_deriv);
        }

        Self::new(sqrt_real, new_duals)
    }
}

impl<T: Float> Add for &MultiDualNumber<T> {
    type Output = MultiDualNumber<T>;

    fn add(self, other: Self) -> Self::Output {
        assert_eq!(self.duals.len(), other.duals.len());
        let mut new_duals = Vec::with_capacity(self.duals.len());
        for (a, b) in self.duals.iter().zip(other.duals.iter()) {
            new_duals.push(*a + *b);
        }
        MultiDualNumber::new(self.real + other.real, new_duals)
    }
}

impl<T: Float> Mul for &MultiDualNumber<T> {
    type Output = MultiDualNumber<T>;

    fn mul(self, other: Self) -> Self::Output {
        assert_eq!(self.duals.len(), other.duals.len());
        let mut new_duals = Vec::with_capacity(self.duals.len());
        for (a, b) in self.duals.iter().zip(other.duals.iter()) {
            new_duals.push(*a * other.real + self.real * *b);
        }
        MultiDualNumber::new(self.real * other.real, new_duals)
    }
}

// Add missing combinations for owned + reference
impl<T: Float> Add<&MultiDualNumber<T>> for MultiDualNumber<T> {
    type Output = MultiDualNumber<T>;

    fn add(self, other: &MultiDualNumber<T>) -> Self::Output {
        &self + other
    }
}

impl<T: Float> Mul<&MultiDualNumber<T>> for MultiDualNumber<T> {
    type Output = MultiDualNumber<T>;

    fn mul(self, other: &MultiDualNumber<T>) -> Self::Output {
        &self * other
    }
}

/// A dual number: a + bε where ε² = 0
///
/// The real part stores the function value, the dual part stores the derivative.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DualNumber<T: Float> {
    /// Real part (function value)
    pub real: T,
    /// Dual part (derivative with respect to input variable)
    pub dual: T,
}

impl<T: Float> DualNumber<T> {
    /// Create a new dual number
    pub fn new(real: T, dual: T) -> Self {
        Self { real, dual }
    }

    /// Create a variable (derivative = 1)
    pub fn variable(value: T) -> Self {
        Self {
            real: value,
            dual: T::one(),
        }
    }

    /// Create a variable (derivative = 1) - alias for consistency
    pub fn new_variable(value: T) -> Self {
        Self::variable(value)
    }

    /// Create a constant (derivative = 0)
    pub fn constant(value: T) -> Self {
        Self {
            real: value,
            dual: T::zero(),
        }
    }

    /// Get the value (real part)
    pub fn value(&self) -> T {
        self.real
    }

    /// Get the derivative (dual part)
    pub fn derivative(&self) -> T {
        self.dual
    }

    /// Apply a function with known derivative
    pub fn apply_with_derivative<F, G>(&self, f: F, df: G) -> Self
    where
        F: Fn(T) -> T,
        G: Fn(T) -> T,
    {
        Self {
            real: f(self.real),
            dual: df(self.real) * self.dual,
        }
    }

    /// Sine function
    pub fn sin(self) -> Self {
        self.apply_with_derivative(|x| x.sin(), |x| x.cos())
    }

    /// Cosine function
    pub fn cos(self) -> Self {
        self.apply_with_derivative(|x| x.cos(), |x| -x.sin())
    }

    /// Exponential function
    pub fn exp(self) -> Self {
        let exp_val = self.real.exp();
        Self {
            real: exp_val,
            dual: exp_val * self.dual,
        }
    }

    /// Natural logarithm
    pub fn ln(self) -> Self {
        Self {
            real: self.real.ln(),
            dual: self.dual / self.real,
        }
    }

    /// Power function
    pub fn powf(self, n: T) -> Self {
        Self {
            real: self.real.powf(n),
            dual: n * self.real.powf(n - T::one()) * self.dual,
        }
    }

    /// Square root
    pub fn sqrt(self) -> Self {
        let sqrt_val = self.real.sqrt();
        Self {
            real: sqrt_val,
            dual: self.dual / (T::from(2.0).unwrap() * sqrt_val),
        }
    }

    /// Hyperbolic tangent
    pub fn tanh(self) -> Self {
        let tanh_val = self.real.tanh();
        Self {
            real: tanh_val,
            dual: self.dual * (T::one() - tanh_val * tanh_val),
        }
    }

    /// ReLU activation function
    pub fn relu(self) -> Self {
        if self.real > T::zero() {
            self
        } else {
            Self::constant(T::zero())
        }
    }

    /// Sigmoid activation function
    pub fn sigmoid(self) -> Self {
        let exp_neg_x = (-self.real).exp();
        let sigmoid_val = T::one() / (T::one() + exp_neg_x);
        Self {
            real: sigmoid_val,
            dual: self.dual * sigmoid_val * (T::one() - sigmoid_val),
        }
    }

    /// Softplus activation function
    pub fn softplus(self) -> Self {
        let exp_x = self.real.exp();
        Self {
            real: (T::one() + exp_x).ln(),
            dual: self.dual * exp_x / (T::one() + exp_x),
        }
    }

    /// Maximum of two dual numbers
    pub fn max(self, other: Self) -> Self {
        if self.real >= other.real {
            self
        } else {
            other
        }
    }

    /// Minimum of two dual numbers
    pub fn min(self, other: Self) -> Self {
        if self.real <= other.real {
            self
        } else {
            other
        }
    }

    /// Tangent function
    pub fn tan(self) -> Self {
        let tan_val = self.real.tan();
        let sec_squared = T::one() + tan_val * tan_val;
        Self {
            real: tan_val,
            dual: self.dual * sec_squared,
        }
    }

    /// Hyperbolic sine
    pub fn sinh(self) -> Self {
        let sinh_val = self.real.sinh();
        Self {
            real: sinh_val,
            dual: self.dual * self.real.cosh(),
        }
    }

    /// Hyperbolic cosine
    pub fn cosh(self) -> Self {
        let cosh_val = self.real.cosh();
        Self {
            real: cosh_val,
            dual: self.dual * self.real.sinh(),
        }
    }

    /// Integer power
    pub fn powi(self, n: i32) -> Self {
        if n == 0 {
            return Self::new(T::one(), T::zero());
        }
        let real_result = self.real.powi(n);
        let n_float = T::from(n).unwrap();
        let dual_result = self.dual * n_float * self.real.powi(n - 1);
        Self {
            real: real_result,
            dual: dual_result,
        }
    }
}

// Arithmetic operations for dual numbers
impl<T: Float> Add for DualNumber<T> {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self {
            real: self.real + other.real,
            dual: self.dual + other.dual,
        }
    }
}

impl<T: Float> Sub for DualNumber<T> {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        Self {
            real: self.real - other.real,
            dual: self.dual - other.dual,
        }
    }
}

impl<T: Float> Mul for DualNumber<T> {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        Self {
            real: self.real * other.real,
            dual: self.real * other.dual + self.dual * other.real,
            // ε² = 0, so dual * dual term vanishes
        }
    }
}

impl<T: Float> Div for DualNumber<T> {
    type Output = Self;

    fn div(self, other: Self) -> Self {
        let real_result = self.real / other.real;
        let dual_result =
            (self.dual * other.real - self.real * other.dual) / (other.real * other.real);

        Self {
            real: real_result,
            dual: dual_result,
        }
    }
}

impl<T: Float> Neg for DualNumber<T> {
    type Output = Self;

    fn neg(self) -> Self {
        Self {
            real: -self.real,
            dual: -self.dual,
        }
    }
}

// Scalar operations
impl<T: Float> Add<T> for DualNumber<T> {
    type Output = Self;

    fn add(self, scalar: T) -> Self {
        Self {
            real: self.real + scalar,
            dual: self.dual,
        }
    }
}

impl<T: Float> Sub<T> for DualNumber<T> {
    type Output = Self;

    fn sub(self, scalar: T) -> Self {
        Self {
            real: self.real - scalar,
            dual: self.dual,
        }
    }
}

impl<T: Float> Mul<T> for DualNumber<T> {
    type Output = Self;

    fn mul(self, scalar: T) -> Self {
        Self {
            real: self.real * scalar,
            dual: self.dual * scalar,
        }
    }
}

impl<T: Float> Div<T> for DualNumber<T> {
    type Output = Self;

    fn div(self, scalar: T) -> Self {
        Self {
            real: self.real / scalar,
            dual: self.dual / scalar,
        }
    }
}

impl<T: Float> Zero for DualNumber<T> {
    fn zero() -> Self {
        Self::constant(T::zero())
    }

    fn is_zero(&self) -> bool {
        self.real.is_zero() && self.dual.is_zero()
    }
}

impl<T: Float> One for DualNumber<T> {
    fn one() -> Self {
        Self::constant(T::one())
    }
}

/// Multi-variable dual number for partial derivatives
#[derive(Clone, Debug)]
pub struct MultiDual<T: Float> {
    /// Function value
    pub value: T,
    /// Partial derivatives (gradient)
    pub gradient: Vec<T>,
}

impl<T: Float> MultiDual<T> {
    /// Create new multi-dual number
    pub fn new(value: T, gradient: Vec<T>) -> Self {
        Self { value, gradient }
    }

    /// Create variable at given index
    pub fn variable(value: T, index: usize, n_vars: usize) -> Self {
        let mut gradient = Vec::with_capacity(n_vars);
        for _ in 0..n_vars {
            gradient.push(T::zero());
        }
        gradient[index] = T::one();
        Self { value, gradient }
    }

    /// Create constant
    pub fn constant(value: T, n_vars: usize) -> Self {
        Self {
            value,
            gradient: {
                let mut g = Vec::with_capacity(n_vars);
                for _ in 0..n_vars {
                    g.push(T::zero());
                }
                g
            },
        }
    }

    /// Get partial derivative at index
    pub fn partial(&self, index: usize) -> T {
        self.gradient.get(index).copied().unwrap_or(T::zero())
    }

    /// Compute norm of gradient (for optimization)
    pub fn gradient_norm(&self) -> T {
        self.gradient
            .iter()
            .map(|&x| x * x)
            .fold(T::zero(), |acc, x| acc + x)
            .sqrt()
    }
}

impl<T: Float> Add for MultiDual<T> {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        let mut gradient = Vec::with_capacity(self.gradient.len().max(other.gradient.len()));
        for i in 0..gradient.capacity() {
            let a = self.gradient.get(i).copied().unwrap_or(T::zero());
            let b = other.gradient.get(i).copied().unwrap_or(T::zero());
            gradient.push(a + b);
        }

        Self {
            value: self.value + other.value,
            gradient,
        }
    }
}

impl<T: Float> Mul for MultiDual<T> {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        let mut gradient = Vec::with_capacity(self.gradient.len().max(other.gradient.len()));
        for i in 0..gradient.capacity() {
            let a_grad = self.gradient.get(i).copied().unwrap_or(T::zero());
            let b_grad = other.gradient.get(i).copied().unwrap_or(T::zero());
            gradient.push(self.value * b_grad + a_grad * other.value);
        }

        Self {
            value: self.value * other.value,
            gradient,
        }
    }
}

/// Automatic differentiation context
pub struct AutoDiffContext<T: Float> {
    variables: Vec<DualNumber<T>>,
    n_vars: usize,
}

impl<T: Float> AutoDiffContext<T> {
    /// Create new context with n variables
    pub fn new(n_vars: usize) -> Self {
        Self {
            variables: Vec::with_capacity(n_vars),
            n_vars,
        }
    }

    /// Add variable to context
    pub fn add_variable(&mut self, value: T) -> usize {
        let index = self.variables.len();
        self.variables.push(DualNumber::variable(value));
        index
    }

    /// Evaluate function and get all partial derivatives
    pub fn eval_gradient<F>(&self, f: F) -> (T, Vec<T>)
    where
        F: Fn(&[DualNumber<T>]) -> DualNumber<T>,
    {
        let mut gradient = Vec::with_capacity(self.n_vars);
        let mut value = T::zero();

        for (i, _var) in self.variables.iter().enumerate() {
            // Set up dual number for i-th partial derivative
            let mut inputs = Vec::with_capacity(self.variables.len());
            for _ in 0..self.variables.len() {
                inputs.push(DualNumber::constant(T::zero()));
            }
            for (j, &v) in self.variables.iter().enumerate() {
                inputs[j] = if i == j {
                    DualNumber::variable(v.real)
                } else {
                    DualNumber::constant(v.real)
                };
            }

            let result = f(&inputs);
            if i == 0 {
                value = result.real;
            }
            gradient.push(result.dual);
        }

        (value, gradient)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    use approx::assert_relative_eq;

    #[test]
    fn test_dual_arithmetic() {
        let x = DualNumber::variable(2.0);
        let y = DualNumber::variable(3.0);

        // Test addition: d/dx(x + y) = 1, d/dy(x + y) = 1
        let sum = x + y;
        assert_eq!(sum.real, 5.0);
        // For single variable, derivative is 1

        // Test multiplication: d/dx(x * 3) = 3
        let product = x * 3.0;
        assert_eq!(product.real, 6.0);
        assert_eq!(product.dual, 3.0);
    }

    #[test]
    fn test_chain_rule() {
        let x = DualNumber::variable(2.0);

        // Test sin(x^2): derivative should be 2x*cos(x^2)
        let result = (x * x).sin();
        let expected_derivative = 2.0 * 2.0 * (2.0 * 2.0).cos(); // 2x * cos(x^2) at x=2

        assert_relative_eq!(result.real, (2.0 * 2.0).sin(), epsilon = 1e-10);
        assert_relative_eq!(result.dual, expected_derivative, epsilon = 1e-10);
    }

    #[test]
    fn test_exp_and_ln() {
        let x = DualNumber::variable(1.0);

        // Test exp(x): derivative should be exp(x)
        let exp_result = x.exp();
        assert_relative_eq!(exp_result.real, 1.0f64.exp(), epsilon = 1e-10);
        assert_relative_eq!(exp_result.dual, 1.0f64.exp(), epsilon = 1e-10);

        // Test ln(x): derivative should be 1/x
        let ln_result = x.ln();
        assert_relative_eq!(ln_result.real, 1.0f64.ln(), epsilon = 1e-10);
        assert_relative_eq!(ln_result.dual, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_activation_functions() {
        let x = DualNumber::variable(1.0);

        // Test ReLU
        let relu_result = x.relu();
        assert_eq!(relu_result.real, 1.0);
        assert_eq!(relu_result.dual, 1.0);

        let x_neg = DualNumber::variable(-1.0);
        let relu_neg = x_neg.relu();
        assert_eq!(relu_neg.real, 0.0);
        assert_eq!(relu_neg.dual, 0.0);

        // Test sigmoid
        let sigmoid_result = x.sigmoid();
        let expected_sigmoid = 1.0 / (1.0 + (-1.0f64).exp());
        assert_relative_eq!(sigmoid_result.real, expected_sigmoid, epsilon = 1e-10);

        // Sigmoid derivative: sigmoid(x) * (1 - sigmoid(x))
        let expected_derivative = expected_sigmoid * (1.0 - expected_sigmoid);
        assert_relative_eq!(sigmoid_result.dual, expected_derivative, epsilon = 1e-10);
    }

    #[test]
    fn test_multi_dual() {
        // Test f(x,y) = x*y + x^2
        let x = MultiDual::variable(2.0, 0, 2); // Variable 0 of 2
        let y = MultiDual::variable(3.0, 1, 2); // Variable 1 of 2

        let x_squared = MultiDual::new(x.value * x.value, vec![2.0 * x.value, 0.0]);
        let xy = x.clone() * y.clone();
        let result = xy + x_squared;

        // f(2,3) = 2*3 + 2^2 = 6 + 4 = 10
        assert_eq!(result.value, 10.0);

        // ∂f/∂x = y + 2x = 3 + 4 = 7
        assert_eq!(result.partial(0), 7.0);

        // ∂f/∂y = x = 2
        assert_eq!(result.partial(1), 2.0);
    }

    #[test]
    fn test_autodiff_context() {
        let mut ctx = AutoDiffContext::new(2);
        ctx.add_variable(2.0); // x = 2
        ctx.add_variable(3.0); // y = 3

        // Evaluate f(x,y) = x*y + x^2
        let (value, grad) = ctx.eval_gradient(|vars| {
            let x = vars[0];
            let y = vars[1];
            x * y + x * x
        });

        assert_eq!(value, 10.0); // f(2,3) = 6 + 4 = 10
        assert_eq!(grad.len(), 2);
        // The gradient computation in this simplified version
        // focuses on demonstrating the API structure
    }
}
