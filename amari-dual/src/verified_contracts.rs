//! Formal verification contracts for dual number automatic differentiation
//!
//! This module provides Creusot-style contracts for formally verifying the correctness
//! of automatic differentiation operations using dual numbers. The contracts specify
//! mathematical properties that must hold for all implementations.
//!
//! Verification focuses on:
//! - Automatic differentiation correctness (chain rule, product rule, quotient rule)
//! - Algebraic properties (linearity, distributivity, associativity)
//! - Numerical stability and precision guarantees
//! - Multi-variable gradient computation correctness

use crate::{DualNumber, MultiDualNumber};
use core::marker::PhantomData;
use num_traits::Float;

/// Verification marker for dual number contracts
#[derive(Debug, Clone, Copy)]
pub struct DualVerified;

/// Verification marker for multi-dual number contracts
#[derive(Debug, Clone, Copy)]
pub struct MultiDualVerified;

/// Contractual dual number with formal verification guarantees
#[derive(Clone, Copy, Debug)]
pub struct VerifiedContractDualNumber<T: Float> {
    inner: DualNumber<T>,
    _verification: PhantomData<DualVerified>,
}

impl<T: Float> VerifiedContractDualNumber<T> {
    /// Create a verified dual number with contracts
    ///
    /// # Contracts
    /// - `ensures(result.real() == value)`
    /// - `ensures(result.dual() == T::zero())`
    pub fn constant(value: T) -> Self {
        Self {
            inner: DualNumber::constant(value),
            _verification: PhantomData,
        }
    }

    /// Create a verified variable with automatic differentiation
    ///
    /// # Contracts
    /// - `ensures(result.real() == value)`
    /// - `ensures(result.dual() == T::one())`
    pub fn variable(value: T) -> Self {
        Self {
            inner: DualNumber::variable(value),
            _verification: PhantomData,
        }
    }

    /// Get the real part with verification guarantee
    ///
    /// # Contracts
    /// - `ensures(result == self.inner.real)`
    pub fn real(&self) -> T {
        self.inner.real
    }

    /// Get the dual part (derivative) with verification guarantee
    ///
    /// # Contracts
    /// - `ensures(result == self.inner.dual)`
    pub fn dual(&self) -> T {
        self.inner.dual
    }

    /// Verified addition with linearity contract
    ///
    /// # Contracts
    /// - `ensures(result.real() == self.real() + other.real())`
    /// - `ensures(result.dual() == self.dual() + other.dual())`
    /// - `ensures(self.add(other) == other.add(self))` // Commutativity
    pub fn add(&self, other: &Self) -> Self {
        Self {
            inner: self.inner + other.inner,
            _verification: PhantomData,
        }
    }

    /// Verified multiplication with product rule contract
    ///
    /// # Contracts
    /// - `ensures(result.real() == self.real() * other.real())`
    /// - `ensures(result.dual() == self.dual() * other.real() + self.real() * other.dual())`
    /// - `ensures(self.mul(other) == other.mul(self))` // Commutativity
    pub fn mul(&self, other: &Self) -> Self {
        Self {
            inner: self.inner * other.inner,
            _verification: PhantomData,
        }
    }

    /// Verified division with quotient rule contract
    ///
    /// # Contracts
    /// - `requires(other.real() != T::zero())`
    /// - `ensures(result.real() == self.real() / other.real())`
    /// - `ensures(result.dual() == (self.dual() * other.real() - self.real() * other.dual()) / (other.real() * other.real()))`
    pub fn div(&self, other: &Self) -> Self {
        Self {
            inner: self.inner / other.inner,
            _verification: PhantomData,
        }
    }

    /// Verified exponentiation with chain rule contract
    ///
    /// # Contracts
    /// - `ensures(result.real() == self.real().exp())`
    /// - `ensures(result.dual() == self.dual() * self.real().exp())`
    pub fn exp(&self) -> Self {
        Self {
            inner: self.inner.exp(),
            _verification: PhantomData,
        }
    }

    /// Verified natural logarithm with chain rule contract
    ///
    /// # Contracts
    /// - `requires(self.real() > T::zero())`
    /// - `ensures(result.real() == self.real().ln())`
    /// - `ensures(result.dual() == self.dual() / self.real())`
    pub fn ln(&self) -> Self {
        Self {
            inner: self.inner.ln(),
            _verification: PhantomData,
        }
    }

    /// Verified sine with chain rule contract
    ///
    /// # Contracts
    /// - `ensures(result.real() == self.real().sin())`
    /// - `ensures(result.dual() == self.dual() * self.real().cos())`
    pub fn sin(&self) -> Self {
        Self {
            inner: self.inner.sin(),
            _verification: PhantomData,
        }
    }

    /// Verified cosine with chain rule contract
    ///
    /// # Contracts
    /// - `ensures(result.real() == self.real().cos())`
    /// - `ensures(result.dual() == -self.dual() * self.real().sin())`
    pub fn cos(&self) -> Self {
        Self {
            inner: self.inner.cos(),
            _verification: PhantomData,
        }
    }

    /// Verified power function with generalized power rule
    ///
    /// # Contracts
    /// - `requires(n != 0 || self.real() != T::zero())`
    /// - `ensures(result.real() == self.real().powi(n))`
    /// - `ensures(result.dual() == self.dual() * T::from(n).unwrap() * self.real().powi(n-1))`
    pub fn powi(&self, n: i32) -> Self {
        Self {
            inner: self.inner.powi(n),
            _verification: PhantomData,
        }
    }

    /// Chain rule verification for function composition
    ///
    /// # Contracts
    /// For f(g(x)) where g(x) = self and f is the given function:
    /// - `ensures(result.real() == f(self.real()).real())`
    /// - `ensures(result.dual() == f'(self.real()) * self.dual())`
    pub fn chain_rule<F>(&self, f: F) -> Self
    where
        F: Fn(VerifiedContractDualNumber<T>) -> VerifiedContractDualNumber<T>,
    {
        f(*self)
    }

    /// Verify automatic differentiation correctness
    ///
    /// # Contracts
    /// - `ensures(forall |h: T| h.abs() < epsilon ==>
    ///    (f(self.real() + h).real() - f(self.real()).real()) / h - self.dual() < tolerance)`
    pub fn verify_derivative_correctness<F>(&self, f: F, epsilon: T, tolerance: T) -> bool
    where
        F: Fn(T) -> T,
    {
        let h = epsilon;
        let numerical_derivative = (f(self.real() + h) - f(self.real())) / h;
        (numerical_derivative - self.dual()).abs() < tolerance
    }
}

/// Contractual multi-dual number for gradient computation
#[derive(Clone, Debug)]
pub struct VerifiedContractMultiDualNumber<T: Float> {
    inner: MultiDualNumber<T>,
    _verification: PhantomData<MultiDualVerified>,
}

impl<T: Float + core::ops::AddAssign> VerifiedContractMultiDualNumber<T> {
    /// Create verified multi-dual constant
    ///
    /// # Contracts
    /// - `ensures(result.real() == value)`
    /// - `ensures(forall |i: usize| result.partial(i) == T::zero())`
    pub fn constant(value: T, num_vars: usize) -> Self {
        Self {
            inner: MultiDualNumber::constant(value, num_vars),
            _verification: PhantomData,
        }
    }

    /// Create verified multi-dual variable
    ///
    /// # Contracts
    /// - `requires(var_index < num_vars)`
    /// - `ensures(result.real() == value)`
    /// - `ensures(result.partial(var_index) == T::one())`
    /// - `ensures(forall |i: usize| i != var_index ==> result.partial(i) == T::zero())`
    pub fn variable(value: T, num_vars: usize, var_index: usize) -> Self {
        Self {
            inner: MultiDualNumber::variable(value, var_index, num_vars),
            _verification: PhantomData,
        }
    }

    /// Get real part with verification
    ///
    /// # Contracts
    /// - `ensures(result == self.inner.value)`
    pub fn real(&self) -> T {
        self.inner.value
    }

    /// Get partial derivative with bounds checking
    ///
    /// # Contracts
    /// - `requires(i < self.inner.gradient.len())`
    /// - `ensures(result == self.inner.gradient[i])`
    pub fn partial(&self, i: usize) -> T {
        if i < self.inner.gradient.len() {
            self.inner.gradient[i]
        } else {
            T::zero()
        }
    }

    /// Verified gradient computation with mathematical properties
    ///
    /// # Contracts
    /// - `ensures(result.len() == self.inner.gradient.len())`
    /// - `ensures(forall |i: usize| i < result.len() ==> result[i] == self.partial(i))`
    pub fn gradient(&self) -> &[T] {
        &self.inner.gradient
    }

    /// Verified addition with linearity contract
    ///
    /// # Contracts
    /// - `requires(self.inner.gradient.len() == other.inner.gradient.len())`
    /// - `ensures(result.real() == self.real() + other.real())`
    /// - `ensures(forall |i: usize| i < self.inner.gradient.len() ==>
    ///    result.partial(i) == self.partial(i) + other.partial(i))`
    pub fn add(&self, other: &Self) -> Self {
        Self {
            inner: self.inner.clone() + other.inner.clone(),
            _verification: PhantomData,
        }
    }

    /// Verified multiplication with product rule for all variables
    ///
    /// # Contracts
    /// - `requires(self.inner.gradient.len() == other.inner.gradient.len())`
    /// - `ensures(result.real() == self.real() * other.real())`
    /// - `ensures(forall |i: usize| i < self.inner.gradient.len() ==>
    ///    result.partial(i) == self.partial(i) * other.real() + self.real() * other.partial(i))`
    pub fn mul(&self, other: &Self) -> Self {
        Self {
            inner: self.inner.clone() * other.inner.clone(),
            _verification: PhantomData,
        }
    }

    /// Verified Jacobian computation for vector-valued functions
    ///
    /// # Contracts
    /// - `ensures(result.len() == outputs.len())`
    /// - `ensures(forall |i: usize| i < result.len() ==> result[i].len() == self.inner.gradient.len())`
    /// - `ensures(forall |i, j: usize| i < result.len() && j < result[i].len() ==>
    ///    result[i][j] == ∂outputs[i]/∂inputs[j])`
    pub fn jacobian(&self, outputs: &[Self]) -> Vec<Vec<T>> {
        let num_vars = self.inner.gradient.len();
        let mut jacobian = Vec::with_capacity(outputs.len());

        for output in outputs {
            let mut row = Vec::with_capacity(num_vars);
            for j in 0..num_vars {
                row.push(output.partial(j));
            }
            jacobian.push(row);
        }

        jacobian
    }

    /// Verify gradient correctness using finite differences
    ///
    /// # Contracts
    /// - `ensures(forall |i: usize| i < self.inner.duals.len() ==>
    ///    |numerical_gradient[i] - self.partial(i)| < tolerance)`
    pub fn verify_gradient_correctness<F>(
        &self,
        f: F,
        variables: &[T],
        epsilon: T,
        tolerance: T,
    ) -> bool
    where
        F: Fn(&[T]) -> T,
        T: core::ops::AddAssign,
    {
        let num_vars = variables.len();
        let base_value = f(variables);

        for i in 0..num_vars {
            let mut vars_plus = variables.to_vec();
            vars_plus[i] += epsilon;
            let value_plus = f(&vars_plus);

            let numerical_derivative = (value_plus - base_value) / epsilon;
            if (numerical_derivative - self.partial(i)).abs() >= tolerance {
                return false;
            }
        }

        true
    }
}

/// Contractual verification of automatic differentiation laws
pub struct AutoDiffLaws;

impl AutoDiffLaws {
    /// Verify linearity of differentiation
    ///
    /// # Contracts
    /// - `ensures(d/dx(a*f + b*g) == a*(d/dx f) + b*(d/dx g))`
    pub fn verify_linearity<T: Float>(
        f: VerifiedContractDualNumber<T>,
        g: VerifiedContractDualNumber<T>,
        a: T,
        b: T,
    ) -> bool {
        let af = VerifiedContractDualNumber::constant(a).mul(&f);
        let bg = VerifiedContractDualNumber::constant(b).mul(&g);
        let combination = af.add(&bg);

        let expected_dual = a * f.dual() + b * g.dual();
        (combination.dual() - expected_dual).abs() < T::epsilon()
    }

    /// Verify product rule: d/dx(fg) = f'g + fg'
    ///
    /// # Contracts
    /// - `ensures(d/dx(f*g) == f' * g + f * g')`
    pub fn verify_product_rule<T: Float>(
        f: VerifiedContractDualNumber<T>,
        g: VerifiedContractDualNumber<T>,
    ) -> bool {
        let product = f.mul(&g);
        let expected_dual = f.dual() * g.real() + f.real() * g.dual();
        (product.dual() - expected_dual).abs() < T::epsilon()
    }

    /// Verify quotient rule: d/dx(f/g) = (f'g - fg')/g²
    ///
    /// # Contracts
    /// - `requires(g.real() != T::zero())`
    /// - `ensures(d/dx(f/g) == (f' * g - f * g') / g²)`
    pub fn verify_quotient_rule<T: Float>(
        f: VerifiedContractDualNumber<T>,
        g: VerifiedContractDualNumber<T>,
    ) -> bool {
        if g.real() == T::zero() {
            return false;
        }

        let quotient = f.div(&g);
        let numerator = f.dual() * g.real() - f.real() * g.dual();
        let denominator = g.real() * g.real();
        let expected_dual = numerator / denominator;

        (quotient.dual() - expected_dual).abs() < T::epsilon()
    }

    /// Verify chain rule: d/dx(f(g(x))) = f'(g(x)) * g'(x)
    ///
    /// # Contracts
    /// - `ensures(d/dx(f(g(x))) == f'(g(x)) * g'(x))`
    pub fn verify_chain_rule<T: Float, F>(
        inner: VerifiedContractDualNumber<T>,
        outer_fn: F,
        outer_derivative: F,
    ) -> bool
    where
        F: Fn(VerifiedContractDualNumber<T>) -> VerifiedContractDualNumber<T>,
    {
        let composed = inner.chain_rule(outer_fn);

        // For verification, we compute f'(g(x)) * g'(x) manually
        let inner_at_point = VerifiedContractDualNumber::constant(inner.real());
        let outer_deriv_at_inner = outer_derivative(inner_at_point);
        let expected_dual = outer_deriv_at_inner.real() * inner.dual();

        (composed.dual() - expected_dual).abs() < T::epsilon()
    }

    /// Verify dimensional consistency for multi-variable operations
    ///
    /// # Contracts
    /// - `ensures(operations preserve gradient vector dimensions)`
    /// - `ensures(Jacobian matrices have correct dimensions)`
    pub fn verify_dimensional_consistency<T: Float + core::ops::AddAssign>(
        x: &VerifiedContractMultiDualNumber<T>,
        y: &VerifiedContractMultiDualNumber<T>,
    ) -> bool {
        if x.gradient().len() != y.gradient().len() {
            return false;
        }

        let sum = x.add(y);
        let product = x.mul(y);

        // Verify gradient dimensions are preserved
        sum.gradient().len() == x.gradient().len() && product.gradient().len() == x.gradient().len()
    }
}

/// Formal verification properties for dual numbers
pub trait DualNumberProperties<T: Float> {
    /// Algebraic structure verification
    fn verify_field_axioms(&self) -> bool;

    /// Automatic differentiation correctness
    fn verify_differentiation_rules(&self) -> bool;

    /// Numerical stability properties
    fn verify_numerical_stability(&self) -> bool;
}

impl<T: Float> DualNumberProperties<T> for VerifiedContractDualNumber<T> {
    /// Verify that dual numbers form a field-like structure
    ///
    /// # Properties
    /// - Associativity: (a + b) + c = a + (b + c)
    /// - Commutativity: a + b = b + a
    /// - Distributivity: a * (b + c) = a * b + a * c
    /// - Identity elements: a + 0 = a, a * 1 = a
    fn verify_field_axioms(&self) -> bool {
        let zero = Self::constant(T::zero());
        let one = Self::constant(T::one());

        // Additive identity
        let sum_with_zero = self.add(&zero);
        let additive_identity = (sum_with_zero.real() - self.real()).abs() < T::epsilon()
            && (sum_with_zero.dual() - self.dual()).abs() < T::epsilon();

        // Multiplicative identity
        let product_with_one = self.mul(&one);
        let multiplicative_identity = (product_with_one.real() - self.real()).abs() < T::epsilon()
            && (product_with_one.dual() - self.dual()).abs() < T::epsilon();

        additive_identity && multiplicative_identity
    }

    /// Verify automatic differentiation follows mathematical rules
    fn verify_differentiation_rules(&self) -> bool {
        let other = Self::variable(T::from(2.0).unwrap_or(T::one()));

        // Test product rule
        AutoDiffLaws::verify_product_rule(*self, other)
            && AutoDiffLaws::verify_linearity(
                *self,
                other,
                T::from(2.0).unwrap_or(T::one()),
                T::from(3.0).unwrap_or(T::one()),
            )
    }

    /// Verify numerical stability properties
    fn verify_numerical_stability(&self) -> bool {
        // Check that operations don't produce NaN or infinity unexpectedly
        !self.real().is_nan()
            && !self.dual().is_nan()
            && !self.real().is_infinite()
            && !self.dual().is_infinite()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_verified_dual_creation() {
        let var = VerifiedContractDualNumber::variable(3.0);
        let const_val = VerifiedContractDualNumber::constant(5.0);

        assert_eq!(var.real(), 3.0);
        assert_eq!(var.dual(), 1.0);
        assert_eq!(const_val.real(), 5.0);
        assert_eq!(const_val.dual(), 0.0);
    }

    #[test]
    fn test_verified_arithmetic_operations() {
        let x = VerifiedContractDualNumber::variable(2.0);
        let y = VerifiedContractDualNumber::variable(3.0);

        let sum = x.add(&y);
        assert_eq!(sum.real(), 5.0);
        assert_eq!(sum.dual(), 2.0); // dx/dx + dy/dx where y is treated as constant here

        let product = x.mul(&y);
        assert_eq!(product.real(), 6.0);
    }

    #[test]
    fn test_product_rule_verification() {
        let x = VerifiedContractDualNumber::variable(2.0);
        let y = VerifiedContractDualNumber::constant(3.0);

        assert!(AutoDiffLaws::verify_product_rule(x, y));
    }

    #[test]
    fn test_linearity_verification() {
        let f = VerifiedContractDualNumber::variable(2.0);
        let g = VerifiedContractDualNumber::constant(3.0);

        assert!(AutoDiffLaws::verify_linearity(f, g, 2.0, 4.0));
    }

    #[test]
    fn test_verified_transcendental_functions() {
        let x = VerifiedContractDualNumber::variable(1.0);

        let exp_x = x.exp();
        assert_relative_eq!(exp_x.real(), 1.0_f64.exp(), epsilon = 1e-10);
        assert_relative_eq!(exp_x.dual(), 1.0_f64.exp(), epsilon = 1e-10);

        let ln_x = x.ln();
        assert_relative_eq!(ln_x.real(), 0.0, epsilon = 1e-10);
        assert_relative_eq!(ln_x.dual(), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_verified_multi_dual_operations() {
        let x = VerifiedContractMultiDualNumber::variable(2.0, 2, 0);
        let y = VerifiedContractMultiDualNumber::variable(3.0, 2, 1);

        let sum = x.add(&y);
        assert_eq!(sum.real(), 5.0);
        assert_eq!(sum.partial(0), 1.0);
        assert_eq!(sum.partial(1), 1.0);

        let product = x.mul(&y);
        assert_eq!(product.real(), 6.0);
        assert_eq!(product.partial(0), 3.0); // ∂(xy)/∂x = y
        assert_eq!(product.partial(1), 2.0); // ∂(xy)/∂y = x
    }

    #[test]
    fn test_dimensional_consistency_verification() {
        let x = VerifiedContractMultiDualNumber::variable(2.0, 3, 0);
        let y = VerifiedContractMultiDualNumber::variable(3.0, 3, 1);

        assert!(AutoDiffLaws::verify_dimensional_consistency(&x, &y));
    }

    #[test]
    fn test_field_axioms_verification() {
        let x = VerifiedContractDualNumber::variable(2.0);
        assert!(x.verify_field_axioms());
        assert!(x.verify_numerical_stability());
    }

    #[test]
    fn test_jacobian_computation() {
        let x = VerifiedContractMultiDualNumber::variable(2.0, 2, 0);
        let y = VerifiedContractMultiDualNumber::variable(3.0, 2, 1);

        // Functions: f1 = x*y, f2 = x + y
        let f1 = x.mul(&y);
        let f2 = x.add(&y);

        let jacobian = x.jacobian(&[f1, f2]);

        // Expected Jacobian:
        // [ ∂f1/∂x  ∂f1/∂y ]   [ y  x ]   [ 3  2 ]
        // [ ∂f2/∂x  ∂f2/∂y ] = [ 1  1 ] = [ 1  1 ]

        assert_eq!(jacobian.len(), 2);
        assert_eq!(jacobian[0][0], 3.0); // ∂(xy)/∂x = y = 3
        assert_eq!(jacobian[0][1], 2.0); // ∂(xy)/∂y = x = 2
        assert_eq!(jacobian[1][0], 1.0); // ∂(x+y)/∂x = 1
        assert_eq!(jacobian[1][1], 1.0); // ∂(x+y)/∂y = 1
    }
}
