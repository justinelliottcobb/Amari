//! Verified dual number automatic differentiation with phantom types
//!
//! This module provides type-safe dual numbers with compile-time guarantees
//! about differentiation properties and dimensional consistency.

use crate::DualNumber;
use core::marker::PhantomData;
use num_traits::Float;

/// Phantom type for tracking variable dependencies
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Variable<const ID: usize>;

/// Phantom type for constant values
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Constant;

/// Phantom type for multi-variable differentiation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MultiVariable<const N: usize>;

/// Type-safe dual number with phantom type tracking
#[derive(Clone, Copy, Debug)]
pub struct VerifiedDual<T: Float, V = Variable<0>> {
    /// The dual number value
    pub value: DualNumber<T>,
    /// Phantom type for variable tracking
    _variable: PhantomData<V>,
}

impl<T: Float, V> VerifiedDual<T, V> {
    /// Get the real part
    pub fn real(&self) -> T {
        self.value.real
    }

    /// Get the dual part (derivative)
    pub fn dual(&self) -> T {
        self.value.dual
    }
}

impl<T: Float> VerifiedDual<T, Constant> {
    /// Create a verified constant (derivative = 0)
    pub fn constant(value: T) -> Self {
        Self {
            value: DualNumber::constant(value),
            _variable: PhantomData,
        }
    }
}

impl<T: Float, const ID: usize> VerifiedDual<T, Variable<ID>> {
    /// Create a verified variable with specific ID
    pub fn variable(value: T) -> Self {
        Self {
            value: DualNumber::variable(value),
            _variable: PhantomData,
        }
    }
}

/// Type-safe operations that preserve variable tracking
impl<T: Float, V> VerifiedDual<T, V> {
    /// Addition preserves variable type (use + operator instead)
    #[allow(clippy::should_implement_trait)]
    pub fn add<W>(self, other: VerifiedDual<T, W>) -> VerifiedDual<T, V>
    where
        V: VariableCompatible<W>,
    {
        VerifiedDual {
            value: self.value + other.value,
            _variable: PhantomData,
        }
    }

    /// Multiplication preserves variable type (use * operator instead)
    #[allow(clippy::should_implement_trait)]
    pub fn mul<W>(self, other: VerifiedDual<T, W>) -> VerifiedDual<T, V>
    where
        V: VariableCompatible<W>,
    {
        VerifiedDual {
            value: self.value * other.value,
            _variable: PhantomData,
        }
    }

    /// Transcendental functions preserve variable type
    pub fn exp(self) -> Self {
        VerifiedDual {
            value: self.value.exp(),
            _variable: PhantomData,
        }
    }

    pub fn ln(self) -> Self {
        VerifiedDual {
            value: self.value.ln(),
            _variable: PhantomData,
        }
    }

    pub fn sin(self) -> Self {
        VerifiedDual {
            value: self.value.sin(),
            _variable: PhantomData,
        }
    }

    pub fn cos(self) -> Self {
        VerifiedDual {
            value: self.value.cos(),
            _variable: PhantomData,
        }
    }
}

/// Trait for variable compatibility in operations
pub trait VariableCompatible<W> {}

// Constants are compatible with everything
impl<W> VariableCompatible<W> for Constant {}

// Variables are compatible with constants
impl<const ID: usize> VariableCompatible<Constant> for Variable<ID> {}

// Same variables are compatible
impl<const ID: usize> VariableCompatible<Variable<ID>> for Variable<ID> {}

// Multi-variables are compatible with constants
impl<const N: usize> VariableCompatible<Constant> for MultiVariable<N> {}

/// Verified multi-dual number for gradient computation
#[derive(Clone, Debug)]
pub struct VerifiedMultiDual<T: Float, const N: usize> {
    /// Real value
    pub real: T,
    /// Gradient vector (partial derivatives)
    pub gradient: [T; N],
    /// Phantom type for compile-time dimension checking
    _phantom: PhantomData<MultiVariable<N>>,
}

impl<T: Float, const N: usize> VerifiedMultiDual<T, N> {
    /// Create a new multi-dual number with specified gradient
    pub fn new(real: T, gradient: [T; N]) -> Self {
        Self {
            real,
            gradient,
            _phantom: PhantomData,
        }
    }

    /// Create a variable at specific index
    pub fn variable(value: T, index: usize) -> Self {
        assert!(index < N, "Variable index out of bounds");
        let mut gradient = [T::zero(); N];
        gradient[index] = T::one();
        Self::new(value, gradient)
    }

    /// Create a constant (zero gradient)
    pub fn constant(value: T) -> Self {
        Self::new(value, [T::zero(); N])
    }

    /// Get partial derivative with respect to variable i
    pub fn partial(&self, i: usize) -> T {
        assert!(i < N, "Index out of bounds");
        self.gradient[i]
    }

    /// Compute the gradient norm
    pub fn gradient_norm(&self) -> T {
        self.gradient
            .iter()
            .map(|&g| g * g)
            .fold(T::zero(), |acc, x| acc + x)
            .sqrt()
    }
}

/// Operations for verified multi-dual numbers
impl<T: Float, const N: usize> VerifiedMultiDual<T, N> {
    /// Addition with dimension checking at compile time
    pub fn add(&self, other: &Self) -> Self {
        let mut gradient = [T::zero(); N];
        for (i, item) in gradient.iter_mut().enumerate().take(N) {
            *item = self.gradient[i] + other.gradient[i];
        }
        Self::new(self.real + other.real, gradient)
    }

    /// Multiplication with automatic differentiation
    pub fn mul(&self, other: &Self) -> Self {
        let mut gradient = [T::zero(); N];
        for (i, item) in gradient.iter_mut().enumerate().take(N) {
            *item = self.gradient[i] * other.real + self.real * other.gradient[i];
        }
        Self::new(self.real * other.real, gradient)
    }

    /// Division with automatic differentiation
    pub fn div(&self, other: &Self) -> Self {
        let mut gradient = [T::zero(); N];
        let divisor_squared = other.real * other.real;
        for (i, item) in gradient.iter_mut().enumerate().take(N) {
            *item =
                (self.gradient[i] * other.real - self.real * other.gradient[i]) / divisor_squared;
        }
        Self::new(self.real / other.real, gradient)
    }
}

/// Differentiation mode phantom types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ForwardMode;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReverseMode;

/// Type-safe differentiation context
pub struct DifferentiationContext<T: Float, Mode> {
    _phantom_t: PhantomData<T>,
    _phantom_mode: PhantomData<Mode>,
}

impl<T: Float> DifferentiationContext<T, ForwardMode> {
    /// Create a forward-mode differentiation context
    pub fn forward() -> Self {
        Self {
            _phantom_t: PhantomData,
            _phantom_mode: PhantomData,
        }
    }

    /// Differentiate a univariate function
    pub fn differentiate<F>(&self, f: F, x: T) -> (T, T)
    where
        F: Fn(DualNumber<T>) -> DualNumber<T>,
    {
        let input = DualNumber::variable(x);
        let output = f(input);
        (output.real, output.dual)
    }

    /// Compute gradient of a multivariate function
    pub fn gradient<F, const N: usize>(&self, f: F, x: [T; N]) -> (T, [T; N])
    where
        F: Fn(&[VerifiedMultiDual<T, N>]) -> VerifiedMultiDual<T, N>,
    {
        let mut inputs = Vec::with_capacity(N);
        for (i, &item) in x.iter().enumerate().take(N) {
            inputs.push(VerifiedMultiDual::variable(item, i));
        }
        let output = f(&inputs);
        (output.real, output.gradient)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_verified_dual_creation() {
        let x: VerifiedDual<f64, Variable<0>> = VerifiedDual::variable(2.0);
        assert_eq!(x.real(), 2.0);
        assert_eq!(x.dual(), 1.0);

        let c: VerifiedDual<f64, Constant> = VerifiedDual::constant(3.0);
        assert_eq!(c.real(), 3.0);
        assert_eq!(c.dual(), 0.0);
    }

    #[test]
    fn test_verified_operations() {
        let x = VerifiedDual::<f64, Variable<0>>::variable(2.0);
        let c = VerifiedDual::<f64, Constant>::constant(3.0);

        let sum = x.add(c);
        assert_eq!(sum.real(), 5.0);
        assert_eq!(sum.dual(), 1.0);

        let product = x.mul(c);
        assert_eq!(product.real(), 6.0);
        assert_eq!(product.dual(), 3.0);
    }

    #[test]
    fn test_verified_multi_dual() {
        let x = VerifiedMultiDual::<f64, 3>::variable(2.0, 0);
        let y = VerifiedMultiDual::<f64, 3>::variable(3.0, 1);

        let product = x.mul(&y);
        assert_eq!(product.real, 6.0);
        assert_eq!(product.partial(0), 3.0); // ∂(xy)/∂x = y
        assert_eq!(product.partial(1), 2.0); // ∂(xy)/∂y = x
        assert_eq!(product.partial(2), 0.0); // ∂(xy)/∂z = 0
    }

    #[test]
    fn test_differentiation_context() {
        let ctx = DifferentiationContext::<f64, ForwardMode>::forward();

        // Differentiate f(x) = x² + 2x + 1
        let (value, derivative) = ctx.differentiate(
            |x| {
                let x_squared = x * x;
                let two_x = x * 2.0;
                x_squared + two_x + DualNumber::constant(1.0)
            },
            3.0,
        );

        assert_eq!(value, 16.0); // 3² + 2*3 + 1
        assert_eq!(derivative, 8.0); // 2*3 + 2
    }

    #[test]
    fn test_gradient_computation() {
        let ctx = DifferentiationContext::<f64, ForwardMode>::forward();

        // Gradient of f(x,y,z) = x²y + yz
        let (value, grad) = ctx.gradient(
            |vars| {
                let x = &vars[0];
                let y = &vars[1];
                let z = &vars[2];

                let x_squared = x.mul(x);
                let x_squared_y = x_squared.mul(y);
                let y_z = y.mul(z);
                x_squared_y.add(&y_z)
            },
            [2.0, 3.0, 4.0],
        );

        assert_eq!(value, 24.0); // 4*3 + 3*4
        assert_eq!(grad[0], 12.0); // 2xy = 2*2*3
        assert_eq!(grad[1], 8.0); // x² + z = 4 + 4
        assert_eq!(grad[2], 3.0); // y = 3
    }
}
