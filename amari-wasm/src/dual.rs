//! WASM bindings for automatic differentiation with dual numbers

use amari_dual::{DualNumber, MultiDualNumber};
use wasm_bindgen::prelude::*;

/// WASM wrapper for single-variable dual numbers
#[wasm_bindgen]
pub struct WasmDualNumber {
    inner: DualNumber<f64>,
}

#[wasm_bindgen]
impl WasmDualNumber {
    /// Create a new dual number with given real and dual parts
    #[wasm_bindgen(constructor)]
    pub fn new(real: f64, dual: f64) -> Self {
        Self {
            inner: DualNumber::new(real, dual),
        }
    }

    /// Create a variable (derivative = 1)
    #[wasm_bindgen(js_name = variable)]
    pub fn variable(value: f64) -> Self {
        Self {
            inner: DualNumber::variable(value),
        }
    }

    /// Create a constant (derivative = 0)
    #[wasm_bindgen(js_name = constant)]
    pub fn constant(value: f64) -> Self {
        Self {
            inner: DualNumber::constant(value),
        }
    }

    /// Get the real part (function value)
    #[wasm_bindgen(js_name = getReal)]
    pub fn get_real(&self) -> f64 {
        self.inner.real
    }

    /// Get the dual part (derivative)
    #[wasm_bindgen(js_name = getDual)]
    pub fn get_dual(&self) -> f64 {
        self.inner.dual
    }

    /// Addition
    pub fn add(&self, other: &WasmDualNumber) -> WasmDualNumber {
        Self {
            inner: self.inner + other.inner,
        }
    }

    /// Subtraction
    pub fn sub(&self, other: &WasmDualNumber) -> WasmDualNumber {
        Self {
            inner: self.inner - other.inner,
        }
    }

    /// Multiplication
    pub fn mul(&self, other: &WasmDualNumber) -> WasmDualNumber {
        Self {
            inner: self.inner * other.inner,
        }
    }

    /// Division
    pub fn div(&self, other: &WasmDualNumber) -> Result<WasmDualNumber, JsValue> {
        if other.inner.real == 0.0 {
            return Err(JsValue::from_str("Division by zero"));
        }
        Ok(Self {
            inner: self.inner / other.inner,
        })
    }

    /// Negation
    pub fn neg(&self) -> WasmDualNumber {
        Self { inner: -self.inner }
    }

    /// Power function
    pub fn pow(&self, exponent: f64) -> WasmDualNumber {
        Self {
            inner: self.inner.powf(exponent),
        }
    }

    /// Exponential function
    pub fn exp(&self) -> WasmDualNumber {
        Self {
            inner: self.inner.exp(),
        }
    }

    /// Natural logarithm
    pub fn ln(&self) -> Result<WasmDualNumber, JsValue> {
        if self.inner.real <= 0.0 {
            return Err(JsValue::from_str("Logarithm of non-positive number"));
        }
        Ok(Self {
            inner: self.inner.ln(),
        })
    }

    /// Sine function
    pub fn sin(&self) -> WasmDualNumber {
        Self {
            inner: self.inner.sin(),
        }
    }

    /// Cosine function
    pub fn cos(&self) -> WasmDualNumber {
        Self {
            inner: self.inner.cos(),
        }
    }

    /// Square root
    pub fn sqrt(&self) -> Result<WasmDualNumber, JsValue> {
        if self.inner.real < 0.0 {
            return Err(JsValue::from_str("Square root of negative number"));
        }
        Ok(Self {
            inner: self.inner.sqrt(),
        })
    }
}

/// WASM wrapper for multi-variable dual numbers
#[wasm_bindgen]
pub struct WasmMultiDualNumber {
    inner: MultiDualNumber<f64>,
}

#[wasm_bindgen]
impl WasmMultiDualNumber {
    /// Create a new multi-dual number
    #[wasm_bindgen(constructor)]
    pub fn new(real: f64, duals: &[f64]) -> Self {
        Self {
            inner: MultiDualNumber::new(real, duals.to_vec()),
        }
    }

    /// Create a variable with derivative 1 at the specified index
    #[wasm_bindgen(js_name = variable)]
    pub fn variable(value: f64, num_vars: usize, var_index: usize) -> Self {
        Self {
            inner: MultiDualNumber::variable(value, num_vars, var_index),
        }
    }

    /// Create a constant (all derivatives are zero)
    #[wasm_bindgen(js_name = constant)]
    pub fn constant(value: f64, num_vars: usize) -> Self {
        Self {
            inner: MultiDualNumber::constant(value, num_vars),
        }
    }

    /// Get the real part (function value)
    #[wasm_bindgen(js_name = getReal)]
    pub fn get_real(&self) -> f64 {
        self.inner.real
    }

    /// Get the gradient (all partial derivatives)
    #[wasm_bindgen(js_name = getGradient)]
    pub fn get_gradient(&self) -> Vec<f64> {
        self.inner.duals.clone()
    }

    /// Get a specific partial derivative
    #[wasm_bindgen(js_name = getPartial)]
    pub fn get_partial(&self, index: usize) -> Result<f64, JsValue> {
        self.inner
            .duals
            .get(index)
            .copied()
            .ok_or_else(|| JsValue::from_str("Index out of bounds"))
    }

    /// Get number of variables
    #[wasm_bindgen(js_name = getNumVars)]
    pub fn get_num_vars(&self) -> usize {
        self.inner.num_vars()
    }

    /// Addition
    pub fn add(&self, other: &WasmMultiDualNumber) -> Result<WasmMultiDualNumber, JsValue> {
        if self.inner.duals.len() != other.inner.duals.len() {
            return Err(JsValue::from_str("Incompatible number of variables"));
        }
        Ok(Self {
            inner: &self.inner + &other.inner,
        })
    }

    /// Multiplication
    pub fn mul(&self, other: &WasmMultiDualNumber) -> Result<WasmMultiDualNumber, JsValue> {
        if self.inner.duals.len() != other.inner.duals.len() {
            return Err(JsValue::from_str("Incompatible number of variables"));
        }
        Ok(Self {
            inner: &self.inner * &other.inner,
        })
    }

    /// Square root
    pub fn sqrt(&self) -> Result<WasmMultiDualNumber, JsValue> {
        if self.inner.real < 0.0 {
            return Err(JsValue::from_str("Square root of negative number"));
        }
        Ok(Self {
            inner: self.inner.sqrt(),
        })
    }
}

/// Automatic differentiation utilities
#[wasm_bindgen]
pub struct AutoDiff;

#[wasm_bindgen]
impl AutoDiff {
    /// Compute numerical derivative using finite differences (fallback implementation)
    #[wasm_bindgen(js_name = numericalDerivative)]
    pub fn numerical_derivative(x: f64, f: &js_sys::Function, h: f64) -> Result<f64, JsValue> {
        let this = &JsValue::null();

        // f(x + h)
        let args_plus = js_sys::Array::new();
        args_plus.push(&JsValue::from(x + h));
        let f_plus = f
            .apply(this, &args_plus)?
            .as_f64()
            .ok_or_else(|| JsValue::from_str("Function must return a number"))?;

        // f(x - h)
        let args_minus = js_sys::Array::new();
        args_minus.push(&JsValue::from(x - h));
        let f_minus = f
            .apply(this, &args_minus)?
            .as_f64()
            .ok_or_else(|| JsValue::from_str("Function must return a number"))?;

        // Central difference: (f(x+h) - f(x-h)) / (2h)
        Ok((f_plus - f_minus) / (2.0 * h))
    }

    /// Create a dual number and evaluate a polynomial
    #[wasm_bindgen(js_name = evaluatePolynomial)]
    pub fn evaluate_polynomial(x: f64, coefficients: &[f64]) -> WasmDualNumber {
        let dual_x = WasmDualNumber::variable(x);
        let mut result = WasmDualNumber::constant(0.0);

        for (i, &coeff) in coefficients.iter().enumerate() {
            let term = if i == 0 {
                WasmDualNumber::constant(coeff)
            } else {
                let power = dual_x.pow(i as f64);
                power.mul(&WasmDualNumber::constant(coeff))
            };
            result = result.add(&term);
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use wasm_bindgen_test::*;

    #[allow(dead_code)]
    #[wasm_bindgen_test]
    fn test_dual_basic() {
        let x = WasmDualNumber::variable(3.0);
        let y = WasmDualNumber::constant(2.0);

        // Test x + y
        let sum = x.add(&y);
        assert_eq!(sum.get_real(), 5.0);
        assert_eq!(sum.get_dual(), 1.0); // d/dx(x + 2) = 1

        // Test x * y
        let prod = x.mul(&y);
        assert_eq!(prod.get_real(), 6.0);
        assert_eq!(prod.get_dual(), 2.0); // d/dx(x * 2) = 2
    }

    #[allow(dead_code)]
    #[wasm_bindgen_test]
    fn test_dual_functions() {
        let x = WasmDualNumber::variable(1.0);

        // Test exp(x)
        let exp_x = x.exp();
        assert!((exp_x.get_real() - std::f64::consts::E).abs() < 1e-10);
        assert!((exp_x.get_dual() - std::f64::consts::E).abs() < 1e-10); // d/dx(exp(x)) = exp(x)

        // Test x^2
        let x_squared = x.pow(2.0);
        assert_eq!(x_squared.get_real(), 1.0);
        assert_eq!(x_squared.get_dual(), 2.0); // d/dx(x^2) = 2x at x=1
    }

    #[allow(dead_code)]
    #[wasm_bindgen_test]
    fn test_multi_dual() {
        // Test f(x,y) = x*y at (2,3)
        let x = WasmMultiDualNumber::variable(2.0, 2, 0); // variable 0
        let y = WasmMultiDualNumber::variable(3.0, 2, 1); // variable 1

        let product = x.mul(&y).unwrap();
        assert_eq!(product.get_real(), 6.0);

        let grad = product.get_gradient();
        assert_eq!(grad[0], 3.0); // ∂/∂x(x*y) = y = 3
        assert_eq!(grad[1], 2.0); // ∂/∂y(x*y) = x = 2
    }
}
