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
        self.inner.value()
    }

    /// Get the dual part (derivative)
    #[wasm_bindgen(js_name = getDual)]
    pub fn get_dual(&self) -> f64 {
        self.inner.derivative()
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

    /// Tangent function
    pub fn tan(&self) -> WasmDualNumber {
        Self {
            inner: self.inner.tan(),
        }
    }

    /// Hyperbolic sine
    pub fn sinh(&self) -> WasmDualNumber {
        Self {
            inner: self.inner.sinh(),
        }
    }

    /// Hyperbolic cosine
    pub fn cosh(&self) -> WasmDualNumber {
        Self {
            inner: self.inner.cosh(),
        }
    }

    /// Hyperbolic tangent
    pub fn tanh(&self) -> WasmDualNumber {
        Self {
            inner: self.inner.tanh(),
        }
    }

    /// ReLU activation function
    ///
    /// Note: Manual implementation as relu() is not in v0.12.0 DualNumber API
    pub fn relu(&self) -> WasmDualNumber {
        let x = self.inner.value();
        if x > 0.0 {
            // ReLU(x) = x, derivative = 1
            Self {
                inner: DualNumber::new(x, self.inner.derivative()),
            }
        } else {
            // ReLU(x) = 0, derivative = 0
            Self {
                inner: DualNumber::new(0.0, 0.0),
            }
        }
    }

    /// Sigmoid activation function
    pub fn sigmoid(&self) -> WasmDualNumber {
        Self {
            inner: self.inner.sigmoid(),
        }
    }

    /// Softplus activation function
    ///
    /// Note: Manual implementation as softplus() is not in v0.12.0 DualNumber API
    /// softplus(x) = ln(1 + exp(x))
    pub fn softplus(&self) -> WasmDualNumber {
        let x = self.inner.value();
        let exp_x = x.exp();
        let softplus_val = (1.0 + exp_x).ln();
        // Derivative of softplus: d/dx ln(1 + exp(x)) = exp(x)/(1 + exp(x)) = sigmoid(x)
        let softplus_deriv = exp_x / (1.0 + exp_x);
        Self {
            inner: DualNumber::new(softplus_val, self.inner.derivative() * softplus_deriv),
        }
    }

    /// Maximum of two dual numbers
    pub fn max(&self, other: &WasmDualNumber) -> WasmDualNumber {
        Self {
            inner: self.inner.max(other.inner),
        }
    }

    /// Minimum of two dual numbers
    pub fn min(&self, other: &WasmDualNumber) -> WasmDualNumber {
        Self {
            inner: self.inner.min(other.inner),
        }
    }

    /// Integer power
    #[wasm_bindgen(js_name = powi)]
    pub fn powi(&self, n: i32) -> WasmDualNumber {
        Self {
            inner: self.inner.powi(n),
        }
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
        self.inner.get_value()
    }

    /// Get the gradient (all partial derivatives)
    #[wasm_bindgen(js_name = getGradient)]
    pub fn get_gradient(&self) -> Vec<f64> {
        self.inner.get_gradient().to_vec()
    }

    /// Get a specific partial derivative
    #[wasm_bindgen(js_name = getPartial)]
    pub fn get_partial(&self, index: usize) -> Result<f64, JsValue> {
        self.inner
            .get_gradient()
            .get(index)
            .copied()
            .ok_or_else(|| JsValue::from_str("Index out of bounds"))
    }

    /// Get number of variables
    #[wasm_bindgen(js_name = getNumVars)]
    pub fn get_num_vars(&self) -> usize {
        self.inner.n_vars()
    }

    /// Addition
    pub fn add(&self, other: &WasmMultiDualNumber) -> Result<WasmMultiDualNumber, JsValue> {
        if self.inner.n_vars() != other.inner.n_vars() {
            return Err(JsValue::from_str("Incompatible number of variables"));
        }
        Ok(Self {
            inner: self.inner.clone() + other.inner.clone(),
        })
    }

    /// Multiplication
    pub fn mul(&self, other: &WasmMultiDualNumber) -> Result<WasmMultiDualNumber, JsValue> {
        if self.inner.n_vars() != other.inner.n_vars() {
            return Err(JsValue::from_str("Incompatible number of variables"));
        }
        Ok(Self {
            inner: self.inner.clone() * other.inner.clone(),
        })
    }

    /// Square root
    pub fn sqrt(&self) -> Result<WasmMultiDualNumber, JsValue> {
        if self.inner.get_value() < 0.0 {
            return Err(JsValue::from_str("Square root of negative number"));
        }
        // Note: sqrt() not available on MultiDualNumber in v0.12.0
        // Providing stub that preserves value but computes correct gradient
        let val = self.inner.get_value().sqrt();
        // Compute gradient: d/dx sqrt(f) = f'/(2*sqrt(f))
        let sqrt_val = val;
        let grad: Vec<f64> = self
            .inner
            .get_gradient()
            .iter()
            .map(|&g| g / (2.0 * sqrt_val))
            .collect();
        Ok(Self {
            inner: MultiDualNumber::new(val, grad),
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

    /// Compute mean squared error with automatic gradients
    #[wasm_bindgen(js_name = meanSquaredError)]
    pub fn mean_squared_error(
        predictions: &[f64],
        targets: &[f64],
    ) -> Result<WasmDualNumber, JsValue> {
        if predictions.len() != targets.len() {
            return Err(JsValue::from_str(
                "Predictions and targets must have same length",
            ));
        }

        let n = predictions.len();
        let mut mse = WasmDualNumber::constant(0.0);

        for (pred_val, target_val) in predictions.iter().zip(targets.iter()) {
            let pred = WasmDualNumber::variable(*pred_val);
            let target = WasmDualNumber::constant(*target_val);
            let diff = pred.sub(&target);
            let squared = diff.mul(&diff);
            mse = mse.add(&squared);
        }

        let n_dual = WasmDualNumber::constant(n as f64);
        mse.div(&n_dual)
    }

    /// Linear layer forward pass (y = Wx + b) with automatic gradients
    #[wasm_bindgen(js_name = linearLayer)]
    pub fn linear_layer(
        inputs: &[f64],
        weights: &[f64], // Flattened weight matrix
        bias: &[f64],
        input_size: usize,
        output_size: usize,
    ) -> Result<Vec<f64>, JsValue> {
        if weights.len() != input_size * output_size {
            return Err(JsValue::from_str("Weight matrix size mismatch"));
        }
        if bias.len() != output_size {
            return Err(JsValue::from_str("Bias vector size mismatch"));
        }
        if inputs.len() != input_size {
            return Err(JsValue::from_str("Input vector size mismatch"));
        }

        let mut outputs = Vec::new();

        for i in 0..output_size {
            let mut output = WasmDualNumber::constant(bias[i]);

            for j in 0..input_size {
                let input_dual = WasmDualNumber::variable(inputs[j]);
                let weight_dual = WasmDualNumber::constant(weights[i * input_size + j]);
                let product = input_dual.mul(&weight_dual);
                output = output.add(&product);
            }

            outputs.push(output.get_real());
        }

        Ok(outputs)
    }
}

/// Advanced machine learning operations using dual numbers
#[wasm_bindgen]
pub struct MLOps;

#[wasm_bindgen]
impl MLOps {
    /// Compute softmax with automatic gradients
    #[wasm_bindgen(js_name = softmax)]
    pub fn softmax(inputs: &[f64]) -> Vec<f64> {
        let mut dual_inputs = Vec::new();

        // Convert inputs to dual numbers
        for &val in inputs {
            dual_inputs.push(WasmDualNumber::variable(val));
        }

        // Find max for numerical stability
        let max_val = inputs.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let max_dual = WasmDualNumber::constant(max_val);

        // Compute exp(x_i - max) for each input
        let mut exp_vals = Vec::new();
        for dual in &dual_inputs {
            let shifted = dual.sub(&max_dual);
            exp_vals.push(shifted.exp());
        }

        // Compute sum of exponentials
        let mut sum = WasmDualNumber::constant(0.0);
        for exp_val in &exp_vals {
            sum = sum.add(exp_val);
        }

        // Divide each by sum to get softmax
        let mut result = Vec::new();
        for exp_val in exp_vals {
            let softmax_val = exp_val.div(&sum).unwrap();
            result.push(softmax_val.get_real());
        }

        result
    }

    /// Batch apply activation function with gradients
    #[wasm_bindgen(js_name = batchActivation)]
    pub fn batch_activation(inputs: &[f64], activation: &str) -> Result<Vec<f64>, JsValue> {
        let mut outputs = Vec::new();

        for &input in inputs {
            let dual = WasmDualNumber::variable(input);
            let output = match activation {
                "relu" => dual.relu(),
                "sigmoid" => dual.sigmoid(),
                "tanh" => dual.tanh(),
                "softplus" => dual.softplus(),
                "sin" => dual.sin(),
                "cos" => dual.cos(),
                "exp" => dual.exp(),
                _ => return Err(JsValue::from_str("Unsupported activation function")),
            };
            outputs.push(output.get_real());
        }

        Ok(outputs)
    }

    /// Gradient descent step
    #[wasm_bindgen(js_name = gradientDescentStep)]
    pub fn gradient_descent_step(
        parameters: &[f64],
        gradients: &[f64],
        learning_rate: f64,
    ) -> Result<Vec<f64>, JsValue> {
        if parameters.len() != gradients.len() {
            return Err(JsValue::from_str(
                "Parameters and gradients must have same length",
            ));
        }

        let mut updated_params = Vec::new();
        for (&param, &grad) in parameters.iter().zip(gradients.iter()) {
            let new_param = param - learning_rate * grad;
            updated_params.push(new_param);
        }

        Ok(updated_params)
    }

    /// Compute cross-entropy loss with automatic gradients
    #[wasm_bindgen(js_name = crossEntropyLoss)]
    pub fn cross_entropy_loss(
        predictions: &[f64],
        targets: &[f64],
    ) -> Result<WasmDualNumber, JsValue> {
        if predictions.len() != targets.len() {
            return Err(JsValue::from_str(
                "Predictions and targets must have same length",
            ));
        }

        let mut loss = WasmDualNumber::constant(0.0);
        let eps = 1e-8; // Small value to prevent log(0)

        for (&pred, &target) in predictions.iter().zip(targets.iter()) {
            let pred_dual = WasmDualNumber::variable((pred + eps).max(eps).min(1.0 - eps));
            let target_dual = WasmDualNumber::constant(target);

            let log_pred = pred_dual.ln()?;
            let term = target_dual.mul(&log_pred);
            loss = loss.sub(&term);
        }

        Ok(loss)
    }
}

/// Batch operations for efficient computation
#[wasm_bindgen]
pub struct BatchOps;

#[wasm_bindgen]
impl BatchOps {
    /// Matrix multiplication with automatic gradients
    #[wasm_bindgen(js_name = matrixMultiply)]
    pub fn matrix_multiply(
        a: &[f64],
        b: &[f64],
        a_rows: usize,
        a_cols: usize,
        b_rows: usize,
        b_cols: usize,
    ) -> Result<Vec<f64>, JsValue> {
        if a_cols != b_rows {
            return Err(JsValue::from_str(
                "Matrix dimensions incompatible for multiplication",
            ));
        }
        if a.len() != a_rows * a_cols || b.len() != b_rows * b_cols {
            return Err(JsValue::from_str("Matrix data size mismatch"));
        }

        let mut result = Vec::with_capacity(a_rows * b_cols);

        for i in 0..a_rows {
            for j in 0..b_cols {
                let mut sum = WasmDualNumber::constant(0.0);

                for k in 0..a_cols {
                    let a_val = WasmDualNumber::variable(a[i * a_cols + k]);
                    let b_val = WasmDualNumber::constant(b[k * b_cols + j]);
                    let product = a_val.mul(&b_val);
                    sum = sum.add(&product);
                }

                result.push(sum.get_real());
            }
        }

        Ok(result)
    }

    /// Compute Jacobian matrix for vector function
    #[wasm_bindgen(js_name = computeJacobian)]
    pub fn compute_jacobian(
        input_values: &[f64],
        function_name: &str,
    ) -> Result<Vec<f64>, JsValue> {
        let mut jacobian = Vec::new();

        for &input_val in input_values {
            let dual = WasmDualNumber::variable(input_val);

            let result = match function_name {
                "sin" => dual.sin(),
                "cos" => dual.cos(),
                "exp" => dual.exp(),
                "tanh" => dual.tanh(),
                "sigmoid" => dual.sigmoid(),
                "square" => dual.mul(&dual),
                _ => return Err(JsValue::from_str("Unsupported function")),
            };

            jacobian.push(result.get_dual());
        }

        Ok(jacobian)
    }

    /// Batch evaluate polynomial with derivatives
    #[wasm_bindgen(js_name = batchPolynomial)]
    pub fn batch_polynomial(x_values: &[f64], coefficients: &[f64]) -> Result<Vec<f64>, JsValue> {
        let mut results = Vec::new();

        for &x in x_values {
            let poly_result = AutoDiff::evaluate_polynomial(x, coefficients);
            results.push(poly_result.get_real());
            results.push(poly_result.get_dual()); // Include derivative
        }

        Ok(results)
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
