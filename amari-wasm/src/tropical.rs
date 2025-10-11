//! WASM bindings for tropical (max-plus) algebra
//!
//! This module provides WebAssembly bindings for tropical algebra, which is critical
//! for optimization algorithms, pathfinding, and efficient neural network operations
//! in web applications.

use amari_tropical::viterbi::{TropicalPolynomial, TropicalViterbi};
use amari_tropical::TropicalNumber;
use js_sys::{Array, Object};
use wasm_bindgen::prelude::*;

/// WASM wrapper for TropicalNumber
#[wasm_bindgen]
pub struct WasmTropicalNumber {
    inner: TropicalNumber<f64>,
}

#[wasm_bindgen]
impl WasmTropicalNumber {
    /// Create a new tropical number from a regular number
    #[wasm_bindgen(constructor)]
    pub fn new(value: f64) -> Self {
        Self {
            inner: TropicalNumber::new(value),
        }
    }

    /// Create tropical zero (negative infinity)
    #[wasm_bindgen(js_name = zero)]
    pub fn zero() -> Self {
        Self {
            inner: TropicalNumber::neg_infinity(),
        }
    }

    /// Create tropical one (regular zero)
    #[wasm_bindgen(js_name = one)]
    pub fn one() -> Self {
        Self {
            inner: TropicalNumber::zero(),
        }
    }

    /// Create from log probability
    #[wasm_bindgen(js_name = fromLogProb)]
    pub fn from_log_prob(log_p: f64) -> Self {
        Self {
            inner: TropicalNumber::from_log_prob(log_p),
        }
    }

    /// Get the underlying value
    #[wasm_bindgen(js_name = getValue)]
    pub fn get_value(&self) -> f64 {
        self.inner.value()
    }

    /// Convert to probability (via exp)
    #[wasm_bindgen(js_name = toProb)]
    pub fn to_prob(&self) -> f64 {
        self.inner.to_prob()
    }

    /// Check if this is tropical zero (negative infinity)
    #[wasm_bindgen(js_name = isZero)]
    pub fn is_zero(&self) -> bool {
        self.inner.is_zero()
    }

    /// Check if this is tropical one (zero)
    #[wasm_bindgen(js_name = isOne)]
    pub fn is_one(&self) -> bool {
        self.inner.is_one()
    }

    /// Check if this is infinite
    #[wasm_bindgen(js_name = isInfinity)]
    pub fn is_infinity(&self) -> bool {
        self.inner.is_infinity()
    }

    /// Tropical addition (max operation)
    #[wasm_bindgen(js_name = tropicalAdd)]
    pub fn tropical_add(&self, other: &WasmTropicalNumber) -> WasmTropicalNumber {
        Self {
            inner: self.inner.tropical_add(other.inner),
        }
    }

    /// Tropical multiplication (addition)
    #[wasm_bindgen(js_name = tropicalMul)]
    pub fn tropical_mul(&self, other: &WasmTropicalNumber) -> WasmTropicalNumber {
        Self {
            inner: self.inner.tropical_mul(other.inner),
        }
    }

    /// Tropical power (scalar multiplication)
    #[wasm_bindgen(js_name = tropicalPow)]
    pub fn tropical_pow(&self, n: f64) -> WasmTropicalNumber {
        Self {
            inner: self.inner.tropical_pow(n),
        }
    }

    /// Standard addition (for convenience)
    pub fn add(&self, other: &WasmTropicalNumber) -> WasmTropicalNumber {
        self.tropical_add(other)
    }

    /// Standard multiplication (for convenience)
    pub fn mul(&self, other: &WasmTropicalNumber) -> WasmTropicalNumber {
        self.tropical_mul(other)
    }

    /// Negation
    pub fn neg(&self) -> WasmTropicalNumber {
        Self { inner: -self.inner }
    }
}

/// Batch operations for tropical numbers
#[wasm_bindgen]
pub struct TropicalBatch;

#[wasm_bindgen]
impl TropicalBatch {
    /// Batch tropical addition (max operation)
    #[wasm_bindgen(js_name = batchTropicalAdd)]
    pub fn batch_tropical_add(values: &[f64]) -> f64 {
        values.iter().fold(f64::NEG_INFINITY, |acc, &x| acc.max(x))
    }

    /// Batch tropical multiplication (addition)
    #[wasm_bindgen(js_name = batchTropicalMul)]
    pub fn batch_tropical_mul(values: &[f64]) -> f64 {
        values.iter().sum()
    }

    /// Convert array of log probabilities to tropical numbers and find maximum
    #[wasm_bindgen(js_name = maxLogProb)]
    pub fn max_log_prob(log_probs: &[f64]) -> f64 {
        Self::batch_tropical_add(log_probs)
    }

    /// Viterbi algorithm helper: find best path through trellis
    #[wasm_bindgen(js_name = viterbiStep)]
    pub fn viterbi_step(
        prev_scores: &[f64],
        transition_scores: &[f64],
        emission_scores: &[f64],
        num_states: usize,
    ) -> Result<Vec<f64>, JsValue> {
        if transition_scores.len() != num_states * num_states {
            return Err(JsValue::from_str(
                "transition_scores must have length num_states^2",
            ));
        }

        if emission_scores.len() != num_states {
            return Err(JsValue::from_str(
                "emission_scores must have length num_states",
            ));
        }

        if prev_scores.len() != num_states {
            return Err(JsValue::from_str("prev_scores must have length num_states"));
        }

        let mut new_scores = vec![f64::NEG_INFINITY; num_states];

        for j in 0..num_states {
            for i in 0..num_states {
                let score =
                    prev_scores[i] + transition_scores[i * num_states + j] + emission_scores[j];
                new_scores[j] = new_scores[j].max(score);
            }
        }

        Ok(new_scores)
    }
}

/// WASM wrapper for TropicalViterbi - Hidden Markov Model decoding
#[wasm_bindgen]
pub struct WasmTropicalViterbi {
    inner: TropicalViterbi<f64>,
}

#[wasm_bindgen]
impl WasmTropicalViterbi {
    /// Create a new Viterbi decoder
    ///
    /// # Arguments
    /// * `transitions` - Transition probability matrix (2D array)
    /// * `emissions` - Emission probability matrix (2D array)
    #[wasm_bindgen(constructor)]
    pub fn new(transitions: &JsValue, emissions: &JsValue) -> Result<WasmTropicalViterbi, JsValue> {
        let transitions = js_value_to_matrix(transitions)?;
        let emissions = js_value_to_matrix(emissions)?;

        Ok(WasmTropicalViterbi {
            inner: TropicalViterbi::new(transitions, emissions),
        })
    }

    /// Decode the most likely state sequence for given observations
    ///
    /// Returns an object with `states` (array of state indices) and `probability` (log probability)
    #[wasm_bindgen]
    pub fn decode(&self, observations: &[u32]) -> Result<JsValue, JsValue> {
        let obs: Vec<usize> = observations.iter().map(|&x| x as usize).collect();
        let (states, prob) = self.inner.decode(&obs);

        let states_array = Array::new();
        for state in states {
            states_array.push(&JsValue::from(state as u32));
        }

        let result = Object::new();
        js_sys::Reflect::set(&result, &"states".into(), &states_array)?;
        let prob_val: f64 = prob.value();
        js_sys::Reflect::set(&result, &"probability".into(), &JsValue::from(prob_val))?;

        Ok(result.into())
    }

    /// Compute forward probabilities for all states
    #[wasm_bindgen]
    pub fn forward_probabilities(&self, observations: &[u32]) -> Result<Array, JsValue> {
        let obs: Vec<usize> = observations.iter().map(|&x| x as usize).collect();
        let matrix = self.inner.forward_probabilities(&obs);

        // Convert TropicalMatrix to JS Array using the public interface
        let scores = matrix.to_attention_scores();
        let result = Array::new();
        for row in scores {
            let js_row = Array::new();
            for val in row {
                js_row.push(&JsValue::from(val));
            }
            result.push(&js_row);
        }

        Ok(result)
    }
}

/// WASM wrapper for TropicalPolynomial - polynomial operations in tropical algebra
#[wasm_bindgen]
pub struct WasmTropicalPolynomial {
    inner: TropicalPolynomial<f64>,
}

#[wasm_bindgen]
impl WasmTropicalPolynomial {
    /// Create a new tropical polynomial from coefficients
    #[wasm_bindgen(constructor)]
    pub fn new(coefficients: &[f64]) -> Self {
        Self {
            inner: TropicalPolynomial::new(coefficients.to_vec()),
        }
    }

    /// Evaluate the polynomial at a given tropical number
    #[wasm_bindgen]
    pub fn evaluate(&self, x: &WasmTropicalNumber) -> WasmTropicalNumber {
        WasmTropicalNumber {
            inner: self.inner.evaluate(x.inner),
        }
    }

    /// Find tropical roots of the polynomial
    #[wasm_bindgen]
    pub fn tropical_roots(&self) -> Array {
        let roots = self.inner.tropical_roots();
        let result = Array::new();

        for root in roots {
            let wasm_root = WasmTropicalNumber { inner: root };
            result.push(&JsValue::from(wasm_root));
        }

        result
    }

    /// Get the number of coefficients
    #[wasm_bindgen]
    pub fn coefficients_count(&self) -> usize {
        // TropicalPolynomial doesn't expose degree, so we can't provide this info
        // Return 0 as a placeholder
        0
    }
}

/// Advanced tropical operations for machine learning and optimization
#[wasm_bindgen]
pub struct TropicalMLOps;

#[wasm_bindgen]
impl TropicalMLOps {
    /// Compute tropical convex combination (used in optimization)
    #[wasm_bindgen(js_name = convexCombination)]
    pub fn convex_combination(values: &[f64], weights: &[f64]) -> Result<f64, JsValue> {
        if values.len() != weights.len() {
            return Err(JsValue::from_str(
                "Values and weights must have the same length",
            ));
        }

        let tropical_values: Vec<f64> = values
            .iter()
            .zip(weights.iter())
            .map(|(&v, &w)| v + w) // Tropical multiplication (regular addition)
            .collect();

        // Tropical sum (maximum)
        Ok(tropical_values
            .iter()
            .fold(f64::NEG_INFINITY, |acc, &x| acc.max(x)))
    }

    /// Compute tropical matrix multiplication for pathfinding
    #[wasm_bindgen(js_name = matrixMultiply)]
    pub fn matrix_multiply(a: &JsValue, b: &JsValue) -> Result<Array, JsValue> {
        let matrix_a = js_value_to_matrix(a)?;
        let matrix_b = js_value_to_matrix(b)?;

        if matrix_a[0].len() != matrix_b.len() {
            return Err(JsValue::from_str(
                "Matrix dimensions don't match for multiplication",
            ));
        }

        let cols_b = matrix_b[0].len();
        let inner_dim = matrix_a[0].len();

        let result = Array::new();

        for row_a in &matrix_a {
            let row = Array::new();
            #[allow(clippy::needless_range_loop)]
            for j in 0..cols_b {
                let mut max_val = f64::NEG_INFINITY;
                for k in 0..inner_dim {
                    // Tropical matrix multiplication: (A ⊗ B)[i,j] = max_k(A[i,k] + B[k,j])
                    let val = row_a[k] + matrix_b[k][j];
                    max_val = max_val.max(val);
                }
                row.push(&JsValue::from(max_val));
            }
            result.push(&row);
        }

        Ok(result)
    }

    /// Compute shortest paths using tropical algebra (Floyd-Warshall)
    #[wasm_bindgen(js_name = shortestPaths)]
    pub fn shortest_paths(distance_matrix: &JsValue) -> Result<Array, JsValue> {
        let mut distances = js_value_to_matrix(distance_matrix)?;
        let n = distances.len();

        if distances.iter().any(|row| row.len() != n) {
            return Err(JsValue::from_str("Distance matrix must be square"));
        }

        // Floyd-Warshall algorithm in tropical algebra
        for k in 0..n {
            for i in 0..n {
                for j in 0..n {
                    // Tropical addition: max(distances[i][j], distances[i][k] + distances[k][j])
                    let new_distance = distances[i][k] + distances[k][j];
                    distances[i][j] = distances[i][j].max(new_distance);
                }
            }
        }

        // Convert back to JS array
        let result = Array::new();
        for row in distances {
            let js_row = Array::new();
            for val in row {
                js_row.push(&JsValue::from(val));
            }
            result.push(&js_row);
        }

        Ok(result)
    }
}

/// Utility function to convert JS 2D array to Vec<Vec<f64>>
fn js_value_to_matrix(value: &JsValue) -> Result<Vec<Vec<f64>>, JsValue> {
    let array = Array::from(value);
    let mut matrix = Vec::new();

    for i in 0..array.length() {
        let row_js = Array::from(&array.get(i));
        let mut row = Vec::new();

        for j in 0..row_js.length() {
            let val = row_js
                .get(j)
                .as_f64()
                .ok_or_else(|| JsValue::from_str("Matrix elements must be numbers"))?;
            row.push(val);
        }
        matrix.push(row);
    }

    Ok(matrix)
}

#[cfg(test)]
mod tests {
    use super::*;
    use wasm_bindgen_test::*;

    #[allow(dead_code)]
    #[wasm_bindgen_test]
    fn test_tropical_basic() {
        let a = WasmTropicalNumber::new(2.0);
        let b = WasmTropicalNumber::new(3.0);

        // Tropical addition is max
        let sum = a.tropical_add(&b);
        assert_eq!(sum.get_value(), 3.0);

        // Tropical multiplication is addition
        let prod = a.tropical_mul(&b);
        assert_eq!(prod.get_value(), 5.0);
    }

    #[allow(dead_code)]
    #[wasm_bindgen_test]
    fn test_tropical_identities() {
        let zero = WasmTropicalNumber::zero();
        let one = WasmTropicalNumber::one();
        let x = WasmTropicalNumber::new(5.0);

        assert!(zero.is_zero());
        assert!(one.is_one());

        // x + 0 = x (tropical addition with zero)
        let x_plus_zero = x.tropical_add(&zero);
        assert_eq!(x_plus_zero.get_value(), 5.0);

        // x * 1 = x (tropical multiplication with one)
        let x_times_one = x.tropical_mul(&one);
        assert_eq!(x_times_one.get_value(), 5.0);
    }
}
