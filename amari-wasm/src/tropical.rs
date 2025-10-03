//! WASM bindings for tropical (max-plus) algebra

use amari_tropical::TropicalNumber;
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
