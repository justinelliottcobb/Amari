//! WASM bindings for tropical-dual-Clifford fusion systems

use amari_fusion::TropicalDualClifford;
use wasm_bindgen::prelude::*;

/// WASM wrapper for TropicalDualClifford fusion system
#[wasm_bindgen]
pub struct WasmTropicalDualClifford {
    inner: TropicalDualClifford<f64, 8>, // Default to 8D for flexibility
}

#[wasm_bindgen]
impl WasmTropicalDualClifford {
    /// Create a zero TDC object
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            inner: TropicalDualClifford::zero(),
        }
    }

    /// Create from logits (array of log-probabilities)
    #[wasm_bindgen(js_name = fromLogits)]
    pub fn from_logits(logits: &[f64]) -> Self {
        Self {
            inner: TropicalDualClifford::from_logits(logits),
        }
    }

    /// Create random TDC for testing
    #[wasm_bindgen(js_name = random)]
    pub fn random() -> Self {
        Self {
            inner: TropicalDualClifford::random(),
        }
    }

    /// Create random TDC with specific scale
    #[wasm_bindgen(js_name = randomWithScale)]
    pub fn random_with_scale(scale: f64) -> Self {
        Self {
            inner: TropicalDualClifford::random_with_scale(scale),
        }
    }

    /// Check if TDC is zero
    #[wasm_bindgen(js_name = isZero)]
    pub fn is_zero(&self) -> bool {
        self.inner.is_zero()
    }

    /// Extract tropical features as array
    #[wasm_bindgen(js_name = getTropicalFeatures)]
    pub fn get_tropical_features(&self) -> Vec<f64> {
        self.inner
            .extract_tropical_features()
            .into_iter()
            .map(|t| t.value())
            .collect()
    }

    /// Extract dual features as array of real parts
    #[wasm_bindgen(js_name = getDualReals)]
    pub fn get_dual_reals(&self) -> Vec<f64> {
        self.inner
            .extract_dual_features()
            .into_iter()
            .map(|d| d.real)
            .collect()
    }

    /// Extract dual features as array of dual parts (derivatives)
    #[wasm_bindgen(js_name = getDualDerivatives)]
    pub fn get_dual_derivatives(&self) -> Vec<f64> {
        self.inner
            .extract_dual_features()
            .into_iter()
            .map(|d| d.dual)
            .collect()
    }

    /// Get Clifford coefficients
    #[wasm_bindgen(js_name = getCliffordCoefficients)]
    pub fn get_clifford_coefficients(&self) -> Vec<f64> {
        (0..8).map(|i| self.inner.clifford.get(i)).collect()
    }

    /// Add two TDC objects
    pub fn add(&self, other: &WasmTropicalDualClifford) -> WasmTropicalDualClifford {
        Self {
            inner: self.inner.add(&other.inner),
        }
    }

    /// Scale TDC object
    pub fn scale(&self, factor: f64) -> WasmTropicalDualClifford {
        Self {
            inner: self.inner.scale(factor),
        }
    }

    /// Compute fusion norm (combined measure across all systems)
    #[wasm_bindgen(js_name = fusionNorm)]
    pub fn fusion_norm(&self) -> f64 {
        // Combine norms from all three systems
        let tropical_norm = self
            .get_tropical_features()
            .iter()
            .map(|&x| if x.is_finite() { x } else { 0.0 })
            .fold(f64::NEG_INFINITY, f64::max);

        let dual_norm = self
            .get_dual_reals()
            .iter()
            .map(|&x| x * x)
            .sum::<f64>()
            .sqrt();

        let clifford_norm = self.inner.clifford.magnitude();

        // Weighted combination
        (tropical_norm.max(0.0) + dual_norm + clifford_norm) / 3.0
    }

    /// Perform tropical attention operation
    #[wasm_bindgen(js_name = tropicalAttention)]
    pub fn tropical_attention(&self, keys: &[f64], values: &[f64]) -> Result<Vec<f64>, JsValue> {
        if keys.len() != values.len() {
            return Err(JsValue::from_str("Keys and values must have same length"));
        }

        let query_features = self.get_tropical_features();
        let mut attention_scores = Vec::new();

        // Compute tropical attention scores (max-based)
        for &key in keys {
            let score = query_features
                .iter()
                .map(|&q| {
                    if q.is_finite() {
                        q + key
                    } else {
                        f64::NEG_INFINITY
                    }
                })
                .fold(f64::NEG_INFINITY, f64::max);
            attention_scores.push(score);
        }

        // Apply tropical softmax (find max and normalize)
        let max_score = attention_scores
            .iter()
            .fold(f64::NEG_INFINITY, |a, &b| a.max(b));

        let normalized_scores: Vec<f64> = attention_scores
            .iter()
            .map(|&score| if score == max_score { 1.0 } else { 0.0 })
            .collect();

        // Compute weighted sum of values
        let mut result = 0.0;
        for (i, &weight) in normalized_scores.iter().enumerate() {
            result += weight * values[i];
        }

        Ok(vec![result])
    }

    /// Extract geometric features using Clifford operations
    #[wasm_bindgen(js_name = extractGeometricFeatures)]
    pub fn extract_geometric_features(&self) -> Vec<f64> {
        let mv = &self.inner.clifford;
        vec![
            mv.get(0),                // scalar
            mv.get(1),                // e1
            mv.get(2),                // e2
            mv.get(3),                // e3
            mv.magnitude(),           // magnitude
            mv.reverse().magnitude(), // reverse magnitude
        ]
    }
}

impl Default for WasmTropicalDualClifford {
    fn default() -> Self {
        Self::new()
    }
}

/// Fusion utilities for batch operations
#[wasm_bindgen]
pub struct FusionBatch;

#[wasm_bindgen]
impl FusionBatch {
    /// Batch tropical attention across multiple queries
    #[wasm_bindgen(js_name = batchTropicalAttention)]
    pub fn batch_tropical_attention(
        queries: &[f64],
        keys: &[f64],
        values: &[f64],
        query_dim: usize,
    ) -> Result<Vec<f64>, JsValue> {
        if !queries.len().is_multiple_of(query_dim) {
            return Err(JsValue::from_str(
                "Queries length must be divisible by query_dim",
            ));
        }

        if keys.len() != values.len() {
            return Err(JsValue::from_str("Keys and values must have same length"));
        }

        let num_queries = queries.len() / query_dim;
        let mut results = Vec::new();

        for i in 0..num_queries {
            let start = i * query_dim;
            let end = start + query_dim;
            let query_logits = &queries[start..end];

            // Create TDC from query
            let tdc = WasmTropicalDualClifford::from_logits(query_logits);

            // Compute attention
            let attention_result = tdc.tropical_attention(keys, values)?;
            results.extend(attention_result);
        }

        Ok(results)
    }

    /// Compute fusion similarity between two TDC objects
    #[wasm_bindgen(js_name = fusionSimilarity)]
    pub fn fusion_similarity(
        tdc1: &WasmTropicalDualClifford,
        tdc2: &WasmTropicalDualClifford,
    ) -> f64 {
        // Tropical similarity (max-based correlation)
        let trop1 = tdc1.get_tropical_features();
        let trop2 = tdc2.get_tropical_features();
        let tropical_sim = trop1
            .iter()
            .zip(trop2.iter())
            .map(|(&a, &b)| {
                if a.is_finite() && b.is_finite() {
                    (a + b).max(0.0)
                } else {
                    0.0
                }
            })
            .sum::<f64>();

        // Dual similarity (dot product of real parts)
        let dual1 = tdc1.get_dual_reals();
        let dual2 = tdc2.get_dual_reals();
        let dual_sim = dual1
            .iter()
            .zip(dual2.iter())
            .map(|(&a, &b)| a * b)
            .sum::<f64>();

        // Clifford similarity (scalar product)
        let cliff1 = tdc1.get_clifford_coefficients();
        let cliff2 = tdc2.get_clifford_coefficients();
        let clifford_sim = cliff1
            .iter()
            .zip(cliff2.iter())
            .map(|(&a, &b)| a * b)
            .sum::<f64>();

        // Combined similarity
        (tropical_sim + dual_sim + clifford_sim) / 3.0
    }

    /// Optimize TDC parameters using gradient information
    #[wasm_bindgen(js_name = gradientStep)]
    pub fn gradient_step(
        tdc: &WasmTropicalDualClifford,
        learning_rate: f64,
    ) -> WasmTropicalDualClifford {
        let dual_derivs = tdc.get_dual_derivatives();

        // Simple gradient descent step on dual components
        let mut new_logits = tdc.get_tropical_features();
        for (i, &grad) in dual_derivs.iter().enumerate() {
            if i < new_logits.len() {
                new_logits[i] -= learning_rate * grad;
            }
        }

        WasmTropicalDualClifford::from_logits(&new_logits)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use wasm_bindgen_test::*;

    #[allow(dead_code)]
    #[wasm_bindgen_test]
    fn test_fusion_basic() {
        let tdc = WasmTropicalDualClifford::new();
        assert!(tdc.is_zero());

        let random_tdc = WasmTropicalDualClifford::random();
        assert!(!random_tdc.is_zero());
    }

    #[allow(dead_code)]
    #[wasm_bindgen_test]
    fn test_fusion_from_logits() {
        let logits = vec![0.1, 0.2, 0.3, 0.4];
        let tdc = WasmTropicalDualClifford::from_logits(&logits);

        let features = tdc.get_tropical_features();
        assert_eq!(features.len(), 8); // Should have 8 features for 8D system
    }

    #[allow(dead_code)]
    #[wasm_bindgen_test]
    fn test_fusion_operations() {
        let logits1 = vec![1.0, 2.0, 3.0];
        let logits2 = vec![0.5, 1.5, 2.5];

        let tdc1 = WasmTropicalDualClifford::from_logits(&logits1);
        let tdc2 = WasmTropicalDualClifford::from_logits(&logits2);

        let sum = tdc1.add(&tdc2);
        assert!(!sum.is_zero());

        let scaled = tdc1.scale(2.0);
        assert!(!scaled.is_zero());
    }

    #[allow(dead_code)]
    #[wasm_bindgen_test]
    fn test_tropical_attention() {
        let logits = vec![1.0, 2.0, 3.0];
        let tdc = WasmTropicalDualClifford::from_logits(&logits);

        let keys = vec![0.1, 0.2, 0.3];
        let values = vec![1.0, 2.0, 3.0];

        let result = tdc.tropical_attention(&keys, &values).unwrap();
        assert_eq!(result.len(), 1);
    }
}
