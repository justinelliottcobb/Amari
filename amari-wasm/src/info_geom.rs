//! WASM bindings for information geometry

use amari_info_geom::{DuallyFlatManifold, FisherInformationMatrix, SimpleAlphaConnection};
use wasm_bindgen::prelude::*;

/// WASM wrapper for DuallyFlatManifold
#[wasm_bindgen]
pub struct WasmDuallyFlatManifold {
    inner: DuallyFlatManifold,
}

#[wasm_bindgen]
impl WasmDuallyFlatManifold {
    /// Create a new dually flat manifold
    #[wasm_bindgen(constructor)]
    pub fn new(dimension: usize, alpha: f64) -> Self {
        Self {
            inner: DuallyFlatManifold::new(dimension, alpha),
        }
    }

    /// Compute Fisher information metric at a point
    #[wasm_bindgen(js_name = fisherMetricAt)]
    pub fn fisher_metric_at(&self, point: &[f64]) -> WasmFisherInformationMatrix {
        WasmFisherInformationMatrix {
            inner: self.inner.fisher_metric_at(point),
        }
    }

    /// Compute Bregman divergence (KL divergence for probability distributions)
    #[wasm_bindgen(js_name = bregmanDivergence)]
    pub fn bregman_divergence(&self, p: &[f64], q: &[f64]) -> f64 {
        self.inner.bregman_divergence(p, q)
    }

    /// Compute KL divergence between two probability distributions
    #[wasm_bindgen(js_name = klDivergence)]
    pub fn kl_divergence(&self, p: &[f64], q: &[f64]) -> Result<f64, JsValue> {
        // Validate probability distributions
        let sum_p: f64 = p.iter().sum();
        let sum_q: f64 = q.iter().sum();

        if (sum_p - 1.0).abs() > 1e-6 {
            return Err(JsValue::from_str("First distribution doesn't sum to 1"));
        }

        if (sum_q - 1.0).abs() > 1e-6 {
            return Err(JsValue::from_str("Second distribution doesn't sum to 1"));
        }

        Ok(self.inner.bregman_divergence(p, q))
    }

    /// Compute JS divergence (symmetric version of KL divergence)
    #[wasm_bindgen(js_name = jsDivergence)]
    pub fn js_divergence(&self, p: &[f64], q: &[f64]) -> Result<f64, JsValue> {
        if p.len() != q.len() {
            return Err(JsValue::from_str("Distributions must have same length"));
        }

        // Compute M = (P + Q) / 2
        let m: Vec<f64> = p
            .iter()
            .zip(q.iter())
            .map(|(&pi, &qi)| (pi + qi) / 2.0)
            .collect();

        // JS(P,Q) = 1/2 * KL(P||M) + 1/2 * KL(Q||M)
        let kl_pm = self.kl_divergence(p, &m)?;
        let kl_qm = self.kl_divergence(q, &m)?;

        Ok(0.5 * kl_pm + 0.5 * kl_qm)
    }

    /// Compute Wasserstein-1 distance (Earth Mover's Distance)
    #[wasm_bindgen(js_name = wassersteinDistance)]
    pub fn wasserstein_distance(&self, p: &[f64], q: &[f64]) -> Result<f64, JsValue> {
        if p.len() != q.len() {
            return Err(JsValue::from_str("Distributions must have same length"));
        }

        // For 1D distributions, Wasserstein-1 is the L1 distance between CDFs
        let mut cdf_p = vec![0.0; p.len()];
        let mut cdf_q = vec![0.0; q.len()];

        // Compute CDFs
        cdf_p[0] = p[0];
        cdf_q[0] = q[0];
        for i in 1..p.len() {
            cdf_p[i] = cdf_p[i - 1] + p[i];
            cdf_q[i] = cdf_q[i - 1] + q[i];
        }

        // Compute L1 distance between CDFs
        let mut distance = 0.0;
        for i in 0..p.len() {
            distance += (cdf_p[i] - cdf_q[i]).abs();
        }

        Ok(distance)
    }
}

/// WASM wrapper for FisherInformationMatrix
#[wasm_bindgen]
pub struct WasmFisherInformationMatrix {
    inner: FisherInformationMatrix,
}

#[wasm_bindgen]
impl WasmFisherInformationMatrix {
    /// Get eigenvalues of the Fisher matrix
    #[wasm_bindgen(js_name = getEigenvalues)]
    pub fn get_eigenvalues(&self) -> Vec<f64> {
        self.inner.eigenvalues()
    }

    /// Check if the matrix is positive definite
    #[wasm_bindgen(js_name = isPositiveDefinite)]
    pub fn is_positive_definite(&self) -> bool {
        self.inner.eigenvalues().iter().all(|&x| x > 1e-12)
    }

    /// Compute condition number (ratio of largest to smallest eigenvalue)
    #[wasm_bindgen(js_name = conditionNumber)]
    pub fn condition_number(&self) -> f64 {
        let eigenvals = self.inner.eigenvalues();
        if eigenvals.is_empty() {
            return f64::INFINITY;
        }

        let max_eval = eigenvals.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let min_eval = eigenvals.iter().fold(f64::INFINITY, |a, &b| a.min(b));

        if min_eval > 1e-12 {
            max_eval / min_eval
        } else {
            f64::INFINITY
        }
    }
}

/// WASM wrapper for AlphaConnection
#[wasm_bindgen]
pub struct WasmAlphaConnection {
    inner: SimpleAlphaConnection,
}

#[wasm_bindgen]
impl WasmAlphaConnection {
    /// Create a new α-connection
    #[wasm_bindgen(constructor)]
    pub fn new(alpha: f64) -> Result<Self, JsValue> {
        if !(-1.0..=1.0).contains(&alpha) {
            return Err(JsValue::from_str("Alpha must be in range [-1, 1]"));
        }

        Ok(Self {
            inner: SimpleAlphaConnection::new(alpha),
        })
    }

    /// Get the α parameter
    #[wasm_bindgen(js_name = getAlpha)]
    pub fn get_alpha(&self) -> f64 {
        self.inner.alpha()
    }

    /// Check if this is the exponential connection (α = 1)
    #[wasm_bindgen(js_name = isExponential)]
    pub fn is_exponential(&self) -> bool {
        (self.inner.alpha() - 1.0).abs() < 1e-12
    }

    /// Check if this is the mixture connection (α = -1)
    #[wasm_bindgen(js_name = isMixture)]
    pub fn is_mixture(&self) -> bool {
        (self.inner.alpha() + 1.0).abs() < 1e-12
    }

    /// Check if this is the Levi-Civita connection (α = 0)
    #[wasm_bindgen(js_name = isLeviCivita)]
    pub fn is_levi_civita(&self) -> bool {
        self.inner.alpha().abs() < 1e-12
    }
}

/// Information geometry utilities
#[wasm_bindgen]
pub struct InfoGeomUtils;

#[wasm_bindgen]
impl InfoGeomUtils {
    /// Normalize array to probability distribution
    #[wasm_bindgen(js_name = normalize)]
    pub fn normalize(values: &[f64]) -> Result<Vec<f64>, JsValue> {
        let sum: f64 = values.iter().sum();
        if sum <= 0.0 {
            return Err(JsValue::from_str("Cannot normalize non-positive values"));
        }

        Ok(values.iter().map(|&x| x / sum).collect())
    }

    /// Convert log probabilities to probabilities
    #[wasm_bindgen(js_name = softmax)]
    pub fn softmax(logits: &[f64]) -> Vec<f64> {
        let max_logit = logits.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

        // Subtract max for numerical stability
        let exp_logits: Vec<f64> = logits.iter().map(|&x| (x - max_logit).exp()).collect();
        let sum_exp: f64 = exp_logits.iter().sum();

        exp_logits.iter().map(|&x| x / sum_exp).collect()
    }

    /// Compute entropy of a probability distribution
    #[wasm_bindgen(js_name = entropy)]
    pub fn entropy(p: &[f64]) -> Result<f64, JsValue> {
        let sum: f64 = p.iter().sum();
        if (sum - 1.0).abs() > 1e-6 {
            return Err(JsValue::from_str("Distribution doesn't sum to 1"));
        }

        let mut entropy = 0.0;
        for &pi in p {
            if pi > 1e-12 {
                entropy -= pi * pi.ln();
            }
        }

        Ok(entropy)
    }

    /// Compute cross-entropy between two distributions
    #[wasm_bindgen(js_name = crossEntropy)]
    pub fn cross_entropy(p: &[f64], q: &[f64]) -> Result<f64, JsValue> {
        if p.len() != q.len() {
            return Err(JsValue::from_str("Distributions must have same length"));
        }

        let mut cross_entropy = 0.0;
        for (&pi, &qi) in p.iter().zip(q.iter()) {
            if pi > 1e-12 && qi > 1e-12 {
                cross_entropy -= pi * qi.ln();
            } else if pi > 1e-12 {
                return Err(JsValue::from_str(
                    "Cross-entropy undefined when q[i] = 0 and p[i] > 0",
                ));
            }
        }

        Ok(cross_entropy)
    }

    /// Compute mutual information between two discrete distributions
    #[wasm_bindgen(js_name = mutualInformation)]
    pub fn mutual_information(
        joint: &[f64],
        marginal_x: &[f64],
        marginal_y: &[f64],
        dim_x: usize,
    ) -> Result<f64, JsValue> {
        let dim_y = marginal_y.len();
        if joint.len() != dim_x * dim_y {
            return Err(JsValue::from_str(
                "Joint distribution size doesn't match marginals",
            ));
        }

        let mut mi = 0.0;
        #[allow(clippy::needless_range_loop)]
        for i in 0..dim_x {
            for j in 0..dim_y {
                let idx = i * dim_y + j;
                let pxy = joint[idx];
                let px = marginal_x[i];
                let py = marginal_y[j];

                if pxy > 1e-12 && px > 1e-12 && py > 1e-12 {
                    mi += pxy * (pxy / (px * py)).ln();
                }
            }
        }

        Ok(mi)
    }

    /// Generate points on the probability simplex for testing
    /// Note: Uses a simple deterministic sequence for reproducibility in WASM
    #[wasm_bindgen(js_name = randomSimplex)]
    pub fn random_simplex(dimension: usize) -> Vec<f64> {
        // Use a simple deterministic sequence for testing
        let mut values: Vec<f64> = (0..dimension)
            .map(|i| 1.0 + (i as f64 * 0.7).sin().abs())
            .collect();

        let sum: f64 = values.iter().sum();
        for val in &mut values {
            *val /= sum;
        }

        values
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use wasm_bindgen_test::*;

    #[allow(dead_code)]
    #[wasm_bindgen_test]
    fn test_manifold_basic() {
        let manifold = WasmDuallyFlatManifold::new(3, 0.5);
        let point = vec![0.3, 0.4, 0.3];

        let fisher = manifold.fisher_metric_at(&point);
        assert!(fisher.is_positive_definite());
    }

    #[allow(dead_code)]
    #[wasm_bindgen_test]
    fn test_kl_divergence() {
        let manifold = WasmDuallyFlatManifold::new(3, 0.0);
        let p = vec![0.5, 0.3, 0.2];
        let q = vec![0.4, 0.4, 0.2];

        let kl = manifold.kl_divergence(&p, &q).unwrap();
        assert!(kl >= 0.0); // KL divergence is non-negative
    }

    #[allow(dead_code)]
    #[wasm_bindgen_test]
    fn test_js_divergence() {
        let manifold = WasmDuallyFlatManifold::new(2, 0.0);
        let p = vec![0.7, 0.3];
        let q = vec![0.4, 0.6];

        let js = manifold.js_divergence(&p, &q).unwrap();
        assert!(js >= 0.0); // JS divergence is non-negative
        assert!(js <= (2.0_f64.ln())); // JS divergence is bounded
    }

    #[allow(dead_code)]
    #[wasm_bindgen_test]
    fn test_alpha_connection() {
        let conn = WasmAlphaConnection::new(0.5).unwrap();
        assert_eq!(conn.get_alpha(), 0.5);
        assert!(!conn.is_exponential());
        assert!(!conn.is_mixture());
        assert!(!conn.is_levi_civita());

        let exp_conn = WasmAlphaConnection::new(1.0).unwrap();
        assert!(exp_conn.is_exponential());
    }

    #[allow(dead_code)]
    #[wasm_bindgen_test]
    fn test_utils() {
        // Test normalization
        let values = vec![1.0, 2.0, 3.0];
        let normalized = InfoGeomUtils::normalize(&values).unwrap();
        let sum: f64 = normalized.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);

        // Test softmax
        let logits = vec![1.0, 2.0, 3.0];
        let probs = InfoGeomUtils::softmax(&logits);
        let prob_sum: f64 = probs.iter().sum();
        assert!((prob_sum - 1.0).abs() < 1e-10);

        // Test entropy
        let uniform = vec![0.25, 0.25, 0.25, 0.25];
        let entropy = InfoGeomUtils::entropy(&uniform).unwrap();
        assert!(entropy > 0.0);
    }
}
