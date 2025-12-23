//! WASM bindings for probabilistic operations on geometric algebra spaces
//!
//! Provides WebAssembly bindings for:
//! - Gaussian distributions on multivector spaces
//! - Uniform distributions on multivector spaces
//! - Sampling and Monte Carlo estimation
//! - Distribution moments (mean, variance, covariance)

#![allow(clippy::needless_range_loop)]

use amari_probabilistic::distribution::{
    Distribution, GaussianMultivector, MultivectorDistribution, UniformMultivector,
};
use wasm_bindgen::prelude::*;

/// WASM wrapper for Gaussian distribution on Cl(3,0,0) multivector space
///
/// A Gaussian distribution over 8-dimensional multivector space with
/// configurable mean and per-component standard deviation.
#[wasm_bindgen]
pub struct WasmGaussianMultivector {
    inner: GaussianMultivector<3, 0, 0>,
}

#[wasm_bindgen]
impl WasmGaussianMultivector {
    /// Create a standard Gaussian (zero mean, unit variance per coefficient)
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            inner: GaussianMultivector::standard(),
        }
    }

    /// Create a Gaussian with specified mean and standard deviation
    ///
    /// # Arguments
    /// * `mean` - 8 coefficients for the mean multivector
    /// * `std_dev` - 8 coefficients for per-coefficient standard deviation
    #[wasm_bindgen(js_name = withParameters)]
    pub fn with_parameters(
        mean: &[f64],
        std_dev: &[f64],
    ) -> Result<WasmGaussianMultivector, JsValue> {
        if mean.len() != 8 {
            return Err(JsValue::from_str(
                "Mean must have 8 coefficients for Cl(3,0,0)",
            ));
        }
        if std_dev.len() != 8 {
            return Err(JsValue::from_str(
                "Std dev must have 8 coefficients for Cl(3,0,0)",
            ));
        }

        // Create multivector from coefficients
        let mean_mv = amari_core::Multivector::<3, 0, 0>::from_coefficients(mean.to_vec());

        GaussianMultivector::new(mean_mv, std_dev.to_vec())
            .map(|inner| Self { inner })
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Create an isotropic Gaussian with given mean and variance
    ///
    /// # Arguments
    /// * `mean` - 8 coefficients for the mean multivector
    /// * `variance` - Variance (same for all components)
    #[wasm_bindgen(js_name = isotropic)]
    pub fn isotropic(mean: &[f64], variance: f64) -> Result<WasmGaussianMultivector, JsValue> {
        if mean.len() != 8 {
            return Err(JsValue::from_str(
                "Mean must have 8 coefficients for Cl(3,0,0)",
            ));
        }

        let mean_mv = amari_core::Multivector::<3, 0, 0>::from_coefficients(mean.to_vec());

        GaussianMultivector::isotropic(mean_mv, variance)
            .map(|inner| Self { inner })
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Create a Gaussian concentrated on a specific grade
    ///
    /// # Arguments
    /// * `grade` - The grade to concentrate on (0=scalar, 1=vectors, 2=bivectors, 3=pseudoscalar)
    /// * `variance` - Variance for coefficients of that grade
    #[wasm_bindgen(js_name = gradeConcentrated)]
    pub fn grade_concentrated(
        grade: usize,
        variance: f64,
    ) -> Result<WasmGaussianMultivector, JsValue> {
        GaussianMultivector::grade_concentrated(grade, variance)
            .map(|inner| Self { inner })
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Draw a sample from this distribution
    ///
    /// Returns 8 coefficients for the sampled multivector
    pub fn sample(&self) -> Vec<f64> {
        use std::f64::consts::PI;
        let mean = self.inner.get_mean();
        let std_devs = self.inner.get_std_devs();

        // Box-Muller for WASM-compatible Gaussian sampling
        let mut result = Vec::with_capacity(8);
        for i in 0..8 {
            let u1: f64 = fastrand::f64().max(1e-10);
            let u2: f64 = fastrand::f64();
            let z = (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos();
            result.push(mean.get(i) + std_devs[i] * z);
        }
        result
    }

    /// Draw multiple samples from this distribution
    ///
    /// Returns flat array of coefficients: num_samples * 8
    #[wasm_bindgen(js_name = sampleBatch)]
    pub fn sample_batch(&self, num_samples: usize) -> Vec<f64> {
        use std::f64::consts::PI;
        let mean = self.inner.get_mean();
        let std_devs = self.inner.get_std_devs();
        let mut result = Vec::with_capacity(num_samples * 8);

        for _ in 0..num_samples {
            for i in 0..8 {
                let u1: f64 = fastrand::f64().max(1e-10);
                let u2: f64 = fastrand::f64();
                let z = (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos();
                result.push(mean.get(i) + std_devs[i] * z);
            }
        }

        result
    }

    /// Compute log probability of a multivector
    ///
    /// # Arguments
    /// * `coefficients` - 8 coefficients of the multivector
    #[wasm_bindgen(js_name = logProb)]
    pub fn log_prob(&self, coefficients: &[f64]) -> Result<f64, JsValue> {
        if coefficients.len() != 8 {
            return Err(JsValue::from_str("Coefficients must have 8 values"));
        }

        let mv = amari_core::Multivector::<3, 0, 0>::from_coefficients(coefficients.to_vec());
        self.inner
            .log_prob(&mv)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Get the mean of this distribution
    ///
    /// Returns 8 coefficients
    #[wasm_bindgen(js_name = getMean)]
    pub fn get_mean(&self) -> Vec<f64> {
        let mean = self.inner.get_mean();
        (0..8).map(|i| mean.get(i)).collect()
    }

    /// Get the standard deviations
    ///
    /// Returns 8 values
    #[wasm_bindgen(js_name = getStdDevs)]
    pub fn get_std_devs(&self) -> Vec<f64> {
        self.inner.get_std_devs().to_vec()
    }

    /// Get the variance (square of std devs)
    ///
    /// Returns 8 values
    #[wasm_bindgen(js_name = getVariance)]
    pub fn get_variance(&self) -> Vec<f64> {
        self.inner.variances()
    }

    /// Get the full covariance matrix (flattened, row-major)
    ///
    /// Returns 64 values (8x8 matrix)
    #[wasm_bindgen(js_name = getCovariance)]
    pub fn get_covariance(&self) -> Vec<f64> {
        self.inner.covariance_matrix().unwrap_or_else(|| {
            // Fallback to diagonal covariance
            let vars = self.inner.variances();
            let mut result = vec![0.0; 64];
            for i in 0..8 {
                result[i * 8 + i] = vars[i];
            }
            result
        })
    }
}

impl Default for WasmGaussianMultivector {
    fn default() -> Self {
        Self::new()
    }
}

/// WASM wrapper for Uniform distribution on Cl(3,0,0) multivector space
///
/// A uniform distribution over a hyperrectangle in 8-dimensional multivector space.
#[wasm_bindgen]
pub struct WasmUniformMultivector {
    inner: UniformMultivector<3, 0, 0>,
    // Store bounds for retrieval
    mins: Vec<f64>,
    maxs: Vec<f64>,
}

#[wasm_bindgen]
impl WasmUniformMultivector {
    /// Create a standard uniform distribution on [-1, 1]^8
    #[wasm_bindgen(constructor)]
    pub fn new() -> Result<WasmUniformMultivector, JsValue> {
        UniformMultivector::hypercube(-1.0, 1.0)
            .map(|inner| Self {
                inner,
                mins: vec![-1.0; 8],
                maxs: vec![1.0; 8],
            })
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Create a uniform distribution on a hypercube [min, max]^8
    #[wasm_bindgen(js_name = hypercube)]
    pub fn hypercube(min: f64, max: f64) -> Result<WasmUniformMultivector, JsValue> {
        UniformMultivector::hypercube(min, max)
            .map(|inner| Self {
                inner,
                mins: vec![min; 8],
                maxs: vec![max; 8],
            })
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Create a uniform distribution with specified bounds
    ///
    /// # Arguments
    /// * `lower` - 8 coefficients for lower bounds
    /// * `upper` - 8 coefficients for upper bounds
    #[wasm_bindgen(js_name = withBounds)]
    pub fn with_bounds(lower: &[f64], upper: &[f64]) -> Result<WasmUniformMultivector, JsValue> {
        if lower.len() != 8 {
            return Err(JsValue::from_str("Lower bounds must have 8 coefficients"));
        }
        if upper.len() != 8 {
            return Err(JsValue::from_str("Upper bounds must have 8 coefficients"));
        }

        UniformMultivector::new(lower.to_vec(), upper.to_vec())
            .map(|inner| Self {
                inner,
                mins: lower.to_vec(),
                maxs: upper.to_vec(),
            })
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Draw a sample from this distribution
    ///
    /// Returns 8 coefficients for the sampled multivector
    pub fn sample(&self) -> Vec<f64> {
        // WASM-compatible uniform sampling
        (0..8)
            .map(|i| {
                let t = fastrand::f64();
                self.mins[i] + t * (self.maxs[i] - self.mins[i])
            })
            .collect()
    }

    /// Draw multiple samples from this distribution
    ///
    /// Returns flat array of coefficients: num_samples * 8
    #[wasm_bindgen(js_name = sampleBatch)]
    pub fn sample_batch(&self, num_samples: usize) -> Vec<f64> {
        let mut result = Vec::with_capacity(num_samples * 8);

        for _ in 0..num_samples {
            for i in 0..8 {
                let t = fastrand::f64();
                result.push(self.mins[i] + t * (self.maxs[i] - self.mins[i]));
            }
        }

        result
    }

    /// Compute log probability of a multivector
    ///
    /// Returns log(1/volume) if inside bounds, error otherwise
    #[wasm_bindgen(js_name = logProb)]
    pub fn log_prob(&self, coefficients: &[f64]) -> Result<f64, JsValue> {
        if coefficients.len() != 8 {
            return Err(JsValue::from_str("Coefficients must have 8 values"));
        }

        let mv = amari_core::Multivector::<3, 0, 0>::from_coefficients(coefficients.to_vec());
        self.inner
            .log_prob(&mv)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Get the lower bounds
    #[wasm_bindgen(js_name = getLower)]
    pub fn get_lower(&self) -> Vec<f64> {
        self.mins.clone()
    }

    /// Get the upper bounds
    #[wasm_bindgen(js_name = getUpper)]
    pub fn get_upper(&self) -> Vec<f64> {
        self.maxs.clone()
    }

    /// Get the mean of this distribution
    #[wasm_bindgen(js_name = getMean)]
    pub fn get_mean(&self) -> Vec<f64> {
        let mean = self.inner.mean();
        (0..8).map(|i| mean.get(i)).collect()
    }

    /// Get the variance of this distribution
    #[wasm_bindgen(js_name = getVariance)]
    pub fn get_variance(&self) -> Vec<f64> {
        self.inner.variances()
    }
}

impl Default for WasmUniformMultivector {
    fn default() -> Self {
        Self::new().expect("Default uniform distribution should be valid")
    }
}

/// Monte Carlo estimation utilities
#[wasm_bindgen]
pub struct WasmMonteCarloEstimator;

#[wasm_bindgen]
impl WasmMonteCarloEstimator {
    /// Compute sample mean from batch samples
    ///
    /// # Arguments
    /// * `samples` - Flat array of samples (num_samples * 8)
    ///
    /// # Returns
    /// 8 coefficients for the mean
    #[wasm_bindgen(js_name = sampleMean)]
    pub fn sample_mean(samples: &[f64]) -> Result<Vec<f64>, JsValue> {
        if !samples.len().is_multiple_of(8) {
            return Err(JsValue::from_str("Sample array must be divisible by 8"));
        }

        let num_samples = samples.len() / 8;
        if num_samples == 0 {
            return Err(JsValue::from_str("Must have at least one sample"));
        }

        let mut mean = vec![0.0; 8];
        for i in 0..num_samples {
            for j in 0..8 {
                mean[j] += samples[i * 8 + j];
            }
        }

        for m in &mut mean {
            *m /= num_samples as f64;
        }

        Ok(mean)
    }

    /// Compute sample variance from batch samples
    ///
    /// # Arguments
    /// * `samples` - Flat array of samples (num_samples * 8)
    ///
    /// # Returns
    /// 8 values for per-coefficient variance
    #[wasm_bindgen(js_name = sampleVariance)]
    pub fn sample_variance(samples: &[f64]) -> Result<Vec<f64>, JsValue> {
        if !samples.len().is_multiple_of(8) {
            return Err(JsValue::from_str("Sample array must be divisible by 8"));
        }

        let num_samples = samples.len() / 8;
        if num_samples < 2 {
            return Err(JsValue::from_str(
                "Must have at least 2 samples for variance",
            ));
        }

        // Compute mean first
        let mean = Self::sample_mean(samples)?;

        // Compute variance
        let mut variance = vec![0.0; 8];
        for i in 0..num_samples {
            for j in 0..8 {
                let diff = samples[i * 8 + j] - mean[j];
                variance[j] += diff * diff;
            }
        }

        // Bessel's correction
        for v in &mut variance {
            *v /= (num_samples - 1) as f64;
        }

        Ok(variance)
    }

    /// Compute sample covariance matrix from batch samples
    ///
    /// # Arguments
    /// * `samples` - Flat array of samples (num_samples * 8)
    ///
    /// # Returns
    /// 64 values for 8x8 covariance matrix (row-major)
    #[wasm_bindgen(js_name = sampleCovariance)]
    pub fn sample_covariance(samples: &[f64]) -> Result<Vec<f64>, JsValue> {
        if !samples.len().is_multiple_of(8) {
            return Err(JsValue::from_str("Sample array must be divisible by 8"));
        }

        let num_samples = samples.len() / 8;
        if num_samples < 2 {
            return Err(JsValue::from_str(
                "Must have at least 2 samples for covariance",
            ));
        }

        // Compute mean first
        let mean = Self::sample_mean(samples)?;

        // Compute covariance
        let mut cov = vec![0.0; 64];
        for i in 0..num_samples {
            for j in 0..8 {
                let diff_j = samples[i * 8 + j] - mean[j];
                for k in 0..8 {
                    let diff_k = samples[i * 8 + k] - mean[k];
                    cov[j * 8 + k] += diff_j * diff_k;
                }
            }
        }

        // Bessel's correction
        for c in &mut cov {
            *c /= (num_samples - 1) as f64;
        }

        Ok(cov)
    }

    /// Monte Carlo estimate of expectation E[f(X)] where f is the geometric product
    ///
    /// Estimates E[X * Y] for independent X ~ dist_x, Y ~ dist_y
    #[wasm_bindgen(js_name = expectationGeometricProduct)]
    pub fn expectation_geometric_product(
        dist_x: &WasmGaussianMultivector,
        dist_y: &WasmGaussianMultivector,
        num_samples: usize,
    ) -> Vec<f64> {
        let x_samples = dist_x.sample_batch(num_samples);
        let y_samples = dist_y.sample_batch(num_samples);

        // Compute geometric products
        let mut sum = vec![0.0; 8];
        for i in 0..num_samples {
            let x_start = i * 8;
            let x = amari_core::Multivector::<3, 0, 0>::from_coefficients(
                x_samples[x_start..x_start + 8].to_vec(),
            );
            let y = amari_core::Multivector::<3, 0, 0>::from_coefficients(
                y_samples[x_start..x_start + 8].to_vec(),
            );
            let product = x.geometric_product(&y);
            for j in 0..8 {
                sum[j] += product.get(j);
            }
        }

        // Compute mean
        for s in &mut sum {
            *s /= num_samples as f64;
        }

        sum
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use wasm_bindgen_test::*;

    #[wasm_bindgen_test]
    #[allow(dead_code)]
    fn test_gaussian_creation() {
        let gaussian = WasmGaussianMultivector::new();
        let mean = gaussian.get_mean();
        assert_eq!(mean.len(), 8);
        for m in &mean {
            assert!(m.abs() < 1e-10);
        }
    }

    #[wasm_bindgen_test]
    #[allow(dead_code)]
    fn test_gaussian_sampling() {
        let gaussian = WasmGaussianMultivector::new();
        let sample = gaussian.sample();
        assert_eq!(sample.len(), 8);
    }

    #[wasm_bindgen_test]
    #[allow(dead_code)]
    fn test_uniform_creation() {
        let uniform = WasmUniformMultivector::new().unwrap();
        let lower = uniform.get_lower();
        let upper = uniform.get_upper();
        assert_eq!(lower.len(), 8);
        assert_eq!(upper.len(), 8);
    }

    #[wasm_bindgen_test]
    #[allow(dead_code)]
    fn test_monte_carlo_mean() {
        let gaussian = WasmGaussianMultivector::new();
        let samples = gaussian.sample_batch(1000);
        let mean = WasmMonteCarloEstimator::sample_mean(&samples).unwrap();
        assert_eq!(mean.len(), 8);
        // Mean should be close to zero for standard Gaussian
        for m in &mean {
            assert!(m.abs() < 0.5);
        }
    }
}
