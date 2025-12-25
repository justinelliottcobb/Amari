//! WASM bindings for probabilistic operations on geometric algebra spaces
//!
//! Provides WebAssembly bindings for:
//! - Gaussian distributions on multivector spaces
//! - Uniform distributions on multivector spaces
//! - Sampling and Monte Carlo estimation
//! - Distribution moments (mean, variance, covariance)
//! - MCMC sampling with Metropolis-Hastings
//! - Uncertainty propagation
//! - Stochastic processes (Geometric Brownian Motion)
//! - Grade-projected distributions

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

// =============================================================================
// MCMC Sampling
// =============================================================================

/// WASM wrapper for MCMC diagnostics
#[wasm_bindgen]
pub struct WasmMCMCDiagnostics {
    /// Total number of steps
    pub num_steps: usize,
    /// Acceptance rate
    pub acceptance_rate: f64,
    /// Effective sample size (if computed)
    effective_sample_size: Option<f64>,
    /// R-hat convergence statistic (if computed)
    r_hat: Option<f64>,
}

#[wasm_bindgen]
impl WasmMCMCDiagnostics {
    /// Check if the sampler has converged (R-hat < 1.1)
    #[wasm_bindgen(js_name = isConverged)]
    pub fn is_converged(&self) -> bool {
        self.r_hat.is_none_or(|r| r < 1.1)
    }

    /// Get effective sample size (or -1 if not computed)
    #[wasm_bindgen(js_name = getEffectiveSampleSize)]
    pub fn get_effective_sample_size(&self) -> f64 {
        self.effective_sample_size.unwrap_or(-1.0)
    }

    /// Get R-hat statistic (or -1 if not computed)
    #[wasm_bindgen(js_name = getRHat)]
    pub fn get_r_hat(&self) -> f64 {
        self.r_hat.unwrap_or(-1.0)
    }
}

/// WASM wrapper for Metropolis-Hastings MCMC sampler
///
/// Samples from a target distribution using random-walk proposals.
#[wasm_bindgen]
pub struct WasmMetropolisHastings {
    /// Proposal standard deviation
    proposal_std: f64,
    /// Current state (8 coefficients)
    current: Vec<f64>,
    /// Current log probability
    current_log_prob: f64,
    /// Number of steps taken
    num_steps: usize,
    /// Number of accepted proposals
    num_accepted: usize,
    /// Target distribution mean
    target_mean: Vec<f64>,
    /// Target distribution std devs
    target_std_devs: Vec<f64>,
}

#[wasm_bindgen]
impl WasmMetropolisHastings {
    /// Create a new Metropolis-Hastings sampler for a Gaussian target
    ///
    /// # Arguments
    /// * `target` - The target Gaussian distribution to sample from
    /// * `proposal_std` - Standard deviation for the proposal distribution
    #[wasm_bindgen(constructor)]
    pub fn new(target: &WasmGaussianMultivector, proposal_std: f64) -> Self {
        let current = target.get_mean();
        let target_mean = target.get_mean();
        let target_std_devs = target.get_std_devs();

        // Compute initial log probability
        let current_log_prob = Self::compute_log_prob(&current, &target_mean, &target_std_devs);

        Self {
            proposal_std,
            current,
            current_log_prob,
            num_steps: 0,
            num_accepted: 0,
            target_mean,
            target_std_devs,
        }
    }

    /// Compute log probability for a Gaussian
    fn compute_log_prob(x: &[f64], mean: &[f64], std_devs: &[f64]) -> f64 {
        let mut log_prob = 0.0;
        for i in 0..8 {
            let diff = x[i] - mean[i];
            let var = std_devs[i] * std_devs[i];
            if var > 0.0 {
                log_prob +=
                    -0.5 * diff * diff / var - 0.5 * (2.0 * std::f64::consts::PI * var).ln();
            }
        }
        log_prob
    }

    /// Take a single MCMC step
    ///
    /// Returns the new sample (8 coefficients)
    pub fn step(&mut self) -> Vec<f64> {
        use std::f64::consts::PI;

        // Generate proposal using Box-Muller
        let mut proposal = Vec::with_capacity(8);
        for i in 0..8 {
            let u1: f64 = fastrand::f64().max(1e-10);
            let u2: f64 = fastrand::f64();
            let z = (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos();
            proposal.push(self.current[i] + self.proposal_std * z);
        }

        // Compute proposal log probability
        let proposal_log_prob =
            Self::compute_log_prob(&proposal, &self.target_mean, &self.target_std_devs);

        // Accept/reject
        let log_alpha = proposal_log_prob - self.current_log_prob;
        let u: f64 = fastrand::f64();

        self.num_steps += 1;

        if u.ln() < log_alpha {
            // Accept
            self.current = proposal;
            self.current_log_prob = proposal_log_prob;
            self.num_accepted += 1;
        }

        self.current.clone()
    }

    /// Run the sampler for multiple steps
    ///
    /// # Arguments
    /// * `num_samples` - Number of samples to collect
    /// * `burnin` - Number of burn-in steps to discard
    ///
    /// Returns flat array of samples (num_samples * 8)
    pub fn run(&mut self, num_samples: usize, burnin: usize) -> Vec<f64> {
        // Burn-in
        for _ in 0..burnin {
            self.step();
        }

        // Collect samples
        let mut samples = Vec::with_capacity(num_samples * 8);
        for _ in 0..num_samples {
            let sample = self.step();
            samples.extend(sample);
        }

        samples
    }

    /// Get sampling diagnostics
    pub fn diagnostics(&self) -> WasmMCMCDiagnostics {
        WasmMCMCDiagnostics {
            num_steps: self.num_steps,
            acceptance_rate: if self.num_steps > 0 {
                self.num_accepted as f64 / self.num_steps as f64
            } else {
                0.0
            },
            effective_sample_size: None,
            r_hat: None,
        }
    }

    /// Get current acceptance rate
    #[wasm_bindgen(js_name = getAcceptanceRate)]
    pub fn get_acceptance_rate(&self) -> f64 {
        if self.num_steps > 0 {
            self.num_accepted as f64 / self.num_steps as f64
        } else {
            0.0
        }
    }

    /// Get current sample
    #[wasm_bindgen(js_name = getCurrent)]
    pub fn get_current(&self) -> Vec<f64> {
        self.current.clone()
    }
}

// =============================================================================
// Uncertainty Propagation
// =============================================================================

/// WASM wrapper for uncertain multivector (mean + covariance)
///
/// Represents a multivector with associated uncertainty, useful for
/// error propagation through geometric operations.
#[wasm_bindgen]
pub struct WasmUncertainMultivector {
    /// Mean (8 coefficients)
    mean: Vec<f64>,
    /// Covariance matrix (64 values, row-major 8x8)
    covariance: Vec<f64>,
}

#[wasm_bindgen]
impl WasmUncertainMultivector {
    /// Create an uncertain multivector with diagonal covariance
    ///
    /// # Arguments
    /// * `mean` - 8 coefficients for the mean
    /// * `variances` - 8 values for per-coefficient variance
    #[wasm_bindgen(constructor)]
    pub fn new(mean: &[f64], variances: &[f64]) -> Result<WasmUncertainMultivector, JsValue> {
        if mean.len() != 8 {
            return Err(JsValue::from_str("Mean must have 8 coefficients"));
        }
        if variances.len() != 8 {
            return Err(JsValue::from_str("Variances must have 8 values"));
        }

        // Build diagonal covariance matrix
        let mut covariance = vec![0.0; 64];
        for i in 0..8 {
            covariance[i * 8 + i] = variances[i];
        }

        Ok(Self {
            mean: mean.to_vec(),
            covariance,
        })
    }

    /// Create with full covariance matrix
    ///
    /// # Arguments
    /// * `mean` - 8 coefficients for the mean
    /// * `covariance` - 64 values for 8x8 covariance matrix (row-major)
    #[wasm_bindgen(js_name = withCovariance)]
    pub fn with_covariance(
        mean: &[f64],
        covariance: &[f64],
    ) -> Result<WasmUncertainMultivector, JsValue> {
        if mean.len() != 8 {
            return Err(JsValue::from_str("Mean must have 8 coefficients"));
        }
        if covariance.len() != 64 {
            return Err(JsValue::from_str("Covariance must have 64 values (8x8)"));
        }

        Ok(Self {
            mean: mean.to_vec(),
            covariance: covariance.to_vec(),
        })
    }

    /// Create a deterministic (zero variance) uncertain multivector
    #[wasm_bindgen(js_name = deterministic)]
    pub fn deterministic(value: &[f64]) -> Result<WasmUncertainMultivector, JsValue> {
        if value.len() != 8 {
            return Err(JsValue::from_str("Value must have 8 coefficients"));
        }

        Ok(Self {
            mean: value.to_vec(),
            covariance: vec![0.0; 64],
        })
    }

    /// Get the mean
    #[wasm_bindgen(js_name = getMean)]
    pub fn get_mean(&self) -> Vec<f64> {
        self.mean.clone()
    }

    /// Get the covariance matrix (64 values, row-major)
    #[wasm_bindgen(js_name = getCovariance)]
    pub fn get_covariance(&self) -> Vec<f64> {
        self.covariance.clone()
    }

    /// Get variances (diagonal of covariance)
    #[wasm_bindgen(js_name = getVariances)]
    pub fn get_variances(&self) -> Vec<f64> {
        (0..8).map(|i| self.covariance[i * 8 + i]).collect()
    }

    /// Get standard deviations
    #[wasm_bindgen(js_name = getStdDevs)]
    pub fn get_std_devs(&self) -> Vec<f64> {
        (0..8).map(|i| self.covariance[i * 8 + i].sqrt()).collect()
    }

    /// Get total variance (trace of covariance)
    #[wasm_bindgen(js_name = getTotalVariance)]
    pub fn get_total_variance(&self) -> f64 {
        (0..8).map(|i| self.covariance[i * 8 + i]).sum()
    }

    /// Linear propagation: scale by a constant
    ///
    /// For Y = aX: E[Y] = aE[X], Var(Y) = a²Var(X)
    #[wasm_bindgen(js_name = scale)]
    pub fn scale(&self, scalar: f64) -> WasmUncertainMultivector {
        let mean: Vec<f64> = self.mean.iter().map(|&m| m * scalar).collect();
        let covariance: Vec<f64> = self
            .covariance
            .iter()
            .map(|&c| c * scalar * scalar)
            .collect();
        WasmUncertainMultivector { mean, covariance }
    }

    /// Add two uncertain multivectors (assuming independence)
    ///
    /// For Z = X + Y: E[Z] = E[X] + E[Y], Var(Z) = Var(X) + Var(Y)
    pub fn add(&self, other: &WasmUncertainMultivector) -> WasmUncertainMultivector {
        let mean: Vec<f64> = self
            .mean
            .iter()
            .zip(&other.mean)
            .map(|(&a, &b)| a + b)
            .collect();
        let covariance: Vec<f64> = self
            .covariance
            .iter()
            .zip(&other.covariance)
            .map(|(&a, &b)| a + b)
            .collect();
        WasmUncertainMultivector { mean, covariance }
    }
}

// =============================================================================
// Stochastic Processes
// =============================================================================

/// WASM wrapper for Geometric Brownian Motion on multivector space
///
/// dX = μX dt + σX dW
///
/// Useful for modeling multiplicative noise processes in geometric algebra.
#[wasm_bindgen]
pub struct WasmGeometricBrownianMotion {
    /// Drift coefficient
    mu: f64,
    /// Diffusion coefficient
    sigma: f64,
}

#[wasm_bindgen]
impl WasmGeometricBrownianMotion {
    /// Create a new Geometric Brownian Motion process
    ///
    /// # Arguments
    /// * `mu` - Drift coefficient (expected growth rate)
    /// * `sigma` - Diffusion coefficient (volatility)
    #[wasm_bindgen(constructor)]
    pub fn new(mu: f64, sigma: f64) -> Self {
        Self { mu, sigma }
    }

    /// Sample a single path of the process
    ///
    /// # Arguments
    /// * `initial` - Initial multivector (8 coefficients)
    /// * `t_end` - End time
    /// * `num_steps` - Number of time steps
    ///
    /// Returns flat array: (num_steps + 1) * 9 values
    /// Each point is [time, coeff0, coeff1, ..., coeff7]
    #[wasm_bindgen(js_name = samplePath)]
    pub fn sample_path(
        &self,
        initial: &[f64],
        t_end: f64,
        num_steps: usize,
    ) -> Result<Vec<f64>, JsValue> {
        use std::f64::consts::PI;

        if initial.len() != 8 {
            return Err(JsValue::from_str("Initial must have 8 coefficients"));
        }

        let dt = t_end / num_steps as f64;
        let sqrt_dt = dt.sqrt();

        let mut path = Vec::with_capacity((num_steps + 1) * 9);

        // Initial point
        path.push(0.0); // time
        path.extend_from_slice(initial);

        let mut current = initial.to_vec();

        for i in 1..=num_steps {
            let t = i as f64 * dt;

            // Euler-Maruyama step for each coefficient
            for j in 0..8 {
                // Generate standard normal using Box-Muller
                let u1: f64 = fastrand::f64().max(1e-10);
                let u2: f64 = fastrand::f64();
                let dw = (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos() * sqrt_dt;

                // GBM update: X(t+dt) = X(t) + μX(t)dt + σX(t)dW
                current[j] += self.mu * current[j] * dt + self.sigma * current[j] * dw;
            }

            path.push(t);
            path.extend_from_slice(&current);
        }

        Ok(path)
    }

    /// Get drift coefficient
    #[wasm_bindgen(js_name = getMu)]
    pub fn get_mu(&self) -> f64 {
        self.mu
    }

    /// Get diffusion coefficient
    #[wasm_bindgen(js_name = getSigma)]
    pub fn get_sigma(&self) -> f64 {
        self.sigma
    }

    /// Compute expected value at time t given initial value
    ///
    /// E[X(t)] = X(0) * exp(μt)
    #[wasm_bindgen(js_name = expectedValue)]
    pub fn expected_value(&self, initial: &[f64], t: f64) -> Result<Vec<f64>, JsValue> {
        if initial.len() != 8 {
            return Err(JsValue::from_str("Initial must have 8 coefficients"));
        }

        let factor = (self.mu * t).exp();
        Ok(initial.iter().map(|&x| x * factor).collect())
    }

    /// Compute variance at time t given initial value
    ///
    /// Var(X(t)) = X(0)² * exp(2μt) * (exp(σ²t) - 1)
    #[wasm_bindgen(js_name = variance)]
    pub fn variance(&self, initial: &[f64], t: f64) -> Result<Vec<f64>, JsValue> {
        if initial.len() != 8 {
            return Err(JsValue::from_str("Initial must have 8 coefficients"));
        }

        let factor = (2.0 * self.mu * t).exp() * ((self.sigma * self.sigma * t).exp() - 1.0);
        Ok(initial.iter().map(|&x| x * x * factor).collect())
    }
}

/// WASM wrapper for Wiener process (Brownian motion)
///
/// Standard Brownian motion W(t) with W(0) = 0
#[wasm_bindgen]
pub struct WasmWienerProcess {
    /// Dimension of the process
    dim: usize,
}

#[wasm_bindgen]
impl WasmWienerProcess {
    /// Create a new Wiener process
    ///
    /// # Arguments
    /// * `dim` - Dimension of the process (default 8 for multivector space)
    #[wasm_bindgen(constructor)]
    pub fn new(dim: Option<usize>) -> Self {
        Self {
            dim: dim.unwrap_or(8),
        }
    }

    /// Sample a path of the Wiener process
    ///
    /// # Arguments
    /// * `t_end` - End time
    /// * `num_steps` - Number of time steps
    ///
    /// Returns flat array: (num_steps + 1) * (dim + 1) values
    /// Each point is [time, w0, w1, ...]
    #[wasm_bindgen(js_name = samplePath)]
    pub fn sample_path(&self, t_end: f64, num_steps: usize) -> Vec<f64> {
        use std::f64::consts::PI;

        let dt = t_end / num_steps as f64;
        let sqrt_dt = dt.sqrt();

        let mut path = Vec::with_capacity((num_steps + 1) * (self.dim + 1));

        // Initial point W(0) = 0
        path.push(0.0); // time
        path.extend(std::iter::repeat_n(0.0, self.dim));

        let mut current = vec![0.0; self.dim];

        for i in 1..=num_steps {
            let t = i as f64 * dt;

            // Increment: W(t+dt) - W(t) ~ N(0, dt)
            for j in 0..self.dim {
                let u1: f64 = fastrand::f64().max(1e-10);
                let u2: f64 = fastrand::f64();
                let dw = (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos() * sqrt_dt;
                current[j] += dw;
            }

            path.push(t);
            path.extend_from_slice(&current);
        }

        path
    }

    /// Get dimension
    #[wasm_bindgen(js_name = getDim)]
    pub fn get_dim(&self) -> usize {
        self.dim
    }
}

// =============================================================================
// Grade-Projected Distributions
// =============================================================================

/// WASM wrapper for grade-projected distribution
///
/// A distribution that only operates on components of a specific grade.
#[wasm_bindgen]
pub struct WasmGradeProjectedDistribution {
    /// The grade (0=scalar, 1=vectors, 2=bivectors, 3=pseudoscalar for Cl(3,0,0))
    grade: usize,
    /// Mean for this grade's components
    mean: Vec<f64>,
    /// Standard deviations for this grade's components
    std_devs: Vec<f64>,
}

#[wasm_bindgen]
impl WasmGradeProjectedDistribution {
    /// Create a grade-projected distribution from a Gaussian
    ///
    /// # Arguments
    /// * `gaussian` - Source Gaussian distribution
    /// * `grade` - Grade to project onto (0, 1, 2, or 3 for Cl(3,0,0))
    #[wasm_bindgen(constructor)]
    pub fn new(
        gaussian: &WasmGaussianMultivector,
        grade: usize,
    ) -> Result<WasmGradeProjectedDistribution, JsValue> {
        if grade > 3 {
            return Err(JsValue::from_str(
                "Grade must be 0, 1, 2, or 3 for Cl(3,0,0)",
            ));
        }

        let full_mean = gaussian.get_mean();
        let full_std_devs = gaussian.get_std_devs();

        // Grade indices for Cl(3,0,0):
        // Grade 0: index 0 (scalar)
        // Grade 1: indices 1, 2, 3 (vectors e1, e2, e3)
        // Grade 2: indices 4, 5, 6 (bivectors e12, e13, e23)
        // Grade 3: index 7 (pseudoscalar e123)
        let indices: Vec<usize> = match grade {
            0 => vec![0],
            1 => vec![1, 2, 3],
            2 => vec![4, 5, 6],
            3 => vec![7],
            _ => unreachable!(),
        };

        let mean: Vec<f64> = indices.iter().map(|&i| full_mean[i]).collect();
        let std_devs: Vec<f64> = indices.iter().map(|&i| full_std_devs[i]).collect();

        Ok(Self {
            grade,
            mean,
            std_devs,
        })
    }

    /// Get the grade
    #[wasm_bindgen(js_name = getGrade)]
    pub fn get_grade(&self) -> usize {
        self.grade
    }

    /// Get number of components in this grade
    #[wasm_bindgen(js_name = getNumComponents)]
    pub fn get_num_components(&self) -> usize {
        self.mean.len()
    }

    /// Get the mean for this grade's components
    #[wasm_bindgen(js_name = getMean)]
    pub fn get_mean(&self) -> Vec<f64> {
        self.mean.clone()
    }

    /// Get standard deviations
    #[wasm_bindgen(js_name = getStdDevs)]
    pub fn get_std_devs(&self) -> Vec<f64> {
        self.std_devs.clone()
    }

    /// Sample from this grade-projected distribution
    ///
    /// Returns components for just this grade
    pub fn sample(&self) -> Vec<f64> {
        use std::f64::consts::PI;

        self.mean
            .iter()
            .zip(&self.std_devs)
            .map(|(&m, &s)| {
                let u1: f64 = fastrand::f64().max(1e-10);
                let u2: f64 = fastrand::f64();
                let z = (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos();
                m + s * z
            })
            .collect()
    }

    /// Sample and embed into full multivector (other grades are zero)
    ///
    /// Returns 8 coefficients
    #[wasm_bindgen(js_name = sampleFull)]
    pub fn sample_full(&self) -> Vec<f64> {
        let grade_sample = self.sample();
        let mut full = vec![0.0; 8];

        let indices: Vec<usize> = match self.grade {
            0 => vec![0],
            1 => vec![1, 2, 3],
            2 => vec![4, 5, 6],
            3 => vec![7],
            _ => unreachable!(),
        };

        for (i, &idx) in indices.iter().enumerate() {
            full[idx] = grade_sample[i];
        }

        full
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
