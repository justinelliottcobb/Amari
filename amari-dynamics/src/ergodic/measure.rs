//! Invariant measures for dynamical systems
//!
//! This module provides tools for computing and representing invariant measures,
//! which characterize the long-term statistical behavior of dynamical systems.
//!
//! # Overview
//!
//! An invariant measure μ satisfies μ(f⁻¹(A)) = μ(A) for all measurable sets A,
//! where f is the dynamics. Key concepts include:
//!
//! - **Physical measures**: Describe typical long-term behavior
//! - **SRB measures**: Sinai-Ruelle-Bowen measures for chaotic systems
//! - **Ergodic measures**: Time averages equal space averages
//!
//! # Numerical Approximation
//!
//! We approximate invariant measures by:
//! 1. Binning phase space into cells
//! 2. Tracking trajectory visits to each cell
//! 3. Normalizing visit frequencies
//!
//! # Example
//!
//! ```ignore
//! use amari_dynamics::ergodic::{InvariantMeasure, HistogramMeasure, MeasureConfig};
//!
//! let config = MeasureConfig::two_dimensional((-2.0, 2.0), (-2.0, 2.0), 100, 100);
//! let measure = HistogramMeasure::from_trajectory(&trajectory, &config);
//!
//! println!("Entropy: {:.4}", measure.entropy());
//! println!("Support fraction: {:.2}%", measure.support_fraction() * 100.0);
//! ```

use amari_core::Multivector;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

use crate::error::{DynamicsError, Result};
use crate::flow::DynamicalSystem;
use crate::solver::{DormandPrince, ODESolver, Trajectory};

/// Trait for invariant measures on phase space
pub trait InvariantMeasure<const P: usize, const Q: usize, const R: usize> {
    /// Evaluate the measure density at a point
    ///
    /// Returns the probability density (for continuous measures) or
    /// probability mass (for discrete approximations).
    fn density(&self, state: &Multivector<P, Q, R>) -> f64;

    /// Sample a point from the measure
    fn sample(&self) -> Option<Multivector<P, Q, R>>;

    /// Compute the entropy of the measure
    ///
    /// For discrete approximations: H = -Σ p_i log(p_i)
    fn entropy(&self) -> f64;

    /// Compute the integral of a function with respect to this measure
    ///
    /// ∫ f(x) dμ(x)
    fn integrate<F>(&self, f: F) -> f64
    where
        F: Fn(&Multivector<P, Q, R>) -> f64;

    /// Get the support of the measure (bounding box)
    fn support_bounds(&self) -> Vec<(f64, f64)>;

    /// Fraction of phase space with non-zero measure
    fn support_fraction(&self) -> f64;
}

/// Configuration for measure computation
#[derive(Debug, Clone)]
pub struct MeasureConfig {
    /// Number of bins in each dimension
    pub resolution: Vec<usize>,

    /// Bounds for each dimension: (min, max)
    pub bounds: Vec<(f64, f64)>,

    /// Which dimensions to track (indices into state space)
    /// If empty, tracks all dimensions
    pub tracked_dimensions: Vec<usize>,

    /// Minimum number of trajectory points for reliable statistics
    pub min_samples: usize,

    /// Whether to use kernel density estimation instead of histograms
    pub use_kde: bool,

    /// Bandwidth for kernel density estimation (if used)
    pub kde_bandwidth: f64,
}

impl Default for MeasureConfig {
    fn default() -> Self {
        Self {
            resolution: vec![50, 50],
            bounds: vec![(-1.0, 1.0), (-1.0, 1.0)],
            tracked_dimensions: Vec::new(),
            min_samples: 1000,
            use_kde: false,
            kde_bandwidth: 0.1,
        }
    }
}

impl MeasureConfig {
    /// Create a 2D measure configuration
    pub fn two_dimensional(
        x_range: (f64, f64),
        y_range: (f64, f64),
        x_resolution: usize,
        y_resolution: usize,
    ) -> Self {
        Self {
            resolution: vec![x_resolution, y_resolution],
            bounds: vec![x_range, y_range],
            tracked_dimensions: vec![0, 1],
            ..Default::default()
        }
    }

    /// Create configuration for a 1D projection
    pub fn one_dimensional(range: (f64, f64), resolution: usize, dimension: usize) -> Self {
        Self {
            resolution: vec![resolution],
            bounds: vec![range],
            tracked_dimensions: vec![dimension],
            ..Default::default()
        }
    }

    /// Get the total number of bins
    pub fn num_bins(&self) -> usize {
        self.resolution.iter().product()
    }

    /// Get the bin size in each dimension
    pub fn bin_sizes(&self) -> Vec<f64> {
        self.resolution
            .iter()
            .zip(self.bounds.iter())
            .map(|(&n, &(min, max))| (max - min) / n as f64)
            .collect()
    }

    /// Convert a state to bin indices
    pub fn state_to_indices<const P: usize, const Q: usize, const R: usize>(
        &self,
        state: &Multivector<P, Q, R>,
    ) -> Option<Vec<usize>> {
        let dims = if self.tracked_dimensions.is_empty() {
            (0..self.bounds.len()).collect()
        } else {
            self.tracked_dimensions.clone()
        };

        let mut indices = Vec::with_capacity(dims.len());

        for (i, &dim) in dims.iter().enumerate() {
            let val = state.get(dim);
            let (min, max) = self.bounds[i];

            if val < min || val > max {
                return None; // Outside bounds
            }

            let n = self.resolution[i];
            let idx = ((val - min) / (max - min) * n as f64).floor() as usize;
            indices.push(idx.min(n - 1));
        }

        Some(indices)
    }

    /// Convert bin indices to a flat index
    pub fn indices_to_flat(&self, indices: &[usize]) -> usize {
        let mut flat = 0;
        let mut stride = 1;

        for (i, &idx) in indices.iter().enumerate() {
            flat += idx * stride;
            stride *= self.resolution[i];
        }

        flat
    }

    /// Convert a flat index to bin indices
    pub fn flat_to_indices(&self, mut flat: usize) -> Vec<usize> {
        let mut indices = Vec::with_capacity(self.resolution.len());

        for &n in &self.resolution {
            indices.push(flat % n);
            flat /= n;
        }

        indices
    }

    /// Get the center of a bin given its indices
    pub fn bin_center<const P: usize, const Q: usize, const R: usize>(
        &self,
        indices: &[usize],
    ) -> Multivector<P, Q, R> {
        let dims = if self.tracked_dimensions.is_empty() {
            (0..self.bounds.len()).collect()
        } else {
            self.tracked_dimensions.clone()
        };

        let mut state = Multivector::<P, Q, R>::zero();

        for (i, (&idx, &dim)) in indices.iter().zip(dims.iter()).enumerate() {
            let (min, max) = self.bounds[i];
            let n = self.resolution[i];
            let center = min + (idx as f64 + 0.5) * (max - min) / n as f64;
            state.set(dim, center);
        }

        state
    }
}

/// Histogram-based approximation of an invariant measure
#[derive(Debug, Clone)]
pub struct HistogramMeasure<const P: usize, const Q: usize, const R: usize> {
    /// Configuration
    config: MeasureConfig,

    /// Bin counts
    pub(crate) counts: Vec<usize>,

    /// Total number of samples
    pub(crate) total_samples: usize,

    /// Normalized probabilities (cached)
    probabilities: Vec<f64>,
}

impl<const P: usize, const Q: usize, const R: usize> HistogramMeasure<P, Q, R> {
    /// Create an empty histogram measure
    pub fn new(config: MeasureConfig) -> Self {
        let num_bins = config.num_bins();
        Self {
            config,
            counts: vec![0; num_bins],
            total_samples: 0,
            probabilities: vec![0.0; num_bins],
        }
    }

    /// Create from a trajectory
    pub fn from_trajectory(trajectory: &Trajectory<P, Q, R>, config: &MeasureConfig) -> Self {
        let mut measure = Self::new(config.clone());

        for state in &trajectory.states {
            measure.add_sample(state);
        }

        measure.normalize();
        measure
    }

    /// Add a sample to the histogram
    pub fn add_sample(&mut self, state: &Multivector<P, Q, R>) {
        if let Some(indices) = self.config.state_to_indices(state) {
            let flat = self.config.indices_to_flat(&indices);
            self.counts[flat] += 1;
            self.total_samples += 1;
        }
    }

    /// Normalize the histogram to get probabilities
    pub fn normalize(&mut self) {
        if self.total_samples > 0 {
            let n = self.total_samples as f64;
            for (prob, &count) in self.probabilities.iter_mut().zip(self.counts.iter()) {
                *prob = count as f64 / n;
            }
        }
    }

    /// Get the configuration
    pub fn config(&self) -> &MeasureConfig {
        &self.config
    }

    /// Get the raw bin counts
    pub fn counts(&self) -> &[usize] {
        &self.counts
    }

    /// Get the total number of samples
    pub fn total_samples(&self) -> usize {
        self.total_samples
    }

    /// Get the probabilities
    pub fn probabilities(&self) -> &[f64] {
        &self.probabilities
    }

    /// Get the probability of a specific bin
    pub fn bin_probability(&self, indices: &[usize]) -> f64 {
        let flat = self.config.indices_to_flat(indices);
        self.probabilities[flat]
    }

    /// Find the bin with maximum probability (mode)
    pub fn mode(&self) -> Option<Multivector<P, Q, R>> {
        let max_idx = self
            .probabilities
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())?
            .0;

        let indices = self.config.flat_to_indices(max_idx);
        Some(self.config.bin_center(&indices))
    }

    /// Compute marginal distribution for a single dimension
    pub fn marginal(&self, dim_index: usize) -> Vec<f64> {
        let n = self.config.resolution[dim_index];
        let mut marginal = vec![0.0; n];

        for (flat, &prob) in self.probabilities.iter().enumerate() {
            let indices = self.config.flat_to_indices(flat);
            marginal[indices[dim_index]] += prob;
        }

        marginal
    }

    /// Compute correlation between two dimensions
    pub fn correlation(&self, dim1: usize, dim2: usize) -> f64 {
        // Get marginals
        let marg1 = self.marginal(dim1);
        let marg2 = self.marginal(dim2);

        // Compute means
        let mean1: f64 = marg1
            .iter()
            .enumerate()
            .map(|(i, p)| i as f64 * p)
            .sum::<f64>();
        let mean2: f64 = marg2
            .iter()
            .enumerate()
            .map(|(i, p)| i as f64 * p)
            .sum::<f64>();

        // Compute standard deviations
        let var1: f64 = marg1
            .iter()
            .enumerate()
            .map(|(i, p)| (i as f64 - mean1).powi(2) * p)
            .sum();
        let var2: f64 = marg2
            .iter()
            .enumerate()
            .map(|(i, p)| (i as f64 - mean2).powi(2) * p)
            .sum();

        let std1 = var1.sqrt();
        let std2 = var2.sqrt();

        if std1 < 1e-10 || std2 < 1e-10 {
            return 0.0;
        }

        // Compute covariance
        let mut cov = 0.0;
        for (flat, &prob) in self.probabilities.iter().enumerate() {
            let indices = self.config.flat_to_indices(flat);
            let x1 = indices[dim1] as f64 - mean1;
            let x2 = indices[dim2] as f64 - mean2;
            cov += x1 * x2 * prob;
        }

        cov / (std1 * std2)
    }
}

impl<const P: usize, const Q: usize, const R: usize> InvariantMeasure<P, Q, R>
    for HistogramMeasure<P, Q, R>
{
    fn density(&self, state: &Multivector<P, Q, R>) -> f64 {
        if let Some(indices) = self.config.state_to_indices(state) {
            let flat = self.config.indices_to_flat(&indices);
            let bin_volume: f64 = self.config.bin_sizes().iter().product();
            self.probabilities[flat] / bin_volume
        } else {
            0.0
        }
    }

    fn sample(&self) -> Option<Multivector<P, Q, R>> {
        // Simple rejection sampling or find first non-zero bin
        for (flat, &prob) in self.probabilities.iter().enumerate() {
            if prob > 0.0 {
                let indices = self.config.flat_to_indices(flat);
                return Some(self.config.bin_center(&indices));
            }
        }
        None
    }

    fn entropy(&self) -> f64 {
        let mut h = 0.0;
        for &p in &self.probabilities {
            if p > 0.0 {
                h -= p * p.ln();
            }
        }
        h
    }

    fn integrate<F>(&self, f: F) -> f64
    where
        F: Fn(&Multivector<P, Q, R>) -> f64,
    {
        let mut sum = 0.0;

        for (flat, &prob) in self.probabilities.iter().enumerate() {
            if prob > 0.0 {
                let indices = self.config.flat_to_indices(flat);
                let center: Multivector<P, Q, R> = self.config.bin_center(&indices);
                sum += f(&center) * prob;
            }
        }

        sum
    }

    fn support_bounds(&self) -> Vec<(f64, f64)> {
        self.config.bounds.clone()
    }

    fn support_fraction(&self) -> f64 {
        let non_zero = self.probabilities.iter().filter(|&&p| p > 0.0).count();
        non_zero as f64 / self.probabilities.len() as f64
    }
}

/// Empirical measure from a finite set of points
#[derive(Debug, Clone)]
pub struct EmpiricalMeasure<const P: usize, const Q: usize, const R: usize> {
    /// Sample points
    points: Vec<Multivector<P, Q, R>>,

    /// Weights for each point (default: uniform)
    weights: Vec<f64>,
}

impl<const P: usize, const Q: usize, const R: usize> EmpiricalMeasure<P, Q, R> {
    /// Create from a set of points with uniform weights
    pub fn from_points(points: Vec<Multivector<P, Q, R>>) -> Self {
        let n = points.len();
        let weights = if n > 0 {
            vec![1.0 / n as f64; n]
        } else {
            Vec::new()
        };

        Self { points, weights }
    }

    /// Create from a trajectory
    pub fn from_trajectory(trajectory: &Trajectory<P, Q, R>) -> Self {
        Self::from_points(trajectory.states.clone())
    }

    /// Create with custom weights
    pub fn with_weights(points: Vec<Multivector<P, Q, R>>, weights: Vec<f64>) -> Result<Self> {
        if points.len() != weights.len() {
            return Err(DynamicsError::DimensionMismatch {
                expected: points.len(),
                actual: weights.len(),
            });
        }

        // Normalize weights
        let sum: f64 = weights.iter().sum();
        let weights: Vec<f64> = weights.iter().map(|w| w / sum).collect();

        Ok(Self { points, weights })
    }

    /// Number of points
    pub fn len(&self) -> usize {
        self.points.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }

    /// Get the points
    pub fn points(&self) -> &[Multivector<P, Q, R>] {
        &self.points
    }

    /// Get the weights
    pub fn weights(&self) -> &[f64] {
        &self.weights
    }

    /// Compute the mean
    pub fn mean(&self) -> Multivector<P, Q, R> {
        let dim = 1 << (P + Q + R);
        let mut result = Multivector::<P, Q, R>::zero();

        for (point, &weight) in self.points.iter().zip(self.weights.iter()) {
            for i in 0..dim {
                result.set(i, result.get(i) + point.get(i) * weight);
            }
        }

        result
    }

    /// Compute variance in each component
    pub fn variance(&self) -> Vec<f64> {
        let dim = 1 << (P + Q + R);
        let mean = self.mean();

        let mut var = vec![0.0; dim];

        for (point, &weight) in self.points.iter().zip(self.weights.iter()) {
            for (i, v) in var.iter_mut().enumerate().take(dim) {
                let diff = point.get(i) - mean.get(i);
                *v += diff * diff * weight;
            }
        }

        var
    }
}

impl<const P: usize, const Q: usize, const R: usize> InvariantMeasure<P, Q, R>
    for EmpiricalMeasure<P, Q, R>
{
    fn density(&self, state: &Multivector<P, Q, R>) -> f64 {
        // For empirical measure, use kernel density estimation
        let bandwidth = 0.1;
        let dim = 1 << (P + Q + R);

        let mut density = 0.0;

        for (point, &weight) in self.points.iter().zip(self.weights.iter()) {
            let mut dist_sq = 0.0;
            for i in 0..dim {
                let diff = state.get(i) - point.get(i);
                dist_sq += diff * diff;
            }

            // Gaussian kernel
            let kernel = (-dist_sq / (2.0 * bandwidth * bandwidth)).exp();
            density += weight * kernel;
        }

        density / (2.0 * std::f64::consts::PI * bandwidth * bandwidth).powf(dim as f64 / 2.0)
    }

    fn sample(&self) -> Option<Multivector<P, Q, R>> {
        self.points.first().cloned()
    }

    fn entropy(&self) -> f64 {
        // Approximate entropy using k-nearest neighbor estimator
        // Simplified: use negative log of average density
        let mut h = 0.0;
        for (point, &weight) in self.points.iter().zip(self.weights.iter()) {
            let d = self.density(point);
            if d > 0.0 {
                h -= weight * d.ln();
            }
        }
        h
    }

    fn integrate<F>(&self, f: F) -> f64
    where
        F: Fn(&Multivector<P, Q, R>) -> f64,
    {
        self.points
            .iter()
            .zip(self.weights.iter())
            .map(|(p, &w)| f(p) * w)
            .sum()
    }

    fn support_bounds(&self) -> Vec<(f64, f64)> {
        if self.points.is_empty() {
            return Vec::new();
        }

        let dim = 1 << (P + Q + R);
        let mut bounds = vec![(f64::INFINITY, f64::NEG_INFINITY); dim];

        for point in &self.points {
            for (i, b) in bounds.iter_mut().enumerate().take(dim) {
                let val = point.get(i);
                if val < b.0 {
                    b.0 = val;
                }
                if val > b.1 {
                    b.1 = val;
                }
            }
        }

        bounds
    }

    fn support_fraction(&self) -> f64 {
        // For empirical measures, all points are in the support
        1.0
    }
}

/// Compute the Wasserstein distance between two measures (1D projection)
pub fn wasserstein_distance_1d(cdf1: &[f64], cdf2: &[f64]) -> f64 {
    if cdf1.len() != cdf2.len() {
        return f64::INFINITY;
    }

    let mut distance = 0.0;
    for (c1, c2) in cdf1.iter().zip(cdf2.iter()) {
        distance += (c1 - c2).abs();
    }

    distance / cdf1.len() as f64
}

/// Compute the Kullback-Leibler divergence D_KL(P || Q)
pub fn kl_divergence(p: &[f64], q: &[f64]) -> f64 {
    if p.len() != q.len() {
        return f64::INFINITY;
    }

    let mut kl = 0.0;
    for (&pi, &qi) in p.iter().zip(q.iter()) {
        if pi > 0.0 && qi > 0.0 {
            kl += pi * (pi / qi).ln();
        } else if pi > 0.0 {
            return f64::INFINITY;
        }
    }

    kl
}

/// Compute the total variation distance between two measures
pub fn total_variation_distance(p: &[f64], q: &[f64]) -> f64 {
    if p.len() != q.len() {
        return 1.0;
    }

    let mut tv = 0.0;
    for (&pi, &qi) in p.iter().zip(q.iter()) {
        tv += (pi - qi).abs();
    }

    0.5 * tv
}

/// Compute an invariant measure by trajectory sampling
///
/// This function integrates the system from the initial condition and builds
/// a histogram measure from the resulting trajectory.
///
/// # Arguments
///
/// * `system` - The dynamical system
/// * `initial` - Initial condition for trajectory
/// * `config` - Configuration for the measure
/// * `total_time` - Total integration time
/// * `transient_time` - Time to discard at the beginning
/// * `dt` - Integration time step
///
/// # Returns
///
/// A histogram measure approximating the invariant measure.
pub fn compute_invariant_measure<S, const P: usize, const Q: usize, const R: usize>(
    system: &S,
    initial: &Multivector<P, Q, R>,
    config: &MeasureConfig,
    total_time: f64,
    transient_time: f64,
    dt: f64,
) -> Result<HistogramMeasure<P, Q, R>>
where
    S: DynamicalSystem<P, Q, R>,
{
    let solver = DormandPrince::new();

    // First, integrate through transient
    let transient_steps = (transient_time / dt) as usize;
    let post_transient = if transient_steps > 0 {
        let trajectory = solver.solve(
            system,
            initial.clone(),
            0.0,
            transient_time,
            transient_steps,
        )?;
        trajectory.final_state().cloned().ok_or_else(|| {
            DynamicsError::numerical_instability(
                "compute_invariant_measure",
                "Empty transient trajectory",
            )
        })?
    } else {
        initial.clone()
    };

    // Now compute the sampling trajectory
    let sampling_time = total_time - transient_time;
    let sampling_steps = (sampling_time / dt) as usize;
    let trajectory = solver.solve(system, post_transient, 0.0, sampling_time, sampling_steps)?;

    // Build histogram from trajectory
    Ok(HistogramMeasure::from_trajectory(&trajectory, config))
}

/// Convenience function to compute a histogram measure from a trajectory
pub fn compute_histogram_measure<const P: usize, const Q: usize, const R: usize>(
    trajectory: &Trajectory<P, Q, R>,
    config: &MeasureConfig,
) -> HistogramMeasure<P, Q, R> {
    HistogramMeasure::from_trajectory(trajectory, config)
}

/// Compute invariant measure in parallel from multiple initial conditions
#[cfg(feature = "parallel")]
pub fn compute_invariant_measure_parallel<S, const P: usize, const Q: usize, const R: usize>(
    system: &S,
    initial_conditions: &[Multivector<P, Q, R>],
    config: &MeasureConfig,
    total_time: f64,
    transient_time: f64,
    dt: f64,
) -> Result<HistogramMeasure<P, Q, R>>
where
    S: DynamicalSystem<P, Q, R> + Sync,
{
    // Compute measures from each initial condition in parallel
    let measures: Vec<HistogramMeasure<P, Q, R>> = initial_conditions
        .par_iter()
        .filter_map(|ic| {
            compute_invariant_measure(system, ic, config, total_time, transient_time, dt).ok()
        })
        .collect();

    // Merge measures by summing counts
    if measures.is_empty() {
        return Err(DynamicsError::invalid_parameter("No measures computed"));
    }

    let mut combined = HistogramMeasure::new(config.clone());

    for measure in &measures {
        for (i, &count) in measure.counts().iter().enumerate() {
            combined.counts[i] += count;
        }
        combined.total_samples += measure.total_samples();
    }

    combined.normalize();
    Ok(combined)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_measure_config_2d() {
        let config = MeasureConfig::two_dimensional((-1.0, 1.0), (-1.0, 1.0), 10, 10);
        assert_eq!(config.num_bins(), 100);
        assert_eq!(config.bin_sizes(), vec![0.2, 0.2]);
    }

    #[test]
    fn test_state_to_indices() {
        let config = MeasureConfig::two_dimensional((0.0, 1.0), (0.0, 1.0), 10, 10);

        let mut state = Multivector::<2, 0, 0>::zero();
        state.set(0, 0.25);
        state.set(1, 0.75);

        let indices = config.state_to_indices(&state).unwrap();
        assert_eq!(indices[0], 2); // 0.25 in [0,1] with 10 bins -> bin 2
        assert_eq!(indices[1], 7); // 0.75 in [0,1] with 10 bins -> bin 7
    }

    #[test]
    fn test_histogram_measure() {
        let config = MeasureConfig::two_dimensional((0.0, 1.0), (0.0, 1.0), 10, 10);
        let mut measure = HistogramMeasure::<2, 0, 0>::new(config);

        // Add samples in the center
        for _ in 0..100 {
            let mut state = Multivector::<2, 0, 0>::zero();
            state.set(0, 0.5);
            state.set(1, 0.5);
            measure.add_sample(&state);
        }

        measure.normalize();

        // Check that center has high probability
        let center_prob = measure.bin_probability(&[5, 5]);
        assert!(center_prob > 0.5);

        // Entropy should be low (concentrated distribution)
        assert!(measure.entropy() < 1.0);
    }

    #[test]
    fn test_empirical_measure_mean() {
        let mut points = Vec::new();
        for i in 0..10 {
            let mut p = Multivector::<2, 0, 0>::zero();
            p.set(0, i as f64);
            p.set(1, (i * 2) as f64);
            points.push(p);
        }

        let measure = EmpiricalMeasure::from_points(points);
        let mean = measure.mean();

        assert!((mean.get(0) - 4.5).abs() < 1e-10);
        assert!((mean.get(1) - 9.0).abs() < 1e-10);
    }

    #[test]
    fn test_support_fraction() {
        let config = MeasureConfig::two_dimensional((0.0, 1.0), (0.0, 1.0), 10, 10);
        let mut measure = HistogramMeasure::<2, 0, 0>::new(config);

        // Add samples in only one bin
        let mut state = Multivector::<2, 0, 0>::zero();
        state.set(0, 0.5);
        state.set(1, 0.5);
        measure.add_sample(&state);
        measure.normalize();

        // Only 1 bin out of 100 should be occupied
        assert!((measure.support_fraction() - 0.01).abs() < 1e-10);
    }

    #[test]
    fn test_kl_divergence() {
        // Same distribution
        let p = vec![0.25, 0.25, 0.25, 0.25];
        let q = vec![0.25, 0.25, 0.25, 0.25];
        assert!((kl_divergence(&p, &q) - 0.0).abs() < 1e-10);

        // Different distributions
        let p = vec![0.5, 0.5, 0.0, 0.0];
        let q = vec![0.25, 0.25, 0.25, 0.25];
        assert!(kl_divergence(&p, &q) > 0.0);
    }

    #[test]
    fn test_total_variation() {
        // Same distribution
        let p = vec![0.25, 0.25, 0.25, 0.25];
        let q = vec![0.25, 0.25, 0.25, 0.25];
        assert!((total_variation_distance(&p, &q) - 0.0).abs() < 1e-10);

        // Completely different
        let p = vec![1.0, 0.0, 0.0, 0.0];
        let q = vec![0.0, 0.0, 0.0, 1.0];
        assert!((total_variation_distance(&p, &q) - 1.0).abs() < 1e-10);
    }
}
