//! Noise-induced transitions
//!
//! This module provides tools for analyzing transitions between attractors
//! induced by stochastic perturbations.
//!
//! # Key Concepts
//!
//! ## Kramers Problem
//!
//! For a particle in a double-well potential V(x) with noise intensity D,
//! the mean first passage time (escape time) from one well to another is:
//!
//! ```text
//! τ ≈ (2π)/(ω_well * ω_barrier) * exp(ΔV/D)
//! ```
//!
//! where ΔV is the barrier height.
//!
//! ## Stochastic Resonance
//!
//! Periodic forcing combined with noise can lead to enhanced signal detection
//! when the noise intensity matches the forcing period.
//!
//! ## First Passage Time
//!
//! The time for a stochastic trajectory to first reach a target region.
//!
//! # Example
//!
//! ```ignore
//! use amari_dynamics::stochastic::{TransitionAnalyzer, first_passage_time};
//! use amari_dynamics::systems::DuffingOscillator;
//!
//! let duffing = DuffingOscillator::double_well();
//! let analyzer = TransitionAnalyzer::new(duffing, 0.1);
//!
//! let mfpt = analyzer.mean_first_passage_time(initial, target, 1000)?;
//! ```

use amari_core::Multivector;
use rand::Rng;
use rand_distr::{Distribution, Normal};

use crate::error::{DynamicsError, Result};
use crate::flow::DynamicalSystem;

/// Configuration for transition analysis
#[derive(Debug, Clone)]
pub struct TransitionConfig {
    /// Noise intensity (diffusion coefficient)
    pub noise_intensity: f64,
    /// Time step for simulation
    pub dt: f64,
    /// Maximum simulation time
    pub max_time: f64,
    /// Number of samples for statistics
    pub n_samples: usize,
}

impl Default for TransitionConfig {
    fn default() -> Self {
        Self {
            noise_intensity: 0.1,
            dt: 0.01,
            max_time: 1000.0,
            n_samples: 100,
        }
    }
}

impl TransitionConfig {
    /// Create configuration with noise intensity
    pub fn with_noise(noise_intensity: f64) -> Self {
        Self {
            noise_intensity,
            ..Default::default()
        }
    }

    /// Set time step
    pub fn dt(mut self, dt: f64) -> Self {
        self.dt = dt;
        self
    }

    /// Set maximum simulation time
    pub fn max_time(mut self, max_time: f64) -> Self {
        self.max_time = max_time;
        self
    }

    /// Set number of samples
    pub fn n_samples(mut self, n_samples: usize) -> Self {
        self.n_samples = n_samples;
        self
    }
}

/// Region definition for first passage time calculations
#[derive(Debug, Clone)]
pub enum Region<const P: usize, const Q: usize, const R: usize> {
    /// Ball of given radius around center
    Ball {
        /// Center of the ball
        center: Multivector<P, Q, R>,
        /// Radius of the ball
        radius: f64,
    },
    /// Half-space: {x : n·(x - x0) > 0}
    HalfSpace {
        /// Reference point on the boundary
        point: Multivector<P, Q, R>,
        /// Component index for the normal direction
        normal_component: usize,
        /// Whether the region is the positive half-space
        positive: bool,
    },
    /// Axis-aligned box region
    Box {
        /// Minimum coordinates for each dimension
        min: Vec<f64>,
        /// Maximum coordinates for each dimension
        max: Vec<f64>,
    },
}

impl<const P: usize, const Q: usize, const R: usize> Region<P, Q, R> {
    /// Create a ball region
    pub fn ball(center: Multivector<P, Q, R>, radius: f64) -> Self {
        Self::Ball { center, radius }
    }

    /// Create a half-space region
    pub fn half_space(point: Multivector<P, Q, R>, component: usize, positive: bool) -> Self {
        Self::HalfSpace {
            point,
            normal_component: component,
            positive,
        }
    }

    /// Check if a state is in the region
    pub fn contains(&self, state: &Multivector<P, Q, R>) -> bool {
        match self {
            Region::Ball { center, radius } => {
                let diff_vec: Vec<f64> = state
                    .to_vec()
                    .iter()
                    .zip(center.to_vec().iter())
                    .map(|(a, b)| a - b)
                    .collect();
                let dist_sq: f64 = diff_vec.iter().map(|x| x * x).sum();
                dist_sq <= radius * radius
            }
            Region::HalfSpace {
                point,
                normal_component,
                positive,
            } => {
                let diff = state.get(*normal_component) - point.get(*normal_component);
                if *positive {
                    diff > 0.0
                } else {
                    diff < 0.0
                }
            }
            Region::Box { min, max } => {
                let state_vec = state.to_vec();
                for (i, &x) in state_vec.iter().enumerate() {
                    if i < min.len() && i < max.len() && (x < min[i] || x > max[i]) {
                        return false;
                    }
                }
                true
            }
        }
    }
}

/// Result of first passage time calculation
#[derive(Debug, Clone)]
pub struct FirstPassageResult {
    /// Mean first passage time
    pub mean_time: f64,
    /// Standard deviation of first passage time
    pub std_time: f64,
    /// Minimum observed time
    pub min_time: f64,
    /// Maximum observed time
    pub max_time: f64,
    /// Number of successful passages
    pub n_passages: usize,
    /// Number of trajectories that didn't reach target
    pub n_timeouts: usize,
    /// Individual passage times
    pub times: Vec<f64>,
}

impl FirstPassageResult {
    /// Compute from vector of passage times
    pub fn from_times(times: Vec<f64>, n_timeouts: usize) -> Self {
        if times.is_empty() {
            return Self {
                mean_time: f64::INFINITY,
                std_time: 0.0,
                min_time: f64::INFINITY,
                max_time: 0.0,
                n_passages: 0,
                n_timeouts,
                times,
            };
        }

        let n = times.len() as f64;
        let mean = times.iter().sum::<f64>() / n;
        let variance = times.iter().map(|t| (t - mean).powi(2)).sum::<f64>() / n;

        Self {
            mean_time: mean,
            std_time: variance.sqrt(),
            min_time: times.iter().cloned().fold(f64::INFINITY, f64::min),
            max_time: times.iter().cloned().fold(0.0, f64::max),
            n_passages: times.len(),
            n_timeouts,
            times,
        }
    }

    /// Success rate (fraction of trajectories that reached target)
    pub fn success_rate(&self) -> f64 {
        let total = self.n_passages + self.n_timeouts;
        if total == 0 {
            0.0
        } else {
            self.n_passages as f64 / total as f64
        }
    }

    /// Coefficient of variation
    pub fn cv(&self) -> f64 {
        if self.mean_time > 0.0 {
            self.std_time / self.mean_time
        } else {
            0.0
        }
    }
}

/// Analyzer for noise-induced transitions
#[derive(Debug, Clone)]
pub struct TransitionAnalyzer<S> {
    /// Underlying dynamical system
    pub system: S,
    /// Configuration
    pub config: TransitionConfig,
}

impl<S> TransitionAnalyzer<S> {
    /// Create new analyzer
    pub fn new(system: S, noise_intensity: f64) -> Self {
        Self {
            system,
            config: TransitionConfig::with_noise(noise_intensity),
        }
    }

    /// Create with full configuration
    pub fn with_config(system: S, config: TransitionConfig) -> Self {
        Self { system, config }
    }
}

impl<S> TransitionAnalyzer<S> {
    /// Compute mean first passage time to a target region
    pub fn mean_first_passage_time<RNG: Rng, const P: usize, const Q: usize, const R: usize>(
        &self,
        initial: &Multivector<P, Q, R>,
        target: &Region<P, Q, R>,
        rng: &mut RNG,
    ) -> Result<FirstPassageResult>
    where
        S: DynamicalSystem<P, Q, R>,
    {
        let mut passage_times = Vec::new();
        let mut n_timeouts = 0;

        let dim = 1 << (P + Q + R);
        let noise_amp = (2.0 * self.config.noise_intensity).sqrt();
        let normal = Normal::new(0.0, 1.0).map_err(|e| {
            DynamicsError::numerical_instability("MFPT", format!("RNG error: {}", e))
        })?;

        let sqrt_dt = self.config.dt.sqrt();
        let max_steps = (self.config.max_time / self.config.dt) as usize;

        for _ in 0..self.config.n_samples {
            let mut state = initial.clone();
            let mut reached = false;

            for step in 0..max_steps {
                // Check if target reached
                if target.contains(&state) {
                    passage_times.push(step as f64 * self.config.dt);
                    reached = true;
                    break;
                }

                // Euler-Maruyama step
                let drift = self.system.vector_field(&state)?;
                let state_vec = state.to_vec();
                let drift_vec = drift.to_vec();

                let mut new_state = Vec::with_capacity(dim);
                for i in 0..dim {
                    let dw = normal.sample(rng) * sqrt_dt;
                    new_state.push(state_vec[i] + drift_vec[i] * self.config.dt + noise_amp * dw);
                }

                state = Multivector::from_coefficients(new_state);
            }

            if !reached {
                n_timeouts += 1;
            }
        }

        Ok(FirstPassageResult::from_times(passage_times, n_timeouts))
    }

    /// Compute transition rate between two regions
    pub fn transition_rate<RNG: Rng, const P: usize, const Q: usize, const R: usize>(
        &self,
        source: &Region<P, Q, R>,
        target: &Region<P, Q, R>,
        initial: &Multivector<P, Q, R>,
        rng: &mut RNG,
    ) -> Result<f64>
    where
        S: DynamicalSystem<P, Q, R>,
    {
        // Transition rate ≈ 1 / MFPT
        let fpt = self.mean_first_passage_time(initial, target, rng)?;

        // Check that initial is in source region
        if !source.contains(initial) {
            return Err(DynamicsError::invalid_parameter(
                "Initial state must be in source region",
            ));
        }

        if fpt.mean_time > 0.0 && fpt.mean_time.is_finite() {
            Ok(1.0 / fpt.mean_time)
        } else {
            Ok(0.0)
        }
    }

    /// Count transitions between regions over a time interval
    /// Count transitions between regions over a time interval
    pub fn count_transitions<RNG: Rng, const P: usize, const Q: usize, const R: usize>(
        &self,
        initial: &Multivector<P, Q, R>,
        region_a: &Region<P, Q, R>,
        region_b: &Region<P, Q, R>,
        total_time: f64,
        rng: &mut RNG,
    ) -> Result<TransitionCounts>
    where
        S: DynamicalSystem<P, Q, R>,
    {
        let dim = 1 << (P + Q + R);
        let noise_amp = (2.0 * self.config.noise_intensity).sqrt();
        let normal = Normal::new(0.0, 1.0).map_err(|e| {
            DynamicsError::numerical_instability("Count transitions", format!("RNG error: {}", e))
        })?;

        let sqrt_dt = self.config.dt.sqrt();
        let n_steps = (total_time / self.config.dt) as usize;

        let mut state = initial.clone();
        let mut a_to_b = 0;
        let mut b_to_a = 0;
        let mut time_in_a = 0.0;
        let mut time_in_b = 0.0;

        let mut in_a = region_a.contains(&state);
        let mut in_b = region_b.contains(&state);

        for _ in 0..n_steps {
            // Euler-Maruyama step
            let drift = self.system.vector_field(&state)?;
            let state_vec = state.to_vec();
            let drift_vec = drift.to_vec();

            let mut new_state = Vec::with_capacity(dim);
            for i in 0..dim {
                let dw = normal.sample(rng) * sqrt_dt;
                new_state.push(state_vec[i] + drift_vec[i] * self.config.dt + noise_amp * dw);
            }
            state = Multivector::from_coefficients(new_state);

            let now_in_a = region_a.contains(&state);
            let now_in_b = region_b.contains(&state);

            // Count transitions
            if in_a && now_in_b && !in_b {
                a_to_b += 1;
            }
            if in_b && now_in_a && !in_a {
                b_to_a += 1;
            }

            // Accumulate residence times
            if now_in_a {
                time_in_a += self.config.dt;
            }
            if now_in_b {
                time_in_b += self.config.dt;
            }

            in_a = now_in_a;
            in_b = now_in_b;
        }

        Ok(TransitionCounts {
            a_to_b,
            b_to_a,
            time_in_a,
            time_in_b,
            total_time,
        })
    }
}

/// Counts of transitions between regions
#[derive(Debug, Clone)]
pub struct TransitionCounts {
    /// Number of A → B transitions
    pub a_to_b: usize,
    /// Number of B → A transitions
    pub b_to_a: usize,
    /// Total time spent in region A
    pub time_in_a: f64,
    /// Total time spent in region B
    pub time_in_b: f64,
    /// Total simulation time
    pub total_time: f64,
}

impl TransitionCounts {
    /// Total number of transitions
    pub fn total_transitions(&self) -> usize {
        self.a_to_b + self.b_to_a
    }

    /// Transition rate A → B
    pub fn rate_a_to_b(&self) -> f64 {
        if self.time_in_a > 0.0 {
            self.a_to_b as f64 / self.time_in_a
        } else {
            0.0
        }
    }

    /// Transition rate B → A
    pub fn rate_b_to_a(&self) -> f64 {
        if self.time_in_b > 0.0 {
            self.b_to_a as f64 / self.time_in_b
        } else {
            0.0
        }
    }

    /// Fraction of time in region A
    pub fn fraction_in_a(&self) -> f64 {
        self.time_in_a / self.total_time
    }

    /// Fraction of time in region B
    pub fn fraction_in_b(&self) -> f64 {
        self.time_in_b / self.total_time
    }
}

/// Estimate escape time using Kramers formula
///
/// For a system with drift f(x) = -dV/dx and barrier height ΔV,
/// the mean escape time is approximately:
///
/// τ ≈ (2π / (ω_min * ω_saddle)) * exp(ΔV / D)
///
/// # Arguments
///
/// * `barrier_height` - Height of the energy barrier ΔV
/// * `diffusion` - Diffusion coefficient D
/// * `omega_minimum` - Angular frequency at the potential minimum
/// * `omega_saddle` - Angular frequency at the saddle point
pub fn kramers_escape_time(
    barrier_height: f64,
    diffusion: f64,
    omega_minimum: f64,
    omega_saddle: f64,
) -> f64 {
    2.0 * std::f64::consts::PI / (omega_minimum * omega_saddle) * (barrier_height / diffusion).exp()
}

/// Estimate noise intensity required for given escape time
///
/// Inverse of Kramers formula: D = ΔV / ln(τ * ω_min * ω_saddle / 2π)
pub fn noise_for_escape_time(
    target_time: f64,
    barrier_height: f64,
    omega_minimum: f64,
    omega_saddle: f64,
) -> f64 {
    let ln_arg = target_time * omega_minimum * omega_saddle / (2.0 * std::f64::consts::PI);
    barrier_height / ln_arg.ln()
}

/// Detect residence times in a region
///
/// Returns vector of times spent in each visit to the region.
pub fn residence_times<S, RNG, const P: usize, const Q: usize, const R: usize>(
    system: &S,
    initial: &Multivector<P, Q, R>,
    region: &Region<P, Q, R>,
    noise_intensity: f64,
    total_time: f64,
    dt: f64,
    rng: &mut RNG,
) -> Result<Vec<f64>>
where
    S: DynamicalSystem<P, Q, R>,
    RNG: Rng,
{
    let dim = 1 << (P + Q + R);
    let noise_amp = (2.0 * noise_intensity).sqrt();
    let normal = Normal::new(0.0, 1.0)
        .map_err(|e| DynamicsError::numerical_instability("Residence times", e.to_string()))?;

    let sqrt_dt = dt.sqrt();
    let n_steps = (total_time / dt) as usize;

    let mut state = initial.clone();
    let mut times = Vec::new();
    let mut current_residence = 0.0;
    let mut was_inside = region.contains(&state);

    for _ in 0..n_steps {
        // Euler-Maruyama step
        let drift = system.vector_field(&state)?;
        let state_vec = state.to_vec();
        let drift_vec = drift.to_vec();

        let mut new_state = Vec::with_capacity(dim);
        for i in 0..dim {
            let dw = normal.sample(rng) * sqrt_dt;
            new_state.push(state_vec[i] + drift_vec[i] * dt + noise_amp * dw);
        }
        state = Multivector::from_coefficients(new_state);

        let is_inside = region.contains(&state);

        if is_inside {
            current_residence += dt;
        } else if was_inside {
            // Just left the region
            times.push(current_residence);
            current_residence = 0.0;
        }

        was_inside = is_inside;
    }

    // Don't forget last residence if we ended inside
    if was_inside && current_residence > 0.0 {
        times.push(current_residence);
    }

    Ok(times)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flow::HarmonicOscillator;
    use rand::SeedableRng;

    #[test]
    fn test_transition_config_default() {
        let config = TransitionConfig::default();
        assert_eq!(config.noise_intensity, 0.1);
        assert_eq!(config.n_samples, 100);
    }

    #[test]
    fn test_region_ball() {
        let center = Multivector::<2, 0, 0>::zero();
        let region = Region::ball(center, 1.0);

        let mut inside = Multivector::zero();
        inside.set(1, 0.5);
        assert!(region.contains(&inside));

        let mut outside = Multivector::zero();
        outside.set(1, 2.0);
        assert!(!region.contains(&outside));
    }

    #[test]
    fn test_region_half_space() {
        let point = Multivector::<2, 0, 0>::zero();
        let region = Region::half_space(point, 1, true);

        let mut positive = Multivector::zero();
        positive.set(1, 1.0);
        assert!(region.contains(&positive));

        let mut negative = Multivector::zero();
        negative.set(1, -1.0);
        assert!(!region.contains(&negative));
    }

    #[test]
    fn test_first_passage_result() {
        let times = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = FirstPassageResult::from_times(times, 2);

        assert_eq!(result.mean_time, 3.0);
        assert_eq!(result.min_time, 1.0);
        assert_eq!(result.max_time, 5.0);
        assert_eq!(result.n_passages, 5);
        assert_eq!(result.n_timeouts, 2);
        assert!((result.success_rate() - 5.0 / 7.0).abs() < 1e-10);
    }

    #[test]
    fn test_first_passage_time_simple() {
        let ho = HarmonicOscillator::new(1.0);
        let analyzer = TransitionAnalyzer::new(ho, 0.5);

        let mut initial = Multivector::<2, 0, 0>::zero();
        initial.set(1, 2.0); // Start away from origin

        let target = Region::ball(Multivector::zero(), 0.5);

        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let result = analyzer
            .mean_first_passage_time(&initial, &target, &mut rng)
            .unwrap();

        // Should have some successful passages
        assert!(result.n_passages > 0);
        assert!(result.mean_time > 0.0);
    }

    #[test]
    fn test_transition_counts() {
        let ho = HarmonicOscillator::new(1.0);
        let config = TransitionConfig::with_noise(0.5)
            .dt(0.01)
            .max_time(100.0)
            .n_samples(10);
        let analyzer = TransitionAnalyzer::with_config(ho, config);

        let initial = Multivector::<2, 0, 0>::zero();
        let region_a = Region::half_space(Multivector::zero(), 1, false); // x < 0
        let region_b = Region::half_space(Multivector::zero(), 1, true); // x > 0

        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let counts = analyzer
            .count_transitions(&initial, &region_a, &region_b, 50.0, &mut rng)
            .unwrap();

        // With noise, should see some transitions
        assert!(counts.total_transitions() > 0 || counts.time_in_a > 0.0 || counts.time_in_b > 0.0);
    }

    #[test]
    fn test_kramers_escape_time() {
        let barrier = 1.0;
        let diffusion = 0.1;
        let omega_min = 1.0;
        let omega_saddle = 1.0;

        let tau = kramers_escape_time(barrier, diffusion, omega_min, omega_saddle);

        // Should be exponentially large for barrier >> D
        assert!(tau > 1.0);

        // Check inverse
        let d_back = noise_for_escape_time(tau, barrier, omega_min, omega_saddle);
        assert!((d_back - diffusion).abs() < 1e-10);
    }

    #[test]
    fn test_residence_times() {
        let ho = HarmonicOscillator::new(1.0);
        let initial = Multivector::<2, 0, 0>::zero();
        let region = Region::ball(Multivector::zero(), 1.0);

        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let times = residence_times(&ho, &initial, &region, 0.5, 10.0, 0.01, &mut rng).unwrap();

        // Starting inside, should have at least one residence period
        assert!(!times.is_empty());
    }
}
