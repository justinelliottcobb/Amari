//! Langevin dynamics
//!
//! Langevin dynamics describes the evolution of a system subject to both
//! deterministic forces and random thermal fluctuations.
//!
//! # Equations
//!
//! The overdamped Langevin equation:
//! ```text
//! dx/dt = f(x) + √(2D) ξ(t)
//! ```
//!
//! where:
//! - f(x) is the deterministic drift (from an underlying dynamical system)
//! - D is the diffusion coefficient (related to temperature: D = kT/γ)
//! - ξ(t) is white Gaussian noise
//!
//! The underdamped (full) Langevin equation includes inertia:
//! ```text
//! m d²x/dt² = -γ dx/dt + F(x) + √(2γkT) ξ(t)
//! ```
//!
//! # Applications
//!
//! - Brownian motion in a potential
//! - Molecular dynamics simulations
//! - Stochastic gradient descent (overdamped limit)
//! - Noise-induced transitions between attractors
//!
//! # Example
//!
//! ```ignore
//! use amari_dynamics::stochastic::LangevinSystem;
//! use amari_dynamics::systems::DuffingOscillator;
//!
//! let duffing = DuffingOscillator::double_well();
//! let langevin = LangevinSystem::new(duffing, 0.1);
//!
//! let mut rng = rand::thread_rng();
//! let trajectory = langevin.simulate(initial, 0.0, 100.0, 10000, &mut rng)?;
//! ```

use amari_core::Multivector;
use rand::Rng;
use rand_distr::{Distribution, Normal};

use crate::error::{DynamicsError, Result};
use crate::flow::DynamicalSystem;
use crate::solver::Trajectory;

/// Configuration for Langevin dynamics simulation
#[derive(Debug, Clone)]
pub struct LangevinConfig {
    /// Diffusion coefficient D (noise intensity)
    pub diffusion: f64,
    /// Temperature (if using physical units, D = kT/γ)
    pub temperature: Option<f64>,
    /// Friction coefficient γ (for underdamped dynamics)
    pub friction: Option<f64>,
    /// Whether to use overdamped approximation
    pub overdamped: bool,
}

impl Default for LangevinConfig {
    fn default() -> Self {
        Self {
            diffusion: 0.1,
            temperature: None,
            friction: None,
            overdamped: true,
        }
    }
}

impl LangevinConfig {
    /// Create configuration with given diffusion coefficient
    pub fn with_diffusion(diffusion: f64) -> Self {
        Self {
            diffusion,
            ..Default::default()
        }
    }

    /// Create configuration from temperature and friction
    pub fn from_temperature(temperature: f64, friction: f64) -> Self {
        Self {
            diffusion: temperature / friction, // D = kT/γ (k_B = 1)
            temperature: Some(temperature),
            friction: Some(friction),
            overdamped: true,
        }
    }

    /// Use underdamped (full) Langevin dynamics
    pub fn underdamped(mut self, friction: f64) -> Self {
        self.friction = Some(friction);
        self.overdamped = false;
        self
    }
}

/// Langevin system wrapping a deterministic dynamical system
///
/// Adds thermal noise to an existing dynamical system.
#[derive(Debug, Clone)]
pub struct LangevinSystem<S> {
    /// Underlying deterministic system
    pub system: S,
    /// Configuration
    pub config: LangevinConfig,
}

impl<S> LangevinSystem<S> {
    /// Create a new Langevin system with given diffusion
    pub fn new(system: S, diffusion: f64) -> Self {
        Self {
            system,
            config: LangevinConfig::with_diffusion(diffusion),
        }
    }

    /// Create a Langevin system with full configuration
    pub fn with_config(system: S, config: LangevinConfig) -> Self {
        Self { system, config }
    }

    /// Get reference to underlying system
    pub fn inner(&self) -> &S {
        &self.system
    }

    /// Get mutable reference to underlying system
    pub fn inner_mut(&mut self) -> &mut S {
        &mut self.system
    }

    /// Get the diffusion coefficient
    pub fn diffusion(&self) -> f64 {
        self.config.diffusion
    }

    /// Get noise amplitude √(2D)
    pub fn noise_amplitude(&self) -> f64 {
        (2.0 * self.config.diffusion).sqrt()
    }
}

impl<S> LangevinSystem<S> {
    /// Take a single Euler-Maruyama step
    pub fn step<RNG: Rng, const P: usize, const Q: usize, const R: usize>(
        &self,
        state: &Multivector<P, Q, R>,
        dt: f64,
        rng: &mut RNG,
    ) -> Result<Multivector<P, Q, R>>
    where
        S: DynamicalSystem<P, Q, R>,
    {
        let dim = 1 << (P + Q + R);

        // Compute deterministic drift
        let drift = self.system.vector_field(state)?;

        // Generate Brownian increments
        let sqrt_dt = dt.sqrt();
        let noise_amp = self.noise_amplitude();
        let normal = Normal::new(0.0, 1.0).map_err(|e| {
            DynamicsError::numerical_instability("Langevin step", format!("RNG error: {}", e))
        })?;

        let state_vec = state.to_vec();
        let drift_vec = drift.to_vec();

        let mut new_state = Vec::with_capacity(dim);

        for i in 0..dim {
            let dw = normal.sample(rng) * sqrt_dt;
            new_state.push(state_vec[i] + drift_vec[i] * dt + noise_amp * dw);
        }

        // Check for numerical instability
        for (i, &x) in new_state.iter().enumerate() {
            if !x.is_finite() {
                return Err(DynamicsError::numerical_instability(
                    "Langevin dynamics",
                    format!("Component {} became non-finite", i),
                ));
            }
        }

        Ok(Multivector::from_coefficients(new_state))
    }

    /// Simulate a trajectory
    pub fn simulate<RNG: Rng, const P: usize, const Q: usize, const R: usize>(
        &self,
        initial: Multivector<P, Q, R>,
        t0: f64,
        t1: f64,
        steps: usize,
        rng: &mut RNG,
    ) -> Result<LangevinTrajectory<P, Q, R>>
    where
        S: DynamicalSystem<P, Q, R>,
    {
        if steps == 0 {
            return Err(DynamicsError::invalid_parameter(
                "Number of steps must be positive",
            ));
        }

        let dt = (t1 - t0) / steps as f64;
        let mut states = Vec::with_capacity(steps + 1);
        let mut times = Vec::with_capacity(steps + 1);
        let mut state = initial;
        let mut t = t0;

        states.push(state.clone());
        times.push(t);

        for _ in 0..steps {
            state = self.step(&state, dt, rng)?;
            t += dt;
            states.push(state.clone());
            times.push(t);
        }

        Ok(LangevinTrajectory {
            trajectory: Trajectory {
                states,
                times,
                step_sizes: Some(vec![dt; steps]),
                error_estimates: None,
            },
            diffusion: self.config.diffusion,
        })
    }

    /// Generate an ensemble of trajectories
    pub fn ensemble<RNG: Rng, const P: usize, const Q: usize, const R: usize>(
        &self,
        initial: Multivector<P, Q, R>,
        t0: f64,
        t1: f64,
        steps: usize,
        n_trajectories: usize,
        rng: &mut RNG,
    ) -> Result<Vec<LangevinTrajectory<P, Q, R>>>
    where
        S: DynamicalSystem<P, Q, R>,
    {
        let mut trajectories = Vec::with_capacity(n_trajectories);

        for _ in 0..n_trajectories {
            trajectories.push(self.simulate(initial.clone(), t0, t1, steps, rng)?);
        }

        Ok(trajectories)
    }

    /// Compute ensemble average at final time
    pub fn ensemble_average<RNG: Rng, const P: usize, const Q: usize, const R: usize>(
        &self,
        initial: Multivector<P, Q, R>,
        t0: f64,
        t1: f64,
        steps: usize,
        n_samples: usize,
        rng: &mut RNG,
    ) -> Result<Multivector<P, Q, R>>
    where
        S: DynamicalSystem<P, Q, R>,
    {
        let dim = 1 << (P + Q + R);
        let mut sum = vec![0.0; dim];

        for _ in 0..n_samples {
            let traj = self.simulate(initial.clone(), t0, t1, steps, rng)?;
            let final_state = traj.trajectory.final_state().ok_or_else(|| {
                DynamicsError::numerical_instability("Ensemble average", "Empty trajectory")
            })?;
            for (i, v) in sum.iter_mut().enumerate().take(dim) {
                *v += final_state.get(i);
            }
        }

        for v in sum.iter_mut() {
            *v /= n_samples as f64;
        }

        Ok(Multivector::from_coefficients(sum))
    }

    /// Compute ensemble variance at final time
    pub fn ensemble_variance<RNG: Rng, const P: usize, const Q: usize, const R: usize>(
        &self,
        initial: Multivector<P, Q, R>,
        t0: f64,
        t1: f64,
        steps: usize,
        n_samples: usize,
        rng: &mut RNG,
    ) -> Result<Vec<f64>>
    where
        S: DynamicalSystem<P, Q, R>,
    {
        let dim = 1 << (P + Q + R);
        let mut sum = vec![0.0; dim];
        let mut sum_sq = vec![0.0; dim];

        for _ in 0..n_samples {
            let traj = self.simulate(initial.clone(), t0, t1, steps, rng)?;
            let final_state = traj.trajectory.final_state().ok_or_else(|| {
                DynamicsError::numerical_instability("Ensemble variance", "Empty trajectory")
            })?;
            for i in 0..dim {
                let x = final_state.get(i);
                sum[i] += x;
                sum_sq[i] += x * x;
            }
        }

        let n = n_samples as f64;
        let variance: Vec<f64> = sum
            .iter()
            .zip(sum_sq.iter())
            .map(|(&s, &s2)| s2 / n - (s / n).powi(2))
            .collect();

        Ok(variance)
    }
}

/// A trajectory from Langevin dynamics simulation
#[derive(Debug, Clone)]
pub struct LangevinTrajectory<const P: usize, const Q: usize, const R: usize> {
    /// Underlying trajectory data
    pub trajectory: Trajectory<P, Q, R>,
    /// Diffusion coefficient used
    pub diffusion: f64,
}

impl<const P: usize, const Q: usize, const R: usize> LangevinTrajectory<P, Q, R> {
    /// Get the final state
    pub fn final_state(&self) -> Option<&Multivector<P, Q, R>> {
        self.trajectory.final_state()
    }

    /// Get the initial state
    pub fn initial_state(&self) -> Option<&Multivector<P, Q, R>> {
        self.trajectory.states.first()
    }

    /// Get the number of points
    pub fn len(&self) -> usize {
        self.trajectory.states.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.trajectory.states.is_empty()
    }

    /// Get state at index
    pub fn state(&self, index: usize) -> Option<&Multivector<P, Q, R>> {
        self.trajectory.states.get(index)
    }

    /// Get time at index
    pub fn time(&self, index: usize) -> Option<f64> {
        self.trajectory.times.get(index).copied()
    }

    /// Iterator over (time, state) pairs
    pub fn iter(&self) -> impl Iterator<Item = (f64, &Multivector<P, Q, R>)> {
        self.trajectory
            .times
            .iter()
            .copied()
            .zip(self.trajectory.states.iter())
    }

    /// Compute mean squared displacement from initial position
    pub fn mean_squared_displacement(&self) -> Vec<f64> {
        let Some(initial) = self.initial_state() else {
            return vec![];
        };

        let initial_vec = initial.to_vec();
        let dim = initial_vec.len();

        self.trajectory
            .states
            .iter()
            .map(|state| {
                let state_vec = state.to_vec();
                let mut msd = 0.0;
                for i in 0..dim {
                    let dx = state_vec[i] - initial_vec[i];
                    msd += dx * dx;
                }
                msd
            })
            .collect()
    }
}

/// Underdamped Langevin dynamics (includes momentum)
///
/// Full Langevin equation:
/// ```text
/// dx/dt = p/m
/// dp/dt = F(x) - γp + √(2γkT) ξ(t)
/// ```
#[derive(Debug, Clone)]
pub struct UnderdampedLangevin<S> {
    /// Underlying force field (F = -∇V for potential V)
    pub force_system: S,
    /// Mass
    pub mass: f64,
    /// Friction coefficient γ
    pub friction: f64,
    /// Temperature (kT)
    pub temperature: f64,
}

impl<S> UnderdampedLangevin<S> {
    /// Create new underdamped Langevin system
    pub fn new(force_system: S, mass: f64, friction: f64, temperature: f64) -> Self {
        Self {
            force_system,
            mass,
            friction,
            temperature,
        }
    }

    /// Get noise amplitude √(2γkT)
    pub fn noise_amplitude(&self) -> f64 {
        (2.0 * self.friction * self.temperature).sqrt()
    }

    /// Get the critical damping friction
    pub fn critical_damping(&self, omega: f64) -> f64 {
        2.0 * self.mass * omega
    }

    /// Check if system is overdamped
    pub fn is_overdamped(&self, omega: f64) -> bool {
        self.friction > self.critical_damping(omega)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flow::HarmonicOscillator;
    use rand::SeedableRng;

    #[test]
    fn test_langevin_config_default() {
        let config = LangevinConfig::default();
        assert_eq!(config.diffusion, 0.1);
        assert!(config.overdamped);
    }

    #[test]
    fn test_langevin_config_from_temperature() {
        let config = LangevinConfig::from_temperature(1.0, 2.0);
        assert_eq!(config.diffusion, 0.5); // D = kT/γ = 1/2
        assert_eq!(config.temperature, Some(1.0));
        assert_eq!(config.friction, Some(2.0));
    }

    #[test]
    fn test_langevin_system_creation() {
        let ho = HarmonicOscillator::new(1.0);
        let langevin = LangevinSystem::new(ho, 0.5);
        assert_eq!(langevin.diffusion(), 0.5);
        assert!((langevin.noise_amplitude() - 1.0).abs() < 1e-10); // √(2*0.5) = 1
    }

    #[test]
    fn test_langevin_step() {
        let ho = HarmonicOscillator::new(1.0);
        let langevin = LangevinSystem::new(ho, 0.01);

        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let mut state = Multivector::<2, 0, 0>::zero();
        state.set(1, 1.0);

        let next = langevin.step(&state, 0.01, &mut rng).unwrap();

        // Should be close to original with small noise
        assert!((next.get(1) - 1.0).abs() < 0.5);
    }

    #[test]
    fn test_langevin_simulation() {
        let ho = HarmonicOscillator::new(1.0);
        let langevin = LangevinSystem::new(ho, 0.01);

        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let mut initial = Multivector::<2, 0, 0>::zero();
        initial.set(1, 1.0);

        let traj = langevin.simulate(initial, 0.0, 1.0, 100, &mut rng).unwrap();

        assert_eq!(traj.len(), 101);
        assert!((traj.time(0).unwrap() - 0.0).abs() < 1e-10);
        assert!((traj.time(100).unwrap() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_langevin_ensemble() {
        let ho = HarmonicOscillator::new(1.0);
        let langevin = LangevinSystem::new(ho, 0.1);

        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let initial = Multivector::<2, 0, 0>::zero();

        let ensemble = langevin
            .ensemble(initial, 0.0, 1.0, 100, 10, &mut rng)
            .unwrap();

        assert_eq!(ensemble.len(), 10);
    }

    #[test]
    fn test_mean_squared_displacement() {
        let ho = HarmonicOscillator::new(1.0);
        let langevin = LangevinSystem::new(ho, 0.1);

        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let initial = Multivector::<2, 0, 0>::zero();

        let traj = langevin.simulate(initial, 0.0, 1.0, 100, &mut rng).unwrap();
        let msd = traj.mean_squared_displacement();

        assert_eq!(msd.len(), 101);
        assert_eq!(msd[0], 0.0); // MSD at t=0 should be 0
    }

    #[test]
    fn test_underdamped_langevin() {
        let ho = HarmonicOscillator::new(1.0);
        let underdamped = UnderdampedLangevin::new(ho, 1.0, 0.5, 1.0);

        assert_eq!(underdamped.mass, 1.0);
        assert_eq!(underdamped.friction, 0.5);
        assert_eq!(underdamped.temperature, 1.0);
        assert!((underdamped.noise_amplitude() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_overdamped_check() {
        let ho = HarmonicOscillator::new(1.0);

        // Underdamped case
        let underdamped = UnderdampedLangevin::new(ho, 1.0, 0.5, 1.0);
        assert!(!underdamped.is_overdamped(1.0)); // γ=0.5 < 2mω=2

        // Overdamped case
        let overdamped = UnderdampedLangevin::new(ho, 1.0, 3.0, 1.0);
        assert!(overdamped.is_overdamped(1.0)); // γ=3 > 2mω=2
    }
}
