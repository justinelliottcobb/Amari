//! Stochastic dynamics
//!
//! This module provides tools for analyzing dynamical systems subject to
//! random perturbations, including Langevin dynamics, Fokker-Planck equations,
//! and noise-induced transitions.
//!
//! # Overview
//!
//! Stochastic dynamics extends deterministic systems by adding noise:
//!
//! ```text
//! Deterministic: dx/dt = f(x)
//! Stochastic:    dx = f(x)dt + √(2D) dW
//! ```
//!
//! where D is the diffusion coefficient (noise intensity) and dW is a Wiener
//! process increment.
//!
//! # Key Concepts
//!
//! ## Langevin Dynamics
//!
//! Describes a particle subject to deterministic forces and thermal fluctuations.
//! In the overdamped limit, inertia is neglected and the equation becomes:
//!
//! ```text
//! dx/dt = f(x) + √(2D) ξ(t)
//! ```
//!
//! where D = kT/γ (thermal energy divided by friction).
//!
//! ## Fokker-Planck Equation
//!
//! The probability density P(x, t) evolves according to:
//!
//! ```text
//! ∂P/∂t = -∇·(f(x)P) + D∇²P
//! ```
//!
//! For gradient systems f(x) = -∇V(x), the stationary distribution is
//! the Boltzmann distribution P_s ∝ exp(-V/D).
//!
//! ## Noise-Induced Transitions
//!
//! Stochastic perturbations can cause transitions between attractors.
//! The Kramers escape rate from a potential well is:
//!
//! ```text
//! r ∝ exp(-ΔV/D)
//! ```
//!
//! where ΔV is the barrier height.
//!
//! # Modules
//!
//! - [`langevin`]: Langevin dynamics simulation
//! - [`fokker_planck`]: Fokker-Planck equation solvers
//! - [`noise_induced`]: Noise-induced transition analysis
//!
//! # Example
//!
//! ```ignore
//! use amari_dynamics::stochastic::{LangevinSystem, FokkerPlanck1D};
//! use amari_dynamics::systems::DuffingOscillator;
//!
//! // Create a noisy Duffing oscillator
//! let duffing = DuffingOscillator::double_well();
//! let langevin = LangevinSystem::new(duffing, 0.1);
//!
//! // Simulate an ensemble of trajectories
//! let mut rng = rand::thread_rng();
//! let ensemble = langevin.ensemble(initial, 0.0, 100.0, 10000, 100, &mut rng)?;
//!
//! // Solve the corresponding Fokker-Planck equation
//! let config = FokkerPlanckConfig::new_1d(-3.0, 3.0, 200, 0.1);
//! let mut fp = FokkerPlanck1D::new(config)?;
//! fp.init_gaussian(1.0, 0.5);
//! fp.evolve(|x| x - x.powi(3), 0.001, 10000)?;
//! ```
//!
//! # Feature Flag
//!
//! This module requires the `stochastic` feature flag:
//!
//! ```toml
//! [dependencies]
//! amari-dynamics = { version = "0.16", features = ["stochastic"] }
//! ```

mod fokker_planck;
mod langevin;
mod noise_induced;

// Re-export Langevin dynamics
pub use langevin::{LangevinConfig, LangevinSystem, LangevinTrajectory, UnderdampedLangevin};

// Re-export Fokker-Planck
pub use fokker_planck::{
    kramers_rate, BoundaryCondition, FokkerPlanck1D, FokkerPlanck2D, FokkerPlanckConfig,
};

// Re-export noise-induced transitions
pub use noise_induced::{
    kramers_escape_time, noise_for_escape_time, residence_times, FirstPassageResult, Region,
    TransitionAnalyzer, TransitionConfig, TransitionCounts,
};

// Re-export from amari-probabilistic if available
#[cfg(feature = "stochastic")]
pub use amari_probabilistic::stochastic::{
    EulerMaruyama, GeometricBrownianMotion, SDESolver, StochasticProcess, WienerProcess,
};
