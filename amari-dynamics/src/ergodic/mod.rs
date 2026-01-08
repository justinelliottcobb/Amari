//! Ergodic theory for dynamical systems
//!
//! This module provides tools for studying the statistical properties of
//! dynamical systems through ergodic theory.
//!
//! # Overview
//!
//! Ergodic theory studies the long-term average behavior of dynamical systems.
//! Key concepts include:
//!
//! - **Invariant measures**: Probability measures preserved by the dynamics
//! - **Birkhoff averages**: Time averages along trajectories
//! - **Ergodicity**: When time averages equal space averages
//! - **Mixing**: When correlations decay over time
//!
//! # Birkhoff Ergodic Theorem
//!
//! For an ergodic system with invariant measure μ:
//!
//! ```text
//! lim (1/T) ∫₀ᵀ f(φₜ(x)) dt = ∫ f dμ
//! ```
//!
//! holds for almost all initial conditions x.
//!
//! # Example
//!
//! ```ignore
//! use amari_dynamics::ergodic::{birkhoff_average, test_ergodicity, BirkhoffConfig};
//!
//! // Test if system is ergodic
//! let test = test_ergodicity(&system, &initial_conditions, observable, &config, 0.05)?;
//! if test.is_ergodic {
//!     println!("System appears ergodic with mean = {}", test.mean_average);
//! }
//! ```

pub mod birkhoff;
pub mod measure;

// Re-export main types and functions
pub use birkhoff::{
    birkhoff_average, birkhoff_average_ensemble_seq, birkhoff_averages,
    compute_average_from_trajectory, observables, test_ergodicity, BirkhoffConfig, BirkhoffResult,
    ErgodicityTest,
};

pub use measure::{
    compute_histogram_measure, compute_invariant_measure, kl_divergence, total_variation_distance,
    wasserstein_distance_1d, EmpiricalMeasure, HistogramMeasure, InvariantMeasure, MeasureConfig,
};

#[cfg(feature = "parallel")]
pub use birkhoff::birkhoff_average_ensemble;

#[cfg(feature = "parallel")]
pub use measure::compute_invariant_measure_parallel;
