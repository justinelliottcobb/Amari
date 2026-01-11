//! Built-in dynamical systems
//!
//! This module provides implementations of well-known dynamical systems for
//! studying chaos, bifurcations, and nonlinear dynamics.
//!
//! # Continuous-Time Systems
//!
//! ## 2D Systems
//!
//! - [`VanDerPolOscillator`]: Self-sustained oscillator with limit cycle
//! - [`ForcedVanDerPol`]: Driven Van der Pol with potential chaos
//! - [`DuffingOscillator`]: Nonlinear oscillator with cubic stiffness
//! - [`ForcedDuffing`]: Driven Duffing with chaotic behavior
//! - [`SimplePendulum`]: Classic nonlinear pendulum
//! - [`DrivenPendulum`]: Driven pendulum with chaos
//!
//! ## 3D Systems
//!
//! - [`LorenzSystem`]: Famous chaotic attractor
//! - [`RosslerSystem`]: Simpler chaotic system with single lobe
//!
//! ## 4D Systems
//!
//! - [`DoublePendulum`]: Paradigmatic example of deterministic chaos
//!
//! # Discrete-Time Maps
//!
//! - [`HenonMap`]: 2D chaotic map
//! - [`LoziMap`]: Piecewise-linear version of Hénon
//! - [`GeneralizedHenon`]: Power-law Hénon variants
//!
//! # Example
//!
//! ```ignore
//! use amari_dynamics::systems::{LorenzSystem, VanDerPolOscillator, HenonMap};
//! use amari_dynamics::solver::{RungeKutta4, ODESolver};
//! use amari_dynamics::flow::DiscreteMap;
//!
//! // Continuous system
//! let lorenz = LorenzSystem::classic();
//! let solver = RungeKutta4::new();
//! let trajectory = solver.solve(&lorenz, lorenz.default_initial_condition(), 0.0, 100.0, 10000)?;
//!
//! // Discrete map
//! let henon = HenonMap::classic();
//! let orbit = henon.orbit(&henon.attractor_initial_condition(), 10000)?;
//! ```
//!
//! # System Properties
//!
//! Each system provides methods for:
//! - Creating common parameter configurations (e.g., `classic()`, `chaotic()`)
//! - Computing fixed points
//! - Default initial conditions on/near attractors
//! - Energy/Hamiltonian functions where applicable
//! - Jacobian matrices for stability analysis

mod duffing;
mod henon;
mod lorenz;
mod pendulum;
mod rossler;
mod vanderpol;

// Re-export continuous-time 2D systems
pub use duffing::{DuffingOscillator, ForcedDuffing};
pub use pendulum::{DrivenPendulum, SimplePendulum};
pub use vanderpol::{ForcedVanDerPol, VanDerPolOscillator};

// Re-export continuous-time 3D systems
pub use lorenz::LorenzSystem;
pub use rossler::RosslerSystem;

// Re-export continuous-time 4D systems
pub use pendulum::DoublePendulum;

// Re-export discrete maps
pub use henon::{GeneralizedHenon, HenonMap, LoziMap};
