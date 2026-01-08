//! ODE solvers for dynamical systems
//!
//! This module provides numerical integration methods for solving ordinary
//! differential equations defined on geometric algebra state spaces.
//!
//! # Available Solvers
//!
//! ## Explicit Methods
//!
//! - [`ForwardEuler`] - First-order, single evaluation per step
//! - [`MidpointMethod`] - Second-order (RK2), two evaluations per step
//! - [`RungeKutta4`] - Fourth-order, four evaluations per step (recommended)
//!
//! ## Choosing a Solver
//!
//! For most non-stiff problems, [`RungeKutta4`] provides a good balance of
//! accuracy and efficiency. Use adaptive methods (coming in future updates)
//! for problems requiring error control.
//!
//! # Example
//!
//! ```ignore
//! use amari_dynamics::{
//!     flow::DynamicalSystem,
//!     solver::{ODESolver, RungeKutta4, Trajectory},
//! };
//!
//! let solver = RungeKutta4::new();
//! let trajectory = solver.solve(&system, initial_state, 0.0, 10.0, 1000)?;
//!
//! for (t, state) in trajectory.iter() {
//!     println!("t = {:.2}: {:?}", t, state);
//! }
//! ```

mod rk4;
mod traits;

// Re-export main types
pub use rk4::{ForwardEuler, MidpointMethod, RungeKutta4};
pub use traits::{
    AdaptiveConfig, AdaptiveODESolver, ImplicitODESolver, ODESolver, StepResult, Trajectory,
};
