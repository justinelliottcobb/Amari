//! ODE solvers for dynamical systems
//!
//! This module provides numerical integration methods for solving ordinary
//! differential equations defined on geometric algebra state spaces.
//!
//! # Available Solvers
//!
//! ## Explicit Methods (Fixed Step)
//!
//! - [`ForwardEuler`] - First-order, single evaluation per step
//! - [`MidpointMethod`] - Second-order (RK2), two evaluations per step
//! - [`RungeKutta4`] - Fourth-order, four evaluations per step
//!
//! ## Adaptive Methods
//!
//! - [`RungeKuttaFehlberg45`] - 4(5) embedded pair with automatic step size control
//! - [`DormandPrince`] - 5(4) embedded pair with FSAL property (more efficient)
//!
//! ## Choosing a Solver
//!
//! - **[`RungeKutta4`]**: Good for non-stiff problems with smooth solutions
//! - **[`RungeKuttaFehlberg45`]**: Automatic error control, good general choice
//! - **[`DormandPrince`]**: Recommended for most problems - more efficient than RKF45
//!   due to FSAL (First Same As Last) property
//!
//! ## Implicit Methods (for stiff systems)
//!
//! - [`BackwardEuler`] - First-order, unconditionally stable, good for stiff problems
//!
//! Use implicit methods when explicit solvers require prohibitively small steps
//! (e.g., systems with fast/slow mode separation, chemical kinetics).
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

mod dopri;
pub mod implicit;
mod rk4;
mod rkf45;
mod traits;

// Re-export main types
pub use dopri::{DormandPrince, DormandPrinceDense};
pub use implicit::BackwardEuler;
pub use rk4::{ForwardEuler, MidpointMethod, RungeKutta4};
pub use rkf45::{RKF45WithConfig, RungeKuttaFehlberg45};
pub use traits::{
    AdaptiveConfig, AdaptiveODESolver, ImplicitODESolver, ODESolver, StepResult, Trajectory,
};
