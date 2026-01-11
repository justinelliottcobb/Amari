//! Implicit ODE solvers for stiff systems
//!
//! This module provides implicit integration methods suitable for stiff
//! ordinary differential equations. Stiff systems have components evolving
//! on very different timescales, causing explicit methods to require
//! prohibitively small step sizes for stability.
//!
//! # When to Use Implicit Solvers
//!
//! - Systems with widely separated timescales (e.g., chemical reactions)
//! - Problems where explicit methods require extremely small steps
//! - Dissipative systems with fast relaxation
//! - Parabolic PDEs discretized as ODEs
//!
//! # Available Methods
//!
//! - [`BackwardEuler`] - First-order implicit method, unconditionally stable
//!
//! # Mathematical Background
//!
//! Implicit methods solve equations of the form:
//!
//! ```text
//! y_{n+1} = y_n + h * f(t_{n+1}, y_{n+1})  (Backward Euler)
//! ```
//!
//! This requires solving a nonlinear system at each step, typically via
//! Newton iteration:
//!
//! ```text
//! G(y) = y - y_n - h*f(t_{n+1}, y) = 0
//! y^{k+1} = y^k - [I - h*J]^{-1} * G(y^k)
//! ```
//!
//! where J = ∂f/∂y is the Jacobian.
//!
//! # Geometric Algebra Context
//!
//! For systems on Clifford algebras, the Jacobian is computed component-wise
//! and the Newton iteration preserves the multivector structure.

mod backward_euler;

pub use backward_euler::BackwardEuler;
