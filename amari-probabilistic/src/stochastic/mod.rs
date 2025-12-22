//! Stochastic processes on geometric algebra spaces
//!
//! This module provides stochastic differential equations (SDEs) and
//! stochastic processes for multivector-valued random variables.
//!
//! # Core Concepts
//!
//! A stochastic process X(t) on Cl(P,Q,R) satisfies an SDE:
//!
//! dX = μ(X, t)dt + σ(X, t)dW
//!
//! where:
//! - μ is the drift (multivector-valued)
//! - σ is the diffusion coefficient
//! - W is a Wiener process
//!
//! # Solvers
//!
//! - **Euler-Maruyama**: First-order SDE solver
//! - **Milstein**: Higher-order solver with gradient correction
//!
//! # Example
//!
//! ```ignore
//! use amari_probabilistic::stochastic::{GeometricBrownianMotion, StochasticProcess};
//!
//! // Geometric Brownian motion on multivector space
//! let gbm = GeometricBrownianMotion::<3, 0, 0>::new(0.1, 0.2);
//! let path = gbm.sample_path(0.0, 1.0, 100)?;
//! ```

mod brownian;
mod sde;

pub use brownian::{GeometricBrownianMotion, WienerProcess};
pub use sde::{EulerMaruyama, SDESolver, StochasticProcess};
