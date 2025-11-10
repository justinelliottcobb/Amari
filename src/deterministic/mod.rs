//! Deterministic computation module for networked physics applications
//!
//! Provides bit-exact reproducibility across platforms (x86-64, ARM64, WASM32)
//! and compiler optimization levels for networked physics simulations.
//!
//! # Overview
//!
//! Standard IEEE 754 floating-point has platform-specific behaviors that break
//! determinism in networked applications. This module provides deterministic
//! alternatives through:
//!
//! - Fixed-iteration algorithms (no early termination)
//! - Disabled FMA (fused multiply-add) instructions
//! - Polynomial approximations for transcendentals
//!
//! # Performance
//!
//! Deterministic operations are ~10-20% slower than native f32 due to disabled
//! optimizations. For non-networked applications, use the standard types.
//!
//! # Example
//!
//! ```
//! # #[cfg(feature = "deterministic")]
//! # {
//! use amari::deterministic::*;
//! use amari::deterministic::ga2d::*;
//!
//! // Create deterministic ship orientation
//! let mut rotation = DetRotor2::IDENTITY;
//!
//! // Rotate by fixed amount (bit-exact on all platforms)
//! let delta = DetRotor2::from_angle(DetF32::from_f32(0.1));
//! rotation = rotation * delta;
//!
//! // Transform vectors
//! let thrust_dir = rotation.transform(DetVector2::X_AXIS);
//! # }
//! ```

mod approx; // Extends scalar with trig functions
pub mod ga2d;
mod scalar;

// Re-export primary types
pub use ga2d::{DetRotor2, DetVector2};
pub use scalar::{DetF32, Deterministic};
