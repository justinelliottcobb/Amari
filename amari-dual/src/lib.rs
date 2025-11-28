//! Dual number automatic differentiation for efficient gradient computation
//!
//! Dual numbers extend real numbers with an infinitesimal unit ε where ε² = 0.
//! This allows for exact computation of derivatives without numerical approximation
//! or computational graphs, making it ideal for forward-mode automatic differentiation.

#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

// Import precision types from amari-core
#[cfg(feature = "high-precision")]
pub use amari_core::HighPrecisionFloat;
pub use amari_core::{ExtendedFloat, PrecisionFloat, StandardFloat};

// Core dual number types
pub mod types;

// Domain modules
pub mod error;
pub mod functions;
pub mod multivector;

#[cfg(feature = "phantom-types")]
pub mod verified;

#[cfg(feature = "contracts")]
pub mod verified_contracts;

// Re-export error types
pub use error::{DualError, DualResult};

// Re-export core types
pub use multivector::{DualMultivector, MultiDualMultivector};
pub use types::{DualNumber, MultiDualNumber, StandardDual, StandardMultiDual};

#[cfg(feature = "high-precision")]
pub use types::{ExtendedDual, ExtendedMultiDual};

// GPU acceleration exports
