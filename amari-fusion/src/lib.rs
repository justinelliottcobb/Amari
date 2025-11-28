//! Tropical-Dual-Clifford fusion system for optimal LLM evaluation
//!
//! This crate combines three exotic number systems:
//! - Tropical algebra: Converts expensive softmax operations to max operations
//! - Dual numbers: Provides automatic differentiation without computational graphs
//! - Clifford algebra: Handles geometric relationships and rotations
//!
//! Together, these systems create a powerful framework for efficient neural network
//! evaluation and optimization.

#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

// Re-export precision types from all constituent crates
pub use amari_core::{ExtendedFloat, PrecisionFloat, StandardFloat};
pub use amari_dual::{ExtendedDual, ExtendedMultiDual, StandardDual, StandardMultiDual};
pub use amari_tropical::{ExtendedTropical, StandardTropical};

#[cfg(feature = "high-precision")]
pub use amari_core::HighPrecisionFloat;

// Core fusion types
pub mod types;

// Domain modules
pub mod attention;
pub mod evaluation;
pub mod optimizer;

// Verification modules
pub mod verified;
pub mod verified_contracts;

// Re-export core types
pub use types::{EvaluationError, EvaluationResult, TropicalDualClifford};

// GPU acceleration exports
