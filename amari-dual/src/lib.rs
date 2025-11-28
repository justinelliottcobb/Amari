//! Dual number automatic differentiation for efficient gradient computation
//!
//! Dual numbers extend real numbers with an infinitesimal unit ε where ε² = 0.
//! This allows for exact computation of derivatives without numerical approximation
//! or computational graphs, making it ideal for forward-mode automatic differentiation.

#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;
use alloc::vec::Vec;
use core::ops::{Add, Div, Mul, Neg, Sub};
use num_traits::{Float, One, Zero};

// Import precision types from amari-core
#[cfg(feature = "high-precision")]
pub use amari_core::HighPrecisionFloat;
pub use amari_core::{ExtendedFloat, PrecisionFloat, StandardFloat};

pub mod comprehensive_tests;
pub mod error;
pub mod functions;
pub mod multivector;
pub mod verified;
pub mod verified_contracts;

// Re-export commonly used types
pub use error::{DualError, DualResult};
pub use multivector::{DualMultivector, MultiDualMultivector};

// GPU acceleration exports
