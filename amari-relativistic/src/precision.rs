//! High-precision arithmetic for relativistic physics calculations
//!
//! This module re-exports precision types from amari-core, providing
//! unified high-precision arithmetic for spacecraft orbital mechanics
//! and relativistic physics calculations.

// Re-export precision types from amari-core
pub use amari_core::precision::{ExtendedFloat, PrecisionFloat, StandardFloat};

#[cfg(any(
    feature = "high-precision",
    feature = "wasm-precision",
    feature = "native-precision"
))]
pub use amari_core::HighPrecisionFloat;

// Type aliases for relativistic physics
/// Standard precision for general relativistic calculations
pub type RelativisticFloat = StandardFloat;

/// Extended precision for critical orbital mechanics calculations
pub type OrbitalFloat = ExtendedFloat;
