//! High-precision arithmetic traits for scientific computing
//!
//! This module provides a unified interface for different precision levels:
//! - Standard f64 for general use
//! - Arbitrary precision using rug::Float for critical calculations
//! - Configurable precision based on application requirements

use num_traits::{Float as NumFloat, FloatConst, FromPrimitive, One, ToPrimitive, Zero};

#[cfg(feature = "std")]
use std::fmt::{Debug, Display};

#[cfg(not(feature = "std"))]
use core::fmt::{Debug, Display};

/// Unified trait for floating-point arithmetic in scientific computing
///
/// This trait abstracts over different precision levels to allow
/// mathematical calculations with appropriate numerical accuracy.
pub trait PrecisionFloat:
    Clone
    + Debug
    + Display
    + PartialEq
    + PartialOrd
    + NumFloat
    + FloatConst
    + FromPrimitive
    + ToPrimitive
    + Zero
    + One
    + Send
    + Sync
    + 'static
{
    /// Create from f64 value
    fn from_f64(value: f64) -> Self;

    /// Convert to f64 (may lose precision)
    fn to_f64(self) -> f64;

    /// Square root with high precision
    fn sqrt_precise(self) -> Self;

    /// Power function with high precision
    fn powf_precise(self, exp: Self) -> Self;

    /// Trigonometric functions with high precision
    fn sin_precise(self) -> Self;
    /// Cosine function with high precision
    fn cos_precise(self) -> Self;
    /// Tangent function with high precision
    fn tan_precise(self) -> Self;

    /// Hyperbolic functions with high precision
    fn sinh_precise(self) -> Self;
    /// Hyperbolic cosine with high precision
    fn cosh_precise(self) -> Self;
    /// Hyperbolic tangent with high precision
    fn tanh_precise(self) -> Self;

    /// Natural logarithm with high precision
    fn ln_precise(self) -> Self;

    /// Exponential function with high precision
    fn exp_precise(self) -> Self;

    /// Absolute value
    fn abs_precise(self) -> Self;

    /// Machine epsilon for this precision level
    fn epsilon() -> Self;

    /// Recommended tolerance for numerical calculations
    fn default_tolerance() -> Self;

    /// Specialized tolerance for orbital mechanics calculations
    fn orbital_tolerance() -> Self;
}

/// Standard f64 implementation for general use
impl PrecisionFloat for f64 {
    #[inline]
    fn from_f64(value: f64) -> Self {
        value
    }

    #[inline]
    fn to_f64(self) -> f64 {
        self
    }

    #[inline]
    fn sqrt_precise(self) -> Self {
        self.sqrt()
    }

    #[inline]
    fn powf_precise(self, exp: Self) -> Self {
        self.powf(exp)
    }

    #[inline]
    fn sin_precise(self) -> Self {
        self.sin()
    }

    #[inline]
    fn cos_precise(self) -> Self {
        self.cos()
    }

    #[inline]
    fn tan_precise(self) -> Self {
        self.tan()
    }

    #[inline]
    fn sinh_precise(self) -> Self {
        self.sinh()
    }

    #[inline]
    fn cosh_precise(self) -> Self {
        self.cosh()
    }

    #[inline]
    fn tanh_precise(self) -> Self {
        self.tanh()
    }

    #[inline]
    fn ln_precise(self) -> Self {
        self.ln()
    }

    #[inline]
    fn exp_precise(self) -> Self {
        self.exp()
    }

    #[inline]
    fn abs_precise(self) -> Self {
        self.abs()
    }

    #[inline]
    fn epsilon() -> Self {
        f64::EPSILON
    }

    #[inline]
    fn default_tolerance() -> Self {
        1e-12
    }

    #[inline]
    fn orbital_tolerance() -> Self {
        1e-15
    }
}

/// High-precision wrapper around rug::Float for critical calculations
#[cfg(feature = "high-precision")]
#[derive(Clone, Debug)]
pub struct HighPrecisionFloat {
    /// The arbitrary precision floating point value
    value: rug::Float,
    /// Precision in bits
    precision: u32,
}

#[cfg(feature = "high-precision")]
impl HighPrecisionFloat {
    /// Create with specified precision (bits)
    pub fn with_precision(value: f64, precision: u32) -> Self {
        Self {
            value: rug::Float::with_val(precision, value),
            precision,
        }
    }

    /// Create with standard precision (128 bits)
    pub fn standard(value: f64) -> Self {
        Self::with_precision(value, 128)
    }

    /// Create with high precision (256 bits) for critical calculations
    pub fn high(value: f64) -> Self {
        Self::with_precision(value, 256)
    }

    /// Get the precision in bits
    pub fn precision(&self) -> u32 {
        self.precision
    }
}

#[cfg(feature = "high-precision")]
impl PartialEq for HighPrecisionFloat {
    fn eq(&self, other: &Self) -> bool {
        self.value == other.value
    }
}

#[cfg(feature = "high-precision")]
impl PartialOrd for HighPrecisionFloat {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.value.partial_cmp(&other.value)
    }
}

#[cfg(feature = "high-precision")]
impl Display for HighPrecisionFloat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.value)
    }
}

// TODO: Implement NumFloat and other required traits for HighPrecisionFloat
// This would require substantial implementation for full compatibility

/// Type alias for standard precision calculations
pub type StandardFloat = f64;

/// Type alias for high precision calculations when available
#[cfg(feature = "high-precision")]
pub type ExtendedFloat = HighPrecisionFloat;

/// Type alias using standard precision when high-precision unavailable
#[cfg(not(feature = "high-precision"))]
pub type ExtendedFloat = f64;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_standard_float_precision() {
        let x = 2.0_f64;
        assert!((x.sqrt_precise() - f64::sqrt(2.0)).abs() < f64::EPSILON);
        assert!((x.powf_precise(3.0) - 8.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_default_tolerance() {
        assert!(f64::default_tolerance() < f64::EPSILON.sqrt());
        assert!(f64::default_tolerance() > 0.0);
    }

    #[cfg(feature = "high-precision")]
    #[test]
    fn test_high_precision_creation() {
        let hp = HighPrecisionFloat::standard(std::f64::consts::PI);
        assert_eq!(hp.precision(), 128);

        let sp = HighPrecisionFloat::high(std::f64::consts::E);
        assert_eq!(sp.precision(), 256);
    }
}
