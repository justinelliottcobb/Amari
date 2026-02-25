//! GF(2) scalar type — the Galois field with two elements.

use core::fmt;
use core::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

/// Element of the Galois field GF(2) = {0, 1}.
///
/// The smallest finite field. Addition is XOR, multiplication is AND.
/// Every nonzero element is its own inverse (both additive and multiplicative).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(transparent)]
pub struct GF2(u8);

impl GF2 {
    pub const ZERO: Self = GF2(0);
    pub const ONE: Self = GF2(1);

    /// Create from a u8 value (reduces mod 2).
    #[inline]
    #[must_use]
    pub const fn new(value: u8) -> Self {
        GF2(value & 1)
    }

    /// Create from a bool.
    #[inline]
    #[must_use]
    pub const fn from_bool(value: bool) -> Self {
        GF2(value as u8)
    }

    /// The underlying value (0 or 1).
    #[inline]
    #[must_use]
    pub const fn value(self) -> u8 {
        self.0
    }

    /// Whether this is the zero element.
    #[inline]
    #[must_use]
    pub const fn is_zero_element(self) -> bool {
        self.0 == 0
    }

    /// Whether this is the one (unity) element.
    #[inline]
    #[must_use]
    pub const fn is_one(self) -> bool {
        self.0 == 1
    }

    /// Multiplicative inverse. Returns None for zero.
    #[inline]
    #[must_use]
    pub const fn inverse(self) -> Option<Self> {
        if self.0 == 1 {
            Some(self)
        } else {
            None
        }
    }
}

// --- Arithmetic trait impls ---

// GF(2) arithmetic: addition = XOR, multiplication = AND. These are intentional.
#[allow(clippy::suspicious_arithmetic_impl)]
impl Add for GF2 {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        GF2(self.0 ^ rhs.0)
    }
}

#[allow(clippy::suspicious_arithmetic_impl)]
impl Sub for GF2 {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        GF2(self.0 ^ rhs.0) // same as add in GF(2)
    }
}

#[allow(clippy::suspicious_arithmetic_impl)]
impl Mul for GF2 {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        GF2(self.0 & rhs.0)
    }
}

impl Neg for GF2 {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        self // -a = a in GF(2)
    }
}

#[allow(clippy::suspicious_op_assign_impl)]
impl AddAssign for GF2 {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        self.0 ^= rhs.0;
    }
}

#[allow(clippy::suspicious_op_assign_impl)]
impl SubAssign for GF2 {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        self.0 ^= rhs.0;
    }
}

#[allow(clippy::suspicious_op_assign_impl)]
impl MulAssign for GF2 {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }
}

// --- Conversion impls ---

impl From<bool> for GF2 {
    #[inline]
    fn from(value: bool) -> Self {
        GF2(value as u8)
    }
}

impl From<u8> for GF2 {
    #[inline]
    fn from(value: u8) -> Self {
        GF2(value & 1)
    }
}

impl From<GF2> for bool {
    #[inline]
    fn from(value: GF2) -> Self {
        value.0 != 0
    }
}

impl From<GF2> for u8 {
    #[inline]
    fn from(value: GF2) -> Self {
        value.0
    }
}

// --- Display ---

impl fmt::Display for GF2 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

// --- num_traits ---

impl num_traits::Zero for GF2 {
    #[inline]
    fn zero() -> Self {
        GF2::ZERO
    }
    #[inline]
    fn is_zero(&self) -> bool {
        self.0 == 0
    }
}

impl num_traits::One for GF2 {
    #[inline]
    fn one() -> Self {
        GF2::ONE
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_field_addition() {
        assert_eq!(GF2::ZERO + GF2::ZERO, GF2::ZERO);
        assert_eq!(GF2::ZERO + GF2::ONE, GF2::ONE);
        assert_eq!(GF2::ONE + GF2::ZERO, GF2::ONE);
        assert_eq!(GF2::ONE + GF2::ONE, GF2::ZERO);
    }

    #[test]
    fn test_field_multiplication() {
        assert_eq!(GF2::ZERO * GF2::ZERO, GF2::ZERO);
        assert_eq!(GF2::ZERO * GF2::ONE, GF2::ZERO);
        assert_eq!(GF2::ONE * GF2::ZERO, GF2::ZERO);
        assert_eq!(GF2::ONE * GF2::ONE, GF2::ONE);
    }

    #[test]
    fn test_distributivity() {
        for &a in &[GF2::ZERO, GF2::ONE] {
            for &b in &[GF2::ZERO, GF2::ONE] {
                for &c in &[GF2::ZERO, GF2::ONE] {
                    assert_eq!(a * (b + c), a * b + a * c);
                }
            }
        }
    }

    #[test]
    fn test_self_inverse() {
        // a + a = 0 for all a in GF(2)
        assert_eq!(GF2::ZERO + GF2::ZERO, GF2::ZERO);
        assert_eq!(GF2::ONE + GF2::ONE, GF2::ZERO);
        // -a = a
        assert_eq!(-GF2::ZERO, GF2::ZERO);
        assert_eq!(-GF2::ONE, GF2::ONE);
    }

    #[test]
    fn test_multiplicative_inverse() {
        assert_eq!(GF2::ONE.inverse(), Some(GF2::ONE));
        assert_eq!(GF2::ZERO.inverse(), None);
    }

    #[test]
    fn test_conversions() {
        assert_eq!(GF2::from(true), GF2::ONE);
        assert_eq!(GF2::from(false), GF2::ZERO);
        assert_eq!(GF2::new(5), GF2::ONE); // 5 & 1 = 1
        assert_eq!(GF2::new(4), GF2::ZERO); // 4 & 1 = 0
        assert!(bool::from(GF2::ONE));
        assert!(!bool::from(GF2::ZERO));
    }
}
