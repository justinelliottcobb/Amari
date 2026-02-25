//! Bit-packed vectors over GF(2).

use super::scalar::GF2;
use alloc::vec;
use alloc::vec::Vec;
use core::fmt;
use core::ops::{Add, BitAnd, BitXor};

const WORD_BITS: usize = 64;

#[inline]
const fn word_index(i: usize) -> usize {
    i / WORD_BITS
}

#[inline]
const fn bit_index(i: usize) -> usize {
    i % WORD_BITS
}

#[inline]
fn num_words(dim: usize) -> usize {
    dim.div_ceil(WORD_BITS)
}

/// Mask for the valid bits in the last word (zeros out trailing bits beyond dim).
#[inline]
fn trailing_mask(dim: usize) -> u64 {
    let rem = dim % WORD_BITS;
    if rem == 0 && dim > 0 {
        u64::MAX
    } else if dim == 0 {
        0
    } else {
        (1u64 << rem) - 1
    }
}

/// A vector in F₂ⁿ, stored as packed bits in u64 words.
///
/// Operations are word-parallel: XOR for addition, AND + popcount for dot product.
/// Vectors of dimension ≤ 64 fit in a single word with no heap allocation overhead.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct GF2Vector {
    words: Vec<u64>,
    dim: usize,
}

impl GF2Vector {
    /// Create the zero vector of given dimension.
    #[must_use]
    pub fn zero(dim: usize) -> Self {
        Self {
            words: vec![0u64; num_words(dim)],
            dim,
        }
    }

    /// Create from a slice of 0/1 u8 values.
    #[must_use]
    pub fn from_bits(bits: &[u8]) -> Self {
        let dim = bits.len();
        let mut v = Self::zero(dim);
        for (i, &b) in bits.iter().enumerate() {
            if b & 1 != 0 {
                v.words[word_index(i)] |= 1u64 << bit_index(i);
            }
        }
        v
    }

    /// Create from a u64 value for dimensions ≤ 64.
    #[must_use]
    pub fn from_u64(dim: usize, value: u64) -> Self {
        assert!(dim <= WORD_BITS, "from_u64 requires dim <= 64");
        let masked = value & trailing_mask(dim);
        Self {
            words: vec![masked],
            dim,
        }
    }

    /// Number of components.
    #[inline]
    #[must_use]
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Get component i.
    #[inline]
    #[must_use]
    pub fn get(&self, i: usize) -> GF2 {
        assert!(
            i < self.dim,
            "index {} out of bounds for dim {}",
            i,
            self.dim
        );
        GF2::new(((self.words[word_index(i)] >> bit_index(i)) & 1) as u8)
    }

    /// Set component i.
    #[inline]
    pub fn set(&mut self, i: usize, value: GF2) {
        assert!(
            i < self.dim,
            "index {} out of bounds for dim {}",
            i,
            self.dim
        );
        let w = word_index(i);
        let b = bit_index(i);
        self.words[w] = (self.words[w] & !(1u64 << b)) | ((value.value() as u64) << b);
    }

    /// Hamming weight (number of 1-bits).
    #[must_use]
    pub fn weight(&self) -> usize {
        self.words.iter().map(|w| w.count_ones() as usize).sum()
    }

    /// Hamming distance to another vector.
    #[must_use]
    pub fn hamming_distance(&self, other: &Self) -> usize {
        assert_eq!(self.dim, other.dim, "dimension mismatch");
        self.words
            .iter()
            .zip(other.words.iter())
            .map(|(a, b)| (a ^ b).count_ones() as usize)
            .sum()
    }

    /// Dot product in F₂ (popcount of AND, mod 2).
    #[must_use]
    pub fn dot(&self, other: &Self) -> GF2 {
        assert_eq!(self.dim, other.dim, "dimension mismatch");
        let parity: u32 = self
            .words
            .iter()
            .zip(other.words.iter())
            .map(|(a, b)| (a & b).count_ones())
            .sum();
        GF2::new((parity & 1) as u8)
    }

    /// Component-wise XOR (addition in F₂ⁿ).
    #[must_use]
    pub fn add(&self, other: &Self) -> Self {
        assert_eq!(self.dim, other.dim, "dimension mismatch");
        let words: Vec<u64> = self
            .words
            .iter()
            .zip(other.words.iter())
            .map(|(a, b)| a ^ b)
            .collect();
        Self {
            words,
            dim: self.dim,
        }
    }

    /// Component-wise AND.
    #[must_use]
    pub fn bitwise_and(&self, other: &Self) -> Self {
        assert_eq!(self.dim, other.dim, "dimension mismatch");
        let words: Vec<u64> = self
            .words
            .iter()
            .zip(other.words.iter())
            .map(|(a, b)| a & b)
            .collect();
        Self {
            words,
            dim: self.dim,
        }
    }

    /// Whether this is the zero vector.
    #[must_use]
    pub fn is_zero(&self) -> bool {
        self.words.iter().all(|&w| w == 0)
    }

    /// Raw access to packed words.
    #[must_use]
    pub fn as_words(&self) -> &[u64] {
        &self.words
    }
}

// --- Operator impls ---

impl Add for &GF2Vector {
    type Output = GF2Vector;
    fn add(self, rhs: Self) -> GF2Vector {
        self.add(rhs)
    }
}

impl BitXor for &GF2Vector {
    type Output = GF2Vector;
    fn bitxor(self, rhs: Self) -> GF2Vector {
        self.add(rhs)
    }
}

impl BitAnd for &GF2Vector {
    type Output = GF2Vector;
    fn bitand(self, rhs: Self) -> GF2Vector {
        self.bitwise_and(rhs)
    }
}

impl fmt::Display for GF2Vector {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[")?;
        for i in 0..self.dim {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}", self.get(i))?;
        }
        write!(f, "]")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zero_vector() {
        let v = GF2Vector::zero(8);
        assert_eq!(v.dim(), 8);
        assert!(v.is_zero());
        assert_eq!(v.weight(), 0);
    }

    #[test]
    fn test_get_set_roundtrip() {
        let mut v = GF2Vector::zero(10);
        v.set(0, GF2::ONE);
        v.set(5, GF2::ONE);
        v.set(9, GF2::ONE);
        assert_eq!(v.get(0), GF2::ONE);
        assert_eq!(v.get(1), GF2::ZERO);
        assert_eq!(v.get(5), GF2::ONE);
        assert_eq!(v.get(9), GF2::ONE);
        assert_eq!(v.weight(), 3);
    }

    #[test]
    fn test_from_bits() {
        let v = GF2Vector::from_bits(&[1, 0, 1, 1, 0]);
        assert_eq!(v.dim(), 5);
        assert_eq!(v.get(0), GF2::ONE);
        assert_eq!(v.get(1), GF2::ZERO);
        assert_eq!(v.get(2), GF2::ONE);
        assert_eq!(v.get(3), GF2::ONE);
        assert_eq!(v.get(4), GF2::ZERO);
    }

    #[test]
    fn test_xor_addition() {
        let a = GF2Vector::from_bits(&[1, 0, 1, 0]);
        let b = GF2Vector::from_bits(&[1, 1, 0, 0]);
        let c = a.add(&b);
        assert_eq!(c, GF2Vector::from_bits(&[0, 1, 1, 0]));
    }

    #[test]
    fn test_dot_product() {
        let a = GF2Vector::from_bits(&[1, 1, 0, 1]);
        let b = GF2Vector::from_bits(&[1, 0, 1, 1]);
        // AND = [1,0,0,1], popcount = 2, mod 2 = 0
        assert_eq!(a.dot(&b), GF2::ZERO);

        let c = GF2Vector::from_bits(&[1, 1, 1]);
        let d = GF2Vector::from_bits(&[1, 0, 1]);
        // AND = [1,0,1], popcount = 2, mod 2 = 0
        assert_eq!(c.dot(&d), GF2::ZERO);

        let e = GF2Vector::from_bits(&[1, 1, 0]);
        let f = GF2Vector::from_bits(&[1, 0, 0]);
        // AND = [1,0,0], popcount = 1, mod 2 = 1
        assert_eq!(e.dot(&f), GF2::ONE);
    }

    #[test]
    fn test_hamming_weight_and_distance() {
        let a = GF2Vector::from_bits(&[1, 0, 1, 1, 0, 1]);
        assert_eq!(a.weight(), 4);

        let b = GF2Vector::from_bits(&[0, 1, 1, 0, 0, 1]);
        assert_eq!(a.hamming_distance(&b), 3);
    }

    #[test]
    fn test_from_u64_roundtrip() {
        let v = GF2Vector::from_u64(8, 0b10110011);
        assert_eq!(v.get(0), GF2::ONE);
        assert_eq!(v.get(1), GF2::ONE);
        assert_eq!(v.get(2), GF2::ZERO);
        assert_eq!(v.get(3), GF2::ZERO);
        assert_eq!(v.get(4), GF2::ONE);
        assert_eq!(v.get(5), GF2::ONE);
        assert_eq!(v.get(6), GF2::ZERO);
        assert_eq!(v.get(7), GF2::ONE);
    }

    #[test]
    fn test_large_vector() {
        let mut v = GF2Vector::zero(128);
        v.set(0, GF2::ONE);
        v.set(63, GF2::ONE);
        v.set(64, GF2::ONE);
        v.set(127, GF2::ONE);
        assert_eq!(v.weight(), 4);
        assert_eq!(v.get(63), GF2::ONE);
        assert_eq!(v.get(64), GF2::ONE);
        assert_eq!(v.get(65), GF2::ZERO);
    }
}
