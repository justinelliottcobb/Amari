//! Binary hologram pattern for DMD display.
//!
//! The output of Lee encoding is a binary pattern that can be
//! displayed on a digital micromirror device (DMD) or similar
//! binary spatial light modulator.

#[cfg(feature = "serialize")]
use serde::{Deserialize, Serialize};

/// Binary hologram pattern for DMD display.
///
/// Stores a bit-packed binary pattern where each pixel is represented
/// by a single bit. This is the output of Lee hologram encoding.
///
/// # Memory Layout
///
/// Bits are packed LSB-first within each byte, in row-major order:
/// - Bit 0 of byte 0 = pixel (0, 0)
/// - Bit 1 of byte 0 = pixel (1, 0)
/// - Bit 0 of byte 1 = pixel (8, 0) (for width > 8)
///
/// # Example
///
/// ```ignore
/// use amari_holographic::optical::BinaryHologram;
///
/// let pattern = vec![true, false, true, false];
/// let hologram = BinaryHologram::from_bools(&pattern, (4, 1));
///
/// assert!(hologram.get(0, 0));
/// assert!(!hologram.get(1, 0));
/// assert!(hologram.get(2, 0));
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub struct BinaryHologram {
    /// Packed binary pattern (1 bit per pixel)
    data: Vec<u8>,
    /// Pattern dimensions (width, height)
    dimensions: (usize, usize),
}

impl BinaryHologram {
    /// Create from boolean array.
    ///
    /// # Arguments
    ///
    /// * `pattern` - Boolean values in row-major order
    /// * `dimensions` - Grid dimensions (width, height)
    ///
    /// # Panics
    ///
    /// Panics if `pattern.len() != dimensions.0 * dimensions.1`.
    pub fn from_bools(pattern: &[bool], dimensions: (usize, usize)) -> Self {
        assert_eq!(pattern.len(), dimensions.0 * dimensions.1);

        let n_bytes = pattern.len().div_ceil(8);
        let mut data = vec![0u8; n_bytes];

        for (i, &b) in pattern.iter().enumerate() {
            if b {
                data[i / 8] |= 1 << (i % 8);
            }
        }

        Self { data, dimensions }
    }

    /// Create an empty (all zeros) hologram.
    pub fn zeros(dimensions: (usize, usize)) -> Self {
        let n = dimensions.0 * dimensions.1;
        let n_bytes = n.div_ceil(8);
        Self {
            data: vec![0u8; n_bytes],
            dimensions,
        }
    }

    /// Create a filled (all ones) hologram.
    pub fn ones(dimensions: (usize, usize)) -> Self {
        let n = dimensions.0 * dimensions.1;
        let n_bytes = n.div_ceil(8);
        let mut data = vec![0xFFu8; n_bytes];

        // Clear extra bits in the last byte
        let extra_bits = n % 8;
        if extra_bits > 0 && !data.is_empty() {
            let last_idx = data.len() - 1;
            data[last_idx] = (1u8 << extra_bits) - 1;
        }

        Self { data, dimensions }
    }

    /// Get pixel value at (x, y).
    ///
    /// # Arguments
    ///
    /// * `x` - X coordinate (column)
    /// * `y` - Y coordinate (row)
    ///
    /// # Panics
    ///
    /// Panics if coordinates are out of bounds.
    pub fn get(&self, x: usize, y: usize) -> bool {
        assert!(x < self.dimensions.0 && y < self.dimensions.1);
        let idx = y * self.dimensions.0 + x;
        (self.data[idx / 8] >> (idx % 8)) & 1 != 0
    }

    /// Set pixel value at (x, y).
    ///
    /// # Arguments
    ///
    /// * `x` - X coordinate (column)
    /// * `y` - Y coordinate (row)
    /// * `value` - New pixel value
    ///
    /// # Panics
    ///
    /// Panics if coordinates are out of bounds.
    pub fn set(&mut self, x: usize, y: usize, value: bool) {
        assert!(x < self.dimensions.0 && y < self.dimensions.1);
        let idx = y * self.dimensions.0 + x;
        if value {
            self.data[idx / 8] |= 1 << (idx % 8);
        } else {
            self.data[idx / 8] &= !(1 << (idx % 8));
        }
    }

    /// Toggle pixel value at (x, y).
    pub fn toggle(&mut self, x: usize, y: usize) {
        assert!(x < self.dimensions.0 && y < self.dimensions.1);
        let idx = y * self.dimensions.0 + x;
        self.data[idx / 8] ^= 1 << (idx % 8);
    }

    /// Grid dimensions (width, height).
    pub fn dimensions(&self) -> (usize, usize) {
        self.dimensions
    }

    /// Total number of pixels.
    pub fn len(&self) -> usize {
        self.dimensions.0 * self.dimensions.1
    }

    /// Check if the hologram is empty (zero dimensions).
    pub fn is_empty(&self) -> bool {
        self.dimensions.0 == 0 || self.dimensions.1 == 0
    }

    /// Raw packed data (for hardware interface).
    pub fn as_bytes(&self) -> &[u8] {
        &self.data
    }

    /// Count of "on" pixels.
    pub fn popcount(&self) -> usize {
        self.data.iter().map(|b| b.count_ones() as usize).sum()
    }

    /// Fill factor (fraction of "on" pixels).
    ///
    /// Returns a value in [0, 1].
    pub fn fill_factor(&self) -> f32 {
        if self.is_empty() {
            return 0.0;
        }
        self.popcount() as f32 / self.len() as f32
    }

    /// Compute XOR of two holograms (Hamming distance base).
    ///
    /// Returns a new hologram where each pixel is the XOR of
    /// the corresponding pixels in the inputs.
    pub fn xor(&self, other: &Self) -> Self {
        assert_eq!(self.dimensions, other.dimensions);
        let data = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a ^ b)
            .collect();
        Self {
            data,
            dimensions: self.dimensions,
        }
    }

    /// Compute Hamming distance between two holograms.
    ///
    /// Returns the number of differing pixels.
    pub fn hamming_distance(&self, other: &Self) -> usize {
        self.xor(other).popcount()
    }

    /// Compute normalized Hamming distance.
    ///
    /// Returns distance / total_pixels, in range [0, 1].
    pub fn normalized_hamming_distance(&self, other: &Self) -> f32 {
        if self.is_empty() {
            return 0.0;
        }
        self.hamming_distance(other) as f32 / self.len() as f32
    }

    /// Convert to boolean vector.
    pub fn to_bools(&self) -> Vec<bool> {
        let n = self.len();
        (0..n)
            .map(|i| (self.data[i / 8] >> (i % 8)) & 1 != 0)
            .collect()
    }

    /// Invert all pixels (NOT operation).
    pub fn invert(&mut self) {
        for byte in &mut self.data {
            *byte = !*byte;
        }
        // Clear extra bits in last byte
        let extra_bits = self.len() % 8;
        if extra_bits > 0 && !self.data.is_empty() {
            let last_idx = self.data.len() - 1;
            self.data[last_idx] &= (1u8 << extra_bits) - 1;
        }
    }

    /// Create an inverted copy.
    pub fn inverted(&self) -> Self {
        let mut result = self.clone();
        result.invert();
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_from_bools() {
        let pattern = vec![true, false, true, true, false, false, true, false];
        let hologram = BinaryHologram::from_bools(&pattern, (8, 1));

        assert_eq!(hologram.dimensions(), (8, 1));
        assert_eq!(hologram.len(), 8);

        for (i, &expected) in pattern.iter().enumerate() {
            assert_eq!(hologram.get(i, 0), expected, "Mismatch at position {}", i);
        }
    }

    #[test]
    fn test_zeros_and_ones() {
        let zeros = BinaryHologram::zeros((16, 16));
        assert_eq!(zeros.popcount(), 0);
        assert!((zeros.fill_factor()).abs() < 1e-6);

        let ones = BinaryHologram::ones((16, 16));
        assert_eq!(ones.popcount(), 256);
        assert!((ones.fill_factor() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_set_and_get() {
        let mut hologram = BinaryHologram::zeros((8, 8));

        hologram.set(3, 2, true);
        assert!(hologram.get(3, 2));
        assert!(!hologram.get(2, 3));

        hologram.set(3, 2, false);
        assert!(!hologram.get(3, 2));
    }

    #[test]
    fn test_toggle() {
        let mut hologram = BinaryHologram::zeros((4, 4));

        hologram.toggle(1, 1);
        assert!(hologram.get(1, 1));

        hologram.toggle(1, 1);
        assert!(!hologram.get(1, 1));
    }

    #[test]
    fn test_xor_and_hamming() {
        let a = BinaryHologram::from_bools(&[true, true, false, false], (4, 1));
        let b = BinaryHologram::from_bools(&[true, false, true, false], (4, 1));

        let xored = a.xor(&b);
        assert!(!xored.get(0, 0)); // true ^ true = false
        assert!(xored.get(1, 0)); // true ^ false = true
        assert!(xored.get(2, 0)); // false ^ true = true
        assert!(!xored.get(3, 0)); // false ^ false = false

        assert_eq!(a.hamming_distance(&b), 2);
        assert!((a.normalized_hamming_distance(&b) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_invert() {
        let original = BinaryHologram::from_bools(&[true, false, true, false], (4, 1));
        let inverted = original.inverted();

        assert!(!inverted.get(0, 0));
        assert!(inverted.get(1, 0));
        assert!(!inverted.get(2, 0));
        assert!(inverted.get(3, 0));
    }

    #[test]
    fn test_to_bools() {
        let pattern = vec![true, false, true, true, false];
        let hologram = BinaryHologram::from_bools(&pattern, (5, 1));
        let recovered = hologram.to_bools();

        assert_eq!(pattern, recovered);
    }

    #[test]
    fn test_2d_indexing() {
        let mut hologram = BinaryHologram::zeros((4, 3));

        // Set a checkerboard pattern: true where (x + y) is even
        for y in 0..3 {
            for x in 0..4 {
                hologram.set(x, y, (x + y) % 2 == 0);
            }
        }

        // Verify pattern: (x + y) even -> true
        assert!(hologram.get(0, 0)); // 0 + 0 = 0 (even) -> true
        assert!(!hologram.get(1, 0)); // 1 + 0 = 1 (odd)  -> false
        assert!(hologram.get(2, 0)); // 2 + 0 = 2 (even) -> true
        assert!(hologram.get(1, 1)); // 1 + 1 = 2 (even) -> true
        assert!(!hologram.get(1, 2)); // 1 + 2 = 3 (odd)  -> false
    }

    #[test]
    fn test_fill_factor() {
        let all_on = BinaryHologram::ones((10, 10));
        assert!((all_on.fill_factor() - 1.0).abs() < 1e-6);

        let half = BinaryHologram::from_bools(&[true, false, true, false], (4, 1));
        assert!((half.fill_factor() - 0.5).abs() < 1e-6);
    }
}
