//! Optical rotor field representation.
//!
//! A spatially-discretized rotor field representing an optical wavefront
//! in the even subalgebra of Cl(2,0).

use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

#[cfg(feature = "serialize")]
use serde::{Deserialize, Serialize};

/// A rotor field discretized on a 2D grid.
///
/// Each point holds an element of the even subalgebra of Cl(2,0),
/// which is isomorphic to the complex numbers.
///
/// # Memory Layout
///
/// Memory layout optimized for SIMD: separate arrays per grade.
/// - `scalar`: cos(φ) terms (grade-0)
/// - `bivector`: sin(φ)·e₁₂ coefficients (grade-2)
/// - `amplitude`: Separate amplitude envelope
///
/// # Mathematical Background
///
/// The even subalgebra of Cl(2,0) has basis {1, e₁₂} where (e₁₂)² = -1.
/// This is isomorphic to complex numbers via:
///
/// ```text
/// z = a + bi  ↔  R = a + b·e₁₂
/// ```
///
/// A rotor R = cos(φ) + sin(φ)·e₁₂ = exp(φ·e₁₂) represents a rotation
/// by angle 2φ in the e₁-e₂ plane.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub struct OpticalRotorField {
    /// Scalar parts: cos(φ) terms
    pub(crate) scalar: Vec<f32>,
    /// Bivector parts: sin(φ)·e₁₂ coefficients
    pub(crate) bivector: Vec<f32>,
    /// Amplitude envelope (separate from phase)
    pub(crate) amplitude: Vec<f32>,
    /// Grid dimensions (width, height)
    pub(crate) dimensions: (usize, usize),
}

impl OpticalRotorField {
    /// Create from explicit phase and amplitude arrays.
    ///
    /// # Arguments
    ///
    /// * `phase` - Phase values in radians
    /// * `amplitude` - Amplitude values (typically in [0, 1])
    /// * `dimensions` - Grid dimensions (width, height)
    ///
    /// # Panics
    ///
    /// Panics if array lengths don't match `dimensions.0 * dimensions.1`.
    pub fn new(phase: Vec<f32>, amplitude: Vec<f32>, dimensions: (usize, usize)) -> Self {
        assert_eq!(phase.len(), dimensions.0 * dimensions.1);
        assert_eq!(amplitude.len(), dimensions.0 * dimensions.1);

        let scalar = phase.iter().map(|&p| p.cos()).collect();
        let bivector = phase.iter().map(|&p| p.sin()).collect();

        Self {
            scalar,
            bivector,
            amplitude,
            dimensions,
        }
    }

    /// Create from phase array with uniform amplitude of 1.0.
    ///
    /// # Arguments
    ///
    /// * `phase` - Phase values in radians
    /// * `dimensions` - Grid dimensions (width, height)
    pub fn from_phase(phase: Vec<f32>, dimensions: (usize, usize)) -> Self {
        let amplitude = vec![1.0; phase.len()];
        Self::new(phase, amplitude, dimensions)
    }

    /// Create with random phase (deterministic from seed).
    ///
    /// Uses ChaCha8 RNG for reproducible random number generation.
    /// Phases are uniformly distributed in [0, 2π).
    ///
    /// # Arguments
    ///
    /// * `dimensions` - Grid dimensions (width, height)
    /// * `seed` - Random seed for deterministic generation
    pub fn random(dimensions: (usize, usize), seed: u64) -> Self {
        use rand::Rng;

        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let n = dimensions.0 * dimensions.1;

        let phase: Vec<f32> = (0..n)
            .map(|_| rng.gen::<f32>() * std::f32::consts::TAU)
            .collect();

        Self::from_phase(phase, dimensions)
    }

    /// Create uniform field (constant phase and amplitude).
    ///
    /// # Arguments
    ///
    /// * `phase` - Uniform phase value in radians
    /// * `amplitude` - Uniform amplitude value
    /// * `dimensions` - Grid dimensions (width, height)
    pub fn uniform(phase: f32, amplitude: f32, dimensions: (usize, usize)) -> Self {
        let n = dimensions.0 * dimensions.1;
        Self::new(vec![phase; n], vec![amplitude; n], dimensions)
    }

    /// Create an identity field (phase = 0 everywhere, amplitude = 1).
    ///
    /// The identity rotor is R = 1 + 0·e₁₂ = 1.
    pub fn identity(dimensions: (usize, usize)) -> Self {
        Self::uniform(0.0, 1.0, dimensions)
    }

    /// Grid dimensions (width, height).
    pub fn dimensions(&self) -> (usize, usize) {
        self.dimensions
    }

    /// Total number of points in the field.
    pub fn len(&self) -> usize {
        self.scalar.len()
    }

    /// Check if the field is empty.
    pub fn is_empty(&self) -> bool {
        self.scalar.is_empty()
    }

    /// Extract phase at a point (in radians, range [-π, π]).
    ///
    /// # Arguments
    ///
    /// * `x` - X coordinate (column)
    /// * `y` - Y coordinate (row)
    pub fn phase_at(&self, x: usize, y: usize) -> f32 {
        let idx = y * self.dimensions.0 + x;
        self.bivector[idx].atan2(self.scalar[idx])
    }

    /// Extract amplitude at a point.
    ///
    /// # Arguments
    ///
    /// * `x` - X coordinate (column)
    /// * `y` - Y coordinate (row)
    pub fn amplitude_at(&self, x: usize, y: usize) -> f32 {
        let idx = y * self.dimensions.0 + x;
        self.amplitude[idx]
    }

    /// Extract the rotor at a specific point.
    ///
    /// Returns (scalar, bivector, amplitude) tuple.
    pub fn rotor_at(&self, x: usize, y: usize) -> (f32, f32, f32) {
        let idx = y * self.dimensions.0 + x;
        (self.scalar[idx], self.bivector[idx], self.amplitude[idx])
    }

    /// Raw access to scalar components (for SIMD operations).
    pub fn scalars(&self) -> &[f32] {
        &self.scalar
    }

    /// Raw access to bivector components (for SIMD operations).
    pub fn bivectors(&self) -> &[f32] {
        &self.bivector
    }

    /// Raw access to amplitudes (for SIMD operations).
    pub fn amplitudes(&self) -> &[f32] {
        &self.amplitude
    }

    /// Mutable access to scalar components for in-place operations.
    pub fn scalars_mut(&mut self) -> &mut [f32] {
        &mut self.scalar
    }

    /// Mutable access to bivector components for in-place operations.
    pub fn bivectors_mut(&mut self) -> &mut [f32] {
        &mut self.bivector
    }

    /// Mutable access to amplitudes for in-place operations.
    pub fn amplitudes_mut(&mut self) -> &mut [f32] {
        &mut self.amplitude
    }

    /// Compute the total energy (sum of squared amplitudes).
    pub fn total_energy(&self) -> f32 {
        self.amplitude.iter().map(|a| a * a).sum()
    }

    /// Normalize the field so total energy equals 1.
    pub fn normalize(&mut self) {
        let energy = self.total_energy();
        if energy > 1e-10 {
            let scale = 1.0 / energy.sqrt();
            for a in &mut self.amplitude {
                *a *= scale;
            }
        }
    }

    /// Create a normalized copy of the field.
    pub fn normalized(&self) -> Self {
        let mut result = self.clone();
        result.normalize();
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::{FRAC_PI_2, FRAC_PI_4, PI, TAU};

    #[test]
    fn test_from_phase() {
        let phases = vec![0.0, FRAC_PI_4, FRAC_PI_2, PI];
        let field = OpticalRotorField::from_phase(phases.clone(), (4, 1));

        assert_eq!(field.dimensions(), (4, 1));
        assert_eq!(field.len(), 4);

        // Check phases are preserved (handling ±π equivalence)
        for (i, &p) in phases.iter().enumerate() {
            let extracted = field.phase_at(i, 0);
            let diff = (extracted - p).abs();
            // Handle wrap-around at ±π
            let wrapped_diff = diff.min(TAU - diff);
            assert!(
                wrapped_diff < 1e-5,
                "Phase mismatch at {}: {} vs {}",
                i,
                extracted,
                p
            );
        }

        // Check amplitudes are 1.0
        for i in 0..4 {
            assert!((field.amplitude_at(i, 0) - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_random_deterministic() {
        let field1 = OpticalRotorField::random((16, 16), 42);
        let field2 = OpticalRotorField::random((16, 16), 42);

        assert_eq!(field1.scalars(), field2.scalars());
        assert_eq!(field1.bivectors(), field2.bivectors());

        // Different seed should produce different result
        let field3 = OpticalRotorField::random((16, 16), 43);
        assert_ne!(field1.scalars(), field3.scalars());
    }

    #[test]
    fn test_uniform() {
        let field = OpticalRotorField::uniform(FRAC_PI_4, 0.5, (8, 8));

        assert_eq!(field.dimensions(), (8, 8));
        assert_eq!(field.len(), 64);

        for y in 0..8 {
            for x in 0..8 {
                assert!((field.phase_at(x, y) - FRAC_PI_4).abs() < 1e-6);
                assert!((field.amplitude_at(x, y) - 0.5).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn test_identity() {
        let field = OpticalRotorField::identity((4, 4));

        for y in 0..4 {
            for x in 0..4 {
                assert!((field.phase_at(x, y)).abs() < 1e-6);
                assert!((field.amplitude_at(x, y) - 1.0).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn test_rotor_normalization() {
        // Rotors should be unit normalized: scalar² + bivector² = 1
        let field = OpticalRotorField::random((32, 32), 12345);

        for i in 0..field.len() {
            let s = field.scalars()[i];
            let b = field.bivectors()[i];
            let norm_sq = s * s + b * b;
            assert!(
                (norm_sq - 1.0).abs() < 1e-6,
                "Rotor not normalized at {}: {}",
                i,
                norm_sq
            );
        }
    }

    #[test]
    fn test_phase_wrap() {
        // Test that phase correctly wraps from atan2
        let phases = vec![-PI, -FRAC_PI_2, 0.0, FRAC_PI_2, PI - 0.001];
        let field = OpticalRotorField::from_phase(phases.clone(), (5, 1));

        for (i, &p) in phases.iter().enumerate() {
            let extracted = field.phase_at(i, 0);
            // Handle wrap-around at ±π
            let diff = (extracted - p).abs();
            let wrapped_diff = (TAU - diff).abs().min(diff);
            assert!(
                wrapped_diff < 1e-5,
                "Phase mismatch at {}: {} vs {}",
                i,
                extracted,
                p
            );
        }
    }

    #[test]
    fn test_normalize() {
        let mut field = OpticalRotorField::uniform(0.0, 2.0, (4, 4));
        let original_energy = field.total_energy();
        assert!((original_energy - 64.0).abs() < 1e-6);

        field.normalize();
        assert!((field.total_energy() - 1.0).abs() < 1e-6);
    }
}
