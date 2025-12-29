//! Lee hologram encoder using geometric algebra.
//!
//! The Lee method encodes complex optical fields as binary patterns
//! by modulating with a spatial carrier frequency and thresholding.
//!
//! In geometric algebra terms, the carrier is a rotor field varying
//! linearly in space:
//!
//! ```text
//! Carrier(x) = exp(2π·f_c·x · e₁₂)
//!            = cos(2π·f_c·x) + sin(2π·f_c·x)·e₁₂
//! ```

use super::hologram::BinaryHologram;
use super::rotor_field::OpticalRotorField;

#[cfg(feature = "serialize")]
use serde::{Deserialize, Serialize};

/// Configuration for Lee hologram encoding.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub struct LeeEncoderConfig {
    /// Carrier spatial frequency (cycles per pixel).
    ///
    /// Higher frequency increases phase resolution but requires
    /// finer spatial resolution. Typical values: 0.1 to 0.5.
    pub carrier_frequency: f32,

    /// Carrier direction angle (radians, 0 = horizontal).
    ///
    /// The carrier wave propagates in this direction.
    /// 0 = horizontal (x direction), π/2 = vertical (y direction).
    pub carrier_angle: f32,

    /// Grid dimensions (width, height).
    pub dimensions: (usize, usize),
}

impl LeeEncoderConfig {
    /// Create a new configuration with default carrier angle (horizontal).
    pub fn new(dimensions: (usize, usize), carrier_frequency: f32) -> Self {
        Self {
            carrier_frequency,
            carrier_angle: 0.0,
            dimensions,
        }
    }

    /// Create configuration with angled carrier.
    pub fn with_angle(
        dimensions: (usize, usize),
        carrier_frequency: f32,
        carrier_angle: f32,
    ) -> Self {
        Self {
            carrier_frequency,
            carrier_angle,
            dimensions,
        }
    }
}

/// Lee hologram encoder using geometric algebra.
///
/// Encodes optical rotor fields as binary patterns suitable for
/// display on DMD (digital micromirror device) or similar binary
/// spatial light modulators.
///
/// # Theory
///
/// The Lee method works by:
/// 1. Multiplying the signal rotor field by a carrier rotor field
/// 2. Thresholding the scalar part of the result
///
/// The carrier is a rotor varying linearly in space:
/// ```text
/// Carrier(x,y) = exp(2π·f_c·(x·cos(θ) + y·sin(θ)) · e₁₂)
/// ```
///
/// The binary threshold condition is:
/// ```text
/// B(x,y) = 1  if  ⟨Carrier · Signal⟩₀ > cos(π·A)
///          0  otherwise
/// ```
///
/// where ⟨R⟩₀ denotes the scalar (grade-0) part of rotor R.
///
/// # Example
///
/// ```ignore
/// use amari_holographic::optical::{GeometricLeeEncoder, OpticalRotorField};
///
/// let encoder = GeometricLeeEncoder::with_frequency((256, 256), 0.25);
/// let field = OpticalRotorField::random((256, 256), 42);
/// let hologram = encoder.encode(&field);
///
/// // Check fill factor is reasonable
/// assert!(hologram.fill_factor() > 0.2 && hologram.fill_factor() < 0.8);
/// ```
pub struct GeometricLeeEncoder {
    config: LeeEncoderConfig,
    /// Pre-computed carrier rotor field
    carrier: OpticalRotorField,
}

impl GeometricLeeEncoder {
    /// Create encoder with specified configuration.
    pub fn new(config: LeeEncoderConfig) -> Self {
        let carrier = Self::compute_carrier(&config);
        Self { config, carrier }
    }

    /// Simple constructor with horizontal carrier.
    ///
    /// # Arguments
    ///
    /// * `dimensions` - Grid dimensions (width, height)
    /// * `carrier_frequency` - Carrier frequency in cycles per pixel
    pub fn with_frequency(dimensions: (usize, usize), carrier_frequency: f32) -> Self {
        Self::new(LeeEncoderConfig::new(dimensions, carrier_frequency))
    }

    /// Compute the carrier rotor field for the given configuration.
    fn compute_carrier(config: &LeeEncoderConfig) -> OpticalRotorField {
        let (w, h) = config.dimensions;
        let mut phase = Vec::with_capacity(w * h);

        let cos_angle = config.carrier_angle.cos();
        let sin_angle = config.carrier_angle.sin();

        for y in 0..h {
            for x in 0..w {
                let t = x as f32 * cos_angle + y as f32 * sin_angle;
                let p = std::f32::consts::TAU * config.carrier_frequency * t;
                phase.push(p);
            }
        }

        OpticalRotorField::from_phase(phase, config.dimensions)
    }

    /// Encode optical field as binary Lee hologram.
    ///
    /// The encoding process:
    /// 1. Multiply signal by carrier (rotor product = phase addition)
    /// 2. Threshold based on scalar part and amplitude
    ///
    /// # Arguments
    ///
    /// * `field` - Input optical rotor field to encode
    ///
    /// # Panics
    ///
    /// Panics if field dimensions don't match encoder dimensions.
    pub fn encode(&self, field: &OpticalRotorField) -> BinaryHologram {
        assert_eq!(
            field.dimensions(),
            self.config.dimensions,
            "Field dimensions {:?} don't match encoder dimensions {:?}",
            field.dimensions(),
            self.config.dimensions
        );

        let n = field.len();
        let mut pattern = Vec::with_capacity(n);

        // Rotor multiplication: (c_s + c_b·e₁₂)(f_s + f_b·e₁₂)
        // = (c_s·f_s - c_b·f_b) + (c_s·f_b + c_b·f_s)·e₁₂
        // Scalar part of product = c_s·f_s - c_b·f_b = cos(φ_c + φ_f)

        let c_s = self.carrier.scalars();
        let c_b = self.carrier.bivectors();
        let f_s = field.scalars();
        let f_b = field.bivectors();
        let amp = field.amplitudes();

        for i in 0..n {
            let modulated_scalar = c_s[i] * f_s[i] - c_b[i] * f_b[i];
            let threshold = (std::f32::consts::PI * amp[i]).cos();
            pattern.push(modulated_scalar > threshold);
        }

        BinaryHologram::from_bools(&pattern, self.config.dimensions)
    }

    /// Encode with custom threshold function.
    ///
    /// Allows specifying a custom threshold instead of the standard
    /// Lee threshold based on amplitude.
    ///
    /// # Arguments
    ///
    /// * `field` - Input optical rotor field
    /// * `threshold_fn` - Function returning threshold for each pixel index
    pub fn encode_with_threshold<F>(
        &self,
        field: &OpticalRotorField,
        threshold_fn: F,
    ) -> BinaryHologram
    where
        F: Fn(usize) -> f32,
    {
        assert_eq!(field.dimensions(), self.config.dimensions);

        let n = field.len();
        let mut pattern = Vec::with_capacity(n);

        let c_s = self.carrier.scalars();
        let c_b = self.carrier.bivectors();
        let f_s = field.scalars();
        let f_b = field.bivectors();

        for i in 0..n {
            let modulated_scalar = c_s[i] * f_s[i] - c_b[i] * f_b[i];
            pattern.push(modulated_scalar > threshold_fn(i));
        }

        BinaryHologram::from_bools(&pattern, self.config.dimensions)
    }

    /// Theoretical diffraction efficiency.
    ///
    /// The first-order diffraction efficiency is:
    /// ```text
    /// η = (2/π)² · ⟨sin²(π·A)⟩
    /// ```
    ///
    /// For uniform A=0.5: η ≈ 40.5%
    ///
    /// # Arguments
    ///
    /// * `field` - Input optical rotor field
    pub fn theoretical_efficiency(&self, field: &OpticalRotorField) -> f32 {
        let scale = (2.0 / std::f32::consts::PI).powi(2);
        let mean_sin_sq: f32 = field
            .amplitudes()
            .iter()
            .map(|&a| (std::f32::consts::PI * a).sin().powi(2))
            .sum::<f32>()
            / field.len() as f32;
        scale * mean_sin_sq
    }

    /// Get encoder configuration.
    pub fn config(&self) -> &LeeEncoderConfig {
        &self.config
    }

    /// Get reference to pre-computed carrier field.
    pub fn carrier(&self) -> &OpticalRotorField {
        &self.carrier
    }

    /// Get encoder dimensions.
    pub fn dimensions(&self) -> (usize, usize) {
        self.config.dimensions
    }

    /// Compute the modulated rotor field (before thresholding).
    ///
    /// This is useful for analysis or visualization.
    ///
    /// # Arguments
    ///
    /// * `field` - Input optical rotor field
    pub fn modulate(&self, field: &OpticalRotorField) -> OpticalRotorField {
        assert_eq!(field.dimensions(), self.config.dimensions);

        let n = field.len();
        let mut scalar = Vec::with_capacity(n);
        let mut bivector = Vec::with_capacity(n);

        let c_s = self.carrier.scalars();
        let c_b = self.carrier.bivectors();
        let f_s = field.scalars();
        let f_b = field.bivectors();

        for i in 0..n {
            // Rotor product
            scalar.push(c_s[i] * f_s[i] - c_b[i] * f_b[i]);
            bivector.push(c_s[i] * f_b[i] + c_b[i] * f_s[i]);
        }

        OpticalRotorField {
            scalar,
            bivector,
            amplitude: field.amplitudes().to_vec(),
            dimensions: self.config.dimensions,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    #[test]
    fn test_encoder_creation() {
        let encoder = GeometricLeeEncoder::with_frequency((64, 64), 0.25);
        assert_eq!(encoder.dimensions(), (64, 64));
        assert!((encoder.config().carrier_frequency - 0.25).abs() < 1e-6);
    }

    #[test]
    fn test_carrier_computation() {
        let encoder = GeometricLeeEncoder::with_frequency((4, 1), 0.25);
        let carrier = encoder.carrier();

        // At x=0, phase should be 0
        assert!(carrier.phase_at(0, 0).abs() < 1e-6);

        // At x=1, phase should be 2π * 0.25 = π/2
        let expected = std::f32::consts::FRAC_PI_2;
        assert!(
            (carrier.phase_at(1, 0) - expected).abs() < 1e-5,
            "Phase at x=1: {} vs expected {}",
            carrier.phase_at(1, 0),
            expected
        );

        // At x=2, phase should be π (but atan2 may return -π)
        let phase_2 = carrier.phase_at(2, 0);
        assert!(
            (phase_2.abs() - PI).abs() < 1e-5,
            "Phase at x=2: {} vs expected ±{}",
            phase_2,
            PI
        );
    }

    #[test]
    fn test_encode_produces_binary() {
        let encoder = GeometricLeeEncoder::with_frequency((128, 128), 0.25);
        // Use amplitude 0.5 for reasonable fill factor
        // (amplitude 1.0 gives threshold cos(π) = -1, so everything passes)
        let phase: Vec<f32> = (0..128 * 128)
            .map(|i| {
                use rand::Rng;
                use rand::SeedableRng;
                let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(42 + i as u64);
                rng.gen::<f32>() * std::f32::consts::TAU
            })
            .collect();
        let amplitude = vec![0.5; 128 * 128];
        let field = OpticalRotorField::new(phase, amplitude, (128, 128));

        let hologram = encoder.encode(&field);

        assert_eq!(hologram.dimensions(), (128, 128));

        // Fill factor should be reasonable (not all 0s or all 1s)
        let fill = hologram.fill_factor();
        assert!(fill > 0.2 && fill < 0.8, "Unexpected fill factor: {}", fill);
    }

    #[test]
    fn test_uniform_field_encoding() {
        let encoder = GeometricLeeEncoder::with_frequency((64, 64), 0.25);

        // Uniform phase = 0 field
        let field = OpticalRotorField::uniform(0.0, 0.5, (64, 64));
        let hologram = encoder.encode(&field);

        // Should produce a regular pattern (carrier-dominated)
        // The fill factor should be around 0.5 for this case
        let fill = hologram.fill_factor();
        assert!(
            fill > 0.3 && fill < 0.7,
            "Unexpected fill factor for uniform field: {}",
            fill
        );
    }

    #[test]
    fn test_theoretical_efficiency() {
        let encoder = GeometricLeeEncoder::with_frequency((64, 64), 0.25);

        // For uniform amplitude = 0.5, efficiency should be ≈ 40.5%
        let field = OpticalRotorField::uniform(0.0, 0.5, (64, 64));
        let efficiency = encoder.theoretical_efficiency(&field);

        let expected = (2.0 / PI).powi(2) * (PI * 0.5_f32).sin().powi(2);
        assert!(
            (efficiency - expected).abs() < 1e-5,
            "Efficiency {} vs expected {}",
            efficiency,
            expected
        );
    }

    #[test]
    fn test_modulate() {
        let encoder = GeometricLeeEncoder::with_frequency((32, 32), 0.25);
        let field = OpticalRotorField::uniform(0.0, 1.0, (32, 32));

        let modulated = encoder.modulate(&field);

        // Modulated should have same dimensions
        assert_eq!(modulated.dimensions(), field.dimensions());

        // Amplitudes should be preserved
        assert_eq!(modulated.amplitudes(), field.amplitudes());

        // Phase should be carrier phase (since signal phase is 0)
        for y in 0..32 {
            for x in 0..32 {
                let carrier_phase = encoder.carrier().phase_at(x, y);
                let mod_phase = modulated.phase_at(x, y);
                let diff = (carrier_phase - mod_phase).abs();
                // Handle wrap-around
                let wrapped_diff = diff.min(std::f32::consts::TAU - diff);
                assert!(
                    wrapped_diff < 1e-5,
                    "Phase mismatch at ({}, {}): {} vs {}",
                    x,
                    y,
                    mod_phase,
                    carrier_phase
                );
            }
        }
    }

    #[test]
    fn test_angled_carrier() {
        use std::f32::consts::FRAC_PI_4;

        let config = LeeEncoderConfig::with_angle((32, 32), 0.25, FRAC_PI_4);
        let encoder = GeometricLeeEncoder::new(config);

        // For 45 degree angle, phase should increase along both x and y
        let carrier = encoder.carrier();

        // Phase at (1, 1) should be greater than at (1, 0)
        let phase_11 = carrier.phase_at(1, 1);
        let phase_10 = carrier.phase_at(1, 0);
        assert!(phase_11 > phase_10);
    }

    #[test]
    #[should_panic]
    fn test_dimension_mismatch() {
        let encoder = GeometricLeeEncoder::with_frequency((64, 64), 0.25);
        let field = OpticalRotorField::random((32, 32), 42);
        encoder.encode(&field); // Should panic
    }
}
