//! Algebra implementation for optical rotor fields.
//!
//! Provides VSA (Vector Symbolic Architecture) operations on optical
//! rotor fields using geometric algebra.

use super::rotor_field::OpticalRotorField;

/// Algebra instance for optical rotor fields.
///
/// Provides binding algebra operations on `OpticalRotorField`:
/// - **Binding**: Rotor multiplication (phase addition)
/// - **Bundling**: Weighted rotor sum with normalization
/// - **Similarity**: Normalized inner product
/// - **Inverse**: Rotor reverse (phase negation)
///
/// # Example
///
/// ```ignore
/// use amari_holographic::optical::{OpticalFieldAlgebra, OpticalRotorField};
/// use std::f32::consts::FRAC_PI_4;
///
/// let algebra = OpticalFieldAlgebra::new((64, 64));
///
/// let field_a = OpticalRotorField::random((64, 64), 1);
/// let field_b = OpticalRotorField::random((64, 64), 2);
///
/// // Binding adds phases
/// let bound = algebra.bind(&field_a, &field_b);
///
/// // Inverse negates phase
/// let inv = algebra.inverse(&field_a);
///
/// // bind(a, inverse(a)) ≈ identity
/// let identity = algebra.bind(&field_a, &inv);
/// assert!((identity.phase_at(0, 0)).abs() < 1e-5);
///
/// // Self-similarity is 1.0
/// let sim = algebra.similarity(&field_a, &field_a);
/// assert!((sim - 1.0).abs() < 1e-5);
/// ```
#[derive(Clone, Debug)]
pub struct OpticalFieldAlgebra {
    dimensions: (usize, usize),
}

impl OpticalFieldAlgebra {
    /// Create a new algebra for fields of the given dimensions.
    pub fn new(dimensions: (usize, usize)) -> Self {
        Self { dimensions }
    }

    /// Get the dimensions this algebra operates on.
    pub fn dimensions(&self) -> (usize, usize) {
        self.dimensions
    }

    /// Total number of pixels.
    pub fn field_size(&self) -> usize {
        self.dimensions.0 * self.dimensions.1
    }

    /// Create the identity field for this algebra.
    ///
    /// The identity is the field with phase = 0 and amplitude = 1 everywhere.
    /// It satisfies: `bind(x, identity()) = x`.
    pub fn identity(&self) -> OpticalRotorField {
        OpticalRotorField::identity(self.dimensions)
    }

    /// Create a random field for this algebra.
    pub fn random(&self, seed: u64) -> OpticalRotorField {
        OpticalRotorField::random(self.dimensions, seed)
    }

    /// Binding via rotor multiplication (= phase addition).
    ///
    /// In geometric algebra terms, this is the geometric product of rotors:
    /// ```text
    /// (a_s + a_b·e₁₂)(b_s + b_b·e₁₂) = (a_s·b_s - a_b·b_b) + (a_s·b_b + a_b·b_s)·e₁₂
    /// ```
    ///
    /// This is isomorphic to complex multiplication, which adds phases:
    /// ```text
    /// exp(φ_a · e₁₂) · exp(φ_b · e₁₂) = exp((φ_a + φ_b) · e₁₂)
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if field dimensions don't match the algebra dimensions.
    pub fn bind(&self, a: &OpticalRotorField, b: &OpticalRotorField) -> OpticalRotorField {
        assert_eq!(a.dimensions(), b.dimensions());
        assert_eq!(a.dimensions(), self.dimensions);

        let n = a.len();
        let mut scalar = Vec::with_capacity(n);
        let mut bivector = Vec::with_capacity(n);
        let mut amplitude = Vec::with_capacity(n);

        let a_s = a.scalars();
        let a_b = a.bivectors();
        let b_s = b.scalars();
        let b_b = b.bivectors();
        let a_amp = a.amplitudes();
        let b_amp = b.amplitudes();

        for i in 0..n {
            // Rotor product: (a_s + a_b·e₁₂)(b_s + b_b·e₁₂)
            // Scalar part: a_s·b_s - a_b·b_b
            // Bivector part: a_s·b_b + a_b·b_s
            scalar.push(a_s[i] * b_s[i] - a_b[i] * b_b[i]);
            bivector.push(a_s[i] * b_b[i] + a_b[i] * b_s[i]);
            // Amplitudes multiply
            amplitude.push(a_amp[i] * b_amp[i]);
        }

        OpticalRotorField {
            scalar,
            bivector,
            amplitude,
            dimensions: self.dimensions,
        }
    }

    /// Bundling via weighted rotor sum.
    ///
    /// Computes a weighted superposition of rotor fields:
    /// ```text
    /// R_sum = Σ wᵢ · Aᵢ · Rᵢ
    /// ```
    ///
    /// The result is normalized so that the rotor part is unit length.
    /// The amplitude is set to the magnitude of the weighted sum.
    ///
    /// # Arguments
    ///
    /// * `elements` - Fields to bundle
    /// * `weights` - Weights for each field
    ///
    /// # Panics
    ///
    /// Panics if `elements` and `weights` have different lengths,
    /// or if any field has wrong dimensions.
    pub fn bundle(&self, elements: &[OpticalRotorField], weights: &[f32]) -> OpticalRotorField {
        assert_eq!(elements.len(), weights.len());
        assert!(!elements.is_empty(), "Cannot bundle empty list");

        for e in elements {
            assert_eq!(e.dimensions(), self.dimensions);
        }

        let n = elements[0].len();
        let mut scalar = vec![0.0f32; n];
        let mut bivector = vec![0.0f32; n];
        let mut amplitude = vec![0.0f32; n];

        for (elem, &w) in elements.iter().zip(weights) {
            for i in 0..n {
                // Weight applies to full rotor (including amplitude)
                let weighted_amp = w * elem.amplitudes()[i];
                scalar[i] += weighted_amp * elem.scalars()[i];
                bivector[i] += weighted_amp * elem.bivectors()[i];
            }
        }

        // Compute resulting amplitude and normalize rotor parts
        for i in 0..n {
            let s = scalar[i];
            let b = bivector[i];
            let r = (s * s + b * b).sqrt();

            if r > 1e-10 {
                amplitude[i] = r;
                scalar[i] = s / r;
                bivector[i] = b / r;
            } else {
                amplitude[i] = 0.0;
                scalar[i] = 1.0;
                bivector[i] = 0.0;
            }
        }

        OpticalRotorField {
            scalar,
            bivector,
            amplitude,
            dimensions: self.dimensions,
        }
    }

    /// Bundle with uniform weights.
    ///
    /// Convenience method that bundles with equal weights (1/n).
    pub fn bundle_uniform(&self, elements: &[OpticalRotorField]) -> OpticalRotorField {
        let weight = 1.0 / elements.len() as f32;
        let weights = vec![weight; elements.len()];
        self.bundle(elements, &weights)
    }

    /// Similarity via normalized rotor inner product.
    ///
    /// Computes:
    /// ```text
    /// sim(a, b) = ⟨R_a† · R_b⟩₀ integrated over field / (||a|| · ||b||)
    /// ```
    ///
    /// where R† is the rotor reverse (complex conjugate).
    ///
    /// # Properties
    ///
    /// - `similarity(a, a) = 1.0`
    /// - `similarity(a, inverse(a)) = 1.0` (same phase magnitude)
    /// - Random fields have near-zero similarity (quasi-orthogonal)
    ///
    /// # Panics
    ///
    /// Panics if field dimensions don't match.
    pub fn similarity(&self, a: &OpticalRotorField, b: &OpticalRotorField) -> f32 {
        assert_eq!(a.dimensions(), b.dimensions());

        let n = a.len();
        let mut sum = 0.0f32;
        let mut norm_a = 0.0f32;
        let mut norm_b = 0.0f32;

        let a_s = a.scalars();
        let a_b = a.bivectors();
        let b_s = b.scalars();
        let b_b = b.bivectors();
        let a_amp = a.amplitudes();
        let b_amp = b.amplitudes();

        for i in 0..n {
            // R_a† · R_b = (a_s - a_b·e₁₂)(b_s + b_b·e₁₂)
            // Scalar part = a_s·b_s + a_b·b_b (note: + because of conjugate)
            let inner = a_s[i] * b_s[i] + a_b[i] * b_b[i];
            sum += a_amp[i] * b_amp[i] * inner;
            norm_a += a_amp[i] * a_amp[i];
            norm_b += b_amp[i] * b_amp[i];
        }

        let denom = (norm_a * norm_b).sqrt();
        if denom > 1e-10 {
            sum / denom
        } else {
            0.0
        }
    }

    /// Inverse via rotor reverse (= phase negation).
    ///
    /// The reverse of a rotor negates its bivector part:
    /// ```text
    /// (a_s + a_b·e₁₂)† = a_s - a_b·e₁₂
    /// ```
    ///
    /// This corresponds to phase negation:
    /// ```text
    /// exp(φ·e₁₂)† = exp(-φ·e₁₂)
    /// ```
    ///
    /// # Property
    ///
    /// `bind(a, inverse(a)) = identity` (approximately, for unit rotors)
    pub fn inverse(&self, a: &OpticalRotorField) -> OpticalRotorField {
        OpticalRotorField {
            scalar: a.scalar.clone(),
            bivector: a.bivector.iter().map(|&b| -b).collect(),
            amplitude: a.amplitude.clone(),
            dimensions: a.dimensions,
        }
    }

    /// Unbind operation: retrieve associated value.
    ///
    /// Given `bound = bind(key, value)`, calling `unbind(key, bound)`
    /// returns (approximately) `value`.
    ///
    /// This is equivalent to `bind(inverse(key), bound)`.
    pub fn unbind(&self, key: &OpticalRotorField, bound: &OpticalRotorField) -> OpticalRotorField {
        let key_inv = self.inverse(key);
        self.bind(&key_inv, bound)
    }

    /// Scale a field's amplitude by a constant factor.
    pub fn scale(&self, field: &OpticalRotorField, factor: f32) -> OpticalRotorField {
        OpticalRotorField {
            scalar: field.scalar.clone(),
            bivector: field.bivector.clone(),
            amplitude: field.amplitude.iter().map(|&a| a * factor).collect(),
            dimensions: field.dimensions,
        }
    }

    /// Add a constant phase to all pixels.
    ///
    /// Equivalent to binding with a uniform field of the given phase.
    pub fn add_phase(&self, field: &OpticalRotorField, phase: f32) -> OpticalRotorField {
        let uniform = OpticalRotorField::uniform(phase, 1.0, self.dimensions);
        self.bind(field, &uniform)
    }

    /// Compute the average phase (circular mean) of a field.
    pub fn mean_phase(&self, field: &OpticalRotorField) -> f32 {
        let n = field.len() as f32;
        let sum_s: f32 = field.scalars().iter().sum();
        let sum_b: f32 = field.bivectors().iter().sum();
        (sum_b / n).atan2(sum_s / n)
    }

    /// Compute phase variance (circular variance) of a field.
    pub fn phase_variance(&self, field: &OpticalRotorField) -> f32 {
        let n = field.len() as f32;
        let sum_s: f32 = field.scalars().iter().sum();
        let sum_b: f32 = field.bivectors().iter().sum();
        let r = ((sum_s / n).powi(2) + (sum_b / n).powi(2)).sqrt();
        1.0 - r
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::{FRAC_PI_2, FRAC_PI_4, PI};

    fn approx_eq(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() < eps
    }

    #[test]
    fn test_binding_is_phase_addition() {
        let algebra = OpticalFieldAlgebra::new((3, 1));

        let phase_a = vec![0.0, FRAC_PI_4, FRAC_PI_2];
        let phase_b = vec![FRAC_PI_4, FRAC_PI_4, FRAC_PI_4];

        let field_a = OpticalRotorField::from_phase(phase_a, (3, 1));
        let field_b = OpticalRotorField::from_phase(phase_b, (3, 1));

        let bound = algebra.bind(&field_a, &field_b);

        // Phases should add
        assert!(approx_eq(bound.phase_at(0, 0), FRAC_PI_4, 1e-5));
        assert!(approx_eq(bound.phase_at(1, 0), FRAC_PI_2, 1e-5));
        assert!(approx_eq(bound.phase_at(2, 0), 3.0 * FRAC_PI_4, 1e-5));
    }

    #[test]
    fn test_inverse_negates_phase() {
        let algebra = OpticalFieldAlgebra::new((4, 1));
        let field = OpticalRotorField::from_phase(vec![0.0, FRAC_PI_4, FRAC_PI_2, PI], (4, 1));

        let inv = algebra.inverse(&field);

        assert!(approx_eq(inv.phase_at(0, 0), 0.0, 1e-5));
        assert!(approx_eq(inv.phase_at(1, 0), -FRAC_PI_4, 1e-5));
        assert!(approx_eq(inv.phase_at(2, 0), -FRAC_PI_2, 1e-5));
        // -π and π are equivalent (wrap around)
        let phase_3 = inv.phase_at(3, 0);
        assert!(
            approx_eq(phase_3.abs(), PI, 1e-5),
            "Expected ±π, got {}",
            phase_3
        );
    }

    #[test]
    fn test_bind_inverse_identity() {
        let algebra = OpticalFieldAlgebra::new((64, 64));
        let field = OpticalRotorField::random((64, 64), 42);

        let inv = algebra.inverse(&field);
        let product = algebra.bind(&field, &inv);

        // Should be identity (phase = 0 everywhere)
        for y in 0..64 {
            for x in 0..64 {
                let phase = product.phase_at(x, y);
                assert!(
                    phase.abs() < 1e-5,
                    "Expected 0, got {} at ({}, {})",
                    phase,
                    x,
                    y
                );
            }
        }
    }

    #[test]
    fn test_similarity_self_is_one() {
        let algebra = OpticalFieldAlgebra::new((64, 64));
        let field = OpticalRotorField::random((64, 64), 42);

        let sim = algebra.similarity(&field, &field);
        assert!(
            approx_eq(sim, 1.0, 1e-5),
            "Self-similarity should be 1.0, got {}",
            sim
        );
    }

    #[test]
    fn test_random_fields_quasi_orthogonal() {
        let algebra = OpticalFieldAlgebra::new((64, 64));

        let fields: Vec<_> = (0..10)
            .map(|i| OpticalRotorField::random((64, 64), i))
            .collect();

        // Self-similarity should be 1
        for f in &fields {
            assert!(approx_eq(algebra.similarity(f, f), 1.0, 1e-5));
        }

        // Cross-similarity should be small (quasi-orthogonal)
        for i in 0..fields.len() {
            for j in (i + 1)..fields.len() {
                let sim = algebra.similarity(&fields[i], &fields[j]);
                assert!(
                    sim.abs() < 0.2,
                    "Fields {} and {} too similar: {}",
                    i,
                    j,
                    sim
                );
            }
        }
    }

    #[test]
    fn test_bundle_preserves_similarity() {
        let algebra = OpticalFieldAlgebra::new((64, 64));

        let field_a = OpticalRotorField::random((64, 64), 1);
        let field_b = OpticalRotorField::random((64, 64), 2);

        let bundled = algebra.bundle(&[field_a.clone(), field_b.clone()], &[0.5, 0.5]);

        // Bundled should be similar to both inputs
        let sim_a = algebra.similarity(&bundled, &field_a);
        let sim_b = algebra.similarity(&bundled, &field_b);

        assert!(
            sim_a > 0.3,
            "Bundled should be similar to field_a: {}",
            sim_a
        );
        assert!(
            sim_b > 0.3,
            "Bundled should be similar to field_b: {}",
            sim_b
        );
    }

    #[test]
    fn test_unbind_retrieves_value() {
        let algebra = OpticalFieldAlgebra::new((32, 32));

        let key = OpticalRotorField::random((32, 32), 1);
        let value = OpticalRotorField::random((32, 32), 2);

        let bound = algebra.bind(&key, &value);
        let retrieved = algebra.unbind(&key, &bound);

        // Retrieved should be highly similar to original value
        let sim = algebra.similarity(&retrieved, &value);
        assert!(
            approx_eq(sim, 1.0, 1e-4),
            "Retrieved should match value, similarity: {}",
            sim
        );
    }

    #[test]
    fn test_identity() {
        let algebra = OpticalFieldAlgebra::new((16, 16));
        let field = OpticalRotorField::random((16, 16), 42);
        let identity = algebra.identity();

        let bound = algebra.bind(&field, &identity);

        // Binding with identity should give same field
        let sim = algebra.similarity(&bound, &field);
        assert!(approx_eq(sim, 1.0, 1e-5));

        // Phases should be identical
        for y in 0..16 {
            for x in 0..16 {
                assert!(approx_eq(bound.phase_at(x, y), field.phase_at(x, y), 1e-5));
            }
        }
    }

    #[test]
    fn test_add_phase() {
        let algebra = OpticalFieldAlgebra::new((4, 1));
        let field = OpticalRotorField::from_phase(vec![0.0, FRAC_PI_4, FRAC_PI_2, PI], (4, 1));

        let shifted = algebra.add_phase(&field, FRAC_PI_4);

        assert!(approx_eq(shifted.phase_at(0, 0), FRAC_PI_4, 1e-5));
        assert!(approx_eq(shifted.phase_at(1, 0), FRAC_PI_2, 1e-5));
    }

    #[test]
    fn test_mean_phase() {
        let algebra = OpticalFieldAlgebra::new((4, 1));

        // Uniform phase should give that phase
        let field = OpticalRotorField::uniform(FRAC_PI_4, 1.0, (4, 1));
        let mean = algebra.mean_phase(&field);
        assert!(approx_eq(mean, FRAC_PI_4, 1e-5));
    }
}
