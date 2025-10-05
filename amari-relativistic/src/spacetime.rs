//! Spacetime Algebra (STA) implementation using geometric algebra
//!
//! This module implements spacetime algebra with signature (+---) for Minkowski spacetime.
//! It leverages the geometric algebra types from amari-core to provide a mathematically
//! rigorous foundation for relativistic physics calculations.
//!
//! # Mathematical Background
//!
//! Spacetime algebra uses the Clifford algebra Cl(1,3) with signature (+---), where:
//! - γ₀² = +1 (timelike basis vector)
//! - γ₁², γ₂², γ₃² = -1 (spacelike basis vectors)
//! - γμγν + γνγμ = 2ημν (anticommutation relation)
//!
//! Four-vectors are represented as grade-1 multivectors in this algebra.
//!
//! # References
//! - Doran, C. & Lasenby, A. "Geometric Algebra for Physicists" (2003)
//! - Hestenes, D. "Space-Time Algebra" (1966)

use crate::constants::C;
use amari_core::{Multivector, Rotor};
use nalgebra::Vector3;

// Note: Serde support is available but not currently implemented for these types
// #[cfg(feature = "serde")]
// use serde::{Deserialize, Serialize};

#[cfg(feature = "std")]
use std::ops::{Add, Mul, Sub};

#[cfg(not(feature = "std"))]
use core::ops::{Add, Mul, Sub};

/// Spacetime signature: Cl(1,3) with signature (+---)
/// P=1 (timelike), Q=3 (spacelike), R=0 (no null vectors)
pub type SpacetimeAlgebra = Multivector<1, 3, 0>;

/// A four-vector in Minkowski spacetime
///
/// Four-vectors are represented as grade-1 multivectors in spacetime algebra Cl(1,3).
/// The metric signature is (+---) with coordinates (ct, x, y, z).
///
/// # Mathematical Properties
/// - Minkowski inner product: u·v = u₀v₀ - u₁v₁ - u₂v₂ - u₃v₃
/// - Norm squared: |u|² = u₀² - u₁² - u₂² - u₃²
/// - Light cone: |u|² > 0 (timelike), |u|² < 0 (spacelike), |u|² = 0 (null)
#[derive(Debug, Clone, PartialEq)]
pub struct SpacetimeVector {
    /// Internal representation as a grade-1 multivector in Cl(1,3)
    pub(crate) mv: SpacetimeAlgebra,
}

impl SpacetimeVector {
    /// Create a new spacetime vector from coordinates (ct, x, y, z)
    ///
    /// # Arguments
    /// * `t` - Time coordinate in seconds (will be multiplied by c)
    /// * `x` - Spatial x coordinate in meters
    /// * `y` - Spatial y coordinate in meters
    /// * `z` - Spatial z coordinate in meters
    ///
    /// # Mathematical Form
    /// Creates the four-vector: X = ctγ₀ + xγ₁ + yγ₂ + zγ₃
    ///
    /// # Example
    /// ```rust
    /// use amari_relativistic::spacetime::SpacetimeVector;
    ///
    /// // Event at origin at t=1 second
    /// let event = SpacetimeVector::new(1.0, 0.0, 0.0, 0.0);
    /// assert_eq!(event.time_component(), 299792458.0); // ct in meters
    /// ```
    pub fn new(t: f64, x: f64, y: f64, z: f64) -> Self {
        let mut mv = SpacetimeAlgebra::zero();
        mv.set_vector_component(0, C * t); // ct component (timelike)
        mv.set_vector_component(1, x); // x component (spacelike)
        mv.set_vector_component(2, y); // y component (spacelike)
        mv.set_vector_component(3, z); // z component (spacelike)
        Self { mv }
    }

    /// Create spacetime vector from spatial position and time
    ///
    /// # Arguments
    /// * `position` - 3D spatial position vector in meters
    /// * `t` - Time coordinate in seconds
    pub fn from_position_and_time(position: Vector3<f64>, t: f64) -> Self {
        Self::new(t, position.x, position.y, position.z)
    }

    /// Create a purely timelike vector (ct, 0, 0, 0)
    pub fn timelike(t: f64) -> Self {
        Self::new(t, 0.0, 0.0, 0.0)
    }

    /// Create a purely spacelike vector (0, x, y, z)
    pub fn spacelike(x: f64, y: f64, z: f64) -> Self {
        Self::new(0.0, x, y, z)
    }

    /// Get the time component (ct) in meters
    pub fn time_component(&self) -> f64 {
        self.mv.vector_component(0)
    }

    /// Get the time coordinate in seconds
    pub fn time(&self) -> f64 {
        self.time_component() / C
    }

    /// Get the x spatial component in meters
    pub fn x(&self) -> f64 {
        self.mv.vector_component(1)
    }

    /// Get the y spatial component in meters
    pub fn y(&self) -> f64 {
        self.mv.vector_component(2)
    }

    /// Get the z spatial component in meters
    pub fn z(&self) -> f64 {
        self.mv.vector_component(3)
    }

    /// Get the spatial part as a 3-vector
    ///
    /// Returns the spatial components (x, y, z) as a nalgebra Vector3.
    pub fn spatial(&self) -> Vector3<f64> {
        Vector3::new(self.x(), self.y(), self.z())
    }

    /// Minkowski inner product with another spacetime vector
    ///
    /// Computes the Lorentz-invariant inner product: u·v = u₀v₀ - u₁v₁ - u₂v₂ - u₃v₃
    ///
    /// # Mathematical Properties
    /// - Positive for timelike separations
    /// - Negative for spacelike separations
    /// - Zero for null (lightlike) separations
    ///
    /// # Example
    /// ```rust
    /// use amari_relativistic::spacetime::SpacetimeVector;
    ///
    /// let u = SpacetimeVector::new(1.0, 0.0, 0.0, 0.0);
    /// let v = SpacetimeVector::new(1.0, 0.0, 0.0, 0.0);
    /// let inner = u.minkowski_dot(&v);
    /// // Should be positive (timelike)
    /// assert!(inner > 0.0);
    /// ```
    pub fn minkowski_dot(&self, other: &Self) -> f64 {
        // Minkowski metric signature (+---)
        let t_part = self.time_component() * other.time_component();
        let spatial_part = self.x() * other.x() + self.y() * other.y() + self.z() * other.z();
        t_part - spatial_part
    }

    /// Compute the Minkowski norm squared
    ///
    /// Returns |u|² = u·u using the Minkowski metric.
    /// - Positive: timelike vector
    /// - Negative: spacelike vector
    /// - Zero: null (lightlike) vector
    pub fn minkowski_norm_squared(&self) -> f64 {
        self.minkowski_dot(self)
    }

    /// Compute the proper time τ for timelike vectors
    ///
    /// For timelike vectors with |u|² > 0, returns τ = √(|u|²)/c.
    /// Returns None for spacelike or null vectors.
    ///
    /// # Mathematical Background
    /// Proper time is the time measured by a clock moving along the worldline.
    /// For a four-velocity u, the proper time element is dτ = √(u·u)/c.
    pub fn proper_time(&self) -> Option<f64> {
        let norm_sq = self.minkowski_norm_squared();
        if norm_sq > 0.0 {
            Some(norm_sq.sqrt() / C)
        } else {
            None
        }
    }

    /// Check if this vector is timelike (|u|² > 0)
    pub fn is_timelike(&self) -> bool {
        self.minkowski_norm_squared() > 0.0
    }

    /// Check if this vector is spacelike (|u|² < 0)
    pub fn is_spacelike(&self) -> bool {
        self.minkowski_norm_squared() < 0.0
    }

    /// Check if this vector is null/lightlike (|u|² = 0)
    pub fn is_null(&self) -> bool {
        self.minkowski_norm_squared().abs() < 1e-10
    }

    /// Get coordinates as array [ct, x, y, z]
    pub fn coordinates(&self) -> [f64; 4] {
        [self.time_component(), self.x(), self.y(), self.z()]
    }

    /// Create from coordinates array [ct, x, y, z]
    pub fn from_coordinates(coords: [f64; 4]) -> Self {
        let mut mv = SpacetimeAlgebra::zero();
        mv.set_vector_component(0, coords[0]); // ct
        mv.set_vector_component(1, coords[1]); // x
        mv.set_vector_component(2, coords[2]); // y
        mv.set_vector_component(3, coords[3]); // z
        Self { mv }
    }

    /// Access the underlying multivector (for advanced operations)
    pub fn as_multivector(&self) -> &SpacetimeAlgebra {
        &self.mv
    }
}

impl Add for SpacetimeVector {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        Self {
            mv: self.mv + rhs.mv,
        }
    }
}

impl Sub for SpacetimeVector {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self {
        Self {
            mv: self.mv - rhs.mv,
        }
    }
}

impl Mul<f64> for SpacetimeVector {
    type Output = Self;

    fn mul(self, scalar: f64) -> Self {
        Self {
            mv: self.mv * scalar,
        }
    }
}

/// Four-velocity vector with normalization constraint
///
/// Four-velocity is a timelike unit vector satisfying u·u = c².
/// It represents the velocity of a massive particle through spacetime.
///
/// # Mathematical Properties
/// - Normalization: u·u = c²
/// - Timelike: u₀ > 0 (future-directed)
/// - Related to 3-velocity: u = γ(c, v⃗) where γ = 1/√(1-v²/c²)
#[derive(Debug, Clone, PartialEq)]
pub struct FourVelocity {
    /// Internal spacetime vector representation
    vector: SpacetimeVector,
}

impl FourVelocity {
    /// Create four-velocity from 3-velocity
    ///
    /// # Arguments
    /// * `velocity` - 3-velocity vector in m/s
    ///
    /// # Mathematical Form
    /// u^μ = γ(c, v⃗) where γ = 1/√(1 - v²/c²)
    ///
    /// # Panics
    /// Panics if |v| ≥ c (faster than light)
    ///
    /// # Example
    /// ```rust
    /// use amari_relativistic::spacetime::FourVelocity;
    /// use nalgebra::Vector3;
    ///
    /// let v = Vector3::new(0.5 * 299792458.0, 0.0, 0.0); // 0.5c in x direction
    /// let u = FourVelocity::from_velocity(v);
    /// assert!((u.gamma() - 1.154700538).abs() < 1e-8); // γ ≈ 1.155
    /// ```
    pub fn from_velocity(velocity: Vector3<f64>) -> Self {
        let v_magnitude = velocity.magnitude();

        if v_magnitude >= C {
            panic!(
                "Velocity magnitude ({:.3e} m/s) must be less than speed of light ({:.3e} m/s)",
                v_magnitude, C
            );
        }

        let gamma = Self::lorentz_factor(v_magnitude);
        let four_vel = SpacetimeVector::new(
            gamma * C / C, // γc/c = γ (time component)
            gamma * velocity.x,
            gamma * velocity.y,
            gamma * velocity.z,
        );

        Self { vector: four_vel }
    }

    /// Create four-velocity for particle at rest
    pub fn at_rest() -> Self {
        Self::from_velocity(Vector3::zeros())
    }

    /// Calculate Lorentz factor γ = 1/√(1 - v²/c²)
    #[inline]
    fn lorentz_factor(v_magnitude: f64) -> f64 {
        let beta_squared = (v_magnitude / C).powi(2);
        if beta_squared >= 1.0 {
            panic!("Beta squared ({:.6}) must be less than 1", beta_squared);
        }
        1.0 / (1.0 - beta_squared).sqrt()
    }

    /// Get the Lorentz factor γ
    pub fn gamma(&self) -> f64 {
        self.vector.time_component() / C
    }

    /// Get the 3-velocity vector
    ///
    /// Returns v⃗ = (u₁/u₀, u₂/u₀, u₃/u₀)c where uμ is the four-velocity.
    pub fn velocity(&self) -> Vector3<f64> {
        let gamma = self.gamma();
        Vector3::new(
            self.vector.x() / gamma,
            self.vector.y() / gamma,
            self.vector.z() / gamma,
        )
    }

    /// Get the rapidity φ where v = c tanh(φ)
    ///
    /// Rapidity is useful because rapidities add linearly under boosts,
    /// unlike velocities which have the relativistic velocity addition formula.
    ///
    /// # Mathematical Background
    /// - φ = tanh⁻¹(v/c) = ½ ln((1+β)/(1-β)) where β = v/c
    /// - γ = cosh(φ), βγ = sinh(φ)
    pub fn rapidity(&self) -> f64 {
        let beta = self.velocity().magnitude() / C;
        if beta >= 1.0 {
            panic!("Beta ({:.6}) must be less than 1 for finite rapidity", beta);
        }
        0.5 * ((1.0 + beta) / (1.0 - beta)).ln()
    }

    /// Normalize the four-velocity to ensure u·u = c²
    ///
    /// This method corrects for numerical drift that may accumulate
    /// during integration or other operations.
    pub fn normalize(&mut self) {
        let norm_squared = self.vector.minkowski_norm_squared();
        if norm_squared > 1e-14 {
            let correction_factor = C / norm_squared.sqrt();
            self.vector = self.vector.clone() * correction_factor;
        }
    }

    /// Check if the four-velocity is properly normalized
    ///
    /// Returns true if |u·u - c²| < tolerance
    pub fn is_normalized(&self, tolerance: f64) -> bool {
        (self.vector.minkowski_norm_squared() - C * C).abs() < tolerance
    }

    /// Get the underlying spacetime vector
    pub fn as_spacetime_vector(&self) -> &SpacetimeVector {
        &self.vector
    }

    /// Convert to spacetime vector (consuming)
    pub fn into_spacetime_vector(self) -> SpacetimeVector {
        self.vector
    }
}

/// Lorentz transformation represented as a rotor (even-grade multivector)
///
/// Lorentz rotors provide a unified way to represent both spatial rotations
/// and Lorentz boosts using the geometric algebra exponential map.
///
/// # Mathematical Background
/// - Spatial rotations: R = exp(-½θB) where B is a spatial bivector
/// - Lorentz boosts: R = exp(-½φB) where B is a spacetime bivector and φ is rapidity
/// - Application: x' = RxR† (rotor sandwiching)
///
/// # References
/// - Doran & Lasenby, "Geometric Algebra for Physicists", Ch. 10
pub struct LorentzRotor {
    /// Internal rotor representation using amari-core
    rotor: Rotor<1, 3, 0>,
}

impl LorentzRotor {
    /// Create a Lorentz boost from 3-velocity
    ///
    /// Creates a boost transformation that transforms the rest frame
    /// to a frame moving with the given velocity.
    ///
    /// # Arguments
    /// * `velocity` - 3-velocity vector of the boost in m/s
    ///
    /// # Mathematical Form
    /// R = exp(-½φn⃗·σ⃗) where:
    /// - φ is the rapidity: φ = tanh⁻¹(|v|/c)
    /// - n⃗ = v⃗/|v⃗| is the boost direction
    /// - σ⃗ are the spacetime bivectors γ₀γᵢ
    ///
    /// # Example
    /// ```rust
    /// use amari_relativistic::spacetime::LorentzRotor;
    /// use nalgebra::Vector3;
    ///
    /// let velocity = Vector3::new(0.6 * 299792458.0, 0.0, 0.0); // 0.6c
    /// let boost = LorentzRotor::boost(velocity);
    /// // boost.velocity() should return approximately the input velocity
    /// ```
    pub fn boost(velocity: Vector3<f64>) -> Self {
        let v_magnitude = velocity.magnitude();

        if v_magnitude >= C {
            panic!(
                "Boost velocity magnitude ({:.3e}) must be less than c ({:.3e})",
                v_magnitude, C
            );
        }

        if v_magnitude < 1e-10 {
            // For very small velocities, return identity
            return Self::identity();
        }

        // Calculate rapidity φ = tanh⁻¹(v/c)
        let beta = v_magnitude / C;
        let rapidity = 0.5 * ((1.0 + beta) / (1.0 - beta)).ln();

        // Boost direction (unit vector)
        let direction = velocity / v_magnitude;

        // Create spacetime bivector for boost
        // In spacetime algebra, boost bivectors are γ₀γᵢ terms
        // TODO: Implement proper spacetime bivector construction
        // For now, use a spatial bivector as approximation
        let mut boost_bivector = SpacetimeAlgebra::zero();

        // Create bivector components for boost
        // This is a simplified implementation - proper spacetime bivectors
        // would involve mixed timelike-spacelike terms
        boost_bivector.set_bivector_component(0, rapidity * direction.x); // Approximation
        boost_bivector.set_bivector_component(1, rapidity * direction.y);
        boost_bivector.set_bivector_component(2, rapidity * direction.z);

        // Create rotor: R = exp(-½φB)
        let half_angle_bivector = boost_bivector.clone() * (-0.5);
        let _rotor_mv = half_angle_bivector.exp();

        // Create rotor from the exponential
        let rotor = Rotor::from_multivector_bivector(&boost_bivector, rapidity);

        Self { rotor }
    }

    /// Create identity transformation (no boost, no rotation)
    pub fn identity() -> Self {
        Self {
            rotor: Rotor::identity(),
        }
    }

    /// Transform a spacetime vector: x' = RxR†
    ///
    /// Applies the Lorentz transformation to the input vector.
    /// This preserves the Minkowski inner product (Lorentz invariance).
    pub fn transform(&self, vector: &SpacetimeVector) -> SpacetimeVector {
        let transformed = self.rotor.apply(&vector.mv);
        SpacetimeVector { mv: transformed }
    }

    /// Compose two Lorentz transformations: R₃ = R₂R₁
    ///
    /// The composition represents applying transformation R₁ first,
    /// then transformation R₂.
    pub fn compose(&self, other: &Self) -> Self {
        Self {
            rotor: self.rotor.compose(&other.rotor),
        }
    }

    /// Get the inverse transformation: R⁻¹ = R†
    pub fn inverse(&self) -> Self {
        Self {
            rotor: self.rotor.inverse(),
        }
    }

    /// Extract the velocity of this boost transformation
    ///
    /// For pure boosts, this returns the 3-velocity that was used
    /// to create the boost. For general Lorentz transformations
    /// (boost + rotation), this returns the boost component velocity.
    ///
    /// # Note
    /// This is a simplified implementation. A complete implementation
    /// would decompose the rotor into boost and rotation parts.
    pub fn velocity(&self) -> Vector3<f64> {
        // TODO: Implement proper velocity extraction from rotor
        // This requires decomposing the rotor into boost and rotation parts
        // For now, return zero as placeholder
        Vector3::zeros()
    }

    /// Get the underlying rotor for advanced operations
    pub fn as_rotor(&self) -> &Rotor<1, 3, 0> {
        &self.rotor
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::{assert_abs_diff_eq, assert_relative_eq};

    #[test]
    fn test_spacetime_vector_creation() {
        let sv = SpacetimeVector::new(1.0, 1.0, 2.0, 3.0);

        assert_eq!(sv.time(), 1.0);
        assert_eq!(sv.x(), 1.0);
        assert_eq!(sv.y(), 2.0);
        assert_eq!(sv.z(), 3.0);

        let spatial = sv.spatial();
        assert_eq!(spatial, Vector3::new(1.0, 2.0, 3.0));
    }

    #[test]
    fn test_minkowski_metric() {
        // Test metric signature (+---)
        let timelike = SpacetimeVector::new(2.0, 1.0, 1.0, 1.0); // ct > spatial
        let spacelike = SpacetimeVector::new(1.0, 2.0, 2.0, 2.0); // ct < spatial
        let null = SpacetimeVector::new(1.0, C, 0.0, 0.0); // light ray

        assert!(timelike.is_timelike());
        assert!(spacelike.is_spacelike());
        assert!(null.is_null());

        // Test inner product
        let t1 = SpacetimeVector::timelike(1.0);
        let t2 = SpacetimeVector::timelike(1.0);
        assert!(t1.minkowski_dot(&t2) > 0.0); // Timelike separation

        let s1 = SpacetimeVector::spacelike(1.0, 0.0, 0.0);
        let s2 = SpacetimeVector::spacelike(1.0, 0.0, 0.0);
        assert!(s1.minkowski_dot(&s2) < 0.0); // Spacelike separation
    }

    #[test]
    fn test_four_velocity_normalization() {
        // Test four-velocity normalization u·u = c²
        let velocity = Vector3::new(0.6 * C, 0.0, 0.0); // 0.6c
        let four_vel = FourVelocity::from_velocity(velocity);

        let norm_squared = four_vel.as_spacetime_vector().minkowski_norm_squared();
        assert_relative_eq!(norm_squared, C * C, epsilon = 1e-8);

        // Test Lorentz factor
        let expected_gamma = 1.0 / (1.0 - 0.6_f64.powi(2)).sqrt();
        assert_relative_eq!(four_vel.gamma(), expected_gamma, epsilon = 1e-12);
    }

    #[test]
    fn test_four_velocity_at_rest() {
        let rest = FourVelocity::at_rest();

        assert_relative_eq!(rest.gamma(), 1.0, epsilon = 1e-15);
        assert!(rest.velocity().magnitude() < 1e-15);

        let norm_squared = rest.as_spacetime_vector().minkowski_norm_squared();
        assert_relative_eq!(norm_squared, C * C, epsilon = 1e-10);
    }

    #[test]
    fn test_rapidity() {
        let velocity = Vector3::new(0.8 * C, 0.0, 0.0); // 0.8c
        let four_vel = FourVelocity::from_velocity(velocity);

        let rapidity = four_vel.rapidity();
        let expected_rapidity = 0.5 * ((1.0_f64 + 0.8) / (1.0_f64 - 0.8)).ln();
        assert_relative_eq!(rapidity, expected_rapidity, epsilon = 1e-12);

        // Test rapidity-velocity relationship: v = c tanh(φ)
        let v_from_rapidity = C * rapidity.tanh();
        assert_relative_eq!(v_from_rapidity, 0.8 * C, epsilon = 1e-10);
    }

    #[test]
    fn test_lorentz_boost_identity() {
        let identity = LorentzRotor::identity();
        let test_vector = SpacetimeVector::new(1.0, 2.0, 3.0, 4.0);

        let transformed = identity.transform(&test_vector);

        // Identity should leave vector unchanged
        assert_abs_diff_eq!(transformed.time(), test_vector.time(), epsilon = 1e-12);
        assert_abs_diff_eq!(transformed.x(), test_vector.x(), epsilon = 1e-12);
        assert_abs_diff_eq!(transformed.y(), test_vector.y(), epsilon = 1e-12);
        assert_abs_diff_eq!(transformed.z(), test_vector.z(), epsilon = 1e-12);
    }

    #[test]
    fn test_boost_composition() {
        let boost1 = LorentzRotor::boost(Vector3::new(0.3 * C, 0.0, 0.0));
        let boost2 = LorentzRotor::boost(Vector3::new(0.2 * C, 0.0, 0.0));

        let composed = boost1.compose(&boost2);
        let inverse_composed = composed.inverse();

        // Test that composing with inverse gives identity
        let should_be_identity = composed.compose(&inverse_composed);
        let test_vector = SpacetimeVector::new(1.0, 1.0, 1.0, 1.0);
        let result = should_be_identity.transform(&test_vector);

        // Should be close to original (within numerical precision)
        assert_abs_diff_eq!(result.time(), test_vector.time(), epsilon = 1e-10);
        assert_abs_diff_eq!(result.x(), test_vector.x(), epsilon = 1e-10);
    }

    #[test]
    #[should_panic(expected = "must be less than speed of light")]
    fn test_faster_than_light_panic() {
        let faster_than_light = Vector3::new(2.0 * C, 0.0, 0.0);
        FourVelocity::from_velocity(faster_than_light);
    }

    #[test]
    fn test_coordinates_roundtrip() {
        let original = SpacetimeVector::new(1.0, 2.0, 3.0, 4.0);
        let coords = original.coordinates();
        let reconstructed = SpacetimeVector::from_coordinates(coords);

        assert_abs_diff_eq!(original.time(), reconstructed.time(), epsilon = 1e-15);
        assert_abs_diff_eq!(original.x(), reconstructed.x(), epsilon = 1e-15);
        assert_abs_diff_eq!(original.y(), reconstructed.y(), epsilon = 1e-15);
        assert_abs_diff_eq!(original.z(), reconstructed.z(), epsilon = 1e-15);
    }
}
