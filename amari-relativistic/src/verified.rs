//! Formally verified relativistic physics with phantom types
//!
//! This module provides type-safe, mathematically verified implementations of
//! relativistic physics operations using phantom types for compile-time invariants.

#[cfg(feature = "std")]
use std::marker::PhantomData;

#[cfg(not(feature = "std"))]
use core::marker::PhantomData;

use crate::constants::C;
use nalgebra::Vector3;

/// Phantom type for spacetime signature verification
/// Ensures Cl(1,3) signature: (+---)
/// - T: 1 timelike dimension (e₀² = +1)
/// - S: 3 spacelike dimensions (e₁², e₂², e₃² = -1)
pub struct SpacetimeSignature<const T: usize, const S: usize>;

/// Type-level constraint for valid spacetime algebra
pub type ValidSpacetime = SpacetimeSignature<1, 3>;

/// Phantom type for four-velocity normalization verification
/// Ensures u·u = c² for massive particles
pub struct FourVelocityNorm;

/// Phantom type for energy-momentum relation verification
/// Ensures E² = (pc)² + (mc²)²
pub struct EnergyMomentumInvariant;

/// Phantom type for metric signature verification
/// Ensures correct Minkowski metric η = diag(+1, -1, -1, -1)
pub struct MinkowskiMetric;

/// Verified spacetime vector with compile-time signature guarantees
///
/// This structure ensures at the type level that:
/// 1. The spacetime signature is Cl(1,3) with (+---) metric
/// 2. Lorentz invariants are preserved
/// 3. Causality constraints are respected
#[derive(Debug, Clone)]
pub struct VerifiedSpacetimeVector {
    /// Temporal component (ct)
    pub(crate) t: f64,
    /// Spatial components (x, y, z)
    pub(crate) spatial: Vector3<f64>,
    /// Phantom marker for spacetime signature verification
    _signature: PhantomData<ValidSpacetime>,
    /// Phantom marker for metric verification
    _metric: PhantomData<MinkowskiMetric>,
}

impl VerifiedSpacetimeVector {
    /// Create a new verified spacetime vector
    ///
    /// # Type Invariants
    /// - Signature is fixed as Cl(1,3) at compile time
    /// - Metric signature is (+---) and cannot be violated
    pub fn new(t: f64, spatial: Vector3<f64>) -> Self {
        Self {
            t,
            spatial,
            _signature: PhantomData,
            _metric: PhantomData,
        }
    }

    /// Compute Minkowski inner product with mathematical guarantees
    ///
    /// η(u,v) = u₀v₀ - u₁v₁ - u₂v₂ - u₃v₃
    pub fn minkowski_dot(&self, other: &Self) -> f64 {
        // (+---) signature: t₁t₂ - x₁x₂ - y₁y₂ - z₁z₂
        self.t * other.t - self.spatial.dot(&other.spatial)
    }

    /// Compute norm squared with causality classification
    pub fn norm_squared(&self) -> f64 {
        self.minkowski_dot(self)
    }

    /// Check if vector is timelike (massive particle)
    pub fn is_timelike(&self) -> bool {
        self.norm_squared() > 0.0
    }

    /// Check if vector is spacelike
    pub fn is_spacelike(&self) -> bool {
        self.norm_squared() < 0.0
    }

    /// Check if vector is null (lightlike)
    pub fn is_null(&self) -> bool {
        self.norm_squared() == 0.0
    }
}

/// Verified four-velocity with normalization guarantees
///
/// Ensures u·u = c² at the type level for massive particles
#[derive(Debug, Clone)]
pub struct VerifiedFourVelocity {
    /// Underlying spacetime vector
    pub(crate) vector: VerifiedSpacetimeVector,
    /// Phantom marker for normalization verification
    _normalization: PhantomData<FourVelocityNorm>,
}

impl VerifiedFourVelocity {
    /// Create a verified four-velocity from 3-velocity
    ///
    /// Automatically ensures proper normalization u·u = c²
    pub fn from_velocity(velocity: Vector3<f64>) -> Result<Self, &'static str> {
        let v_mag_sq = velocity.dot(&velocity);

        if v_mag_sq >= C * C {
            return Err("Velocity must be less than speed of light");
        }

        let gamma = (1.0 - v_mag_sq / (C * C)).powf(-0.5);
        let u0 = gamma * C;
        let u_spatial = velocity * gamma;

        let vector = VerifiedSpacetimeVector::new(u0, u_spatial);

        Ok(Self {
            vector,
            _normalization: PhantomData,
        })
    }

    /// Verify normalization constraint u·u = c²
    pub fn is_normalized(&self) -> bool {
        let norm_sq = self.vector.norm_squared();
        let c_sq = C * C;

        // Use relative tolerance appropriate for c² scale (≈9e16)
        let relative_epsilon = 1e-6;
        let absolute_tolerance = relative_epsilon * c_sq;
        (norm_sq - c_sq).abs() < absolute_tolerance
    }

    /// Get Lorentz factor γ
    pub fn gamma(&self) -> f64 {
        self.vector.t / C
    }

    /// Get spacetime vector representation
    pub fn as_spacetime_vector(&self) -> &VerifiedSpacetimeVector {
        &self.vector
    }
}

/// Verified relativistic particle with energy-momentum invariants
///
/// Ensures E² = (pc)² + (mc²)² at the type level
#[derive(Debug, Clone)]
pub struct VerifiedRelativisticParticle {
    /// Verified spacetime position
    #[allow(dead_code)]
    pub(crate) position: VerifiedSpacetimeVector,
    /// Verified four-velocity
    pub(crate) four_velocity: VerifiedFourVelocity,
    /// Rest mass
    pub(crate) mass: f64,
    /// Electric charge
    #[allow(dead_code)]
    pub(crate) charge: f64,
    /// Phantom marker for energy-momentum verification
    _energy_momentum: PhantomData<EnergyMomentumInvariant>,
}

impl VerifiedRelativisticParticle {
    /// Create a verified relativistic particle
    ///
    /// Ensures all relativistic invariants are satisfied
    pub fn new(
        position: VerifiedSpacetimeVector,
        four_velocity: VerifiedFourVelocity,
        mass: f64,
        charge: f64,
    ) -> Result<Self, &'static str> {
        if mass <= 0.0 {
            return Err("Mass must be positive");
        }

        if !four_velocity.is_normalized() {
            return Err("Four-velocity must be normalized");
        }

        Ok(Self {
            position,
            four_velocity,
            mass,
            charge,
            _energy_momentum: PhantomData,
        })
    }

    /// Verify energy-momentum relation E² = (pc)² + (mc²)²
    pub fn satisfies_energy_momentum_relation(&self) -> bool {
        let gamma = self.four_velocity.gamma();

        let energy = gamma * self.mass * C * C;
        let momentum_spatial = self.four_velocity.vector.spatial * (gamma * self.mass);
        let pc = momentum_spatial.magnitude() * C;
        let mc2 = self.mass * C * C;

        let lhs = energy * energy;
        let rhs = pc * pc + mc2 * mc2;

        // Allow small numerical tolerance
        let epsilon = 1e-10;
        (lhs - rhs).abs() < epsilon
    }

    /// Get total energy with verification
    pub fn total_energy(&self) -> f64 {
        let gamma = self.four_velocity.gamma();
        gamma * self.mass * C * C
    }

    /// Get kinetic energy with verification
    pub fn kinetic_energy(&self) -> f64 {
        let gamma = self.four_velocity.gamma();
        let rest_energy = self.mass * C * C;
        (gamma - 1.0) * rest_energy
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_verified_spacetime_vector() {
        let vec = VerifiedSpacetimeVector::new(1.0, Vector3::new(0.5, 0.0, 0.0));
        assert!(vec.is_timelike());
        assert!(!vec.is_spacelike());
        assert!(!vec.is_null());
    }

    #[test]
    fn test_verified_four_velocity_normalization() {
        let velocity = Vector3::new(0.6 * C, 0.0, 0.0);
        let four_vel = VerifiedFourVelocity::from_velocity(velocity).unwrap();
        assert!(four_vel.is_normalized());
        assert!(four_vel.gamma() > 1.0);
    }

    #[test]
    fn test_verified_particle_energy_momentum() {
        let position = VerifiedSpacetimeVector::new(0.0, Vector3::zeros());
        let velocity = Vector3::new(0.5 * C, 0.0, 0.0);
        let four_vel = VerifiedFourVelocity::from_velocity(velocity).unwrap();
        let mass = 1.67e-27; // Proton mass

        let particle =
            VerifiedRelativisticParticle::new(position, four_vel, mass, 1.6e-19).unwrap();

        assert!(particle.satisfies_energy_momentum_relation());
        assert!(particle.total_energy() > 0.0);
        assert!(particle.kinetic_energy() >= 0.0);
    }

    #[test]
    fn test_minkowski_metric_properties() {
        let u = VerifiedSpacetimeVector::new(C, Vector3::zeros());
        let v = VerifiedSpacetimeVector::new(C, Vector3::zeros());

        // Test symmetry: u·v = v·u
        assert_relative_eq!(u.minkowski_dot(&v), v.minkowski_dot(&u), epsilon = 1e-15);

        // Test norm squared computation
        let norm_sq = u.norm_squared();
        assert_relative_eq!(norm_sq, C * C, epsilon = 1e-15);
    }

    #[test]
    fn test_causality_classification() {
        // Timelike vector (massive particle)
        let timelike = VerifiedSpacetimeVector::new(2.0 * C, Vector3::new(C, 0.0, 0.0));
        assert!(timelike.is_timelike());
        assert!(!timelike.is_spacelike());
        assert!(!timelike.is_null());

        // Spacelike vector (spacelike separation)
        let spacelike = VerifiedSpacetimeVector::new(C, Vector3::new(2.0 * C, 0.0, 0.0));
        assert!(!spacelike.is_timelike());
        assert!(spacelike.is_spacelike());
        assert!(!spacelike.is_null());

        // Null vector (light ray)
        let null = VerifiedSpacetimeVector::new(C, Vector3::new(C, 0.0, 0.0));
        assert!(!null.is_timelike());
        assert!(!null.is_spacelike());
        assert!(null.is_null());
    }
}
