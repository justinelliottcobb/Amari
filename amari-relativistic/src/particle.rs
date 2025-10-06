//! Relativistic charged particle dynamics
//!
//! This module provides types and functions for relativistic charged particles
//! moving through curved spacetime. It handles both massive and massless particles,
//! with proper treatment of relativistic energy-momentum relations.
//!
//! # Mathematical Background
//!
//! For relativistic particles, the energy-momentum relation is:
//! ```text
//! E² = (pc)² + (mc²)²
//! ```
//!
//! Where:
//! - E is the total energy
//! - p is the momentum magnitude
//! - m is the rest mass
//! - c is the speed of light
//!
//! The four-momentum is: p^μ = m u^μ where u^μ is the four-velocity.
//!
//! # Key Features
//! - Relativistic energy-momentum calculations
//! - Proper time evolution
//! - Integration with geodesic equation
//! - Support for charged particles (future electromagnetic field extension)
//!
//! # References
//! - Jackson, "Classical Electrodynamics" Ch. 12 (1999)
//! - Landau & Lifshitz, "Classical Theory of Fields" Ch. 3 (1975)

use crate::constants::{C, E_CHARGE};
use crate::geodesic::GeodesicIntegrator;
use crate::spacetime::{FourVelocity, SpacetimeVector};
use nalgebra::Vector3;

// Note: Serde support is available but not currently implemented for these types
// #[cfg(feature = "serde")]
// use serde::{Deserialize, Serialize};

#[cfg(feature = "std")]
use std::f64::consts::PI;

#[cfg(not(feature = "std"))]
use core::f64::consts::PI;

#[cfg(not(feature = "std"))]
extern crate alloc;

#[cfg(not(feature = "std"))]
use alloc::{format, string::String, string::ToString, vec::Vec};

#[cfg(feature = "std")]
use thiserror::Error;

#[cfg(not(feature = "std"))]
use core::fmt;

/// Error types for particle dynamics
#[derive(Debug)]
#[cfg_attr(feature = "std", derive(Error))]
pub enum ParticleError {
    #[cfg_attr(feature = "std", error("Invalid particle parameters: {reason}"))]
    /// Invalid parameters provided for particle creation
    InvalidParameters {
        /// Description of invalid parameter
        reason: String,
    },

    #[cfg_attr(
        feature = "std",
        error("Energy {energy:.6e} J is insufficient for particle with mass {mass:.6e} kg")
    )]
    /// Energy is insufficient for particle with given mass
    InsufficientEnergy {
        /// Provided energy in Joules
        energy: f64,
        /// Particle mass in kg
        mass: f64,
    },

    #[cfg_attr(
        feature = "std",
        error("Velocity magnitude {velocity:.6e} m/s exceeds speed of light")
    )]
    /// Velocity exceeds speed of light
    SuperluminalVelocity {
        /// Velocity magnitude in m/s
        velocity: f64,
    },

    #[cfg_attr(feature = "std", error("Particle propagation failed: {reason}"))]
    /// Particle propagation through spacetime failed
    PropagationFailure {
        /// Description of propagation failure
        reason: String,
    },

    #[cfg_attr(
        feature = "std",
        error("Invalid charge-to-mass ratio: q/m = {charge_to_mass:.6e}")
    )]
    /// Charge-to-mass ratio is physically invalid
    InvalidChargeToMass {
        /// Charge-to-mass ratio in C/kg
        charge_to_mass: f64,
    },
}

/// Result type for particle operations
pub type ParticleResult<T> = Result<T, ParticleError>;

/// Trajectory point containing time, position, and velocity
pub type TrajectoryPoint = (f64, Vector3<f64>, Vector3<f64>);

/// Trajectory result type
pub type TrajectoryResult = ParticleResult<Vec<TrajectoryPoint>>;

#[cfg(not(feature = "std"))]
impl fmt::Display for ParticleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ParticleError::InvalidParameters { reason } => {
                write!(f, "Invalid particle parameters: {}", reason)
            }
            ParticleError::InsufficientEnergy { energy, mass } => {
                write!(
                    f,
                    "Energy {:.6e} J is insufficient for particle with mass {:.6e} kg",
                    energy, mass
                )
            }
            ParticleError::SuperluminalVelocity { velocity } => {
                write!(
                    f,
                    "Velocity magnitude {:.6e} m/s exceeds speed of light",
                    velocity
                )
            }
            ParticleError::PropagationFailure { reason } => {
                write!(f, "Particle propagation failed: {}", reason)
            }
            ParticleError::InvalidChargeToMass { charge_to_mass } => {
                write!(
                    f,
                    "Invalid charge-to-mass ratio: q/m = {:.6e}",
                    charge_to_mass
                )
            }
        }
    }
}

/// Relativistic particle with charge and mass
///
/// Represents a relativistic particle (massive or massless) with charge,
/// position, and velocity in spacetime. Handles proper relativistic
/// energy-momentum relations and four-velocity normalization.
///
/// # Examples
/// ```rust
/// use amari_relativistic::particle::RelativisticParticle;
/// use amari_relativistic::constants::{E_CHARGE, energy::KEV};
/// use nalgebra::Vector3;
///
/// // Create a 100 keV proton
/// let position = Vector3::new(1e6, 0.0, 0.0); // 1000 km from origin
/// let velocity = Vector3::new(1e6, 0.0, 0.0);  // 1000 km/s
/// let particle = RelativisticParticle::new(
///     position, velocity, 0.0,
///     1.67e-27, // proton mass
///     E_CHARGE  // proton charge
/// );
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct RelativisticParticle {
    /// Current spacetime position
    pub position: SpacetimeVector,

    /// Current four-velocity (normalized: u·u = c²)
    pub four_velocity: FourVelocity,

    /// Rest mass in kg (0 for photons)
    pub mass: f64,

    /// Electric charge in Coulombs
    pub charge: f64,

    /// Particle identifier (for tracking in simulations)
    pub id: Option<u64>,
}

impl RelativisticParticle {
    /// Create a new relativistic particle
    ///
    /// # Arguments
    /// * `position` - Initial 3D position in meters
    /// * `velocity` - Initial 3D velocity in m/s
    /// * `t` - Initial time coordinate in seconds
    /// * `mass` - Rest mass in kg (use 0 for photons)
    /// * `charge` - Electric charge in Coulombs
    ///
    /// # Returns
    /// `Ok(particle)` on success, `Err(ParticleError)` if parameters are invalid
    ///
    /// # Example
    /// ```rust
    /// use amari_relativistic::particle::RelativisticParticle;
    /// use amari_relativistic::constants::{E_CHARGE, atomic_masses::HYDROGEN};
    /// use nalgebra::Vector3;
    ///
    /// let pos = Vector3::new(0.0, 0.0, 1e6); // 1000 km altitude
    /// let vel = Vector3::new(1e4, 0.0, 0.0);  // 10 km/s
    /// let ion = RelativisticParticle::new(pos, vel, 0.0, HYDROGEN, E_CHARGE)?;
    /// # Ok::<(), amari_relativistic::particle::ParticleError>(())
    /// ```
    pub fn new(
        position: Vector3<f64>,
        velocity: Vector3<f64>,
        t: f64,
        mass: f64,
        charge: f64,
    ) -> ParticleResult<Self> {
        // Validate parameters
        if mass < 0.0 {
            return Err(ParticleError::InvalidParameters {
                reason: format!("Mass cannot be negative: {:.6e} kg", mass),
            });
        }

        let v_magnitude = velocity.magnitude();
        if v_magnitude >= C {
            return Err(ParticleError::SuperluminalVelocity {
                velocity: v_magnitude,
            });
        }

        // Create spacetime position
        let spacetime_position = SpacetimeVector::from_position_and_time(position, t);

        // Create four-velocity
        let four_velocity = if mass > 1e-30 {
            // Massive particle
            FourVelocity::from_velocity(velocity)
        } else {
            // Massless particle (photon) - create null four-velocity
            // For photons, we need |u|² = 0, not c²
            // This is a special case that requires careful handling
            if v_magnitude > 1e-10 {
                let direction = velocity / v_magnitude;
                FourVelocity::from_velocity(direction * (C - 1.0)) // Approach light speed
            } else {
                return Err(ParticleError::InvalidParameters {
                    reason: "Massless particle must have non-zero velocity".to_string(),
                });
            }
        };

        Ok(Self {
            position: spacetime_position,
            four_velocity,
            mass,
            charge,
            id: None,
        })
    }

    /// Create particle with specific kinetic energy
    ///
    /// Uses the relativistic energy-momentum relation to determine the
    /// appropriate velocity for the given kinetic energy.
    ///
    /// # Arguments
    /// * `position` - Initial 3D position in meters
    /// * `direction` - Velocity direction (will be normalized)
    /// * `kinetic_energy_j` - Kinetic energy in Joules
    /// * `mass` - Rest mass in kg
    /// * `charge` - Electric charge in Coulombs
    ///
    /// # Mathematical Background
    /// For relativistic particles:
    /// - Total energy: E = γmc²
    /// - Kinetic energy: T = (γ - 1)mc²
    /// - Velocity: v = c√(1 - 1/γ²) where γ = 1 + T/(mc²)
    ///
    /// # Example
    /// ```rust
    /// use amari_relativistic::particle::RelativisticParticle;
    /// use amari_relativistic::constants::{E_CHARGE, energy::KEV, atomic_masses::IRON_56};
    /// use nalgebra::Vector3;
    ///
    /// let pos = Vector3::new(2e11, 0.0, 0.0); // 2e11 m from center
    /// let dir = Vector3::new(-1.0, 0.0, 0.1); // Direction vector
    /// let iron_ion = RelativisticParticle::with_energy(
    ///     pos, dir, 100.0 * KEV, IRON_56, E_CHARGE
    /// )?;
    /// # Ok::<(), amari_relativistic::particle::ParticleError>(())
    /// ```
    pub fn with_energy(
        position: Vector3<f64>,
        direction: Vector3<f64>,
        kinetic_energy_j: f64,
        mass: f64,
        charge: f64,
    ) -> ParticleResult<Self> {
        if kinetic_energy_j < 0.0 {
            return Err(ParticleError::InvalidParameters {
                reason: format!(
                    "Kinetic energy cannot be negative: {:.6e} J",
                    kinetic_energy_j
                ),
            });
        }

        if mass <= 0.0 {
            return Err(ParticleError::InvalidParameters {
                reason: format!(
                    "Mass must be positive for energy calculation: {:.6e} kg",
                    mass
                ),
            });
        }

        let direction_normalized = if direction.magnitude() > 1e-14 {
            direction.normalize()
        } else {
            return Err(ParticleError::InvalidParameters {
                reason: "Direction vector cannot be zero".to_string(),
            });
        };

        // Calculate relativistic velocity from kinetic energy
        // T = (γ - 1)mc² → γ = 1 + T/(mc²)
        let rest_energy = mass * C * C;
        let gamma = 1.0 + kinetic_energy_j / rest_energy;

        if gamma < 1.0 {
            return Err(ParticleError::InsufficientEnergy {
                energy: kinetic_energy_j,
                mass,
            });
        }

        // v = c√(1 - 1/γ²)
        let beta_squared = 1.0 - 1.0 / (gamma * gamma);
        if !(0.0..1.0).contains(&beta_squared) {
            return Err(ParticleError::InvalidParameters {
                reason: format!("Invalid β² = {:.6e} from γ = {:.6e}", beta_squared, gamma),
            });
        }

        let velocity_magnitude = C * beta_squared.sqrt();
        let velocity = direction_normalized * velocity_magnitude;

        Self::new(position, velocity, 0.0, mass, charge)
    }

    /// Create an electron with given energy
    pub fn electron(position: Vector3<f64>, kinetic_energy_j: f64) -> ParticleResult<Self> {
        use crate::constants::ELECTRON_MASS;
        let direction = Vector3::new(1.0, 0.0, 0.0); // Default direction
        Self::with_energy(
            position,
            direction,
            kinetic_energy_j,
            ELECTRON_MASS,
            -E_CHARGE,
        )
    }

    /// Create a proton with given energy
    pub fn proton(position: Vector3<f64>, kinetic_energy_j: f64) -> ParticleResult<Self> {
        use crate::constants::PROTON_MASS;
        let direction = Vector3::new(1.0, 0.0, 0.0); // Default direction
        Self::with_energy(position, direction, kinetic_energy_j, PROTON_MASS, E_CHARGE)
    }

    /// Create an iron ion with given energy
    pub fn iron_ion(
        position: Vector3<f64>,
        kinetic_energy_j: f64,
        charge_state: i32,
    ) -> ParticleResult<Self> {
        use crate::constants::atomic_masses;
        let direction = Vector3::new(1.0, 0.0, 0.0); // Default direction
        let charge = charge_state as f64 * E_CHARGE;
        Self::with_energy(
            position,
            direction,
            kinetic_energy_j,
            atomic_masses::IRON_56,
            charge,
        )
    }

    /// Get the Lorentz factor γ = 1/√(1 - v²/c²)
    pub fn gamma(&self) -> f64 {
        self.four_velocity.gamma()
    }

    /// Get the relativistic factor β = v/c
    pub fn beta(&self) -> f64 {
        self.velocity_3d().magnitude() / C
    }

    /// Get kinetic energy T = (γ - 1)mc²
    pub fn kinetic_energy(&self) -> f64 {
        if self.mass > 1e-30 {
            (self.gamma() - 1.0) * self.mass * C * C
        } else {
            // For massless particles, kinetic energy equals total energy
            self.total_energy()
        }
    }

    /// Get total energy E = γmc²
    pub fn total_energy(&self) -> f64 {
        if self.mass > 1e-30 {
            self.gamma() * self.mass * C * C
        } else {
            // For massless particles: E = pc where p = hf/c for photons
            // We'll use the momentum magnitude times c
            self.momentum() * C
        }
    }

    /// Get relativistic momentum magnitude p = γmv
    pub fn momentum(&self) -> f64 {
        if self.mass > 1e-30 {
            self.gamma() * self.mass * self.velocity_3d().magnitude()
        } else {
            // For massless particles: p = E/c
            // Use a reasonable energy for photons based on their direction
            let energy = 1e-19; // Default photon energy ~1 eV
            energy / C
        }
    }

    /// Get 3D spatial position
    pub fn position_3d(&self) -> Vector3<f64> {
        self.position.spatial()
    }

    /// Get 3D velocity
    pub fn velocity_3d(&self) -> Vector3<f64> {
        self.four_velocity.velocity()
    }

    /// Get time coordinate in seconds
    pub fn time(&self) -> f64 {
        self.position.time()
    }

    /// Get rapidity φ where v = c tanh(φ)
    pub fn rapidity(&self) -> f64 {
        self.four_velocity.rapidity()
    }

    /// Get charge-to-mass ratio q/m
    pub fn charge_to_mass_ratio(&self) -> f64 {
        if self.mass > 1e-30 {
            self.charge / self.mass
        } else {
            f64::INFINITY // Massless charged particle (unphysical)
        }
    }

    /// Get the cyclotron frequency ωc = qB/m for uniform magnetic field
    ///
    /// # Arguments
    /// * `magnetic_field` - Magnetic field strength in Tesla
    ///
    /// # Returns
    /// Cyclotron frequency in rad/s, or None for massless particles
    pub fn cyclotron_frequency(&self, magnetic_field: f64) -> Option<f64> {
        if self.mass > 1e-30 {
            Some(self.charge.abs() * magnetic_field / self.mass)
        } else {
            None
        }
    }

    /// Get gyroradius for uniform magnetic field
    ///
    /// The gyroradius is: r_L = mv⊥/(qB) where v⊥ is velocity perpendicular to B
    ///
    /// # Arguments
    /// * `magnetic_field` - Magnetic field strength in Tesla
    /// * `field_direction` - Unit vector in field direction
    ///
    /// # Returns
    /// Gyroradius in meters, or None for massless particles
    pub fn gyroradius(&self, magnetic_field: f64, field_direction: Vector3<f64>) -> Option<f64> {
        if self.mass <= 1e-30 || magnetic_field.abs() < 1e-14 {
            return None;
        }

        let velocity = self.velocity_3d();
        let b_hat = field_direction.normalize();

        // Velocity perpendicular to magnetic field
        let v_parallel = velocity.dot(&b_hat) * b_hat;
        let v_perp = velocity - v_parallel;
        let v_perp_magnitude = v_perp.magnitude();

        if v_perp_magnitude < 1e-14 {
            return Some(0.0); // Motion purely parallel to field
        }

        let momentum_perp = self.gamma() * self.mass * v_perp_magnitude;
        Some(momentum_perp / (self.charge.abs() * magnetic_field))
    }

    /// Set particle ID for tracking
    pub fn set_id(&mut self, id: u64) {
        self.id = Some(id);
    }

    /// Update position (maintaining proper time synchronization)
    pub fn set_position(&mut self, position: Vector3<f64>, time: f64) {
        self.position = SpacetimeVector::from_position_and_time(position, time);
    }

    /// Update velocity (automatically updates four-velocity)
    pub fn set_velocity(&mut self, velocity: Vector3<f64>) -> ParticleResult<()> {
        if velocity.magnitude() >= C {
            return Err(ParticleError::SuperluminalVelocity {
                velocity: velocity.magnitude(),
            });
        }

        if self.mass > 1e-30 {
            self.four_velocity = FourVelocity::from_velocity(velocity);
        } else {
            // For massless particles, maintain null four-velocity constraint
            if velocity.magnitude() > 1e-10 {
                let direction = velocity.normalize();
                self.four_velocity = FourVelocity::from_velocity(direction * (C - 1.0));
            }
        }

        Ok(())
    }

    /// Check if particle is relativistic (v > 0.1c)
    pub fn is_relativistic(&self) -> bool {
        self.beta() > 0.1
    }

    /// Check if particle is ultra-relativistic (γ > 10)
    pub fn is_ultra_relativistic(&self) -> bool {
        self.gamma() > 10.0
    }

    /// Get de Broglie wavelength λ = h/p for quantum effects
    pub fn de_broglie_wavelength(&self) -> f64 {
        use crate::constants::H_PLANCK;
        H_PLANCK / self.momentum()
    }

    /// Check if classical mechanics is valid (λ_dB << characteristic length scale)
    pub fn is_classical(&self, length_scale: f64) -> bool {
        self.de_broglie_wavelength() < 0.01 * length_scale
    }
}

/// Propagate a relativistic particle through curved spacetime
///
/// Integrates the geodesic equation for the particle's worldline,
/// updating position and velocity according to the spacetime curvature.
///
/// # Arguments
/// * `particle` - Particle to propagate (modified in place)
/// * `integrator` - Geodesic integrator with spacetime metric
/// * `duration` - Time duration in seconds (coordinate time)
/// * `dtau` - Integration step size in proper time
///
/// # Returns
/// Trajectory as vector of (time, position, velocity) tuples
///
/// # Example
/// ```rust
/// use amari_relativistic::particle::{RelativisticParticle, propagate_relativistic};
/// use amari_relativistic::geodesic::GeodesicIntegrator;
/// use amari_relativistic::schwarzschild::SchwarzschildMetric;
/// use amari_relativistic::constants::{E_CHARGE, energy::KEV, atomic_masses::IRON_56};
/// use nalgebra::Vector3;
/// use std::sync::Arc;
///
/// // Create iron ion
/// let position = Vector3::new(2e11, 0.0, 0.0); // Large distance from center
/// let direction = Vector3::new(-1.0, 0.0, 0.1);
/// let mut ion = RelativisticParticle::with_energy(
///     position, direction, 100.0 * KEV, IRON_56, E_CHARGE
/// )?;
///
/// // Create Schwarzschild metric for massive object
/// let massive_object_metric = Box::new(SchwarzschildMetric::sun());
/// let mut integrator = GeodesicIntegrator::with_metric(massive_object_metric);
///
/// // Propagate through gravitational field
/// let trajectory = propagate_relativistic(
///     &mut ion, &mut integrator, 30.0 * 86400.0, 100.0
/// )?;
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn propagate_relativistic(
    particle: &mut RelativisticParticle,
    integrator: &mut GeodesicIntegrator,
    duration: f64,
    dtau: f64,
) -> TrajectoryResult {
    // Extract initial state
    let mut position = particle.position.clone();
    let mut four_velocity = particle.four_velocity.clone();

    // Perform integration
    let trajectory = integrator
        .propagate(&mut position, &mut four_velocity, duration, dtau)
        .map_err(|e| ParticleError::PropagationFailure {
            reason: e.to_string(),
        })?;

    // Update particle state
    particle.position = position;
    particle.four_velocity = four_velocity;

    // Convert trajectory to 3D format
    let trajectory_3d = trajectory
        .into_iter()
        .map(|(time, pos, vel)| (time, pos.spatial(), vel.velocity()))
        .collect();

    Ok(trajectory_3d)
}

/// Calculate deflection angle for light or particle passing near massive body
///
/// Uses the weak-field approximation for small deflections:
/// θ ≈ 4GM/(bc²) where b is the impact parameter
///
/// # Arguments
/// * `impact_parameter` - Closest approach distance in meters
/// * `mass` - Mass of deflecting body in kg
///
/// # Returns
/// Deflection angle in radians
pub fn light_deflection_angle(impact_parameter: f64, mass: f64) -> f64 {
    use crate::constants::G;

    if impact_parameter <= 1e-14 {
        return PI; // Head-on collision
    }

    // Einstein's light deflection formula (weak field approximation)
    4.0 * G * mass / (impact_parameter * C * C)
}

/// Calculate time delay for light passing near massive body
///
/// Shapiro time delay: Δt ≈ (2GM/c³) ln(r₁r₂/b²)
/// where r₁, r₂ are distances from mass to source and observer
///
/// # Arguments
/// * `r1` - Distance from mass to source in meters
/// * `r2` - Distance from mass to observer in meters
/// * `impact_parameter` - Closest approach distance in meters
/// * `mass` - Mass of deflecting body in kg
///
/// # Returns
/// Time delay in seconds
pub fn shapiro_time_delay(r1: f64, r2: f64, impact_parameter: f64, mass: f64) -> f64 {
    use crate::constants::G;

    if impact_parameter <= 1e-14 || r1 <= 1e-14 || r2 <= 1e-14 {
        return 0.0;
    }

    let gm = G * mass;
    2.0 * gm / (C * C * C) * (r1 * r2 / (impact_parameter * impact_parameter)).ln()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constants::{atomic_masses, energy};
    use approx::{assert_abs_diff_eq, assert_relative_eq};

    #[test]
    fn test_particle_creation() {
        let position = Vector3::new(1e6, 0.0, 0.0);
        let velocity = Vector3::new(1e5, 0.0, 0.0); // 100 km/s
        let mass = atomic_masses::HYDROGEN;

        let particle = RelativisticParticle::new(position, velocity, 0.0, mass, E_CHARGE);
        assert!(particle.is_ok());

        let p = particle.unwrap();
        assert_eq!(p.position_3d(), position);
        assert_relative_eq!(p.velocity_3d().magnitude(), 1e5, epsilon = 1e-10);
        assert_eq!(p.mass, mass);
        assert_eq!(p.charge, E_CHARGE);
    }

    #[test]
    fn test_energy_momentum_relations() {
        let position = Vector3::zeros();
        let velocity = Vector3::new(0.6 * C, 0.0, 0.0); // 0.6c
        let mass = atomic_masses::HYDROGEN;

        let particle = RelativisticParticle::new(position, velocity, 0.0, mass, E_CHARGE)
            .expect("Particle creation should succeed");

        // Test Lorentz factor
        let expected_gamma = 1.0 / (1.0 - 0.6_f64.powi(2)).sqrt();
        assert_relative_eq!(particle.gamma(), expected_gamma, epsilon = 1e-12);

        // Test energy-momentum relation: E² = (pc)² + (mc²)²
        let total_energy = particle.total_energy();
        let momentum = particle.momentum();
        let rest_energy = mass * C * C;

        let lhs = total_energy * total_energy;
        let rhs = (momentum * C) * (momentum * C) + rest_energy * rest_energy;
        assert_relative_eq!(lhs, rhs, epsilon = 1e-10);
    }

    #[test]
    fn test_particle_with_energy() {
        let position = Vector3::new(1e7, 0.0, 0.0);
        let direction = Vector3::new(-1.0, 0.0, 0.0);
        let kinetic_energy = 100.0 * energy::KEV;
        let mass = atomic_masses::IRON_56;

        let particle =
            RelativisticParticle::with_energy(position, direction, kinetic_energy, mass, E_CHARGE)
                .expect("Particle creation should succeed");

        // Check that kinetic energy matches
        assert_relative_eq!(particle.kinetic_energy(), kinetic_energy, epsilon = 1e-6);

        // Check that direction is correct
        let velocity = particle.velocity_3d();
        let velocity_direction = velocity.normalize();
        assert_relative_eq!(velocity_direction.x, -1.0, epsilon = 1e-12);
        assert_abs_diff_eq!(velocity_direction.y, 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(velocity_direction.z, 0.0, epsilon = 1e-12);
    }

    #[test]
    fn test_non_relativistic_limit() {
        let position = Vector3::zeros();
        let velocity = Vector3::new(1e4, 0.0, 0.0); // 10 km/s << c
        let mass = atomic_masses::HYDROGEN;

        let particle = RelativisticParticle::new(position, velocity, 0.0, mass, E_CHARGE)
            .expect("Particle creation should succeed");

        // γ should be very close to 1
        assert_relative_eq!(particle.gamma(), 1.0, epsilon = 1e-6);

        // Kinetic energy should be approximately ½mv²
        let classical_ke = 0.5 * mass * velocity.magnitude_squared();
        assert_relative_eq!(particle.kinetic_energy(), classical_ke, epsilon = 1e-8);
    }

    #[test]
    fn test_ultra_relativistic_limit() {
        let position = Vector3::zeros();
        let velocity = Vector3::new(0.995 * C, 0.0, 0.0); // 0.995c, γ ≈ 10
        let mass = atomic_masses::HYDROGEN;

        let particle = RelativisticParticle::new(position, velocity, 0.0, mass, E_CHARGE)
            .expect("Particle creation should succeed");

        assert!(particle.is_ultra_relativistic());

        // For ultra-relativistic particles, E ≈ pc
        let total_energy = particle.total_energy();
        let momentum_energy = particle.momentum() * C;
        assert_relative_eq!(total_energy, momentum_energy, epsilon = 0.1);
    }

    #[test]
    fn test_time_tracking() {
        // Test that time() method returns the correct time
        let position = Vector3::new(1e6, 0.0, 0.0);
        let velocity = Vector3::new(1e4, 0.0, 0.0);
        let initial_time = 42.0; // seconds
        let mass = atomic_masses::HYDROGEN;

        let particle = RelativisticParticle::new(position, velocity, initial_time, mass, E_CHARGE)
            .expect("Particle creation should succeed");

        // Verify time() method returns correct initial time
        assert_abs_diff_eq!(particle.time(), initial_time, epsilon = 1e-10);

        // Test set_position updates time
        let mut particle2 = particle.clone();
        let new_time = 100.0;
        particle2.set_position(position, new_time);
        assert_abs_diff_eq!(particle2.time(), new_time, epsilon = 1e-10);
    }

    #[test]
    fn test_propagation_time_update() {
        use crate::geodesic::GeodesicIntegrator;
        use crate::schwarzschild::SchwarzschildMetric;

        // Create particle at initial time
        let position = Vector3::new(2e11, 0.0, 0.0); // Far from center
        let velocity = Vector3::new(0.0, 1e4, 0.0); // 10 km/s tangential
        let initial_time = 0.0;
        let mass = atomic_masses::HYDROGEN;

        let mut particle =
            RelativisticParticle::new(position, velocity, initial_time, mass, E_CHARGE)
                .expect("Particle creation should succeed");

        // Create simple integrator (flat spacetime approximation far from mass)
        let central_mass = 1e30; // kg
        let center = Vector3::zeros();
        let metric = Box::new(SchwarzschildMetric::new(central_mass, center));
        let config = crate::geodesic::IntegrationConfig::default();
        let mut integrator = GeodesicIntegrator::new(metric, config);

        // Propagate for 100 seconds
        let duration = 100.0;
        let dtau = 1.0;

        let initial_particle_time = particle.time();
        let trajectory = propagate_relativistic(&mut particle, &mut integrator, duration, dtau)
            .expect("Propagation should succeed");

        // Check that particle's time has been updated
        let final_particle_time = particle.time();
        assert!(
            final_particle_time > initial_particle_time,
            "Particle time should increase after propagation"
        );

        // For non-relativistic speeds, proper time ≈ coordinate time
        // So final time should be approximately initial_time + duration
        assert_abs_diff_eq!(
            final_particle_time - initial_particle_time,
            duration,
            epsilon = 1.0
        ); // Allow 1 second tolerance

        // Check trajectory points have increasing time
        assert!(!trajectory.is_empty(), "Trajectory should not be empty");
        let mut prev_time = trajectory[0].0;
        for (time, _, _) in trajectory.iter().skip(1) {
            assert!(*time > prev_time, "Trajectory times should be increasing");
            prev_time = *time;
        }
    }

    #[test]
    fn test_gyroradius() {
        let position = Vector3::zeros();
        let velocity = Vector3::new(1e6, 0.0, 0.0); // 1000 km/s
        let mass = atomic_masses::HYDROGEN;

        let particle = RelativisticParticle::new(position, velocity, 0.0, mass, E_CHARGE)
            .expect("Particle creation should succeed");

        let magnetic_field = 1e-5; // 10 μT magnetic field
        let field_direction = Vector3::new(0.0, 0.0, 1.0); // Vertical

        let gyroradius = particle.gyroradius(magnetic_field, field_direction);
        assert!(gyroradius.is_some());

        let r_l = gyroradius.unwrap();

        // Check against classical formula: r_L = mv/(qB)
        let expected = mass * velocity.magnitude() / (E_CHARGE * magnetic_field);
        assert_relative_eq!(r_l, expected, epsilon = 1e-2);
    }

    #[test]
    fn test_cyclotron_frequency() {
        let position = Vector3::zeros();
        let velocity = Vector3::new(1e5, 0.0, 0.0);
        let mass = atomic_masses::HYDROGEN;

        let particle = RelativisticParticle::new(position, velocity, 0.0, mass, E_CHARGE)
            .expect("Particle creation should succeed");

        let magnetic_field = 1e-4; // 100 μT
        let omega_c = particle.cyclotron_frequency(magnetic_field);
        assert!(omega_c.is_some());

        let expected = E_CHARGE * magnetic_field / mass;
        assert_relative_eq!(omega_c.unwrap(), expected, epsilon = 1e-12);
    }

    #[test]
    fn test_light_deflection() {
        let object_radius = 6.957e8; // m
        let mass = crate::constants::SOLAR_MASS;

        // Light grazing the massive object's surface
        let deflection = light_deflection_angle(object_radius, mass);

        // Should be approximately 1.75 arcseconds = 8.48e-6 radians
        let expected_arcsec = 1.75;
        let deflection_arcsec = deflection * 180.0 * 3600.0 / PI;
        assert_relative_eq!(deflection_arcsec, expected_arcsec, epsilon = 0.1);
    }

    #[test]
    fn test_shapiro_delay() {
        let large_distance = 1.496e11; // Large distance unit
        let mass = crate::constants::SOLAR_MASS;
        let object_radius = 6.957e8;

        // Communication path past massive object
        let r1 = large_distance; // Distance to source
        let r2 = 1.5 * large_distance; // Distance to observer
        let impact = 2.0 * object_radius; // Grazing trajectory

        let delay = shapiro_time_delay(r1, r2, impact, mass);

        // Should be on the order of 100 microseconds
        assert!(delay > 1e-6 && delay < 1e-3);
    }

    #[test]
    fn test_superluminal_velocity_error() {
        let position = Vector3::zeros();
        let velocity = Vector3::new(2.0 * C, 0.0, 0.0); // 2c
        let mass = atomic_masses::HYDROGEN;

        let result = RelativisticParticle::new(position, velocity, 0.0, mass, E_CHARGE);
        assert!(result.is_err());
        match result {
            Err(ParticleError::SuperluminalVelocity { velocity: v }) => {
                assert!(v > C);
            }
            _ => panic!("Expected SuperluminalVelocity error"),
        }
    }

    #[test]
    fn test_invalid_mass() {
        let position = Vector3::zeros();
        let velocity = Vector3::new(1e5, 0.0, 0.0);

        let result = RelativisticParticle::new(position, velocity, 0.0, -1.0, E_CHARGE);
        assert!(matches!(
            result,
            Err(ParticleError::InvalidParameters { .. })
        ));
    }

    #[test]
    fn test_de_broglie_wavelength() {
        let position = Vector3::zeros();
        let velocity = Vector3::new(1e6, 0.0, 0.0); // 1000 km/s
        let mass = atomic_masses::HYDROGEN;

        let particle = RelativisticParticle::new(position, velocity, 0.0, mass, E_CHARGE)
            .expect("Particle creation should succeed");

        let lambda_db = particle.de_broglie_wavelength();

        // Should be on the order of 10^-12 m for keV particles
        assert!(lambda_db > 1e-15 && lambda_db < 1e-9);

        // Check against expected formula: λ = h/p
        let expected = crate::constants::H_PLANCK / particle.momentum();
        assert_relative_eq!(lambda_db, expected, epsilon = 1e-12);
    }
}
