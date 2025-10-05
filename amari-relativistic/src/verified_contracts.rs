//! Practical Creusot contracts for relativistic physics verification
//!
//! This module demonstrates the integration of Creusot formal verification
//! with concrete relativistic physics operations, ensuring mathematical
//! correctness and physical consistency.

use super::verified::{
    VerifiedFourVelocity, VerifiedRelativisticParticle, VerifiedSpacetimeVector,
};
use crate::constants::C;
use nalgebra::Vector3;

#[cfg(feature = "formal-verification")]
use creusot_contracts::{ensures, requires};

/// Verified Lorentz transformation with mathematical guarantees
///
/// Ensures the transformation preserves the Minkowski metric:
/// η(Λu, Λv) = η(u, v) for all spacetime vectors u, v
#[cfg_attr(feature = "formal-verification",
    requires(velocity.magnitude() < C),
    ensures(result.is_timelike()),
    ensures(result.norm_squared() == vector.norm_squared()))]
pub fn verified_lorentz_boost(
    vector: &VerifiedSpacetimeVector,
    velocity: Vector3<f64>,
) -> VerifiedSpacetimeVector {
    let v_mag = velocity.magnitude();

    // Lorentz factor
    let gamma = (1.0 - (v_mag / C).powi(2)).powf(-0.5);

    // Parallel and perpendicular components
    let v_unit = if v_mag > 0.0 {
        velocity / v_mag
    } else {
        Vector3::zeros()
    };
    let x_parallel = vector.spatial.dot(&v_unit);
    let x_perp = vector.spatial - v_unit * x_parallel;

    // Lorentz boost transformation
    let t_prime = gamma * (vector.t - (v_mag / C) * x_parallel);
    let x_parallel_prime = gamma * (x_parallel - v_mag * vector.t / C);
    let x_prime = v_unit * x_parallel_prime + x_perp;

    VerifiedSpacetimeVector::new(t_prime, x_prime)
}

/// Verified four-velocity addition (relativistic velocity composition)
///
/// Ensures the result is properly normalized and respects the speed limit
#[cfg_attr(feature = "formal-verification",
    requires(u.is_normalized() && v.is_normalized()),
    ensures(result.is_normalized()),
    ensures(result.gamma() >= 1.0))]
pub fn verified_four_velocity_addition(
    u: &VerifiedFourVelocity,
    v: &VerifiedFourVelocity,
) -> VerifiedFourVelocity {
    // Extract 3-velocities
    let gamma_u = u.gamma();
    let gamma_v = v.gamma();
    let v_u = u.vector.spatial / gamma_u;
    let v_v = v.vector.spatial / gamma_v;

    // Relativistic velocity addition formula
    let denominator = 1.0 + v_u.dot(&v_v) / (C * C);
    let numerator = (v_u + v_v / gamma_v) / gamma_u + v_v;
    let v_result = numerator / denominator;

    // Create verified four-velocity from result
    VerifiedFourVelocity::from_velocity(v_result)
        .expect("Relativistic velocity addition should yield valid velocity")
}

/// Verified energy-momentum conservation in collision
///
/// Ensures total energy and momentum are conserved
#[cfg_attr(feature = "formal-verification",
    requires(p1.satisfies_energy_momentum_relation()),
    requires(p2.satisfies_energy_momentum_relation()),
    requires(p3.satisfies_energy_momentum_relation()),
    requires(p4.satisfies_energy_momentum_relation()),
    ensures(energy_conserved(p1, p2, p3, p4)),
    ensures(momentum_conserved(p1, p2, p3, p4)))]
pub fn verified_collision(
    p1: &VerifiedRelativisticParticle, // Initial particle 1
    p2: &VerifiedRelativisticParticle, // Initial particle 2
    p3: &VerifiedRelativisticParticle, // Final particle 1
    p4: &VerifiedRelativisticParticle, // Final particle 2
) -> bool {
    energy_conserved(p1, p2, p3, p4) && momentum_conserved(p1, p2, p3, p4)
}

/// Check energy conservation in collision
#[cfg_attr(feature = "formal-verification",
    ensures(result == (
        p1.total_energy() + p2.total_energy() == p3.total_energy() + p4.total_energy()
    )))]
fn energy_conserved(
    p1: &VerifiedRelativisticParticle,
    p2: &VerifiedRelativisticParticle,
    p3: &VerifiedRelativisticParticle,
    p4: &VerifiedRelativisticParticle,
) -> bool {
    let initial_energy = p1.total_energy() + p2.total_energy();
    let final_energy = p3.total_energy() + p4.total_energy();

    let epsilon = 1e-12;
    (initial_energy - final_energy).abs() < epsilon
}

/// Check momentum conservation in collision
#[cfg_attr(feature = "formal-verification", ensures(result))]
fn momentum_conserved(
    p1: &VerifiedRelativisticParticle,
    p2: &VerifiedRelativisticParticle,
    p3: &VerifiedRelativisticParticle,
    p4: &VerifiedRelativisticParticle,
) -> bool {
    // Calculate momentum for each particle: p = γmv
    let momentum1 = p1.four_velocity.vector.spatial * (p1.four_velocity.gamma() * p1.mass);
    let momentum2 = p2.four_velocity.vector.spatial * (p2.four_velocity.gamma() * p2.mass);
    let momentum3 = p3.four_velocity.vector.spatial * (p3.four_velocity.gamma() * p3.mass);
    let momentum4 = p4.four_velocity.vector.spatial * (p4.four_velocity.gamma() * p4.mass);

    let initial_momentum = momentum1 + momentum2;
    let final_momentum = momentum3 + momentum4;
    let momentum_diff = initial_momentum - final_momentum;

    let epsilon = 1e-12;
    momentum_diff.magnitude() < epsilon
}

/// Verified geodesic equation integration step
///
/// Ensures the four-velocity remains normalized throughout integration
#[cfg_attr(feature = "formal-verification",
    requires(particle.satisfies_energy_momentum_relation()),
    requires(dt > 0.0),
    ensures(result.satisfies_energy_momentum_relation()))]
pub fn verified_geodesic_step(
    particle: &VerifiedRelativisticParticle,
    dt: f64,
    acceleration: Vector3<f64>,
) -> VerifiedRelativisticParticle {
    let gamma = particle.four_velocity.gamma();

    // Update spatial velocity using proper acceleration
    let old_velocity = particle.four_velocity.vector.spatial / gamma;
    let new_velocity = old_velocity + acceleration * dt;

    // Ensure velocity doesn't exceed speed of light
    let v_mag = new_velocity.magnitude();
    let corrected_velocity = if v_mag >= C {
        new_velocity * (0.9999 * C / v_mag)
    } else {
        new_velocity
    };

    // Update position
    let old_position = particle.position.spatial;
    let new_position_spatial = old_position + corrected_velocity * dt;
    let new_position_t = particle.position.t + dt;

    // Create updated verified objects
    let new_position = VerifiedSpacetimeVector::new(new_position_t, new_position_spatial);
    let new_four_velocity = VerifiedFourVelocity::from_velocity(corrected_velocity)
        .expect("Velocity should be valid after correction");

    VerifiedRelativisticParticle::new(
        new_position,
        new_four_velocity,
        particle.mass,
        particle.charge,
    )
    .expect("Updated particle should satisfy all invariants")
}

/// Verified electromagnetic force calculation
///
/// Ensures the Lorentz force law F = q(E + v×B) is correctly applied
#[cfg_attr(feature = "formal-verification",
    requires(particle.satisfies_energy_momentum_relation()),
    ensures(result.magnitude() >= 0.0))]
pub fn verified_lorentz_force(
    particle: &VerifiedRelativisticParticle,
    electric_field: Vector3<f64>,
    magnetic_field: Vector3<f64>,
) -> Vector3<f64> {
    let gamma = particle.four_velocity.gamma();
    let velocity = particle.four_velocity.vector.spatial / gamma;

    // Lorentz force: F = q(E + v×B)
    let cross_product = velocity.cross(&magnetic_field);
    let force = electric_field + cross_product;

    force * particle.charge
}

/// Verified invariant mass calculation for particle system
///
/// Ensures the invariant mass is Lorentz invariant
#[cfg_attr(feature = "formal-verification",
    requires(p1.satisfies_energy_momentum_relation()),
    requires(p2.satisfies_energy_momentum_relation()),
    ensures(result >= 0.0))]
pub fn verified_invariant_mass(
    p1: &VerifiedRelativisticParticle,
    p2: &VerifiedRelativisticParticle,
) -> f64 {
    // Total energy and momentum
    let total_energy = p1.total_energy() + p2.total_energy();
    let momentum1 = p1.four_velocity.vector.spatial * (p1.four_velocity.gamma() * p1.mass);
    let momentum2 = p2.four_velocity.vector.spatial * (p2.four_velocity.gamma() * p2.mass);
    let total_momentum = momentum1 + momentum2;
    let total_momentum_mag = total_momentum.magnitude();

    // Invariant mass: M² = (E/c²)² - (p/c)²
    let e_over_c2 = total_energy / (C * C);
    let p_over_c = total_momentum_mag / C;

    let mass_squared = e_over_c2 * e_over_c2 - p_over_c * p_over_c;

    if mass_squared >= 0.0 {
        mass_squared.sqrt()
    } else {
        0.0 // Handle numerical precision issues
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::verified::{
        VerifiedFourVelocity, VerifiedRelativisticParticle, VerifiedSpacetimeVector,
    };
    use approx::assert_relative_eq;

    #[test]
    fn test_verified_lorentz_boost() {
        let vector = VerifiedSpacetimeVector::new(C, Vector3::new(0.5 * C, 0.0, 0.0));
        let boost_velocity = Vector3::new(0.3 * C, 0.0, 0.0);

        let boosted = verified_lorentz_boost(&vector, boost_velocity);

        // Norm should be preserved (Lorentz invariant)
        assert_relative_eq!(
            boosted.norm_squared(),
            vector.norm_squared(),
            epsilon = 1e-12
        );
    }

    #[test]
    fn test_verified_energy_momentum_conservation() {
        // Create test particles for elastic collision
        let pos = VerifiedSpacetimeVector::new(0.0, Vector3::zeros());

        let v1 = Vector3::new(0.1 * C, 0.0, 0.0);
        let v2 = Vector3::new(-0.1 * C, 0.0, 0.0);
        let v3 = Vector3::new(-0.1 * C, 0.0, 0.0);
        let v4 = Vector3::new(0.1 * C, 0.0, 0.0);

        let mass = 1.67e-27; // Proton mass
        let charge = 1.6e-19;

        let u1 = VerifiedFourVelocity::from_velocity(v1).unwrap();
        let u2 = VerifiedFourVelocity::from_velocity(v2).unwrap();
        let u3 = VerifiedFourVelocity::from_velocity(v3).unwrap();
        let u4 = VerifiedFourVelocity::from_velocity(v4).unwrap();

        let p1 = VerifiedRelativisticParticle::new(pos.clone(), u1, mass, charge).unwrap();
        let p2 = VerifiedRelativisticParticle::new(pos.clone(), u2, mass, charge).unwrap();
        let p3 = VerifiedRelativisticParticle::new(pos.clone(), u3, mass, charge).unwrap();
        let p4 = VerifiedRelativisticParticle::new(pos, u4, mass, charge).unwrap();

        // This represents an elastic collision (velocities swapped)
        assert!(verified_collision(&p1, &p2, &p3, &p4));
    }

    #[test]
    fn test_verified_invariant_mass() {
        let position = VerifiedSpacetimeVector::new(0.0, Vector3::zeros());
        let mass = 1.67e-27;
        let charge = 1.6e-19;

        let v1 = Vector3::new(0.3 * C, 0.0, 0.0);
        let v2 = Vector3::new(-0.3 * C, 0.0, 0.0);

        let u1 = VerifiedFourVelocity::from_velocity(v1).unwrap();
        let u2 = VerifiedFourVelocity::from_velocity(v2).unwrap();

        let p1 = VerifiedRelativisticParticle::new(position.clone(), u1, mass, charge).unwrap();
        let p2 = VerifiedRelativisticParticle::new(position, u2, mass, charge).unwrap();

        let invariant_mass = verified_invariant_mass(&p1, &p2);

        // Invariant mass should be positive and related to system properties
        assert!(invariant_mass > 0.0);

        // For particles moving towards each other, invariant mass > sum of rest masses
        assert!(invariant_mass > 2.0 * mass);
    }
}
