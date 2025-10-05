//! # Amari Relativistic Physics Library
//!
//! A comprehensive Rust library for relativistic physics simulations using geometric algebra,
//! designed for charged particle simulations and plasma physics applications involving
//! particle trajectories through curved spacetime.
//!
//! ## Overview
//!
//! This crate leverages the geometric algebra (Clifford algebra) framework from `amari-core`
//! to provide a mathematically rigorous foundation for relativistic physics calculations.
//! It implements spacetime algebra (STA) for four-dimensional spacetime, geodesic integration
//! for curved spacetime, and relativistic particle dynamics.
//!
//! ## Key Features
//!
//! - **Spacetime Algebra**: Four-vectors, Lorentz transformations using geometric algebra
//! - **Geodesic Integration**: Numerical integration of Einstein's geodesic equation
//! - **Schwarzschild Metric**: Spacetime around spherically symmetric masses
//! - **Relativistic Particles**: Energy-momentum relations, proper time evolution
//! - **Charged Particle Applications**: Ion beam trajectories, particle deflection
//!
//! ## Mathematical Foundation
//!
//! The library is built on spacetime algebra Cl(1,3) with signature (+---):
//! - γ₀² = +1 (timelike basis vector)
//! - γ₁², γ₂², γ₃² = -1 (spacelike basis vectors)
//! - Four-vectors: X = ctγ₀ + xγ₁ + yγ₂ + zγ₃
//! - Minkowski inner product: X·Y = X₀Y₀ - X₁Y₁ - X₂Y₂ - X₃Y₃
//!
//! ## Example Usage
//!
//! ### Basic Charged Particle Trajectory Simulation
//!
//! ```rust
//! use amari_relativistic::*;
//! use nalgebra::Vector3;
//!
//! // Create gravitational field from massive object
//! let massive_object = schwarzschild::SchwarzschildMetric::sun();
//! let mut integrator = geodesic::GeodesicIntegrator::with_metric(Box::new(massive_object));
//!
//! // Create 100 keV iron ion in strong gravitational field
//! let position = Vector3::new(2e11, 0.0, 0.0); // Distance from center
//! let direction = Vector3::new(-1.0, 0.0, 0.1); // Initial trajectory
//! let mut ion = particle::RelativisticParticle::with_energy(
//!     position,
//!     direction,
//!     100e3 * constants::E_CHARGE, // 100 keV
//!     55.845 * constants::AMU,     // Iron mass
//!     constants::E_CHARGE,         // Singly ionized
//! )?;
//!
//! // Propagate through gravitational field
//! let trajectory = particle::propagate_relativistic(
//!     &mut ion,
//!     &mut integrator,
//!     30.0 * 86400.0,  // Integration time
//!     100.0,            // Time step size
//! )?;
//!
//! println!("Final position: {:?}", ion.position_3d());
//! println!("Deflection: {:.2e} m",
//!          trajectory.last().unwrap().1.magnitude() - position.magnitude());
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! ### Light Deflection by Massive Objects
//!
//! ```rust
//! use amari_relativistic::*;
//!
//! // Calculate light deflection for photon grazing massive object surface
//! let object_radius = 6.957e8; // meters
//! let deflection_angle = particle::light_deflection_angle(
//!     object_radius,
//!     constants::SOLAR_MASS
//! );
//!
//! // Convert to arcseconds
//! let arcseconds = deflection_angle * 180.0 * 3600.0 / std::f64::consts::PI;
//! println!("Light deflection: {:.2} arcseconds", arcseconds);
//! // Expected: ~1.75 arcseconds (Einstein's prediction)
//! ```
//!
//! ### Four-Velocity and Spacetime Intervals
//!
//! ```rust
//! use amari_relativistic::spacetime::*;
//! use nalgebra::Vector3;
//!
//! // Create relativistic four-velocity
//! let velocity = Vector3::new(0.8 * constants::C, 0.0, 0.0); // 0.8c
//! let four_vel = FourVelocity::from_velocity(velocity);
//!
//! println!("Lorentz factor γ = {:.3}", four_vel.gamma());
//! println!("Rapidity φ = {:.3}", four_vel.rapidity());
//!
//! // Spacetime interval between events
//! let event1 = SpacetimeVector::new(0.0, 0.0, 0.0, 0.0);
//! let event2 = SpacetimeVector::new(1.0, 0.5 * constants::C, 0.0, 0.0);
//! let interval = event1.minkowski_dot(&event2);
//!
//! if interval > 0.0 {
//!     println!("Timelike separation: Δτ = {:.6} s",
//!              interval.sqrt() / constants::C);
//! }
//! ```
//!
//! ### Compile-Time Verification with Phantom Types
//!
//! When the `phantom-types` feature is enabled (default), you get compile-time guarantees:
//!
//! ```rust
//! # #[cfg(feature = "phantom-types")] {
//! use amari_relativistic::prelude::*;
//!
//! // Verified spacetime vector with compile-time signature guarantees
//! let position = VerifiedSpacetimeVector::new(0.0, Vector3::zeros());
//! let velocity = Vector3::new(0.6 * C, 0.0, 0.0);
//!
//! // Four-velocity is automatically normalized and verified at creation
//! let four_vel = VerifiedFourVelocity::from_velocity(velocity)?;
//! assert!(four_vel.is_normalized()); // Guaranteed by phantom types
//!
//! // Particles must satisfy energy-momentum relation E² = (pc)² + (mc²)²
//! let particle = VerifiedRelativisticParticle::new(
//!     position, four_vel, 9.109e-31, -1.602e-19
//! )?;
//! assert!(particle.satisfies_energy_momentum_relation()); // Type-guaranteed
//! # }
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! ### Formal Verification with Creusot Contracts
//!
//! With the `formal-verification` feature on nightly Rust:
//!
//! ```rust,ignore
//! use amari_relativistic::prelude::*;
//!
//! // Functions with mathematical guarantees proven at compile time
//! let boosted = verified_lorentz_boost(&vector, boost_velocity);
//! // Guaranteed: Minkowski norm preserved by Lorentz transformation
//!
//! let collision_valid = verified_collision(&p1, &p2, &p3, &p4);
//! // Guaranteed: Energy and momentum conservation in particle collisions
//! ```
//!
//! ## Application Domains
//!
//! ### Charged Particle Simulations
//! - Ion beam focusing and deflection
//! - Particle accelerator beam dynamics
//! - Plasma physics particle trajectories
//! - Industrial ion implantation processes
//!
//! ### Astrophysics
//! - Planetary motion around compact objects
//! - Photon trajectories near black holes
//! - Gravitational lensing calculations
//! - Tidal effects and frame dragging
//!
//! ### Fundamental Physics
//! - Tests of general relativity
//! - Precision orbit determination
//! - Relativistic corrections to classical mechanics
//! - Coordinate transformation verification
//!
//! ## Performance Considerations
//!
//! - Uses `#[inline]` for hot path functions
//! - Leverages SIMD through `amari-core` geometric algebra
//! - Symplectic integrators preserve energy conservation
//! - Adaptive step sizing for numerical stability
//!
//! ## Integration with Amari Ecosystem
//!
//! This crate builds upon:
//! - [`amari-core`]: Geometric algebra operations and multivectors
//! - [`amari-gpu`]: GPU acceleration for large-scale simulations
//! - [`amari-dual`]: Automatic differentiation for optimization
//! - [`amari-info-geom`]: Information geometry for parameter estimation
//!
//! ## References
//!
//! - Misner, Thorne & Wheeler, "Gravitation" (1973)
//! - Doran & Lasenby, "Geometric Algebra for Physicists" (2003)
//! - Jackson, "Classical Electrodynamics" (1999)
//! - Weinberg, "Gravitation and Cosmology" (1972)

#![cfg_attr(not(feature = "std"), no_std)]
#![warn(missing_docs)]
#![warn(clippy::missing_docs_in_private_items)]
#![doc(
    html_logo_url = "https://raw.githubusercontent.com/justinelliottcobb/Amari/main/logo.png",
    html_favicon_url = "https://raw.githubusercontent.com/justinelliottcobb/Amari/main/favicon.ico"
)]

// Re-export key types for convenience
pub use constants::*;
pub use geodesic::{GeodesicIntegrator, GeodesicResult, IntegrationConfig, Metric};
pub use particle::light_deflection_angle;
pub use particle::{
    propagate_relativistic, ParticleError, ParticleResult, RelativisticParticle, TrajectoryPoint,
    TrajectoryResult,
};
pub use schwarzschild::{effective_potential, SchwarzschildMetric};
pub use spacetime::{FourVelocity, LorentzRotor, SpacetimeVector};

// Re-export phantom types and verified constructs when features are enabled
#[cfg(feature = "phantom-types")]
pub use verified::{
    EnergyMomentumInvariant, FourVelocityNorm, MinkowskiMetric, SpacetimeSignature, ValidSpacetime,
    VerifiedFourVelocity, VerifiedRelativisticParticle, VerifiedSpacetimeVector,
};

#[cfg(feature = "formal-verification")]
pub use verified_contracts::{
    verified_collision, verified_four_velocity_addition, verified_geodesic_step,
    verified_invariant_mass, verified_lorentz_boost, verified_lorentz_force,
};

/// Physical constants for relativistic physics
pub mod constants;

/// Spacetime algebra implementation using geometric algebra
pub mod spacetime;

/// Geodesic integration for curved spacetime
pub mod geodesic;

/// Schwarzschild metric for spherically symmetric gravitational fields
pub mod schwarzschild;

/// Relativistic charged particle dynamics
pub mod particle;

#[cfg(feature = "phantom-types")]
/// Formally verified relativistic physics with phantom types
pub mod verified;

#[cfg(feature = "formal-verification")]
/// Creusot contracts for formal verification
pub mod verified_contracts;

/// Common error types used throughout the library
pub mod error {
    pub use crate::geodesic::GeodesicError;
    pub use crate::particle::ParticleError;
}

/// Common result types used throughout the library
pub mod result {
    pub use crate::geodesic::GeodesicResult;
    pub use crate::particle::ParticleResult;
}

/// Prelude module for common imports
///
/// Common imports for relativistic physics calculations
///
/// This prelude module re-exports the most commonly used types and functions
/// for convenient importing with `use amari_relativistic::prelude::*;`
pub mod prelude {

    // Core constants
    pub use crate::constants::{AMU, C, E_CHARGE, G, SOLAR_MASS};

    // Core spacetime types
    pub use crate::spacetime::{FourVelocity, LorentzRotor, SpacetimeVector};

    // Particle physics
    pub use crate::particle::{
        light_deflection_angle, propagate_relativistic, ParticleError, ParticleResult,
        RelativisticParticle, TrajectoryPoint, TrajectoryResult,
    };

    // Geodesic integration
    pub use crate::geodesic::{GeodesicIntegrator, GeodesicResult, IntegrationConfig, Metric};

    // Schwarzschild geometry
    pub use crate::schwarzschild::{effective_potential, SchwarzschildMetric};

    // External dependencies commonly used
    pub use nalgebra::Vector3;

    // Phantom types for compile-time verification (stable Rust)
    #[cfg(feature = "phantom-types")]
    pub use crate::verified::{
        EnergyMomentumInvariant, FourVelocityNorm, MinkowskiMetric, SpacetimeSignature,
        ValidSpacetime, VerifiedFourVelocity, VerifiedRelativisticParticle,
        VerifiedSpacetimeVector,
    };

    // Formal verification contracts (nightly Rust)
    #[cfg(feature = "formal-verification")]
    pub use crate::verified_contracts::{
        verified_collision, verified_four_velocity_addition, verified_geodesic_step,
        verified_invariant_mass, verified_lorentz_boost, verified_lorentz_force,
    };
}

// Version information
/// Current version of the amari-relativistic crate
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Library information
pub const INFO: &str = concat!(
    "Amari Relativistic Physics v",
    env!("CARGO_PKG_VERSION"),
    " - Relativistic physics using geometric algebra"
);

#[cfg(test)]
mod integration_tests {
    //! Integration tests demonstrating complete workflows
    use super::prelude::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_complete_ion_simulation() {
        // Complete simulation: 100 keV iron ion near Sun
        let sun = SchwarzschildMetric::sun();
        let mut integrator = GeodesicIntegrator::with_metric(Box::new(sun));

        // Ion starting at 2 AU, aimed toward Sun with small impact parameter
        let position = Vector3::new(2.0 * 1.496e11, 0.0, 0.0);
        let direction = Vector3::new(-1.0, 0.0, 0.1);
        let mut ion = RelativisticParticle::with_energy(
            position,
            direction,
            100e3 * E_CHARGE, // 100 keV
            55.845 * AMU,     // Iron-56
            E_CHARGE,         // Singly ionized
        )
        .expect("Ion creation should succeed");

        // Verify initial conditions
        assert_relative_eq!(ion.kinetic_energy(), 100e3 * E_CHARGE, epsilon = 1e-6);
        assert!(ion.position_3d().magnitude() > 2.9e11); // > 1.9 AU

        // Short propagation to test integration
        let trajectory = propagate_relativistic(
            &mut ion,
            &mut integrator,
            86400.0, // 1 day
            3600.0,  // 1 hour steps
        )
        .expect("Propagation should succeed");

        // Verify trajectory properties
        assert!(trajectory.len() > 10);
        assert!(trajectory.last().unwrap().0 > 80000.0); // Time advanced

        // Ion should have moved closer to Sun
        let final_distance = ion.position_3d().magnitude();
        let initial_distance = position.magnitude();
        assert!(final_distance < initial_distance);
    }

    #[test]
    fn test_light_deflection_calculation() {
        // Test Einstein's light deflection prediction
        let solar_radius = 6.957e8;
        let deflection = crate::particle::light_deflection_angle(solar_radius, SOLAR_MASS);

        // Convert to arcseconds
        let arcseconds = deflection * 180.0 * 3600.0 / std::f64::consts::PI;

        // Should be approximately 1.75 arcseconds
        assert_relative_eq!(arcseconds, 1.75, epsilon = 0.1);
    }

    #[test]
    fn test_four_velocity_consistency() {
        // Test four-velocity normalization and consistency
        let velocity = Vector3::new(0.6 * C, 0.8 * C, 0.0);

        // This should fail because |v| > c
        assert!(velocity.magnitude() > C);

        // Test with valid velocity
        let valid_velocity = Vector3::new(0.36 * C, 0.48 * C, 0.0); // |v| = 0.6c
        let four_vel = FourVelocity::from_velocity(valid_velocity);

        // Check normalization: u·u = c²
        let norm_sq = four_vel.as_spacetime_vector().minkowski_norm_squared();
        assert_relative_eq!(norm_sq, C * C, epsilon = 1e-10);

        // Check Lorentz factor
        let expected_gamma = 1.0 / (1.0 - 0.6_f64.powi(2)).sqrt();
        assert_relative_eq!(four_vel.gamma(), expected_gamma, epsilon = 1e-12);
    }

    #[test]
    fn test_energy_conservation_in_flat_spacetime() {
        // Test energy conservation in flat spacetime (should be exact)
        use crate::geodesic::{GeodesicIntegrator, Metric};

        #[derive(Debug)]
        struct FlatSpacetime;

        impl Metric for FlatSpacetime {
            fn metric_tensor(&self, _: &SpacetimeVector) -> [[f64; 4]; 4] {
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, -1.0, 0.0, 0.0],
                    [0.0, 0.0, -1.0, 0.0],
                    [0.0, 0.0, 0.0, -1.0],
                ]
            }

            fn christoffel(&self, _: &SpacetimeVector) -> [[[f64; 4]; 4]; 4] {
                [[[0.0; 4]; 4]; 4] // All zeros for flat spacetime
            }

            fn name(&self) -> &str {
                "Flat Spacetime"
            }
        }

        let flat = Box::new(FlatSpacetime);
        let mut integrator = GeodesicIntegrator::with_metric(flat);

        let position = Vector3::new(1e6, 0.0, 0.0);
        let velocity = Vector3::new(1e5, 0.0, 0.0); // 100 km/s
        let mut particle = RelativisticParticle::new(
            position,
            velocity,
            0.0,
            crate::constants::atomic_masses::HYDROGEN,
            E_CHARGE,
        )
        .expect("Particle creation should succeed");

        let initial_energy = particle.total_energy();

        // Propagate in flat spacetime
        let _trajectory = propagate_relativistic(
            &mut particle,
            &mut integrator,
            3600.0, // 1 hour
            60.0,   // 1 minute steps
        )
        .expect("Flat spacetime propagation should succeed");

        let final_energy = particle.total_energy();

        // Energy should be conserved in flat spacetime
        assert_relative_eq!(final_energy, initial_energy, epsilon = 1e-12);
    }

    #[test]
    fn test_schwarzschild_coordinate_singularity() {
        // Test behavior near Schwarzschild radius
        let sun = SchwarzschildMetric::sun();

        // Position just outside Schwarzschild radius
        let rs = sun.schwarzschild_radius;
        let position = SpacetimeVector::new(0.0, rs * 1.1, 0.0, 0.0);

        // Should not be singular
        assert!(!sun.has_singularity(&position));

        // Position at Schwarzschild radius should be singular
        let horizon = SpacetimeVector::new(0.0, rs, 0.0, 0.0);
        assert!(sun.has_singularity(&horizon));

        // Test metric behavior
        let g = sun.metric_tensor(&position);
        assert!(g[0][0].is_finite()); // Should not be infinite
        assert!(g[0][0] < 0.0); // Should be negative (timelike)
    }
}

#[cfg(doctest)]
mod doctests {
    //! Additional documentation tests to ensure examples compile and run correctly

    /// Test that the main example in the crate documentation works
    ///
    /// ```rust
    /// use amari_relativistic::prelude::*;
    ///
    /// // This should compile without errors
    /// let sun = SchwarzschildMetric::sun();
    /// assert!(sun.schwarzschild_radius > 1000.0); // > 1 km
    ///
    /// let position = Vector3::new(1e6, 0.0, 0.0);
    /// let velocity = Vector3::new(1e5, 0.0, 0.0);
    /// let particle = RelativisticParticle::new(
    ///     position, velocity, 0.0,
    ///     crate::constants::atomic_masses::HYDROGEN,
    ///     E_CHARGE
    /// );
    /// assert!(particle.is_ok());
    /// ```
    fn _doctest_main_example() {}
}
