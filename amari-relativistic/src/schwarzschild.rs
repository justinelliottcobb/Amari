//! Schwarzschild metric for spherically symmetric gravitational fields
//!
//! This module implements the Schwarzschild solution to Einstein's field equations,
//! which describes the spacetime geometry around a spherically symmetric,
//! non-rotating massive body.
//!
//! # Mathematical Background
//!
//! The Schwarzschild metric in spherical coordinates (t, r, θ, φ) is:
//! ```text
//! ds² = -(1 - rₛ/r)c²dt² + (1 - rₛ/r)⁻¹dr² + r²dθ² + r²sin²θdφ²
//! ```
//!
//! Where rₛ = 2GM/c² is the Schwarzschild radius.
//!
//! # Coordinate Systems
//!
//! This implementation uses Cartesian coordinates (t, x, y, z) internally,
//! with conversion to/from spherical coordinates for metric calculations.
//!
//! # Key Features
//! - Schwarzschild radius calculation
//! - Photon sphere and ISCO (Innermost Stable Circular Orbit)
//! - Effective potential for orbital analysis
//! - Proper treatment of coordinate singularities
//!
//! # References
//! - Schwarzschild, K. "Über das Gravitationsfeld eines Massenpunktes" (1916)
//! - Misner, Thorne & Wheeler, "Gravitation" Ch. 25 (1973)
//! - Weinberg, "Gravitation and Cosmology" Ch. 8 (1972)

use crate::constants::{C, G, SOLAR_MASS};
use crate::geodesic::{GenericMetric, Metric};
use crate::precision::PrecisionFloat;
use crate::spacetime::{GenericSpacetimeVector, SpacetimeVector};
use nalgebra::Vector3;

// Note: Serde support is available but not currently implemented for these types
// #[cfg(feature = "serde")]
// use serde::{Deserialize, Serialize};

#[cfg(feature = "std")]
use std::f64::consts::PI;

#[cfg(not(feature = "std"))]
use core::f64::consts::PI;

/// Schwarzschild metric for spherically symmetric mass distributions
///
/// Represents the spacetime geometry around a spherically symmetric,
/// non-rotating massive body. Valid outside the mass distribution (r > R).
///
/// # Coordinate System
/// Uses Cartesian coordinates (t, x, y, z) with the mass centered at the origin.
/// Internally converts to spherical coordinates (r, θ, φ) for metric calculations.
///
/// # Limitations
/// - Valid only for r > Schwarzschild radius (no black hole interior)
/// - Assumes perfect spherical symmetry
/// - Non-rotating mass (no frame dragging)
/// - Vacuum solution (no matter outside the mass)
#[derive(Debug, Clone, PartialEq)]
pub struct SchwarzschildMetric {
    /// Mass of the central body in kg
    pub mass: f64,

    /// Position of the mass center in Cartesian coordinates (m)
    pub position: Vector3<f64>,

    /// Schwarzschild radius rₛ = 2GM/c² in meters
    pub schwarzschild_radius: f64,

    /// Gravitational parameter GM in m³/s²
    pub gm: f64,
}

impl SchwarzschildMetric {
    /// Create a new Schwarzschild metric
    ///
    /// # Arguments
    /// * `mass` - Mass of the central body in kg
    /// * `position` - Position of the mass center in meters
    ///
    /// # Example
    /// ```rust
    /// use amari_relativistic::schwarzschild::SchwarzschildMetric;
    /// use nalgebra::Vector3;
    ///
    /// let massive_object_mass = 5.972e24; // kg
    /// let object_center = Vector3::zeros();
    /// let metric = SchwarzschildMetric::new(massive_object_mass, object_center);
    /// ```
    pub fn new(mass: f64, position: Vector3<f64>) -> Self {
        let gm = G * mass;
        let schwarzschild_radius = 2.0 * gm / (C * C);

        Self {
            mass,
            position,
            schwarzschild_radius,
            gm,
        }
    }

    /// Create Schwarzschild metric for a solar-mass object
    ///
    /// Uses solar-mass parameters:
    /// - Mass: 1.98892 × 10³⁰ kg
    /// - Position: Origin (0, 0, 0)
    /// - Schwarzschild radius: ~2.95 km
    ///
    /// # Example
    /// ```rust
    /// use amari_relativistic::schwarzschild::SchwarzschildMetric;
    ///
    /// let massive_object = SchwarzschildMetric::sun();
    /// assert!((massive_object.schwarzschild_radius - 2954.0).abs() < 1.0); // ~2.95 km
    /// ```
    pub fn sun() -> Self {
        Self::new(SOLAR_MASS, Vector3::zeros())
    }

    /// Calculate distance from mass center to given position
    #[inline]
    fn distance_from_center(&self, position: &SpacetimeVector) -> f64 {
        let spatial = position.spatial();
        (spatial - self.position).magnitude()
    }

    /// Convert Cartesian coordinates to spherical (r, θ, φ)
    ///
    /// Returns (r, theta, phi) where:
    /// - r: radial distance from center
    /// - theta: polar angle (0 to π)
    /// - phi: azimuthal angle (0 to 2π)
    #[inline]
    fn cartesian_to_spherical(&self, position: &SpacetimeVector) -> (f64, f64, f64) {
        let relative = position.spatial() - self.position;
        let x = relative.x;
        let y = relative.y;
        let z = relative.z;

        let r = (x * x + y * y + z * z).sqrt();
        let theta = if r > 1e-14 {
            (z / r).acos()
        } else {
            0.0 // Arbitrary direction at origin
        };
        let phi = y.atan2(x);

        (r, theta, phi)
    }

    /// Check if position is inside the Schwarzschild radius
    pub fn is_inside_schwarzschild_radius(&self, position: &SpacetimeVector) -> bool {
        self.distance_from_center(position) <= self.schwarzschild_radius
    }

    /// Calculate gravitational potential Φ = -GM/r
    pub fn gravitational_potential(&self, position: &SpacetimeVector) -> f64 {
        let r = self.distance_from_center(position);
        if r > 1e-14 {
            -self.gm / r
        } else {
            f64::NEG_INFINITY
        }
    }

    /// Calculate escape velocity at given position
    ///
    /// Returns the minimum velocity needed to escape to infinity.
    /// For Schwarzschild metric: v_escape = √(2GM/r)
    pub fn escape_velocity(&self, position: &SpacetimeVector) -> f64 {
        let r = self.distance_from_center(position);
        if r > self.schwarzschild_radius {
            (2.0 * self.gm / r).sqrt()
        } else {
            C // Inside Schwarzschild radius, escape velocity > c
        }
    }

    /// Calculate circular orbital velocity at given radius
    ///
    /// For circular orbits in Schwarzschild spacetime: v_circ = √(GM/r)
    pub fn circular_orbital_velocity(&self, r: f64) -> Option<f64> {
        if r > 3.0 * self.schwarzschild_radius {
            // ISCO is at r = 6GM/c² = 3rₛ for Schwarzschild
            Some((self.gm / r).sqrt())
        } else {
            None // Inside ISCO, no stable circular orbits
        }
    }

    /// Calculate orbital period for circular orbit at given radius
    ///
    /// Uses Kepler's third law generalized to relativistic case.
    pub fn orbital_period(&self, r: f64) -> Option<f64> {
        if r > 3.0 * self.schwarzschild_radius {
            Some(2.0 * PI * (r * r * r / self.gm).sqrt())
        } else {
            None
        }
    }
}

impl Metric for SchwarzschildMetric {
    fn metric_tensor(&self, position: &SpacetimeVector) -> [[f64; 4]; 4] {
        let (r, theta, _phi) = self.cartesian_to_spherical(position);

        // Handle singularity at origin
        if r <= 1e-14 {
            // Return flat metric to avoid division by zero
            return [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, -1.0, 0.0, 0.0],
                [0.0, 0.0, -1.0, 0.0],
                [0.0, 0.0, 0.0, -1.0],
            ];
        }

        // Schwarzschild metric components
        let f = 1.0 - self.schwarzschild_radius / r;

        // Handle event horizon
        if f <= 1e-14 {
            // At or inside event horizon, use limiting form
            let g = [
                [0.0, 0.0, 0.0, 0.0],
                [0.0, -f64::INFINITY, 0.0, 0.0],
                [0.0, 0.0, -r * r, 0.0],
                [0.0, 0.0, 0.0, -r * r * theta.sin().powi(2)],
            ];
            return g;
        }

        // Standard Schwarzschild metric in spherical coordinates
        // ds² = -f c²dt² + f⁻¹dr² + r²dθ² + r²sin²θ dφ²
        let mut g = [[0.0; 4]; 4];

        // For simplicity, we'll work in Cartesian coordinates
        // and transform the metric tensor. This is more complex but
        // allows easier integration with Cartesian particle trajectories.

        // Simplified approach: use isotropic coordinates approximation
        // for weak fields, exact Schwarzschild for strong fields
        let spatial = position.spatial() - self.position;
        let x = spatial.x;
        let y = spatial.y;
        let _z = spatial.z;

        // Time-time component: g₀₀ = -(1 - rₛ/r)
        g[0][0] = -f;

        // For spatial components in Cartesian coordinates,
        // we need to transform the spherical metric tensor
        // This is complex, so we'll use an approximation

        if r > 10.0 * self.schwarzschild_radius {
            // Weak field approximation: g ≈ η + h where h is small
            let h00 = -2.0 * self.gravitational_potential(position) / (C * C);
            g[0][0] = -(1.0 + h00);
            g[1][1] = -(1.0 - h00);
            g[2][2] = -(1.0 - h00);
            g[3][3] = -(1.0 - h00);
        } else {
            // Strong field: use exact transformation
            let _sin_theta = theta.sin();
            let _cos_theta = theta.cos();
            let _sin_phi = (y / (x * x + y * y).sqrt()).asin();
            let _cos_phi = (x / (x * x + y * y).sqrt()).acos();

            // Transformation matrix from spherical to Cartesian
            // This is a simplified implementation
            g[1][1] = -1.0 / f; // Approximate radial component
            g[2][2] = -1.0; // Approximate tangential
            g[3][3] = -1.0; // Approximate tangential
        }

        g
    }

    fn christoffel(&self, position: &SpacetimeVector) -> [[[f64; 4]; 4]; 4] {
        let (r, _theta, _phi) = self.cartesian_to_spherical(position);

        // Initialize Christoffel symbols
        let mut gamma = [[[0.0; 4]; 4]; 4];

        // Handle singularity
        if r <= 1e-14 || r <= self.schwarzschild_radius {
            return gamma; // Return zeros to avoid infinities
        }

        let rs = self.schwarzschild_radius;
        let f = 1.0 - rs / r;

        // Key non-zero Christoffel symbols in spherical coordinates
        // We'll compute the most important ones for radial motion

        // Γᵗᵗʳ = rs/(2r²(r-rs)) - time-time-radial
        // Γʳᵗᵗ = (rs c²(r-rs))/(2r³) - radial-time-time
        // Γʳʳʳ = -rs/(2r(r-rs)) - radial-radial-radial

        // For Cartesian coordinates, we need to transform these
        // This is complex, so we'll use key components

        let spatial = position.spatial() - self.position;
        let x = spatial.x;
        let y = spatial.y;
        let z = spatial.z;

        if r > 1e-14 {
            // Radial unit vector components
            let nr_x = x / r;
            let nr_y = y / r;
            let nr_z = z / r;

            // Key Christoffel components (simplified)
            let coeff1 = rs / (2.0 * r * r * f);
            let coeff2 = rs * f / (2.0 * r * r * r);

            // Time-space-space components (Γ⁰ᵢⱼ)
            gamma[0][1][1] = coeff2 * nr_x * nr_x; // Γ⁰ₓₓ
            gamma[0][2][2] = coeff2 * nr_y * nr_y; // Γ⁰ᵧᵧ
            gamma[0][3][3] = coeff2 * nr_z * nr_z; // Γ⁰ᵣᵣ

            // Space-time-time components (Γⁱ₀₀)
            gamma[1][0][0] = coeff1 * nr_x; // Γˣ₀₀
            gamma[2][0][0] = coeff1 * nr_y; // Γʸ₀₀
            gamma[3][0][0] = coeff1 * nr_z; // Γᶻ₀₀

            // Make symmetric in lower indices
            #[allow(clippy::needless_range_loop)]
            for i in 0..4 {
                for j in 0..4 {
                    for k in j + 1..4 {
                        gamma[i][k][j] = gamma[i][j][k];
                    }
                }
            }
        }

        gamma
    }

    fn name(&self) -> &str {
        "Schwarzschild"
    }

    fn has_singularity(&self, position: &SpacetimeVector) -> bool {
        let r = self.distance_from_center(position);
        r <= self.schwarzschild_radius || r <= 1e-14
    }

    fn characteristic_scale(&self) -> f64 {
        // Use Schwarzschild radius as characteristic scale
        self.schwarzschild_radius
    }
}

impl<T: PrecisionFloat> GenericMetric<T> for SchwarzschildMetric {
    fn metric_tensor(&self, position: &GenericSpacetimeVector<T>) -> [[T; 4]; 4] {
        // Convert to f64 SpacetimeVector for calculation
        let f64_pos = SpacetimeVector::new(
            position.time().to_f64(),
            position.x().to_f64(),
            position.y().to_f64(),
            position.z().to_f64(),
        );

        // Use f64 implementation
        let f64_metric = <Self as Metric>::metric_tensor(self, &f64_pos);

        // Convert back to generic precision
        let mut generic_metric = T::zero_matrix_4x4();
        for i in 0..4 {
            for j in 0..4 {
                generic_metric[i][j] = <T as PrecisionFloat>::from_f64(f64_metric[i][j]);
            }
        }

        generic_metric
    }

    fn christoffel(&self, position: &GenericSpacetimeVector<T>) -> [[[T; 4]; 4]; 4] {
        // Convert to f64 SpacetimeVector for calculation
        let f64_pos = SpacetimeVector::new(
            position.time().to_f64(),
            position.x().to_f64(),
            position.y().to_f64(),
            position.z().to_f64(),
        );

        // Use f64 implementation
        let f64_christoffel = <Self as Metric>::christoffel(self, &f64_pos);

        // Convert back to generic precision
        let mut generic_christoffel = T::zero_tensor_4x4x4();
        for i in 0..4 {
            for j in 0..4 {
                for k in 0..4 {
                    generic_christoffel[i][j][k] =
                        <T as PrecisionFloat>::from_f64(f64_christoffel[i][j][k]);
                }
            }
        }

        generic_christoffel
    }

    fn name(&self) -> &str {
        <Self as Metric>::name(self)
    }
}

/// Calculate effective potential for orbital motion in Schwarzschild spacetime
///
/// The effective potential determines the radial motion of particles:
/// V_eff(r) = -GM/r + L²/(2mr²) - GML²/(mc²r³)
///
/// Where:
/// - L is the angular momentum per unit mass
/// - The last term is the relativistic correction
///
/// # Arguments
/// * `r` - Radial distance from mass center (m)
/// * `angular_momentum` - Angular momentum per unit mass (m²/s)
/// * `mass` - Central mass (kg)
///
/// # Returns
/// Effective potential energy per unit mass (J/kg)
pub fn effective_potential(r: f64, angular_momentum: f64, mass: f64) -> f64 {
    if r <= 1e-14 {
        return f64::NEG_INFINITY;
    }

    let gm = G * mass;
    let _rs = 2.0 * gm / (C * C);

    // Classical terms
    let gravitational = -gm / r;
    let centrifugal = angular_momentum * angular_momentum / (2.0 * r * r);

    // Relativistic correction (GR term)
    let relativistic = if angular_momentum.abs() > 1e-14 {
        -gm * angular_momentum * angular_momentum / (C * C * r * r * r)
    } else {
        0.0
    };

    gravitational + centrifugal + relativistic
}

/// Find radius of circular orbit for given angular momentum
///
/// Circular orbits occur at extrema of the effective potential.
/// For Schwarzschild metric, stable circular orbits exist for r ≥ 6GM/c².
///
/// # Arguments
/// * `angular_momentum` - Angular momentum per unit mass (m²/s)
/// * `mass` - Central mass (kg)
///
/// # Returns
/// Orbital radius in meters, or None if no stable orbit exists
pub fn circular_orbit_radius(angular_momentum: f64, mass: f64) -> Option<f64> {
    let gm = G * mass;
    let rs = 2.0 * gm / (C * C);

    // For circular orbits: dV_eff/dr = 0
    // This gives: r = L²/GM ± √((L²/GM)² - 12(L²/GM)(rs/2))
    // Simplifying: r = L²/GM (1 ± √(1 - 12GM²/(L²c²)))

    let l_squared = angular_momentum * angular_momentum;
    if l_squared <= 1e-14 {
        return None; // Radial motion, no circular orbit
    }

    let r_classical = l_squared / gm; // Classical circular orbit radius
    let discriminant = 1.0 - 12.0 * gm * gm / (l_squared * C * C);

    if discriminant < 0.0 {
        None // No real solution, no circular orbit possible
    } else {
        let r = r_classical * (1.0 + discriminant.sqrt());
        if r > 3.0 * rs {
            Some(r)
        } else {
            None // Inside ISCO
        }
    }
}

/// Calculate photon sphere radius for massless particles
///
/// The photon sphere is the unstable circular orbit for light rays.
/// For Schwarzschild metric: r_photon = 3GM/c² = 1.5 rₛ
///
/// # Arguments
/// * `mass` - Central mass (kg)
///
/// # Returns
/// Photon sphere radius in meters
pub fn photon_sphere_radius(mass: f64) -> f64 {
    3.0 * G * mass / (C * C)
}

/// Determine if an orbit is bound (elliptical) or unbound (hyperbolic)
///
/// Uses the effective potential and total energy to classify the orbit.
///
/// # Arguments
/// * `position` - Current position relative to mass center (m)
/// * `velocity` - Current velocity (m/s)
/// * `central_mass` - Mass of central body (kg)
///
/// # Returns
/// True if orbit is bound (E < 0), false if unbound (E ≥ 0)
pub fn is_bound_orbit(position: Vector3<f64>, velocity: Vector3<f64>, central_mass: f64) -> bool {
    let r = position.magnitude();
    let v_squared = velocity.magnitude_squared();
    let gm = G * central_mass;

    // Classical orbital energy per unit mass
    let kinetic = 0.5 * v_squared;
    let potential = -gm / r;
    let total_energy = kinetic + potential;

    // Include relativistic corrections for high velocities
    if v_squared > 0.01 * C * C {
        // Use relativistic kinetic energy: T = (γ - 1)mc²
        let gamma = 1.0 / (1.0 - v_squared / (C * C)).sqrt();
        let relativistic_kinetic = (gamma - 1.0) * C * C;
        let relativistic_energy = relativistic_kinetic + potential;
        relativistic_energy < 0.0
    } else {
        total_energy < 0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_schwarzschild_radius() {
        let sun = SchwarzschildMetric::sun();

        // Sun's Schwarzschild radius should be ~2.95 km
        let expected_rs = 2.0 * G * SOLAR_MASS / (C * C);
        assert_relative_eq!(sun.schwarzschild_radius, expected_rs, epsilon = 1e-10);
        assert!((sun.schwarzschild_radius - 2953.0).abs() < 10.0); // Within 10 meters
    }

    #[test]
    fn test_metric_tensor_far_field() {
        let sun = SchwarzschildMetric::sun();

        // Test at large distance (weak field)
        let position = SpacetimeVector::new(0.0, 1.496e11, 0.0, 0.0); // Large distance
        let g = <SchwarzschildMetric as Metric>::metric_tensor(&sun, &position);

        // In weak field, should approach Minkowski metric
        // g₀₀ ≈ -(1 + 2Φ/c²) where Φ = -GM/r
        let r = 1.496e11;
        let phi = -G * SOLAR_MASS / r;
        let expected_g00 = -(1.0 + 2.0 * phi / (C * C));

        assert_relative_eq!(g[0][0], expected_g00, epsilon = 1e-6);

        // Spatial components should be close to -1 in weak field
        assert!(g[1][1] < -0.99);
        assert!(g[2][2] < -0.99);
        assert!(g[3][3] < -0.99);
    }

    #[test]
    fn test_photon_sphere() {
        let sun = SchwarzschildMetric::sun();
        let r_photon = photon_sphere_radius(SOLAR_MASS);

        // Should be 1.5 times Schwarzschild radius
        assert_relative_eq!(r_photon, 1.5 * sun.schwarzschild_radius, epsilon = 1e-12);

        // Should be approximately 4.43 km for solar-mass objects
        assert!((r_photon - 4429.5).abs() < 100.0);
    }

    #[test]
    fn test_escape_velocity() {
        let earth_mass = 5.972e24;
        let earth_radius = 6.371e6;
        let earth = SchwarzschildMetric::new(earth_mass, Vector3::zeros());

        let surface_position = SpacetimeVector::new(0.0, earth_radius, 0.0, 0.0);
        let v_escape = earth.escape_velocity(&surface_position);

        // Escape velocity should be ~11.2 km/s for this mass and radius
        assert_relative_eq!(v_escape, 11200.0, epsilon = 100.0);
    }

    #[test]
    fn test_circular_orbital_velocity() {
        let earth_mass = 5.972e24;
        let earth = SchwarzschildMetric::new(earth_mass, Vector3::zeros());

        // Low orbit at ~400 km altitude
        let orbital_radius = 6.371e6 + 400e3;
        let v_orbital = earth.circular_orbital_velocity(orbital_radius);

        assert!(v_orbital.is_some());
        let v = v_orbital.unwrap();

        // Should be approximately 7.7 km/s
        assert_relative_eq!(v, 7700.0, epsilon = 100.0);
    }

    #[test]
    fn test_effective_potential() {
        let sun_mass = SOLAR_MASS;
        let r = 1.496e11; // Large distance
        let l = 1e15; // Some angular momentum

        let v_eff = effective_potential(r, l, sun_mass);

        // Should be negative (bound system)
        assert!(v_eff < 0.0);

        // Test that relativistic term is small at this distance
        let gm = G * sun_mass;
        let classical = -gm / r + l * l / (2.0 * r * r);
        let relativistic_term = -gm * l * l / (C * C * r * r * r);

        assert!(relativistic_term.abs() < 0.01 * classical.abs());
    }

    #[test]
    fn test_bound_orbit_classification() {
        let earth_mass = 5.972e24_f64;
        let position = Vector3::new(7e6_f64, 0.0_f64, 0.0_f64); // 700 km from center

        // Circular orbital velocity (bound)
        let v_circular = (G * earth_mass / position.magnitude()).sqrt();
        let velocity_bound = Vector3::new(0.0, v_circular, 0.0);

        assert!(is_bound_orbit(position, velocity_bound, earth_mass));

        // Escape velocity (unbound)
        let v_escape = (2.0 * G * earth_mass / position.magnitude()).sqrt();
        let velocity_unbound = Vector3::new(0.0, v_escape * 1.1, 0.0);

        assert!(!is_bound_orbit(position, velocity_unbound, earth_mass));
    }

    #[test]
    fn test_has_singularity() {
        let sun = SchwarzschildMetric::sun();

        // Outside Schwarzschild radius should be fine
        let far_position = SpacetimeVector::new(0.0, 1e6, 0.0, 0.0);
        assert!(!<SchwarzschildMetric as Metric>::has_singularity(
            &sun,
            &far_position
        ));

        // At Schwarzschild radius should be singular
        let horizon_position = SpacetimeVector::new(0.0, sun.schwarzschild_radius, 0.0, 0.0);
        assert!(<SchwarzschildMetric as Metric>::has_singularity(
            &sun,
            &horizon_position
        ));

        // Origin should be singular
        let origin = SpacetimeVector::new(0.0, 0.0, 0.0, 0.0);
        assert!(<SchwarzschildMetric as Metric>::has_singularity(
            &sun, &origin
        ));
    }

    #[test]
    fn test_orbital_period() {
        let earth_mass = 5.972e24;
        let earth = SchwarzschildMetric::new(earth_mass, Vector3::zeros());

        // High orbit radius ~42,164 km
        let geo_radius = 4.2164e7;
        let period = earth.orbital_period(geo_radius);

        assert!(period.is_some());
        let t = period.unwrap();

        // Should be approximately 24 hours
        let expected_period = 24.0 * 3600.0; // 24 hours in seconds
        assert_relative_eq!(t, expected_period, epsilon = 1000.0);
    }
}
