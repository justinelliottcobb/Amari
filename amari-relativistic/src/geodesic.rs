//! Geodesic integration for general relativistic trajectories
//!
//! This module provides numerical integration of the geodesic equation in curved spacetime.
//! The geodesic equation describes the motion of free particles (both massive and massless)
//! in gravitational fields according to general relativity.
//!
//! # Mathematical Background
//!
//! The geodesic equation in curved spacetime is:
//! ```text
//! d²xᵘ/dτ² + Γᵘ_αβ (dxᵅ/dτ)(dxᵝ/dτ) = 0
//! ```
//!
//! Where:
//! - xᵘ are the spacetime coordinates
//! - τ is the proper time (or affine parameter for null geodesics)
//! - Γᵘ_αβ are the Christoffel symbols derived from the metric
//!
//! # Numerical Methods
//!
//! This implementation uses the velocity Verlet integrator, which is:
//! - Symplectic (preserves phase space volume)
//! - Second-order accurate in time
//! - Stable for long integrations
//! - Conserves energy for conservative systems
//!
//! # References
//! - Misner, Thorne & Wheeler, "Gravitation" (1973)
//! - Wald, "General Relativity" (1984)
//! - Press et al., "Numerical Recipes" (2007)

use crate::spacetime::{FourVelocity, SpacetimeVector};
use nalgebra::Vector3;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

#[cfg(feature = "std")]
use std::fmt;

#[cfg(not(feature = "std"))]
use core::fmt;

#[cfg(not(feature = "std"))]
extern crate alloc;

#[cfg(not(feature = "std"))]
use alloc::{boxed::Box, format, string::String, string::ToString, vec::Vec};

#[cfg(feature = "std")]
use thiserror::Error;

/// Error types for geodesic integration
#[derive(Debug)]
#[cfg_attr(feature = "std", derive(Error))]
pub enum GeodesicError {
    #[cfg_attr(feature = "std", error("Integration failed: {reason}"))]
    /// Integration step failed due to numerical instability or invalid parameters
    IntegrationFailure {
        /// Description of what caused the integration failure
        reason: String,
    },

    #[cfg_attr(feature = "std", error("Invalid metric configuration: {details}"))]
    /// Invalid metric tensor or singularity encountered
    InvalidMetric {
        /// Details about the metric problem
        details: String,
    },

    #[cfg_attr(
        feature = "std",
        error("Numerical instability detected at τ = {proper_time:.6e}")
    )]
    /// Numerical instability detected during integration
    NumericalInstability {
        /// Proper time at which instability was detected
        proper_time: f64,
    },

    #[cfg_attr(feature = "std", error("Four-velocity normalization failed: |u|² = {norm_squared:.6e}, expected c² = {expected:.6e}"))]
    /// Four-velocity normalization constraint violated
    NormalizationFailure {
        /// Actual norm squared of four-velocity
        norm_squared: f64,
        /// Expected norm squared (c²)
        expected: f64,
    },

    #[cfg_attr(
        feature = "std",
        error("Time step too large: Δτ = {step_size:.6e} > maximum allowed {max_allowed:.6e}")
    )]
    /// Time step exceeds maximum allowed for stability
    StepSizeTooLarge {
        /// Attempted step size
        step_size: f64,
        /// Maximum allowed step size
        max_allowed: f64,
    },
}

/// Result type for geodesic operations
pub type GeodesicResult<T> = Result<T, GeodesicError>;

#[cfg(not(feature = "std"))]
impl fmt::Display for GeodesicError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GeodesicError::IntegrationFailure { reason } => {
                write!(f, "Integration failed: {}", reason)
            }
            GeodesicError::InvalidMetric { details } => {
                write!(f, "Invalid metric configuration: {}", details)
            }
            GeodesicError::NumericalInstability { proper_time } => {
                write!(
                    f,
                    "Numerical instability detected at τ = {:.6e}",
                    proper_time
                )
            }
            GeodesicError::NormalizationFailure {
                norm_squared,
                expected,
            } => {
                write!(
                    f,
                    "Four-velocity normalization failed: |u|² = {:.6e}, expected c² = {:.6e}",
                    norm_squared, expected
                )
            }
            GeodesicError::StepSizeTooLarge {
                step_size,
                max_allowed,
            } => {
                write!(
                    f,
                    "Time step too large: Δτ = {:.6e} > maximum allowed {:.6e}",
                    step_size, max_allowed
                )
            }
        }
    }
}

/// Trait for spacetime metrics
///
/// This trait defines the interface for spacetime metrics used in geodesic integration.
/// Implementations must provide the metric tensor and Christoffel symbols at any
/// spacetime point.
///
/// # Mathematical Requirements
/// - Metric signature: (-,+,+,+) or (+,-,-,-) (Lorentzian)
/// - Symmetric: gμν = gνμ
/// - Non-degenerate: det(g) ≠ 0
/// - Christoffel symbols: Γᵘ_αβ = ½gᵘᵛ(∂gᵥα/∂xᵝ + ∂gᵥβ/∂xᵅ - ∂gαβ/∂xᵛ)
pub trait Metric: Send + Sync + fmt::Debug {
    /// Compute the metric tensor at a given spacetime point
    ///
    /// Returns the 4×4 metric tensor gμν with signature (-,+,+,+) or (+,-,-,-).
    /// The tensor should be symmetric: gμν = gνμ.
    ///
    /// # Arguments
    /// * `position` - Spacetime coordinates (ct, x, y, z)
    ///
    /// # Returns
    /// 4×4 matrix representing the metric tensor gμν
    fn metric_tensor(&self, position: &SpacetimeVector) -> [[f64; 4]; 4];

    /// Compute Christoffel symbols at a given spacetime point
    ///
    /// Returns the Christoffel symbols Γᵘ_αβ of the second kind.
    /// These are computed from the metric using:
    /// Γᵘ_αβ = ½gᵘᵛ(∂gᵥα/∂xᵝ + ∂gᵥβ/∂xᵅ - ∂gαβ/∂xᵛ)
    ///
    /// # Arguments
    /// * `position` - Spacetime coordinates (ct, x, y, z)
    ///
    /// # Returns
    /// 4×4×4 array representing Γᵘ_αβ where the first index is μ,
    /// second is α, third is β
    fn christoffel(&self, position: &SpacetimeVector) -> [[[f64; 4]; 4]; 4];

    /// Get a descriptive name for this metric
    fn name(&self) -> &str;

    /// Check if this metric has any singularities at the given point
    ///
    /// Default implementation checks if the metric determinant is near zero.
    /// Specific metrics can override this for more sophisticated checks.
    fn has_singularity(&self, position: &SpacetimeVector) -> bool {
        let g = self.metric_tensor(position);
        let det = metric_determinant(&g);
        det.abs() < 1e-14
    }

    /// Get characteristic length scale for this metric
    ///
    /// This is used for adaptive step size control. Default implementation
    /// returns a generic scale, but specific metrics should override.
    fn characteristic_scale(&self) -> f64 {
        1e6 // 1000 km default scale
    }
}

/// Calculate determinant of 4×4 metric tensor
#[inline]
fn metric_determinant(g: &[[f64; 4]; 4]) -> f64 {
    // Compute determinant using cofactor expansion
    // det(g) = g₀₀ * M₀₀ - g₀₁ * M₀₁ + g₀₂ * M₀₂ - g₀₃ * M₀₃
    // where Mᵢⱼ are 3×3 minors

    let m00 = g[1][1] * (g[2][2] * g[3][3] - g[2][3] * g[3][2])
        - g[1][2] * (g[2][1] * g[3][3] - g[2][3] * g[3][1])
        + g[1][3] * (g[2][1] * g[3][2] - g[2][2] * g[3][1]);

    let m01 = g[1][0] * (g[2][2] * g[3][3] - g[2][3] * g[3][2])
        - g[1][2] * (g[2][0] * g[3][3] - g[2][3] * g[3][0])
        + g[1][3] * (g[2][0] * g[3][2] - g[2][2] * g[3][0]);

    let m02 = g[1][0] * (g[2][1] * g[3][3] - g[2][3] * g[3][1])
        - g[1][1] * (g[2][0] * g[3][3] - g[2][3] * g[3][0])
        + g[1][3] * (g[2][0] * g[3][1] - g[2][1] * g[3][0]);

    let m03 = g[1][0] * (g[2][1] * g[3][2] - g[2][2] * g[3][1])
        - g[1][1] * (g[2][0] * g[3][2] - g[2][2] * g[3][0])
        + g[1][2] * (g[2][0] * g[3][1] - g[2][1] * g[3][0]);

    g[0][0] * m00 - g[0][1] * m01 + g[0][2] * m02 - g[0][3] * m03
}

/// Configuration for geodesic integration
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct IntegrationConfig {
    /// Maximum allowed step size in proper time (seconds)
    pub max_step_size: f64,

    /// Minimum allowed step size in proper time (seconds)
    pub min_step_size: f64,

    /// Tolerance for four-velocity normalization
    pub normalization_tolerance: f64,

    /// Frequency of four-velocity renormalization (every N steps)
    pub renormalization_frequency: usize,

    /// Maximum number of integration steps
    pub max_steps: usize,

    /// Relative error tolerance for adaptive stepping
    pub error_tolerance: f64,

    /// Safety factor for step size adjustment
    pub safety_factor: f64,
}

impl Default for IntegrationConfig {
    fn default() -> Self {
        Self {
            max_step_size: 100.0,           // 100 seconds max step
            min_step_size: 1e-6,            // 1 microsecond min step
            normalization_tolerance: 1e-10, // Very tight normalization
            renormalization_frequency: 100, // Renormalize every 100 steps
            max_steps: 1_000_000,           // 1M steps maximum
            error_tolerance: 1e-8,          // 1e-8 relative error
            safety_factor: 0.9,             // Conservative step adjustment
        }
    }
}

/// Geodesic integrator for curved spacetime
///
/// This struct provides numerical integration of geodesic equations using
/// the velocity Verlet method. It maintains the metric, integration state,
/// and configuration parameters.
///
/// # Example
/// ```rust
/// use amari_relativistic::geodesic::{GeodesicIntegrator, IntegrationConfig};
/// use amari_relativistic::schwarzschild::SchwarzschildMetric;
/// use std::sync::Arc;
///
/// // Create metric (example with Schwarzschild)
/// let metric = Arc::new(SchwarzschildMetric::sun());
/// let config = IntegrationConfig::default();
/// let mut integrator = GeodesicIntegrator::new(metric, config);
/// ```
pub struct GeodesicIntegrator {
    /// Spacetime metric implementation
    metric: Box<dyn Metric>,

    /// Current proper time
    pub proper_time: f64,

    /// Integration configuration
    config: IntegrationConfig,

    /// Number of steps taken
    steps_taken: usize,

    /// Last step size used
    last_step_size: f64,
}

impl GeodesicIntegrator {
    /// Create a new geodesic integrator
    ///
    /// # Arguments
    /// * `metric` - Implementation of the spacetime metric
    /// * `config` - Integration configuration parameters
    pub fn new(metric: Box<dyn Metric>, config: IntegrationConfig) -> Self {
        Self {
            metric,
            proper_time: 0.0,
            config,
            steps_taken: 0,
            last_step_size: 0.0,
        }
    }

    /// Create integrator with default configuration
    pub fn with_metric(metric: Box<dyn Metric>) -> Self {
        Self::new(metric, IntegrationConfig::default())
    }

    /// Perform a single integration step using velocity Verlet method
    ///
    /// The velocity Verlet algorithm for the geodesic equation:
    /// 1. x_{n+1} = x_n + u_n * Δτ + ½ * a_n * (Δτ)²
    /// 2. a_{n+1} = acceleration(x_{n+1}, u_{n+1/2})
    /// 3. u_{n+1} = u_n + ½ * (a_n + a_{n+1}) * Δτ
    ///
    /// where acceleration = -Γᵘ_αβ u^α u^β
    ///
    /// # Arguments
    /// * `position` - Current spacetime position (modified in place)
    /// * `four_velocity` - Current four-velocity (modified in place)
    /// * `dtau` - Proper time step size
    ///
    /// # Returns
    /// `Ok(actual_step_size)` on success, `Err(GeodesicError)` on failure
    #[inline]
    pub fn step(
        &mut self,
        position: &mut SpacetimeVector,
        four_velocity: &mut FourVelocity,
        dtau: f64,
    ) -> GeodesicResult<f64> {
        // Validate step size
        if dtau > self.config.max_step_size {
            return Err(GeodesicError::StepSizeTooLarge {
                step_size: dtau,
                max_allowed: self.config.max_step_size,
            });
        }

        if dtau < self.config.min_step_size {
            return Err(GeodesicError::IntegrationFailure {
                reason: format!(
                    "Step size {:.2e} below minimum {:.2e}",
                    dtau, self.config.min_step_size
                ),
            });
        }

        // Check for singularities
        if self.metric.has_singularity(position) {
            return Err(GeodesicError::NumericalInstability {
                proper_time: self.proper_time,
            });
        }

        // Extract current coordinates and velocity
        let x = position.coordinates();
        let u = four_velocity.as_spacetime_vector().coordinates();

        // Compute initial acceleration: a_n = -Γᵘ_αβ u^α u^β
        let a_n = self.compute_acceleration(position, four_velocity)?;

        // Velocity Verlet step 1: x_{n+1} = x_n + u_n * Δτ + ½ * a_n * (Δτ)²
        let dtau_sq = dtau * dtau;
        let mut x_new = [0.0; 4];
        for μ in 0..4 {
            x_new[μ] = x[μ] + u[μ] * dtau + 0.5 * a_n[μ] * dtau_sq;
        }

        // Update position
        *position = SpacetimeVector::from_coordinates(x_new);

        // Compute intermediate velocity: u_{n+1/2} = u_n + ½ * a_n * Δτ
        let mut u_half = [0.0; 4];
        for μ in 0..4 {
            u_half[μ] = u[μ] + 0.5 * a_n[μ] * dtau;
        }

        // Create temporary four-velocity for acceleration calculation
        // Note: This may not be exactly normalized, but it's used only for computing forces
        let temp_spacetime_vec = SpacetimeVector::from_coordinates(u_half);
        let temp_four_vel = {
            let normalized = temp_spacetime_vec.minkowski_norm_squared();
            if normalized > 0.0 {
                // For timelike vectors, create a proper four-velocity
                let gamma = (normalized / (crate::constants::C * crate::constants::C)).sqrt();
                let spatial_part =
                    Vector3::new(u_half[1] / gamma, u_half[2] / gamma, u_half[3] / gamma);
                FourVelocity::from_velocity(spatial_part)
            } else {
                // For spacelike or null, fall back to original velocity
                four_velocity.clone()
            }
        };

        // Compute acceleration at new position: a_{n+1}
        let a_new = self.compute_acceleration(position, &temp_four_vel)?;

        // Velocity Verlet step 3: u_{n+1} = u_n + ½ * (a_n + a_{n+1}) * Δτ
        let mut u_new = [0.0; 4];
        for μ in 0..4 {
            u_new[μ] = u[μ] + 0.5 * (a_n[μ] + a_new[μ]) * dtau;
        }

        // Reconstruct four-velocity and normalize if needed
        let new_spacetime_vec = SpacetimeVector::from_coordinates(u_new);

        // For massive particles, reconstruct proper four-velocity
        if new_spacetime_vec.is_timelike() {
            let gamma = new_spacetime_vec.time_component() / crate::constants::C;
            if gamma > 1e-10 {
                let spatial_velocity =
                    Vector3::new(u_new[1] / gamma, u_new[2] / gamma, u_new[3] / gamma);

                // Check for superluminal velocity
                if spatial_velocity.magnitude() < crate::constants::C {
                    *four_velocity = FourVelocity::from_velocity(spatial_velocity);
                } else {
                    return Err(GeodesicError::NumericalInstability {
                        proper_time: self.proper_time,
                    });
                }
            }
        }

        // Periodic renormalization to prevent drift
        if self
            .steps_taken
            .is_multiple_of(self.config.renormalization_frequency)
        {
            four_velocity.normalize();

            // Check normalization quality
            if !four_velocity.is_normalized(self.config.normalization_tolerance) {
                return Err(GeodesicError::NormalizationFailure {
                    norm_squared: four_velocity.as_spacetime_vector().minkowski_norm_squared(),
                    expected: crate::constants::C * crate::constants::C,
                });
            }
        }

        // Update state
        self.proper_time += dtau;
        self.steps_taken += 1;
        self.last_step_size = dtau;

        Ok(dtau)
    }

    /// Compute geodesic acceleration: a^μ = -Γᵘ_αβ u^α u^β
    #[inline]
    fn compute_acceleration(
        &self,
        position: &SpacetimeVector,
        four_velocity: &FourVelocity,
    ) -> GeodesicResult<[f64; 4]> {
        let christoffel = self.metric.christoffel(position);
        let u = four_velocity.as_spacetime_vector().coordinates();
        let mut acceleration = [0.0; 4];

        // a^μ = -Γᵘ_αβ u^α u^β
        for μ in 0..4 {
            let mut sum = 0.0;
            for α in 0..4 {
                for β in 0..4 {
                    sum += christoffel[μ][α][β] * u[α] * u[β];
                }
            }
            acceleration[μ] = -sum;
        }

        Ok(acceleration)
    }

    /// Propagate geodesic for a given duration
    ///
    /// Integrates the geodesic equation from the current state for the specified
    /// duration, returning the complete trajectory.
    ///
    /// # Arguments
    /// * `position` - Initial spacetime position (modified to final position)
    /// * `four_velocity` - Initial four-velocity (modified to final velocity)
    /// * `duration` - Integration duration in proper time (seconds)
    /// * `dtau` - Base step size in proper time (seconds)
    ///
    /// # Returns
    /// Vector of trajectory points: (proper_time, position, velocity)
    pub fn propagate(
        &mut self,
        position: &mut SpacetimeVector,
        four_velocity: &mut FourVelocity,
        duration: f64,
        dtau: f64,
    ) -> GeodesicResult<Vec<(f64, SpacetimeVector, FourVelocity)>> {
        let start_time = self.proper_time;
        let end_time = start_time + duration;
        let mut trajectory = Vec::new();

        // Store initial point
        trajectory.push((self.proper_time, position.clone(), four_velocity.clone()));

        let mut current_step = dtau.min(self.config.max_step_size);

        while self.proper_time < end_time && self.steps_taken < self.config.max_steps {
            // Ensure we don't overshoot the end time
            if self.proper_time + current_step > end_time {
                current_step = end_time - self.proper_time;
            }

            // Perform integration step
            match self.step(position, four_velocity, current_step) {
                Ok(actual_step) => {
                    // Store trajectory point
                    trajectory.push((self.proper_time, position.clone(), four_velocity.clone()));

                    // Update step size for next iteration
                    current_step = actual_step;
                }
                Err(e) => {
                    // Try smaller step size on error
                    current_step *= 0.5;
                    if current_step < self.config.min_step_size {
                        return Err(e);
                    }
                    // Don't increment time on failed step
                    continue;
                }
            }
        }

        if self.steps_taken >= self.config.max_steps {
            return Err(GeodesicError::IntegrationFailure {
                reason: format!("Maximum steps ({}) exceeded", self.config.max_steps),
            });
        }

        Ok(trajectory)
    }

    /// Reset integrator state
    pub fn reset(&mut self) {
        self.proper_time = 0.0;
        self.steps_taken = 0;
        self.last_step_size = 0.0;
    }

    /// Get current integration statistics
    pub fn stats(&self) -> IntegrationStats {
        IntegrationStats {
            proper_time: self.proper_time,
            steps_taken: self.steps_taken,
            last_step_size: self.last_step_size,
            metric_name: self.metric.name().to_string(),
        }
    }

    /// Get reference to the metric
    pub fn metric(&self) -> &dyn Metric {
        self.metric.as_ref()
    }

    /// Get the current configuration
    pub fn config(&self) -> &IntegrationConfig {
        &self.config
    }

    /// Update configuration
    pub fn set_config(&mut self, config: IntegrationConfig) {
        self.config = config;
    }
}

/// Integration statistics
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct IntegrationStats {
    /// Current proper time
    pub proper_time: f64,

    /// Number of steps taken
    pub steps_taken: usize,

    /// Last step size used
    pub last_step_size: f64,

    /// Name of the metric being used
    pub metric_name: String,
}

impl fmt::Display for IntegrationStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Integration Stats:\n  Metric: {}\n  Proper time: {:.6e} s\n  Steps: {}\n  Last step: {:.6e} s",
            self.metric_name, self.proper_time, self.steps_taken, self.last_step_size
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::{assert_abs_diff_eq, assert_relative_eq};
    use nalgebra::Vector3;

    /// Mock flat spacetime metric for testing
    #[derive(Debug)]
    struct FlatSpacetime;

    impl Metric for FlatSpacetime {
        fn metric_tensor(&self, _position: &SpacetimeVector) -> [[f64; 4]; 4] {
            // Minkowski metric: diag(+1, -1, -1, -1)
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, -1.0, 0.0, 0.0],
                [0.0, 0.0, -1.0, 0.0],
                [0.0, 0.0, 0.0, -1.0],
            ]
        }

        fn christoffel(&self, _position: &SpacetimeVector) -> [[[f64; 4]; 4]; 4] {
            // All Christoffel symbols are zero in flat spacetime
            [[[0.0; 4]; 4]; 4]
        }

        fn name(&self) -> &str {
            "Flat Spacetime (Minkowski)"
        }
    }

    #[test]
    fn test_flat_spacetime_geodesic() {
        // In flat spacetime, geodesics should be straight lines
        let metric = Box::new(FlatSpacetime);
        let mut integrator = GeodesicIntegrator::with_metric(metric);

        let mut position = SpacetimeVector::new(0.0, 0.0, 0.0, 0.0);
        let velocity = Vector3::new(0.5 * crate::constants::C, 0.0, 0.0);
        let mut four_velocity = FourVelocity::from_velocity(velocity);

        // Integrate for 1 second
        let trajectory = integrator
            .propagate(&mut position, &mut four_velocity, 1.0, 0.01)
            .expect("Integration should succeed in flat spacetime");

        // Final position should be approximately (1.0, 0.5c, 0, 0)
        let final_pos = &trajectory.last().unwrap().1;
        assert_relative_eq!(final_pos.time(), 1.0, epsilon = 1e-6);
        assert_relative_eq!(final_pos.x(), 0.5 * crate::constants::C, epsilon = 1e-3);
        assert_abs_diff_eq!(final_pos.y(), 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(final_pos.z(), 0.0, epsilon = 1e-10);

        // Four-velocity should remain constant
        let final_vel = &trajectory.last().unwrap().2;
        let final_3vel = final_vel.velocity();
        assert_relative_eq!(final_3vel.magnitude(), velocity.magnitude(), epsilon = 1e-6);
    }

    #[test]
    fn test_four_velocity_normalization() {
        let metric = Box::new(FlatSpacetime);
        let mut integrator = GeodesicIntegrator::with_metric(metric);

        let mut position = SpacetimeVector::new(0.0, 0.0, 0.0, 0.0);
        let velocity = Vector3::new(0.8 * crate::constants::C, 0.0, 0.0);
        let mut four_velocity = FourVelocity::from_velocity(velocity);

        // Take several steps
        for _ in 0..50 {
            integrator
                .step(&mut position, &mut four_velocity, 0.01)
                .expect("Step should succeed");
        }

        // Four-velocity should remain normalized
        let norm_sq = four_velocity.as_spacetime_vector().minkowski_norm_squared();
        let expected = crate::constants::C * crate::constants::C;
        assert_relative_eq!(norm_sq, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_integration_config() {
        let config = IntegrationConfig {
            max_step_size: 1.0,
            min_step_size: 1e-6,
            normalization_tolerance: 1e-12,
            renormalization_frequency: 10,
            max_steps: 1000,
            error_tolerance: 1e-10,
            safety_factor: 0.8,
        };

        let metric = Box::new(FlatSpacetime);
        let integrator = GeodesicIntegrator::new(metric, config.clone());

        assert_eq!(integrator.config().max_step_size, 1.0);
        assert_eq!(integrator.config().renormalization_frequency, 10);
    }

    #[test]
    fn test_step_size_validation() {
        let metric = Box::new(FlatSpacetime);
        let mut integrator = GeodesicIntegrator::with_metric(metric);

        let mut position = SpacetimeVector::new(0.0, 0.0, 0.0, 0.0);
        let mut four_velocity = FourVelocity::at_rest();

        // Test step size too large
        let result = integrator.step(&mut position, &mut four_velocity, 1000.0);
        assert!(matches!(
            result,
            Err(GeodesicError::StepSizeTooLarge { .. })
        ));

        // Test step size too small
        let result = integrator.step(&mut position, &mut four_velocity, 1e-12);
        assert!(matches!(
            result,
            Err(GeodesicError::IntegrationFailure { .. })
        ));
    }

    #[test]
    fn test_integration_stats() {
        let metric = Box::new(FlatSpacetime);
        let mut integrator = GeodesicIntegrator::with_metric(metric);

        let mut position = SpacetimeVector::new(0.0, 0.0, 0.0, 0.0);
        let mut four_velocity = FourVelocity::at_rest();

        integrator
            .step(&mut position, &mut four_velocity, 0.1)
            .expect("Step should succeed");

        let stats = integrator.stats();
        assert_eq!(stats.steps_taken, 1);
        assert_eq!(stats.proper_time, 0.1);
        assert_eq!(stats.last_step_size, 0.1);
        assert_eq!(stats.metric_name, "Flat Spacetime (Minkowski)");
    }
}
