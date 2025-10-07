//! High-precision geodesic integration for spacecraft orbital mechanics
//!
//! This module extends the standard geodesic integrator with arbitrary precision
//! arithmetic for critical spacecraft orbital calculations where numerical precision
//! is essential for accurate trajectory prediction.

use crate::precision::PrecisionFloat;
use crate::spacetime::SpacetimeVector;
use nalgebra::Vector3;

#[cfg(feature = "std")]
use std::fmt;

#[cfg(not(feature = "std"))]
use core::fmt;

#[cfg(not(feature = "std"))]
extern crate alloc;

#[cfg(not(feature = "std"))]
use alloc::{boxed::Box, format, string::String, vec::Vec};

/// High-precision metric trait for spacecraft orbital mechanics
pub trait PrecisionMetric<T: PrecisionFloat> {
    /// Compute metric tensor components at given spacetime point
    fn metric_tensor_precise(&self, position: &PrecisionSpacetimeVector<T>) -> [[T; 4]; 4];

    /// Compute Christoffel symbols at given spacetime point
    fn christoffel_precise(&self, position: &PrecisionSpacetimeVector<T>) -> [[[T; 4]; 4]; 4];

    /// Check for coordinate singularities
    fn has_singularity_precise(&self, position: &PrecisionSpacetimeVector<T>) -> bool;

    /// Get metric name for debugging
    fn name(&self) -> &str;

    /// Characteristic length scale for adaptive stepping
    fn characteristic_scale_precise(&self) -> T {
        <T as PrecisionFloat>::from_f64(1e6) // 1000 km default
    }
}

/// High-precision spacetime vector for orbital mechanics
#[derive(Clone, Debug)]
pub struct PrecisionSpacetimeVector<T: PrecisionFloat> {
    /// Temporal component (ct)
    pub t: T,
    /// Spatial x component
    pub x: T,
    /// Spatial y component
    pub y: T,
    /// Spatial z component
    pub z: T,
}

impl<T: PrecisionFloat> PrecisionSpacetimeVector<T> {
    /// Create new precision spacetime vector
    pub fn new(t: T, x: T, y: T, z: T) -> Self {
        Self { t, x, y, z }
    }

    /// Create from standard spacetime vector
    pub fn from_spacetime_vector(sv: &SpacetimeVector) -> Self {
        Self {
            t: <T as PrecisionFloat>::from_f64(sv.time()),
            x: <T as PrecisionFloat>::from_f64(sv.x()),
            y: <T as PrecisionFloat>::from_f64(sv.y()),
            z: <T as PrecisionFloat>::from_f64(sv.z()),
        }
    }

    /// Convert to standard spacetime vector (may lose precision)
    pub fn to_spacetime_vector(&self) -> SpacetimeVector {
        SpacetimeVector::new(
            self.t.to_f64(),
            self.x.to_f64(),
            self.y.to_f64(),
            self.z.to_f64(),
        )
    }

    /// Get spatial components as Vector3
    pub fn spatial(&self) -> Vector3<f64> {
        Vector3::new(self.x.to_f64(), self.y.to_f64(), self.z.to_f64())
    }

    /// Minkowski inner product with high precision
    pub fn minkowski_dot(&self, other: &Self) -> T {
        self.t * other.t - self.x * other.x - self.y * other.y - self.z * other.z
    }

    /// Minkowski norm squared
    pub fn norm_squared(&self) -> T {
        self.minkowski_dot(self)
    }

    /// Spatial magnitude squared
    pub fn spatial_magnitude_squared(&self) -> T {
        self.x * self.x + self.y * self.y + self.z * self.z
    }

    /// Component access by index
    pub fn component(&self, i: usize) -> T {
        match i {
            0 => self.t,
            1 => self.x,
            2 => self.y,
            3 => self.z,
            _ => panic!("Invalid component index {}", i),
        }
    }

    /// Mutable component access by index
    pub fn component_mut(&mut self, i: usize) -> &mut T {
        match i {
            0 => &mut self.t,
            1 => &mut self.x,
            2 => &mut self.y,
            3 => &mut self.z,
            _ => panic!("Invalid component index {}", i),
        }
    }
}

/// High-precision integration configuration for orbital mechanics
#[derive(Clone, Debug)]
pub struct PrecisionIntegrationConfig<T: PrecisionFloat> {
    /// Maximum step size in proper time
    pub max_step_size: T,
    /// Minimum step size in proper time
    pub min_step_size: T,
    /// Tolerance for four-velocity normalization
    pub normalization_tolerance: T,
    /// Frequency of renormalization
    pub renormalization_frequency: usize,
    /// Maximum integration steps
    pub max_steps: usize,
    /// Error tolerance for adaptive stepping
    pub error_tolerance: T,
    /// Safety factor for step adjustment
    pub safety_factor: T,
}

impl<T: PrecisionFloat> Default for PrecisionIntegrationConfig<T> {
    fn default() -> Self {
        Self {
            max_step_size: <T as PrecisionFloat>::from_f64(100.0),
            min_step_size: <T as PrecisionFloat>::from_f64(1e-9), // Smaller for high precision
            normalization_tolerance: T::orbital_tolerance(),
            renormalization_frequency: 50, // More frequent for precision
            max_steps: 10_000_000,         // More steps allowed
            error_tolerance: T::orbital_tolerance(),
            safety_factor: <T as PrecisionFloat>::from_f64(0.8), // More conservative
        }
    }
}

/// High-precision geodesic integrator for spacecraft orbital mechanics
pub struct PrecisionGeodesicIntegrator<T: PrecisionFloat> {
    /// High-precision metric
    metric: Box<dyn PrecisionMetric<T>>,
    /// Current proper time
    pub proper_time: T,
    /// Integration configuration
    config: PrecisionIntegrationConfig<T>,
    /// Current step size
    current_step: T,
    /// Integration step counter
    step_count: usize,
}

impl<T: PrecisionFloat> PrecisionGeodesicIntegrator<T> {
    /// Create new high-precision integrator
    pub fn new(metric: Box<dyn PrecisionMetric<T>>, config: PrecisionIntegrationConfig<T>) -> Self {
        Self {
            metric,
            proper_time: T::zero(),
            current_step: config.max_step_size,
            config,
            step_count: 0,
        }
    }

    /// Create with default configuration
    pub fn with_metric(metric: Box<dyn PrecisionMetric<T>>) -> Self {
        Self::new(metric, PrecisionIntegrationConfig::default())
    }

    /// Perform single integration step using velocity Verlet method
    pub fn step(
        &mut self,
        position: &mut PrecisionSpacetimeVector<T>,
        velocity: &mut PrecisionSpacetimeVector<T>,
        step_size: T,
    ) -> Result<(), String> {
        // Check for singularities
        if self.metric.has_singularity_precise(position) {
            return Err("Metric singularity encountered".to_string());
        }

        // Compute initial acceleration
        let accel_initial = self.compute_acceleration(position, velocity)?;

        // Position update: x_{n+1} = x_n + v_n * dt + 0.5 * a_n * dt^2
        let half_step_sq = step_size * step_size * <T as PrecisionFloat>::from_f64(0.5);
        for i in 0..4 {
            *position.component_mut(i) = position.component(i)
                + velocity.component(i) * step_size
                + accel_initial.component(i) * half_step_sq;
        }

        // Compute acceleration at new position
        let accel_final = self.compute_acceleration(position, velocity)?;

        // Velocity update: v_{n+1} = v_n + 0.5 * (a_n + a_{n+1}) * dt
        let half_step = step_size * <T as PrecisionFloat>::from_f64(0.5);
        for i in 0..4 {
            *velocity.component_mut(i) = velocity.component(i)
                + (accel_initial.component(i) + accel_final.component(i)) * half_step;
        }

        // Renormalize four-velocity periodically for massive particles
        if self
            .step_count
            .is_multiple_of(self.config.renormalization_frequency)
        {
            self.renormalize_four_velocity(velocity)?;
        }

        self.proper_time = self.proper_time + step_size;
        self.step_count += 1;

        Ok(())
    }

    /// Compute geodesic acceleration d²x^μ/dτ²
    #[allow(clippy::needless_range_loop)] // Christoffel symbol indexing requires explicit loops
    fn compute_acceleration(
        &self,
        position: &PrecisionSpacetimeVector<T>,
        velocity: &PrecisionSpacetimeVector<T>,
    ) -> Result<PrecisionSpacetimeVector<T>, String> {
        let christoffel = self.metric.christoffel_precise(position);

        let mut acceleration =
            PrecisionSpacetimeVector::new(T::zero(), T::zero(), T::zero(), T::zero());

        // a^μ = -Γ^μ_αβ v^α v^β
        for mu in 0..4 {
            let mut sum = T::zero();
            for alpha in 0..4 {
                for beta in 0..4 {
                    sum = sum
                        + christoffel[mu][alpha][beta]
                            * velocity.component(alpha)
                            * velocity.component(beta);
                }
            }
            *acceleration.component_mut(mu) = T::zero() - sum;
        }

        Ok(acceleration)
    }

    /// Renormalize four-velocity to maintain u·u = c²
    fn renormalize_four_velocity(
        &self,
        velocity: &mut PrecisionSpacetimeVector<T>,
    ) -> Result<(), String> {
        let c = <T as PrecisionFloat>::from_f64(crate::constants::C);
        let norm_squared = velocity.norm_squared();
        let expected = c * c;

        // Check if renormalization is needed
        let deviation = (norm_squared - expected).abs_precise();
        if deviation < self.config.normalization_tolerance {
            return Ok(()); // Already normalized
        }

        // Renormalize: u^μ = u^μ * c / |u|
        let norm = norm_squared.sqrt_precise();
        if norm <= T::zero() {
            return Err("Zero four-velocity cannot be renormalized".to_string());
        }

        let factor = c / norm;
        velocity.t = velocity.t * factor;
        velocity.x = velocity.x * factor;
        velocity.y = velocity.y * factor;
        velocity.z = velocity.z * factor;

        Ok(())
    }

    /// Adaptive step size control based on error estimation
    pub fn adaptive_step(
        &mut self,
        position: &mut PrecisionSpacetimeVector<T>,
        velocity: &mut PrecisionSpacetimeVector<T>,
    ) -> Result<T, String> {
        let mut step_size = self.current_step;
        let max_attempts = 10;

        for _ in 0..max_attempts {
            // Store current state
            let pos_backup = position.clone();
            let vel_backup = velocity.clone();

            // Try full step
            match self.step(position, velocity, step_size.clone()) {
                Ok(()) => {
                    // Step succeeded, check if we can increase step size
                    if step_size < self.config.max_step_size {
                        step_size = (step_size * <T as PrecisionFloat>::from_f64(1.1))
                            .min(self.config.max_step_size.clone());
                    }
                    self.current_step = step_size.clone();
                    return Ok(step_size);
                }
                Err(_) => {
                    // Step failed, restore state and reduce step size
                    *position = pos_backup;
                    *velocity = vel_backup;
                    step_size = step_size * self.config.safety_factor.clone();

                    if step_size < self.config.min_step_size {
                        return Err("Step size below minimum threshold".to_string());
                    }
                }
            }
        }

        Err("Adaptive stepping failed after maximum attempts".to_string())
    }

    /// Get current integration statistics
    pub fn stats(&self) -> PrecisionIntegrationStats<T> {
        PrecisionIntegrationStats {
            proper_time: self.proper_time.clone(),
            step_count: self.step_count,
            current_step_size: self.current_step.clone(),
            metric_name: self.metric.name().to_string(),
        }
    }
}

/// Integration statistics for monitoring
#[derive(Debug, Clone)]
pub struct PrecisionIntegrationStats<T: PrecisionFloat> {
    /// Current proper time in the integration
    pub proper_time: T,
    /// Number of integration steps completed
    pub step_count: usize,
    /// Current adaptive step size
    pub current_step_size: T,
    /// Name of the metric being used
    pub metric_name: String,
}

impl<T: PrecisionFloat> fmt::Display for PrecisionIntegrationStats<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Integration Stats: τ={:.6e}, steps={}, Δτ={:.3e}, metric={}",
            self.proper_time.to_f64(),
            self.step_count,
            self.current_step_size.to_f64(),
            self.metric_name
        )
    }
}

/// Result type for precision geodesic operations
pub type PrecisionGeodesicResult<T> = Result<T, String>;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::precision::StandardFloat;

    #[test]
    fn test_precision_spacetime_vector() {
        let vec = PrecisionSpacetimeVector::<StandardFloat>::new(1.0, 0.5, 0.0, 0.0);
        assert!(vec.norm_squared() > 0.0); // Timelike
    }

    #[test]
    fn test_precision_config_defaults() {
        let config = PrecisionIntegrationConfig::<StandardFloat>::default();
        assert!(config.max_step_size > config.min_step_size);
        assert!(config.normalization_tolerance > 0.0);
    }

    #[test]
    fn test_component_access() {
        let mut vec = PrecisionSpacetimeVector::<StandardFloat>::new(1.0, 2.0, 3.0, 4.0);
        assert_eq!(vec.component(0), 1.0);
        assert_eq!(vec.component(1), 2.0);
        assert_eq!(vec.component(2), 3.0);
        assert_eq!(vec.component(3), 4.0);

        *vec.component_mut(0) = 5.0;
        assert_eq!(vec.component(0), 5.0);
    }
}
