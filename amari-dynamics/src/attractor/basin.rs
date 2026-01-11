//! Basin of attraction computation
//!
//! This module provides algorithms for computing and visualizing the basin
//! of attraction for different attractors in a dynamical system.
//!
//! # Overview
//!
//! The basin of attraction is the set of initial conditions that converge
//! to a particular attractor. For systems with multiple attractors, the
//! phase space is partitioned into separate basins.
//!
//! # Algorithms
//!
//! - **Grid-based**: Sample initial conditions on a regular grid
//! - **Monte Carlo**: Random sampling for high-dimensional systems
//! - **Boundary refinement**: Adaptive refinement near basin boundaries
//!
//! # Example
//!
//! ```ignore
//! use amari_dynamics::attractor::{compute_basins, BasinConfig};
//!
//! let config = BasinConfig {
//!     grid_resolution: vec![100, 100],
//!     bounds: vec![(-2.0, 2.0), (-2.0, 2.0)],
//!     ..Default::default()
//! };
//!
//! let basins = compute_basins(&system, &attractors, &config)?;
//! ```

use amari_core::Multivector;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

use crate::error::{DynamicsError, Result};
use crate::flow::DynamicalSystem;
use crate::solver::{DormandPrince, ODESolver};

use super::traits::{Attractor, Basin};

/// Configuration for basin computation
#[derive(Debug, Clone)]
pub struct BasinConfig {
    /// Resolution of the sampling grid for each dimension
    pub grid_resolution: Vec<usize>,

    /// Bounds for each dimension: (min, max)
    pub bounds: Vec<(f64, f64)>,

    /// Maximum integration time
    pub max_time: f64,

    /// Integration time step
    pub dt: f64,

    /// Tolerance for attractor detection
    pub attractor_tolerance: f64,

    /// Which dimensions to sample (indices into state space)
    /// If empty, samples all dimensions
    pub sample_dimensions: Vec<usize>,

    /// Fixed values for non-sampled dimensions
    pub fixed_values: Vec<(usize, f64)>,
}

impl Default for BasinConfig {
    fn default() -> Self {
        Self {
            grid_resolution: vec![50, 50],
            bounds: vec![(-1.0, 1.0), (-1.0, 1.0)],
            max_time: 100.0,
            dt: 0.01,
            attractor_tolerance: 0.1,
            sample_dimensions: Vec::new(),
            fixed_values: Vec::new(),
        }
    }
}

impl BasinConfig {
    /// Create a 2D basin configuration
    pub fn two_dimensional(
        x_range: (f64, f64),
        y_range: (f64, f64),
        x_resolution: usize,
        y_resolution: usize,
    ) -> Self {
        Self {
            grid_resolution: vec![x_resolution, y_resolution],
            bounds: vec![x_range, y_range],
            sample_dimensions: vec![0, 1],
            ..Default::default()
        }
    }

    /// Create a configuration with specified dimensions
    pub fn with_dimensions(dims: Vec<usize>, bounds: Vec<(f64, f64)>, resolution: usize) -> Self {
        let n = dims.len();
        Self {
            grid_resolution: vec![resolution; n],
            bounds,
            sample_dimensions: dims,
            ..Default::default()
        }
    }

    /// Get the total number of grid points
    pub fn num_grid_points(&self) -> usize {
        self.grid_resolution.iter().product()
    }
}

/// Result of basin computation for a single attractor
#[derive(Debug, Clone)]
pub struct BasinResult<const P: usize, const Q: usize, const R: usize> {
    /// Index of the attractor
    pub attractor_index: usize,

    /// Points in this basin
    pub basin: Basin<P, Q, R>,

    /// Fraction of sampled points in this basin
    pub fraction: f64,
}

/// Full result of basin computation for multiple attractors
#[derive(Debug, Clone)]
pub struct MultiBasinResult<const P: usize, const Q: usize, const R: usize> {
    /// Results for each attractor
    pub basins: Vec<BasinResult<P, Q, R>>,

    /// Points that didn't converge to any attractor
    pub unclassified: Basin<P, Q, R>,

    /// Total number of sampled points
    pub total_points: usize,

    /// Configuration used
    pub config: BasinConfig,
}

impl<const P: usize, const Q: usize, const R: usize> MultiBasinResult<P, Q, R> {
    /// Get the basin for a specific attractor
    pub fn basin_for(&self, attractor_index: usize) -> Option<&BasinResult<P, Q, R>> {
        self.basins
            .iter()
            .find(|b| b.attractor_index == attractor_index)
    }

    /// Get the fraction of points in each basin
    pub fn fractions(&self) -> Vec<(usize, f64)> {
        self.basins
            .iter()
            .map(|b| (b.attractor_index, b.fraction))
            .collect()
    }
}

/// Generate grid points for sampling
pub fn generate_grid_points<const P: usize, const Q: usize, const R: usize>(
    config: &BasinConfig,
) -> Vec<Multivector<P, Q, R>> {
    let n_dims = config.bounds.len();
    let total_points = config.num_grid_points();
    let mut points = Vec::with_capacity(total_points);

    // Generate all combinations of grid indices
    let mut indices = vec![0usize; n_dims];

    loop {
        // Create point at current indices
        let mut point = Multivector::<P, Q, R>::zero();

        // Set fixed values first
        for &(dim, val) in &config.fixed_values {
            point.set(dim, val);
        }

        // Set sampled values
        for (i, &idx) in indices.iter().enumerate() {
            let dim = if config.sample_dimensions.is_empty() {
                i
            } else {
                config.sample_dimensions[i]
            };

            let (min, max) = config.bounds[i];
            let n = config.grid_resolution[i];
            let val = if n > 1 {
                min + (max - min) * (idx as f64) / ((n - 1) as f64)
            } else {
                (min + max) / 2.0
            };

            point.set(dim, val);
        }

        points.push(point);

        // Increment indices
        let mut carry = true;
        for (i, idx) in indices.iter_mut().enumerate().take(n_dims) {
            if carry {
                *idx += 1;
                if *idx >= config.grid_resolution[i] {
                    *idx = 0;
                } else {
                    carry = false;
                }
            }
        }

        if carry {
            break; // All combinations generated
        }
    }

    points
}

/// Classify a single initial condition to an attractor
fn classify_point<S, A, const P: usize, const Q: usize, const R: usize>(
    system: &S,
    initial: &Multivector<P, Q, R>,
    attractors: &[A],
    config: &BasinConfig,
) -> Option<usize>
where
    S: DynamicalSystem<P, Q, R>,
    A: Attractor<P, Q, R>,
{
    let solver = DormandPrince::new();
    let steps = (config.max_time / config.dt) as usize;

    // Integrate trajectory
    let trajectory = match solver.solve(system, initial.clone(), 0.0, config.max_time, steps) {
        Ok(t) => t,
        Err(_) => return None,
    };

    // Check final state against attractors
    if let Some(final_state) = trajectory.final_state() {
        for (idx, attractor) in attractors.iter().enumerate() {
            if attractor.contains(final_state, config.attractor_tolerance) {
                return Some(idx);
            }
        }
    }

    None
}

/// Compute basins of attraction for multiple attractors (sequential)
pub fn compute_basins<S, A, const P: usize, const Q: usize, const R: usize>(
    system: &S,
    attractors: &[A],
    config: &BasinConfig,
) -> Result<MultiBasinResult<P, Q, R>>
where
    S: DynamicalSystem<P, Q, R>,
    A: Attractor<P, Q, R>,
{
    if attractors.is_empty() {
        return Err(DynamicsError::invalid_parameter(
            "At least one attractor required",
        ));
    }

    let grid_points = generate_grid_points(config);
    let total_points = grid_points.len();

    let mut basin_points: Vec<Vec<Multivector<P, Q, R>>> = vec![Vec::new(); attractors.len()];
    let mut unclassified_points = Vec::new();

    for point in grid_points {
        match classify_point(system, &point, attractors, config) {
            Some(idx) => basin_points[idx].push(point),
            None => unclassified_points.push(point),
        }
    }

    let basins: Vec<BasinResult<P, Q, R>> = basin_points
        .into_iter()
        .enumerate()
        .map(|(idx, points)| {
            let fraction = points.len() as f64 / total_points as f64;
            BasinResult {
                attractor_index: idx,
                basin: Basin::from_points(points),
                fraction,
            }
        })
        .collect();

    Ok(MultiBasinResult {
        basins,
        unclassified: Basin::from_points(unclassified_points),
        total_points,
        config: config.clone(),
    })
}

/// Compute basins of attraction in parallel
#[cfg(feature = "parallel")]
pub fn compute_basins_parallel<S, A, const P: usize, const Q: usize, const R: usize>(
    system: &S,
    attractors: &[A],
    config: &BasinConfig,
) -> Result<MultiBasinResult<P, Q, R>>
where
    S: DynamicalSystem<P, Q, R> + Sync,
    A: Attractor<P, Q, R> + Sync,
{
    if attractors.is_empty() {
        return Err(DynamicsError::invalid_parameter(
            "At least one attractor required",
        ));
    }

    let grid_points = generate_grid_points(config);
    let total_points = grid_points.len();

    // Classify all points in parallel
    let classifications: Vec<(Multivector<P, Q, R>, Option<usize>)> = grid_points
        .into_par_iter()
        .map(|point| {
            let class = classify_point(system, &point, attractors, config);
            (point, class)
        })
        .collect();

    // Separate into basins
    let mut basin_points: Vec<Vec<Multivector<P, Q, R>>> = vec![Vec::new(); attractors.len()];
    let mut unclassified_points = Vec::new();

    for (point, class) in classifications {
        match class {
            Some(idx) => basin_points[idx].push(point),
            None => unclassified_points.push(point),
        }
    }

    let basins: Vec<BasinResult<P, Q, R>> = basin_points
        .into_iter()
        .enumerate()
        .map(|(idx, points)| {
            let fraction = points.len() as f64 / total_points as f64;
            BasinResult {
                attractor_index: idx,
                basin: Basin::from_points(points),
                fraction,
            }
        })
        .collect();

    Ok(MultiBasinResult {
        basins,
        unclassified: Basin::from_points(unclassified_points),
        total_points,
        config: config.clone(),
    })
}

/// Compute basin boundary by adaptive refinement
///
/// Refines sampling near basin boundaries to get a more accurate picture.
pub fn refine_boundary<S, A, const P: usize, const Q: usize, const R: usize>(
    system: &S,
    attractors: &[A],
    initial_result: &MultiBasinResult<P, Q, R>,
    refinement_levels: usize,
) -> Result<Vec<Multivector<P, Q, R>>>
where
    S: DynamicalSystem<P, Q, R>,
    A: Attractor<P, Q, R>,
{
    let mut boundary_points = Vec::new();
    let config = &initial_result.config;

    // Find points near boundaries (where neighbors belong to different basins)
    // This is a simplified implementation - a full version would track which
    // basin each grid point belongs to

    // For now, just return unclassified points as potential boundary region
    for point in &initial_result.unclassified.points {
        boundary_points.push(point.clone());
    }

    // Refine around boundary points
    for _level in 0..refinement_levels {
        let mut new_boundary = Vec::new();

        for point in &boundary_points {
            // Sample nearby points
            for i in 0..config.bounds.len() {
                let dim = if config.sample_dimensions.is_empty() {
                    i
                } else {
                    config.sample_dimensions[i]
                };

                let (min, max) = config.bounds[i];
                let delta = (max - min) / (config.grid_resolution[i] as f64 * 4.0);

                for &offset in &[-delta, delta] {
                    let mut neighbor = point.clone();
                    neighbor.set(dim, neighbor.get(dim) + offset);

                    // Check if in bounds
                    let val = neighbor.get(dim);
                    if val >= min
                        && val <= max
                        && classify_point(system, &neighbor, attractors, config).is_none()
                    {
                        new_boundary.push(neighbor);
                    }
                }
            }
        }

        boundary_points = new_boundary;
    }

    Ok(boundary_points)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basin_config_default() {
        let config = BasinConfig::default();
        assert_eq!(config.grid_resolution.len(), 2);
        assert_eq!(config.num_grid_points(), 2500); // 50 * 50
    }

    #[test]
    fn test_basin_config_2d() {
        let config = BasinConfig::two_dimensional((-2.0, 2.0), (-2.0, 2.0), 10, 10);
        assert_eq!(config.grid_resolution, vec![10, 10]);
        assert_eq!(config.sample_dimensions, vec![0, 1]);
        assert_eq!(config.num_grid_points(), 100);
    }

    #[test]
    fn test_generate_grid_points() {
        let config = BasinConfig {
            grid_resolution: vec![3, 3],
            bounds: vec![(0.0, 1.0), (0.0, 1.0)],
            sample_dimensions: vec![0, 1],
            ..Default::default()
        };

        let points = generate_grid_points::<2, 0, 0>(&config);
        assert_eq!(points.len(), 9);

        // Check corners
        let has_origin = points
            .iter()
            .any(|p| (p.get(0) - 0.0).abs() < 1e-10 && (p.get(1) - 0.0).abs() < 1e-10);
        assert!(has_origin);

        let has_corner = points
            .iter()
            .any(|p| (p.get(0) - 1.0).abs() < 1e-10 && (p.get(1) - 1.0).abs() < 1e-10);
        assert!(has_corner);
    }

    #[test]
    fn test_grid_with_fixed_values() {
        let config = BasinConfig {
            grid_resolution: vec![3],
            bounds: vec![(0.0, 1.0)],
            sample_dimensions: vec![0],
            fixed_values: vec![(1, 5.0), (2, -3.0)],
            ..Default::default()
        };

        let points = generate_grid_points::<3, 0, 0>(&config);
        assert_eq!(points.len(), 3);

        // All points should have fixed values
        for point in &points {
            assert_eq!(point.get(1), 5.0);
            assert_eq!(point.get(2), -3.0);
        }
    }
}
