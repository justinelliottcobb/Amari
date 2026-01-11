//! Nullcline computation for 2D systems
//!
//! This module provides algorithms for computing nullclines (isoclines)
//! of 2D dynamical systems, where one component of the vector field is zero.
//!
//! # Overview
//!
//! For a 2D system dx/dt = f(x,y), dy/dt = g(x,y):
//!
//! - **x-nullcline**: The curve where f(x,y) = 0 (dx/dt = 0)
//! - **y-nullcline**: The curve where g(x,y) = 0 (dy/dt = 0)
//!
//! Fixed points occur at intersections of nullclines.
//!
//! # Example
//!
//! ```ignore
//! use amari_dynamics::phase::{compute_nullclines, NullclineConfig};
//!
//! let config = NullclineConfig::default();
//! let nullclines = compute_nullclines(&system, &config)?;
//!
//! println!("x-nullcline has {} points", nullclines.x_nullcline.len());
//! println!("y-nullcline has {} points", nullclines.y_nullcline.len());
//! ```

use amari_core::Multivector;

use crate::error::Result;
use crate::flow::DynamicalSystem;

/// Configuration for nullcline computation
#[derive(Debug, Clone)]
pub struct NullclineConfig {
    /// Bounds for x dimension: (min, max)
    pub x_bounds: (f64, f64),

    /// Bounds for y dimension: (min, max)
    pub y_bounds: (f64, f64),

    /// Which component index corresponds to x
    pub x_component: usize,

    /// Which component index corresponds to y
    pub y_component: usize,

    /// Resolution for contour detection
    pub resolution: usize,

    /// Tolerance for zero detection
    pub zero_tolerance: f64,

    /// Fixed values for other components
    pub fixed_values: Vec<(usize, f64)>,
}

impl Default for NullclineConfig {
    fn default() -> Self {
        Self {
            x_bounds: (-2.0, 2.0),
            y_bounds: (-2.0, 2.0),
            x_component: 1,
            y_component: 2,
            resolution: 100,
            zero_tolerance: 1e-6,
            fixed_values: Vec::new(),
        }
    }
}

impl NullclineConfig {
    /// Create configuration for specified bounds
    pub fn new(x_bounds: (f64, f64), y_bounds: (f64, f64)) -> Self {
        Self {
            x_bounds,
            y_bounds,
            ..Default::default()
        }
    }

    /// Set resolution
    pub fn with_resolution(mut self, res: usize) -> Self {
        self.resolution = res;
        self
    }

    /// Set component indices
    pub fn with_components(mut self, x: usize, y: usize) -> Self {
        self.x_component = x;
        self.y_component = y;
        self
    }
}

/// A point on a nullcline
#[derive(Debug, Clone)]
pub struct NullclinePoint<const P: usize, const Q: usize, const R: usize> {
    /// Position on the nullcline
    pub position: Multivector<P, Q, R>,

    /// x coordinate
    pub x: f64,

    /// y coordinate
    pub y: f64,

    /// Value of the other component of the vector field
    /// (e.g., dy/dt value on the x-nullcline)
    pub other_component_value: f64,
}

/// Collection of nullcline segments
#[derive(Debug, Clone)]
pub struct NullclineSegment<const P: usize, const Q: usize, const R: usize> {
    /// Points along this segment
    pub points: Vec<NullclinePoint<P, Q, R>>,
}

/// Result of nullcline computation
#[derive(Debug, Clone)]
pub struct NullclineResult<const P: usize, const Q: usize, const R: usize> {
    /// X-nullcline points (where dx/dt = 0)
    pub x_nullcline: Vec<NullclineSegment<P, Q, R>>,

    /// Y-nullcline points (where dy/dt = 0)
    pub y_nullcline: Vec<NullclineSegment<P, Q, R>>,

    /// Intersection points (approximate fixed points)
    pub intersections: Vec<Multivector<P, Q, R>>,

    /// Configuration used
    pub config: NullclineConfig,
}

impl<const P: usize, const Q: usize, const R: usize> NullclineResult<P, Q, R> {
    /// Get total number of x-nullcline points
    pub fn num_x_nullcline_points(&self) -> usize {
        self.x_nullcline.iter().map(|s| s.points.len()).sum()
    }

    /// Get total number of y-nullcline points
    pub fn num_y_nullcline_points(&self) -> usize {
        self.y_nullcline.iter().map(|s| s.points.len()).sum()
    }

    /// Get all x-nullcline points as (x, y) pairs
    pub fn x_nullcline_coords(&self) -> Vec<(f64, f64)> {
        self.x_nullcline
            .iter()
            .flat_map(|s| s.points.iter().map(|p| (p.x, p.y)))
            .collect()
    }

    /// Get all y-nullcline points as (x, y) pairs
    pub fn y_nullcline_coords(&self) -> Vec<(f64, f64)> {
        self.y_nullcline
            .iter()
            .flat_map(|s| s.points.iter().map(|p| (p.x, p.y)))
            .collect()
    }
}

/// Compute nullclines for a 2D system
pub fn compute_nullclines<S, const P: usize, const Q: usize, const R: usize>(
    system: &S,
    config: &NullclineConfig,
) -> Result<NullclineResult<P, Q, R>>
where
    S: DynamicalSystem<P, Q, R>,
{
    let n = config.resolution;
    let dx = (config.x_bounds.1 - config.x_bounds.0) / (n - 1) as f64;
    let dy = (config.y_bounds.1 - config.y_bounds.0) / (n - 1) as f64;

    // Evaluate vector field on grid
    let mut fx_grid = vec![vec![0.0; n]; n];
    let mut fy_grid = vec![vec![0.0; n]; n];

    for i in 0..n {
        for j in 0..n {
            let x = config.x_bounds.0 + i as f64 * dx;
            let y = config.y_bounds.0 + j as f64 * dy;

            let mut state = Multivector::<P, Q, R>::zero();
            state.set(config.x_component, x);
            state.set(config.y_component, y);

            for &(idx, val) in &config.fixed_values {
                state.set(idx, val);
            }

            if let Ok(vf) = system.vector_field(&state) {
                fx_grid[i][j] = vf.get(config.x_component);
                fy_grid[i][j] = vf.get(config.y_component);
            }
        }
    }

    // Find x-nullcline (where fx = 0)
    let x_nullcline_points = find_zero_contour(&fx_grid, &fy_grid, config, true)?;
    let x_nullcline = segment_nullcline_points(&x_nullcline_points, dx.max(dy) * 2.0);

    // Find y-nullcline (where fy = 0)
    let y_nullcline_points = find_zero_contour(&fy_grid, &fx_grid, config, false)?;
    let y_nullcline = segment_nullcline_points(&y_nullcline_points, dx.max(dy) * 2.0);

    // Find intersections
    let intersections = find_nullcline_intersections(&x_nullcline, &y_nullcline, config);

    Ok(NullclineResult {
        x_nullcline,
        y_nullcline,
        intersections,
        config: config.clone(),
    })
}

/// Find zero contour using marching squares algorithm
fn find_zero_contour<const P: usize, const Q: usize, const R: usize>(
    values: &[Vec<f64>],
    other_values: &[Vec<f64>],
    config: &NullclineConfig,
    is_x_nullcline: bool,
) -> Result<Vec<NullclinePoint<P, Q, R>>> {
    let n = config.resolution;
    let dx = (config.x_bounds.1 - config.x_bounds.0) / (n - 1) as f64;
    let dy = (config.y_bounds.1 - config.y_bounds.0) / (n - 1) as f64;

    let mut points = Vec::new();

    for i in 0..(n - 1) {
        for j in 0..(n - 1) {
            // Get values at cell corners
            let v00 = values[i][j];
            let v10 = values[i + 1][j];
            let v01 = values[i][j + 1];
            let v11 = values[i + 1][j + 1];

            // Check for sign changes (zero crossings)
            let has_crossing =
                (v00 * v10 < 0.0) || (v00 * v01 < 0.0) || (v10 * v11 < 0.0) || (v01 * v11 < 0.0);

            if has_crossing {
                // Use linear interpolation to find crossing points
                let cell_x = config.x_bounds.0 + i as f64 * dx;
                let cell_y = config.y_bounds.0 + j as f64 * dy;

                // Check each edge
                let crossings = find_edge_crossings(cell_x, cell_y, dx, dy, v00, v10, v01, v11);

                for (px, py) in crossings {
                    // Get the other component value at this point
                    let fi = ((px - config.x_bounds.0) / dx) as usize;
                    let fj = ((py - config.y_bounds.0) / dy) as usize;
                    let fi = fi.min(n - 1);
                    let fj = fj.min(n - 1);
                    let other_val = other_values[fi][fj];

                    let mut position = Multivector::<P, Q, R>::zero();
                    position.set(config.x_component, px);
                    position.set(config.y_component, py);

                    for &(idx, val) in &config.fixed_values {
                        position.set(idx, val);
                    }

                    let _is_x = is_x_nullcline; // Mark as used

                    points.push(NullclinePoint {
                        position,
                        x: px,
                        y: py,
                        other_component_value: other_val,
                    });
                }
            }
        }
    }

    Ok(points)
}

/// Find zero crossings on cell edges
#[allow(clippy::too_many_arguments)]
fn find_edge_crossings(
    x0: f64,
    y0: f64,
    dx: f64,
    dy: f64,
    v00: f64,
    v10: f64,
    v01: f64,
    v11: f64,
) -> Vec<(f64, f64)> {
    let mut crossings = Vec::new();

    // Bottom edge (y = y0)
    if v00 * v10 < 0.0 {
        let t = -v00 / (v10 - v00);
        crossings.push((x0 + t * dx, y0));
    }

    // Top edge (y = y0 + dy)
    if v01 * v11 < 0.0 {
        let t = -v01 / (v11 - v01);
        crossings.push((x0 + t * dx, y0 + dy));
    }

    // Left edge (x = x0)
    if v00 * v01 < 0.0 {
        let t = -v00 / (v01 - v00);
        crossings.push((x0, y0 + t * dy));
    }

    // Right edge (x = x0 + dx)
    if v10 * v11 < 0.0 {
        let t = -v10 / (v11 - v10);
        crossings.push((x0 + dx, y0 + t * dy));
    }

    crossings
}

/// Segment nullcline points into connected segments
fn segment_nullcline_points<const P: usize, const Q: usize, const R: usize>(
    points: &[NullclinePoint<P, Q, R>],
    max_gap: f64,
) -> Vec<NullclineSegment<P, Q, R>> {
    if points.is_empty() {
        return Vec::new();
    }

    // Sort points by x then y for consistent ordering
    let mut sorted_points: Vec<_> = points.to_vec();
    sorted_points.sort_by(|a, b| {
        a.x.partial_cmp(&b.x)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.y.partial_cmp(&b.y).unwrap_or(std::cmp::Ordering::Equal))
    });

    let mut segments = Vec::new();
    let mut current_segment = vec![sorted_points[0].clone()];

    for point in sorted_points.iter().skip(1) {
        let last = current_segment.last().unwrap();
        let dist = ((point.x - last.x).powi(2) + (point.y - last.y).powi(2)).sqrt();

        if dist <= max_gap {
            current_segment.push(point.clone());
        } else {
            if !current_segment.is_empty() {
                segments.push(NullclineSegment {
                    points: current_segment,
                });
            }
            current_segment = vec![point.clone()];
        }
    }

    if !current_segment.is_empty() {
        segments.push(NullclineSegment {
            points: current_segment,
        });
    }

    segments
}

/// Find intersection points of x and y nullclines
fn find_nullcline_intersections<const P: usize, const Q: usize, const R: usize>(
    x_nullcline: &[NullclineSegment<P, Q, R>],
    y_nullcline: &[NullclineSegment<P, Q, R>],
    config: &NullclineConfig,
) -> Vec<Multivector<P, Q, R>> {
    let mut intersections = Vec::new();
    let tolerance = 3.0 * (config.x_bounds.1 - config.x_bounds.0) / config.resolution as f64;

    for x_seg in x_nullcline {
        for y_seg in y_nullcline {
            for x_point in &x_seg.points {
                for y_point in &y_seg.points {
                    let dist =
                        ((x_point.x - y_point.x).powi(2) + (x_point.y - y_point.y).powi(2)).sqrt();
                    if dist < tolerance {
                        // Average the two points
                        let avg_x = (x_point.x + y_point.x) / 2.0;
                        let avg_y = (x_point.y + y_point.y) / 2.0;

                        // Check if we already have a nearby intersection
                        let is_new = !intersections.iter().any(|p: &Multivector<P, Q, R>| {
                            let dx = p.get(config.x_component) - avg_x;
                            let dy = p.get(config.y_component) - avg_y;
                            (dx * dx + dy * dy).sqrt() < tolerance
                        });

                        if is_new {
                            let mut intersection = Multivector::<P, Q, R>::zero();
                            intersection.set(config.x_component, avg_x);
                            intersection.set(config.y_component, avg_y);

                            for &(idx, val) in &config.fixed_values {
                                intersection.set(idx, val);
                            }

                            intersections.push(intersection);
                        }
                    }
                }
            }
        }
    }

    intersections
}

/// Compute direction field on nullclines
pub fn nullcline_flow_direction<S, const P: usize, const Q: usize, const R: usize>(
    system: &S,
    nullclines: &NullclineResult<P, Q, R>,
) -> Result<NullclineFlowInfo<P, Q, R>>
where
    S: DynamicalSystem<P, Q, R>,
{
    let config = &nullclines.config;

    // For x-nullcline: compute sign of dy/dt
    let mut x_nullcline_directions = Vec::new();
    for segment in &nullclines.x_nullcline {
        let mut segment_dirs = Vec::new();
        for point in &segment.points {
            if let Ok(vf) = system.vector_field(&point.position) {
                let dy = vf.get(config.y_component);
                segment_dirs.push(FlowDirection {
                    position: point.position.clone(),
                    direction: if dy > 0.0 {
                        Direction::Up
                    } else if dy < 0.0 {
                        Direction::Down
                    } else {
                        Direction::None
                    },
                    magnitude: dy.abs(),
                });
            }
        }
        x_nullcline_directions.push(segment_dirs);
    }

    // For y-nullcline: compute sign of dx/dt
    let mut y_nullcline_directions = Vec::new();
    for segment in &nullclines.y_nullcline {
        let mut segment_dirs = Vec::new();
        for point in &segment.points {
            if let Ok(vf) = system.vector_field(&point.position) {
                let dx = vf.get(config.x_component);
                segment_dirs.push(FlowDirection {
                    position: point.position.clone(),
                    direction: if dx > 0.0 {
                        Direction::Right
                    } else if dx < 0.0 {
                        Direction::Left
                    } else {
                        Direction::None
                    },
                    magnitude: dx.abs(),
                });
            }
        }
        y_nullcline_directions.push(segment_dirs);
    }

    Ok(NullclineFlowInfo {
        x_nullcline_directions,
        y_nullcline_directions,
    })
}

/// Direction of flow
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Direction {
    /// Upward flow (positive y direction)
    Up,
    /// Downward flow (negative y direction)
    Down,
    /// Leftward flow (negative x direction)
    Left,
    /// Rightward flow (positive x direction)
    Right,
    /// No flow (at fixed point)
    None,
}

/// Flow direction at a point
#[derive(Debug, Clone)]
pub struct FlowDirection<const P: usize, const Q: usize, const R: usize> {
    /// Position in phase space
    pub position: Multivector<P, Q, R>,
    /// Direction of flow
    pub direction: Direction,
    /// Magnitude of the flow velocity
    pub magnitude: f64,
}

/// Flow direction information along nullclines
#[derive(Debug, Clone)]
pub struct NullclineFlowInfo<const P: usize, const Q: usize, const R: usize> {
    /// Flow directions on x-nullcline segments
    pub x_nullcline_directions: Vec<Vec<FlowDirection<P, Q, R>>>,

    /// Flow directions on y-nullcline segments
    pub y_nullcline_directions: Vec<Vec<FlowDirection<P, Q, R>>>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nullcline_config_default() {
        let config = NullclineConfig::default();
        assert_eq!(config.x_component, 1);
        assert_eq!(config.y_component, 2);
        assert!(config.resolution > 0);
    }

    #[test]
    fn test_nullcline_config_builder() {
        let config = NullclineConfig::new((-1.0, 1.0), (-1.0, 1.0))
            .with_resolution(50)
            .with_components(0, 1);

        assert_eq!(config.x_bounds, (-1.0, 1.0));
        assert_eq!(config.resolution, 50);
        assert_eq!(config.x_component, 0);
        assert_eq!(config.y_component, 1);
    }

    #[test]
    fn test_edge_crossings_bottom() {
        // Zero crossing on bottom edge and left edge
        // v00=-1, v10=1, v01=1, v11=1
        // bottom: v00*v10 = -1 < 0 -> crossing
        // left: v00*v01 = -1 < 0 -> crossing
        let crossings = find_edge_crossings(0.0, 0.0, 1.0, 1.0, -1.0, 1.0, 1.0, 1.0);
        assert_eq!(crossings.len(), 2);

        // Find the bottom edge crossing (y = 0)
        let bottom_crossing = crossings.iter().find(|c| c.1.abs() < 1e-10);
        assert!(bottom_crossing.is_some());
        let bc = bottom_crossing.unwrap();
        assert!((bc.0 - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_edge_crossings_all() {
        // Zero crossings on all edges
        let crossings = find_edge_crossings(0.0, 0.0, 1.0, 1.0, -1.0, 1.0, 1.0, -1.0);
        // Should have 4 crossings
        assert_eq!(crossings.len(), 4);
    }

    #[test]
    fn test_direction_enum() {
        assert_ne!(Direction::Up, Direction::Down);
        assert_eq!(Direction::None, Direction::None);
    }

    #[test]
    fn test_nullcline_result_methods() {
        let segment1 = NullclineSegment {
            points: vec![
                NullclinePoint {
                    position: Multivector::<2, 0, 0>::zero(),
                    x: 0.0,
                    y: 0.0,
                    other_component_value: 1.0,
                },
                NullclinePoint {
                    position: Multivector::<2, 0, 0>::zero(),
                    x: 1.0,
                    y: 0.0,
                    other_component_value: 1.0,
                },
            ],
        };

        let result = NullclineResult {
            x_nullcline: vec![segment1],
            y_nullcline: vec![],
            intersections: vec![],
            config: NullclineConfig::default(),
        };

        assert_eq!(result.num_x_nullcline_points(), 2);
        assert_eq!(result.num_y_nullcline_points(), 0);

        let coords = result.x_nullcline_coords();
        assert_eq!(coords.len(), 2);
    }
}
