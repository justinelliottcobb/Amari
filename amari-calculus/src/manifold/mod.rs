//! Manifold calculus - covariant derivatives and connections
//!
//! This module provides tools for calculus on curved manifolds, including:
//! - Metric tensors and Riemannian geometry
//! - Christoffel symbols and connections
//! - Covariant derivatives
//! - Parallel transport and geodesics
//!
//! ## Mathematical Background
//!
//! ### Metric Tensor
//!
//! The metric tensor g_ij defines the geometry of a manifold:
//! ```text
//! ds² = g_ij dx^i dx^j
//! ```
//!
//! ### Christoffel Symbols
//!
//! The Christoffel symbols Γ^k_ij encode how basis vectors change:
//! ```text
//! Γ^k_ij = (1/2) g^kl (∂_i g_jl + ∂_j g_il - ∂_l g_ij)
//! ```
//!
//! ### Covariant Derivative
//!
//! The covariant derivative ∇_i generalizes the partial derivative:
//! ```text
//! ∇_i V^j = ∂_i V^j + Γ^j_ik V^k
//! ```
//!
//! ## Examples
//!
//! ```
//! use amari_calculus::manifold::{MetricTensor, RiemannianManifold};
//!
//! // Define 2D Euclidean metric (flat space)
//! let metric = MetricTensor::<2>::euclidean();
//!
//! // Create manifold
//! let manifold = RiemannianManifold::new(metric);
//!
//! // Christoffel symbols are zero for flat space
//! let gamma = manifold.christoffel_symbol(0, 0, 0, &[0.0, 0.0]);
//! assert!(gamma.abs() < 1e-10);
//! ```

use crate::{CalculusResult, ScalarField};

pub mod connection;
pub mod covariant;
pub mod curvature;
pub mod geodesic;
pub mod metric;

pub use connection::Connection;
pub use covariant::CovariantDerivative;
pub use curvature::{RicciTensor, RiemannTensor, ScalarCurvature};
pub use geodesic::{GeodesicSolver, ParallelTransport};
pub use metric::MetricTensor;

/// A Riemannian manifold with metric tensor and connection
///
/// Provides the infrastructure for differential geometry on curved spaces.
pub struct RiemannianManifold<const DIM: usize> {
    /// Metric tensor g_ij
    metric: MetricTensor<DIM>,
    /// Connection (Christoffel symbols)
    connection: Connection<DIM>,
}

impl<const DIM: usize> RiemannianManifold<DIM> {
    /// Create a new Riemannian manifold from a metric tensor
    ///
    /// # Arguments
    ///
    /// * `metric` - The metric tensor g_ij defining the geometry
    ///
    /// # Examples
    ///
    /// ```
    /// use amari_calculus::manifold::{MetricTensor, RiemannianManifold};
    ///
    /// let metric = MetricTensor::<2>::euclidean();
    /// let manifold = RiemannianManifold::new(metric);
    /// ```
    pub fn new(metric: MetricTensor<DIM>) -> Self {
        let connection = Connection::from_metric(&metric);
        Self { metric, connection }
    }

    /// Get the metric tensor
    pub fn metric(&self) -> &MetricTensor<DIM> {
        &self.metric
    }

    /// Get the connection
    pub fn connection(&self) -> &Connection<DIM> {
        &self.connection
    }

    /// Compute Christoffel symbol Γ^k_ij at a point
    ///
    /// # Arguments
    ///
    /// * `k` - Upper index
    /// * `i` - First lower index
    /// * `j` - Second lower index
    /// * `coords` - Point at which to evaluate
    ///
    /// # Returns
    ///
    /// The value of Γ^k_ij at the given point
    pub fn christoffel_symbol(&self, k: usize, i: usize, j: usize, coords: &[f64]) -> f64 {
        self.connection.christoffel(k, i, j, coords)
    }

    /// Compute covariant derivative of a vector field
    ///
    /// For a vector field V^j, computes ∇_i V^j = ∂_i V^j + Γ^j_ik V^k
    pub fn covariant_derivative_vector(
        &self,
        vector_field: &[ScalarField<3, 0, 0>],
        direction: usize,
        coords: &[f64],
    ) -> CalculusResult<Vec<f64>> {
        CovariantDerivative::vector(&self.connection, vector_field, direction, coords)
    }

    /// Compute Riemann curvature tensor component R^i_jkl at a point
    ///
    /// The Riemann tensor measures how the manifold deviates from flat space.
    ///
    /// # Arguments
    ///
    /// * `i` - Upper index
    /// * `j` - First lower index
    /// * `k` - Second lower index
    /// * `l` - Third lower index
    /// * `coords` - Point at which to evaluate
    ///
    /// # Returns
    ///
    /// The value of R^i_jkl at the given point
    pub fn riemann_tensor(&self, i: usize, j: usize, k: usize, l: usize, coords: &[f64]) -> f64 {
        let riemann = RiemannTensor::new(&self.connection);
        riemann.component(i, j, k, l, coords)
    }

    /// Compute Ricci curvature tensor component R_ij at a point
    ///
    /// The Ricci tensor is a contraction of the Riemann tensor.
    ///
    /// # Arguments
    ///
    /// * `i` - First index
    /// * `j` - Second index
    /// * `coords` - Point at which to evaluate
    ///
    /// # Returns
    ///
    /// The value of R_ij at the given point
    pub fn ricci_tensor(&self, i: usize, j: usize, coords: &[f64]) -> f64 {
        let ricci = RicciTensor::new(&self.connection);
        ricci.component(i, j, coords)
    }

    /// Compute scalar curvature R at a point
    ///
    /// The scalar curvature is a single number characterizing the overall
    /// curvature of the manifold:
    /// - R > 0: Positive curvature (sphere-like)
    /// - R = 0: Flat space
    /// - R < 0: Negative curvature (hyperbolic)
    ///
    /// # Arguments
    ///
    /// * `coords` - Point at which to evaluate
    ///
    /// # Returns
    ///
    /// The scalar curvature R at the given point
    pub fn scalar_curvature(&self, coords: &[f64]) -> f64 {
        let scalar_curv = ScalarCurvature::new(&self.connection);
        scalar_curv.compute(coords)
    }

    /// Compute geodesic trajectory from initial conditions
    ///
    /// Geodesics are curves that locally minimize distance, generalizing
    /// straight lines to curved spaces.
    ///
    /// # Arguments
    ///
    /// * `initial_pos` - Starting position
    /// * `initial_vel` - Starting velocity (tangent vector)
    /// * `t_max` - Total integration time
    /// * `dt` - Time step for numerical integration
    ///
    /// # Returns
    ///
    /// Vector of (position, velocity) pairs along the geodesic
    pub fn geodesic(
        &self,
        initial_pos: &[f64],
        initial_vel: &[f64],
        t_max: f64,
        dt: f64,
    ) -> Vec<(Vec<f64>, Vec<f64>)> {
        let solver = GeodesicSolver::new(&self.connection);
        solver.trajectory(initial_pos, initial_vel, t_max, dt)
    }

    /// Parallel transport a vector along a curve
    ///
    /// Moves a vector along a curve while keeping it "as constant as possible"
    /// relative to the manifold's connection.
    ///
    /// # Arguments
    ///
    /// * `vector` - Vector to transport
    /// * `curve_pos` - Position on curve
    /// * `curve_tangent` - Tangent to curve
    /// * `dt` - Time step
    ///
    /// # Returns
    ///
    /// Transported vector
    pub fn parallel_transport(
        &self,
        vector: &[f64],
        curve_pos: &[f64],
        curve_tangent: &[f64],
        dt: f64,
    ) -> Vec<f64> {
        let transport = ParallelTransport::new(&self.connection);
        transport.step(curve_pos, curve_tangent, vector, dt)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_euclidean_manifold() {
        // Euclidean space has zero Christoffel symbols
        let metric = MetricTensor::<2>::euclidean();
        let manifold = RiemannianManifold::new(metric);

        // Check all Christoffel symbols are zero
        for k in 0..2 {
            for i in 0..2 {
                for j in 0..2 {
                    let gamma = manifold.christoffel_symbol(k, i, j, &[1.0, 2.0]);
                    assert!(
                        gamma.abs() < 1e-10,
                        "Euclidean Christoffel symbol Γ^{}_{}{} should be 0, got {}",
                        k,
                        i,
                        j,
                        gamma
                    );
                }
            }
        }
    }

    #[test]
    fn test_spherical_manifold() {
        // Sphere has non-zero Christoffel symbols
        let metric = MetricTensor::<2>::sphere(1.0);
        let manifold = RiemannianManifold::new(metric);

        // At θ=π/4, φ=0, some Christoffel symbols should be non-zero
        let theta = std::f64::consts::PI / 4.0;
        let phi = 0.0;

        // Γ^θ_φφ = -sin(θ)cos(θ) should be non-zero
        let gamma_theta_phi_phi = manifold.christoffel_symbol(0, 1, 1, &[theta, phi]);

        // For θ=π/4: -sin(π/4)cos(π/4) = -(√2/2)(√2/2) = -1/2
        assert!(
            (gamma_theta_phi_phi + 0.5).abs() < 0.1,
            "Expected Γ^θ_φφ ≈ -0.5, got {}",
            gamma_theta_phi_phi
        );
    }

    #[test]
    fn test_hyperbolic_manifold() {
        // Hyperbolic space has negative curvature
        let metric = MetricTensor::<2>::hyperbolic();
        let manifold = RiemannianManifold::new(metric);

        // Test point inside Poincaré disk
        let coords = &[0.3, 0.2];

        // Christoffel symbols should be non-zero for hyperbolic geometry
        let gamma_000 = manifold.christoffel_symbol(0, 0, 0, coords);
        assert!(
            gamma_000.abs() > 1e-6,
            "Hyperbolic geometry should have non-zero Christoffel symbols"
        );

        // Scalar curvature should be negative for hyperbolic space
        let r = manifold.scalar_curvature(coords);
        assert!(
            r < 0.0,
            "Hyperbolic space should have negative curvature, got {}",
            r
        );
    }

    #[test]
    fn test_euclidean_3d() {
        // Test 3D Euclidean metric
        let metric = MetricTensor::<3>::euclidean();
        let manifold = RiemannianManifold::new(metric);

        // All Christoffel symbols should be zero
        for k in 0..3 {
            for i in 0..3 {
                for j in 0..3 {
                    let gamma = manifold.christoffel_symbol(k, i, j, &[1.0, 2.0, 3.0]);
                    assert!(
                        gamma.abs() < 1e-10,
                        "3D Euclidean Γ^{}_{}{} should be 0",
                        k,
                        i,
                        j
                    );
                }
            }
        }

        // Scalar curvature should be zero
        let r = manifold.scalar_curvature(&[1.0, 2.0, 3.0]);
        assert!(r.abs() < 1e-8, "3D Euclidean scalar curvature should be 0");
    }

    #[test]
    fn test_manifold_curvature_methods() {
        // Test high-level curvature methods on manifold
        let metric = MetricTensor::<2>::sphere(1.0);
        let manifold = RiemannianManifold::new(metric);

        let coords = &[std::f64::consts::PI / 4.0, 0.0];

        // Test Riemann tensor method
        let riemann = manifold.riemann_tensor(0, 1, 0, 1, coords);
        assert!(
            riemann.abs() > 0.1,
            "Sphere should have non-zero Riemann tensor"
        );

        // Test Ricci tensor method
        let ricci_00 = manifold.ricci_tensor(0, 0, coords);
        let ricci_11 = manifold.ricci_tensor(1, 1, coords);
        assert!(
            ricci_00.abs() > 0.1 || ricci_11.abs() > 0.1,
            "Sphere should have non-zero Ricci tensor"
        );

        // Test scalar curvature method
        let scalar = manifold.scalar_curvature(coords);
        assert!(
            (scalar - 2.0).abs() < 0.5,
            "Unit sphere scalar curvature should be ~2, got {}",
            scalar
        );
    }

    #[test]
    fn test_manifold_geodesic_method() {
        // Test high-level geodesic method on manifold
        let metric = MetricTensor::<2>::euclidean();
        let manifold = RiemannianManifold::new(metric);

        let initial_pos = vec![0.0, 0.0];
        let initial_vel = vec![1.0, 0.5];

        let trajectory = manifold.geodesic(&initial_pos, &initial_vel, 1.0, 0.1);

        assert!(
            trajectory.len() > 1,
            "Geodesic should produce multiple points"
        );

        // In Euclidean space, geodesic should be straight line
        let (final_pos, _) = &trajectory[trajectory.len() - 1];
        assert!(
            (final_pos[0] - 1.0).abs() < 0.05,
            "Euclidean geodesic x should be ~1.0"
        );
        assert!(
            (final_pos[1] - 0.5).abs() < 0.05,
            "Euclidean geodesic y should be ~0.5"
        );
    }

    #[test]
    fn test_manifold_parallel_transport_method() {
        // Test high-level parallel transport method
        let metric = MetricTensor::<2>::euclidean();
        let manifold = RiemannianManifold::new(metric);

        let vector = vec![1.0, 0.0];
        let pos = vec![0.0, 0.0];
        let tangent = vec![1.0, 0.0];

        let transported = manifold.parallel_transport(&vector, &pos, &tangent, 0.1);

        // In Euclidean space, vector should not change
        assert!(
            (transported[0] - 1.0).abs() < 1e-6,
            "Euclidean parallel transport should preserve vector"
        );
        assert!(transported[1].abs() < 1e-6);
    }
}
