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
pub mod metric;

pub use connection::Connection;
pub use covariant::CovariantDerivative;
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
}
