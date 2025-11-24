//! Curvature tensors for Riemannian manifolds
//!
//! This module implements various curvature tensors that measure how
//! a manifold deviates from flat Euclidean space:
//!
//! - **Riemann curvature tensor** R^i_jkl: Full curvature information
//! - **Ricci tensor** R_ij: Contraction of Riemann tensor
//! - **Ricci scalar** R: Trace of Ricci tensor
//! - **Weyl tensor** C^i_jkl: Traceless part of Riemann tensor (coming soon)

use super::Connection;

/// Riemann curvature tensor R^i_jkl
///
/// The Riemann tensor measures the failure of parallel transport to be path-independent.
/// It's defined as:
///
/// ```text
/// R^i_jkl = ∂_k Γ^i_jl - ∂_l Γ^i_jk + Γ^i_mk Γ^m_jl - Γ^i_ml Γ^m_jk
/// ```
///
/// ## Properties
///
/// - Antisymmetric in last two indices: R^i_jkl = -R^i_jlk
/// - First Bianchi identity: R^i_jkl + R^i_klj + R^i_ljk = 0
/// - Second Bianchi identity: ∇_m R^i_jkl + ∇_k R^i_jlm + ∇_l R^i_jmk = 0
///
/// ## Physical Interpretation
///
/// In general relativity, the Riemann tensor encodes tidal forces:
/// parallel transporting a vector around a closed loop produces a change
/// proportional to the Riemann tensor.
pub struct RiemannTensor<'a, const DIM: usize> {
    connection: &'a Connection<DIM>,
    h: f64, // Step size for numerical differentiation
}

impl<'a, const DIM: usize> RiemannTensor<'a, DIM> {
    /// Create a new Riemann tensor from a connection
    ///
    /// # Arguments
    ///
    /// * `connection` - The connection (Christoffel symbols)
    ///
    /// # Examples
    ///
    /// ```
    /// use amari_calculus::manifold::{MetricTensor, Connection, RiemannTensor};
    ///
    /// let metric = MetricTensor::<2>::euclidean();
    /// let connection = Connection::from_metric(&metric);
    /// let riemann = RiemannTensor::new(&connection);
    ///
    /// // Euclidean space has zero curvature
    /// let r = riemann.component(0, 0, 0, 0, &[0.0, 0.0]);
    /// assert!(r.abs() < 1e-8);
    /// ```
    pub fn new(connection: &'a Connection<DIM>) -> Self {
        Self {
            connection,
            h: 1e-5,
        }
    }

    /// Compute component R^i_jkl at a point
    ///
    /// Uses the formula:
    /// R^i_jkl = ∂_k Γ^i_jl - ∂_l Γ^i_jk + Γ^i_mk Γ^m_jl - Γ^i_ml Γ^m_jk
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
    pub fn component(&self, i: usize, j: usize, k: usize, l: usize, coords: &[f64]) -> f64 {
        // ∂_k Γ^i_jl
        let d_k_gamma_ijl = self.derivative_christoffel(i, j, l, k, coords);

        // ∂_l Γ^i_jk
        let d_l_gamma_ijk = self.derivative_christoffel(i, j, k, l, coords);

        // Γ^i_mk Γ^m_jl - sum over m
        let mut product_term = 0.0;
        #[allow(clippy::needless_range_loop)]
        for m in 0..DIM {
            let gamma_imk = self.connection.christoffel(i, m, k, coords);
            let gamma_mjl = self.connection.christoffel(m, j, l, coords);
            let gamma_iml = self.connection.christoffel(i, m, l, coords);
            let gamma_mjk = self.connection.christoffel(m, j, k, coords);

            product_term += gamma_imk * gamma_mjl - gamma_iml * gamma_mjk;
        }

        d_k_gamma_ijl - d_l_gamma_ijk + product_term
    }

    /// Compute numerical derivative ∂_k Γ^i_jl
    fn derivative_christoffel(
        &self,
        i: usize,
        j: usize,
        l: usize,
        k: usize,
        coords: &[f64],
    ) -> f64 {
        let mut coords_plus = coords.to_vec();
        let mut coords_minus = coords.to_vec();

        coords_plus[k] += self.h;
        coords_minus[k] -= self.h;

        let gamma_plus = self.connection.christoffel(i, j, l, &coords_plus);
        let gamma_minus = self.connection.christoffel(i, j, l, &coords_minus);

        (gamma_plus - gamma_minus) / (2.0 * self.h)
    }
}

/// Ricci curvature tensor R_ij
///
/// The Ricci tensor is obtained by contracting the Riemann tensor:
///
/// ```text
/// R_ij = R^k_ikj = g^kl R_likj
/// ```
///
/// ## Properties
///
/// - Symmetric: R_ij = R_ji
/// - Trace gives scalar curvature: R = g^ij R_ij
///
/// ## Physical Interpretation
///
/// In general relativity, the Ricci tensor appears in Einstein's field equations
/// and describes how matter curves spacetime.
pub struct RicciTensor<'a, const DIM: usize> {
    riemann: RiemannTensor<'a, DIM>,
}

impl<'a, const DIM: usize> RicciTensor<'a, DIM> {
    /// Create a new Ricci tensor from a connection
    pub fn new(connection: &'a Connection<DIM>) -> Self {
        Self {
            riemann: RiemannTensor::new(connection),
        }
    }

    /// Compute component R_ij at a point
    ///
    /// Uses contraction: R_ij = R^k_ikj
    ///
    /// # Arguments
    ///
    /// * `i` - First index
    /// * `j` - Second index
    /// * `coords` - Point at which to evaluate
    pub fn component(&self, i: usize, j: usize, coords: &[f64]) -> f64 {
        let mut ricci = 0.0;

        // Contract over k: R_ij = R^k_ikj
        #[allow(clippy::needless_range_loop)]
        for k in 0..DIM {
            ricci += self.riemann.component(k, i, k, j, coords);
        }

        ricci
    }
}

/// Scalar curvature R
///
/// The scalar curvature is obtained by tracing the Ricci tensor with the metric:
///
/// ```text
/// R = g^ij R_ij
/// ```
///
/// ## Properties
///
/// - Single number characterizing overall curvature
/// - For 2D surfaces: R = 2K (twice the Gaussian curvature)
/// - For 3-sphere of radius a: R = 6/a²
///
/// ## Physical Interpretation
///
/// - R > 0: Positive curvature (sphere-like)
/// - R = 0: Flat space (Euclidean)
/// - R < 0: Negative curvature (hyperbolic)
pub struct ScalarCurvature<'a, const DIM: usize> {
    ricci: RicciTensor<'a, DIM>,
    connection: &'a Connection<DIM>,
}

impl<'a, const DIM: usize> ScalarCurvature<'a, DIM> {
    /// Create a new scalar curvature calculator
    pub fn new(connection: &'a Connection<DIM>) -> Self {
        Self {
            ricci: RicciTensor::new(connection),
            connection,
        }
    }

    /// Compute scalar curvature R at a point
    ///
    /// Uses: R = g^ij R_ij
    ///
    /// # Arguments
    ///
    /// * `coords` - Point at which to evaluate
    pub fn compute(&self, coords: &[f64]) -> f64 {
        // Get inverse metric
        let g_inv = self.connection.metric().inverse(coords);

        let mut r = 0.0;

        // Trace: R = g^ij R_ij
        #[allow(clippy::needless_range_loop)]
        for i in 0..DIM {
            #[allow(clippy::needless_range_loop)]
            for j in 0..DIM {
                let ricci_ij = self.ricci.component(i, j, coords);
                r += g_inv[i][j] * ricci_ij;
            }
        }

        r
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::manifold::MetricTensor;

    #[test]
    fn test_euclidean_riemann_tensor() {
        // Euclidean space has zero Riemann tensor
        let metric = MetricTensor::<2>::euclidean();
        let connection = Connection::from_metric(&metric);
        let riemann = RiemannTensor::new(&connection);

        // Check all components are zero
        for i in 0..2 {
            for j in 0..2 {
                for k in 0..2 {
                    for l in 0..2 {
                        let r = riemann.component(i, j, k, l, &[1.0, 2.0]);
                        assert!(
                            r.abs() < 1e-6,
                            "Euclidean R^{}_{{{}{}{}}} should be 0, got {}",
                            i,
                            j,
                            k,
                            l,
                            r
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn test_sphere_riemann_tensor() {
        // 2-sphere has non-zero curvature
        let metric = MetricTensor::<2>::sphere(1.0);
        let connection = Connection::from_metric(&metric);
        let riemann = RiemannTensor::new(&connection);

        // At θ=π/4, φ=0
        let theta = std::f64::consts::PI / 4.0;
        let phi = 0.0;

        // R^θ_φθφ should be non-zero (related to Gaussian curvature)
        let r_theta_phi_theta_phi = riemann.component(0, 1, 0, 1, &[theta, phi]);

        // For unit sphere, expect curvature ~ 1
        assert!(
            r_theta_phi_theta_phi.abs() > 0.1,
            "Sphere should have non-zero curvature, got {}",
            r_theta_phi_theta_phi
        );
    }

    #[test]
    fn test_riemann_antisymmetry() {
        // R^i_jkl = -R^i_jlk
        let metric = MetricTensor::<2>::sphere(1.0);
        let connection = Connection::from_metric(&metric);
        let riemann = RiemannTensor::new(&connection);

        let coords = &[std::f64::consts::PI / 3.0, 0.5];

        let r_0101 = riemann.component(0, 1, 0, 1, coords);
        let r_0110 = riemann.component(0, 1, 1, 0, coords);

        assert!(
            (r_0101 + r_0110).abs() < 0.1,
            "Riemann tensor should be antisymmetric in last two indices"
        );
    }

    #[test]
    fn test_euclidean_ricci_tensor() {
        // Euclidean space has zero Ricci tensor
        let metric = MetricTensor::<2>::euclidean();
        let connection = Connection::from_metric(&metric);
        let ricci = RicciTensor::new(&connection);

        for i in 0..2 {
            for j in 0..2 {
                let r_ij = ricci.component(i, j, &[1.0, 2.0]);
                assert!(
                    r_ij.abs() < 1e-6,
                    "Euclidean Ricci tensor R_{{{}{}}} should be 0, got {}",
                    i,
                    j,
                    r_ij
                );
            }
        }
    }

    #[test]
    fn test_ricci_symmetry() {
        // R_ij = R_ji
        let metric = MetricTensor::<2>::sphere(1.0);
        let connection = Connection::from_metric(&metric);
        let ricci = RicciTensor::new(&connection);

        let coords = &[std::f64::consts::PI / 4.0, 0.0];

        let r_01 = ricci.component(0, 1, coords);
        let r_10 = ricci.component(1, 0, coords);

        assert!(
            (r_01 - r_10).abs() < 0.1,
            "Ricci tensor should be symmetric, got R_01={}, R_10={}",
            r_01,
            r_10
        );
    }

    #[test]
    fn test_euclidean_scalar_curvature() {
        // Euclidean space has zero scalar curvature
        let metric = MetricTensor::<2>::euclidean();
        let connection = Connection::from_metric(&metric);
        let scalar_curv = ScalarCurvature::new(&connection);

        let r = scalar_curv.compute(&[1.0, 2.0]);
        assert!(
            r.abs() < 1e-6,
            "Euclidean scalar curvature should be 0, got {}",
            r
        );
    }

    #[test]
    fn test_sphere_scalar_curvature() {
        // Unit 2-sphere should have scalar curvature R = 2
        let metric = MetricTensor::<2>::sphere(1.0);
        let connection = Connection::from_metric(&metric);
        let scalar_curv = ScalarCurvature::new(&connection);

        let r = scalar_curv.compute(&[std::f64::consts::PI / 4.0, 0.0]);

        // For unit sphere: R = 2K = 2(1/a²) = 2
        // Allow for numerical error
        assert!(
            (r - 2.0).abs() < 0.5,
            "Unit sphere scalar curvature should be ~2, got {}",
            r
        );
    }
}
