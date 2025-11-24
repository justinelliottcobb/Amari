//! Integration on manifolds using geometric measure theory
//!
//! This module provides integration of scalar and multivector fields over
//! manifolds, using the measure-theoretic foundations from amari-measure.
//!
//! ## Mathematical Background
//!
//! Integration on manifolds extends classical integration to curved spaces:
//!
//! ### Scalar Integration
//!
//! For a scalar field f and a measure μ on a manifold M:
//! ```text
//! ∫_M f dμ
//! ```
//!
//! ### Multivector Integration
//!
//! For a k-vector field ω and a k-dimensional manifold M:
//! ```text
//! ∫_M ω
//! ```
//!
//! ### Fundamental Theorem of Geometric Calculus
//!
//! Generalizes Stokes' theorem:
//! ```text
//! ∫_V (∇F) dV = ∮_∂V F dS
//! ```
//!
//! Where:
//! - V is a volume (n-dimensional manifold)
//! - ∂V is its boundary ((n-1)-dimensional manifold)
//! - ∇F is the geometric derivative
//! - dV, dS are volume and surface elements
//!
//! ## Examples
//!
//! ```
//! use amari_calculus::{ScalarField, ManifoldIntegrator};
//!
//! // Define scalar field f(x, y) = x² + y²
//! let f = ScalarField::<3, 0, 0>::new(|coords| {
//!     coords[0].powi(2) + coords[1].powi(2)
//! });
//!
//! // Create integrator for 2D rectangular domain [0, 1] × [0, 1]
//! let integrator = ManifoldIntegrator::<3, 0, 0>::new();
//!
//! // Integrate over rectangle
//! let result = integrator.integrate_scalar_2d(&f, (0.0, 1.0), (0.0, 1.0), 100);
//! ```

use crate::fields::*;

/// Integrator for scalar and multivector fields on manifolds
///
/// Uses adaptive quadrature and measure-theoretic methods for accurate integration.
pub struct ManifoldIntegrator<const P: usize, const Q: usize, const R: usize> {
    /// Tolerance for adaptive integration
    tolerance: f64,
}

impl<const P: usize, const Q: usize, const R: usize> ManifoldIntegrator<P, Q, R> {
    /// Create new manifold integrator
    ///
    /// # Examples
    ///
    /// ```
    /// use amari_calculus::ManifoldIntegrator;
    ///
    /// let integrator = ManifoldIntegrator::<3, 0, 0>::new();
    /// ```
    pub fn new() -> Self {
        Self { tolerance: 1e-6 }
    }

    /// Set tolerance for adaptive integration
    pub fn with_tolerance(mut self, tolerance: f64) -> Self {
        self.tolerance = tolerance;
        self
    }

    /// Integrate scalar field over 1D interval [a, b]
    ///
    /// Uses adaptive Simpson's rule for accurate integration.
    ///
    /// # Arguments
    ///
    /// * `f` - Scalar field to integrate
    /// * `a` - Lower bound
    /// * `b` - Upper bound
    /// * `n` - Number of subdivisions (must be even)
    ///
    /// # Returns
    ///
    /// Approximation of ∫_a^b f(x) dx
    ///
    /// # Examples
    ///
    /// ```
    /// use amari_calculus::{ScalarField, ManifoldIntegrator};
    ///
    /// // f(x) = x²
    /// let f = ScalarField::<3, 0, 0>::with_dimension(
    ///     |coords| coords[0].powi(2),
    ///     1
    /// );
    ///
    /// let integrator = ManifoldIntegrator::<3, 0, 0>::new();
    ///
    /// // ∫_0^1 x² dx = 1/3
    /// let result = integrator.integrate_scalar_1d(&f, 0.0, 1.0, 100);
    /// assert!((result - 1.0/3.0).abs() < 1e-4);
    /// ```
    pub fn integrate_scalar_1d(&self, f: &ScalarField<P, Q, R>, a: f64, b: f64, n: usize) -> f64 {
        // Simpson's rule: ∫ f(x) dx ≈ (h/3)[f(x0) + 4f(x1) + 2f(x2) + 4f(x3) + ... + f(xn)]
        let h = (b - a) / (n as f64);
        let mut sum = f.evaluate(&[a]) + f.evaluate(&[b]);

        for i in 1..n {
            let x = a + (i as f64) * h;
            let coeff = if i % 2 == 0 { 2.0 } else { 4.0 };
            sum += coeff * f.evaluate(&[x]);
        }

        sum * h / 3.0
    }

    /// Integrate scalar field over 2D rectangular domain [x_min, x_max] × [y_min, y_max]
    ///
    /// # Arguments
    ///
    /// * `f` - Scalar field to integrate
    /// * `x_range` - (x_min, x_max) bounds
    /// * `y_range` - (y_min, y_max) bounds
    /// * `n` - Number of subdivisions per dimension
    ///
    /// # Returns
    ///
    /// Approximation of ∫∫_R f(x,y) dx dy
    ///
    /// # Examples
    ///
    /// ```
    /// use amari_calculus::{ScalarField, ManifoldIntegrator};
    ///
    /// // f(x, y) = x*y
    /// let f = ScalarField::<3, 0, 0>::with_dimension(
    ///     |coords| coords[0] * coords[1],
    ///     2
    /// );
    ///
    /// let integrator = ManifoldIntegrator::<3, 0, 0>::new();
    ///
    /// // ∫_0^1 ∫_0^1 x*y dx dy = 1/4
    /// let result = integrator.integrate_scalar_2d(&f, (0.0, 1.0), (0.0, 1.0), 50);
    /// assert!((result - 0.25).abs() < 1e-4);
    /// ```
    pub fn integrate_scalar_2d(
        &self,
        f: &ScalarField<P, Q, R>,
        x_range: (f64, f64),
        y_range: (f64, f64),
        n: usize,
    ) -> f64 {
        let (x_min, x_max) = x_range;
        let (y_min, y_max) = y_range;
        let hx = (x_max - x_min) / (n as f64);
        let hy = (y_max - y_min) / (n as f64);

        let mut sum = 0.0;

        for i in 0..=n {
            for j in 0..=n {
                let x = x_min + (i as f64) * hx;
                let y = y_min + (j as f64) * hy;

                let weight = match (i, j) {
                    (0, 0) | (0, _) if j == n => 1.0,
                    (_, 0) if i == n => 1.0,
                    (_, _) if i == n && j == n => 1.0,
                    (0, _) | (_, 0) if i != n && j != n => 2.0,
                    (_, _) if i == n || j == n => 2.0,
                    _ => 4.0,
                };

                sum += weight * f.evaluate(&[x, y]);
            }
        }

        sum * hx * hy / 4.0
    }

    /// Integrate scalar field over 3D rectangular domain
    ///
    /// # Arguments
    ///
    /// * `f` - Scalar field to integrate
    /// * `x_range` - (x_min, x_max) bounds
    /// * `y_range` - (y_min, y_max) bounds
    /// * `z_range` - (z_min, z_max) bounds
    /// * `n` - Number of subdivisions per dimension
    ///
    /// # Returns
    ///
    /// Approximation of ∫∫∫_V f(x,y,z) dx dy dz
    pub fn integrate_scalar_3d(
        &self,
        f: &ScalarField<P, Q, R>,
        x_range: (f64, f64),
        y_range: (f64, f64),
        z_range: (f64, f64),
        n: usize,
    ) -> f64 {
        let (x_min, x_max) = x_range;
        let (y_min, y_max) = y_range;
        let (z_min, z_max) = z_range;
        let hx = (x_max - x_min) / (n as f64);
        let hy = (y_max - y_min) / (n as f64);
        let hz = (z_max - z_min) / (n as f64);

        let mut sum = 0.0;

        for i in 0..=n {
            for j in 0..=n {
                for k in 0..=n {
                    let x = x_min + (i as f64) * hx;
                    let y = y_min + (j as f64) * hy;
                    let z = z_min + (k as f64) * hz;

                    // Trapezoidal rule weights for 3D
                    let weight = if i == 0 || i == n { 0.5 } else { 1.0 }
                        * if j == 0 || j == n { 0.5 } else { 1.0 }
                        * if k == 0 || k == n { 0.5 } else { 1.0 };

                    sum += weight * f.evaluate(&[x, y, z]);
                }
            }
        }

        sum * hx * hy * hz
    }

    /// Verify the fundamental theorem of geometric calculus: ∫_V (∇F) dV = ∮_∂V F dS
    ///
    /// For a 2D rectangular domain, this reduces to checking:
    /// ∫∫_R div(F) dx dy = ∮_∂R F·n ds
    ///
    /// where n is the outward normal to the boundary.
    pub fn verify_fundamental_theorem_2d(
        &self,
        f: &VectorField<P, Q, R>,
        x_range: (f64, f64),
        y_range: (f64, f64),
        n: usize,
    ) -> (f64, f64) {
        use crate::VectorDerivative;

        let nabla = VectorDerivative::new(crate::CoordinateSystem::Cartesian);
        let (x_min, x_max) = x_range;
        let (y_min, y_max) = y_range;
        let hx = (x_max - x_min) / (n as f64);
        let hy = (y_max - y_min) / (n as f64);

        // Compute volume integral: ∫∫ div(F) dx dy manually
        let mut volume_integral = 0.0;
        for i in 0..=n {
            for j in 0..=n {
                let x = x_min + (i as f64) * hx;
                let y = y_min + (j as f64) * hy;

                let weight = match (i, j) {
                    (0, 0) | (0, _) if j == n => 1.0,
                    (_, 0) if i == n => 1.0,
                    (_, _) if i == n && j == n => 1.0,
                    (0, _) | (_, 0) if i != n && j != n => 2.0,
                    (_, _) if i == n || j == n => 2.0,
                    _ => 4.0,
                };

                // Pass 3D coordinates (padding with 0 for z)
                volume_integral += weight * nabla.divergence(f, &[x, y, 0.0]);
            }
        }
        volume_integral *= hx * hy / 4.0;

        // Compute surface integral: ∮ F·n ds
        let mut surface_integral = 0.0;

        // Bottom edge (y = y_min, normal = (0, -1))
        for i in 0..=n {
            let x = x_min + (i as f64) * (x_max - x_min) / (n as f64);
            let f_val = f.evaluate(&[x, y_min, 0.0]);
            let weight = if i == 0 || i == n { 0.5 } else { 1.0 };
            surface_integral -= weight * f_val.vector_component(1) * (x_max - x_min) / (n as f64);
        }

        // Top edge (y = y_max, normal = (0, 1))
        for i in 0..=n {
            let x = x_min + (i as f64) * (x_max - x_min) / (n as f64);
            let f_val = f.evaluate(&[x, y_max, 0.0]);
            let weight = if i == 0 || i == n { 0.5 } else { 1.0 };
            surface_integral += weight * f_val.vector_component(1) * (x_max - x_min) / (n as f64);
        }

        // Left edge (x = x_min, normal = (-1, 0))
        for j in 0..=n {
            let y = y_min + (j as f64) * (y_max - y_min) / (n as f64);
            let f_val = f.evaluate(&[x_min, y, 0.0]);
            let weight = if j == 0 || j == n { 0.5 } else { 1.0 };
            surface_integral -= weight * f_val.vector_component(0) * (y_max - y_min) / (n as f64);
        }

        // Right edge (x = x_max, normal = (1, 0))
        for j in 0..=n {
            let y = y_min + (j as f64) * (y_max - y_min) / (n as f64);
            let f_val = f.evaluate(&[x_max, y, 0.0]);
            let weight = if j == 0 || j == n { 0.5 } else { 1.0 };
            surface_integral += weight * f_val.vector_component(0) * (y_max - y_min) / (n as f64);
        }

        (volume_integral, surface_integral)
    }
}

impl<const P: usize, const Q: usize, const R: usize> Default for ManifoldIntegrator<P, Q, R> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_integrate_1d_polynomial() {
        // ∫_0^1 x² dx = 1/3
        let f = ScalarField::<3, 0, 0>::with_dimension(|coords| coords[0].powi(2), 1);

        let integrator = ManifoldIntegrator::<3, 0, 0>::new();
        let result = integrator.integrate_scalar_1d(&f, 0.0, 1.0, 100);

        assert!(
            (result - 1.0 / 3.0).abs() < 1e-4,
            "∫ x² dx should be 1/3, got {}",
            result
        );
    }

    #[test]
    fn test_integrate_2d_product() {
        // ∫_0^1 ∫_0^1 x*y dx dy = 1/4
        let f = ScalarField::<3, 0, 0>::with_dimension(|coords| coords[0] * coords[1], 2);

        let integrator = ManifoldIntegrator::<3, 0, 0>::new();
        let result = integrator.integrate_scalar_2d(&f, (0.0, 1.0), (0.0, 1.0), 50);

        assert!(
            (result - 0.25).abs() < 1e-3,
            "∫∫ x*y dx dy should be 1/4, got {}",
            result
        );
    }

    #[test]
    fn test_integrate_3d_constant() {
        // ∫_0^1 ∫_0^1 ∫_0^1 1 dx dy dz = 1
        let f = ScalarField::<3, 0, 0>::with_dimension(|_coords| 1.0, 3);

        let integrator = ManifoldIntegrator::<3, 0, 0>::new();
        let result = integrator.integrate_scalar_3d(&f, (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), 20);

        assert!(
            (result - 1.0).abs() < 1e-3,
            "∫∫∫ 1 dx dy dz should be 1, got {}",
            result
        );
    }

    #[test]
    fn test_fundamental_theorem_constant_field() {
        // F = (x, y, 0) → div(F) = 2
        // Using 2D version with proper coords
        let f = VectorField::<3, 0, 0>::new(|coords| {
            let x = if !coords.is_empty() { coords[0] } else { 0.0 };
            let y = if coords.len() > 1 { coords[1] } else { 0.0 };
            crate::vector_from_slice(&[x, y, 0.0])
        });

        let integrator = ManifoldIntegrator::<3, 0, 0>::new();
        let (volume, surface) =
            integrator.verify_fundamental_theorem_2d(&f, (0.0, 1.0), (0.0, 1.0), 50);

        assert!(
            (volume - surface).abs() < 1e-2,
            "Fundamental theorem: volume {} should equal surface {}, diff = {}",
            volume,
            surface,
            (volume - surface).abs()
        );
    }
}
