//! Vector derivative operator ∇ = e^i ∂_i

use crate::{fields::*, CalculusResult, CoordinateSystem};
use amari_core::Multivector;

/// Vector derivative operator ∇
///
/// The fundamental differential operator in geometric calculus that combines
/// gradient, divergence, and curl into a single geometric operation.
///
/// ## Mathematical Background
///
/// The vector derivative is defined as:
/// ```text
/// ∇ = e^i ∂_i  (sum over basis vectors)
/// ```
///
/// When applied to fields:
/// - Scalar field: ∇f = gradient
/// - Vector field: ∇·F = divergence, ∇∧F = curl
/// - General: ∇F = ∇·F + ∇∧F (full geometric derivative)
pub struct VectorDerivative<const P: usize, const Q: usize, const R: usize> {
    /// Coordinate system
    coordinates: CoordinateSystem,
    /// Step size for numerical differentiation
    h: f64,
}

impl<const P: usize, const Q: usize, const R: usize> VectorDerivative<P, Q, R> {
    /// Create new vector derivative operator
    ///
    /// # Arguments
    ///
    /// * `coordinates` - Coordinate system (Cartesian, spherical, etc.)
    ///
    /// # Examples
    ///
    /// ```
    /// use amari_calculus::{VectorDerivative, CoordinateSystem};
    ///
    /// let nabla = VectorDerivative::<3, 0, 0>::new(CoordinateSystem::Cartesian);
    /// ```
    pub fn new(coordinates: CoordinateSystem) -> Self {
        Self {
            coordinates,
            h: 1e-5, // Default step size
        }
    }

    /// Set step size for numerical differentiation
    pub fn with_step_size(mut self, h: f64) -> Self {
        self.h = h;
        self
    }

    /// Compute gradient of scalar field: ∇f
    ///
    /// Returns a vector field representing the gradient.
    pub fn gradient(&self, f: &ScalarField<P, Q, R>, coords: &[f64]) -> Multivector<P, Q, R> {
        let dim = P + Q + R;
        let mut components = vec![0.0; dim];

        // Compute partial derivatives along each axis
        for i in 0..dim {
            components[i] = f.partial_derivative(coords, i, self.h).unwrap_or(0.0);
        }

        Multivector::from_vector(&components)
    }

    /// Compute divergence of vector field: ∇·F
    ///
    /// Returns a scalar representing the divergence.
    pub fn divergence(&self, f: &VectorField<P, Q, R>, coords: &[f64]) -> f64 {
        let dim = P + Q + R;
        let mut div = 0.0;

        // Sum of partial derivatives: ∂F_i/∂x_i
        for i in 0..dim {
            if let Ok(df_dxi) = f.partial_derivative(coords, i, self.h) {
                // Extract i-th component of the derivative
                div += df_dxi.get_component(1 << i);
            }
        }

        div
    }

    /// Compute curl of vector field: ∇∧F
    ///
    /// Returns a bivector representing the curl.
    pub fn curl(&self, f: &VectorField<P, Q, R>, coords: &[f64]) -> Multivector<P, Q, R> {
        // For 3D: curl = (∂F_z/∂y - ∂F_y/∂z, ∂F_x/∂z - ∂F_z/∂x, ∂F_y/∂x - ∂F_x/∂y)
        // Represented as bivector (e_2∧e_3, e_3∧e_1, e_1∧e_2)

        let dim = P + Q + R;
        if dim != 3 {
            // Curl only well-defined in 3D for now
            return Multivector::zero();
        }

        // Compute partial derivatives
        let df_dx = f
            .partial_derivative(coords, 0, self.h)
            .unwrap_or(Multivector::zero());
        let df_dy = f
            .partial_derivative(coords, 1, self.h)
            .unwrap_or(Multivector::zero());
        let df_dz = f
            .partial_derivative(coords, 2, self.h)
            .unwrap_or(Multivector::zero());

        // Extract components (basis: e_1, e_2, e_3 → indices 1, 2, 4)
        let fx_x = df_dx.get_component(1); // ∂F_x/∂x
        let fy_x = df_dx.get_component(2); // ∂F_y/∂x
        let fz_x = df_dx.get_component(4); // ∂F_z/∂x

        let fx_y = df_dy.get_component(1); // ∂F_x/∂y
        let fy_y = df_dy.get_component(2); // ∂F_y/∂y
        let fz_y = df_dy.get_component(4); // ∂F_z/∂y

        let fx_z = df_dz.get_component(1); // ∂F_x/∂z
        let fy_z = df_dz.get_component(2); // ∂F_y/∂z
        let fz_z = df_dz.get_component(4); // ∂F_z/∂z

        // Curl components (bivector representation)
        // e_2∧e_3: ∂F_z/∂y - ∂F_y/∂z
        // e_3∧e_1: ∂F_x/∂z - ∂F_z/∂x
        // e_1∧e_2: ∂F_y/∂x - ∂F_x/∂y

        let mut curl = Multivector::zero();
        curl.set_component(2 | 4, fz_y - fy_z); // e_2∧e_3
        curl.set_component(4 | 1, fx_z - fz_x); // e_3∧e_1
        curl.set_component(1 | 2, fy_x - fx_y); // e_1∧e_2

        curl
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gradient() {
        // f(x, y) = x² + y²
        // ∇f = (2x, 2y, 0)
        let f = ScalarField::<3, 0, 0>::new(|coords| coords[0].powi(2) + coords[1].powi(2));

        let nabla = VectorDerivative::<3, 0, 0>::new(CoordinateSystem::Cartesian);
        let grad = nabla.gradient(&f, &[3.0, 4.0, 0.0]);

        // Check components
        assert!(
            (grad.get_component(1) - 6.0).abs() < 1e-4,
            "∂f/∂x should be 6.0"
        );
        assert!(
            (grad.get_component(2) - 8.0).abs() < 1e-4,
            "∂f/∂y should be 8.0"
        );
    }

    #[test]
    fn test_divergence() {
        // F(x, y, z) = (x, y, z)
        // ∇·F = 1 + 1 + 1 = 3
        let f = VectorField::<3, 0, 0>::new(|coords| {
            Multivector::from_vector(&[coords[0], coords[1], coords[2]])
        });

        let nabla = VectorDerivative::<3, 0, 0>::new(CoordinateSystem::Cartesian);
        let div = nabla.divergence(&f, &[1.0, 1.0, 1.0]);

        assert!(
            (div - 3.0).abs() < 1e-4,
            "Divergence should be 3.0, got {}",
            div
        );
    }

    #[test]
    fn test_curl_of_rotation_field() {
        // F(x, y, z) = (-y, x, 0) - rotation around z-axis
        // ∇×F = (0, 0, 2) in traditional notation
        // In GA: bivector e_1∧e_2 with magnitude 2
        let f = VectorField::<3, 0, 0>::new(|coords| {
            Multivector::from_vector(&[-coords[1], coords[0], 0.0])
        });

        let nabla = VectorDerivative::<3, 0, 0>::new(CoordinateSystem::Cartesian);
        let curl = nabla.curl(&f, &[0.0, 0.0, 0.0]);

        // Check e_1∧e_2 component (index 1|2 = 3)
        let curl_z = curl.get_component(1 | 2);
        assert!(
            (curl_z - 2.0).abs() < 1e-4,
            "Curl z-component should be 2.0, got {}",
            curl_z
        );
    }
}
