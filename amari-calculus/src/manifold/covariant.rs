//! Covariant derivative implementation

use super::Connection;
use crate::{CalculusError, CalculusResult, ScalarField};

/// Covariant derivative operator
///
/// The covariant derivative ∇_i generalizes the partial derivative to curved spaces:
/// ∇_i V^j = ∂_i V^j + Γ^j_ik V^k
pub struct CovariantDerivative;

impl CovariantDerivative {
    /// Compute covariant derivative of a vector field
    ///
    /// For a vector field V^j represented as an array of scalar fields,
    /// computes ∇_i V^j = ∂_i V^j + Γ^j_ik V^k
    ///
    /// # Arguments
    ///
    /// * `connection` - The connection (Christoffel symbols)
    /// * `vector_field` - Array of scalar fields representing V^j(x)
    /// * `direction` - Index i for ∇_i
    /// * `coords` - Point at which to evaluate
    ///
    /// # Returns
    ///
    /// Vector representing ∇_i V^j for all j
    pub fn vector<const DIM: usize>(
        connection: &Connection<DIM>,
        vector_field: &[ScalarField<3, 0, 0>],
        direction: usize,
        coords: &[f64],
    ) -> CalculusResult<Vec<f64>>
    where
        [(); DIM]:,
    {
        if vector_field.len() != DIM {
            return Err(CalculusError::InvalidDimension {
                expected: DIM,
                got: vector_field.len(),
            });
        }

        let h = 1e-5;
        let mut result = vec![0.0; DIM];

        for j in 0..DIM {
            // Compute partial derivative ∂_i V^j
            let partial = vector_field[j]
                .partial_derivative(coords, direction, h)
                .unwrap_or(0.0);

            // Compute connection term: Γ^j_ik V^k
            let mut connection_term = 0.0;
            #[allow(clippy::needless_range_loop)]
            for k in 0..DIM {
                let gamma = connection.christoffel(j, direction, k, coords);
                let v_k = vector_field[k].evaluate(coords);
                connection_term += gamma * v_k;
            }

            result[j] = partial + connection_term;
        }

        Ok(result)
    }

    /// Compute covariant derivative of a scalar field
    ///
    /// For scalar fields, the covariant derivative reduces to the partial derivative:
    /// ∇_i f = ∂_i f
    pub fn scalar(
        field: &ScalarField<3, 0, 0>,
        direction: usize,
        coords: &[f64],
    ) -> CalculusResult<f64> {
        let h = 1e-5;
        field.partial_derivative(coords, direction, h)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::manifold::MetricTensor;

    #[test]
    fn test_covariant_derivative_euclidean() {
        // In flat space, covariant derivative = partial derivative
        let metric = MetricTensor::<2>::euclidean();
        let connection = Connection::from_metric(&metric);

        // Define constant vector field V = (1, 2)
        let vector_field = [
            ScalarField::with_dimension(|_| 1.0, 2),
            ScalarField::with_dimension(|_| 2.0, 2),
        ];

        // Covariant derivative of constant field should be zero
        let nabla_0 =
            CovariantDerivative::vector(&connection, &vector_field, 0, &[1.0, 2.0]).unwrap();
        let nabla_1 =
            CovariantDerivative::vector(&connection, &vector_field, 1, &[1.0, 2.0]).unwrap();

        for j in 0..2 {
            assert!(
                nabla_0[j].abs() < 1e-6,
                "∇_0 V^{} should be 0, got {}",
                j,
                nabla_0[j]
            );
            assert!(
                nabla_1[j].abs() < 1e-6,
                "∇_1 V^{} should be 0, got {}",
                j,
                nabla_1[j]
            );
        }
    }

    #[test]
    fn test_covariant_derivative_linear_field() {
        // In flat space, for V = (x, y), ∇_x V = (1, 0), ∇_y V = (0, 1)
        let metric = MetricTensor::<2>::euclidean();
        let connection = Connection::from_metric(&metric);

        let vector_field = [
            ScalarField::with_dimension(|coords| coords[0], 2),
            ScalarField::with_dimension(|coords| coords[1], 2),
        ];

        let nabla_0 =
            CovariantDerivative::vector(&connection, &vector_field, 0, &[1.0, 2.0]).unwrap();
        let nabla_1 =
            CovariantDerivative::vector(&connection, &vector_field, 1, &[1.0, 2.0]).unwrap();

        // ∇_x V = (1, 0)
        assert!(
            (nabla_0[0] - 1.0).abs() < 1e-6,
            "∇_x V^x should be 1, got {}",
            nabla_0[0]
        );
        assert!(
            nabla_0[1].abs() < 1e-6,
            "∇_x V^y should be 0, got {}",
            nabla_0[1]
        );

        // ∇_y V = (0, 1)
        assert!(
            nabla_1[0].abs() < 1e-6,
            "∇_y V^x should be 0, got {}",
            nabla_1[0]
        );
        assert!(
            (nabla_1[1] - 1.0).abs() < 1e-6,
            "∇_y V^y should be 1, got {}",
            nabla_1[1]
        );
    }

    #[test]
    fn test_scalar_covariant_derivative() {
        // Scalar covariant derivative = partial derivative
        let f = ScalarField::with_dimension(|coords| coords[0] * coords[0] + coords[1], 2);

        // ∂f/∂x = 2x
        let nabla_x = CovariantDerivative::scalar(&f, 0, &[3.0, 4.0]).unwrap();
        assert!((nabla_x - 6.0).abs() < 1e-4);

        // ∂f/∂y = 1
        let nabla_y = CovariantDerivative::scalar(&f, 1, &[3.0, 4.0]).unwrap();
        assert!((nabla_y - 1.0).abs() < 1e-4);
    }
}
