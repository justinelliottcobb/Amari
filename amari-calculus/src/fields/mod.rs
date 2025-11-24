//! Field types for geometric calculus
//!
//! This module provides types for representing scalar, vector, and multivector fields
//! defined over geometric spaces.

mod multivector_field;
mod scalar_field;
mod vector_field;

pub use multivector_field::MultivectorField;
pub use scalar_field::ScalarField;
pub use vector_field::VectorField;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scalar_field_evaluation() {
        // f(x, y) = x² + y²
        let f = ScalarField::<3, 0, 0>::new(|coords| coords[0].powi(2) + coords[1].powi(2));

        let val = f.evaluate(&[3.0, 4.0, 0.0]);
        assert!((val - 25.0).abs() < 1e-10);
    }

    #[test]
    fn test_vector_field_evaluation() {
        // F(x, y, z) = (x, y, z)
        let f = VectorField::<3, 0, 0>::new(|coords| {
            crate::vector_from_slice(&[coords[0], coords[1], coords[2]])
        });

        let val = f.evaluate(&[1.0, 2.0, 3.0]);
        let expected = crate::vector_from_slice::<3, 0, 0>(&[1.0, 2.0, 3.0]);

        // Check that components match
        for i in 0..3 {
            assert!((val.vector_component(i) - expected.vector_component(i)).abs() < 1e-10);
        }
    }
}
