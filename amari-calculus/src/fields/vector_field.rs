//! Vector field implementation

use crate::{CalculusError, CalculusResult};
use amari_core::Multivector;

/// A vector field F: ℝⁿ → Cl(p,q,r)_1 (grade-1 multivectors)
///
/// Represents a function that maps points in n-dimensional space to vectors (grade-1 multivectors).
///
/// # Examples
///
/// ```
/// use amari_calculus::VectorField;
/// use amari_core::Multivector;
///
/// // Define F(x, y, z) = (x, y, z) - radial vector field
/// let f = VectorField::<3, 0, 0>::new(|coords| {
///     Multivector::from_vector(&[coords[0], coords[1], coords[2]])
/// });
///
/// // Evaluate at (1, 2, 3)
/// let value = f.evaluate(&[1.0, 2.0, 3.0]);
/// ```
#[derive(Clone)]
pub struct VectorField<const P: usize, const Q: usize, const R: usize> {
    /// The function defining the field
    function: fn(&[f64]) -> Multivector<P, Q, R>,
    /// Domain dimension
    dim: usize,
}

impl<const P: usize, const Q: usize, const R: usize> VectorField<P, Q, R> {
    /// Create a new vector field from a function
    ///
    /// # Arguments
    ///
    /// * `function` - Function mapping coordinates to vectors
    ///
    /// # Examples
    ///
    /// ```
    /// use amari_calculus::VectorField;
    /// use amari_core::Multivector;
    ///
    /// // Rotation field F(x, y) = (-y, x, 0)
    /// let f = VectorField::<3, 0, 0>::new(|coords| {
    ///     Multivector::from_vector(&[-coords[1], coords[0], 0.0])
    /// });
    /// ```
    pub fn new(function: fn(&[f64]) -> Multivector<P, Q, R>) -> Self {
        Self {
            function,
            dim: P + Q + R,
        }
    }

    /// Create a vector field with explicit dimension
    pub fn with_dimension(function: fn(&[f64]) -> Multivector<P, Q, R>, dim: usize) -> Self {
        Self { function, dim }
    }

    /// Evaluate the vector field at a point
    ///
    /// # Arguments
    ///
    /// * `coords` - Coordinates of the point
    ///
    /// # Returns
    ///
    /// The vector value at the point
    pub fn evaluate(&self, coords: &[f64]) -> Multivector<P, Q, R> {
        (self.function)(coords)
    }

    /// Get the domain dimension
    pub fn dimension(&self) -> usize {
        self.dim
    }

    /// Compute numerical derivative of vector field component along coordinate axis
    ///
    /// # Arguments
    ///
    /// * `coords` - Point at which to compute derivative
    /// * `axis` - Coordinate axis index
    /// * `h` - Step size (default: 1e-5)
    pub fn partial_derivative(
        &self,
        coords: &[f64],
        axis: usize,
        h: f64,
    ) -> CalculusResult<Multivector<P, Q, R>> {
        if axis >= self.dim {
            return Err(CalculusError::InvalidDimension {
                expected: self.dim,
                got: axis,
            });
        }

        let mut coords_plus = coords.to_vec();
        let mut coords_minus = coords.to_vec();

        coords_plus[axis] += h;
        coords_minus[axis] -= h;

        let f_plus = self.evaluate(&coords_plus);
        let f_minus = self.evaluate(&coords_minus);

        // Compute (f_plus - f_minus) / (2h)
        let mut result = f_plus;
        result = result.add(&f_minus.scale(-1.0));
        result = result.scale(1.0 / (2.0 * h));

        Ok(result)
    }

    /// Add two vector fields pointwise
    pub fn add(&self, other: &Self) -> Self {
        let f1 = self.function;
        let f2 = other.function;
        Self::new(move |coords| f1(coords).add(&f2(coords)))
    }

    /// Scale vector field by constant
    pub fn scale(&self, c: f64) -> Self {
        let f = self.function;
        Self::new(move |coords| f(coords).scale(c))
    }

    /// Dot product of two vector fields (returns scalar field)
    pub fn dot(&self, other: &Self) -> crate::ScalarField<P, Q, R> {
        let f1 = self.function;
        let f2 = other.function;
        crate::ScalarField::new(move |coords| {
            let v1 = f1(coords);
            let v2 = f2(coords);
            v1.dot(&v2)
        })
    }
}

impl<const P: usize, const Q: usize, const R: usize> std::fmt::Debug for VectorField<P, Q, R> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VectorField")
            .field("dim", &self.dim)
            .field("signature", &format!("Cl({},{},{})", P, Q, R))
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vector_field_evaluation() {
        // F(x, y, z) = (x, y, z) - radial field
        let f = VectorField::<3, 0, 0>::new(|coords| {
            Multivector::from_vector(&[coords[0], coords[1], coords[2]])
        });

        let v = f.evaluate(&[1.0, 2.0, 3.0]);
        let expected = Multivector::<3, 0, 0>::from_vector(&[1.0, 2.0, 3.0]);

        // Check components
        for i in 0..3 {
            assert!(
                (v.get_component(1 << i) - expected.get_component(1 << i)).abs() < 1e-10,
                "Component {} mismatch",
                i
            );
        }
    }

    #[test]
    fn test_vector_field_arithmetic() {
        let f = VectorField::<3, 0, 0>::new(|coords| {
            Multivector::from_vector(&[coords[0], coords[1], 0.0])
        });

        let g = VectorField::<3, 0, 0>::new(|coords| {
            Multivector::from_vector(&[coords[1], coords[0], 0.0])
        });

        // Test addition
        let h = f.add(&g);
        let v = h.evaluate(&[2.0, 3.0, 0.0]);
        // Should be (2+3, 3+2, 0) = (5, 5, 0)
        assert!((v.get_component(1 << 0) - 5.0).abs() < 1e-10);
        assert!((v.get_component(1 << 1) - 5.0).abs() < 1e-10);

        // Test scaling
        let h = f.scale(2.0);
        let v = h.evaluate(&[2.0, 3.0, 0.0]);
        // Should be (4, 6, 0)
        assert!((v.get_component(1 << 0) - 4.0).abs() < 1e-10);
        assert!((v.get_component(1 << 1) - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_partial_derivative() {
        // F(x, y) = (x², y², 0)
        // ∂F/∂x = (2x, 0, 0), ∂F/∂y = (0, 2y, 0)
        let f = VectorField::<3, 0, 0>::new(|coords| {
            Multivector::from_vector(&[coords[0].powi(2), coords[1].powi(2), 0.0])
        });

        let df_dx = f.partial_derivative(&[3.0, 4.0, 0.0], 0, 1e-5).unwrap();
        let df_dy = f.partial_derivative(&[3.0, 4.0, 0.0], 1, 1e-5).unwrap();

        // ∂F/∂x at (3,4) should be approximately (6, 0, 0)
        assert!(
            (df_dx.get_component(1 << 0) - 6.0).abs() < 1e-4,
            "∂F_x/∂x should be 6.0"
        );
        assert!(
            df_dx.get_component(1 << 1).abs() < 1e-4,
            "∂F_y/∂x should be 0.0"
        );

        // ∂F/∂y at (3,4) should be approximately (0, 8, 0)
        assert!(
            df_dy.get_component(1 << 0).abs() < 1e-4,
            "∂F_x/∂y should be 0.0"
        );
        assert!(
            (df_dy.get_component(1 << 1) - 8.0).abs() < 1e-4,
            "∂F_y/∂y should be 8.0"
        );
    }
}
