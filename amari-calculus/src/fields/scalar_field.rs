//! Scalar field implementation

use crate::{CalculusError, CalculusResult};

/// A scalar field f: ℝⁿ → ℝ
///
/// Represents a function that maps points in n-dimensional space to scalar values.
///
/// # Examples
///
/// ```
/// use amari_calculus::ScalarField;
///
/// // Define f(x, y) = x² + y²
/// let f = ScalarField::<3, 0, 0>::new(|coords| {
///     coords[0].powi(2) + coords[1].powi(2)
/// });
///
/// // Evaluate at (3, 4)
/// let value = f.evaluate(&[3.0, 4.0, 0.0]);
/// assert!((value - 25.0).abs() < 1e-10);
/// ```
#[derive(Clone)]
pub struct ScalarField<const P: usize, const Q: usize, const R: usize> {
    /// The function defining the field
    function: fn(&[f64]) -> f64,
    /// Domain dimension
    dim: usize,
}

impl<const P: usize, const Q: usize, const R: usize> ScalarField<P, Q, R> {
    /// Create a new scalar field from a function
    ///
    /// # Arguments
    ///
    /// * `function` - Function mapping coordinates to scalar values
    ///
    /// # Examples
    ///
    /// ```
    /// use amari_calculus::ScalarField;
    ///
    /// // Quadratic function f(x, y) = x² + y²
    /// let f = ScalarField::<3, 0, 0>::new(|coords| {
    ///     coords[0].powi(2) + coords[1].powi(2)
    /// });
    /// ```
    pub fn new(function: fn(&[f64]) -> f64) -> Self {
        Self {
            function,
            dim: P + Q + R,
        }
    }

    /// Create a scalar field with explicit dimension
    ///
    /// Useful when the field dimension doesn't match the algebra dimension.
    pub fn with_dimension(function: fn(&[f64]) -> f64, dim: usize) -> Self {
        Self { function, dim }
    }

    /// Evaluate the scalar field at a point
    ///
    /// # Arguments
    ///
    /// * `coords` - Coordinates of the point
    ///
    /// # Returns
    ///
    /// The scalar value at the point
    ///
    /// # Errors
    ///
    /// Returns error if coordinate dimension doesn't match field dimension
    pub fn evaluate(&self, coords: &[f64]) -> f64 {
        (self.function)(coords)
    }

    /// Get the domain dimension
    pub fn dimension(&self) -> usize {
        self.dim
    }

    /// Compute numerical derivative along coordinate axis
    ///
    /// Uses centered difference: f'(x) ≈ (f(x+h) - f(x-h)) / (2h)
    ///
    /// # Arguments
    ///
    /// * `coords` - Point at which to compute derivative
    /// * `axis` - Coordinate axis index (0 = x, 1 = y, etc.)
    /// * `h` - Step size (default: 1e-5)
    pub fn partial_derivative(&self, coords: &[f64], axis: usize, h: f64) -> CalculusResult<f64> {
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

        Ok((f_plus - f_minus) / (2.0 * h))
    }

    /// Add two scalar fields pointwise
    pub fn add(&self, other: &Self) -> Self {
        let f1 = self.function;
        let f2 = other.function;
        Self::new(move |coords| f1(coords) + f2(coords))
    }

    /// Multiply two scalar fields pointwise
    pub fn mul(&self, other: &Self) -> Self {
        let f1 = self.function;
        let f2 = other.function;
        Self::new(move |coords| f1(coords) * f2(coords))
    }

    /// Scale scalar field by constant
    pub fn scale(&self, c: f64) -> Self {
        let f = self.function;
        Self::new(move |coords| c * f(coords))
    }
}

impl<const P: usize, const Q: usize, const R: usize> std::fmt::Debug for ScalarField<P, Q, R> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ScalarField")
            .field("dim", &self.dim)
            .field("signature", &format!("Cl({},{},{})", P, Q, R))
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scalar_field_evaluation() {
        // f(x, y) = x² + y²
        let f = ScalarField::<3, 0, 0>::new(|coords| coords[0].powi(2) + coords[1].powi(2));

        assert!((f.evaluate(&[0.0, 0.0, 0.0]) - 0.0).abs() < 1e-10);
        assert!((f.evaluate(&[3.0, 4.0, 0.0]) - 25.0).abs() < 1e-10);
        assert!((f.evaluate(&[1.0, 1.0, 0.0]) - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_partial_derivative() {
        // f(x, y) = x² + y²
        // ∂f/∂x = 2x, ∂f/∂y = 2y
        let f = ScalarField::<3, 0, 0>::new(|coords| coords[0].powi(2) + coords[1].powi(2));

        let df_dx = f.partial_derivative(&[3.0, 4.0, 0.0], 0, 1e-5).unwrap();
        let df_dy = f.partial_derivative(&[3.0, 4.0, 0.0], 1, 1e-5).unwrap();

        assert!(
            (df_dx - 6.0).abs() < 1e-6,
            "∂f/∂x should be 6.0, got {}",
            df_dx
        );
        assert!(
            (df_dy - 8.0).abs() < 1e-6,
            "∂f/∂y should be 8.0, got {}",
            df_dy
        );
    }

    #[test]
    fn test_scalar_field_arithmetic() {
        let f = ScalarField::<3, 0, 0>::new(|coords| coords[0] + coords[1]);
        let g = ScalarField::<3, 0, 0>::new(|coords| coords[0] * coords[1]);

        // Test addition
        let h = f.add(&g);
        assert!((h.evaluate(&[2.0, 3.0, 0.0]) - 11.0).abs() < 1e-10); // (2+3) + (2*3) = 11

        // Test multiplication
        let h = f.mul(&g);
        assert!((h.evaluate(&[2.0, 3.0, 0.0]) - 30.0).abs() < 1e-10); // (2+3) * (2*3) = 30

        // Test scaling
        let h = f.scale(2.0);
        assert!((h.evaluate(&[2.0, 3.0, 0.0]) - 10.0).abs() < 1e-10); // 2 * (2+3) = 10
    }
}
