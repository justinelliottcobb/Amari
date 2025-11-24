//! Classical differential operators (gradient, divergence, curl, Laplacian)

use crate::{fields::*, CoordinateSystem, VectorDerivative};
use amari_core::Multivector;

/// Compute gradient of scalar field: ∇f
///
/// Convenience function wrapping VectorDerivative::gradient
pub fn gradient<const P: usize, const Q: usize, const R: usize>(
    f: &ScalarField<P, Q, R>,
    coords: &[f64],
) -> Multivector<P, Q, R> {
    let nabla = VectorDerivative::<P, Q, R>::new(CoordinateSystem::Cartesian);
    nabla.gradient(f, coords)
}

/// Compute divergence of vector field: ∇·F
///
/// Convenience function wrapping VectorDerivative::divergence
pub fn divergence<const P: usize, const Q: usize, const R: usize>(
    f: &VectorField<P, Q, R>,
    coords: &[f64],
) -> f64 {
    let nabla = VectorDerivative::<P, Q, R>::new(CoordinateSystem::Cartesian);
    nabla.divergence(f, coords)
}

/// Compute curl of vector field: ∇∧F
///
/// Convenience function wrapping VectorDerivative::curl
pub fn curl<const P: usize, const Q: usize, const R: usize>(
    f: &VectorField<P, Q, R>,
    coords: &[f64],
) -> Multivector<P, Q, R> {
    let nabla = VectorDerivative::<P, Q, R>::new(CoordinateSystem::Cartesian);
    nabla.curl(f, coords)
}

/// Compute Laplacian of scalar field: ∇²f = ∇·(∇f) = ∂²f/∂x² + ∂²f/∂y² + ...
///
/// Computes the sum of second partial derivatives.
pub fn laplacian<const P: usize, const Q: usize, const R: usize>(
    f: &ScalarField<P, Q, R>,
    coords: &[f64],
) -> f64 {
    let dim = P + Q + R;
    let h = 1e-5;
    let mut laplacian = 0.0;

    // Sum of second partial derivatives: ∂²f/∂x_i²
    for i in 0..dim {
        let mut coords_plus = coords.to_vec();
        let mut coords_minus = coords.to_vec();
        let coords_center = coords;

        coords_plus[i] += h;
        coords_minus[i] -= h;

        let f_plus = f.evaluate(&coords_plus);
        let f_minus = f.evaluate(&coords_minus);
        let f_center = f.evaluate(coords_center);

        // Second derivative: (f(x+h) - 2f(x) + f(x-h)) / h²
        laplacian += (f_plus - 2.0 * f_center + f_minus) / (h * h);
    }

    laplacian
}
