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

/// Compute Laplacian of scalar field: ∇²f = ∇·(∇f)
pub fn laplacian<const P: usize, const Q: usize, const R: usize>(
    f: &ScalarField<P, Q, R>,
    coords: &[f64],
) -> f64 {
    let nabla = VectorDerivative::<P, Q, R>::new(CoordinateSystem::Cartesian);

    // First compute gradient as a vector field
    let grad_f = VectorField::<P, Q, R>::new(move |c| nabla.gradient(f, c));

    // Then compute divergence of gradient
    nabla.divergence(&grad_f, coords)
}
