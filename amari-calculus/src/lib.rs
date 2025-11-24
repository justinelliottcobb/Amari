//! # Amari Calculus
//!
//! Geometric calculus - a unified framework for differential and integral calculus using geometric algebra.
//!
//! ## Overview
//!
//! This crate provides geometric calculus operations that unify:
//! - Vector calculus (gradient, divergence, curl)
//! - Differential forms
//! - Tensor calculus
//! - Covariant derivatives on manifolds
//!
//! ## Mathematical Foundation
//!
//! Geometric calculus is built on the **vector derivative operator**:
//!
//! ```text
//! ∇ = e^i ∂_i  (sum over basis vectors)
//! ```
//!
//! This operator combines:
//! - Dot product → divergence (∇·F)
//! - Wedge product → curl (∇∧F)
//! - Full geometric product → complete derivative (∇F = ∇·F + ∇∧F)
//!
//! ## Key Features
//!
//! - **Vector Derivative Operator**: The fundamental ∇ operator
//! - **Classical Operators**: Gradient, divergence, curl, Laplacian
//! - **Manifold Calculus**: Covariant derivatives, connections, geodesics
//! - **Lie Derivatives**: Derivatives along vector fields
//! - **Integration**: Integration on manifolds using amari-measure
//! - **Fundamental Theorem**: ∫_V (∇F) dV = ∮_∂V F dS
//!
//! ## Examples
//!
//! ### Gradient of a scalar field
//!
//! ```rust
//! use amari_calculus::{ScalarField, VectorDerivative, CoordinateSystem};
//! use amari_core::Multivector;
//!
//! // Define scalar field f(x, y) = x² + y²
//! let f = ScalarField::<3, 0, 0>::new(|coords| {
//!     coords[0].powi(2) + coords[1].powi(2)
//! });
//!
//! // Create vector derivative operator
//! let nabla = VectorDerivative::<3, 0, 0>::new(CoordinateSystem::Cartesian);
//!
//! // Compute gradient at point (1, 2)
//! let grad_f = nabla.gradient(&f, &[1.0, 2.0, 0.0]);
//!
//! // Gradient should be approximately (2, 4, 0)
//! ```
//!
//! ### Divergence of a vector field
//!
//! ```rust
//! use amari_calculus::{VectorField, VectorDerivative, CoordinateSystem, vector_from_slice};
//!
//! // Define vector field F(x, y, z) = (x, y, z)
//! let f = VectorField::<3, 0, 0>::new(|coords| {
//!     vector_from_slice(&[coords[0], coords[1], coords[2]])
//! });
//!
//! let nabla = VectorDerivative::<3, 0, 0>::new(CoordinateSystem::Cartesian);
//!
//! // Compute divergence (should be 3)
//! let div_f = nabla.divergence(&f, &[1.0, 1.0, 1.0]);
//! ```
//!
//! ### Curl of a vector field
//!
//! ```rust
//! use amari_calculus::{VectorField, VectorDerivative, CoordinateSystem, vector_from_slice};
//!
//! // Define vector field F(x, y, z) = (-y, x, 0) (rotation around z-axis)
//! let f = VectorField::<3, 0, 0>::new(|coords| {
//!     vector_from_slice(&[-coords[1], coords[0], 0.0])
//! });
//!
//! let nabla = VectorDerivative::<3, 0, 0>::new(CoordinateSystem::Cartesian);
//!
//! // Compute curl (should be (0, 0, 2) bivector representing rotation)
//! let curl_f = nabla.curl(&f, &[0.0, 0.0, 0.0]);
//! ```

#![cfg_attr(not(feature = "std"), no_std)]

// Error handling
mod error;
pub use error::*;

// Core derivative types
mod derivative;
pub use derivative::*;

// Field types
pub mod fields;
pub use fields::{MultivectorField, ScalarField, VectorField};

// Classical differential operators
pub mod operators;
pub use operators::{curl, divergence, gradient, laplacian};

// Covariant derivatives on manifolds
pub mod manifold;
pub use manifold::RiemannianManifold;

// Lie derivatives
mod lie;
pub use lie::*;

// Integration on manifolds
mod integration;
pub use integration::ManifoldIntegrator;

/// Utility function to create a vector multivector from a slice of components
///
/// # Arguments
///
/// * `components` - Slice of f64 values representing vector components
///
/// # Returns
///
/// A Multivector containing the vector (grade-1 elements only)
pub fn vector_from_slice<const P: usize, const Q: usize, const R: usize>(
    components: &[f64],
) -> amari_core::Multivector<P, Q, R> {
    let mut mv = amari_core::Multivector::zero();
    let dim = P + Q + R;
    for (i, &val) in components.iter().enumerate() {
        if i < dim {
            mv.set_vector_component(i, val);
        }
    }
    mv
}

/// Coordinate systems for differential operators
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CoordinateSystem {
    /// Cartesian coordinates (x, y, z, ...)
    Cartesian,
    /// Spherical coordinates (r, θ, φ)
    Spherical,
    /// Cylindrical coordinates (ρ, φ, z)
    Cylindrical,
    /// Polar coordinates (r, θ) - 2D only
    Polar,
}

/// Prelude module for convenient imports
pub mod prelude {
    pub use crate::fields::{MultivectorField, ScalarField, VectorField};
    pub use crate::operators::{curl, divergence, gradient, laplacian};
    pub use crate::CoordinateSystem;
    pub use crate::VectorDerivative;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_crate_compiles() {
        // Basic smoke test to ensure crate compiles
        let _coords = CoordinateSystem::Cartesian;
    }
}
