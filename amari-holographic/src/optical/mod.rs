//! GA-native optical field operations for Lee hologram encoding.
//!
//! This module provides geometric algebra representations of optical fields,
//! replacing complex number representations with rotors from Cl(2,0).
//!
//! # Background
//!
//! Complex numbers are isomorphic to the even subalgebra of Cl(2,0):
//!
//! ```text
//! Complex: z = a + bi           where i² = -1
//! GA:      R = a + b·e₁₂        where (e₁₂)² = -1
//! ```
//!
//! An optical field at position (x,y) is represented as a rotor field:
//!
//! ```text
//! E(x,y) = A(x,y) · exp(φ(x,y) · e₁₂)
//!        = A(x,y) · (cos(φ) + sin(φ)·e₁₂)
//! ```
//!
//! # VSA Operations
//!
//! | Operation | Complex Form | GA Form |
//! |-----------|--------------|---------|
//! | Binding | z₁ · z₂ | R₁ · R₂ (rotor product) |
//! | Unbinding | z₁ · z₂* | R₁ · R₂† (product with reverse) |
//! | Bundling | Σ wᵢzᵢ | Σ wᵢRᵢ (weighted sum) |
//! | Similarity | Re(z₁*z₂)/\|z₁\|\|z₂\| | ⟨R₁†R₂⟩₀ / (\|R₁\|\|R₂\|) |
//!
//! # Example
//!
//! ```ignore
//! use amari_holographic::optical::{OpticalRotorField, OpticalFieldAlgebra, GeometricLeeEncoder};
//!
//! // Create rotor field from phase
//! let field = OpticalRotorField::from_phase(
//!     vec![0.0, std::f32::consts::FRAC_PI_4, std::f32::consts::FRAC_PI_2],
//!     (3, 1),
//! );
//!
//! // Create algebra instance
//! let algebra = OpticalFieldAlgebra::new((3, 1));
//!
//! // Binding adds phases
//! let bound = algebra.bind(&field, &field);
//!
//! // Encode as binary Lee hologram
//! let encoder = GeometricLeeEncoder::with_frequency((3, 1), 0.25);
//! let hologram = encoder.encode(&field);
//! ```

mod algebra;
mod codebook;
mod hologram;
mod lee_encoder;
mod rotor_field;
mod tropical;

#[cfg(test)]
mod tests;

pub use algebra::OpticalFieldAlgebra;
pub use codebook::{CodebookConfig, OpticalCodebook, SymbolId};
pub use hologram::BinaryHologram;
pub use lee_encoder::{GeometricLeeEncoder, LeeEncoderConfig};
pub use rotor_field::OpticalRotorField;
pub use tropical::TropicalOpticalAlgebra;
