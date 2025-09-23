//! Amari: Advanced Mathematical Algebra for Robust Intelligence
//!
//! A comprehensive mathematical computing library combining:
//! - Geometric algebra (Clifford algebras)
//! - Tropical algebra (max-plus semiring)
//! - Dual number automatic differentiation
//! - Information geometry
//! - Fusion systems for neural network optimization

pub use amari_core as core;
pub use amari_dual as dual;
pub use amari_fusion as fusion;
pub use amari_info_geom as info_geom;
pub use amari_tropical as tropical;

// Re-export common types
pub use amari_core::{Bivector, Multivector, Scalar, Vector};
pub use amari_dual::{DualMultivector, DualNumber};
pub use amari_fusion::TropicalDualClifford;
pub use amari_info_geom::{DuallyFlatManifold, FisherInformationMatrix, SimpleAlphaConnection};
pub use amari_tropical::{TropicalMatrix, TropicalMultivector, TropicalNumber};
