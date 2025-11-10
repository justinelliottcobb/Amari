//! Amari: Advanced Mathematical Algebra for Robust Intelligence
//!
//! A comprehensive mathematical computing library combining:
//! - Geometric algebra (Clifford algebras)
//! - Tropical algebra (max-plus semiring)
//! - Dual number automatic differentiation
//! - Geometric network analysis
//! - Information geometry
//! - Fusion systems for neural network optimization
//! - Deterministic physics for networked applications (opt-in)

pub use amari_automata as automata;
pub use amari_core as core;
pub use amari_dual as dual;
pub use amari_enumerative as enumerative;
pub use amari_fusion as fusion;
pub use amari_info_geom as info_geom;
pub use amari_network as network;
pub use amari_relativistic as relativistic;
pub use amari_tropical as tropical;

// Deterministic computation module (opt-in via feature flag)
#[cfg(feature = "deterministic")]
pub mod deterministic;

use thiserror::Error;

/// Unified error type for the Amari library
#[derive(Error, Debug)]
pub enum AmariError {
    /// Core geometric algebra error
    #[error(transparent)]
    Core(#[from] amari_core::CoreError),

    /// Automata error
    #[error("{0}")]
    Automata(amari_automata::AutomataError),

    /// Enumerative geometry error
    #[error(transparent)]
    Enumerative(#[from] amari_enumerative::EnumerativeError),

    /// GPU computation error
    #[cfg(feature = "gpu")]
    #[error(transparent)]
    Gpu(#[from] amari_gpu::GpuError),

    /// Information geometry error
    #[error(transparent)]
    InfoGeom(#[from] amari_info_geom::InfoGeomError),

    /// Network analysis error
    #[error(transparent)]
    Network(#[from] amari_network::NetworkError),

    /// Fusion system error
    #[error(transparent)]
    Fusion(#[from] amari_fusion::FusionError),

    /// Tropical algebra error
    #[error(transparent)]
    Tropical(#[from] amari_tropical::TropicalError),

    /// Dual number error
    #[error(transparent)]
    Dual(#[from] amari_dual::DualError),
}

/// Result type for Amari operations
pub type AmariResult<T> = Result<T, AmariError>;

// Manual implementation for AutomataError since it doesn't use thiserror
impl From<amari_automata::AutomataError> for AmariError {
    fn from(err: amari_automata::AutomataError) -> Self {
        AmariError::Automata(err)
    }
}

// Re-export common types
pub use amari_core::{Bivector, Multivector, Scalar, Vector};
pub use amari_dual::{DualMultivector, DualNumber};
pub use amari_fusion::TropicalDualClifford;
pub use amari_info_geom::{DuallyFlatManifold, FisherInformationMatrix, SimpleAlphaConnection};
pub use amari_network::{
    Community, GeometricEdge, GeometricNetwork, NodeMetadata, PropagationAnalysis,
};
pub use amari_tropical::{TropicalMatrix, TropicalMultivector, TropicalNumber};
