//! Holographic memory using TropicalDualClifford representations.
//!
//! This module provides Vector Symbolic Architecture (VSA) operations
//! built on the tropical-dual-Clifford fusion, enabling:
//!
//! - Compositional key-value storage in superposition
//! - Content-addressable retrieval with automatic cleanup
//! - Retrieval attribution via dual number gradients
//! - Temperature-controlled soft↔hard retrieval
//!
//! # Holographic Reduced Representations
//!
//! A holographic memory stores associations `(key, value)` by:
//! 1. **Binding**: `bound = key ⊛ value` (creates association, result dissimilar to both inputs)
//! 2. **Bundling**: `memory = bound₁ ⊕ bound₂ ⊕ ... ⊕ boundₙ` (superposition in single vector)
//! 3. **Retrieval**: `value ≈ key⁻¹ ⊛ memory` (unbind with inverse, get target + noise)
//!
//! The noise term is the sum/max of cross-talk from other items. Capacity is O(DIM / log DIM) items.
//!
//! # Temperature-Parameterized Operations
//!
//! The Maslov dequantization connects soft (standard) and hard (tropical) operations:
//!
//! ```text
//! β → 0:   soft operations (sum, product)
//! β → ∞:   hard operations (max, +)
//! ```
//!
//! Encoding should be soft (preserves superposition), retrieval can be hard (automatic cleanup).
//!
//! # Example
//!
//! ```rust,ignore
//! use amari_fusion::holographic::{HolographicMemory, BindingAlgebra};
//! use amari_fusion::TropicalDualClifford;
//!
//! let mut memory = HolographicMemory::<f64, 8>::new(BindingAlgebra::default());
//!
//! let key = TropicalDualClifford::from_logits(&key_logits);
//! let value = TropicalDualClifford::from_logits(&value_logits);
//!
//! memory.store(&key, &value);
//!
//! let retrieved = memory.retrieve(&key);
//! assert!(retrieved.confidence > 0.9);
//! ```

mod binding;
mod error;
mod memory;
mod resonator;
mod verified;

pub use binding::{Bindable, BindingAlgebra};
pub use error::{HolographicError, HolographicResult};
pub use memory::{CapacityInfo, HolographicMemory, RetrievalResult};
pub use resonator::{CleanupResult, FactorizationResult, Resonator, ResonatorConfig};
pub use verified::{VerifiedBindable, VerifiedHolographicMemory};

#[cfg(test)]
mod tests;
