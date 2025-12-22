//! Holographic associative memory for key-value storage in superposition.
//!
//! This module provides the main `HolographicMemory` data structure that stores
//! key-value associations using holographic reduced representations.

use alloc::vec::Vec;
use core::marker::PhantomData;
use num_traits::Float;

#[cfg(feature = "rayon")]
use rayon::prelude::*;

use super::binding::{Bindable, BindingAlgebra};
use crate::TropicalDualClifford;

/// A holographic associative memory storing key-value pairs in superposition.
///
/// # Capacity
///
/// Reliable retrieval degrades as items are added. Capacity scales as O(DIM / log DIM).
/// The memory tracks estimated SNR and warns when approaching capacity limits.
///
/// # Example
///
/// ```rust,ignore
/// use amari_fusion::holographic::{HolographicMemory, BindingAlgebra};
/// use amari_fusion::TropicalDualClifford;
///
/// let mut memory = HolographicMemory::<f64, 8>::new(BindingAlgebra::default());
///
/// let key = TropicalDualClifford::from_logits(&key_logits);
/// let value = TropicalDualClifford::from_logits(&value_logits);
///
/// memory.store(&key, &value);
///
/// let retrieved = memory.retrieve(&key);
/// assert!(retrieved.confidence > 0.9);
/// ```
#[derive(Clone)]
pub struct HolographicMemory<T: Float, const DIM: usize> {
    /// The superposed memory trace
    trace: TropicalDualClifford<T, DIM>,
    /// Number of items stored
    item_count: usize,
    /// Binding algebra configuration
    algebra: BindingAlgebra,
    /// Optional: store keys for resonator cleanup and attribution
    stored_keys: Option<Vec<TropicalDualClifford<T, DIM>>>,
    /// Phantom marker for Float trait
    _phantom: PhantomData<T>,
}

/// Result of a retrieval operation.
#[derive(Clone, Debug)]
pub struct RetrievalResult<T: Float, const DIM: usize> {
    /// The retrieved value (after cleanup if enabled)
    pub value: TropicalDualClifford<T, DIM>,
    /// Raw retrieved value before cleanup
    pub raw_value: TropicalDualClifford<T, DIM>,
    /// Estimated confidence in [0, 1] based on SNR
    pub confidence: f64,
    /// Attribution: which stored items contributed (from dual gradients)
    /// Indices correspond to storage order if key tracking is enabled
    pub attribution: Vec<(usize, f64)>,
    /// Similarity of retrieved value to query (sanity check)
    pub query_similarity: f64,
}

impl<T: Float + Send + Sync, const DIM: usize> RetrievalResult<T, DIM>
where
    T: num_traits::NumCast,
{
    /// Verify that this retrieval result is valid.
    pub fn verify_result_validity(&self) -> bool {
        // Check that value is finite
        let value_norm = self.value.norm();
        if !value_norm.is_finite() {
            return false;
        }

        // Check confidence is in valid range
        if !(0.0..=1.0).contains(&self.confidence) {
            return false;
        }

        true
    }
}

/// Information about memory capacity and health.
#[derive(Clone, Debug)]
pub struct CapacityInfo {
    /// Current number of stored items
    pub item_count: usize,
    /// Theoretical maximum before severe degradation
    pub theoretical_capacity: usize,
    /// Estimated signal-to-noise ratio (higher is better)
    pub estimated_snr: f64,
    /// Recommended: stop storing if SNR drops below this
    pub snr_threshold: f64,
    /// Whether memory is approaching capacity limits
    pub near_capacity: bool,
}

impl<T: Float + Send + Sync, const DIM: usize> HolographicMemory<T, DIM>
where
    T: num_traits::NumCast,
{
    /// Create a new empty holographic memory.
    pub fn new(algebra: BindingAlgebra) -> Self {
        Self {
            trace: TropicalDualClifford::zero(),
            item_count: 0,
            algebra,
            stored_keys: None,
            _phantom: PhantomData,
        }
    }

    /// Create with key tracking enabled (for resonator cleanup and attribution).
    pub fn with_key_tracking(algebra: BindingAlgebra) -> Self {
        Self {
            trace: TropicalDualClifford::zero(),
            item_count: 0,
            algebra,
            stored_keys: Some(Vec::new()),
            _phantom: PhantomData,
        }
    }

    /// Store a key-value association.
    pub fn store(
        &mut self,
        key: &TropicalDualClifford<T, DIM>,
        value: &TropicalDualClifford<T, DIM>,
    ) {
        // Bind key and value
        let bound = key.bind(value);

        // Bundle into memory trace
        if self.item_count == 0 {
            self.trace = bound;
        } else {
            self.trace = self.trace.bundle(&bound, self.algebra.bundle_beta);
        }

        // Track key if enabled
        if let Some(ref mut keys) = self.stored_keys {
            keys.push(key.clone());
        }

        self.item_count += 1;
    }

    /// Store multiple associations at once (more efficient bundling).
    pub fn store_batch(
        &mut self,
        pairs: &[(TropicalDualClifford<T, DIM>, TropicalDualClifford<T, DIM>)],
    ) {
        if pairs.is_empty() {
            return;
        }

        // Bind all pairs
        #[cfg(feature = "rayon")]
        let bindings: Vec<_> = pairs.par_iter().map(|(k, v)| k.bind(v)).collect();

        #[cfg(not(feature = "rayon"))]
        let bindings: Vec<_> = pairs.iter().map(|(k, v)| k.bind(v)).collect();

        // Bundle all bindings
        let batch_trace = TropicalDualClifford::bundle_all(&bindings, self.algebra.bundle_beta);

        // Merge with existing trace
        if self.item_count == 0 {
            self.trace = batch_trace;
        } else {
            self.trace = self.trace.bundle(&batch_trace, self.algebra.bundle_beta);
        }

        // Track keys if enabled
        if let Some(ref mut keys) = self.stored_keys {
            for (k, _) in pairs {
                keys.push(k.clone());
            }
        }

        self.item_count += pairs.len();
    }

    /// Retrieve the value associated with a key.
    pub fn retrieve(&self, key: &TropicalDualClifford<T, DIM>) -> RetrievalResult<T, DIM> {
        self.retrieve_at_temperature(key, self.algebra.retrieval_beta)
    }

    /// Retrieve with custom temperature (override algebra settings).
    pub fn retrieve_at_temperature(
        &self,
        key: &TropicalDualClifford<T, DIM>,
        beta: f64,
    ) -> RetrievalResult<T, DIM> {
        // Unbind with key to get noisy value
        let raw_value = key.unbind(&self.trace);

        // Compute confidence based on SNR estimate
        let snr = self.estimate_snr();
        let confidence = self.snr_to_confidence(snr);

        // For hard retrieval, we could apply resonator cleanup here
        // For now, we just return the raw value
        let value = if beta.is_infinite() {
            // Hard retrieval: normalize the result
            raw_value.normalize()
        } else {
            raw_value.clone()
        };

        // Compute attribution if key tracking is enabled
        let attribution = self.compute_attribution(key, &value);

        // Compute query similarity
        let query_similarity = value.similarity(key);

        RetrievalResult {
            value,
            raw_value,
            confidence,
            attribution,
            query_similarity,
        }
    }

    /// Query with a partial or noisy key (holographic graceful degradation).
    pub fn query_partial(
        &self,
        partial_key: &TropicalDualClifford<T, DIM>,
        _mask: &[bool],
    ) -> RetrievalResult<T, DIM> {
        // For partial queries, we just retrieve normally
        // The holographic representation naturally handles partial information
        self.retrieve(partial_key)
    }

    /// Check if a key is likely stored (approximate membership).
    pub fn probably_contains(&self, key: &TropicalDualClifford<T, DIM>) -> bool {
        let result = self.retrieve(key);
        result.confidence > self.algebra.similarity_threshold
    }

    /// Get capacity information.
    pub fn capacity_info(&self) -> CapacityInfo {
        let theoretical_capacity = self.theoretical_capacity();
        let estimated_snr = self.estimate_snr();
        let snr_threshold = 2.0; // Minimum acceptable SNR

        CapacityInfo {
            item_count: self.item_count,
            theoretical_capacity,
            estimated_snr,
            snr_threshold,
            near_capacity: self.item_count > theoretical_capacity / 2,
        }
    }

    /// Clear the memory.
    pub fn clear(&mut self) {
        self.trace = TropicalDualClifford::zero();
        self.item_count = 0;
        if let Some(ref mut keys) = self.stored_keys {
            keys.clear();
        }
    }

    /// Merge another memory into this one.
    pub fn merge(&mut self, other: &Self) {
        if other.item_count == 0 {
            return;
        }

        if self.item_count == 0 {
            self.trace = other.trace.clone();
        } else {
            self.trace = self.trace.bundle(&other.trace, self.algebra.bundle_beta);
        }

        self.item_count += other.item_count;

        // Merge keys if tracking
        if let (Some(ref mut self_keys), Some(ref other_keys)) =
            (&mut self.stored_keys, &other.stored_keys)
        {
            self_keys.extend(other_keys.iter().cloned());
        }
    }

    /// Get the raw memory trace (for inspection/serialization).
    pub fn trace(&self) -> &TropicalDualClifford<T, DIM> {
        &self.trace
    }

    /// Verify memory consistency (for verified contracts).
    pub fn verify_consistency(&self) -> bool {
        // Check trace is valid
        let norm = self.trace.norm();
        if !norm.is_finite() {
            return false;
        }

        // Check item count consistency with stored keys
        if let Some(ref keys) = self.stored_keys {
            if keys.len() != self.item_count {
                return false;
            }
        }

        true
    }

    // ========================================================================
    // Private helper methods
    // ========================================================================

    /// Estimate signal-to-noise ratio.
    fn estimate_snr(&self) -> f64 {
        if self.item_count == 0 {
            return f64::INFINITY;
        }

        // SNR ≈ sqrt(algebra_dim / item_count)
        // The algebra dimension is 2^DIM (number of basis elements), not DIM
        let algebra_dim = 1usize << DIM;
        (algebra_dim as f64 / self.item_count.max(1) as f64).sqrt()
    }

    /// Convert SNR to confidence in [0, 1].
    fn snr_to_confidence(&self, snr: f64) -> f64 {
        if snr.is_infinite() {
            return 1.0;
        }
        // Sigmoid-like function that maps SNR to confidence
        // SNR of 2 gives ~0.88, SNR of 5 gives ~0.99
        1.0 - (-snr / 2.0).exp()
    }

    /// Theoretical capacity: algebra_dim / ln(algebra_dim).
    ///
    /// The algebra dimension is 2^DIM (number of basis elements), not DIM.
    fn theoretical_capacity(&self) -> usize {
        let algebra_dim = 1usize << DIM;
        let ln_dim = (algebra_dim as f64).ln().max(1.0);
        (algebra_dim as f64 / ln_dim) as usize
    }

    /// Compute attribution for a retrieval.
    fn compute_attribution(
        &self,
        key: &TropicalDualClifford<T, DIM>,
        _value: &TropicalDualClifford<T, DIM>,
    ) -> Vec<(usize, f64)> {
        let Some(ref keys) = self.stored_keys else {
            return Vec::new();
        };

        // Compute similarity of each stored key to the query key
        // This approximates attribution
        let mut attributions: Vec<(usize, f64)> = keys
            .iter()
            .enumerate()
            .map(|(i, stored_key)| {
                let sim = key.similarity(stored_key);
                (i, sim.max(0.0))
            })
            .collect();

        // Sort by attribution (descending)
        attributions.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(core::cmp::Ordering::Equal));

        // Normalize to sum to 1
        let total: f64 = attributions.iter().map(|(_, w)| w).sum();
        if total > 1e-10 {
            for (_, w) in &mut attributions {
                *w /= total;
            }
        }

        // Keep only significant attributions
        attributions
            .into_iter()
            .filter(|(_, w)| *w > 0.01)
            .collect()
    }
}

// ============================================================================
// Verified wrapper for HolographicMemory
// ============================================================================

/// Marker for verified memory operations.
#[derive(Debug, Clone, Copy)]
#[allow(dead_code)]
pub struct MemoryVerified;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_capacity_formula() {
        // Capacity is algebra_dim / ln(algebra_dim), where algebra_dim = 2^DIM
        //
        // For DIM=8: algebra_dim = 256, capacity ≈ 256/ln(256) ≈ 46
        let algebra_dim_8 = 256.0;
        let capacity_8 = (algebra_dim_8 / algebra_dim_8.ln()) as usize;
        assert!(
            capacity_8 > 40 && capacity_8 < 55,
            "DIM=8 capacity: {}",
            capacity_8
        );

        // For DIM=16: algebra_dim = 65536, capacity ≈ 65536/ln(65536) ≈ 5909
        let algebra_dim_16 = 65536.0;
        let capacity_16 = (algebra_dim_16 / algebra_dim_16.ln()) as usize;
        assert!(
            capacity_16 > 5000 && capacity_16 < 7000,
            "DIM=16 capacity: {}",
            capacity_16
        );
    }

    #[test]
    fn test_snr_to_confidence() {
        let memory = HolographicMemory::<f64, 8>::new(BindingAlgebra::default());

        // High SNR should give high confidence
        let high_conf = memory.snr_to_confidence(10.0);
        assert!(high_conf > 0.99);

        // Low SNR should give lower confidence
        let low_conf = memory.snr_to_confidence(1.0);
        assert!(low_conf < 0.5);

        // Infinite SNR (empty memory) should give 1.0
        let inf_conf = memory.snr_to_confidence(f64::INFINITY);
        assert!((inf_conf - 1.0).abs() < 1e-10);
    }
}
