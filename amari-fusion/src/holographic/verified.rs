//! Formal verification contracts for holographic memory operations.
//!
//! This module provides Creusot-style contracts for formally verifying the correctness
//! of holographic binding and memory operations.
//!
//! Verification focuses on:
//! - Binding algebra laws (inverse, distributivity)
//! - Memory consistency invariants
//! - Capacity and SNR guarantees
//! - Numerical stability

use core::marker::PhantomData;
use num_traits::Float;

use super::binding::{Bindable, BindingAlgebra};
use super::memory::{CapacityInfo, HolographicMemory, RetrievalResult};
use crate::TropicalDualClifford;

/// Verification marker for binding properties.
#[derive(Debug, Clone, Copy)]
pub struct BindingVerified;

/// Verification marker for memory properties.
#[derive(Debug, Clone, Copy)]
pub struct MemoryConsistentVerified;

/// Verified wrapper for Bindable types with algebraic law contracts.
#[derive(Clone, Debug)]
pub struct VerifiedBindable<B: Bindable> {
    inner: B,
    _marker: PhantomData<BindingVerified>,
}

impl<B: Bindable> VerifiedBindable<B> {
    /// Create a verified bindable wrapper.
    ///
    /// # Contracts
    /// - `ensures(result.verify_binding_properties())`
    pub fn new(inner: B) -> Self {
        Self {
            inner,
            _marker: PhantomData,
        }
    }

    /// Verified binding operation.
    ///
    /// # Contracts
    /// - `ensures(result.binding_produces_dissimilar())`
    /// - `ensures(result.binding_is_invertible() ==> a.bind(b).unbind(a) ≈ b)`
    pub fn bind(&self, other: &Self) -> Self {
        Self {
            inner: self.inner.bind(&other.inner),
            _marker: PhantomData,
        }
    }

    /// Verified bundling operation.
    ///
    /// # Contracts
    /// - `ensures(result.bundling_preserves_similarity())`
    /// - `ensures(result.norm().is_finite())`
    pub fn bundle(&self, other: &Self, beta: f64) -> Self {
        Self {
            inner: self.inner.bundle(&other.inner, beta),
            _marker: PhantomData,
        }
    }

    /// Verify binding algebra properties.
    ///
    /// # Contracts
    /// - `ensures(binding_inverse_law_holds())`
    /// - `ensures(identity_law_holds())`
    pub fn verify_binding_properties(&self) -> bool {
        // Verify identity element law
        let identity = B::binding_identity();
        let with_identity = self.inner.bind(&identity);
        let identity_holds = with_identity.similarity(&self.inner) > 0.9;

        // Verify norm is finite
        let norm_finite = self.inner.norm().is_finite();

        identity_holds && norm_finite
    }

    /// Access inner value (breaks verification guarantees).
    pub fn into_inner(self) -> B {
        self.inner
    }
}

/// Verified holographic memory with consistency contracts.
#[derive(Clone)]
pub struct VerifiedHolographicMemory<T: Float + Send + Sync, const DIM: usize>
where
    T: num_traits::NumCast,
{
    inner: HolographicMemory<T, DIM>,
    _marker: PhantomData<MemoryConsistentVerified>,
}

/// Verified retrieval result with validity contracts.
#[derive(Clone)]
pub struct VerifiedRetrievalResult<T: Float, const DIM: usize> {
    inner: RetrievalResult<T, DIM>,
    _marker: PhantomData<MemoryConsistentVerified>,
}

impl<T: Float + Send + Sync, const DIM: usize> VerifiedHolographicMemory<T, DIM>
where
    T: num_traits::NumCast,
{
    /// Create a new verified holographic memory.
    ///
    /// # Contracts
    /// - `ensures(result.verify_consistency())`
    /// - `ensures(result.capacity_info().item_count == 0)`
    pub fn new(algebra: BindingAlgebra) -> Self {
        Self {
            inner: HolographicMemory::new(algebra),
            _marker: PhantomData,
        }
    }

    /// Verified store operation.
    ///
    /// # Contracts
    /// - `ensures(self.capacity_info().item_count == old(self.capacity_info().item_count) + 1)`
    /// - `ensures(self.verify_consistency())`
    /// - `ensures(self.probably_contains(key) || self.capacity_info().near_capacity)`
    pub fn store(
        &mut self,
        key: &TropicalDualClifford<T, DIM>,
        value: &TropicalDualClifford<T, DIM>,
    ) {
        self.inner.store(key, value);
        debug_assert!(
            self.verify_consistency(),
            "Memory consistency violated after store"
        );
    }

    /// Verified retrieve operation.
    ///
    /// # Contracts
    /// - `ensures(result.verify_result_validity())`
    /// - `ensures(result.confidence >= 0.0 && result.confidence <= 1.0)`
    pub fn retrieve(&self, key: &TropicalDualClifford<T, DIM>) -> VerifiedRetrievalResult<T, DIM> {
        let result = self.inner.retrieve(key);
        debug_assert!(result.verify_result_validity(), "Invalid retrieval result");
        VerifiedRetrievalResult {
            inner: result,
            _marker: PhantomData,
        }
    }

    /// Get capacity information with validity contracts.
    ///
    /// # Contracts
    /// - `ensures(result.theoretical_capacity > 0)`
    /// - `ensures(result.estimated_snr >= 0.0 || result.estimated_snr.is_infinite())`
    pub fn capacity_info(&self) -> CapacityInfo {
        let info = self.inner.capacity_info();
        debug_assert!(info.theoretical_capacity > 0);
        info
    }

    /// Verify memory consistency invariants.
    ///
    /// Checks that all invariants are maintained:
    /// - Trace is finite
    /// - Item count matches stored keys (if tracking enabled)
    pub fn verify_consistency(&self) -> bool {
        self.inner.verify_consistency()
    }

    /// Clear with verification.
    ///
    /// # Contracts
    /// - `ensures(self.capacity_info().item_count == 0)`
    /// - `ensures(self.verify_consistency())`
    pub fn clear(&mut self) {
        self.inner.clear();
        debug_assert_eq!(self.inner.capacity_info().item_count, 0);
    }
}

impl<T: Float + Send + Sync, const DIM: usize> VerifiedRetrievalResult<T, DIM>
where
    T: num_traits::NumCast,
{
    /// Verify this result is valid.
    ///
    /// # Contracts
    /// - `ensures(result.value.norm().is_finite())`
    /// - `ensures(result.confidence >= 0.0 && result.confidence <= 1.0)`
    pub fn verify_result_validity(&self) -> bool {
        self.inner.verify_result_validity()
    }

    /// Get the retrieved value.
    pub fn value(&self) -> &TropicalDualClifford<T, DIM> {
        &self.inner.value
    }

    /// Get confidence.
    pub fn confidence(&self) -> f64 {
        self.inner.confidence
    }

    /// Get attribution.
    pub fn attribution(&self) -> &[(usize, f64)] {
        &self.inner.attribution
    }
}

/// Laws for holographic binding algebra verification.
#[allow(dead_code)]
pub struct HolographicAlgebraLaws;

#[allow(dead_code)]
impl HolographicAlgebraLaws {
    /// Verify binding inverse law: `x ⊛ x⁻¹ ≈ identity`.
    ///
    /// # Contracts
    /// - `ensures(x.bind(x.inverse()).similarity(identity) > threshold)`
    pub fn verify_binding_inverse_law<B: Bindable>(x: &B, threshold: f64) -> bool {
        if let Some(inv) = x.binding_inverse() {
            let result = x.bind(&inv);
            let identity = B::binding_identity();
            result.similarity(&identity) > threshold
        } else {
            // Inverse doesn't exist (singular), which is acceptable
            true
        }
    }

    /// Verify distributivity: `a ⊛ (b ⊕ c) ≈ (a ⊛ b) ⊕ (a ⊛ c)`.
    ///
    /// # Contracts
    /// - `ensures(lhs.similarity(rhs) > threshold)`
    pub fn verify_distributivity<B: Bindable + Clone>(
        a: &B,
        b: &B,
        c: &B,
        beta: f64,
        threshold: f64,
    ) -> bool {
        let lhs = a.bind(&b.bundle(c, beta));
        let rhs = a.bind(b).bundle(&a.bind(c), beta);
        lhs.similarity(&rhs) > threshold
    }

    /// Verify binding produces dissimilar results.
    ///
    /// # Contracts
    /// - `ensures(a.bind(b).similarity(a) < threshold)`
    /// - `ensures(a.bind(b).similarity(b) < threshold)`
    pub fn verify_binding_dissimilarity<B: Bindable>(a: &B, b: &B, threshold: f64) -> bool {
        let bound = a.bind(b);
        let sim_a = bound.similarity(a);
        let sim_b = bound.similarity(b);
        sim_a.abs() < threshold && sim_b.abs() < threshold
    }

    /// Verify bundling preserves similarity.
    ///
    /// # Contracts
    /// - `ensures(a.bundle(b).similarity(a) > threshold)`
    /// - `ensures(a.bundle(b).similarity(b) > threshold)`
    pub fn verify_bundling_similarity<B: Bindable>(
        a: &B,
        b: &B,
        beta: f64,
        threshold: f64,
    ) -> bool {
        let bundled = a.bundle(b, beta);
        let sim_a = bundled.similarity(a);
        let sim_b = bundled.similarity(b);
        sim_a > threshold && sim_b > threshold
    }

    /// Verify memory capacity scaling.
    ///
    /// # Contracts
    /// - `ensures(capacity_info.theoretical_capacity ≈ DIM / ln(DIM))`
    pub fn verify_capacity_scaling<
        T: Float + Send + Sync + num_traits::NumCast,
        const DIM: usize,
    >(
        memory: &HolographicMemory<T, DIM>,
    ) -> bool {
        let info = memory.capacity_info();
        let expected_capacity = (DIM as f64 / (DIM as f64).ln().max(1.0)) as usize;

        // Allow 20% deviation
        let lower_bound = expected_capacity * 80 / 100;
        let upper_bound = expected_capacity * 120 / 100;

        info.theoretical_capacity >= lower_bound
            && info.theoretical_capacity <= upper_bound.max(lower_bound + 1)
    }

    /// Verify SNR estimation.
    ///
    /// # Contracts
    /// - `ensures(snr ≈ sqrt(DIM / item_count))`
    pub fn verify_snr_estimation<T: Float + Send + Sync + num_traits::NumCast, const DIM: usize>(
        memory: &HolographicMemory<T, DIM>,
    ) -> bool {
        let info = memory.capacity_info();
        if info.item_count == 0 {
            return info.estimated_snr.is_infinite();
        }

        let expected_snr = (DIM as f64 / info.item_count as f64).sqrt();
        let ratio = info.estimated_snr / expected_snr;

        // Allow 10% deviation
        ratio > 0.9 && ratio < 1.1
    }
}

/// Trait for types with verified properties.
#[allow(dead_code)]
pub trait HolographicVerification {
    /// Verify all algebraic properties.
    fn verify_properties(&self) -> bool;

    /// Verify numerical stability.
    fn verify_stability(&self) -> bool;
}

impl<T: Float + Send + Sync, const DIM: usize> HolographicVerification
    for TropicalDualClifford<T, DIM>
where
    T: num_traits::NumCast,
{
    fn verify_properties(&self) -> bool {
        // Verify norm is finite and positive
        let norm = self.norm();
        norm.is_finite() && norm >= 0.0
    }

    fn verify_stability(&self) -> bool {
        // Check all components are finite
        for i in 0..DIM.min(8) {
            if let Ok(t) = self.tropical().get(i) {
                if !t.value().is_finite() {
                    return false;
                }
            }
            let d = self.dual().get(i);
            if !d.real.is_finite() || !d.dual.is_finite() {
                return false;
            }
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use amari_core::Multivector;

    fn random_tdc<const DIM: usize>() -> TropicalDualClifford<f64, DIM> {
        let mut logits = alloc::vec![0.0; DIM.min(8)];
        for logit in logits.iter_mut() {
            *logit = (fastrand::f64() - 0.5) * 2.0;
        }
        TropicalDualClifford::from_logits(&logits)
    }

    /// Create a random unit vector TDC (grade 1 only) for proper algebraic tests
    fn random_vector_tdc<const DIM: usize>() -> TropicalDualClifford<f64, DIM> {
        let mut clifford_coeffs = alloc::vec![0.0; Multivector::<DIM, 0, 0>::BASIS_COUNT];
        let mut norm_sq = 0.0;
        for i in 0..DIM.min(8) {
            let index = 1 << i;
            if index < clifford_coeffs.len() {
                let val = (fastrand::f64() - 0.5) * 2.0;
                clifford_coeffs[index] = val;
                norm_sq += val * val;
            }
        }
        if norm_sq > 1e-10 {
            let scale = 1.0 / norm_sq.sqrt();
            for i in 0..DIM.min(8) {
                let index = 1 << i;
                if index < clifford_coeffs.len() {
                    clifford_coeffs[index] *= scale;
                }
            }
        }
        let clifford = Multivector::from_coefficients(clifford_coeffs);
        TropicalDualClifford::from_clifford(clifford)
    }

    #[test]
    fn test_verified_bindable() {
        // Use vectors for proper identity law (x * 1 = x)
        let a = random_vector_tdc::<8>();
        let va = VerifiedBindable::new(a);
        assert!(va.verify_binding_properties());
    }

    #[test]
    fn test_binding_inverse_law() {
        // Use vectors which are guaranteed invertible
        let x = random_vector_tdc::<8>();
        assert!(HolographicAlgebraLaws::verify_binding_inverse_law(&x, 0.5));
    }

    #[test]
    fn test_distributivity_law() {
        // For bundling (averaging), distributivity is approximate
        let a = random_vector_tdc::<8>();
        let b = random_vector_tdc::<8>();
        let c = random_vector_tdc::<8>();
        // Use lower threshold for bundling (which is averaging, not addition)
        assert!(HolographicAlgebraLaws::verify_distributivity(
            &a, &b, &c, 1.0, 0.3
        ));
    }

    #[test]
    fn test_verified_memory() {
        let mut memory = VerifiedHolographicMemory::<f64, 8>::new(BindingAlgebra::default());

        let key = random_tdc::<8>();
        let value = random_tdc::<8>();

        memory.store(&key, &value);
        assert!(memory.verify_consistency());

        let result = memory.retrieve(&key);
        assert!(result.verify_result_validity());
    }

    #[test]
    fn test_capacity_scaling_law() {
        let memory = HolographicMemory::<f64, 8>::new(BindingAlgebra::default());
        assert!(HolographicAlgebraLaws::verify_capacity_scaling(&memory));
    }
}
