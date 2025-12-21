//! Core binding algebra for holographic representations.
//!
//! This module implements the fundamental operations for Vector Symbolic Architectures:
//! - **Binding**: Creates associations between representations (dissimilar to both inputs)
//! - **Bundling**: Superposition of multiple representations (similar to all inputs)
//! - **Inverse**: Enables unbinding to retrieve associated values

use num_traits::Float;

#[cfg(feature = "rayon")]
use rayon::prelude::*;

use crate::TropicalDualClifford;
use amari_core::Multivector;
use amari_dual::DualNumber;

/// Trait for types that support holographic binding operations.
///
/// # Algebraic Laws
///
/// Implementations should satisfy:
/// - **Binding inverse**: `x.bind(x.binding_inverse()) ≈ identity`
/// - **Distributivity**: `a.bind(b.bundle(c)) ≈ a.bind(b).bundle(a.bind(c))`
/// - **Dissimilarity**: `a.bind(b)` should be dissimilar to both `a` and `b`
pub trait Bindable: Sized + Clone {
    /// Bind two representations (association/structure-creation).
    ///
    /// Result should be dissimilar to both inputs.
    /// Must distribute over bundling: `a.bind(b.bundle(c)) = a.bind(b).bundle(a.bind(c))`
    fn bind(&self, other: &Self) -> Self;

    /// Compute the binding inverse such that `x.bind(x.binding_inverse()) ≈ identity`.
    ///
    /// For Clifford algebra, this is the versor inverse: `x̃ / (x x̃)`
    /// Returns `None` if the inverse doesn't exist (magnitude too small).
    fn binding_inverse(&self) -> Option<Self>;

    /// Unbind: retrieve associated value.
    ///
    /// Equivalent to `self.binding_inverse().bind(other)`
    fn unbind(&self, other: &Self) -> Self {
        if let Some(inv) = self.binding_inverse() {
            inv.bind(other)
        } else {
            // Fallback: return other if inverse doesn't exist
            other.clone()
        }
    }

    /// Bundle two representations (superposition/aggregation).
    ///
    /// Result should be similar to both inputs.
    /// `beta` controls soft (1.0) vs hard (∞) bundling.
    fn bundle(&self, other: &Self, beta: f64) -> Self;

    /// Bundle multiple items efficiently.
    ///
    /// Uses parallel processing when the `rayon` feature is enabled.
    fn bundle_all(items: &[Self], beta: f64) -> Self
    where
        Self: Send + Sync;

    /// Similarity measure between two representations.
    ///
    /// Returns a value in `[-1, 1]` (normalized inner product).
    fn similarity(&self, other: &Self) -> f64;

    /// The identity element for binding (`x.bind(identity) = x`).
    fn binding_identity() -> Self;

    /// The zero element for bundling (`x.bundle(zero) ≈ x`).
    fn bundling_zero() -> Self;

    /// Compute the norm of this representation.
    fn norm(&self) -> f64;

    /// Normalize to unit norm.
    fn normalize(&self) -> Self;
}

/// Configuration for the binding algebra.
#[derive(Clone, Debug)]
pub struct BindingAlgebra {
    /// Temperature for bundling: 1.0 = soft (logsumexp), f64::INFINITY = hard (max)
    pub bundle_beta: f64,
    /// Temperature for retrieval
    pub retrieval_beta: f64,
    /// Whether to normalize after binding
    pub normalize_bindings: bool,
    /// Similarity threshold for "match" in retrieval
    pub similarity_threshold: f64,
}

impl Default for BindingAlgebra {
    fn default() -> Self {
        Self {
            bundle_beta: 1.0,              // soft encoding by default
            retrieval_beta: f64::INFINITY, // hard retrieval by default
            normalize_bindings: true,
            similarity_threshold: 0.5,
        }
    }
}

impl BindingAlgebra {
    /// Create algebra optimized for soft encoding and hard retrieval.
    pub fn soft_encode_hard_retrieve() -> Self {
        Self::default()
    }

    /// Create algebra with fully soft operations.
    pub fn all_soft() -> Self {
        Self {
            bundle_beta: 1.0,
            retrieval_beta: 1.0,
            normalize_bindings: true,
            similarity_threshold: 0.5,
        }
    }

    /// Create algebra with fully hard (tropical) operations.
    pub fn all_hard() -> Self {
        Self {
            bundle_beta: f64::INFINITY,
            retrieval_beta: f64::INFINITY,
            normalize_bindings: false,
            similarity_threshold: 0.5,
        }
    }
}

// ============================================================================
// Implementation of Bindable for TropicalDualClifford
// ============================================================================
//
// HYBRID APPROACH: Clifford algebra is the source of truth.
//
// The binding operation uses the Clifford geometric product, and the tropical
// representation is derived from Clifford grade magnitudes. This ensures:
// - Binding inverse law: x ⊛ x⁻¹ ≈ identity (via Clifford versor inverse)
// - Identity law: x ⊛ identity = x (via Clifford geometric product with scalar 1)
// - Tropical provides efficient "which grade dominates" view
// ============================================================================

impl<T: Float + Send + Sync, const DIM: usize> Bindable for TropicalDualClifford<T, DIM>
where
    T: num_traits::NumCast,
{
    fn bind(&self, other: &Self) -> Self {
        // PRIMARY OPERATION: Clifford geometric product
        // This is the algebraically correct binding operation
        let clifford_result = self.clifford().geometric_product(other.clifford());

        // Create result from Clifford (source of truth)
        let mut result = TropicalDualClifford::from_clifford(clifford_result);

        // Propagate dual derivatives using product rule
        // d(f·g) = f·dg + df·g
        for i in 0..DIM.min(8) {
            let a = self.dual().get(i);
            let b = other.dual().get(i);
            // Product rule for dual numbers
            let product_dual = a.real * b.dual + a.dual * b.real;
            let existing = result.dual().get(i);
            result
                .dual_mut()
                .set(i, DualNumber::new(existing.real, product_dual));
        }

        result
    }

    fn binding_inverse(&self) -> Option<Self> {
        // Use Clifford versor inverse: x⁻¹ = x̃ / (x·x̃)
        // This is the mathematically correct inverse for geometric product
        let clifford_inv = self.clifford().inverse()?;

        // Create result from Clifford inverse
        let mut result = TropicalDualClifford::from_clifford(clifford_inv);

        // Propagate dual derivatives using chain rule: d(1/x) = -dx/x²
        const MIN_MAGNITUDE: f64 = 1e-10;
        for i in 0..DIM.min(8) {
            let d = self.dual().get(i);
            if d.real.abs() > T::from(MIN_MAGNITUDE).unwrap_or(T::epsilon()) {
                let inv_dual = -d.dual / (d.real * d.real);
                let existing = result.dual().get(i);
                result
                    .dual_mut()
                    .set(i, DualNumber::new(existing.real, inv_dual));
            }
        }

        Some(result)
    }

    fn bundle(&self, other: &Self, beta: f64) -> Self {
        if beta.is_infinite() {
            // Hard bundling: select element with larger Clifford norm
            self.bundle_hard(other)
        } else {
            // Soft bundling: weighted average in Clifford space
            self.bundle_soft(other, beta)
        }
    }

    fn bundle_all(items: &[Self], beta: f64) -> Self
    where
        Self: Send + Sync,
    {
        if items.is_empty() {
            return Self::bundling_zero();
        }
        if items.len() == 1 {
            return items[0].clone();
        }

        #[cfg(feature = "rayon")]
        {
            // Parallel reduction using rayon
            items
                .par_iter()
                .cloned()
                .reduce(Self::bundling_zero, |a, b| a.bundle(&b, beta))
        }

        #[cfg(not(feature = "rayon"))]
        {
            // Sequential reduction
            let mut result = items[0].clone();
            for item in items.iter().skip(1) {
                result = result.bundle(item, beta);
            }
            result
        }
    }

    fn similarity(&self, other: &Self) -> f64 {
        // Use normalized inner product in Clifford space
        // Uses proper inner product: <A B̃>₀ / (|A| |B|)
        let self_norm = self.clifford().norm();
        let other_norm = other.clifford().norm();

        if self_norm < 1e-10 || other_norm < 1e-10 {
            return 0.0;
        }

        // Compute inner product using scalar product with reverse: <A B̃>₀
        let inner = self.clifford().scalar_product(&other.clifford().reverse());

        // Normalize
        inner / (self_norm * other_norm)
    }

    fn binding_identity() -> Self {
        // The identity for geometric product is the scalar 1
        let clifford = Multivector::<DIM, 0, 0>::scalar(1.0);

        // Create from Clifford (will derive tropical from grade magnitudes)
        TropicalDualClifford::from_clifford(clifford)
    }

    fn bundling_zero() -> Self {
        // The zero for bundling is the additive identity
        TropicalDualClifford::zero()
    }

    fn norm(&self) -> f64 {
        self.clifford().norm()
    }

    fn normalize(&self) -> Self {
        let n = self.norm();
        if n < 1e-10 {
            return self.clone();
        }

        // Normalize in Clifford space
        let normalized_clifford = self
            .clifford()
            .normalize()
            .unwrap_or_else(|| self.clifford().clone());

        // Create result from normalized Clifford
        let mut result = TropicalDualClifford::from_clifford(normalized_clifford);

        // Preserve dual structure (scale derivatives)
        let scale = T::from(1.0 / n).unwrap_or(T::one());
        for i in 0..DIM.min(8) {
            let d = self.dual().get(i);
            let existing = result.dual().get(i);
            result
                .dual_mut()
                .set(i, DualNumber::new(existing.real, d.dual * scale));
        }

        result
    }
}

// ============================================================================
// Helper methods for TropicalDualClifford bundling (Hybrid approach)
// ============================================================================

impl<T: Float + Send + Sync, const DIM: usize> TropicalDualClifford<T, DIM>
where
    T: num_traits::NumCast,
{
    /// Soft bundling using weighted average in Clifford space.
    ///
    /// The beta parameter controls the weighting:
    /// - beta = 1.0: Equal weighting (simple average)
    /// - beta > 1.0: Favor the element with larger norm
    fn bundle_soft(&self, other: &Self, beta: f64) -> Self {
        // Compute weights based on norms and beta
        let self_norm = self.clifford().norm();
        let other_norm = other.clifford().norm();

        // Softmax-style weighting based on norms
        let (w1, w2) = if beta <= 0.0 || (self_norm < 1e-10 && other_norm < 1e-10) {
            // Equal weighting
            (0.5, 0.5)
        } else {
            // Softmax weights: exp(beta * norm) / sum
            let max_norm = self_norm.max(other_norm);
            let exp1 = (beta * (self_norm - max_norm)).exp();
            let exp2 = (beta * (other_norm - max_norm)).exp();
            let sum = exp1 + exp2;
            (exp1 / sum, exp2 / sum)
        };

        // Weighted average in Clifford space
        let clifford_result = self.clifford().clone() * w1 + other.clifford().clone() * w2;

        // Create result from Clifford (source of truth)
        let mut result = TropicalDualClifford::from_clifford(clifford_result);

        // Weighted average of dual derivatives
        let w1_t = T::from(w1).unwrap_or(T::from(0.5).unwrap());
        let w2_t = T::from(w2).unwrap_or(T::from(0.5).unwrap());
        for i in 0..DIM.min(8) {
            let a = self.dual().get(i);
            let b = other.dual().get(i);
            let existing = result.dual().get(i);
            result.dual_mut().set(
                i,
                DualNumber::new(existing.real, a.dual * w1_t + b.dual * w2_t),
            );
        }

        result
    }

    /// Hard bundling: select element with larger Clifford norm.
    ///
    /// This is winner-take-all bundling, equivalent to tropical max
    /// applied to the overall representations.
    fn bundle_hard(&self, other: &Self) -> Self {
        let self_norm = self.clifford().norm();
        let other_norm = other.clifford().norm();

        // Winner-take-all: select the element with larger norm
        if self_norm >= other_norm {
            self.clone()
        } else {
            other.clone()
        }
    }
}

/// Compute log-sum-exp with temperature: `(1/β) * ln(exp(β*a) + exp(β*b))`
///
/// This smoothly interpolates between max (β→∞) and average (β→0).
/// Currently used in tests; may be used in future soft bundling implementations.
#[allow(dead_code)]
fn logsumexp(a: f64, b: f64, beta: f64) -> f64 {
    if beta.is_infinite() {
        return a.max(b);
    }
    if beta <= 0.0 {
        return (a + b) / 2.0;
    }

    // Numerically stable logsumexp
    let max_val = a.max(b);
    let sum = (beta * (a - max_val)).exp() + (beta * (b - max_val)).exp();
    max_val + sum.ln() / beta
}

/// Phantom type marker for verified binding properties.
#[derive(Debug, Clone, Copy)]
#[allow(dead_code)]
pub struct BindingVerified;

/// Phantom type marker for verified distributivity.
#[derive(Debug, Clone, Copy)]
#[allow(dead_code)]
pub struct DistributivityVerified;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_logsumexp_hard_limit() {
        let result = logsumexp(5.0, 3.0, f64::INFINITY);
        assert!((result - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_logsumexp_soft() {
        let result = logsumexp(5.0, 5.0, 1.0);
        // ln(2*exp(5)) = 5 + ln(2) ≈ 5.693
        assert!((result - (5.0 + 2.0_f64.ln())).abs() < 0.01);
    }

    #[test]
    fn test_binding_algebra_constructors() {
        let soft = BindingAlgebra::all_soft();
        assert_eq!(soft.retrieval_beta, 1.0);

        let hard = BindingAlgebra::all_hard();
        assert!(hard.retrieval_beta.is_infinite());
    }
}
