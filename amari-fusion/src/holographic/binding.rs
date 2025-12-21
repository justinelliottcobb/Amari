//! Core binding algebra for holographic representations.
//!
//! This module implements the fundamental operations for Vector Symbolic Architectures:
//! - **Binding**: Creates associations between representations (dissimilar to both inputs)
//! - **Bundling**: Superposition of multiple representations (similar to all inputs)
//! - **Inverse**: Enables unbinding to retrieve associated values

use alloc::vec::Vec;
use num_traits::Float;

#[cfg(feature = "rayon")]
use rayon::prelude::*;

use crate::TropicalDualClifford;
use amari_core::Multivector;
use amari_dual::DualNumber;
use amari_tropical::TropicalNumber;

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

impl<T: Float + Send + Sync, const DIM: usize> Bindable for TropicalDualClifford<T, DIM>
where
    T: num_traits::NumCast,
{
    fn bind(&self, other: &Self) -> Self {
        // Use geometric product in Clifford space
        let clifford_result = self.clifford().geometric_product(other.clifford());

        // For tropical: use addition (log-space multiplication)
        let mut tropical_coeffs = Vec::with_capacity(DIM.min(8));
        for i in 0..DIM.min(8) {
            if let (Ok(a), Ok(b)) = (self.tropical().get(i), other.tropical().get(i)) {
                // Tropical multiplication is addition in log-space
                tropical_coeffs.push(TropicalNumber::new(a.value() + b.value()));
            } else {
                tropical_coeffs.push(TropicalNumber::zero());
            }
        }

        // For dual: multiply (preserves derivatives)
        let mut dual_coeffs = Vec::with_capacity(DIM.min(8));
        for i in 0..DIM.min(8) {
            let a = self.dual().get(i);
            let b = other.dual().get(i);
            dual_coeffs.push(a * b);
        }

        TropicalDualClifford::from_components(tropical_coeffs, dual_coeffs, clifford_result)
    }

    fn binding_inverse(&self) -> Option<Self> {
        // For Clifford versors, inverse is x̃ / (x·x̃) where x̃ is the reverse
        let clifford = self.clifford();
        let reversed = clifford.reverse();
        let mag_sq = clifford.geometric_product(&reversed).scalar_part();

        // Check for near-zero magnitude
        const MIN_MAGNITUDE: f64 = 1e-10;
        if mag_sq.abs() < MIN_MAGNITUDE {
            return None;
        }

        let scale = 1.0 / mag_sq;
        let clifford_inv = reversed * scale;

        // For tropical: negate (log-space reciprocal)
        let mut tropical_coeffs = Vec::with_capacity(DIM.min(8));
        for i in 0..DIM.min(8) {
            if let Ok(val) = self.tropical().get(i) {
                tropical_coeffs.push(TropicalNumber::new(-val.value()));
            } else {
                tropical_coeffs.push(TropicalNumber::zero());
            }
        }

        // For dual: compute inverse (1/x with derivatives)
        let mut dual_coeffs = Vec::with_capacity(DIM.min(8));
        for i in 0..DIM.min(8) {
            let d = self.dual().get(i);
            if d.real.abs() > T::from(MIN_MAGNITUDE).unwrap_or(T::epsilon()) {
                // d(1/x) = -1/x² dx
                let inv_real = T::one() / d.real;
                let inv_dual = -d.dual / (d.real * d.real);
                dual_coeffs.push(DualNumber::new(inv_real, inv_dual));
            } else {
                dual_coeffs.push(DualNumber::constant(T::zero()));
            }
        }

        Some(TropicalDualClifford::from_components(
            tropical_coeffs,
            dual_coeffs,
            clifford_inv,
        ))
    }

    fn bundle(&self, other: &Self, beta: f64) -> Self {
        if beta.is_infinite() {
            // Hard bundling: component-wise max
            self.bundle_hard(other)
        } else {
            // Soft bundling: logsumexp / weighted average
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
        let self_norm = self.clifford().norm();
        let other_norm = other.clifford().norm();

        if self_norm < 1e-10 || other_norm < 1e-10 {
            return 0.0;
        }

        // Compute inner product via symmetric scalar part of geometric product
        let product = self.clifford().geometric_product(other.clifford());
        let inner = product.scalar_part();

        // Normalize
        inner / (self_norm * other_norm)
    }

    fn binding_identity() -> Self {
        // The identity for geometric product is the scalar 1
        let mut clifford_coeffs = alloc::vec![0.0; Multivector::<DIM, 0, 0>::BASIS_COUNT];
        clifford_coeffs[0] = 1.0; // Scalar component

        let tropical_coeffs = (0..DIM.min(8))
            .map(|_| TropicalNumber::new(T::zero()))
            .collect();

        let dual_coeffs = (0..DIM.min(8))
            .map(|_| DualNumber::constant(T::one()))
            .collect();

        let clifford = Multivector::from_coefficients(clifford_coeffs);

        TropicalDualClifford::from_components(tropical_coeffs, dual_coeffs, clifford)
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

        let scale = T::from(1.0 / n).unwrap_or(T::one());
        self.scale(scale)
    }
}

// ============================================================================
// Helper methods for TropicalDualClifford bundling
// ============================================================================

impl<T: Float, const DIM: usize> TropicalDualClifford<T, DIM>
where
    T: num_traits::NumCast,
{
    /// Soft bundling using logsumexp-style aggregation.
    fn bundle_soft(&self, other: &Self, beta: f64) -> Self {
        // Weighted average for Clifford
        let clifford_sum = self.clifford().clone() + other.clifford().clone();
        let clifford_result = clifford_sum * 0.5;

        // For tropical: soft-max (logsumexp)
        let mut tropical_coeffs = Vec::with_capacity(DIM.min(8));
        for i in 0..DIM.min(8) {
            if let (Ok(a), Ok(b)) = (self.tropical().get(i), other.tropical().get(i)) {
                let a_val = a.value().to_f64().unwrap_or(0.0);
                let b_val = b.value().to_f64().unwrap_or(0.0);
                let soft_max = logsumexp(a_val, b_val, beta);
                tropical_coeffs.push(TropicalNumber::new(T::from(soft_max).unwrap_or(T::zero())));
            } else {
                tropical_coeffs.push(TropicalNumber::zero());
            }
        }

        // For dual: average
        let mut dual_coeffs = Vec::with_capacity(DIM.min(8));
        for i in 0..DIM.min(8) {
            let a = self.dual().get(i);
            let b = other.dual().get(i);
            let half = T::from(0.5).unwrap_or(T::one());
            dual_coeffs.push(DualNumber::new(
                (a.real + b.real) * half,
                (a.dual + b.dual) * half,
            ));
        }

        TropicalDualClifford::from_components(tropical_coeffs, dual_coeffs, clifford_result)
    }

    /// Hard bundling using max operations.
    fn bundle_hard(&self, other: &Self) -> Self {
        // Take element with larger norm for Clifford
        let self_norm = self.clifford().norm();
        let other_norm = other.clifford().norm();
        let clifford_result = if self_norm >= other_norm {
            self.clifford().clone()
        } else {
            other.clifford().clone()
        };

        // For tropical: max
        let mut tropical_coeffs = Vec::with_capacity(DIM.min(8));
        for i in 0..DIM.min(8) {
            if let (Ok(a), Ok(b)) = (self.tropical().get(i), other.tropical().get(i)) {
                let max_val = a.value().max(b.value());
                tropical_coeffs.push(TropicalNumber::new(max_val));
            } else {
                tropical_coeffs.push(TropicalNumber::zero());
            }
        }

        // For dual: take dominant
        let mut dual_coeffs = Vec::with_capacity(DIM.min(8));
        for i in 0..DIM.min(8) {
            let a = self.dual().get(i);
            let b = other.dual().get(i);
            if a.real.abs() >= b.real.abs() {
                dual_coeffs.push(a);
            } else {
                dual_coeffs.push(b);
            }
        }

        TropicalDualClifford::from_components(tropical_coeffs, dual_coeffs, clifford_result)
    }
}

/// Compute log-sum-exp with temperature: `(1/β) * ln(exp(β*a) + exp(β*b))`
///
/// This smoothly interpolates between max (β→∞) and average (β→0).
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
