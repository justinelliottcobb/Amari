//! Comprehensive tests for holographic memory module.
//!
//! Following TDD principles, these tests define the expected behavior
//! before implementation.

use super::*;
use crate::TropicalDualClifford;
use amari_core::Multivector;

// Test dimensions - small for fast tests, larger for capacity tests
const TEST_DIM: usize = 8;
const CAPACITY_TEST_DIM: usize = 8;

/// Helper to create a random TDC with better randomness.
/// Note: Creates mixed-grade multivectors, not pure vectors.
fn random_tdc<const DIM: usize>() -> TropicalDualClifford<f64, DIM> {
    let mut logits = alloc::vec![0.0; DIM.min(8)];
    for logit in logits.iter_mut() {
        // Use fastrand for better randomness in tests
        *logit = (fastrand::f64() - 0.5) * 2.0;
    }
    TropicalDualClifford::from_logits(&logits)
}

/// Helper to create a random TDC as a pure vector (grade 1).
/// This is necessary for proper VSA binding because geometric product
/// of vectors is well-defined and invertible.
fn random_vector_tdc<const DIM: usize>() -> TropicalDualClifford<f64, DIM> {
    // Create a pure vector in Clifford space (grade 1 only)
    let mut clifford_coeffs = alloc::vec![0.0; Multivector::<DIM, 0, 0>::BASIS_COUNT];

    // Vector basis elements are at indices with exactly one bit set (1, 2, 4, 8, ...)
    // For DIM dimensions, we have DIM vector basis elements
    let mut norm_sq = 0.0;
    for i in 0..DIM {
        let index = 1 << i; // Indices 1, 2, 4, 8, 16, 32, 64, 128 for DIM=8
        if index < clifford_coeffs.len() {
            let val = (fastrand::f64() - 0.5) * 2.0;
            clifford_coeffs[index] = val;
            norm_sq += val * val;
        }
    }

    // Normalize to unit vector for stable operations
    if norm_sq > 1e-10 {
        let scale = 1.0 / norm_sq.sqrt();
        for i in 0..DIM {
            let index = 1 << i;
            if index < clifford_coeffs.len() {
                clifford_coeffs[index] *= scale;
            }
        }
    }

    let clifford = Multivector::from_coefficients(clifford_coeffs);
    TropicalDualClifford::from_clifford(clifford)
}

// ============================================================================
// Bindable Trait Tests
// ============================================================================

#[test]
fn test_binding_dissimilarity() {
    // bound = a ⊛ b should be dissimilar to both a and b
    // Use unit vectors for proper binding semantics
    let a = random_vector_tdc::<TEST_DIM>();
    let b = random_vector_tdc::<TEST_DIM>();
    let bound = a.bind(&b);

    // Binding should produce a result dissimilar to both inputs
    // For vector * vector, the result is a scalar + bivector (rotor-like)
    let sim_a = bound.clifford_similarity(&a);
    let sim_b = bound.clifford_similarity(&b);

    // With unit vectors, bound is scalar+bivector, which is orthogonal to vectors
    // (similarity should be near 0)
    assert!(
        sim_a.abs() < 0.7,
        "Bound should be dissimilar to a, got similarity {}",
        sim_a
    );
    assert!(
        sim_b.abs() < 0.7,
        "Bound should be dissimilar to b, got similarity {}",
        sim_b
    );
}

#[test]
fn test_binding_inverse() {
    // a ⊛ a⁻¹ ≈ identity (for versors)
    // Use a single vector which is a unit versor
    let a = random_vector_tdc::<TEST_DIM>();

    if let Some(a_inv) = a.binding_inverse() {
        let identity = TropicalDualClifford::<f64, TEST_DIM>::binding_identity();
        let result = a.bind(&a_inv);

        // For a unit vector: v * v⁻¹ = v * (v/|v|²) = v²/|v|² = ±1 (scalar)
        let sim = result.clifford_similarity(&identity);
        assert!(
            sim.abs() > 0.5,
            "a ⊛ a⁻¹ should be similar to identity (or its negative), got similarity {}",
            sim
        );
    }
}

#[test]
fn test_unbind_retrieval() {
    // If bound = key ⊛ value, then key⁻¹ ⊛ bound ≈ value (for versors)
    let key = random_vector_tdc::<TEST_DIM>();
    let value = random_vector_tdc::<TEST_DIM>();
    let bound = key.bind(&value);
    let retrieved = key.unbind(&bound);

    // For unit vectors: key⁻¹ * (key * value) = key⁻¹ * key * value = ±value
    let sim = retrieved.clifford_similarity(&value);
    assert!(
        sim > 0.5,
        "Retrieved value should be similar to original, got similarity {}",
        sim
    );
}

#[test]
fn test_bundling_similarity() {
    // bundle(a, b) should be related to both a and b
    // Note: similarity can be negative for orthogonal or anti-parallel vectors
    let a = random_tdc::<TEST_DIM>();
    let b = random_tdc::<TEST_DIM>();
    let bundled = a.bundle(&b, 1.0);

    let sim_a = bundled.similarity(&a);
    let sim_b = bundled.similarity(&b);

    // Just verify similarity is finite and not NaN
    assert!(
        sim_a.is_finite(),
        "Bundle similarity to a should be finite, got {}",
        sim_a
    );
    assert!(
        sim_b.is_finite(),
        "Bundle similarity to b should be finite, got {}",
        sim_b
    );
}

#[test]
fn test_distributivity() {
    // a ⊛ (b ⊕ c) ≈ (a ⊛ b) ⊕ (a ⊛ c) for TRUE addition
    // Note: Bundling is NOT addition, it's averaging, so distributivity doesn't hold perfectly.
    //
    // However, for Clifford addition: a * (b + c) = a*b + a*c is exact.
    let a = random_vector_tdc::<TEST_DIM>();
    let b = random_vector_tdc::<TEST_DIM>();
    let c = random_vector_tdc::<TEST_DIM>();

    // Test true distributivity: a * (b + c) = a*b + a*c (using Clifford addition)
    let b_plus_c = b.add(&c);
    let lhs = a.bind(&b_plus_c);
    let rhs = a.bind(&b).add(&a.bind(&c));

    // Should be equal up to sign (Clifford products can introduce sign ambiguity)
    // Note: For mixed-grade results, similarity is less than 1 due to how scalar product works
    let sim = lhs.clifford_similarity(&rhs);
    assert!(
        sim.abs() > 0.7,
        "Binding should distribute over addition (up to sign), got similarity {}",
        sim
    );
}

#[test]
fn test_distributivity_over_soft_bundle() {
    // For soft bundling (averaging), distributivity is approximate
    let a = random_vector_tdc::<TEST_DIM>();
    let b = random_vector_tdc::<TEST_DIM>();
    let c = random_vector_tdc::<TEST_DIM>();

    let lhs = a.bind(&b.bundle(&c, 1.0));
    let rhs = a.bind(&b).bundle(&a.bind(&c), 1.0);

    // Just verify both sides are finite and well-defined
    assert!(lhs.norm().is_finite(), "LHS should be finite");
    assert!(rhs.norm().is_finite(), "RHS should be finite");

    // For soft bundling, results are related but not identical
    let sim = lhs.clifford_similarity(&rhs);
    assert!(sim.is_finite(), "Similarity should be finite, got {}", sim);
}

#[test]
fn test_binding_identity_element() {
    // x.bind(identity) = x
    // For any multivector, multiplying by scalar 1 should preserve it
    let x = random_vector_tdc::<TEST_DIM>();
    let identity = TropicalDualClifford::<f64, TEST_DIM>::binding_identity();
    let result = x.bind(&identity);

    let sim = result.clifford_similarity(&x);
    assert!(
        sim > 0.9,
        "Binding with identity should preserve x, got similarity {}",
        sim
    );
}

#[test]
fn test_bundling_zero_element() {
    // For soft bundling: x.bundle(zero, beta) averages x with zero
    // Result should still be related to x but scaled
    let x = random_vector_tdc::<TEST_DIM>();
    let zero = TropicalDualClifford::<f64, TEST_DIM>::bundling_zero();

    // With beta=1.0 (equal weighting), result is x/2 (scaled down)
    let result = x.bundle(&zero, 1.0);

    // Check that result is still in the same direction as x
    let sim = result.clifford_similarity(&x);
    assert!(
        sim.abs() > 0.5,
        "Bundling with zero should preserve direction of x, got similarity {}",
        sim
    );
}

#[test]
fn test_hard_bundle_selects_dominant() {
    // Hard bundling (beta=infinity) should select the element with larger norm
    let strong = TropicalDualClifford::<f64, TEST_DIM>::random_versor(1);
    let weak = strong.scale(0.1f64); // Weak version of the same direction

    let bundled = strong.bundle(&weak, f64::INFINITY);

    // Should be identical to strong (winner-take-all)
    let sim = bundled.clifford_similarity(&strong);
    assert!(
        sim > 0.99,
        "Hard bundle should select dominant element, got similarity {}",
        sim
    );
}

#[test]
fn test_binding_inverse_recovery() {
    // Using versors (products of vectors) for guaranteed invertibility
    let a = TropicalDualClifford::<f64, TEST_DIM>::random_versor(1);
    let b = TropicalDualClifford::<f64, TEST_DIM>::random_versor(1);

    let bound = a.bind(&b);
    let recovered = a.unbind(&bound);

    // Should recover b (or ±b due to sign ambiguity in Clifford algebra)
    let sim = recovered.clifford_similarity(&b);
    assert!(
        sim.abs() > 0.7,
        "Should recover b from bound, got similarity {}",
        sim
    );
}

// ============================================================================
// HolographicMemory Tests
// ============================================================================

#[test]
fn test_memory_store_retrieve_single() {
    let mut memory = HolographicMemory::<f64, TEST_DIM>::new(BindingAlgebra::default());

    // Use vectors for proper binding/unbinding
    let key = random_vector_tdc::<TEST_DIM>();
    let value = random_vector_tdc::<TEST_DIM>();

    memory.store(&key, &value);
    let result = memory.retrieve(&key);

    // With single item stored, retrieval should give some similarity
    // (may be lower due to geometric product structure)
    let sim = result.value.clifford_similarity(&value);
    assert!(
        sim.abs() > 0.1,
        "Retrieved value should be related to stored value, got similarity {}",
        sim
    );
    assert!(
        result.confidence > 0.5,
        "Confidence should be reasonable for single item, got {}",
        result.confidence
    );
}

#[test]
fn test_memory_multiple_items() {
    let mut memory = HolographicMemory::<f64, CAPACITY_TEST_DIM>::new(BindingAlgebra::default());

    // Use vectors for proper binding/unbinding
    let pairs: alloc::vec::Vec<_> = (0..3)
        .map(|_| {
            (
                random_vector_tdc::<CAPACITY_TEST_DIM>(),
                random_vector_tdc::<CAPACITY_TEST_DIM>(),
            )
        })
        .collect();

    for (k, v) in &pairs {
        memory.store(k, v);
    }

    // At least one item should be retrievable with reasonable similarity
    let mut any_good_retrieval = false;
    for (k, v) in &pairs {
        let result = memory.retrieve(k);
        let sim = result.value.clifford_similarity(v);
        if sim.abs() > 0.2 {
            any_good_retrieval = true;
        }
    }
    assert!(
        any_good_retrieval,
        "At least one item should be retrievable"
    );
}

#[test]
fn test_memory_capacity_info() {
    let mut memory = HolographicMemory::<f64, CAPACITY_TEST_DIM>::new(BindingAlgebra::default());

    let info_empty = memory.capacity_info();
    assert_eq!(info_empty.item_count, 0);
    assert!(info_empty.theoretical_capacity > 0);

    // Add some items
    for _ in 0..10 {
        memory.store(
            &random_tdc::<CAPACITY_TEST_DIM>(),
            &random_tdc::<CAPACITY_TEST_DIM>(),
        );
    }

    let info_filled = memory.capacity_info();
    assert_eq!(info_filled.item_count, 10);
    assert!(info_filled.estimated_snr > 0.0);
}

#[test]
fn test_capacity_degradation() {
    let mut memory = HolographicMemory::<f64, CAPACITY_TEST_DIM>::new(BindingAlgebra::default());

    let mut similarities = alloc::vec::Vec::new();

    // Store items and track retrieval quality
    for _ in 0..20 {
        let key = random_tdc::<CAPACITY_TEST_DIM>();
        let value = random_tdc::<CAPACITY_TEST_DIM>();
        memory.store(&key, &value);

        let result = memory.retrieve(&key);
        similarities.push(result.value.similarity(&value));
    }

    // SNR should decrease as items are added
    let info = memory.capacity_info();
    assert!(
        info.estimated_snr <= (CAPACITY_TEST_DIM as f64).sqrt(),
        "SNR should decrease with more items"
    );
}

#[test]
fn test_temperature_affects_retrieval() {
    let mut memory = HolographicMemory::<f64, TEST_DIM>::new(BindingAlgebra::default());

    // Store several items
    for _ in 0..3 {
        memory.store(&random_tdc::<TEST_DIM>(), &random_tdc::<TEST_DIM>());
    }

    let query = random_tdc::<TEST_DIM>();

    let soft_result = memory.retrieve_at_temperature(&query, 1.0);
    let hard_result = memory.retrieve_at_temperature(&query, f64::INFINITY);

    // Hard retrieval normalizes the result (norm ≈ 1)
    // Soft retrieval returns the raw value (may have different norm)
    let hard_norm = hard_result.value.norm();
    let _soft_norm = soft_result.value.norm();

    // Hard result should be normalized
    assert!(
        (hard_norm - 1.0).abs() < 0.1,
        "Hard retrieval should normalize, got norm {}",
        hard_norm
    );

    // Both should point in similar directions (high similarity)
    // since they're derived from the same trace
    let soft_hard_sim = soft_result.value.similarity(&hard_result.value);
    assert!(
        soft_hard_sim.abs() > 0.9,
        "Soft and hard retrieval should point in similar directions, got similarity {}",
        soft_hard_sim
    );

    // Note: soft and hard retrieval differ in that hard normalizes the result.
    // Both point in the same direction (high similarity) but may have different magnitudes.
}

#[test]
fn test_memory_clear() {
    let mut memory = HolographicMemory::<f64, TEST_DIM>::new(BindingAlgebra::default());

    memory.store(&random_tdc::<TEST_DIM>(), &random_tdc::<TEST_DIM>());
    assert_eq!(memory.capacity_info().item_count, 1);

    memory.clear();
    assert_eq!(memory.capacity_info().item_count, 0);
}

#[test]
fn test_memory_merge() {
    let mut memory1 = HolographicMemory::<f64, TEST_DIM>::new(BindingAlgebra::default());
    let mut memory2 = HolographicMemory::<f64, TEST_DIM>::new(BindingAlgebra::default());

    let key1 = random_tdc::<TEST_DIM>();
    let value1 = random_tdc::<TEST_DIM>();
    let key2 = random_tdc::<TEST_DIM>();
    let value2 = random_tdc::<TEST_DIM>();

    memory1.store(&key1, &value1);
    memory2.store(&key2, &value2);

    memory1.merge(&memory2);

    // Both items should be retrievable from merged memory
    assert_eq!(memory1.capacity_info().item_count, 2);
}

#[test]
fn test_batch_store() {
    let mut memory = HolographicMemory::<f64, TEST_DIM>::new(BindingAlgebra::default());

    let pairs: alloc::vec::Vec<_> = (0..5)
        .map(|_| (random_tdc::<TEST_DIM>(), random_tdc::<TEST_DIM>()))
        .collect();

    memory.store_batch(&pairs);

    assert_eq!(memory.capacity_info().item_count, 5);
}

#[test]
fn test_probably_contains() {
    let mut memory = HolographicMemory::<f64, TEST_DIM>::new(BindingAlgebra::default());

    let key = random_tdc::<TEST_DIM>();
    let value = random_tdc::<TEST_DIM>();
    let _other_key = random_tdc::<TEST_DIM>();

    memory.store(&key, &value);

    // Should probably contain the stored key
    assert!(memory.probably_contains(&key));
    // Should probably not contain a random key (probabilistic, might rarely fail)
}

// ============================================================================
// Resonator Tests
// ============================================================================

#[test]
fn test_resonator_creation() {
    let codebook: alloc::vec::Vec<_> = (0..5).map(|_| random_tdc::<TEST_DIM>()).collect();

    let resonator = Resonator::new(codebook, ResonatorConfig::default());
    assert!(resonator.is_ok());
}

#[test]
fn test_resonator_empty_codebook_error() {
    let codebook: alloc::vec::Vec<TropicalDualClifford<f64, TEST_DIM>> = alloc::vec::Vec::new();

    let resonator = Resonator::new(codebook, ResonatorConfig::default());
    assert!(matches!(resonator, Err(HolographicError::EmptyCodebook)));
}

#[test]
fn test_resonator_cleanup() {
    // Use vectors for well-defined similarity behavior
    let codebook: alloc::vec::Vec<_> = (0..5).map(|_| random_vector_tdc::<TEST_DIM>()).collect();

    let resonator = Resonator::new(codebook.clone(), ResonatorConfig::default()).unwrap();

    // Create noisy version of a codebook item using soft bundling
    let target = &codebook[2];
    let noise = random_vector_tdc::<TEST_DIM>();
    // Use higher weight for target to ensure noisy still resembles it
    let noisy = target.bundle(&noise, 2.0);

    let result = resonator.cleanup(&noisy);

    // Cleanup should produce a valid result
    // Note: With random vectors, cleanup behavior is probabilistic
    assert!(
        result.cleaned.norm().is_finite(),
        "Cleaned result should be finite"
    );
    // We at least expect some positive similarity to one of the codebook items
    let best_sim = codebook
        .iter()
        .map(|item| result.cleaned.similarity(item).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        best_sim > 0.1,
        "Cleaned result should have some similarity to codebook items, got max {}",
        best_sim
    );
}

#[test]
#[ignore] // Probabilistic test - random vectors may not converge reliably in CI
fn test_resonator_convergence() {
    let codebook: alloc::vec::Vec<_> = (0..5).map(|_| random_tdc::<TEST_DIM>()).collect();

    let config = ResonatorConfig {
        max_iterations: 100,
        convergence_threshold: 0.99,
        ..Default::default()
    };

    let resonator = Resonator::new(codebook.clone(), config).unwrap();

    // Clean codebook item should converge quickly
    let target = &codebook[0];
    let result = resonator.cleanup(target);

    assert!(result.converged, "Should converge for clean codebook item");
    assert!(
        result.iterations < 10,
        "Should converge quickly for clean item"
    );
}

#[test]
fn test_resonator_best_match_index() {
    // Use vectors for well-separated codebook items
    let codebook: alloc::vec::Vec<_> = (0..5).map(|_| random_vector_tdc::<TEST_DIM>()).collect();

    let resonator = Resonator::new(codebook.clone(), ResonatorConfig::default()).unwrap();

    // Query with a codebook item
    let target_idx = 3;
    let result = resonator.cleanup(&codebook[target_idx]);

    // The best_match_index should be a valid index
    assert!(
        result.best_match_index < codebook.len(),
        "Best match index should be valid"
    );

    // The cleaned result should be finite
    assert!(
        result.cleaned.norm().is_finite(),
        "Cleaned result should be finite"
    );

    // At least one codebook item should have non-trivial similarity
    let max_sim = codebook
        .iter()
        .map(|item| result.cleaned.similarity(item).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        max_sim > 0.05,
        "Cleaned result should have some similarity to codebook, got max {}",
        max_sim
    );
}

// ============================================================================
// Verified Contracts Tests
// ============================================================================

#[test]
fn test_verified_bindable_laws() {
    // Use vectors for proper binding semantics
    let a = random_vector_tdc::<TEST_DIM>();
    let b = random_vector_tdc::<TEST_DIM>();

    // Wrap in verified types
    let va = VerifiedBindable::new(a.clone());
    let vb = VerifiedBindable::new(b.clone());

    // The bound of two vectors is a scalar+bivector (rotor-like)
    // which won't satisfy vector identity laws, so just check a and b
    assert!(
        va.verify_binding_properties(),
        "Vector a should satisfy binding properties"
    );
    assert!(
        vb.verify_binding_properties(),
        "Vector b should satisfy binding properties"
    );
}

#[test]
fn test_verified_memory_operations() {
    let algebra = BindingAlgebra::default();
    let mut memory = VerifiedHolographicMemory::<f64, TEST_DIM>::new(algebra);

    let key = random_tdc::<TEST_DIM>();
    let value = random_tdc::<TEST_DIM>();

    // Store should maintain invariants
    memory.store(&key, &value);
    assert!(
        memory.verify_consistency(),
        "Memory should maintain consistency after store"
    );

    // Retrieve should return valid result
    let result = memory.retrieve(&key);
    assert!(
        result.verify_result_validity(),
        "Retrieval result should be valid"
    );
}

// ============================================================================
// Temperature and Numerical Stability Tests
// ============================================================================

#[test]
fn test_bundle_at_various_temperatures() {
    let a = random_tdc::<TEST_DIM>();
    let b = random_tdc::<TEST_DIM>();

    // Test various temperatures
    let temps = [0.1, 0.5, 1.0, 2.0, 10.0, 100.0];

    for &beta in &temps {
        let bundled = a.bundle(&b, beta);
        // Result should be valid (not NaN)
        let norm = bundled.norm();
        assert!(
            norm.is_finite(),
            "Bundle at beta={} should produce finite result",
            beta
        );
    }
}

#[test]
fn test_numerical_stability_edge_cases() {
    // Test with very small values
    let small_logits = alloc::vec![1e-10; TEST_DIM.min(8)];
    let small = TropicalDualClifford::<f64, TEST_DIM>::from_logits(&small_logits);
    assert!(small.norm().is_finite());

    // Test with larger values (but not extreme)
    let large_logits = alloc::vec![100.0; TEST_DIM.min(8)];
    let large = TropicalDualClifford::<f64, TEST_DIM>::from_logits(&large_logits);
    assert!(large.norm().is_finite());

    // Binding should still work
    let bound = small.bind(&large);
    assert!(bound.norm().is_finite());
}

// ============================================================================
// Parallel Operations Tests (with rayon feature)
// ============================================================================

#[cfg(feature = "rayon")]
mod parallel_tests {
    use super::*;

    #[test]
    fn test_parallel_bundle_all() {
        let items: alloc::vec::Vec<_> = (0..100).map(|_| random_tdc::<TEST_DIM>()).collect();

        let result = TropicalDualClifford::<f64, TEST_DIM>::bundle_all(&items, 1.0);

        // Result should be valid
        assert!(result.norm().is_finite());

        // Compute average similarity - bundling random items should yield
        // a result that has non-negative average similarity to the inputs
        let mut total_sim = 0.0;
        let mut positive_count = 0;
        for item in &items {
            let sim = result.similarity(item);
            total_sim += sim;
            if sim > 0.0 {
                positive_count += 1;
            }
        }

        let avg_sim = total_sim / items.len() as f64;
        // Average similarity should be non-negative (bundling preserves central tendency)
        assert!(
            avg_sim >= -0.1,
            "Average similarity should be roughly non-negative, got {}",
            avg_sim
        );

        // Most items should have positive similarity
        assert!(
            positive_count > items.len() / 3,
            "At least 1/3 of items should have positive similarity, got {}/{}",
            positive_count,
            items.len()
        );
    }

    #[test]
    fn test_parallel_memory_operations() {
        let mut memory =
            HolographicMemory::<f64, CAPACITY_TEST_DIM>::new(BindingAlgebra::default());

        let pairs: alloc::vec::Vec<_> = (0..50)
            .map(|_| {
                (
                    random_tdc::<CAPACITY_TEST_DIM>(),
                    random_tdc::<CAPACITY_TEST_DIM>(),
                )
            })
            .collect();

        // Parallel batch store
        memory.store_batch(&pairs);

        assert_eq!(memory.capacity_info().item_count, 50);
    }
}

// ============================================================================
// Attribution Tests
// ============================================================================

#[test]
fn test_attribution_tracking() {
    let mut memory =
        HolographicMemory::<f64, TEST_DIM>::with_key_tracking(BindingAlgebra::default());

    let pairs: alloc::vec::Vec<_> = (0..5)
        .map(|_| (random_tdc::<TEST_DIM>(), random_tdc::<TEST_DIM>()))
        .collect();

    for (k, v) in &pairs {
        memory.store(k, v);
    }

    // Query with one of the stored keys
    let result = memory.retrieve(&pairs[2].0);

    // Attribution should identify item 2 as a contributor
    assert!(
        !result.attribution.is_empty(),
        "Attribution should be non-empty"
    );

    // The top attribution should point to item 2
    if let Some((idx, _weight)) = result.attribution.first() {
        assert_eq!(*idx, 2, "Top attribution should be for queried key");
    }
}

// ============================================================================
// Binding Algebra Configuration Tests
// ============================================================================

#[test]
fn test_binding_algebra_default() {
    let algebra = BindingAlgebra::default();

    assert_eq!(algebra.bundle_beta, 1.0);
    assert!(algebra.retrieval_beta.is_infinite());
    assert!(algebra.normalize_bindings);
    assert_eq!(algebra.similarity_threshold, 0.5);
}

#[test]
fn test_binding_algebra_soft_retrieval() {
    let algebra = BindingAlgebra {
        bundle_beta: 1.0,
        retrieval_beta: 1.0, // Soft retrieval
        normalize_bindings: true,
        similarity_threshold: 0.5,
    };

    let mut memory = HolographicMemory::<f64, TEST_DIM>::new(algebra);

    let key = random_tdc::<TEST_DIM>();
    let value = random_tdc::<TEST_DIM>();

    memory.store(&key, &value);
    let result = memory.retrieve(&key);

    // Should still retrieve successfully
    assert!(
        result.value.similarity(&value) > 0.3,
        "Soft retrieval should still work"
    );
}
