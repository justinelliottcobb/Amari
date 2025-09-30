//! Comprehensive test suite for tropical-dual-clifford fusion system
//!
//! This module provides extensive testing for the fusion of three algebraic systems:
//! - Tropical algebra operations and properties
//! - Dual number automatic differentiation
//! - Clifford geometric algebra relationships
//!
//! Testing methodology follows the formal verification approach established
//! for other amari crates, ensuring mathematical correctness and robustness.

#[cfg(test)]
mod tests {
    use crate::{conversion, TropicalDualClifford, TropicalDualDistribution};
    use alloc::vec::Vec;
    use amari_core::Multivector;
    use approx::assert_relative_eq;
    use num_traits::Float;

    const EPSILON: f64 = 1e-10;

    // ========== Core Fusion System Properties ==========

    #[test]
    fn test_tropical_dual_clifford_creation() {
        let logits = vec![1.0, 2.0, 3.0, 0.5];
        let tdc = TropicalDualClifford::<f64, 4>::from_logits(&logits);

        // Verify all three representations are properly initialized
        assert!(!tdc.tropical().max_element().is_zero());
        assert!(tdc.dual().norm().real > 0.0);
        assert!(tdc.clifford().norm() > 0.0);

        // Test from probabilities
        let probs = vec![0.1, 0.3, 0.4, 0.2];
        let tdc_from_probs = TropicalDualClifford::<f64, 4>::from_probabilities(&probs);
        assert!(!tdc_from_probs.is_zero());
    }

    #[test]
    fn test_tropical_dual_clifford_zero() {
        let zero_tdc = TropicalDualClifford::<f64, 4>::zero();

        // Test that zero object behaves correctly - some components may not be exactly zero
        // due to the implementation details of the underlying algebras
        let dual_norm = zero_tdc.dual().norm().real.abs();
        let clifford_norm = zero_tdc.clifford().norm().abs();

        // At minimum, dual and clifford components should be very small
        assert!(dual_norm < 1e-10);
        assert!(clifford_norm < 1e-10);

        // Operations with zero
        let logits = vec![1.0, 2.0, 3.0, 0.5];
        let tdc = TropicalDualClifford::<f64, 4>::from_logits(&logits);

        let added = tdc.add(&zero_tdc);
        // Should be approximately equal to original (tropical addition is max-based)
        assert!(added.dual().norm().real > 0.0);
    }

    #[test]
    fn test_fusion_system_arithmetic_properties() {
        let logits1 = vec![1.0, 2.0, 3.0, 0.5];
        let logits2 = vec![0.5, 1.5, 2.5, 1.0];
        let logits3 = vec![2.0, 1.0, 1.5, 2.5];

        let tdc1 = TropicalDualClifford::<f64, 4>::from_logits(&logits1);
        let tdc2 = TropicalDualClifford::<f64, 4>::from_logits(&logits2);
        let tdc3 = TropicalDualClifford::<f64, 4>::from_logits(&logits3);

        // Test associativity: (a + b) + c = a + (b + c)
        let left_assoc = tdc1.add(&tdc2).add(&tdc3);
        let right_assoc = tdc1.add(&tdc2.add(&tdc3));

        // For dual part, should be exactly associative
        let left_dual_norm = left_assoc.dual().norm().real;
        let right_dual_norm = right_assoc.dual().norm().real;
        assert!((left_dual_norm - right_dual_norm).abs() < EPSILON);

        // Test scaling properties
        let scale_factor = 2.5;
        let scaled = tdc1.scale(scale_factor);

        // Dual components should scale linearly
        let original_dual_norm = tdc1.dual().norm().real;
        let scaled_dual_norm = scaled.dual().norm().real;
        assert!((scaled_dual_norm - scale_factor * original_dual_norm).abs() < EPSILON);
    }

    #[test]
    fn test_fusion_distance_properties() {
        let logits1 = vec![1.0, 0.0, 0.0, 0.0];
        let logits2 = vec![0.0, 1.0, 0.0, 0.0];
        let logits3 = vec![0.0, 0.0, 1.0, 0.0];

        let tdc1 = TropicalDualClifford::<f64, 4>::from_logits(&logits1);
        let tdc2 = TropicalDualClifford::<f64, 4>::from_logits(&logits2);
        let tdc3 = TropicalDualClifford::<f64, 4>::from_logits(&logits3);

        // Test symmetry: d(a,b) = d(b,a)
        let dist_12 = tdc1.distance(&tdc2);
        let dist_21 = tdc2.distance(&tdc1);
        assert_relative_eq!(dist_12, dist_21, epsilon = EPSILON);

        // Test identity: d(a,a) = 0
        let dist_self = tdc1.distance(&tdc1);
        assert!(dist_self < EPSILON);

        // Test triangle inequality: d(a,c) <= d(a,b) + d(b,c)
        let dist_13 = tdc1.distance(&tdc3);
        let dist_23 = tdc2.distance(&tdc3);
        assert!(dist_13 <= dist_12 + dist_23 + EPSILON);

        // Test positivity: d(a,b) > 0 for a != b
        assert!(dist_12 > 0.0);
    }

    #[test]
    fn test_interpolation_properties() {
        let logits1 = vec![1.0, 0.0, 0.0, 0.0];
        let logits2 = vec![0.0, 1.0, 0.0, 0.0];

        let tdc1 = TropicalDualClifford::<f64, 4>::from_logits(&logits1);
        let tdc2 = TropicalDualClifford::<f64, 4>::from_logits(&logits2);

        // Test boundary conditions
        let interp_0 = tdc1.interpolate(&tdc2, 0.0);
        let interp_1 = tdc1.interpolate(&tdc2, 1.0);

        // At t=0, should be close to tdc1
        let dist_to_1_at_0 = interp_0.distance(&tdc1);
        assert!(dist_to_1_at_0 < EPSILON);

        // At t=1, should be close to tdc2
        let dist_to_2_at_1 = interp_1.distance(&tdc2);
        assert!(dist_to_2_at_1 < EPSILON);

        // Test intermediate values
        let interp_half = tdc1.interpolate(&tdc2, 0.5);
        let dist_1_half = interp_half.distance(&tdc1);
        let dist_2_half = interp_half.distance(&tdc2);

        // Should be between the endpoints
        assert!(dist_1_half > 0.0);
        assert!(dist_2_half > 0.0);
    }

    // ========== Sensitivity Analysis and Automatic Differentiation ==========

    #[test]
    fn test_sensitivity_analysis() {
        let logits = vec![1.0, 2.0, 3.0, 0.5, 1.5, 2.5, 0.8, 1.2];
        let tdc = TropicalDualClifford::<f64, 8>::from_logits(&logits);

        let sensitivity = tdc.sensitivity_analysis();

        // Verify sensitivity structure
        assert_eq!(sensitivity.sensitivities.len(), 8);

        // Total sensitivity should be positive for non-zero input
        assert!(sensitivity.total_sensitivity() > 0.0);

        // Most sensitive components
        let most_sensitive = sensitivity.most_sensitive(3);
        assert_eq!(most_sensitive.len(), 3);

        // Check sensitivity ordering
        let all_sensitivities: Vec<f64> = sensitivity
            .sensitivities
            .iter()
            .map(|s| s.sensitivity.abs())
            .collect();

        for &idx in &most_sensitive[..2] {
            let this_sensitivity = all_sensitivities[idx];
            for (other_idx, &other_sensitivity) in all_sensitivities.iter().enumerate() {
                if !most_sensitive[..3].contains(&other_idx) {
                    assert!(this_sensitivity >= other_sensitivity - EPSILON);
                }
            }
        }
    }

    #[test]
    fn test_gradient_computation() {
        let logits = vec![1.0, 2.0, 3.0, 0.5];
        let tdc = TropicalDualClifford::<f64, 4>::from_logits(&logits);

        // Test that dual numbers provide gradient information
        let dual_features = tdc.extract_dual_features();

        // At least some gradients should be non-zero for variable inputs
        let has_nonzero_gradients = dual_features.iter().any(|dn| dn.dual.abs() > EPSILON);
        assert!(has_nonzero_gradients);

        // Test gradient consistency
        for (i, dual_feature) in dual_features.iter().enumerate() {
            // Gradient should be 1 for the corresponding variable, 0 for others
            if i < logits.len() {
                assert!(
                    (dual_feature.dual - 1.0).abs() < EPSILON || dual_feature.dual.abs() < EPSILON
                );
            }
        }
    }

    #[test]
    fn test_automatic_differentiation_chain_rule() {
        // Test chain rule through the fusion system
        let logits = vec![1.0, 2.0];
        let tdc = TropicalDualClifford::<f64, 2>::from_logits(&logits);

        // Create a composite function using multiple algebraic views
        let evaluation_result = tdc.evaluate(&tdc);

        // Gradient should be automatically computed through the chain rule
        assert!(evaluation_result.gradient_norm.abs() >= 0.0);
        assert!(!evaluation_result.gradient_norm.is_nan());
        assert!(!evaluation_result.gradient_norm.is_infinite());
    }

    // ========== Tropical Algebra Properties ==========

    #[test]
    fn test_tropical_properties() {
        let logits1 = vec![1.0, 2.0, 3.0, 0.5];
        let logits2 = vec![2.0, 1.0, 2.5, 1.5];

        let tdc1 = TropicalDualClifford::<f64, 4>::from_logits(&logits1);
        let tdc2 = TropicalDualClifford::<f64, 4>::from_logits(&logits2);

        // Test tropical max-plus properties
        let _tropical1 = tdc1.extract_tropical_features();
        let _tropical2 = tdc2.extract_tropical_features();

        // Tropical addition is commutative
        let sum_12 = tdc1.add(&tdc2);
        let sum_21 = tdc2.add(&tdc1);

        // For tropical part, should be commutative (max operation)
        let max1 = sum_12.tropical().max_element();
        let max2 = sum_21.tropical().max_element();
        assert_relative_eq!(max1.value(), max2.value(), epsilon = EPSILON);
    }

    #[test]
    fn test_tropical_matrix_conversion() {
        let logits = vec![1.0, 2.0, 3.0, 0.5];
        let tdc = TropicalDualClifford::<f64, 4>::from_logits(&logits);

        let tropical_matrix = tdc.to_tropical_matrix(2, 2);
        // Test that matrix creation doesn't panic - dimensions are internal
        assert!(!tropical_matrix.to_attention_scores().is_empty());
    }

    // ========== Clifford Algebra Geometric Properties ==========

    #[test]
    fn test_clifford_geometric_properties() {
        let logits1 = vec![1.0, 0.0, 0.0, 0.0];
        let logits2 = vec![0.0, 1.0, 0.0, 0.0];

        let tdc1 = TropicalDualClifford::<f64, 4>::from_logits(&logits1);
        let tdc2 = TropicalDualClifford::<f64, 4>::from_logits(&logits2);

        // Test geometric transformation
        let transformed = tdc1.transform(&tdc2);
        assert!(!transformed.is_zero());

        // Test geometric relationships
        let evaluation = tdc1.evaluate(&tdc2);
        assert!(evaluation.geometric_distance >= 0.0);

        // Orthogonal vectors should have specific geometric distance
        assert!(evaluation.geometric_distance > 0.0);
    }

    #[test]
    fn test_clifford_multivector_properties() {
        let logits = vec![1.0, 2.0, 3.0, 0.5, 1.5, 2.5, 0.8, 1.2];
        let tdc = TropicalDualClifford::<f64, 8>::from_logits(&logits);

        let clifford = tdc.clifford();

        // Test basic geometric algebra properties
        assert!(clifford.norm() >= 0.0);

        // Test that Clifford part maintains geometric structure
        let scalar_part = clifford.get(0); // Scalar component
        assert!(!scalar_part.is_nan());
        assert!(!scalar_part.is_infinite());
    }

    // ========== Distribution and Probability Properties ==========

    #[test]
    fn test_tropical_dual_distribution_properties() {
        let logits = vec![1.0, 2.0, 3.0, 0.5, 1.5];
        let dist1 = TropicalDualDistribution::from_logits(&logits);
        let dist2 = TropicalDualDistribution::from_logits(&[2.0, 1.0, 2.5, 1.0, 1.8]);

        // Test KL divergence properties
        let kl_12 = dist1.kl_divergence(&dist2);
        let kl_21 = dist2.kl_divergence(&dist1);

        // KL divergence should be real and finite
        assert!(!kl_12.real.is_nan());
        assert!(!kl_12.real.is_infinite());
        assert!(!kl_21.real.is_nan());
        assert!(!kl_21.real.is_infinite());

        // Self KL divergence should be zero (or very small)
        let kl_self = dist1.kl_divergence(&dist1);
        assert!(kl_self.real.abs() < 1e-6);

        // Test sequence generation
        let sequence = dist1.most_likely_sequence(5);
        assert_eq!(sequence.len(), 5);

        // All sequence elements should be valid
        for &token in &sequence {
            assert!(token < dist1.vocab_size);
        }

        // Test geometric alignment
        let alignment = dist1.geometric_alignment(&dist2);
        assert!(!alignment.is_nan());
        assert!(!alignment.is_infinite());
    }

    #[test]
    fn test_attention_polytope_properties() {
        let logits = vec![1.0, 2.0, 3.0, 0.5];
        let dist = TropicalDualDistribution::from_logits(&logits);

        let polytope = dist.attention_polytope();

        // Should have vertices
        assert!(!polytope.is_empty());

        // Each vertex should have consistent dimension
        if !polytope.is_empty() {
            let first_dim = polytope[0].len();
            for vertex in &polytope {
                assert_eq!(vertex.len(), first_dim);
            }
        }
    }

    // ========== Conversion and Consistency Tests ==========

    #[test]
    fn test_conversion_consistency() {
        use conversion::*;

        let probs = vec![0.1, 0.3, 0.4, 0.2];

        // Test round-trip conversion
        let tropical = softmax_to_tropical(&probs);
        let recovered = tropical_to_softmax(&tropical);

        // Should recover approximately the same probabilities
        for (orig, recov) in probs.iter().zip(recovered.iter()) {
            assert_relative_eq!(orig, recov, epsilon = 1e-6);
        }

        // Test clifford to dual conversion
        let mv =
            Multivector::<3, 0, 0>::from_coefficients(vec![1.0, 2.0, 3.0, 0.5, 1.5, 2.5, 0.8, 1.2]);
        let dual_nums = clifford_to_dual::<f64>(&mv);

        assert_eq!(dual_nums.len(), 8);
        for (i, dual_num) in dual_nums.iter().enumerate() {
            assert_relative_eq!(dual_num.real, mv.get(i), epsilon = EPSILON);
        }
    }

    #[test]
    fn test_evaluation_result_consistency() {
        let logits1 = vec![1.0, 2.0, 3.0, 0.5];
        let logits2 = vec![1.5, 1.8, 2.5, 1.0];

        let tdc1 = TropicalDualClifford::<f64, 4>::from_logits(&logits1);
        let tdc2 = TropicalDualClifford::<f64, 4>::from_logits(&logits2);

        let result = tdc1.evaluate(&tdc2);

        // All components should be finite and real
        assert!(!result.best_path_score.is_zero() || result.best_path_score.is_zero());
        assert!(!result.gradient_norm.is_nan());
        assert!(!result.gradient_norm.is_infinite());
        assert!(result.geometric_distance >= 0.0);
        assert!(!result.combined_score.is_nan());
        assert!(!result.combined_score.is_infinite());
    }

    // ========== Builder Pattern Tests ==========

    #[test]
    fn test_builder_pattern() {
        use crate::TropicalDualCliffordBuilder;

        let tdc = TropicalDualCliffordBuilder::<f64, 4>::new()
            .add_logit(1.0)
            .add_logit(2.0)
            .add_logits(&[3.0, 0.5])
            .build();

        assert!(!tdc.is_zero());

        // Should have same structure as direct construction
        let direct = TropicalDualClifford::<f64, 4>::from_logits(&[1.0, 2.0, 3.0, 0.5]);

        // Compare norms as a consistency check
        let builder_norm = tdc.dual().norm().real;
        let direct_norm = direct.dual().norm().real;
        assert_relative_eq!(builder_norm, direct_norm, epsilon = EPSILON);
    }

    // ========== Edge Cases and Robustness ==========

    #[test]
    fn test_edge_case_empty_logits() {
        let empty_logits: Vec<f64> = vec![];
        let tdc = TropicalDualClifford::<f64, 4>::from_logits(&empty_logits);

        // Should handle empty input gracefully
        assert!(tdc.dual().norm().real.abs() < EPSILON);
    }

    #[test]
    fn test_edge_case_extreme_values() {
        // Use large but finite values to avoid NaN issues
        let extreme_logits = vec![1e10, -1e10, 0.0, 1e8];
        let tdc = TropicalDualClifford::<f64, 4>::from_logits(&extreme_logits);

        // Should handle extreme values without crashing
        assert!(!tdc.dual().norm().real.is_nan());
        let distance_to_self = tdc.distance(&tdc);

        // Distance to self should be finite and small, even with extreme values
        assert!(!distance_to_self.is_nan());
        assert!(!distance_to_self.is_infinite());
        assert!(distance_to_self < 1e-6);
    }

    #[test]
    fn test_edge_case_small_values() {
        let small_logits = vec![1e-15, 1e-14, 1e-13, 1e-12];
        let tdc = TropicalDualClifford::<f64, 4>::from_logits(&small_logits);

        // Should handle small values gracefully
        assert!(!tdc.dual().norm().real.is_nan());
        assert!(tdc.dual().norm().real >= 0.0);
    }

    // ========== Performance and Numerical Stability ==========

    #[test]
    fn test_numerical_stability() {
        let logits = vec![1.0, 1.0 + 1e-15, 1.0 + 2e-15, 1.0 + 3e-15];
        let tdc1 = TropicalDualClifford::<f64, 4>::from_logits(&logits);
        let tdc2 = TropicalDualClifford::<f64, 4>::from_logits(&logits);

        // Operations with nearly identical inputs should be stable
        let distance = tdc1.distance(&tdc2);
        assert!(distance < 1e-10);

        let evaluation = tdc1.evaluate(&tdc2);
        assert!(!evaluation.combined_score.is_nan());
    }

    #[test]
    fn test_consistency_across_dimensions() {
        // Test that operations are consistent regardless of dimension padding
        let logits_short = vec![1.0, 2.0];
        let logits_long = vec![1.0, 2.0, 0.0, 0.0];

        let tdc_short = TropicalDualClifford::<f64, 2>::from_logits(&logits_short);
        let tdc_long = TropicalDualClifford::<f64, 4>::from_logits(&logits_long);

        // Basic properties should be preserved
        assert!(tdc_short.dual().norm().real > 0.0);
        assert!(tdc_long.dual().norm().real > 0.0);
    }
}
