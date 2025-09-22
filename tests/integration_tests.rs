//! Integration tests for consistency across algebraic systems

use amari_core::Multivector;
use amari_dual::{DualNumber, functions::{softmax, cross_entropy_loss}};
use amari_tropical::{TropicalNumber, TropicalMultivector, viterbi::TropicalViterbi};
use amari_fusion::TropicalDualClifford;
use approx::assert_relative_eq;

/// Test consistency between tropical and standard softmax
#[test]
fn test_softmax_consistency() {
    let logits = vec![1.0, 2.0, 3.0, 0.5, 1.5];
    
    // Standard softmax using dual numbers
    let dual_logits: Vec<DualNumber<f64>> = logits.iter()
        .map(|&x| DualNumber::variable(x))
        .collect();
    let standard_softmax = softmax(&dual_logits);
    
    // Tropical approximation
    let tropical_logits: Vec<TropicalNumber<f64>> = logits.iter()
        .map(|&x| TropicalNumber(x))
        .collect();
    
    // Find max for normalization
    let max_val = tropical_logits.iter()
        .fold(TropicalNumber::neg_infinity(), |acc, &x| acc.tropical_add(x));
    
    let tropical_normalized: Vec<f64> = tropical_logits.iter()
        .map(|&x| (x.0 - max_val.0).exp())
        .collect();
    
    let tropical_sum: f64 = tropical_normalized.iter().sum();
    let tropical_softmax: Vec<f64> = tropical_normalized.iter()
        .map(|&x| x / tropical_sum)
        .collect();
    
    // Compare results - should be similar for well-conditioned inputs
    for (standard, tropical) in standard_softmax.iter().zip(tropical_softmax.iter()) {
        assert_relative_eq!(standard.real, *tropical, epsilon = 1e-10);
    }
}

/// Test consistency between dual and manual gradient computation
#[test]
fn test_gradient_consistency() {
    let inputs = vec![0.5, 1.0, 1.5, 0.8];
    let targets = vec![1.0, 0.0, 0.0, 0.0]; // One-hot target
    
    // Dual number automatic differentiation
    let dual_inputs: Vec<DualNumber<f64>> = inputs.iter()
        .map(|&x| DualNumber::variable(x))
        .collect();
    
    let dual_loss = cross_entropy_loss(&dual_inputs, &targets);
    let auto_gradient: Vec<f64> = dual_inputs.iter()
        .map(|d| d.dual)
        .collect();
    
    // Manual gradient computation
    let epsilon = 1e-8;
    let mut manual_gradient = Vec::with_capacity(inputs.len());
    
    for i in 0..inputs.len() {
        let mut inputs_plus = inputs.clone();
        let mut inputs_minus = inputs.clone();
        inputs_plus[i] += epsilon;
        inputs_minus[i] -= epsilon;
        
        let loss_plus = manual_cross_entropy(&inputs_plus, &targets);
        let loss_minus = manual_cross_entropy(&inputs_minus, &targets);
        
        let finite_diff = (loss_plus - loss_minus) / (2.0 * epsilon);
        manual_gradient.push(finite_diff);
    }
    
    // Compare gradients
    for (auto, manual) in auto_gradient.iter().zip(manual_gradient.iter()) {
        assert_relative_eq!(*auto, *manual, epsilon = 1e-5);
    }
}

/// Manual cross-entropy computation for testing
fn manual_cross_entropy(inputs: &[f64], targets: &[f64]) -> f64 {
    // Compute softmax
    let max_val = inputs.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let exp_vals: Vec<f64> = inputs.iter().map(|&x| (x - max_val).exp()).collect();
    let sum_exp: f64 = exp_vals.iter().sum();
    let softmax_vals: Vec<f64> = exp_vals.iter().map(|&x| x / sum_exp).collect();
    
    // Compute cross-entropy
    let mut loss = 0.0;
    for (prob, target) in softmax_vals.iter().zip(targets.iter()) {
        if *prob > 0.0 && *target > 0.0 {
            loss -= target * prob.ln();
        }
    }
    loss
}

/// Test consistency between Clifford and standard geometric operations
#[test]
fn test_clifford_consistency() {
    let coeffs1 = vec![1.0, 0.5, 0.3, 0.2, 0.1, 0.0, 0.0, 0.0];
    let coeffs2 = vec![0.8, 0.6, 0.4, 0.3, 0.2, 0.1, 0.0, 0.0];
    
    let mv1 = Multivector::<3, 0, 0>::from_coefficients(coeffs1.clone());
    let mv2 = Multivector::<3, 0, 0>::from_coefficients(coeffs2.clone());
    
    // Clifford geometric product
    let clifford_product = mv1.geometric_product(&mv2);
    
    // Manual computation of geometric product (simplified for scalars + vectors)
    let scalar1 = coeffs1[0];
    let scalar2 = coeffs2[0];
    let vec1 = [coeffs1[1], coeffs1[2], coeffs1[3]];
    let vec2 = [coeffs2[1], coeffs2[2], coeffs2[3]];
    
    // Scalar part: s1*s2 - v1·v2
    let dot_product = vec1[0]*vec2[0] + vec1[1]*vec2[1] + vec1[2]*vec2[2];
    let expected_scalar = scalar1 * scalar2 - dot_product;
    
    assert_relative_eq!(clifford_product.get(0), expected_scalar, epsilon = 1e-10);
    
    // Vector parts: s1*v2 + s2*v1 + v1×v2
    let cross_product = [
        vec1[1]*vec2[2] - vec1[2]*vec2[1],
        vec1[2]*vec2[0] - vec1[0]*vec2[2],
        vec1[0]*vec2[1] - vec1[1]*vec2[0],
    ];
    
    let expected_vector = [
        scalar1 * vec2[0] + scalar2 * vec1[0] + cross_product[0],
        scalar1 * vec2[1] + scalar2 * vec1[1] + cross_product[1],
        scalar1 * vec2[2] + scalar2 * vec1[2] + cross_product[2],
    ];
    
    for i in 0..3 {
        assert_relative_eq!(clifford_product.get(i + 1), expected_vector[i], epsilon = 1e-10);
    }
}

/// Test TDC self-consistency across all three algebras
#[test]
fn test_tdc_self_consistency() {
    let logits = vec![1.5, 2.0, 0.8, 1.2, 0.5, 1.8, 0.3, 0.9];
    let tdc = TropicalDualClifford::<f64, 8>::from_logits(&logits);
    
    // Test that transformations preserve essential properties
    let identity_transform = TropicalDualClifford::from_logits(&vec![0.0; 8]);
    let transformed = tdc.transform(&identity_transform);
    
    // Should be close to original for identity-like transform
    let distance = tdc.distance(&transformed);
    assert!(distance < 10.0); // Generous bound for transformation consistency
    
    // Test sensitivity analysis consistency
    let sensitivity = tdc.sensitivity_analysis();
    assert!(!sensitivity.sensitivities.is_empty());
    assert!(sensitivity.total_sensitivity() >= 0.0);
    
    // Most sensitive components should make sense
    let most_sensitive = sensitivity.most_sensitive(3);
    assert_eq!(most_sensitive.len(), 3);
    for &idx in &most_sensitive {
        assert!(idx < 8);
    }
}

/// Test tropical algebra consistency with dynamic programming
#[test]
fn test_tropical_dp_consistency() {
    let size = 5;
    
    // Create simple transition matrix for comparison
    let transitions: Vec<Vec<f64>> = (0..size)
        .map(|i| (0..size).map(|j| if i == j { 0.0 } else { -1.0 }).collect())
        .collect();
    
    let emissions: Vec<Vec<f64>> = (0..3)
        .map(|t| (0..size).map(|s| (t * size + s) as f64 * 0.1).collect())
        .collect();
    
    let observations = vec![0, 1, 2];
    
    // Tropical Viterbi
    let decoder = TropicalViterbi::new(transitions.clone(), emissions.clone());
    let (tropical_path, _prob) = decoder.decode(&observations);
    
    // Standard Viterbi in log space
    let standard_path = standard_viterbi(&transitions, &emissions, &observations);
    
    // Paths should be identical for this simple case
    assert_eq!(tropical_path.len(), standard_path.len());
    assert_eq!(tropical_path, standard_path);
}

/// Standard Viterbi implementation for comparison
fn standard_viterbi(
    transitions: &[Vec<f64>],
    emissions: &[Vec<f64>],
    observations: &[usize],
) -> Vec<usize> {
    let num_states = transitions.len();
    let seq_len = observations.len();
    
    if seq_len == 0 || num_states == 0 {
        return Vec::new();
    }
    
    // Initialize DP table
    let mut dp = vec![vec![f64::NEG_INFINITY; num_states]; seq_len];
    let mut path = vec![vec![0; num_states]; seq_len];
    
    // Initialize first column
    for state in 0..num_states {
        if observations[0] < emissions[0].len() {
            dp[0][state] = emissions[0][observations[0]];
        }
    }
    
    // Fill DP table
    for t in 1..seq_len {
        for curr_state in 0..num_states {
            for prev_state in 0..num_states {
                let emission_score = if t < emissions.len() && observations[t] < emissions[t].len() {
                    emissions[t][observations[t]]
                } else {
                    0.0
                };
                
                let score = dp[t-1][prev_state] + 
                           transitions[prev_state][curr_state] + 
                           emission_score;
                
                if score > dp[t][curr_state] {
                    dp[t][curr_state] = score;
                    path[t][curr_state] = prev_state;
                }
            }
        }
    }
    
    // Backtrack
    let mut result = vec![0; seq_len];
    
    // Find best final state
    let mut best_state = 0;
    let mut best_score = f64::NEG_INFINITY;
    for state in 0..num_states {
        if dp[seq_len-1][state] > best_score {
            best_score = dp[seq_len-1][state];
            best_state = state;
        }
    }
    
    result[seq_len-1] = best_state;
    
    // Backtrack through path
    for t in (1..seq_len).rev() {
        result[t-1] = path[t][result[t]];
    }
    
    result
}

/// Test numerical stability across systems
#[test]
fn test_numerical_stability() {
    // Test with extreme values
    let extreme_logits = vec![100.0, -100.0, 50.0, -50.0];
    let tdc = TropicalDualClifford::<f64, 4>::from_logits(&extreme_logits);
    
    // Should not produce NaN or infinite values
    let evaluation = tdc.evaluate(&tdc);
    assert!(evaluation.combined_score.is_finite());
    assert!(!evaluation.combined_score.is_nan());
    
    // Distance should be zero (or very small) for self-distance
    let self_distance = tdc.distance(&tdc);
    assert!(self_distance < 1e-10);
    
    // Sensitivity analysis should handle extreme values
    let sensitivity = tdc.sensitivity_analysis();
    assert!(sensitivity.total_sensitivity().is_finite());
    assert!(!sensitivity.total_sensitivity().is_nan());
}

/// Test interpolation consistency
#[test]
fn test_interpolation_consistency() {
    let logits1 = vec![1.0, 0.0, 0.0, 0.0];
    let logits2 = vec![0.0, 1.0, 0.0, 0.0];
    
    let tdc1 = TropicalDualClifford::<f64, 4>::from_logits(&logits1);
    let tdc2 = TropicalDualClifford::<f64, 4>::from_logits(&logits2);
    
    // Test interpolation at boundaries
    let interp_0 = tdc1.interpolate(&tdc2, 0.0);
    let interp_1 = tdc1.interpolate(&tdc2, 1.0);
    
    let dist_0 = tdc1.distance(&interp_0);
    let dist_1 = tdc2.distance(&interp_1);
    
    assert!(dist_0 < 1e-10); // Should be close to tdc1
    assert!(dist_1 < 1e-10); // Should be close to tdc2
    
    // Test midpoint interpolation
    let interp_mid = tdc1.interpolate(&tdc2, 0.5);
    let dist_mid_1 = tdc1.distance(&interp_mid);
    let dist_mid_2 = tdc2.distance(&interp_mid);
    
    // Midpoint should be roughly equidistant (within reasonable bounds)
    assert!((dist_mid_1 - dist_mid_2).abs() < 1.0);
}

/// Test conversion consistency
#[test]
fn test_conversion_consistency() {
    use amari_fusion::conversion::*;
    
    // Test softmax to tropical conversion round-trip
    let probs = vec![0.1, 0.3, 0.4, 0.2];
    let tropical = softmax_to_tropical(&probs);
    let recovered = tropical_to_softmax(&tropical);
    
    // Should recover approximately the same probabilities
    for (orig, recov) in probs.iter().zip(recovered.iter()) {
        assert_relative_eq!(*orig, *recov, epsilon = 1e-10);
    }
    
    // Test Clifford to dual conversion
    let mv = Multivector::<3, 0, 0>::from_coefficients(vec![1.0, 0.5, 0.3, 0.2, 0.1, 0.0, 0.0, 0.0]);
    let dual_coeffs = clifford_to_dual::<f64>(&mv);
    
    // Should preserve coefficient values
    for i in 0..8 {
        assert_relative_eq!(dual_coeffs[i].real, mv.get(i), epsilon = 1e-10);
        assert_relative_eq!(dual_coeffs[i].dual, 1.0, epsilon = 1e-10); // Variables have unit derivative
    }
}

/// Test system integration under stress conditions
#[test]
fn test_stress_integration() {
    let large_size = 100;
    let logits: Vec<f64> = (0..large_size).map(|i| (i as f64) * 0.01 - 0.5).collect();
    
    // This should not panic or produce invalid results
    let tdc = TropicalDualClifford::<f64, 8>::from_logits(&logits[..8]);
    
    // Multiple operations in sequence
    let mut current = tdc.clone();
    for i in 0..10 {
        let modifier_logits: Vec<f64> = (0..8).map(|j| (i + j) as f64 * 0.01).collect();
        let modifier = TropicalDualClifford::<f64, 8>::from_logits(&modifier_logits);
        
        current = current.transform(&modifier);
        
        // Verify no degenerate states
        assert!(current.distance(&tdc).is_finite());
        assert!(!current.distance(&tdc).is_nan());
    }
    
    // Final state should still be reasonable
    let final_sensitivity = current.sensitivity_analysis();
    assert!(final_sensitivity.total_sensitivity().is_finite());
}

/// Test consistency across different floating point types
#[test]
fn test_floating_point_consistency() {
    let logits_f64 = vec![1.0, 2.0, 0.5, 1.5];
    let logits_f32: Vec<f32> = logits_f64.iter().map(|&x| x as f32).collect();
    
    let tdc_f64 = TropicalDualClifford::<f64, 4>::from_logits(&logits_f64);
    let tdc_f32 = TropicalDualClifford::<f32, 4>::from_logits(&logits_f32);
    
    // Convert both to comparable form
    let sens_f64 = tdc_f64.sensitivity_analysis();
    let sens_f32 = tdc_f32.sensitivity_analysis();
    
    // Should have similar total sensitivity (within float precision)
    let total_f64 = sens_f64.total_sensitivity();
    let total_f32 = sens_f32.total_sensitivity() as f64;
    
    assert_relative_eq!(total_f64, total_f32, epsilon = 1e-6);
}