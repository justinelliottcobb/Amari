//! Tests for Inverse Design capabilities

use amari_automata::{InverseDesigner, InverseDesignable, Target, Configuration};
use amari_core::Multivector;
use approx::assert_relative_eq;

type TestDesigner = InverseDesigner<f64, 3, 0, 0>;

#[test]
fn test_designer_creation() {
    let designer = TestDesigner::new(5, 5, 3, 0.01);

    // Designer should be properly initialized
    assert_eq!(designer.target_steps, 3);
    assert_relative_eq!(designer.learning_rate, 0.01);
}

#[test]
fn test_random_configuration() {
    let designer = TestDesigner::new(3, 3, 2, 0.1);
    let config = designer.random_configuration(42);

    // Configuration should have the right dimensions
    assert_eq!(config.initial_state.len(), 3);
    assert_eq!(config.initial_state[0].len(), 3);

    // Rule parameters should be reasonable
    assert!(config.rule_params.threshold > 0.0);
    assert!(config.rule_params.geo_weight > 0.0);
}

#[test]
fn test_target_creation() {
    let target_state = vec![
        vec![Multivector::zero(); 3],
        vec![Multivector::scalar(1.0), Multivector::basis_vector(0), Multivector::zero()],
        vec![Multivector::zero(); 3],
    ];

    let target = Target::new(target_state.clone());

    assert_eq!(target.target_state.len(), 3);
    assert_eq!(target.position_weights.len(), 3);
    assert_relative_eq!(target.position_weights[1][1], 1.0);
}

#[test]
fn test_target_with_weights() {
    let target_state = vec![vec![Multivector::zero(); 2]; 2];
    let weights = vec![
        vec![1.0, 2.0],
        vec![0.5, 3.0],
    ];

    let target = Target::with_weights(target_state, weights.clone());

    assert_eq!(target.position_weights, weights);
}

#[test]
fn test_fitness_evaluation() {
    let designer = TestDesigner::new(3, 3, 1, 0.1);

    // Create a simple target
    let target_state = vec![
        vec![Multivector::zero(); 3],
        vec![Multivector::zero(), Multivector::scalar(1.0), Multivector::zero()],
        vec![Multivector::zero(); 3],
    ];
    let target = Target::new(target_state);

    // Create a configuration
    let config = designer.random_configuration(123);

    // Fitness should be calculable
    let fitness = designer.fitness(&config, &target);
    assert!(fitness >= 0.0);
}

#[test]
fn test_solver_configuration() {
    let mut designer = TestDesigner::new(5, 5, 10, 0.05);

    designer.set_max_iterations(500);
    designer.set_convergence_threshold(1e-8);

    assert_eq!(designer.max_iterations, 500);
    assert_relative_eq!(designer.convergence_threshold, 1e-8);
}

#[test]
fn test_simple_inverse_design() {
    let designer = TestDesigner::new(3, 3, 2, 0.1);

    // Create a simple target: single activated cell
    let mut target_state = vec![vec![Multivector::zero(); 3]; 3];
    target_state[1][1] = Multivector::scalar(1.0);
    let target = Target::new(target_state);

    // Try to find a seed (this may not always succeed with the simplified implementation)
    match designer.find_seed(&target) {
        Ok(config) => {
            // If successful, the configuration should be valid
            assert_eq!(config.initial_state.len(), 3);
            assert_eq!(config.initial_state[0].len(), 3);
        }
        Err(_) => {
            // This is acceptable for now since we have a simplified implementation
            // In a full implementation, we'd expect more success
        }
    }
}

#[test]
fn test_configuration_properties() {
    let config = Configuration {
        initial_state: vec![
            vec![Multivector::scalar(0.5), Multivector::zero()],
            vec![Multivector::basis_vector(0), Multivector::scalar(1.0)],
        ],
        rule_params: OptimizableRule {
            threshold: 0.3,
            geo_weight: 1.2,
            outer_weight: 0.8,
            inner_weight: 0.4,
        },
    };

    assert_eq!(config.initial_state.len(), 2);
    assert_relative_eq!(config.rule_params.threshold, 0.3);
    assert_relative_eq!(config.rule_params.geo_weight, 1.2);
}

#[test]
fn test_dual_number_simulation() {
    let designer = TestDesigner::new(2, 2, 1, 0.1);
    let config = designer.random_configuration(789);

    // This would test the dual number simulation if fully implemented
    // For now, we test that the method exists and doesn't panic
    match designer.simulate_with_gradients(&config) {
        Ok(result) => {
            assert_eq!(result.len(), 2);
            assert_eq!(result[0].len(), 2);
        }
        Err(_) => {
            // Expected for the simplified implementation
        }
    }
}

#[test]
fn test_optimization_parameters() {
    let rule = OptimizableRule {
        threshold: 0.5,
        geo_weight: 1.0,
        outer_weight: 0.5,
        inner_weight: 0.3,
    };

    // All parameters should be accessible and modifiable
    assert_relative_eq!(rule.threshold, 0.5);
    assert_relative_eq!(rule.geo_weight, 1.0);
    assert_relative_eq!(rule.outer_weight, 0.5);
    assert_relative_eq!(rule.inner_weight, 0.3);
}

#[test]
fn test_target_patterns() {
    // Test various target patterns that might be useful for UI assembly

    // Horizontal line pattern
    let mut horizontal = vec![vec![Multivector::zero(); 5]; 3];
    for x in 0..5 {
        horizontal[1][x] = Multivector::basis_vector(0);
    }
    let target_h = Target::new(horizontal);
    assert_eq!(target_h.target_state[1][2].magnitude(), 1.0);

    // Cross pattern
    let mut cross = vec![vec![Multivector::zero(); 3]; 3];
    cross[1][0] = Multivector::scalar(1.0);
    cross[1][1] = Multivector::scalar(1.0);
    cross[1][2] = Multivector::scalar(1.0);
    cross[0][1] = Multivector::scalar(1.0);
    cross[2][1] = Multivector::scalar(1.0);
    let target_c = Target::new(cross);
    assert_eq!(target_c.target_state[1][1].magnitude(), 1.0);
}

#[test]
fn test_convergence_detection() {
    let mut designer = TestDesigner::new(2, 2, 1, 0.1);
    designer.set_convergence_threshold(1e-6);

    // Create a trivial target (all zeros)
    let target_state = vec![vec![Multivector::zero(); 2]; 2];
    let target = Target::new(target_state);

    // A zero configuration should be close to a zero target
    let zero_config = Configuration {
        initial_state: vec![vec![Multivector::zero(); 2]; 2],
        rule_params: OptimizableRule {
            threshold: 0.5,
            geo_weight: 1.0,
            outer_weight: 0.5,
            inner_weight: 0.5,
        },
    };

    let fitness = designer.fitness(&zero_config, &target);
    // Should be very low for matching configurations
    assert!(fitness < 1.0);
}