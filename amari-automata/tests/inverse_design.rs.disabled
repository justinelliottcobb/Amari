//! Comprehensive Tests for Inverse Design

use amari_automata::{InverseCADesigner, TargetPattern, TropicalConstraint, Objective, GeometricCA};
use amari_dual::DualMultivector;
use approx::assert_relative_eq;

#[test]
fn test_find_seed_for_target_pattern() {
    let target = TargetPattern::from_grid(&[
        [1, 0, 1],
        [0, 1, 0],
        [1, 0, 1],
    ]);

    let designer = InverseCADesigner::new();
    let seed = designer.find_seed(&target, 10); // 10 steps

    let mut ca = GeometricCA::from_seed(&seed);
    for _ in 0..10 {
        ca.step();
    }

    assert_eq!(ca.as_pattern(), target);
}

#[test]
fn test_inverse_design_with_dual_numbers() {
    // Dual numbers for gradient-based search
    let target = TargetPattern::checkerboard(10, 10);
    let mut designer = InverseCADesigner::with_dual_optimization();
    let mut seed = DualMultivector::<f64, 3, 0, 0>::random_grid(10, 10);

    for _ in 0..100 {
        let evolved = designer.evolve_dual(&seed, 20);
        let loss = designer.pattern_distance(&evolved.value, &target);

        if loss < 0.01 {
            break;
        }

        // Update using gradients
        seed = designer.gradient_step(seed, evolved.dual, 0.01);
    }

    assert!(designer.verify_seed(&seed.value, &target, 20));
}

#[test]
fn test_tropical_constraint_satisfaction() {
    // Tropical algebra linearizes constraints
    let constraints = vec![
        TropicalConstraint::has_pattern_at(5, 5, &[
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
        ]),
        TropicalConstraint::sparsity(0.2),
    ];

    let designer = InverseCADesigner::with_tropical_solver();
    let seed = designer.solve_constraints(&constraints);

    let mut ca = GeometricCA::from_seed(&seed);
    for _ in 0..10 {
        ca.step();
    }

    assert!(ca.has_pattern_at(5, 5, &[
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1],
    ]));
    assert!(ca.density() < 0.2);
}

#[test]
fn test_gradient_based_optimization() {
    // Test automatic differentiation through CA evolution
    let target = TargetPattern::single_glider();
    let mut designer = InverseCADesigner::with_gradient_descent();

    let initial_seed = designer.random_seed(20, 20);
    let optimized_seed = designer.optimize_seed(&initial_seed, &target, 100);

    let fitness_initial = designer.evaluate_fitness(&initial_seed, &target);
    let fitness_optimized = designer.evaluate_fitness(&optimized_seed, &target);

    assert!(fitness_optimized > fitness_initial);
}

#[test]
fn test_multi_objective_optimization() {
    // Multiple objectives: pattern match + sparsity + stability
    let objectives = vec![
        Objective::PatternMatch(TargetPattern::cross(5)),
        Objective::Sparsity(0.3),
        Objective::Stability(10), // stable for 10 steps
    ];

    let designer = InverseCADesigner::multi_objective();
    let pareto_front = designer.find_pareto_optimal(&objectives);

    assert!(pareto_front.len() > 1);
    for solution in pareto_front {
        assert!(solution.satisfies_constraints(&objectives));
    }
}

#[test]
fn test_pattern_morphing() {
    // Find transition from pattern A to pattern B
    let pattern_a = TargetPattern::horizontal_line(5);
    let pattern_b = TargetPattern::vertical_line(5);

    let designer = InverseCADesigner::new();
    let transition = designer.find_morphing_sequence(&pattern_a, &pattern_b, 20);

    assert_eq!(transition.initial_pattern(), pattern_a);
    assert_eq!(transition.final_pattern(), pattern_b);
    assert!(transition.is_smooth());
}

#[test]
fn test_reverse_engineering_rules() {
    // Given patterns, infer the CA rule
    let examples = vec![
        (TargetPattern::single_cell(), TargetPattern::cross(3)),
        (TargetPattern::two_cells_adjacent(), TargetPattern::diamond(3)),
    ];

    let designer = InverseCADesigner::rule_inference();
    let inferred_rule = designer.infer_rule(&examples);

    // Test the inferred rule
    for (input, expected_output) in examples {
        let mut ca = GeometricCA::with_rule(&inferred_rule);
        ca.set_pattern(&input);
        ca.evolve_to_stable();

        assert!(ca.pattern_similarity(&expected_output) > 0.9);
    }
}

#[test]
fn test_constraint_satisfaction_with_topology() {
    // Complex topological constraints
    let constraints = vec![
        TropicalConstraint::connected_components(1),
        TropicalConstraint::has_holes(0),
        TropicalConstraint::genus(0), // sphere topology
        TropicalConstraint::boundary_length(12),
    ];

    let designer = InverseCADesigner::with_topology_constraints();
    let seed = designer.solve_topological_constraints(&constraints);

    let mut ca = GeometricCA::from_seed(&seed);
    ca.evolve_to_stable();

    assert_eq!(ca.connected_components(), 1);
    assert_eq!(ca.holes(), 0);
    assert_eq!(ca.genus(), 0);
    assert_relative_eq!(ca.boundary_length() as f64, 12.0, epsilon = 1.0);
}

#[test]
fn test_inverse_design_performance() {
    // Large-scale inverse design should be tractable
    let target = TargetPattern::complex_fractal(100, 100);
    let designer = InverseCADesigner::with_performance_optimization();

    let start_time = std::time::Instant::now();
    let seed = designer.find_seed_fast(&target, 50);
    let duration = start_time.elapsed();

    // Should complete in reasonable time
    assert!(duration.as_secs() < 30);

    // Verify the solution
    let mut ca = GeometricCA::from_seed(&seed);
    for _ in 0..50 {
        ca.step();
    }

    assert!(ca.pattern_similarity(&target) > 0.8);
}

#[test]
fn test_dual_number_chain_rule() {
    // Test that chain rule works correctly through CA evolution
    let designer = InverseCADesigner::with_dual_optimization();
    let seed = DualMultivector::<f64, 3, 0, 0>::with_gradient(
        10, 10,
        |i, j| if i == 5 && j == 5 { 1.0 } else { 0.0 }
    );

    let evolved = designer.evolve_dual(&seed, 5);

    // Gradient should propagate through all evolution steps
    assert!(evolved.has_gradient_at(4, 5));
    assert!(evolved.has_gradient_at(6, 5));
    assert!(evolved.has_gradient_at(5, 4));
    assert!(evolved.has_gradient_at(5, 6));
}

#[test]
fn test_stochastic_optimization() {
    // Test evolutionary/genetic algorithm approaches
    let target = TargetPattern::maze_solution(20, 20);
    let mut designer = InverseCADesigner::evolutionary();

    designer.set_population_size(100);
    designer.set_mutation_rate(0.1);
    designer.set_crossover_rate(0.7);

    let best_individual = designer.evolve_population(&target, 50);

    assert!(designer.evaluate_fitness(&best_individual, &target) > 0.9);
}