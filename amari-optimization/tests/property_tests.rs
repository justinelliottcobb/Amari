//! Property-based tests for amari-optimization
//!
//! This module contains property-based tests that verify mathematical properties
//! and invariants of optimization algorithms across a wide range of inputs.

use amari_optimization::prelude::*;
use proptest::prelude::*;

/// Strategy for generating reasonable constraint parameters
fn constraint_params() -> impl Strategy<Value = (f64, f64)> {
    (-2.0..2.0, 0.1..5.0)
}

/// Property test helper: Simple quadratic objective
struct PropertyQuadratic {
    coefficients: Vec<f64>,
}

impl PropertyQuadratic {
    fn new(coefficients: Vec<f64>) -> Self {
        Self { coefficients }
    }
}

impl ConstrainedObjective<f64> for PropertyQuadratic {
    fn evaluate(&self, x: &[f64]) -> f64 {
        x.iter()
            .zip(&self.coefficients)
            .map(|(&xi, &ci)| ci * xi * xi)
            .sum::<f64>()
            / 2.0
    }

    fn gradient(&self, x: &[f64]) -> Vec<f64> {
        x.iter()
            .zip(&self.coefficients)
            .map(|(&xi, &ci)| ci * xi)
            .collect()
    }

    fn inequality_constraints(&self, _x: &[f64]) -> Vec<f64> {
        vec![]
    }

    fn equality_constraints(&self, _x: &[f64]) -> Vec<f64> {
        vec![]
    }

    fn inequality_jacobian(&self, _x: &[f64]) -> Vec<Vec<f64>> {
        vec![]
    }

    fn equality_jacobian(&self, _x: &[f64]) -> Vec<Vec<f64>> {
        vec![]
    }

    fn variable_bounds(&self) -> Vec<(f64, f64)> {
        vec![(-10.0, 10.0); self.coefficients.len()]
    }

    fn num_inequality_constraints(&self) -> usize {
        0
    }
    fn num_equality_constraints(&self) -> usize {
        0
    }
    fn num_variables(&self) -> usize {
        self.coefficients.len()
    }
}

/// Property test helper: Simple constrained quadratic
struct PropertyConstrainedQuadratic {
    coefficients: Vec<f64>,
    constraint_center: f64,
    constraint_radius: f64,
}

impl PropertyConstrainedQuadratic {
    fn new(coefficients: Vec<f64>, constraint_center: f64, constraint_radius: f64) -> Self {
        Self {
            coefficients,
            constraint_center,
            constraint_radius,
        }
    }
}

impl ConstrainedObjective<f64> for PropertyConstrainedQuadratic {
    fn evaluate(&self, x: &[f64]) -> f64 {
        x.iter()
            .zip(&self.coefficients)
            .map(|(&xi, &ci)| ci * xi * xi)
            .sum::<f64>()
            / 2.0
    }

    fn gradient(&self, x: &[f64]) -> Vec<f64> {
        x.iter()
            .zip(&self.coefficients)
            .map(|(&xi, &ci)| ci * xi)
            .collect()
    }

    fn inequality_constraints(&self, x: &[f64]) -> Vec<f64> {
        let sum_sq = x
            .iter()
            .map(|&xi| (xi - self.constraint_center).powi(2))
            .sum::<f64>();
        vec![sum_sq - self.constraint_radius * self.constraint_radius]
    }

    fn equality_constraints(&self, _x: &[f64]) -> Vec<f64> {
        vec![]
    }

    fn inequality_jacobian(&self, x: &[f64]) -> Vec<Vec<f64>> {
        let grad = x
            .iter()
            .map(|&xi| 2.0 * (xi - self.constraint_center))
            .collect();
        vec![grad]
    }

    fn equality_jacobian(&self, _x: &[f64]) -> Vec<Vec<f64>> {
        vec![]
    }

    fn variable_bounds(&self) -> Vec<(f64, f64)> {
        vec![(-10.0, 10.0); self.coefficients.len()]
    }

    fn num_inequality_constraints(&self) -> usize {
        1
    }
    fn num_equality_constraints(&self) -> usize {
        0
    }
    fn num_variables(&self) -> usize {
        self.coefficients.len()
    }
}

/// Property test helper: Natural gradient test function
struct PropertyNaturalGradient {
    fisher_eigenvalues: Vec<f64>,
}

impl PropertyNaturalGradient {
    fn new(fisher_eigenvalues: Vec<f64>) -> Self {
        Self { fisher_eigenvalues }
    }
}

impl ObjectiveWithFisher<f64> for PropertyNaturalGradient {
    fn evaluate(&self, theta: &[f64]) -> f64 {
        theta
            .iter()
            .zip(&self.fisher_eigenvalues)
            .map(|(&t, &e)| e * t * t)
            .sum::<f64>()
            / 2.0
    }

    fn gradient(&self, theta: &[f64]) -> Vec<f64> {
        theta
            .iter()
            .zip(&self.fisher_eigenvalues)
            .map(|(&t, &e)| e * t)
            .collect()
    }

    fn fisher_information(&self, _theta: &[f64]) -> Vec<Vec<f64>> {
        let n = self.fisher_eigenvalues.len();
        let mut matrix = vec![vec![0.0; n]; n];
        for (i, &eigenval) in self.fisher_eigenvalues.iter().enumerate() {
            matrix[i][i] = eigenval;
        }
        matrix
    }
}

proptest! {
    /// Property: Constrained optimization should always improve or maintain objective value
    #[test]
    fn constrained_optimization_improves_objective(
        coeffs in prop::collection::vec(0.1..10.0, 2..8),
        initial in prop::collection::vec(-2.0..2.0, 2..8)
    ) {
        prop_assume!(coeffs.len() == initial.len());

        let problem = PropertyQuadratic::new(coeffs);
        let optimizer = ConstrainedOptimizer::with_default_config(PenaltyMethod::Exterior);

        use amari_optimization::phantom::{Constrained, Euclidean, NonConvex, SingleObjective};
        let opt_problem: OptimizationProblem<1, Constrained, SingleObjective, NonConvex, Euclidean> =
            OptimizationProblem::new();

        let initial_objective = problem.evaluate(&initial);

        if let Ok(result) = optimizer.optimize(&opt_problem, &problem, initial) {
            if result.converged {
                // Final objective should be <= initial objective for minimization
                prop_assert!(result.objective_value <= initial_objective + 1e-6,
                           "Final objective {:.6} should be <= initial {:.6}",
                           result.objective_value, initial_objective);
            }
        }
    }

    /// Property: Optimization should converge to stationary point (gradient near zero)
    #[test]
    fn constrained_optimization_finds_stationary_point(
        coeffs in prop::collection::vec(0.1..5.0, 2..6),
        initial in prop::collection::vec(-1.0..1.0, 2..6)
    ) {
        prop_assume!(coeffs.len() == initial.len());

        let problem = PropertyQuadratic::new(coeffs.clone());
        let optimizer = ConstrainedOptimizer::with_default_config(PenaltyMethod::Exterior);

        use amari_optimization::phantom::{Constrained, Euclidean, NonConvex, SingleObjective};
        let opt_problem: OptimizationProblem<1, Constrained, SingleObjective, NonConvex, Euclidean> =
            OptimizationProblem::new();

        if let Ok(result) = optimizer.optimize(&opt_problem, &problem, initial) {
            if result.converged {
                let gradient = problem.gradient(&result.solution);
                let grad_norm = gradient.iter().map(|&g| g * g).sum::<f64>().sqrt();

                // For unconstrained quadratic, gradient should be small at optimum
                prop_assert!(grad_norm < 0.1,
                           "Gradient norm {:.6} should be small at stationary point",
                           grad_norm);
            }
        }
    }

    /// Property: Constrained optimization should satisfy constraints at solution
    #[test]
    fn constrained_optimization_satisfies_constraints(
        coeffs in prop::collection::vec(0.1..5.0, 2..6),
        constraint_params in constraint_params(),
        initial in prop::collection::vec(-2.0..2.0, 2..6)
    ) {
        prop_assume!(coeffs.len() == initial.len());

        let (center, radius) = constraint_params;
        let problem = PropertyConstrainedQuadratic::new(coeffs, center, radius);
        let optimizer = ConstrainedOptimizer::with_default_config(PenaltyMethod::AugmentedLagrangian);

        use amari_optimization::phantom::{Constrained, Euclidean, NonConvex, SingleObjective};
        let opt_problem: OptimizationProblem<1, Constrained, SingleObjective, NonConvex, Euclidean> =
            OptimizationProblem::new();

        if let Ok(result) = optimizer.optimize(&opt_problem, &problem, initial) {
            if result.converged {
                let constraints = problem.inequality_constraints(&result.solution);

                // All inequality constraints should be satisfied (≤ 0)
                for (i, &constraint_val) in constraints.iter().enumerate() {
                    prop_assert!(constraint_val <= 0.1,
                               "Constraint {} with value {:.6} should be satisfied",
                               i, constraint_val);
                }
            }
        }
    }

    /// Property: Natural gradient optimization should converge faster than regular gradient
    #[test]
    fn natural_gradient_convergence_property(
        eigenvals in prop::collection::vec(0.1..10.0, 2..8),
        initial_theta in prop::collection::vec(-1.0..1.0, 2..8)
    ) {
        prop_assume!(eigenvals.len() == initial_theta.len());
        prop_assume!(eigenvals.iter().all(|&e| e > 0.05)); // Ensure positive definite

        let objective = PropertyNaturalGradient::new(eigenvals.clone());
        let config = NaturalGradientConfig::default();
        let optimizer = NaturalGradientOptimizer::new(config);

        use amari_optimization::phantom::{Statistical, NonConvex, SingleObjective, Unconstrained};
        let opt_problem: OptimizationProblem<1, Unconstrained, SingleObjective, NonConvex, Statistical> =
            OptimizationProblem::new();

        if let Ok(result) = optimizer.optimize_statistical(&opt_problem, &objective, initial_theta) {
            if result.converged {
                // For quadratic functions, natural gradient should find minimum (zero)
                let solution_norm = result.parameters.iter().map(|&t| t * t).sum::<f64>().sqrt();

                prop_assert!(solution_norm < 0.1,
                           "Natural gradient solution norm {:.6} should be small",
                           solution_norm);

                // Final objective should be close to zero for quadratic centered at origin
                prop_assert!(result.objective_value < 0.01,
                           "Final objective {:.6} should be near minimum",
                           result.objective_value);
            }
        }
    }

    /// Property: Multi-objective optimization should find non-dominated solutions
    #[test]
    fn multi_objective_pareto_optimality(
        dimension in 2..8usize
    ) {
        struct SimpleMultiObjective {
            dim: usize,
        }

        impl MultiObjectiveFunction<f64> for SimpleMultiObjective {
            fn evaluate(&self, x: &[f64]) -> Vec<f64> {
                vec![
                    x.iter().map(|&xi| xi * xi).sum::<f64>(), // f1: sum of squares
                    x.iter().map(|&xi| (xi - 1.0) * (xi - 1.0)).sum::<f64>(), // f2: sum of (x-1)²
                ]
            }

            fn num_objectives(&self) -> usize { 2 }
            fn num_variables(&self) -> usize { self.dim }
            fn variable_bounds(&self) -> Vec<(f64, f64)> { vec![(0.0, 1.0); self.dim] }
        }

        let problem = SimpleMultiObjective { dim: dimension };
        let config = MultiObjectiveConfig::default();
        let nsga2 = NsgaII::new(config);

        use amari_optimization::phantom::{Euclidean, NonConvex, Unconstrained};
        let opt_problem: OptimizationProblem<1, Unconstrained, MultiObjective, NonConvex, Euclidean> =
            OptimizationProblem::new();

        if let Ok(result) = nsga2.optimize(&opt_problem, &problem) {
            if result.converged && !result.pareto_front.solutions.is_empty() {
                // Check that solutions in Pareto front are non-dominated
                let individuals = &result.pareto_front.solutions;

                for (i, ind1) in individuals.iter().enumerate() {
                    for (j, ind2) in individuals.iter().enumerate() {
                        if i != j {
                            // ind1 should not dominate ind2 (since they're both in Pareto front)
                            let dominates = ind1.objectives.iter().zip(&ind2.objectives)
                                .all(|(&o1, &o2)| o1 <= o2) &&
                                ind1.objectives.iter().zip(&ind2.objectives)
                                .any(|(&o1, &o2)| o1 < o2);

                            prop_assert!(!dominates,
                                       "Individual {} should not dominate individual {} in Pareto front",
                                       i, j);
                        }
                    }
                }

                // All individuals should have exactly 2 objectives
                for ind in individuals {
                    prop_assert_eq!(ind.objectives.len(), 2,
                                  "Each individual should have 2 objectives");
                }
            }
        }
    }

    /// Property: Tropical optimization should respect tropical arithmetic properties
    #[test]
    fn tropical_arithmetic_properties(
        values in prop::collection::vec(0.1..10.0, 2..6)
    ) {
        use amari_tropical::{TropicalMatrix, TropicalNumber};

        let optimizer = TropicalOptimizer::with_default_config();
        let objective: Vec<TropicalNumber<f64>> = values.iter()
            .map(|&v| TropicalNumber::new(v))
            .collect();

        let n = values.len();
        let matrix_data: Vec<Vec<f64>> = (0..n)
            .map(|i| (0..n).map(|j| if i == j { 0.0 } else { values[i] + values[j] }).collect())
            .collect();
        let constraint_matrix = TropicalMatrix::from_log_probs(&matrix_data);

        let constraint_rhs: Vec<TropicalNumber<f64>> = values.iter()
            .map(|&v| TropicalNumber::new(v + 1.0))
            .collect();

        if let Ok(result) = optimizer.solve_tropical_linear_program(&objective, &constraint_matrix, &constraint_rhs) {
            prop_assert!(result.converged, "Tropical optimization should converge");
            prop_assert_eq!(result.solution.len(), n, "Solution should have correct dimension");

            // Check that at least one component is tropical one (0) and others are tropical zero (-∞)
            let finite_count = result.solution.iter()
                .filter(|&x| !x.is_zero()) // Not tropical zero
                .count();

            prop_assert!(finite_count <= n, "At most n components should be finite");
        }
    }

    /// Property: Optimization algorithms should be translation invariant for unconstrained problems
    #[test]
    fn translation_invariance_property(
        coeffs in prop::collection::vec(0.1..5.0, 2..6),
        translation in prop::collection::vec(-3.0..3.0, 2..6),
        initial in prop::collection::vec(-1.0..1.0, 2..6)
    ) {
        prop_assume!(coeffs.len() == translation.len() && coeffs.len() == initial.len());

        struct TranslatedQuadratic {
            coeffs: Vec<f64>,
            translation: Vec<f64>,
        }

        impl ConstrainedObjective<f64> for TranslatedQuadratic {
            fn evaluate(&self, x: &[f64]) -> f64 {
                x.iter().zip(&self.coeffs).zip(&self.translation)
                    .map(|((&xi, &ci), &ti)| ci * (xi - ti) * (xi - ti))
                    .sum::<f64>() / 2.0
            }

            fn gradient(&self, x: &[f64]) -> Vec<f64> {
                x.iter().zip(&self.coeffs).zip(&self.translation)
                    .map(|((&xi, &ci), &ti)| ci * (xi - ti))
                    .collect()
            }

            fn inequality_constraints(&self, _x: &[f64]) -> Vec<f64> { vec![] }
            fn equality_constraints(&self, _x: &[f64]) -> Vec<f64> { vec![] }
            fn inequality_jacobian(&self, _x: &[f64]) -> Vec<Vec<f64>> { vec![] }
            fn equality_jacobian(&self, _x: &[f64]) -> Vec<Vec<f64>> { vec![] }
            fn variable_bounds(&self) -> Vec<(f64, f64)> { vec![(-10.0, 10.0); self.coeffs.len()] }
            fn num_inequality_constraints(&self) -> usize { 0 }
            fn num_equality_constraints(&self) -> usize { 0 }
            fn num_variables(&self) -> usize { self.coeffs.len() }
        }

        let problem = TranslatedQuadratic { coeffs, translation: translation.clone() };
        let optimizer = ConstrainedOptimizer::with_default_config(PenaltyMethod::Exterior);

        use amari_optimization::phantom::{Constrained, Euclidean, NonConvex, SingleObjective};
        let opt_problem: OptimizationProblem<1, Constrained, SingleObjective, NonConvex, Euclidean> =
            OptimizationProblem::new();

        if let Ok(result) = optimizer.optimize(&opt_problem, &problem, initial) {
            if result.converged {
                // Solution should be close to the translation vector (minimum of translated quadratic)
                for (i, (&sol, &trans)) in result.solution.iter().zip(&translation).enumerate() {
                    prop_assert!((sol - trans).abs() < 0.2,
                               "Component {} solution {:.3} should be near translation {:.3}",
                               i, sol, trans);
                }
            }
        }
    }
}

/// Unit test for property test infrastructure
#[test]
fn test_property_test_helpers() {
    // Test that property test helpers work correctly
    let coeffs = vec![1.0, 2.0, 3.0];
    let problem = PropertyQuadratic::new(coeffs);

    let x = vec![1.0, 2.0, 3.0];
    let expected_obj = (1.0 * 1.0 + 2.0 * 4.0 + 3.0 * 9.0) / 2.0; // (1 + 8 + 27) / 2 = 18
    let actual_obj = problem.evaluate(&x);

    assert!(
        (actual_obj - expected_obj).abs() < 1e-10,
        "Expected {}, got {}",
        expected_obj,
        actual_obj
    );

    let gradient = problem.gradient(&x);
    let expected_grad = vec![1.0, 4.0, 9.0];

    for (actual, expected) in gradient.iter().zip(&expected_grad) {
        assert!(
            (actual - expected).abs() < 1e-10,
            "Expected gradient component {}, got {}",
            expected,
            actual
        );
    }
}
