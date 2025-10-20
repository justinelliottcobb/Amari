//! # Constrained Optimization
//!
//! This module implements advanced constrained optimization algorithms for handling
//! equality and inequality constraints in optimization problems.
//!
//! ## Mathematical Background
//!
//! Constrained optimization solves problems of the form:
//!
//! ```text
//! minimize f(x)
//! subject to:
//!   g_i(x) ≤ 0,  i = 1, ..., m  (inequality constraints)
//!   h_j(x) = 0,  j = 1, ..., p  (equality constraints)
//!   x ∈ X                       (bound constraints)
//! ```
//!
//! ## Key Algorithms
//!
//! - **Penalty Methods**: Transform constrained problems into unconstrained ones
//! - **Barrier Methods**: Use barrier functions to enforce inequality constraints
//! - **Lagrange Multipliers**: Direct handling of KKT conditions
//! - **Augmented Lagrangian**: Combine penalties with Lagrange multipliers
//! - **Sequential Quadratic Programming (SQP)**: Newton-like methods for constraints
//!
//! ## KKT Conditions
//!
//! For a local minimum x*, the Karush-Kuhn-Tucker conditions must hold:
//! ```text
//! ∇f(x*) + Σᵢ λᵢ ∇gᵢ(x*) + Σⱼ μⱼ ∇hⱼ(x*) = 0
//! gᵢ(x*) ≤ 0,  λᵢ ≥ 0,  λᵢ gᵢ(x*) = 0  (complementary slackness)
//! hⱼ(x*) = 0
//! ```

use crate::phantom::{Constrained, OptimizationProblem};
use crate::OptimizationResult;

use num_traits::Float;
use std::marker::PhantomData;

/// Configuration for constrained optimization algorithms
#[derive(Clone, Debug)]
pub struct ConstrainedConfig<T: Float> {
    /// Maximum number of outer iterations
    pub max_iterations: usize,
    /// Maximum number of inner iterations (for penalty/barrier methods)
    pub max_inner_iterations: usize,
    /// Tolerance for constraint violations
    pub constraint_tolerance: T,
    /// Tolerance for optimality conditions
    pub optimality_tolerance: T,
    /// Initial penalty parameter
    pub initial_penalty: T,
    /// Penalty growth factor
    pub penalty_growth: T,
    /// Initial barrier parameter
    pub initial_barrier: T,
    /// Barrier reduction factor
    pub barrier_reduction: T,
    /// Line search parameters
    pub line_search_tolerance: T,
    /// Enable feasibility restoration
    pub enable_feasibility_restoration: bool,
}

impl<T: Float> Default for ConstrainedConfig<T> {
    fn default() -> Self {
        Self {
            max_iterations: 100,
            max_inner_iterations: 1000,
            constraint_tolerance: T::from(1e-6).unwrap(),
            optimality_tolerance: T::from(1e-6).unwrap(),
            initial_penalty: T::from(1.0).unwrap(),
            penalty_growth: T::from(10.0).unwrap(),
            initial_barrier: T::from(1.0).unwrap(),
            barrier_reduction: T::from(0.1).unwrap(),
            line_search_tolerance: T::from(1e-4).unwrap(),
            enable_feasibility_restoration: true,
        }
    }
}

/// Results from constrained optimization
#[derive(Clone, Debug)]
pub struct ConstrainedResult<T: Float> {
    /// Optimal solution
    pub solution: Vec<T>,
    /// Optimal objective value
    pub objective_value: T,
    /// Lagrange multipliers for inequality constraints
    pub lambda: Vec<T>,
    /// Lagrange multipliers for equality constraints
    pub mu: Vec<T>,
    /// Final constraint violations
    pub constraint_violations: Vec<T>,
    /// Number of iterations performed
    pub iterations: usize,
    /// Convergence status
    pub converged: bool,
    /// KKT error (measure of optimality)
    pub kkt_error: T,
}

/// Trait for constrained objective functions
pub trait ConstrainedObjective<T: Float> {
    /// Evaluate the objective function
    fn evaluate(&self, x: &[T]) -> T;

    /// Compute the gradient of the objective function
    fn gradient(&self, x: &[T]) -> Vec<T>;

    /// Compute the Hessian of the objective function (optional)
    fn hessian(&self, _x: &[T]) -> Option<Vec<Vec<T>>> {
        None
    }

    /// Evaluate inequality constraints g(x) ≤ 0
    fn inequality_constraints(&self, x: &[T]) -> Vec<T>;

    /// Evaluate equality constraints h(x) = 0
    fn equality_constraints(&self, x: &[T]) -> Vec<T>;

    /// Jacobian of inequality constraints
    fn inequality_jacobian(&self, x: &[T]) -> Vec<Vec<T>>;

    /// Jacobian of equality constraints
    fn equality_jacobian(&self, x: &[T]) -> Vec<Vec<T>>;

    /// Get variable bounds (lower, upper)
    fn variable_bounds(&self) -> Vec<(T, T)>;

    /// Number of inequality constraints
    fn num_inequality_constraints(&self) -> usize;

    /// Number of equality constraints
    fn num_equality_constraints(&self) -> usize;

    /// Number of variables
    fn num_variables(&self) -> usize;
}

/// Penalty method types
#[derive(Clone, Copy, Debug)]
pub enum PenaltyMethod {
    /// Exterior penalty method
    Exterior,
    /// Interior penalty method (barrier)
    Interior,
    /// Augmented Lagrangian method
    AugmentedLagrangian,
}

/// Constrained optimization solver
#[derive(Clone, Debug)]
pub struct ConstrainedOptimizer<T: Float> {
    config: ConstrainedConfig<T>,
    method: PenaltyMethod,
    _phantom: PhantomData<T>,
}

impl<T: Float> ConstrainedOptimizer<T> {
    /// Create a new constrained optimizer
    pub fn new(config: ConstrainedConfig<T>, method: PenaltyMethod) -> Self {
        Self {
            config,
            method,
            _phantom: PhantomData,
        }
    }

    /// Create optimizer with default configuration
    pub fn with_default_config(method: PenaltyMethod) -> Self {
        Self::new(ConstrainedConfig::default(), method)
    }

    /// Solve constrained optimization problem
    pub fn optimize<const DIM: usize>(
        &self,
        _problem: &OptimizationProblem<
            DIM,
            Constrained,
            impl crate::phantom::ObjectiveState,
            impl crate::phantom::ConvexityState,
            impl crate::phantom::ManifoldState,
        >,
        objective: &impl ConstrainedObjective<T>,
        initial_point: Vec<T>,
    ) -> OptimizationResult<ConstrainedResult<T>> {
        match self.method {
            PenaltyMethod::Exterior => self.exterior_penalty_method(objective, initial_point),
            PenaltyMethod::Interior => self.interior_penalty_method(objective, initial_point),
            PenaltyMethod::AugmentedLagrangian => {
                self.augmented_lagrangian_method(objective, initial_point)
            }
        }
    }

    /// Exterior penalty method
    fn exterior_penalty_method(
        &self,
        objective: &impl ConstrainedObjective<T>,
        mut x: Vec<T>,
    ) -> OptimizationResult<ConstrainedResult<T>> {
        let mut penalty_param = self.config.initial_penalty;
        let mut best_objective = T::infinity();

        for iteration in 0..self.config.max_iterations {
            // Define penalty function
            let penalty_objective = |vars: &[T]| -> T {
                let obj = objective.evaluate(vars);
                let mut penalty = T::zero();

                // Inequality constraints penalty: max(0, g(x))²
                let ineq_constraints = objective.inequality_constraints(vars);
                for &g in &ineq_constraints {
                    if g > T::zero() {
                        penalty = penalty + g * g;
                    }
                }

                // Equality constraints penalty: h(x)²
                let eq_constraints = objective.equality_constraints(vars);
                for &h in &eq_constraints {
                    penalty = penalty + h * h;
                }

                obj + penalty_param * penalty
            };

            // Solve unconstrained subproblem
            x = self.solve_unconstrained_subproblem(&penalty_objective, x)?;

            // Check convergence
            let current_objective = objective.evaluate(&x);
            let constraint_violation = self.compute_constraint_violation(objective, &x);

            if constraint_violation < self.config.constraint_tolerance {
                if (current_objective - best_objective).abs() < self.config.optimality_tolerance {
                    // Converged
                    let lambda = self.estimate_lagrange_multipliers(objective, &x, penalty_param);
                    let kkt_error = self.compute_kkt_error(objective, &x, &lambda.0, &lambda.1);

                    return Ok(ConstrainedResult {
                        solution: x.clone(),
                        objective_value: current_objective,
                        lambda: lambda.0,
                        mu: lambda.1,
                        constraint_violations: self.get_constraint_violations(objective, &x),
                        iterations: iteration + 1,
                        converged: true,
                        kkt_error,
                    });
                }
                best_objective = current_objective;
            }

            // Update penalty parameter
            penalty_param = penalty_param * self.config.penalty_growth;
        }

        Err(crate::OptimizationError::ConvergenceFailure {
            iterations: self.config.max_iterations,
        })
    }

    /// Interior penalty method (barrier method)
    fn interior_penalty_method(
        &self,
        objective: &impl ConstrainedObjective<T>,
        mut x: Vec<T>,
    ) -> OptimizationResult<ConstrainedResult<T>> {
        // First, find a feasible starting point
        if !self.is_feasible(objective, &x) {
            x = self.find_feasible_point(objective, x)?;
        }

        let mut barrier_param = self.config.initial_barrier;

        for iteration in 0..self.config.max_iterations {
            // Define barrier function
            let barrier_objective = |vars: &[T]| -> T {
                let obj = objective.evaluate(vars);
                let mut barrier = T::zero();

                // Logarithmic barrier for inequality constraints: -log(-g(x))
                let ineq_constraints = objective.inequality_constraints(vars);
                for &g in &ineq_constraints {
                    if g >= T::zero() {
                        return T::infinity(); // Infeasible
                    }
                    barrier = barrier - g.ln();
                }

                obj + barrier_param * barrier
            };

            // Solve unconstrained subproblem
            x = self.solve_unconstrained_subproblem(&barrier_objective, x)?;

            // Check convergence
            let constraint_violation = self.compute_constraint_violation(objective, &x);

            if constraint_violation < self.config.constraint_tolerance {
                let current_objective = objective.evaluate(&x);
                let lambda = self.estimate_lagrange_multipliers(objective, &x, barrier_param);
                let kkt_error = self.compute_kkt_error(objective, &x, &lambda.0, &lambda.1);

                if kkt_error < self.config.optimality_tolerance {
                    return Ok(ConstrainedResult {
                        solution: x.clone(),
                        objective_value: current_objective,
                        lambda: lambda.0,
                        mu: lambda.1,
                        constraint_violations: self.get_constraint_violations(objective, &x),
                        iterations: iteration + 1,
                        converged: true,
                        kkt_error,
                    });
                }
            }

            // Update barrier parameter
            barrier_param = barrier_param * self.config.barrier_reduction;
        }

        Err(crate::OptimizationError::ConvergenceFailure {
            iterations: self.config.max_iterations,
        })
    }

    /// Augmented Lagrangian method
    fn augmented_lagrangian_method(
        &self,
        objective: &impl ConstrainedObjective<T>,
        mut x: Vec<T>,
    ) -> OptimizationResult<ConstrainedResult<T>> {
        let n_ineq = objective.num_inequality_constraints();
        let n_eq = objective.num_equality_constraints();

        let mut lambda = vec![T::zero(); n_ineq]; // Inequality multipliers
        let mut mu = vec![T::zero(); n_eq]; // Equality multipliers
        let mut penalty_param = self.config.initial_penalty;

        for iteration in 0..self.config.max_iterations {
            // Define augmented Lagrangian
            let aug_lag_objective = |vars: &[T]| -> T {
                let obj = objective.evaluate(vars);
                let ineq_constraints = objective.inequality_constraints(vars);
                let eq_constraints = objective.equality_constraints(vars);

                let mut augmented = obj;

                // Inequality constraints: λᵢ gᵢ(x) + (ρ/2) max(0, gᵢ(x))²
                for (i, &g) in ineq_constraints.iter().enumerate() {
                    let max_g = if g > T::zero() { g } else { T::zero() };
                    augmented = augmented
                        + lambda[i] * g
                        + penalty_param / T::from(2.0).unwrap() * max_g * max_g;
                }

                // Equality constraints: μⱼ hⱼ(x) + (ρ/2) hⱼ(x)²
                for (j, &h) in eq_constraints.iter().enumerate() {
                    augmented =
                        augmented + mu[j] * h + penalty_param / T::from(2.0).unwrap() * h * h;
                }

                augmented
            };

            // Solve unconstrained subproblem
            x = self.solve_unconstrained_subproblem(&aug_lag_objective, x)?;

            // Update multipliers
            let ineq_constraints = objective.inequality_constraints(&x);
            let eq_constraints = objective.equality_constraints(&x);

            // Update inequality multipliers: λᵢ ← max(0, λᵢ + ρ gᵢ(x))
            for (i, &g) in ineq_constraints.iter().enumerate() {
                lambda[i] = (lambda[i] + penalty_param * g).max(T::zero());
            }

            // Update equality multipliers: μⱼ ← μⱼ + ρ hⱼ(x)
            for (j, &h) in eq_constraints.iter().enumerate() {
                mu[j] = mu[j] + penalty_param * h;
            }

            // Check convergence
            let constraint_violation = self.compute_constraint_violation(objective, &x);
            let kkt_error = self.compute_kkt_error(objective, &x, &lambda, &mu);

            if constraint_violation < self.config.constraint_tolerance
                && kkt_error < self.config.optimality_tolerance
            {
                return Ok(ConstrainedResult {
                    solution: x.clone(),
                    objective_value: objective.evaluate(&x),
                    lambda,
                    mu,
                    constraint_violations: self.get_constraint_violations(objective, &x),
                    iterations: iteration + 1,
                    converged: true,
                    kkt_error,
                });
            }

            // Update penalty parameter if needed
            if constraint_violation > self.config.constraint_tolerance * T::from(0.1).unwrap() {
                penalty_param = penalty_param * self.config.penalty_growth;
            }
        }

        Err(crate::OptimizationError::ConvergenceFailure {
            iterations: self.config.max_iterations,
        })
    }

    /// Solve unconstrained subproblem using gradient descent
    fn solve_unconstrained_subproblem(
        &self,
        objective: &impl Fn(&[T]) -> T,
        mut x: Vec<T>,
    ) -> OptimizationResult<Vec<T>> {
        let learning_rate = T::from(0.01).unwrap();
        let eps = T::from(1e-8).unwrap();

        for _ in 0..self.config.max_inner_iterations {
            // Compute numerical gradient
            let gradient = self.numerical_gradient(objective, &x, eps);
            let grad_norm = self.vector_norm(&gradient);

            if grad_norm < self.config.optimality_tolerance {
                break;
            }

            // Gradient descent step with line search
            let step_size = self.line_search(objective, &x, &gradient, learning_rate);
            for (i, &grad) in gradient.iter().enumerate() {
                x[i] = x[i] - step_size * grad;
            }
        }

        Ok(x)
    }

    /// Numerical gradient computation
    fn numerical_gradient(&self, f: &impl Fn(&[T]) -> T, x: &[T], eps: T) -> Vec<T> {
        let mut gradient = vec![T::zero(); x.len()];

        for i in 0..x.len() {
            let mut x_plus = x.to_vec();
            let mut x_minus = x.to_vec();

            x_plus[i] = x_plus[i] + eps;
            x_minus[i] = x_minus[i] - eps;

            gradient[i] = (f(&x_plus) - f(&x_minus)) / (T::from(2.0).unwrap() * eps);
        }

        gradient
    }

    /// Simple line search
    fn line_search(&self, f: &impl Fn(&[T]) -> T, x: &[T], direction: &[T], initial_step: T) -> T {
        let mut step = initial_step;
        let current_value = f(x);

        for _ in 0..20 {
            let mut x_trial = x.to_vec();
            for (i, &dir) in direction.iter().enumerate() {
                x_trial[i] = x_trial[i] - step * dir;
            }

            if f(&x_trial) < current_value {
                return step;
            }

            step = step * T::from(0.5).unwrap();
        }

        initial_step * T::from(0.01).unwrap()
    }

    /// Check if point is feasible
    fn is_feasible(&self, objective: &impl ConstrainedObjective<T>, x: &[T]) -> bool {
        let ineq_constraints = objective.inequality_constraints(x);
        let eq_constraints = objective.equality_constraints(x);

        let ineq_feasible = ineq_constraints.iter().all(|&g| g <= T::zero());
        let eq_feasible = eq_constraints
            .iter()
            .all(|&h| h.abs() < self.config.constraint_tolerance);

        ineq_feasible && eq_feasible
    }

    /// Find a feasible starting point (simplified)
    fn find_feasible_point(
        &self,
        objective: &impl ConstrainedObjective<T>,
        x: Vec<T>,
    ) -> OptimizationResult<Vec<T>> {
        // Simplified feasibility restoration - project to bounds
        let bounds = objective.variable_bounds();
        let mut feasible_x = x;

        for (i, &(lower, upper)) in bounds.iter().enumerate() {
            feasible_x[i] = feasible_x[i].max(lower).min(upper);
        }

        Ok(feasible_x)
    }

    /// Compute constraint violation
    fn compute_constraint_violation(&self, objective: &impl ConstrainedObjective<T>, x: &[T]) -> T {
        let ineq_constraints = objective.inequality_constraints(x);
        let eq_constraints = objective.equality_constraints(x);

        let mut violation = T::zero();

        for &g in &ineq_constraints {
            if g > T::zero() {
                violation = violation + g * g;
            }
        }

        for &h in &eq_constraints {
            violation = violation + h * h;
        }

        violation.sqrt()
    }

    /// Get constraint violations as vector
    fn get_constraint_violations(
        &self,
        objective: &impl ConstrainedObjective<T>,
        x: &[T],
    ) -> Vec<T> {
        let mut violations = objective.inequality_constraints(x);
        violations.extend(objective.equality_constraints(x));
        violations
    }

    /// Estimate Lagrange multipliers
    fn estimate_lagrange_multipliers(
        &self,
        objective: &impl ConstrainedObjective<T>,
        x: &[T],
        penalty: T,
    ) -> (Vec<T>, Vec<T>) {
        let ineq_constraints = objective.inequality_constraints(x);
        let eq_constraints = objective.equality_constraints(x);

        // Simple estimation based on penalty parameter
        let lambda: Vec<T> = ineq_constraints
            .iter()
            .map(|&g| {
                if g > T::zero() {
                    penalty * g
                } else {
                    T::zero()
                }
            })
            .collect();

        let mu: Vec<T> = eq_constraints.iter().map(|&h| penalty * h).collect();

        (lambda, mu)
    }

    /// Compute KKT error
    fn compute_kkt_error(
        &self,
        objective: &impl ConstrainedObjective<T>,
        x: &[T],
        lambda: &[T],
        mu: &[T],
    ) -> T {
        let obj_grad = objective.gradient(x);
        let ineq_jac = objective.inequality_jacobian(x);
        let eq_jac = objective.equality_jacobian(x);

        // Compute ∇L = ∇f + Σλᵢ∇gᵢ + Σμⱼ∇hⱼ
        let mut lagrangian_grad = obj_grad;

        for (i, lambda_i) in lambda.iter().enumerate() {
            for (j, &grad_g_ij) in ineq_jac[i].iter().enumerate() {
                lagrangian_grad[j] = lagrangian_grad[j] + *lambda_i * grad_g_ij;
            }
        }

        for (i, mu_i) in mu.iter().enumerate() {
            for (j, &grad_h_ij) in eq_jac[i].iter().enumerate() {
                lagrangian_grad[j] = lagrangian_grad[j] + *mu_i * grad_h_ij;
            }
        }

        self.vector_norm(&lagrangian_grad)
    }

    /// Compute vector norm
    fn vector_norm(&self, v: &[T]) -> T {
        v.iter()
            .map(|&x| x * x)
            .fold(T::zero(), |acc, x| acc + x)
            .sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Simple test problem: minimize (x-1)² + (y-1)² subject to x + y ≤ 1
    struct SimpleConstrained;

    impl ConstrainedObjective<f64> for SimpleConstrained {
        fn evaluate(&self, x: &[f64]) -> f64 {
            (x[0] - 1.0).powi(2) + (x[1] - 1.0).powi(2)
        }

        fn gradient(&self, x: &[f64]) -> Vec<f64> {
            vec![2.0 * (x[0] - 1.0), 2.0 * (x[1] - 1.0)]
        }

        fn inequality_constraints(&self, x: &[f64]) -> Vec<f64> {
            vec![x[0] + x[1] - 1.0] // x + y ≤ 1
        }

        fn equality_constraints(&self, _x: &[f64]) -> Vec<f64> {
            vec![] // No equality constraints
        }

        fn inequality_jacobian(&self, _x: &[f64]) -> Vec<Vec<f64>> {
            vec![vec![1.0, 1.0]] // ∇(x + y - 1) = [1, 1]
        }

        fn equality_jacobian(&self, _x: &[f64]) -> Vec<Vec<f64>> {
            vec![] // No equality constraints
        }

        fn variable_bounds(&self) -> Vec<(f64, f64)> {
            vec![(-10.0, 10.0), (-10.0, 10.0)]
        }

        fn num_inequality_constraints(&self) -> usize {
            1
        }

        fn num_equality_constraints(&self) -> usize {
            0
        }

        fn num_variables(&self) -> usize {
            2
        }
    }

    #[test]
    fn test_constrained_optimizer_creation() {
        let config = ConstrainedConfig::<f64>::default();
        let _optimizer = ConstrainedOptimizer::new(config, PenaltyMethod::Exterior);

        let _default_optimizer =
            ConstrainedOptimizer::<f64>::with_default_config(PenaltyMethod::AugmentedLagrangian);
    }

    #[test]
    fn test_exterior_penalty_method() {
        let problem = SimpleConstrained;
        let optimizer = ConstrainedOptimizer::<f64>::with_default_config(PenaltyMethod::Exterior);

        use crate::phantom::{Euclidean, NonConvex, SingleObjective};
        let opt_problem: OptimizationProblem<
            2,
            Constrained,
            SingleObjective,
            NonConvex,
            Euclidean,
        > = OptimizationProblem::new();

        let initial_point = vec![0.0, 0.0];
        let result = optimizer.optimize(&opt_problem, &problem, initial_point);

        assert!(result.is_ok());
        let solution = result.unwrap();

        // Solution should be near (0.5, 0.5) which satisfies x + y = 1
        assert!(solution.solution[0] + solution.solution[1] <= 1.1); // Allow some tolerance
        assert!(solution.iterations > 0);
    }

    #[test]
    fn test_augmented_lagrangian_method() {
        let problem = SimpleConstrained;
        let optimizer =
            ConstrainedOptimizer::<f64>::with_default_config(PenaltyMethod::AugmentedLagrangian);

        use crate::phantom::{Euclidean, NonConvex, SingleObjective};
        let opt_problem: OptimizationProblem<
            2,
            Constrained,
            SingleObjective,
            NonConvex,
            Euclidean,
        > = OptimizationProblem::new();

        let initial_point = vec![0.0, 0.0];
        let result = optimizer.optimize(&opt_problem, &problem, initial_point);

        assert!(result.is_ok());
        let solution = result.unwrap();

        // Check constraint satisfaction
        assert!(solution.solution[0] + solution.solution[1] <= 1.1);
        assert!(solution.iterations > 0);
    }

    #[test]
    fn test_constraint_violation_computation() {
        let problem = SimpleConstrained;
        let optimizer = ConstrainedOptimizer::<f64>::with_default_config(PenaltyMethod::Exterior);

        let feasible_point = vec![0.25, 0.25]; // Satisfies x + y ≤ 1
        let infeasible_point = vec![1.0, 1.0]; // Violates x + y ≤ 1

        let violation_feasible = optimizer.compute_constraint_violation(&problem, &feasible_point);
        let violation_infeasible =
            optimizer.compute_constraint_violation(&problem, &infeasible_point);

        assert!(violation_feasible < 0.1); // Should be small or zero
        assert!(violation_infeasible > 0.5); // Should be significant
    }

    #[test]
    fn test_lagrange_multiplier_estimation() {
        let problem = SimpleConstrained;
        let optimizer = ConstrainedOptimizer::<f64>::with_default_config(PenaltyMethod::Exterior);

        let point = vec![0.5, 0.5]; // On the constraint boundary
        let (lambda, mu) = optimizer.estimate_lagrange_multipliers(&problem, &point, 1.0);

        assert_eq!(lambda.len(), 1); // One inequality constraint
        assert_eq!(mu.len(), 0); // No equality constraints
    }

    #[test]
    fn test_feasibility_check() {
        let problem = SimpleConstrained;
        let optimizer = ConstrainedOptimizer::<f64>::with_default_config(PenaltyMethod::Interior);

        let feasible_point = vec![0.25, 0.25];
        let infeasible_point = vec![1.0, 1.0];

        assert!(optimizer.is_feasible(&problem, &feasible_point));
        assert!(!optimizer.is_feasible(&problem, &infeasible_point));
    }
}
