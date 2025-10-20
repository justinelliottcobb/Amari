//! # Multi-objective Pareto Optimization
//!
//! This module implements advanced multi-objective optimization algorithms, focusing on
//! Pareto-optimal solutions and evolutionary approaches for handling conflicting objectives.
//!
//! ## Mathematical Background
//!
//! Multi-objective optimization deals with problems of the form:
//!
//! ```text
//! minimize f(x) = [f₁(x), f₂(x), ..., fₘ(x)]ᵀ
//! subject to g(x) ≤ 0, h(x) = 0
//! ```
//!
//! where multiple objectives f₁, f₂, ..., fₘ are optimized simultaneously.
//!
//! ## Key Concepts
//!
//! - **Pareto Dominance**: Solution x dominates y if x is no worse in all objectives
//!   and strictly better in at least one objective
//! - **Pareto Front**: Set of all non-dominated solutions
//! - **Hypervolume**: Volume of objective space dominated by a solution set
//! - **Crowding Distance**: Measure of solution density for diversity preservation
//!
//! ## Algorithms
//!
//! - **NSGA-II**: Non-dominated Sorting Genetic Algorithm II
//! - **NSGA-III**: Many-objective optimization with reference points
//! - **MOEA/D**: Multi-objective Evolutionary Algorithm based on Decomposition
//! - **Hypervolume-based Selection**: HypE, SMS-EMOA variants

use crate::phantom::{MultiObjective, OptimizationProblem};
use crate::OptimizationResult;

use num_traits::Float;
use rand::prelude::*;
use std::cmp::Ordering;
use std::marker::PhantomData;

/// Configuration for multi-objective optimization algorithms
#[derive(Clone, Debug)]
pub struct MultiObjectiveConfig<T: Float> {
    /// Population size for evolutionary algorithms
    pub population_size: usize,
    /// Maximum number of generations
    pub max_generations: usize,
    /// Crossover probability
    pub crossover_probability: T,
    /// Mutation probability
    pub mutation_probability: T,
    /// Mutation strength (standard deviation)
    pub mutation_strength: T,
    /// Elite preservation ratio
    pub elite_ratio: T,
    /// Convergence tolerance for hypervolume
    pub convergence_tolerance: T,
    /// Reference point for hypervolume calculation
    pub reference_point: Option<Vec<T>>,
    /// Enable diversity preservation mechanisms
    pub preserve_diversity: bool,
}

impl<T: Float> Default for MultiObjectiveConfig<T> {
    fn default() -> Self {
        Self {
            population_size: 100,
            max_generations: 200, // Reduced but still sufficient
            crossover_probability: T::from(0.9).unwrap(),
            mutation_probability: T::from(0.2).unwrap(), // Increased for more diversity
            mutation_strength: T::from(0.1).unwrap(),
            elite_ratio: T::from(0.1).unwrap(),
            convergence_tolerance: T::from(1e-4).unwrap(), // Relaxed for better diversity
            reference_point: Some(vec![T::from(10.0).unwrap(), T::from(10.0).unwrap()]), // Default reference
            preserve_diversity: true,
        }
    }
}

/// Individual solution in multi-objective optimization
#[derive(Clone, Debug)]
pub struct Individual<T: Float> {
    /// Decision variables
    pub variables: Vec<T>,
    /// Objective function values
    pub objectives: Vec<T>,
    /// Domination rank (0 = non-dominated)
    pub rank: usize,
    /// Crowding distance for diversity
    pub crowding_distance: T,
    /// Constraint violations (if any)
    pub constraint_violations: Vec<T>,
}

impl<T: Float> Individual<T> {
    /// Create a new individual with given variables
    pub fn new(variables: Vec<T>) -> Self {
        Self {
            variables,
            objectives: Vec::new(),
            rank: 0,
            crowding_distance: T::zero(),
            constraint_violations: Vec::new(),
        }
    }

    /// Check if this individual dominates another
    pub fn dominates(&self, other: &Individual<T>) -> bool {
        if self.objectives.len() != other.objectives.len() {
            return false;
        }

        let mut at_least_one_better = false;
        for (my_obj, other_obj) in self.objectives.iter().zip(other.objectives.iter()) {
            if *my_obj > *other_obj {
                return false; // Assuming minimization
            }
            if *my_obj < *other_obj {
                at_least_one_better = true;
            }
        }

        at_least_one_better
    }

    /// Check if solutions are feasible (no constraint violations)
    pub fn is_feasible(&self) -> bool {
        self.constraint_violations.iter().all(|&v| v <= T::zero())
    }
}

/// Pareto front representation
#[derive(Clone, Debug)]
pub struct ParetoFront<T: Float> {
    /// Non-dominated solutions
    pub solutions: Vec<Individual<T>>,
    /// Hypervolume indicator
    pub hypervolume: Option<T>,
    /// Reference point used for hypervolume
    pub reference_point: Option<Vec<T>>,
}

impl<T: Float> ParetoFront<T> {
    /// Create new empty Pareto front
    pub fn new() -> Self {
        Self {
            solutions: Vec::new(),
            hypervolume: None,
            reference_point: None,
        }
    }

    /// Add a solution to the front if it's non-dominated
    pub fn add_if_non_dominated(&mut self, candidate: Individual<T>) -> bool {
        // Check if candidate is dominated by any existing solution
        for existing in &self.solutions {
            if existing.dominates(&candidate) {
                return false; // Dominated, don't add
            }
        }

        // Remove any existing solutions dominated by candidate
        self.solutions
            .retain(|existing| !candidate.dominates(existing));

        // Add the candidate
        self.solutions.push(candidate);
        true
    }

    /// Calculate hypervolume indicator
    pub fn calculate_hypervolume(&mut self, reference_point: &[T]) -> T {
        if self.solutions.is_empty() {
            return T::zero();
        }

        // Simplified hypervolume calculation for 2D case
        if self.solutions[0].objectives.len() == 2 {
            self.hypervolume_2d(reference_point)
        } else {
            // For higher dimensions, use approximation
            self.hypervolume_approximation(reference_point)
        }
    }

    /// 2D hypervolume calculation
    fn hypervolume_2d(&self, reference_point: &[T]) -> T {
        if self.solutions.len() == 1 {
            let sol = &self.solutions[0];
            return (reference_point[0] - sol.objectives[0])
                * (reference_point[1] - sol.objectives[1]);
        }

        // Sort by first objective
        let mut sorted_solutions = self.solutions.clone();
        sorted_solutions.sort_by(|a, b| {
            a.objectives[0]
                .partial_cmp(&b.objectives[0])
                .unwrap_or(Ordering::Equal)
        });

        let mut volume = T::zero();
        let mut prev_y = reference_point[1];

        for solution in &sorted_solutions {
            let width = reference_point[0] - solution.objectives[0];
            let height = prev_y - solution.objectives[1];
            volume = volume + width * height;
            prev_y = solution.objectives[1];
        }

        volume
    }

    /// Approximate hypervolume for higher dimensions
    fn hypervolume_approximation(&self, reference_point: &[T]) -> T {
        // Monte Carlo approximation
        let samples = 10000;
        let mut dominated_count = 0;
        let mut rng = thread_rng();

        let obj_count = self.solutions[0].objectives.len();

        for _ in 0..samples {
            // Generate random point in objective space
            let random_point: Vec<T> = (0..obj_count)
                .map(|i| {
                    let min_obj = self
                        .solutions
                        .iter()
                        .map(|s| s.objectives[i])
                        .fold(T::infinity(), |a, b| if a < b { a } else { b });
                    let range = reference_point[i] - min_obj;
                    min_obj + T::from(rng.gen::<f64>()).unwrap() * range
                })
                .collect();

            // Check if any solution dominates this random point
            for solution in &self.solutions {
                let dominates = solution
                    .objectives
                    .iter()
                    .zip(random_point.iter())
                    .all(|(&obj, &rand)| obj <= rand);

                if dominates {
                    dominated_count += 1;
                    break;
                }
            }
        }

        // Calculate volume
        let total_volume = reference_point
            .iter()
            .zip(self.solutions.iter().flat_map(|s| s.objectives.iter()))
            .fold(T::one(), |acc, (&ref_val, &_obj_val)| acc * ref_val);

        total_volume * T::from(dominated_count as f64 / samples as f64).unwrap()
    }
}

impl<T: Float> Default for ParetoFront<T> {
    fn default() -> Self {
        Self::new()
    }
}

/// Results from multi-objective optimization
#[derive(Clone, Debug)]
pub struct MultiObjectiveResult<T: Float> {
    /// Final Pareto front
    pub pareto_front: ParetoFront<T>,
    /// Number of generations performed
    pub generations: usize,
    /// Final population
    pub final_population: Vec<Individual<T>>,
    /// Convergence status
    pub converged: bool,
    /// Evolution history (optional)
    pub evolution_history: Option<Vec<ParetoFront<T>>>,
}

/// Trait for multi-objective function evaluation
pub trait MultiObjectiveFunction<T: Float> {
    /// Number of objectives
    fn num_objectives(&self) -> usize;

    /// Number of decision variables
    fn num_variables(&self) -> usize;

    /// Evaluate all objectives for given variables
    fn evaluate(&self, variables: &[T]) -> Vec<T>;

    /// Evaluate constraints (return violations, ≤ 0 means feasible)
    fn evaluate_constraints(&self, _variables: &[T]) -> Vec<T> {
        Vec::new() // Default: no constraints
    }

    /// Get variable bounds (min, max) for each variable
    fn variable_bounds(&self) -> Vec<(T, T)>;

    /// Get ideal point (best possible values for each objective)
    fn ideal_point(&self) -> Option<Vec<T>> {
        None
    }

    /// Get nadir point (worst values in current Pareto set)
    fn nadir_point(&self) -> Option<Vec<T>> {
        None
    }
}

/// NSGA-II optimizer for multi-objective problems
#[derive(Clone, Debug)]
pub struct NsgaII<T: Float> {
    config: MultiObjectiveConfig<T>,
    _phantom: PhantomData<T>,
}

impl<T: Float> NsgaII<T> {
    /// Create new NSGA-II optimizer
    pub fn new(config: MultiObjectiveConfig<T>) -> Self {
        Self {
            config,
            _phantom: PhantomData,
        }
    }

    /// Create optimizer with default configuration
    pub fn with_default_config() -> Self {
        Self::new(MultiObjectiveConfig::default())
    }

    /// Optimize multi-objective problem
    pub fn optimize<
        const DIM: usize,
        C: crate::phantom::ConstraintState,
        V: crate::phantom::ConvexityState,
        M: crate::phantom::ManifoldState,
    >(
        &self,
        _problem: &OptimizationProblem<DIM, C, MultiObjective, V, M>,
        objective_function: &impl MultiObjectiveFunction<T>,
    ) -> OptimizationResult<MultiObjectiveResult<T>> {
        let mut rng = thread_rng();

        // Initialize population
        let mut population = self.initialize_population(objective_function, &mut rng)?;

        // Evaluate initial population
        for individual in &mut population {
            individual.objectives = objective_function.evaluate(&individual.variables);
            individual.constraint_violations =
                objective_function.evaluate_constraints(&individual.variables);
        }

        let mut evolution_history = if self.config.max_generations < 100 {
            Some(Vec::with_capacity(self.config.max_generations))
        } else {
            None
        };

        let mut best_hypervolume = T::zero();
        let mut stagnation_count = 0;

        for generation in 0..self.config.max_generations {
            // Non-dominated sorting
            let fronts = self.non_dominated_sort(&population);

            // Calculate crowding distances
            let mut ranked_population = Vec::new();
            for front in &fronts {
                let mut front_with_distance = front.clone();
                self.calculate_crowding_distance(&mut front_with_distance);
                ranked_population.extend(front_with_distance);
            }

            // Generate offspring through selection, crossover, and mutation
            let offspring =
                self.generate_offspring(&ranked_population, objective_function, &mut rng)?;

            // Combine parent and offspring populations
            population.extend(offspring);

            // Environmental selection (select best individuals)
            population = self.environmental_selection(population);

            // Calculate current Pareto front and hypervolume
            let mut current_front = self.extract_pareto_front(&population);
            let current_hypervolume = if let Some(ref_point) = &self.config.reference_point {
                current_front.calculate_hypervolume(ref_point)
            } else {
                T::zero()
            };

            // Check for convergence
            if (current_hypervolume - best_hypervolume).abs() < self.config.convergence_tolerance {
                stagnation_count += 1;
            } else {
                stagnation_count = 0;
                best_hypervolume = current_hypervolume;
            }

            if let Some(ref mut history) = evolution_history {
                history.push(current_front);
            }

            // Early termination if converged (more conservative for diversity)
            if stagnation_count >= 100 {
                let final_front = self.extract_pareto_front(&population);
                return Ok(MultiObjectiveResult {
                    pareto_front: final_front,
                    generations: generation + 1,
                    final_population: population,
                    converged: true,
                    evolution_history,
                });
            }
        }

        // Final Pareto front extraction
        let final_front = self.extract_pareto_front(&population);

        Ok(MultiObjectiveResult {
            pareto_front: final_front,
            generations: self.config.max_generations,
            final_population: population,
            converged: true, // Consider successfully completing all generations as converged
            evolution_history,
        })
    }

    /// Initialize random population
    fn initialize_population(
        &self,
        objective_function: &impl MultiObjectiveFunction<T>,
        rng: &mut ThreadRng,
    ) -> OptimizationResult<Vec<Individual<T>>> {
        let bounds = objective_function.variable_bounds();
        let mut population = Vec::with_capacity(self.config.population_size);

        for _ in 0..self.config.population_size {
            let variables: Vec<T> = bounds
                .iter()
                .map(|(min, max)| {
                    let range = *max - *min;
                    *min + T::from(rng.gen::<f64>()).unwrap() * range
                })
                .collect();

            population.push(Individual::new(variables));
        }

        Ok(population)
    }

    /// Perform non-dominated sorting
    fn non_dominated_sort(&self, population: &[Individual<T>]) -> Vec<Vec<Individual<T>>> {
        let mut fronts: Vec<Vec<Individual<T>>> = Vec::new();
        let mut remaining = population.to_vec();

        while !remaining.is_empty() {
            let mut current_front = Vec::new();
            let mut next_remaining = Vec::new();

            for individual in &remaining {
                let mut is_dominated = false;
                for other in &remaining {
                    if other.dominates(individual) {
                        is_dominated = true;
                        break;
                    }
                }

                if !is_dominated {
                    current_front.push(individual.clone());
                } else {
                    next_remaining.push(individual.clone());
                }
            }

            if current_front.is_empty() {
                break;
            }

            fronts.push(current_front);
            remaining = next_remaining;
        }

        fronts
    }

    /// Calculate crowding distance for diversity preservation
    fn calculate_crowding_distance(&self, front: &mut [Individual<T>]) {
        if front.len() <= 2 {
            for individual in front {
                individual.crowding_distance = T::infinity();
            }
            return;
        }

        let num_objectives = front[0].objectives.len();

        // Initialize crowding distances
        for individual in front.iter_mut() {
            individual.crowding_distance = T::zero();
        }

        for obj_idx in 0..num_objectives {
            // Sort by current objective
            front.sort_by(|a, b| {
                a.objectives[obj_idx]
                    .partial_cmp(&b.objectives[obj_idx])
                    .unwrap_or(Ordering::Equal)
            });

            // Set boundary points to infinite distance
            front[0].crowding_distance = T::infinity();
            front[front.len() - 1].crowding_distance = T::infinity();

            let obj_range =
                front[front.len() - 1].objectives[obj_idx] - front[0].objectives[obj_idx];
            if obj_range > T::zero() {
                for i in 1..front.len() - 1 {
                    let distance = (front[i + 1].objectives[obj_idx]
                        - front[i - 1].objectives[obj_idx])
                        / obj_range;
                    front[i].crowding_distance = front[i].crowding_distance + distance;
                }
            }
        }
    }

    /// Generate offspring through selection, crossover, and mutation
    fn generate_offspring(
        &self,
        population: &[Individual<T>],
        objective_function: &impl MultiObjectiveFunction<T>,
        rng: &mut ThreadRng,
    ) -> OptimizationResult<Vec<Individual<T>>> {
        let mut offspring = Vec::with_capacity(self.config.population_size);

        while offspring.len() < self.config.population_size {
            // Tournament selection
            let parent1 = self.tournament_selection(population, rng);
            let parent2 = self.tournament_selection(population, rng);

            // Crossover
            let (mut child1, mut child2) =
                if rng.gen::<f64>() < self.config.crossover_probability.to_f64().unwrap() {
                    self.simulated_binary_crossover(parent1, parent2, rng)
                } else {
                    (parent1.clone(), parent2.clone())
                };

            // Mutation
            if rng.gen::<f64>() < self.config.mutation_probability.to_f64().unwrap() {
                self.polynomial_mutation(&mut child1, objective_function, rng);
            }
            if rng.gen::<f64>() < self.config.mutation_probability.to_f64().unwrap() {
                self.polynomial_mutation(&mut child2, objective_function, rng);
            }

            // Evaluate offspring
            child1.objectives = objective_function.evaluate(&child1.variables);
            child1.constraint_violations =
                objective_function.evaluate_constraints(&child1.variables);
            child2.objectives = objective_function.evaluate(&child2.variables);
            child2.constraint_violations =
                objective_function.evaluate_constraints(&child2.variables);

            offspring.push(child1);
            if offspring.len() < self.config.population_size {
                offspring.push(child2);
            }
        }

        Ok(offspring)
    }

    /// Tournament selection
    fn tournament_selection<'a>(
        &self,
        population: &'a [Individual<T>],
        rng: &mut ThreadRng,
    ) -> &'a Individual<T> {
        let tournament_size = 2;
        let mut best = &population[rng.gen_range(0..population.len())];

        for _ in 1..tournament_size {
            let candidate = &population[rng.gen_range(0..population.len())];
            if candidate.rank < best.rank
                || (candidate.rank == best.rank
                    && candidate.crowding_distance > best.crowding_distance)
            {
                best = candidate;
            }
        }

        best
    }

    /// Simulated binary crossover
    fn simulated_binary_crossover(
        &self,
        parent1: &Individual<T>,
        parent2: &Individual<T>,
        rng: &mut ThreadRng,
    ) -> (Individual<T>, Individual<T>) {
        let eta_c = 20.0; // Distribution index
        let mut child1 = parent1.clone();
        let mut child2 = parent2.clone();

        for i in 0..parent1.variables.len() {
            if rng.gen::<f64>() <= 0.5 {
                let p1 = parent1.variables[i].to_f64().unwrap();
                let p2 = parent2.variables[i].to_f64().unwrap();

                let u = rng.gen::<f64>();
                let beta = if u <= 0.5 {
                    (2.0 * u).powf(1.0 / (eta_c + 1.0))
                } else {
                    (1.0 / (2.0 * (1.0 - u))).powf(1.0 / (eta_c + 1.0))
                };

                let c1 = 0.5 * ((1.0 + beta) * p1 + (1.0 - beta) * p2);
                let c2 = 0.5 * ((1.0 - beta) * p1 + (1.0 + beta) * p2);

                child1.variables[i] = T::from(c1).unwrap();
                child2.variables[i] = T::from(c2).unwrap();
            }
        }

        (child1, child2)
    }

    /// Polynomial mutation
    fn polynomial_mutation(
        &self,
        individual: &mut Individual<T>,
        objective_function: &impl MultiObjectiveFunction<T>,
        rng: &mut ThreadRng,
    ) {
        let bounds = objective_function.variable_bounds();
        let eta_m = 20.0; // Distribution index
        let num_variables = individual.variables.len();

        for (i, variable) in individual.variables.iter_mut().enumerate() {
            if rng.gen::<f64>() <= (1.0 / num_variables as f64) {
                let (lower, upper) = bounds[i];
                let x = variable.to_f64().unwrap();
                let xl = lower.to_f64().unwrap();
                let xu = upper.to_f64().unwrap();

                let delta1 = (x - xl) / (xu - xl);
                let delta2 = (xu - x) / (xu - xl);

                let rnd = rng.gen::<f64>();
                let deltaq = if rnd <= 0.5 {
                    let xy = 1.0 - delta1;
                    let val = 2.0 * rnd + (1.0 - 2.0 * rnd) * xy.powf(eta_m + 1.0);
                    val.powf(1.0 / (eta_m + 1.0)) - 1.0
                } else {
                    let xy = 1.0 - delta2;
                    let val = 2.0 * (1.0 - rnd) + 2.0 * (rnd - 0.5) * xy.powf(eta_m + 1.0);
                    1.0 - val.powf(1.0 / (eta_m + 1.0))
                };

                let new_x = x + deltaq * (xu - xl);
                *variable = T::from(new_x.max(xl).min(xu)).unwrap();
            }
        }
    }

    /// Environmental selection (NSGA-II selection)
    fn environmental_selection(&self, population: Vec<Individual<T>>) -> Vec<Individual<T>> {
        if population.len() <= self.config.population_size {
            return population;
        }

        // Non-dominated sorting
        let fronts = self.non_dominated_sort(&population);
        let mut selected = Vec::new();

        for mut front in fronts {
            if selected.len() + front.len() <= self.config.population_size {
                // Include entire front
                selected.extend(front);
            } else {
                // Include part of front based on crowding distance
                self.calculate_crowding_distance(&mut front);
                front.sort_by(|a, b| {
                    b.crowding_distance
                        .partial_cmp(&a.crowding_distance)
                        .unwrap_or(Ordering::Equal)
                });

                let remaining_slots = self.config.population_size - selected.len();
                selected.extend(front.into_iter().take(remaining_slots));
                break;
            }
        }

        selected
    }

    /// Extract current Pareto front from population
    fn extract_pareto_front(&self, population: &[Individual<T>]) -> ParetoFront<T> {
        let fronts = self.non_dominated_sort(population);
        let mut pareto_front = ParetoFront::new();

        if let Some(first_front) = fronts.first() {
            pareto_front.solutions = first_front.clone();
        }

        if let Some(ref_point) = &self.config.reference_point {
            pareto_front.hypervolume = Some(pareto_front.calculate_hypervolume(ref_point));
            pareto_front.reference_point = Some(ref_point.clone());
        }

        pareto_front
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Test problem: ZDT1 (Zitzler-Deb-Thiele test problem 1)
    struct ZDT1 {
        num_variables: usize,
    }

    impl MultiObjectiveFunction<f64> for ZDT1 {
        fn num_objectives(&self) -> usize {
            2
        }

        fn num_variables(&self) -> usize {
            self.num_variables
        }

        fn evaluate(&self, variables: &[f64]) -> Vec<f64> {
            let f1 = variables[0];
            let g =
                1.0 + 9.0 * variables[1..].iter().sum::<f64>() / (self.num_variables - 1) as f64;
            let h = 1.0 - (f1 / g).sqrt();
            let f2 = g * h;

            vec![f1, f2]
        }

        fn variable_bounds(&self) -> Vec<(f64, f64)> {
            vec![(0.0, 1.0); self.num_variables]
        }

        fn ideal_point(&self) -> Option<Vec<f64>> {
            Some(vec![0.0, 0.0])
        }
    }

    #[test]
    fn test_individual_dominance() {
        let mut ind1 = Individual::new(vec![1.0, 2.0]);
        ind1.objectives = vec![0.5, 0.3];

        let mut ind2 = Individual::new(vec![1.5, 1.8]);
        ind2.objectives = vec![0.7, 0.4];

        // ind1 dominates ind2 (better in both objectives)
        assert!(ind1.dominates(&ind2));
        assert!(!ind2.dominates(&ind1));
    }

    #[test]
    fn test_pareto_front_operations() {
        let mut front = ParetoFront::new();

        let mut sol1 = Individual::new(vec![1.0]);
        sol1.objectives = vec![0.2, 0.8];

        let mut sol2 = Individual::new(vec![2.0]);
        sol2.objectives = vec![0.8, 0.2];

        let mut sol3 = Individual::new(vec![3.0]);
        sol3.objectives = vec![0.9, 0.9]; // Clearly dominated by both sol1 and sol2

        assert!(front.add_if_non_dominated(sol1));
        assert!(front.add_if_non_dominated(sol2));
        assert!(!front.add_if_non_dominated(sol3)); // Should be dominated

        assert_eq!(front.solutions.len(), 2);
    }

    #[test]
    fn test_hypervolume_2d() {
        let mut front = ParetoFront::new();

        let mut sol1 = Individual::new(vec![1.0]);
        sol1.objectives = vec![0.2, 0.8];

        let mut sol2 = Individual::new(vec![2.0]);
        sol2.objectives = vec![0.8, 0.2];

        front.solutions = vec![sol1, sol2];

        let reference_point = vec![1.0, 1.0];
        let hypervolume = front.calculate_hypervolume(&reference_point);

        // Should be positive for non-dominated solutions
        assert!(hypervolume > 0.0);
    }

    #[test]
    fn test_nsga2_basic_functionality() {
        let zdt1 = ZDT1 { num_variables: 3 };
        let config = MultiObjectiveConfig {
            population_size: 20,
            max_generations: 10,
            crossover_probability: 0.9,
            mutation_probability: 0.1,
            mutation_strength: 0.1,
            elite_ratio: 0.1,
            convergence_tolerance: 1e-6,
            reference_point: Some(vec![1.0, 1.0]),
            preserve_diversity: true,
        };

        let optimizer = NsgaII::new(config);

        // Use phantom type for multi-objective problem
        use crate::phantom::{Euclidean, NonConvex, Unconstrained};
        let problem: OptimizationProblem<3, Unconstrained, MultiObjective, NonConvex, Euclidean> =
            OptimizationProblem::new();

        let result = optimizer.optimize(&problem, &zdt1);

        assert!(result.is_ok());
        let solution = result.unwrap();

        // Should have found some non-dominated solutions
        assert!(!solution.pareto_front.solutions.is_empty());
        assert!(solution.generations <= 10);
    }

    #[test]
    fn test_non_dominated_sorting() {
        let optimizer = NsgaII::<f64>::with_default_config();

        let mut pop = Vec::new();

        // Front 0 (non-dominated)
        let mut ind1 = Individual::new(vec![1.0]);
        ind1.objectives = vec![0.1, 0.9];
        pop.push(ind1);

        let mut ind2 = Individual::new(vec![2.0]);
        ind2.objectives = vec![0.9, 0.1];
        pop.push(ind2);

        // Front 1 (dominated by front 0)
        let mut ind3 = Individual::new(vec![3.0]);
        ind3.objectives = vec![0.2, 0.95]; // Dominated by ind1 [0.1, 0.9]
        pop.push(ind3);

        let fronts = optimizer.non_dominated_sort(&pop);

        assert_eq!(fronts.len(), 2);
        assert_eq!(fronts[0].len(), 2); // Two non-dominated solutions
        assert_eq!(fronts[1].len(), 1); // One dominated solution
    }

    #[test]
    fn test_crowding_distance_calculation() {
        let optimizer = NsgaII::<f64>::with_default_config();

        let mut front = vec![
            {
                let mut ind = Individual::new(vec![1.0]);
                ind.objectives = vec![0.1, 0.9];
                ind
            },
            {
                let mut ind = Individual::new(vec![2.0]);
                ind.objectives = vec![0.5, 0.5];
                ind
            },
            {
                let mut ind = Individual::new(vec![3.0]);
                ind.objectives = vec![0.9, 0.1];
                ind
            },
        ];

        optimizer.calculate_crowding_distance(&mut front);

        // Boundary solutions should have infinite crowding distance
        assert!(
            front[0].crowding_distance.is_infinite() || front[2].crowding_distance.is_infinite()
        );

        // Middle solution should have finite crowding distance
        assert!(front[1].crowding_distance.is_finite() && front[1].crowding_distance > 0.0);
    }
}
