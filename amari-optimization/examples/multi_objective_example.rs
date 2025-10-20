//! Multi-Objective Optimization Example
//!
//! This example demonstrates how to use NSGA-II for multi-objective optimization
//! to find Pareto-optimal solutions with conflicting objectives.

use amari_optimization::prelude::*;

/// Engineering design problem: Minimize weight and cost while maximizing strength
/// This represents a typical engineering trade-off scenario
struct EngineeringDesign {
    material_density: f64,
    material_cost_per_kg: f64,
    strength_factor: f64,
}

impl EngineeringDesign {
    fn new() -> Self {
        Self {
            material_density: 2.7,     // kg/m³ (aluminum-like)
            material_cost_per_kg: 3.5, // $/kg
            strength_factor: 100.0,    // MPa
        }
    }
}

impl MultiObjectiveFunction<f64> for EngineeringDesign {
    fn num_objectives(&self) -> usize {
        3 // weight, cost, negative strength (for minimization)
    }

    fn num_variables(&self) -> usize {
        3 // thickness, width, height
    }

    fn evaluate(&self, variables: &[f64]) -> Vec<f64> {
        let thickness = variables[0];
        let width = variables[1];
        let height = variables[2];

        // Calculate volume
        let volume = thickness * width * height;

        // Objective 1: Weight (minimize)
        let weight = volume * self.material_density;

        // Objective 2: Cost (minimize)
        let cost = weight * self.material_cost_per_kg;

        // Objective 3: Negative strength (minimize = maximize strength)
        // Strength increases with thickness and area, but has diminishing returns
        let cross_section_area = width * height;
        let strength = self.strength_factor * thickness.sqrt() * cross_section_area.powf(0.3);
        let negative_strength = -strength;

        vec![weight, cost, negative_strength]
    }

    fn variable_bounds(&self) -> Vec<(f64, f64)> {
        vec![
            (0.001, 0.1), // thickness: 1mm to 10cm
            (0.1, 2.0),   // width: 10cm to 2m
            (0.1, 2.0),   // height: 10cm to 2m
        ]
    }

    fn ideal_point(&self) -> Option<Vec<f64>> {
        // Ideally we want zero weight, zero cost, and maximum strength
        Some(vec![0.0, 0.0, f64::NEG_INFINITY])
    }
}

/// Environmental optimization problem: Balance economic and environmental objectives
struct EnvironmentalOptimization;

impl MultiObjectiveFunction<f64> for EnvironmentalOptimization {
    fn num_objectives(&self) -> usize {
        2 // economic cost, environmental impact
    }

    fn num_variables(&self) -> usize {
        4 // production levels for 4 different processes
    }

    fn evaluate(&self, variables: &[f64]) -> Vec<f64> {
        let production = variables;

        // Objective 1: Economic cost (minimize)
        // Different processes have different costs per unit
        let unit_costs = [10.0, 15.0, 20.0, 8.0];
        let economic_cost: f64 = production
            .iter()
            .zip(unit_costs.iter())
            .map(|(&prod, &cost)| prod * cost)
            .sum();

        // Objective 2: Environmental impact (minimize)
        // Different processes have different environmental impacts
        let environmental_factors = [2.0, 1.5, 0.8, 3.0];
        let environmental_impact: f64 = production
            .iter()
            .zip(environmental_factors.iter())
            .map(|(&prod, &impact)| prod * impact)
            .sum();

        vec![economic_cost, environmental_impact]
    }

    fn evaluate_constraints(&self, variables: &[f64]) -> Vec<f64> {
        // Constraint: Total production must meet demand
        let total_production: f64 = variables.iter().sum();
        let demand = 100.0;

        // Return constraint violation (≤ 0 means feasible)
        vec![demand - total_production] // total_production >= demand
    }

    fn variable_bounds(&self) -> Vec<(f64, f64)> {
        vec![(0.0, 100.0); 4] // Each process can produce 0 to 100 units
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Multi-Objective Optimization Example");
    println!("====================================");

    // Example 1: Engineering Design Optimization
    println!("\nExample 1: Engineering Design Trade-offs");
    println!("----------------------------------------");
    println!("Optimizing: thickness, width, height of a structural component");
    println!("Objectives: minimize weight, minimize cost, maximize strength");

    let design_problem = EngineeringDesign::new();

    // Configure NSGA-II
    let config = MultiObjectiveConfig {
        population_size: 50,
        max_generations: 100,
        crossover_probability: 0.9,
        mutation_probability: 0.1,
        mutation_strength: 0.1,
        elite_ratio: 0.1,
        convergence_tolerance: 1e-6,
        reference_point: Some(vec![10.0, 50.0, -200.0]), // For hypervolume calculation
        preserve_diversity: true,
    };

    let nsga2 = NsgaII::new(config);

    // Create phantom type for multi-objective problem
    use amari_optimization::phantom::{Euclidean, NonConvex, Unconstrained};
    let problem: OptimizationProblem<3, Unconstrained, MultiObjective, NonConvex, Euclidean> =
        OptimizationProblem::new();

    println!("\nRunning NSGA-II optimization...");
    let result = nsga2.optimize(&problem, &design_problem)?;

    println!("Optimization completed!");
    println!("Converged: {}", result.converged);
    println!("Generations: {}", result.generations);
    println!("Pareto front size: {}", result.pareto_front.solutions.len());

    if let Some(hypervolume) = result.pareto_front.hypervolume {
        println!("Final hypervolume: {:.6}", hypervolume);
    }

    // Display some Pareto-optimal solutions
    println!("\nTop 5 Pareto-optimal designs:");
    println!("Thickness(m) | Width(m) | Height(m) | Weight(kg) | Cost($) | Strength(MPa)");
    println!("{}", "-".repeat(80));

    for (i, solution) in result.pareto_front.solutions.iter().take(5).enumerate() {
        let thickness = solution.variables[0];
        let width = solution.variables[1];
        let height = solution.variables[2];
        let weight = solution.objectives[0];
        let cost = solution.objectives[1];
        let strength = -solution.objectives[2]; // Convert back from negative

        println!(
            "{:2}: {:8.4} | {:7.3} | {:8.3} | {:9.3} | {:6.2} | {:11.2}",
            i + 1,
            thickness,
            width,
            height,
            weight,
            cost,
            strength
        );
    }

    // Example 2: Environmental vs Economic Trade-off
    println!("\n{}", "=".repeat(60));
    println!("Example 2: Environmental vs Economic Optimization");
    println!("------------------------------------------------");
    println!("Optimizing: production levels for 4 processes");
    println!("Objectives: minimize economic cost, minimize environmental impact");
    println!("Constraint: total production ≥ 100 units");

    let env_problem = EnvironmentalOptimization;

    let env_config = MultiObjectiveConfig {
        population_size: 40,
        max_generations: 50,
        crossover_probability: 0.9,
        mutation_probability: 0.15,
        mutation_strength: 0.2,
        elite_ratio: 0.1,
        convergence_tolerance: 1e-6,
        reference_point: Some(vec![2000.0, 300.0]), // For hypervolume calculation
        preserve_diversity: true,
    };

    let _env_nsga2 = NsgaII::new(env_config);

    // Create phantom type for constrained multi-objective problem
    use amari_optimization::phantom::Constrained;
    let env_problem_type: OptimizationProblem<
        4,
        Constrained,
        MultiObjective,
        NonConvex,
        Euclidean,
    > = OptimizationProblem::new();

    println!("\nRunning environmental optimization...");
    let env_result = nsga2.optimize(&env_problem_type, &env_problem)?;

    println!("Environmental optimization completed!");
    println!("Converged: {}", env_result.converged);
    println!("Generations: {}", env_result.generations);
    println!(
        "Pareto front size: {}",
        env_result.pareto_front.solutions.len()
    );

    println!("\nTop 5 Pareto-optimal production strategies:");
    println!("Process1 | Process2 | Process3 | Process4 | Total | Econ.Cost | Env.Impact");
    println!("{}", "-".repeat(75));

    for (i, solution) in env_result.pareto_front.solutions.iter().take(5).enumerate() {
        let p1 = solution.variables[0];
        let p2 = solution.variables[1];
        let p3 = solution.variables[2];
        let p4 = solution.variables[3];
        let total = p1 + p2 + p3 + p4;
        let cost = solution.objectives[0];
        let impact = solution.objectives[1];

        println!(
            "{:2}: {:7.1} | {:7.1} | {:7.1} | {:7.1} | {:5.1} | {:8.2} | {:9.2}",
            i + 1,
            p1,
            p2,
            p3,
            p4,
            total,
            cost,
            impact
        );
    }

    // Analysis of the Pareto front
    println!("\n{}", "=".repeat(60));
    println!("Pareto Front Analysis");
    println!("--------------------");

    if !env_result.pareto_front.solutions.is_empty() {
        let solutions = &env_result.pareto_front.solutions;

        let min_cost = solutions
            .iter()
            .map(|s| s.objectives[0])
            .fold(f64::INFINITY, f64::min);
        let max_cost = solutions
            .iter()
            .map(|s| s.objectives[0])
            .fold(f64::NEG_INFINITY, f64::max);

        let min_impact = solutions
            .iter()
            .map(|s| s.objectives[1])
            .fold(f64::INFINITY, f64::min);
        let max_impact = solutions
            .iter()
            .map(|s| s.objectives[1])
            .fold(f64::NEG_INFINITY, f64::max);

        println!("Cost range: {:.2} to {:.2}", min_cost, max_cost);
        println!(
            "Environmental impact range: {:.2} to {:.2}",
            min_impact, max_impact
        );
        println!(
            "Trade-off span: {:.1}% cost increase for {:.1}% impact reduction",
            ((max_cost - min_cost) / min_cost) * 100.0,
            ((max_impact - min_impact) / min_impact) * 100.0
        );

        println!("\nThe Pareto front represents optimal trade-offs where:");
        println!("• No solution can improve one objective without worsening another");
        println!("• Decision makers can choose based on their preferences");
        println!("• Each point represents a different compromise between objectives");
    }

    Ok(())
}
