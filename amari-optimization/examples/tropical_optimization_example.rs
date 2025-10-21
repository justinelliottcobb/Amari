//! Tropical Optimization Example
//!
//! This example demonstrates how to use tropical optimization algorithms
//! for solving scheduling and resource allocation problems using max-plus algebra.

use amari_optimization::prelude::*;
use amari_tropical::{TropicalMatrix, TropicalNumber};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Tropical Optimization Example");
    println!("=============================");
    println!("Using max-plus algebra for scheduling optimization\n");

    // Example: Task scheduling problem
    // We have 4 tasks with dependencies and want to find optimal start times
    println!("Problem: Task Scheduling with Dependencies");
    println!("-----------------------------------------");

    // Task durations (in hours)
    let task_durations = vec![2.0, 3.0, 1.0, 4.0];
    println!("Task durations: {:?} hours", task_durations);

    // Dependency matrix: entry (i,j) represents the minimum delay
    // from completion of task i to start of task j
    let dependency_data = vec![
        vec![0.0, 2.0, f64::NEG_INFINITY, 1.0], // Task 0 dependencies
        vec![f64::NEG_INFINITY, 0.0, 1.0, 2.0], // Task 1 dependencies
        vec![3.0, f64::NEG_INFINITY, 0.0, 1.0], // Task 2 dependencies
        vec![f64::NEG_INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY, 0.0], // Task 3 dependencies
    ];

    println!("Dependency matrix (delays between tasks):");
    for (i, row) in dependency_data.iter().enumerate() {
        println!("  Task {}: {:?}", i, row);
    }

    // Convert to tropical matrix
    let dependency_matrix = TropicalMatrix::from_log_probs(&dependency_data);

    // Set up the tropical optimizer
    let optimizer = TropicalOptimizer::with_default_config();

    // Solve tropical eigenvalue problem to find the critical cycle
    println!("\nSolving tropical eigenvalue problem...");
    match optimizer.solve_tropical_eigenvalue(&dependency_matrix) {
        Ok(eigen_result) => {
            if let Some(eigenvalue) = eigen_result.eigenvalue {
                println!("Critical cycle length: {:?} hours", eigenvalue);
                println!(
                    "Optimal start times (relative): {:?}",
                    eigen_result.solution
                );
            }
        }
        Err(e) => println!("Eigenvalue computation: {:?}", e),
    }

    // Example: Resource allocation using tropical linear programming
    println!("\n{}", "=".repeat(50));
    println!("Problem: Resource Allocation");
    println!("---------------------------");

    // We want to minimize the maximum resource usage across different time slots
    // Objective: minimize max(r1, r2, r3) where ri is resource usage in slot i
    let objective = vec![
        TropicalNumber::new(1.0), // Weight for resource slot 1
        TropicalNumber::new(1.0), // Weight for resource slot 2
        TropicalNumber::new(1.0), // Weight for resource slot 3
    ];

    // Constraints: Each task must be assigned to exactly one time slot
    // and resource capacity constraints must be satisfied
    let constraint_data = vec![
        vec![2.0, 3.0, 4.0], // Resource requirements for task A in each slot
        vec![1.0, 2.0, 3.0], // Resource requirements for task B in each slot
        vec![3.0, 1.0, 2.0], // Resource requirements for task C in each slot
    ];

    let constraint_matrix = TropicalMatrix::from_log_probs(&constraint_data);

    // Right-hand side: maximum available resources in each slot
    let constraint_rhs = vec![
        TropicalNumber::new(5.0), // Max resources in slot 1
        TropicalNumber::new(4.0), // Max resources in slot 2
        TropicalNumber::new(6.0), // Max resources in slot 3
    ];

    println!("Resource requirements matrix:");
    for (i, row) in constraint_data.iter().enumerate() {
        println!("  Task {}: {:?}", ['A', 'B', 'C'][i], row);
    }
    println!("Resource capacity limits: {:?}", [5.0, 4.0, 6.0]);

    // Solve tropical linear program
    println!("\nSolving tropical linear program...");
    match optimizer.solve_tropical_linear_program(&objective, &constraint_matrix, &constraint_rhs) {
        Ok(lp_result) => {
            println!("Optimization converged: {}", lp_result.converged);
            println!("Iterations: {}", lp_result.iterations);
            println!("Optimal objective value: {:?}", lp_result.objective_value);

            println!("\nOptimal assignment (tropical solution):");
            for (i, val) in lp_result.solution.iter().enumerate() {
                let assignment = if val.is_zero() {
                    "Not assigned".to_string()
                } else {
                    format!("Slot {} (weight: {:?})", i + 1, val)
                };
                println!("  Variable {}: {}", i, assignment);
            }
        }
        Err(e) => println!("Linear program failed: {:?}", e),
    }

    // Example: Shortest path problem using tropical algebra
    println!("\n{}", "=".repeat(50));
    println!("Problem: Shortest Path");
    println!("---------------------");

    // Distance matrix for a simple graph (4 nodes)
    let distances = [
        vec![0.0, 3.0, f64::INFINITY, 7.0],
        vec![f64::INFINITY, 0.0, 1.0, 2.0],
        vec![f64::INFINITY, f64::INFINITY, 0.0, 4.0],
        vec![f64::INFINITY, f64::INFINITY, f64::INFINITY, 0.0],
    ];

    println!("Distance matrix:");
    for (i, row) in distances.iter().enumerate() {
        println!("  Node {}: {:?}", i, row);
    }

    // Convert distances to tropical representation (negate for min-plus)
    let tropical_distances: Vec<Vec<f64>> = distances
        .iter()
        .map(|row| {
            row.iter()
                .map(|&d| {
                    if d == f64::INFINITY {
                        f64::NEG_INFINITY
                    } else {
                        -d
                    }
                })
                .collect()
        })
        .collect();

    let path_matrix = TropicalMatrix::from_log_probs(&tropical_distances);

    match optimizer.solve_shortest_path(&path_matrix, 0, 3) {
        Ok(path_result) => {
            println!("\nShortest path from node 0 to node 3:");
            println!("Distance: {:?}", path_result.objective_value);
            println!("Path: {:?}", path_result.solution);
        }
        Err(e) => println!("Shortest path computation failed: {:?}", e),
    }

    println!("\n{}", "=".repeat(50));
    println!("Tropical algebra uses max-plus operations where:");
    println!("• Addition becomes max: a ⊕ b = max(a, b)");
    println!("• Multiplication becomes addition: a ⊗ b = a + b");
    println!("• This is ideal for scheduling, path, and timing problems");
    println!("• The tropical eigenvalue gives the critical cycle time");
    println!("• Tropical linear programming finds optimal resource allocation");

    Ok(())
}
