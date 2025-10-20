use amari_optimization::prelude::*;

/// Test problem: Rosenbrock function with constraints
/// minimize (1-x)² + 100(y-x²)²
/// subject to: x² + y² ≤ 1, x + y ≥ 0
struct RosenbrockConstrained;

impl ConstrainedObjective<f64> for RosenbrockConstrained {
    fn evaluate(&self, x: &[f64]) -> f64 {
        let (x, y) = (x[0], x[1]);
        (1.0 - x).powi(2) + 100.0 * (y - x.powi(2)).powi(2)
    }

    fn gradient(&self, x: &[f64]) -> Vec<f64> {
        let (x_val, y_val) = (x[0], x[1]);
        vec![
            -2.0 * (1.0 - x_val) - 400.0 * x_val * (y_val - x_val.powi(2)),
            200.0 * (y_val - x_val.powi(2)),
        ]
    }

    fn inequality_constraints(&self, x: &[f64]) -> Vec<f64> {
        vec![
            x[0].powi(2) + x[1].powi(2) - 1.0, // x² + y² ≤ 1
            -(x[0] + x[1]),                    // x + y ≥ 0
        ]
    }

    fn equality_constraints(&self, _x: &[f64]) -> Vec<f64> {
        vec![] // No equality constraints
    }

    fn inequality_jacobian(&self, x: &[f64]) -> Vec<Vec<f64>> {
        vec![
            vec![2.0 * x[0], 2.0 * x[1]], // ∇(x² + y² - 1)
            vec![-1.0, -1.0],             // ∇(-(x + y))
        ]
    }

    fn equality_jacobian(&self, _x: &[f64]) -> Vec<Vec<f64>> {
        vec![]
    }

    fn variable_bounds(&self) -> Vec<(f64, f64)> {
        vec![(-2.0, 2.0), (-2.0, 2.0)]
    }

    fn num_inequality_constraints(&self) -> usize {
        2
    }

    fn num_equality_constraints(&self) -> usize {
        0
    }

    fn num_variables(&self) -> usize {
        2
    }
}

fn main() {
    let problem = RosenbrockConstrained;
    let test_point = vec![0.5, 0.5];
    
    println!("Testing point: [{}, {}]", test_point[0], test_point[1]);
    
    let constraints = problem.inequality_constraints(&test_point);
    println!("Constraint 1 (x² + y² - 1): {}", constraints[0]);
    println!("Constraint 2 (-(x + y)): {}", constraints[1]);
    
    println!("Is feasible?");
    println!("  x² + y² ≤ 1: {} ≤ 1 -> {}", 
             test_point[0].powi(2) + test_point[1].powi(2),
             constraints[0] <= 0.0);
    println!("  x + y ≥ 0: {} ≥ 0 -> {}", 
             test_point[0] + test_point[1],
             constraints[1] <= 0.0);
}
