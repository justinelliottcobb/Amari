//! Natural Gradient Optimization Example
//!
//! This example demonstrates how to use the natural gradient optimizer
//! for information geometric optimization on statistical manifolds.

use amari_optimization::prelude::*;

/// Example exponential family distribution for natural gradient optimization
struct GaussianExponentialFamily {
    data: Vec<f64>,
}

impl GaussianExponentialFamily {
    fn new(data: Vec<f64>) -> Self {
        Self { data }
    }
}

impl ObjectiveWithFisher<f64> for GaussianExponentialFamily {
    fn evaluate(&self, params: &[f64]) -> f64 {
        let mu = params[0];
        let log_sigma = params[1];
        let sigma = log_sigma.exp();

        // Negative log-likelihood for Gaussian distribution
        let mut nll = 0.0;
        for &x in &self.data {
            let diff = x - mu;
            nll += 0.5 * (diff * diff / (sigma * sigma) + 2.0 * log_sigma);
        }
        nll + 0.5 * self.data.len() as f64 * (2.0 * std::f64::consts::PI).ln()
    }

    fn gradient(&self, params: &[f64]) -> Vec<f64> {
        let mu = params[0];
        let log_sigma = params[1];
        let sigma = log_sigma.exp();
        let sigma2 = sigma * sigma;

        let _n = self.data.len() as f64;
        let mut grad_mu = 0.0;
        let mut grad_log_sigma = 0.0;

        for &x in &self.data {
            let diff = x - mu;
            grad_mu += diff / sigma2;
            grad_log_sigma += (diff * diff / sigma2) - 1.0;
        }

        vec![grad_mu, grad_log_sigma]
    }

    fn fisher_information(&self, params: &[f64]) -> Vec<Vec<f64>> {
        let log_sigma = params[1];
        let sigma = log_sigma.exp();
        let sigma2 = sigma * sigma;
        let n = self.data.len() as f64;

        // Fisher information matrix for Gaussian distribution
        vec![vec![n / sigma2, 0.0], vec![0.0, 2.0 * n]]
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Natural Gradient Optimization Example");
    println!("=====================================");

    // Generate some sample data from a Gaussian distribution
    let true_mu = 2.0;
    let true_sigma = 1.5;
    let data: Vec<f64> = (0..100)
        .map(|i| {
            let u1: f64 = (i as f64 + 1.0) / 101.0; // Avoid 0 and 1
            let u2: f64 = ((i * 17 + 42) % 100 + 1) as f64 / 101.0;

            // Box-Muller transform for normal distribution
            let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
            true_mu + true_sigma * z
        })
        .collect();

    println!(
        "Generated {} data points from N({}, {}²)",
        data.len(),
        true_mu,
        true_sigma
    );

    // Set up the optimization problem
    let objective = GaussianExponentialFamily::new(data);

    // Configure natural gradient optimizer
    let config = NaturalGradientConfig {
        learning_rate: 0.1,
        max_iterations: 100,
        gradient_tolerance: 1e-6,
        parameter_tolerance: 1e-8,
        fisher_regularization: 1e-6,
        use_line_search: false,
        line_search_beta: 0.5,
        line_search_alpha: 1.0,
    };

    let optimizer = NaturalGradientOptimizer::new(config);

    // Create phantom type for statistical manifold optimization
    use amari_optimization::phantom::{NonConvex, SingleObjective, Statistical, Unconstrained};
    let problem: OptimizationProblem<2, Unconstrained, SingleObjective, NonConvex, Statistical> =
        OptimizationProblem::new();

    // Initial parameter estimates (mu, log_sigma)
    let initial_params: Vec<f64> = vec![0.0, 0.0]; // Start with mu=0, sigma=1

    println!("\nStarting optimization...");
    println!(
        "Initial parameters: μ = {:.3}, σ = {:.3}",
        initial_params[0],
        initial_params[1].exp()
    );

    // Perform optimization
    let result = optimizer.optimize_statistical(&problem, &objective, initial_params)?;

    // Display results
    println!("\nOptimization completed!");
    println!("Converged: {}", result.converged);
    println!("Iterations: {}", result.iterations);
    println!("Final gradient norm: {:.2e}", result.gradient_norm);

    let final_mu = result.parameters[0];
    let final_sigma = result.parameters[1].exp();

    println!("\nFinal estimates:");
    println!(
        "μ = {:.6} (true: {:.6}, error: {:.6})",
        final_mu,
        true_mu,
        (final_mu - true_mu).abs()
    );
    println!(
        "σ = {:.6} (true: {:.6}, error: {:.6})",
        final_sigma,
        true_sigma,
        (final_sigma - true_sigma).abs()
    );

    println!(
        "\nFinal negative log-likelihood: {:.6}",
        result.objective_value
    );

    // Demonstrate the advantage of natural gradient
    println!("\nNatural gradient optimization uses the Fisher information matrix");
    println!("to provide invariance under reparameterization and typically");
    println!("converges faster than standard gradient descent on statistical manifolds.");

    Ok(())
}
