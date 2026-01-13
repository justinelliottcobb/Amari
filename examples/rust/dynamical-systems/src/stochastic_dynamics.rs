//! # Stochastic Dynamics Example
//!
//! Demonstrates stochastic dynamical systems including Langevin dynamics,
//! noise-induced transitions, and the Fokker-Planck equation.
//!
//! ## Mathematical Background
//!
//! Stochastic differential equations (SDEs):
//! ```text
//! dx = f(x)dt + g(x)dW
//! ```
//!
//! The Fokker-Planck equation describes probability density evolution:
//! ```text
//! ∂p/∂t = -∂/∂x[f(x)p] + (D/2)∂²p/∂x²
//! ```
//!
//! Run with: `cargo run --bin stochastic_dynamics`

use amari_core::Multivector;
use amari_dynamics::{
    stochastic::{
        LangevinDynamics, LangevinConfig,
        FokkerPlanckSolver, FokkerPlanckConfig,
        NoiseInducedTransition, TransitionConfig,
    },
    systems::DuffingOscillator,
};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("═══════════════════════════════════════════════════════════════");
    println!("                  STOCHASTIC DYNAMICS DEMO");
    println!("═══════════════════════════════════════════════════════════════\n");

    // Use seeded RNG for reproducibility
    let mut rng = ChaCha8Rng::seed_from_u64(42);

    // =========================================================================
    // Part 1: Langevin Dynamics in a Double-Well
    // =========================================================================
    println!("Part 1: Langevin Dynamics in Double-Well Potential");
    println!("───────────────────────────────────────────────────\n");

    println!("System: dx = -dV/dx dt + √(2D) dW");
    println!("Potential: V(x) = -x²/2 + x⁴/4  (double-well)");
    println!("Minima at x = ±1, barrier at x = 0\n");

    // Define double-well potential derivative
    let drift = |x: f64| -> f64 { x - x.powi(3) };  // -dV/dx

    let config = LangevinConfig {
        dt: 0.01,
        num_steps: 10000,
        noise_strength: 0.1,  // D = 0.1 (low noise)
    };

    // Run ensemble of trajectories
    let num_trajectories = 50;
    let mut final_positions_low_noise: Vec<f64> = Vec::new();

    println!("Low noise (D=0.1): Running {} trajectories from x₀=0.1", num_trajectories);

    for _ in 0..num_trajectories {
        let langevin = LangevinDynamics::new(drift, config.clone());
        let trajectory = langevin.simulate(0.1, &mut rng)?;
        final_positions_low_noise.push(*trajectory.last().unwrap());
    }

    // Count which well each trajectory ended in
    let in_left = final_positions_low_noise.iter().filter(|&&x| x < 0.0).count();
    let in_right = num_trajectories - in_left;

    println!("  Final distribution: {} left well, {} right well",
             in_left, in_right);

    // Higher noise - more transitions
    let config_high = LangevinConfig {
        dt: 0.01,
        num_steps: 10000,
        noise_strength: 0.5,  // D = 0.5 (higher noise)
    };

    let mut final_positions_high_noise: Vec<f64> = Vec::new();

    println!("\nHigh noise (D=0.5): Running {} trajectories from x₀=0.1", num_trajectories);

    for _ in 0..num_trajectories {
        let langevin = LangevinDynamics::new(drift, config_high.clone());
        let trajectory = langevin.simulate(0.1, &mut rng)?;
        final_positions_high_noise.push(*trajectory.last().unwrap());
    }

    let in_left_high = final_positions_high_noise.iter().filter(|&&x| x < 0.0).count();
    let in_right_high = num_trajectories - in_left_high;

    println!("  Final distribution: {} left well, {} right well",
             in_left_high, in_right_high);

    println!("\n  Higher noise → more frequent inter-well transitions");
    println!("  → more equilibrated distribution");

    // =========================================================================
    // Part 2: Single Trajectory Analysis
    // =========================================================================
    println!("\n\nPart 2: Single Trajectory Analysis");
    println!("───────────────────────────────────\n");

    let config_long = LangevinConfig {
        dt: 0.01,
        num_steps: 50000,
        noise_strength: 0.3,
    };

    let langevin = LangevinDynamics::new(drift, config_long);
    let trajectory = langevin.simulate(-1.0, &mut rng)?;  // Start in left well

    // Count well transitions
    let mut transitions = 0;
    let mut last_well = -1;  // -1 = left, 1 = right
    let mut residence_times: Vec<(i32, usize)> = Vec::new();
    let mut current_residence = 0;

    for &x in &trajectory {
        current_residence += 1;
        let current_well = if x < 0.0 { -1 } else { 1 };
        if current_well != last_well {
            if last_well != 0 {  // Not the first point
                transitions += 1;
                residence_times.push((last_well, current_residence));
                current_residence = 0;
            }
            last_well = current_well;
        }
    }

    println!("Single trajectory (D=0.3, 50000 steps, starting in left well):");
    println!("  Total transitions: {}", transitions);

    // Compute mean residence times
    let left_times: Vec<_> = residence_times.iter()
        .filter(|(w, _)| *w == -1)
        .map(|(_, t)| *t)
        .collect();
    let right_times: Vec<_> = residence_times.iter()
        .filter(|(w, _)| *w == 1)
        .map(|(_, t)| *t)
        .collect();

    if !left_times.is_empty() {
        let mean_left: f64 = left_times.iter().sum::<usize>() as f64 / left_times.len() as f64;
        println!("  Mean residence time (left well): {:.1} steps", mean_left);
    }
    if !right_times.is_empty() {
        let mean_right: f64 = right_times.iter().sum::<usize>() as f64 / right_times.len() as f64;
        println!("  Mean residence time (right well): {:.1} steps", mean_right);
    }

    // Time-averaged statistics
    let mean_x: f64 = trajectory.iter().sum::<f64>() / trajectory.len() as f64;
    let mean_x2: f64 = trajectory.iter().map(|x| x*x).sum::<f64>() / trajectory.len() as f64;
    let variance = mean_x2 - mean_x * mean_x;

    println!("\n  Time-averaged statistics:");
    println!("    <x> = {:.4} (symmetric potential → should be ~0)", mean_x);
    println!("    <x²> = {:.4}", mean_x2);
    println!("    Var(x) = {:.4}", variance);

    // =========================================================================
    // Part 3: Fokker-Planck Equation
    // =========================================================================
    println!("\n\nPart 3: Fokker-Planck Equation");
    println!("───────────────────────────────\n");

    println!("The Fokker-Planck equation describes how probability density evolves:");
    println!("  ∂p/∂t = -∂/∂x[(x-x³)p] + D ∂²p/∂x²\n");

    let fp_config = FokkerPlanckConfig {
        x_min: -3.0,
        x_max: 3.0,
        nx: 200,
        dt: 0.001,
        diffusion: 0.3,
    };

    let mut fp_solver = FokkerPlanckSolver::new(drift, fp_config)?;

    // Initial condition: Gaussian centered at x = -1
    fp_solver.set_gaussian_initial(-1.0, 0.2)?;

    println!("Initial condition: Gaussian at x=-1 (σ=0.2)");
    println!("Evolving probability density...\n");

    // Evolve and report
    let times = [0.0, 1.0, 5.0, 20.0, 100.0];

    println!("  Time  | Peak position | Peak height | Spread (std)");
    println!("  ──────┼───────────────┼─────────────┼─────────────");

    for &t in &times {
        if t > 0.0 {
            fp_solver.evolve_to(t)?;
        }

        let stats = fp_solver.get_statistics()?;
        println!("  {:5.1} | {:13.3} | {:11.4} | {:11.4}",
                 t, stats.peak_position, stats.peak_height, stats.std_dev);
    }

    println!("\n  Stationary distribution is bimodal (two peaks at ±1)");

    // =========================================================================
    // Part 4: Kramers Escape Rate
    // =========================================================================
    println!("\n\nPart 4: Kramers Escape Rate Theory");
    println!("────────────────────────────────────\n");

    println!("Kramers formula for escape rate from a metastable well:");
    println!("  k = (ω_a ω_b / 2π) exp(-ΔV / D)");
    println!("  where ω_a, ω_b are frequencies at minimum and barrier\n");

    // For our double-well: V(x) = -x²/2 + x⁴/4
    // Minimum at x=1: V(1) = -1/2 + 1/4 = -1/4
    // Barrier at x=0: V(0) = 0
    // ΔV = 0 - (-1/4) = 1/4
    // ω_a² = V''(1) = -1 + 3 = 2, so ω_a = √2
    // ω_b² = |V''(0)| = 1, so ω_b = 1

    let delta_v = 0.25;
    let omega_a = 2.0_f64.sqrt();
    let omega_b = 1.0;

    println!("For our double-well (V = -x²/2 + x⁴/4):");
    println!("  Barrier height ΔV = {:.2}", delta_v);
    println!("  ω_a = {:.3}, ω_b = {:.3}", omega_a, omega_b);

    println!("\nPredicted escape rates:");
    println!("  D     | k (Kramers)  | τ = 1/k (mean escape time)");
    println!("  ──────┼──────────────┼────────────────────────────");

    for d in [0.1, 0.2, 0.3, 0.5, 1.0] {
        let k = (omega_a * omega_b / (2.0 * std::f64::consts::PI))
                * (-delta_v / d).exp();
        let tau = 1.0 / k;
        println!("  {:5.2} | {:12.4e} | {:12.2}", d, k, tau);
    }

    println!("\n  Lower noise → exponentially longer escape times");

    // =========================================================================
    // Part 5: Noise-Induced Phenomena
    // =========================================================================
    println!("\n\nPart 5: Noise-Induced Phenomena");
    println!("────────────────────────────────\n");

    // Stochastic resonance setup
    println!("Stochastic Resonance:");
    println!("  Adding weak periodic forcing to a bistable system,");
    println!("  noise can enhance the signal (counterintuitively).\n");

    // Noise-induced transitions
    let transition_config = TransitionConfig {
        num_trials: 100,
        max_time: 1000.0,
        dt: 0.01,
    };

    println!("Measuring first passage times from x=-1 to x>0.5:");
    println!("  D     | Mean FPT   | Std FPT    | Success rate");
    println!("  ──────┼────────────┼────────────┼─────────────");

    for d in [0.1, 0.2, 0.3, 0.5] {
        let config = LangevinConfig {
            dt: 0.01,
            num_steps: 100000,
            noise_strength: d,
        };

        let mut fpts: Vec<f64> = Vec::new();
        let mut successes = 0;

        for _ in 0..transition_config.num_trials {
            let langevin = LangevinDynamics::new(drift, config.clone());
            let trajectory = langevin.simulate(-1.0, &mut rng)?;

            // Find first passage time to x > 0.5
            for (i, &x) in trajectory.iter().enumerate() {
                if x > 0.5 {
                    fpts.push(i as f64 * config.dt);
                    successes += 1;
                    break;
                }
            }
        }

        if fpts.is_empty() {
            println!("  {:5.2} |    N/A     |    N/A     | {:4}/{:4}",
                     d, successes, transition_config.num_trials);
        } else {
            let mean_fpt: f64 = fpts.iter().sum::<f64>() / fpts.len() as f64;
            let var: f64 = fpts.iter().map(|t| (t - mean_fpt).powi(2)).sum::<f64>()
                          / fpts.len() as f64;
            let std_fpt = var.sqrt();

            println!("  {:5.2} | {:10.2} | {:10.2} | {:4}/{:4}",
                     d, mean_fpt, std_fpt, successes, transition_config.num_trials);
        }
    }

    println!("\n═══════════════════════════════════════════════════════════════");
    println!("                    DEMO COMPLETE");
    println!("═══════════════════════════════════════════════════════════════");

    Ok(())
}
