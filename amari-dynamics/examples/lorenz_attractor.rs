//! Lorenz Attractor Example
//!
//! This example demonstrates the famous Lorenz system, a classic example
//! of deterministic chaos. The system was originally developed by Edward
//! Lorenz as a simplified model of atmospheric convection.
//!
//! # The Lorenz Equations
//!
//! ```text
//! dx/dt = σ(y - x)
//! dy/dt = x(ρ - z) - y
//! dz/dt = xy - βz
//! ```
//!
//! With classic parameters σ=10, ρ=28, β=8/3, the system exhibits chaotic
//! behavior and the trajectory traces out the famous "butterfly" attractor.
//!
//! # Run this example
//!
//! ```bash
//! cargo run --example lorenz_attractor
//! ```

use amari_core::Multivector;
use amari_dynamics::{LorenzSystem, ODESolver, RungeKutta4, Trajectory};

fn main() {
    println!("=== Lorenz Attractor Demonstration ===\n");

    // Create the classic Lorenz system (σ=10, ρ=28, β=8/3)
    let lorenz = LorenzSystem::classic();
    println!("System parameters:");
    println!("  σ (sigma) = {:.2}", lorenz.sigma);
    println!("  ρ (rho)   = {:.2}", lorenz.rho);
    println!("  β (beta)  = {:.4}\n", lorenz.beta);

    // Create initial condition
    let mut initial: Multivector<3, 0, 0> = Multivector::zero();
    initial.set(1, 1.0); // x = 1
    initial.set(2, 1.0); // y = 1
    initial.set(4, 1.0); // z = 1
    println!(
        "Initial condition: ({:.1}, {:.1}, {:.1})",
        initial.get(1),
        initial.get(2),
        initial.get(4)
    );

    // Create RK4 solver
    let solver = RungeKutta4::new();
    let t0 = 0.0;
    let t1 = 50.0;
    let steps = 5000;

    println!("\nIntegration parameters:");
    println!("  t0 = {}", t0);
    println!("  t1 = {}", t1);
    println!("  steps = {}", steps);
    println!("  dt = {:.4}\n", (t1 - t0) / steps as f64);

    // Integrate the trajectory
    println!("Computing trajectory...");
    let trajectory: Trajectory<3, 0, 0> = solver
        .solve(&lorenz, initial.clone(), t0, t1, steps)
        .expect("Integration failed");

    println!("Trajectory computed: {} points\n", trajectory.len());

    // Analyze the attractor
    println!("=== Attractor Statistics ===\n");

    // Skip transient (first 20% of points)
    let skip = trajectory.len() / 5;

    // Compute bounds
    let mut x_min = f64::INFINITY;
    let mut x_max = f64::NEG_INFINITY;
    let mut y_min = f64::INFINITY;
    let mut y_max = f64::NEG_INFINITY;
    let mut z_min = f64::INFINITY;
    let mut z_max = f64::NEG_INFINITY;

    let mut x_sum = 0.0;
    let mut y_sum = 0.0;
    let mut z_sum = 0.0;
    let mut count = 0;

    for (i, (_, state)) in trajectory.iter().enumerate() {
        if i < skip {
            continue;
        }
        let x = state.get(1);
        let y = state.get(2);
        let z = state.get(4);
        x_min = x_min.min(x);
        x_max = x_max.max(x);
        y_min = y_min.min(y);
        y_max = y_max.max(y);
        z_min = z_min.min(z);
        z_max = z_max.max(z);
        x_sum += x;
        y_sum += y;
        z_sum += z;
        count += 1;
    }

    println!("Attractor bounds (after transient):");
    println!("  x: [{:.2}, {:.2}]", x_min, x_max);
    println!("  y: [{:.2}, {:.2}]", y_min, y_max);
    println!("  z: [{:.2}, {:.2}]", z_min, z_max);

    // Compute mean position (approximate center of attractor)
    let n = count as f64;
    let x_mean = x_sum / n;
    let y_mean = y_sum / n;
    let z_mean = z_sum / n;

    println!("\nApproximate attractor center:");
    println!("  ({:.2}, {:.2}, {:.2})", x_mean, y_mean, z_mean);

    // Demonstrate sensitivity to initial conditions (butterfly effect)
    println!("\n=== Sensitivity to Initial Conditions ===\n");

    let epsilon = 1e-10;
    let mut perturbed = initial.clone();
    perturbed.set(1, initial.get(1) + epsilon);

    println!("Perturbing x by ε = {:.0e}", epsilon);

    // Integrate both trajectories
    let traj1 = solver.solve(&lorenz, initial, t0, t1, steps).unwrap();
    let traj2 = solver.solve(&lorenz, perturbed, t0, t1, steps).unwrap();

    // Compute separation over time
    println!("\nTrajectory separation:");
    let check_points = [0, 1000, 2000, 3000, 4000, 5000];
    for &i in &check_points {
        if let (Some((t, s1)), Some((_, s2))) = (traj1.get(i), traj2.get(i)) {
            let dx = s1.get(1) - s2.get(1);
            let dy = s1.get(2) - s2.get(2);
            let dz = s1.get(4) - s2.get(4);
            let dist = (dx * dx + dy * dy + dz * dz).sqrt();
            println!("  t = {:.1}s: separation = {:.2e}", t, dist);
        }
    }

    // Show fixed points
    println!("\n=== Fixed Points ===\n");
    println!("The Lorenz system has three fixed points:");
    println!("  Origin: (0, 0, 0) - saddle point");

    let c1_x = (lorenz.beta * (lorenz.rho - 1.0)).sqrt();
    let c1_y = c1_x;
    let c1_z = lorenz.rho - 1.0;
    println!("  C+: ({:.2}, {:.2}, {:.2})", c1_x, c1_y, c1_z);
    println!("  C-: ({:.2}, {:.2}, {:.2})", -c1_x, -c1_y, c1_z);

    println!(
        "\nFor ρ > 24.74 (ρ = {:.2}), the C+/C- fixed points are unstable,",
        lorenz.rho
    );
    println!("and the system exhibits chaotic behavior on the strange attractor.");

    println!("\n=== Example Complete ===");
}
