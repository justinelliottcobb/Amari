//! # Lorenz Attractor Example
//!
//! Demonstrates the classic Lorenz system - one of the first examples of deterministic chaos.
//! The system models atmospheric convection and exhibits sensitive dependence on initial conditions.
//!
//! ## Mathematical Background
//!
//! The Lorenz equations are:
//! ```text
//! dx/dt = σ(y - x)
//! dy/dt = x(ρ - z) - y
//! dz/dt = xy - βz
//! ```
//!
//! With classic parameters σ=10, ρ=28, β=8/3, the system exhibits chaotic behavior
//! with a strange attractor having fractal dimension ~2.06.
//!
//! Run with: `cargo run --bin lorenz_attractor`

use amari_core::Multivector;
use amari_dynamics::{
    systems::LorenzSystem,
    solver::{RungeKutta4, ODESolver, DormandPrince},
    lyapunov::{LyapunovConfig, compute_lyapunov_spectrum},
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("═══════════════════════════════════════════════════════════════");
    println!("                    LORENZ ATTRACTOR DEMO");
    println!("═══════════════════════════════════════════════════════════════\n");

    // =========================================================================
    // Part 1: Basic Trajectory Integration
    // =========================================================================
    println!("Part 1: Basic Trajectory Integration");
    println!("─────────────────────────────────────\n");

    // Create classic Lorenz system with σ=10, ρ=28, β=8/3
    let lorenz = LorenzSystem::classic();
    println!("Lorenz parameters: σ={}, ρ={}, β={:.4}",
             lorenz.sigma(), lorenz.rho(), lorenz.beta());

    // Initial condition as a multivector in Cl(3,0,0)
    // We use basis vectors e₁, e₂, e₄ for x, y, z coordinates
    let mut initial: Multivector<3, 0, 0> = Multivector::zero();
    initial.set(1, 1.0);  // x = 1
    initial.set(2, 1.0);  // y = 1
    initial.set(4, 1.0);  // z = 1

    println!("Initial condition: ({}, {}, {})",
             initial.get(1), initial.get(2), initial.get(4));

    // Create RK4 solver and integrate
    let rk4 = RungeKutta4::new();
    let trajectory = rk4.solve(&lorenz, initial.clone(), 0.0, 50.0, 5000)?;

    println!("Integrated {} points over t ∈ [0, 50]", trajectory.len());

    // Sample some trajectory points
    println!("\nSample trajectory points:");
    for i in [0, 1000, 2000, 3000, 4000, 4999] {
        if let Some((t, state)) = trajectory.get(i) {
            println!("  t={:6.2}: ({:8.4}, {:8.4}, {:8.4})",
                     t, state.get(1), state.get(2), state.get(4));
        }
    }

    // =========================================================================
    // Part 2: Sensitive Dependence on Initial Conditions (Butterfly Effect)
    // =========================================================================
    println!("\n\nPart 2: Butterfly Effect");
    println!("─────────────────────────\n");

    // Create two initial conditions differing by 1e-10
    let mut initial1: Multivector<3, 0, 0> = Multivector::zero();
    initial1.set(1, 1.0);
    initial1.set(2, 1.0);
    initial1.set(4, 1.0);

    let mut initial2: Multivector<3, 0, 0> = Multivector::zero();
    initial2.set(1, 1.0 + 1e-10);  // Tiny perturbation
    initial2.set(2, 1.0);
    initial2.set(4, 1.0);

    println!("Initial separation: {:.2e}", 1e-10);

    let traj1 = rk4.solve(&lorenz, initial1, 0.0, 30.0, 3000)?;
    let traj2 = rk4.solve(&lorenz, initial2, 0.0, 30.0, 3000)?;

    // Track divergence over time
    println!("\nTrajectory divergence:");
    for i in [0, 500, 1000, 1500, 2000, 2500, 2999] {
        if let (Some((t, s1)), Some((_, s2))) = (traj1.get(i), traj2.get(i)) {
            let dx = s1.get(1) - s2.get(1);
            let dy = s1.get(2) - s2.get(2);
            let dz = s1.get(4) - s2.get(4);
            let dist = (dx*dx + dy*dy + dz*dz).sqrt();
            println!("  t={:5.1}: separation = {:12.6e}", t, dist);
        }
    }

    // =========================================================================
    // Part 3: Adaptive Solver Comparison
    // =========================================================================
    println!("\n\nPart 3: Adaptive vs Fixed-Step Solvers");
    println!("───────────────────────────────────────\n");

    // Use Dormand-Prince adaptive solver
    let dopri = DormandPrince::new(1e-10, 1e-10);  // Tight tolerances

    let mut initial3: Multivector<3, 0, 0> = Multivector::zero();
    initial3.set(1, 1.0);
    initial3.set(2, 1.0);
    initial3.set(4, 1.0);

    let adaptive_traj = dopri.solve(&lorenz, initial3.clone(), 0.0, 50.0, 10000)?;
    let fixed_traj = rk4.solve(&lorenz, initial3, 0.0, 50.0, 10000)?;

    println!("Fixed-step RK4:     {} points", fixed_traj.len());
    println!("Adaptive Dormand-Prince: {} points", adaptive_traj.len());

    // Compare final states
    if let (Some((_, af)), Some((_, ff))) = (adaptive_traj.final_state(), fixed_traj.final_state()) {
        let dx = af.get(1) - ff.get(1);
        let dy = af.get(2) - ff.get(2);
        let dz = af.get(4) - ff.get(4);
        let diff = (dx*dx + dy*dy + dz*dz).sqrt();
        println!("Final state difference: {:.6e}", diff);
    }

    // =========================================================================
    // Part 4: Lyapunov Exponents and Chaos Detection
    // =========================================================================
    println!("\n\nPart 4: Lyapunov Exponents");
    println!("──────────────────────────\n");

    let mut lyap_initial: Multivector<3, 0, 0> = Multivector::zero();
    lyap_initial.set(1, 1.0);
    lyap_initial.set(2, 1.0);
    lyap_initial.set(4, 1.0);

    let lyap_config = LyapunovConfig {
        num_steps: 10000,
        dt: 0.01,
        transient_steps: 1000,
        orthonormalization_steps: 10,
    };

    let spectrum = compute_lyapunov_spectrum(&lorenz, &lyap_initial, &lyap_config)?;

    println!("Lyapunov exponents:");
    for (i, &exp) in spectrum.exponents.iter().enumerate() {
        let sign = if exp > 0.0 { "+" } else { "" };
        println!("  λ_{} = {}{:.4}", i + 1, sign, exp);
    }

    println!("\nSum of exponents: {:.4}", spectrum.sum());
    println!("(Negative sum indicates dissipative system)");

    if spectrum.exponents[0] > 0.0 {
        println!("\n✓ Positive largest exponent confirms CHAOTIC dynamics!");
        println!("  Lyapunov time (1/λ₁): {:.2} time units", 1.0 / spectrum.exponents[0]);
    }

    // Kaplan-Yorke dimension
    let ky_dim = spectrum.kaplan_yorke_dimension();
    println!("\nKaplan-Yorke dimension: {:.3}", ky_dim);
    println!("(Fractal dimension of the strange attractor)");

    // =========================================================================
    // Part 5: Parameter Exploration
    // =========================================================================
    println!("\n\nPart 5: Parameter Exploration");
    println!("─────────────────────────────\n");

    // Explore different ρ values
    println!("Behavior at different ρ values:");
    for rho in [10.0, 15.0, 24.74, 28.0, 100.0] {
        let system = LorenzSystem::new(10.0, rho, 8.0/3.0);

        let mut ic: Multivector<3, 0, 0> = Multivector::zero();
        ic.set(1, 1.0);
        ic.set(2, 1.0);
        ic.set(4, 1.0);

        let config = LyapunovConfig {
            num_steps: 5000,
            dt: 0.01,
            transient_steps: 500,
            orthonormalization_steps: 10,
        };

        match compute_lyapunov_spectrum(&system, &ic, &config) {
            Ok(spec) => {
                let behavior = if spec.exponents[0] > 0.01 {
                    "chaotic"
                } else if spec.exponents[0] > -0.01 {
                    "periodic/quasi-periodic"
                } else {
                    "stable"
                };
                println!("  ρ = {:6.2}: λ₁ = {:+.4}  → {}", rho, spec.exponents[0], behavior);
            }
            Err(e) => println!("  ρ = {:6.2}: Error - {}", rho, e),
        }
    }

    println!("\n═══════════════════════════════════════════════════════════════");
    println!("                    DEMO COMPLETE");
    println!("═══════════════════════════════════════════════════════════════");

    Ok(())
}
