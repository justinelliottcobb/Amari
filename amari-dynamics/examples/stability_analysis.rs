//! Stability Analysis Example
//!
//! This example demonstrates stability analysis concepts for dynamical systems,
//! including eigenvalue analysis and stability classification.
//!
//! # Concepts Demonstrated
//!
//! 1. **Eigenvalue Analysis** - Stability from eigenvalues
//! 2. **Classification** - Nodes, spirals, saddles, centers
//! 3. **System Behavior** - Connecting theory to practice
//!
//! # Run this example
//!
//! ```bash
//! cargo run --example stability_analysis
//! ```

use amari_core::Multivector;
use amari_dynamics::{
    DuffingOscillator, LorenzSystem, ODESolver, RosslerSystem, RungeKutta4, VanDerPolOscillator,
};

fn main() {
    println!("=== Stability Analysis ===\n");

    // ============================================================
    // 1. Linear System Classification
    // ============================================================
    println!("--- 1. Linear System Stability Types ---\n");
    println!("For a 2D linear system dx/dt = Ax, stability depends on eigenvalues:\n");

    let cases: [(f64, f64, f64, f64, &str); 6] = [
        (-1.0, 0.0, -2.0, 0.0, "Stable node (both λ < 0)"),
        (1.0, 0.0, 2.0, 0.0, "Unstable node (both λ > 0)"),
        (-1.0, 0.0, 1.0, 0.0, "Saddle (λ₁ < 0 < λ₂)"),
        (
            -1.0,
            2.0,
            -1.0,
            -2.0,
            "Stable spiral (Re(λ) < 0, Im(λ) ≠ 0)",
        ),
        (
            1.0,
            2.0,
            1.0,
            -2.0,
            "Unstable spiral (Re(λ) > 0, Im(λ) ≠ 0)",
        ),
        (0.0, 2.0, 0.0, -2.0, "Center (Re(λ) = 0, Im(λ) ≠ 0)"),
    ];

    for (re1, im1, re2, im2, desc) in cases {
        let trace = re1 + re2;
        let det = re1 * re2 - im1 * im2;
        let discriminant = trace * trace - 4.0 * det;

        println!("  λ₁ = {:.1}{:+.1}i, λ₂ = {:.1}{:+.1}i", re1, im1, re2, im2);
        println!(
            "    τ = {:.1}, Δ = {:.1}, τ²-4Δ = {:.1}",
            trace, det, discriminant
        );
        println!("    → {}\n", desc);
    }

    // ============================================================
    // 2. Van der Pol Oscillator: Unstable Origin, Stable Limit Cycle
    // ============================================================
    println!("--- 2. Van der Pol Oscillator ---\n");
    println!("Equations: dx/dt = y, dy/dt = μ(1-x²)y - x");
    println!("Jacobian at origin (x=0, y=0):");
    println!("  | 0   1 |");
    println!("  | -1  μ |");
    println!();

    let mu: f64 = 1.0;
    let trace = mu; // For origin
    let det: f64 = 1.0; // For origin
    let discriminant = trace * trace - 4.0 * det;

    println!("At origin with μ = {}:", mu);
    println!("  τ = {:.2}, Δ = {:.2}", trace, det);

    if discriminant < 0.0 {
        let real_part = trace / 2.0;
        let imag_part = (-discriminant).sqrt() / 2.0;
        println!("  Eigenvalues: {:.2} ± {:.2}i", real_part, imag_part);
        if real_part > 0.0 {
            println!("  Classification: UNSTABLE SPIRAL");
        }
    }
    println!("  → Trajectories spiral outward from origin to stable limit cycle\n");

    // Demonstrate by simulation
    let solver = RungeKutta4::new();
    let vdp = VanDerPolOscillator::new(mu);

    let mut initial: Multivector<2, 0, 0> = Multivector::zero();
    initial.set(1, 0.1); // Small initial condition
    initial.set(2, 0.0);

    let traj = solver.solve(&vdp, initial, 0.0, 50.0, 5000).unwrap();

    // Measure amplitude growth
    let mut amplitudes = Vec::new();
    for (i, (t, state)) in traj.iter().enumerate() {
        if i % 1000 == 0 {
            let x = state.get(1);
            let y = state.get(2);
            let amp = (x * x + y * y).sqrt();
            amplitudes.push((t, amp));
        }
    }

    println!("  Amplitude growth from small initial condition:");
    for (t, amp) in amplitudes {
        println!("    t = {:.0}s: amplitude = {:.3}", t, amp);
    }

    // ============================================================
    // 3. Duffing Oscillator: Bistability
    // ============================================================
    println!("\n--- 3. Duffing Oscillator (Double-Well) ---\n");
    println!("Equations: dx/dt = y, dy/dt = -δy + x - x³");
    println!("Fixed points and their Jacobians:\n");

    let delta = 0.05;

    // Origin: x = 0
    println!("Origin (x*, y*) = (0, 0):");
    println!("  Jacobian: | 0   1  |");
    println!("            | 1  -δ  |");
    let det_origin = 0.0 * (-delta) - 1.0 * 1.0; // -1
    println!("  Det = {:.2} < 0 → SADDLE (unstable)\n", det_origin);

    // x = +1: Jacobian has df₂/dx = 1 - 3x² = 1 - 3 = -2
    println!("Right well (x*, y*) = (1, 0):");
    println!("  Jacobian: | 0   1  |");
    println!("            | -2  -δ |");
    let trace_1 = -delta;
    let det_1 = 0.0 * (-delta) - 1.0 * (-2.0); // 2
    let disc_1 = trace_1 * trace_1 - 4.0 * det_1;
    println!(
        "  τ = {:.2}, Δ = {:.2}, τ²-4Δ = {:.2}",
        trace_1, det_1, disc_1
    );
    if disc_1 < 0.0 {
        println!(
            "  Complex eigenvalues with Re(λ) = {:.3} < 0",
            trace_1 / 2.0
        );
        println!("  → STABLE SPIRAL\n");
    }

    // Demonstrate basins of attraction
    let duffing = DuffingOscillator::new(delta, -1.0, 1.0);

    println!("Basins of attraction:");
    let test_points: [(f64, &str); 4] = [
        (-2.0, "far left"),
        (-0.1, "near origin left"),
        (0.1, "near origin right"),
        (2.0, "far right"),
    ];

    for (x0, desc) in test_points {
        let mut initial: Multivector<2, 0, 0> = Multivector::zero();
        initial.set(1, x0);
        initial.set(2, 0.0);

        let traj = solver.solve(&duffing, initial, 0.0, 100.0, 10000).unwrap();
        let final_state = traj.final_state().unwrap();
        let x_final = final_state.get(1);

        let basin = if x_final < 0.0 { "LEFT" } else { "RIGHT" };
        println!(
            "  x₀ = {:+.1} ({}) → x_∞ ≈ {:+.2} ({} well)",
            x0, desc, x_final, basin
        );
    }

    // ============================================================
    // 4. Lorenz System: Chaotic Attractor
    // ============================================================
    println!("\n--- 4. Lorenz System Fixed Points ---\n");

    let lorenz = LorenzSystem::classic();
    println!(
        "Parameters: σ = {:.2}, ρ = {:.2}, β = {:.4}\n",
        lorenz.sigma, lorenz.rho, lorenz.beta
    );

    // Fixed points
    let c = (lorenz.beta * (lorenz.rho - 1.0)).sqrt();
    let z_c = lorenz.rho - 1.0;

    println!("Fixed points:");
    println!("  Origin: (0, 0, 0)");
    println!("    Jacobian eigenvalues (analytical): λ₁ ≈ -22.8, λ₂ ≈ 11.8, λ₃ = -2.67");
    println!("    → SADDLE (1 stable, 2 unstable directions)");
    println!();
    println!("  C+: ({:.2}, {:.2}, {:.2})", c, c, z_c);
    println!("  C-: ({:.2}, {:.2}, {:.2})", -c, -c, z_c);
    println!("    For ρ > 24.74, eigenvalues: one real < 0, two complex with Re > 0");
    println!("    → UNSTABLE SPIRAL (trajectories repelled to strange attractor)");

    // Demonstrate the strange attractor
    println!("\n  Demonstrating strange attractor behavior:");

    let mut initial: Multivector<3, 0, 0> = Multivector::zero();
    initial.set(1, c + 0.1); // Near C+
    initial.set(2, c);
    initial.set(4, z_c);

    let traj = solver.solve(&lorenz, initial, 0.0, 20.0, 2000).unwrap();

    // Count switches between wings
    let mut switches = 0;
    let mut last_wing: i8 = 0;
    for (_, state) in traj.iter() {
        let x = state.get(1);
        let current_wing: i8 = if x > 0.0 { 1 } else { -1 };
        if last_wing != 0 && current_wing != last_wing {
            switches += 1;
        }
        last_wing = current_wing;
    }
    println!(
        "    Starting near C+, trajectory switches between wings {} times in 20s",
        switches
    );
    println!("    → Characteristic chaotic switching behavior");

    // ============================================================
    // 5. Rössler System: Period-Doubling Route
    // ============================================================
    println!("\n--- 5. Rössler System ---\n");

    let rossler = RosslerSystem::new(0.2, 0.2, 5.7);
    println!(
        "Parameters: a = {:.2}, b = {:.2}, c = {:.2}",
        rossler.a, rossler.b, rossler.c
    );
    println!("For these parameters: system exhibits chaotic behavior");
    println!("  - One unstable fixed point with complex eigenvalues");
    println!("  - Period-doubling route to chaos as c increases");

    let mut initial: Multivector<3, 0, 0> = Multivector::zero();
    initial.set(1, 1.0);
    initial.set(2, 1.0);
    initial.set(4, 1.0);

    let traj = solver.solve(&rossler, initial, 0.0, 100.0, 10000).unwrap();

    // Measure x extrema
    let mut x_max: f64 = f64::NEG_INFINITY;
    let mut x_min: f64 = f64::INFINITY;
    for (i, (_, state)) in traj.iter().enumerate() {
        if i > 5000 {
            // After transient
            let x = state.get(1);
            x_max = x_max.max(x);
            x_min = x_min.min(x);
        }
    }
    println!(
        "\n  Attractor bounds (after transient): x ∈ [{:.2}, {:.2}]",
        x_min, x_max
    );

    // ============================================================
    // 6. Phase Diagram Summary
    // ============================================================
    println!("\n--- 6. Trace-Determinant Phase Diagram ---\n");
    println!("The (τ, Δ) plane for 2D systems:\n");
    println!("           Δ");
    println!("           ↑");
    println!("           |     Stable      τ²=4Δ     Unstable");
    println!("           |      Spiral   /  |  \\      Spiral");
    println!("           |             /    |    \\");
    println!("           |    Stable /      |      \\  Unstable");
    println!("           |     Node  -------|-------  Node");
    println!("    -------+--------------------------------→ τ");
    println!("           |          Saddle");
    println!("           |");
    println!("\nStability boundaries:");
    println!("  - Δ = 0: Transition to saddle (one zero eigenvalue)");
    println!("  - τ = 0: Hopf bifurcation (stability change)");
    println!("  - τ² = 4Δ: Node-focus transition (complex eigenvalues)");

    // ============================================================
    // Summary
    // ============================================================
    println!("\n=== Stability Analysis Summary ===\n");
    println!("Key concepts:");
    println!("  1. Eigenvalues determine local behavior near fixed points");
    println!("  2. Stable: all Re(λ) < 0, Unstable: any Re(λ) > 0");
    println!("  3. Spiral: complex λ, Node: real λ, Saddle: mixed signs");
    println!("  4. Chaotic systems: unstable fixed points + bounded attractor");

    println!("\n=== Example Complete ===");
}
