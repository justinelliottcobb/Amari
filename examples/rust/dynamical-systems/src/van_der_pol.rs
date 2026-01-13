//! # Van der Pol Oscillator Example
//!
//! Demonstrates the Van der Pol oscillator - a classic example of a self-sustained
//! relaxation oscillation with a stable limit cycle.
//!
//! ## Mathematical Background
//!
//! The Van der Pol equations are:
//! ```text
//! dx/dt = y
//! dy/dt = μ(1 - x²)y - x
//! ```
//!
//! The parameter μ controls the nonlinearity:
//! - μ = 0: Simple harmonic oscillator
//! - μ small: Nearly sinusoidal oscillations
//! - μ large: Relaxation oscillations with sharp transitions
//!
//! Run with: `cargo run --bin van_der_pol`

use amari_core::Multivector;
use amari_dynamics::{
    systems::VanDerPolOscillator,
    solver::{RungeKutta4, ODESolver},
    stability::{find_fixed_point, analyze_stability, FixedPointConfig},
    attractor::{detect_limit_cycle, LimitCycleConfig},
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("═══════════════════════════════════════════════════════════════");
    println!("                 VAN DER POL OSCILLATOR DEMO");
    println!("═══════════════════════════════════════════════════════════════\n");

    // =========================================================================
    // Part 1: Basic Limit Cycle Behavior
    // =========================================================================
    println!("Part 1: Limit Cycle Behavior");
    println!("────────────────────────────\n");

    let mu = 1.0;
    let vdp = VanDerPolOscillator::new(mu);
    println!("Van der Pol oscillator with μ = {}", mu);

    let rk4 = RungeKutta4::new();

    // Multiple initial conditions all converge to the same limit cycle
    println!("\nConvergence from different initial conditions:");

    let initial_conditions = vec![
        (0.1, 0.0, "small displacement"),
        (3.0, 0.0, "large displacement"),
        (0.0, 2.0, "velocity kick"),
        (-2.0, -1.0, "negative quadrant"),
    ];

    for (x0, y0, desc) in initial_conditions {
        let mut initial: Multivector<2, 0, 0> = Multivector::zero();
        initial.set(1, x0);
        initial.set(2, y0);

        let trajectory = rk4.solve(&vdp, initial, 0.0, 50.0, 5000)?;

        if let Some((_, final_state)) = trajectory.final_state() {
            let x = final_state.get(1);
            let y = final_state.get(2);
            let amplitude = (x*x + y*y).sqrt();
            println!("  ({:5.1}, {:5.1}) {}: final amplitude ≈ {:.3}",
                     x0, y0, desc, amplitude);
        }
    }

    println!("\nAll trajectories converge to limit cycle with amplitude ≈ 2");

    // =========================================================================
    // Part 2: Fixed Point Stability Analysis
    // =========================================================================
    println!("\n\nPart 2: Fixed Point Analysis");
    println!("────────────────────────────\n");

    // Find the fixed point at origin
    let mut guess: Multivector<2, 0, 0> = Multivector::zero();
    guess.set(1, 0.1);
    guess.set(2, 0.1);

    let fp_config = FixedPointConfig::default();
    let fp_result = find_fixed_point(&vdp, &guess, &fp_config)?;

    if fp_result.converged {
        let fp = &fp_result.point;
        println!("Fixed point found: ({:.6}, {:.6})",
                 fp.get(1), fp.get(2));
        println!("Convergence in {} iterations", fp_result.iterations);

        // Analyze stability
        let stability = analyze_stability(&vdp, fp)?;
        println!("\nStability analysis:");
        println!("  Type: {:?}", stability.stability_type);
        println!("  Eigenvalues:");
        for (i, (re, im)) in stability.eigenvalues.iter().enumerate() {
            if im.abs() < 1e-10 {
                println!("    λ_{} = {:.4}", i + 1, re);
            } else {
                println!("    λ_{} = {:.4} ± {:.4}i", i + 1, re, im.abs());
            }
        }

        let trace = stability.trace;
        let det = stability.determinant;
        println!("  Trace = {:.4}, Det = {:.4}", trace, det);

        if trace > 0.0 {
            println!("\n✓ Positive trace confirms UNSTABLE fixed point");
            println!("  (Trajectories spiral away from origin)");
        }
    }

    // =========================================================================
    // Part 3: Effect of μ Parameter
    // =========================================================================
    println!("\n\nPart 3: Effect of Damping Parameter μ");
    println!("──────────────────────────────────────\n");

    println!("Period and amplitude vs μ:");
    println!("  μ      | Period  | Amplitude | Character");
    println!("  ───────┼─────────┼───────────┼──────────────────");

    for mu_val in [0.1, 0.5, 1.0, 2.0, 4.0, 8.0] {
        let system = VanDerPolOscillator::new(mu_val);

        let mut ic: Multivector<2, 0, 0> = Multivector::zero();
        ic.set(1, 0.1);
        ic.set(2, 0.0);

        let traj = rk4.solve(&system, ic, 0.0, 100.0, 10000)?;

        // Find period by detecting zero crossings after transient
        let mut last_x = 0.0;
        let mut crossings = Vec::new();
        let mut last_t = 0.0;

        for i in 5000..10000 {  // Skip transient
            if let Some((t, state)) = traj.get(i) {
                let x = state.get(1);
                if last_x < 0.0 && x >= 0.0 {
                    crossings.push(t);
                }
                last_x = x;
                last_t = t;
            }
        }

        let period = if crossings.len() >= 2 {
            crossings[crossings.len() - 1] - crossings[crossings.len() - 2]
        } else {
            0.0
        };

        // Find max amplitude in last portion
        let mut max_amp = 0.0f64;
        for i in 8000..10000 {
            if let Some((_, state)) = traj.get(i) {
                max_amp = max_amp.max(state.get(1).abs());
            }
        }

        let character = if mu_val < 0.5 {
            "nearly sinusoidal"
        } else if mu_val < 2.0 {
            "moderate nonlinearity"
        } else {
            "relaxation oscillation"
        };

        println!("  {:5.1}  | {:7.3} | {:9.3} | {}", mu_val, period, max_amp, character);
    }

    // =========================================================================
    // Part 4: Limit Cycle Detection
    // =========================================================================
    println!("\n\nPart 4: Limit Cycle Detection");
    println!("─────────────────────────────\n");

    let vdp_mu2 = VanDerPolOscillator::new(2.0);

    let mut ic: Multivector<2, 0, 0> = Multivector::zero();
    ic.set(1, 0.1);
    ic.set(2, 0.0);

    let lc_config = LimitCycleConfig {
        transient_time: 50.0,
        detection_time: 20.0,
        dt: 0.01,
        tolerance: 1e-4,
    };

    match detect_limit_cycle(&vdp_mu2, &ic, &lc_config) {
        Ok(limit_cycle) => {
            println!("Limit cycle detected:");
            println!("  Period: {:.4}", limit_cycle.period);
            println!("  Points on cycle: {}", limit_cycle.points.len());

            // Compute approximate area enclosed
            let mut area = 0.0;
            let n = limit_cycle.points.len();
            for i in 0..n {
                let p1 = &limit_cycle.points[i];
                let p2 = &limit_cycle.points[(i + 1) % n];
                area += p1.get(1) * p2.get(2) - p2.get(1) * p1.get(2);
            }
            area = area.abs() / 2.0;
            println!("  Enclosed area: {:.4}", area);
        }
        Err(e) => println!("Limit cycle detection failed: {}", e),
    }

    // =========================================================================
    // Part 5: Phase Space Visualization Data
    // =========================================================================
    println!("\n\nPart 5: Phase Space Data");
    println!("────────────────────────\n");

    let vdp_vis = VanDerPolOscillator::new(1.0);

    // Generate trajectory for visualization
    let mut vis_ic: Multivector<2, 0, 0> = Multivector::zero();
    vis_ic.set(1, 0.1);
    vis_ic.set(2, 0.0);

    let vis_traj = rk4.solve(&vdp_vis, vis_ic, 0.0, 30.0, 3000)?;

    // Output first and last 5 points for external plotting
    println!("Trajectory data (x, y) - first 10 points:");
    for i in 0..10 {
        if let Some((t, state)) = vis_traj.get(i) {
            println!("  t={:5.2}: ({:8.4}, {:8.4})", t, state.get(1), state.get(2));
        }
    }

    println!("\n... {} more points ...", vis_traj.len() - 20);

    println!("\nLast 10 points (on limit cycle):");
    let n = vis_traj.len();
    for i in (n-10)..n {
        if let Some((t, state)) = vis_traj.get(i) {
            println!("  t={:5.2}: ({:8.4}, {:8.4})", t, state.get(1), state.get(2));
        }
    }

    println!("\n═══════════════════════════════════════════════════════════════");
    println!("                    DEMO COMPLETE");
    println!("═══════════════════════════════════════════════════════════════");

    Ok(())
}
