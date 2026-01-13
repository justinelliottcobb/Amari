//! # Phase Portraits Example
//!
//! Demonstrates phase space analysis including nullclines, flow fields,
//! and trajectory visualization for 2D dynamical systems.
//!
//! ## Mathematical Background
//!
//! Phase portraits provide geometric insight into system dynamics:
//! - Nullclines: Curves where dx/dt = 0 (x-nullcline) or dy/dt = 0 (y-nullcline)
//! - Fixed points occur at nullcline intersections
//! - Flow field shows direction of motion at each point
//! - Trajectories are integral curves of the flow field
//!
//! Run with: `cargo run --bin phase_portraits`

use amari_core::Multivector;
use amari_dynamics::{
    systems::{VanDerPolOscillator, DuffingOscillator, SimplePendulum},
    solver::{RungeKutta4, ODESolver},
    phase::{PhasePortrait, PortraitConfig, compute_nullclines, NullclineConfig},
    stability::{find_fixed_point, analyze_stability, FixedPointConfig},
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("═══════════════════════════════════════════════════════════════");
    println!("                   PHASE PORTRAITS DEMO");
    println!("═══════════════════════════════════════════════════════════════\n");

    let rk4 = RungeKutta4::new();

    // =========================================================================
    // Part 1: Van der Pol Oscillator
    // =========================================================================
    println!("Part 1: Van der Pol Oscillator Phase Portrait");
    println!("──────────────────────────────────────────────\n");

    let vdp = VanDerPolOscillator::new(1.0);
    println!("System: dx/dt = y");
    println!("        dy/dt = μ(1-x²)y - x,  μ = 1.0\n");

    // Compute nullclines
    println!("Nullclines:");
    println!("  x-nullcline (dx/dt = 0): y = 0");
    println!("  y-nullcline (dy/dt = 0): y = x / (μ(1-x²))\n");

    let nc_config = NullclineConfig {
        x_range: (-3.0, 3.0),
        y_range: (-3.0, 3.0),
        resolution: 100,
    };

    match compute_nullclines(&vdp, &nc_config) {
        Ok(nullclines) => {
            println!("  x-nullcline: {} points computed", nullclines.x_nullcline.len());
            println!("  y-nullcline: {} points computed", nullclines.y_nullcline.len());

            // Sample some y-nullcline points
            println!("\n  Sample y-nullcline points:");
            for (i, (x, y)) in nullclines.y_nullcline.iter().enumerate() {
                if i % 20 == 0 {
                    println!("    ({:6.3}, {:6.3})", x, y);
                }
            }
        }
        Err(e) => println!("  Nullcline computation error: {}", e),
    }

    // Compute trajectories from various initial conditions
    println!("\nTrajectories from different initial conditions:");

    let initial_conditions = vec![
        (0.1, 0.0),
        (2.5, 0.0),
        (0.0, 2.0),
        (-1.5, 1.5),
    ];

    for (x0, y0) in initial_conditions {
        let mut ic: Multivector<2, 0, 0> = Multivector::zero();
        ic.set(1, x0);
        ic.set(2, y0);

        let traj = rk4.solve(&vdp, ic, 0.0, 30.0, 3000)?;

        // Find max amplitude on limit cycle
        let mut max_x = 0.0f64;
        for i in 2000..3000 {
            if let Some((_, state)) = traj.get(i) {
                max_x = max_x.max(state.get(1).abs());
            }
        }

        println!("  ({:5.1}, {:5.1}) → limit cycle, max|x| ≈ {:.3}", x0, y0, max_x);
    }

    // =========================================================================
    // Part 2: Simple Pendulum
    // =========================================================================
    println!("\n\nPart 2: Simple Pendulum Phase Portrait");
    println!("───────────────────────────────────────\n");

    let pendulum = SimplePendulum::new(1.0, 9.81, 0.1);  // L=1m, g=9.81, damping=0.1
    println!("System: dθ/dt = ω");
    println!("        dω/dt = -(g/L)sin(θ) - γω");
    println!("        L=1.0, g=9.81, γ=0.1 (light damping)\n");

    // Fixed points
    println!("Fixed points:");
    println!("  θ = 0 (hanging down)  - stable");
    println!("  θ = π (inverted)      - unstable (saddle)");

    // Analyze stability at origin
    let mut fp_down: Multivector<2, 0, 0> = Multivector::zero();
    fp_down.set(1, 0.0);
    fp_down.set(2, 0.0);

    match analyze_stability(&pendulum, &fp_down) {
        Ok(stability) => {
            println!("\n  At θ=0: {:?}", stability.stability_type);
            print!("    Eigenvalues: ");
            for (re, im) in &stability.eigenvalues {
                if im.abs() < 1e-10 {
                    print!("{:+.3}  ", re);
                } else {
                    print!("{:+.3}±{:.3}i  ", re, im.abs());
                }
            }
            println!();
        }
        Err(e) => println!("  Error: {}", e),
    }

    // Trajectories showing different behaviors
    println!("\nTrajectory types:");

    // Small oscillation
    let mut ic_small: Multivector<2, 0, 0> = Multivector::zero();
    ic_small.set(1, 0.3);  // 0.3 rad ≈ 17°
    ic_small.set(2, 0.0);

    let traj_small = rk4.solve(&pendulum, ic_small, 0.0, 20.0, 2000)?;

    println!("  Small angle (θ₀=0.3 rad): Damped oscillation");
    if let Some((_, final_state)) = traj_small.final_state() {
        println!("    Final: θ={:.4}, ω={:.4}", final_state.get(1), final_state.get(2));
    }

    // Large oscillation (nearly rotating)
    let mut ic_large: Multivector<2, 0, 0> = Multivector::zero();
    ic_large.set(1, 2.5);  // 2.5 rad ≈ 143°
    ic_large.set(2, 0.0);

    let traj_large = rk4.solve(&pendulum, ic_large, 0.0, 20.0, 2000)?;

    println!("  Large angle (θ₀=2.5 rad): Large-amplitude damped motion");
    if let Some((_, final_state)) = traj_large.final_state() {
        println!("    Final: θ={:.4}, ω={:.4}", final_state.get(1), final_state.get(2));
    }

    // Rotation (enough energy to go over the top)
    let mut ic_rotate: Multivector<2, 0, 0> = Multivector::zero();
    ic_rotate.set(1, 0.0);
    ic_rotate.set(2, 8.0);  // High initial angular velocity

    let pendulum_undamped = SimplePendulum::new(1.0, 9.81, 0.0);  // No damping
    let traj_rotate = rk4.solve(&pendulum_undamped, ic_rotate, 0.0, 5.0, 500)?;

    // Count rotations
    let mut rotations = 0;
    let mut last_theta = 0.0;
    for i in 0..500 {
        if let Some((_, state)) = traj_rotate.get(i) {
            let theta = state.get(1);
            if last_theta > 3.0 && theta < -3.0 {
                rotations += 1;
            } else if last_theta < -3.0 && theta > 3.0 {
                rotations -= 1;
            }
            last_theta = theta;
        }
    }

    println!("  Fast rotation (ω₀=8.0): {} rotations in 5 seconds (undamped)", rotations.abs());

    // =========================================================================
    // Part 3: Duffing Double-Well
    // =========================================================================
    println!("\n\nPart 3: Duffing Double-Well Phase Portrait");
    println!("───────────────────────────────────────────\n");

    let duffing = DuffingOscillator::new(1.0, -1.0, 0.1, 0.0, 1.0);
    println!("System: dx/dt = y");
    println!("        dy/dt = x - x³ - δy,  δ=0.1 (damping)\n");

    println!("Potential: V(x) = -x²/2 + x⁴/4  (double-well)");
    println!("Fixed points: x = -1, 0, +1\n");

    // Show basins of attraction
    println!("Basin of attraction test:");
    println!("  Initial x | Final x  | Attractor");
    println!("  ──────────┼──────────┼───────────");

    for x0 in [-2.0, -1.5, -0.5, -0.1, 0.1, 0.5, 1.5, 2.0] {
        let mut ic: Multivector<2, 0, 0> = Multivector::zero();
        ic.set(1, x0);
        ic.set(2, 0.0);

        let traj = rk4.solve(&duffing, ic, 0.0, 50.0, 5000)?;

        if let Some((_, final_state)) = traj.final_state() {
            let final_x = final_state.get(1);
            let attractor = if final_x < -0.5 { "Left well" }
                           else if final_x > 0.5 { "Right well" }
                           else { "Origin" };
            println!("  {:10.1} | {:8.4} | {}", x0, final_x, attractor);
        }
    }

    // =========================================================================
    // Part 4: Flow Field Visualization Data
    // =========================================================================
    println!("\n\nPart 4: Flow Field Data (Van der Pol, μ=1)");
    println!("───────────────────────────────────────────\n");

    println!("Vector field at grid points (for external plotting):");
    println!("  x       y      | dx/dt     dy/dt   | |v|");
    println!("  ───────────────┼───────────────────┼──────");

    let vdp_vis = VanDerPolOscillator::new(1.0);

    for ix in -3..=3 {
        for iy in -3..=3 {
            let x = ix as f64;
            let y = iy as f64;

            let mut state: Multivector<2, 0, 0> = Multivector::zero();
            state.set(1, x);
            state.set(2, y);

            if let Ok(vf) = vdp_vis.vector_field(&state) {
                let dx = vf.get(1);
                let dy = vf.get(2);
                let mag = (dx*dx + dy*dy).sqrt();
                println!("  {:6.1} {:6.1} | {:9.3} {:9.3} | {:6.3}",
                         x, y, dx, dy, mag);
            }
        }
    }

    // =========================================================================
    // Part 5: Separatrix
    // =========================================================================
    println!("\n\nPart 5: Separatrices in Duffing System");
    println!("───────────────────────────────────────\n");

    println!("The saddle point at origin has stable/unstable manifolds");
    println!("that form separatrices dividing the phase space into basins.\n");

    // Find approximate separatrix by starting near the saddle
    println!("Trajectories near the separatrix:");

    for offset in [0.001, -0.001] {
        let mut ic: Multivector<2, 0, 0> = Multivector::zero();
        ic.set(1, offset);
        ic.set(2, 0.0);

        let traj = rk4.solve(&duffing, ic, 0.0, 30.0, 3000)?;

        let mut path = String::from("  ");
        path.push_str(&format!("Start: ({:+.3}, 0) → ", offset));

        for i in [100, 500, 1000, 2000] {
            if let Some((_, state)) = traj.get(i) {
                path.push_str(&format!("({:+.2},{:+.2}) → ", state.get(1), state.get(2)));
            }
        }

        if let Some((_, final_state)) = traj.final_state() {
            let attractor = if final_state.get(1) < -0.5 { "Left" }
                           else if final_state.get(1) > 0.5 { "Right" }
                           else { "Origin" };
            path.push_str(attractor);
        }

        println!("{}", path);
    }

    println!("\n═══════════════════════════════════════════════════════════════");
    println!("                    DEMO COMPLETE");
    println!("═══════════════════════════════════════════════════════════════");

    Ok(())
}
