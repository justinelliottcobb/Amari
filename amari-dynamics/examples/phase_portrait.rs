//! Phase Portrait Example
//!
//! This example demonstrates phase portrait analysis for 2D dynamical systems.
//! Phase portraits show the trajectories of a system in state space, revealing
//! the qualitative behavior: fixed points, limit cycles, and flow patterns.
//!
//! # Systems Demonstrated
//!
//! 1. **Van der Pol Oscillator** - Limit cycle attractor
//! 2. **Duffing Oscillator** - Double-well potential (bistable)
//! 3. **Simple Pendulum** - Oscillations and rotations
//!
//! # Run this example
//!
//! ```bash
//! cargo run --example phase_portrait
//! ```

use amari_core::Multivector;
use amari_dynamics::{
    DuffingOscillator, ODESolver, RungeKutta4, SimplePendulum, VanDerPolOscillator,
};

fn main() {
    println!("=== Phase Portrait Analysis ===\n");

    let solver = RungeKutta4::new();

    // ============================================================
    // 1. Van der Pol Oscillator
    // ============================================================
    println!("--- 1. Van der Pol Oscillator (μ = 1.0) ---\n");
    println!("Equations: dx/dt = y, dy/dt = μ(1-x²)y - x");
    println!("This system has a stable limit cycle.\n");

    let vdp = VanDerPolOscillator::new(1.0);

    // Test convergence to limit cycle from inside and outside
    let vdp_ics = vec![
        (0.1, 0.0, "inside"),
        (0.5, 0.0, "inside"),
        (3.0, 0.0, "outside"),
        (4.0, 0.0, "outside"),
    ];

    println!("Trajectories converging to limit cycle:");
    for (x0, y0, location) in vdp_ics {
        let mut initial: Multivector<2, 0, 0> = Multivector::zero();
        initial.set(1, x0);
        initial.set(2, y0);

        // Run for longer to reach limit cycle
        let traj = solver.solve(&vdp, initial, 0.0, 50.0, 5000).unwrap();

        // Measure amplitude at end (should be ~2 for the limit cycle)
        let mut max_x: f64 = 0.0;
        for (i, (_, state)) in traj.iter().enumerate() {
            if i > 4000 {
                max_x = max_x.max(state.get(1).abs());
            }
        }

        println!(
            "  IC: ({:.1}, {:.1}) [{:7}] → Final amplitude ≈ {:.2}",
            x0, y0, location, max_x
        );
    }

    // ============================================================
    // 2. Duffing Oscillator (Double-Well)
    // ============================================================
    println!("\n--- 2. Duffing Oscillator (Double-Well) ---\n");
    println!("Equations: dx/dt = y, dy/dt = -δy + x - x³");
    println!("Potential: V(x) = -x²/2 + x⁴/4 (two wells at x = ±1)\n");

    // Light damping
    let duffing = DuffingOscillator::new(0.05, -1.0, 1.0);

    println!("Fixed points:");
    println!("  x = -1 (stable well)");
    println!("  x =  0 (unstable saddle)");
    println!("  x = +1 (stable well)\n");

    // Show trajectories falling into different wells
    let duffing_ics = vec![
        (-1.5, 0.0, "left well"),
        (-0.3, 0.0, "left well"),
        (0.3, 0.0, "right well"),
        (1.5, 0.0, "right well"),
    ];

    println!("Trajectories in the double-well potential:");
    for (x0, y0, _expected) in duffing_ics {
        let mut initial: Multivector<2, 0, 0> = Multivector::zero();
        initial.set(1, x0);
        initial.set(2, y0);

        let traj = solver.solve(&duffing, initial, 0.0, 100.0, 10000).unwrap();
        let final_state = traj.final_state().unwrap();
        let final_x = final_state.get(1);

        let actual_well = if final_x < 0.0 {
            "left well"
        } else {
            "right well"
        };

        println!(
            "  IC: ({:+.1}, {:.1}) → x_final = {:+.3} ({})",
            x0, y0, final_x, actual_well
        );
    }

    // ============================================================
    // 3. Simple Pendulum Phase Portrait
    // ============================================================
    println!("\n--- 3. Simple Pendulum ---\n");
    println!("Equations: dθ/dt = ω, dω/dt = -(g/L)sin(θ)");
    println!("Phase portrait shows oscillations, librations, and separatrix.\n");

    let pendulum = SimplePendulum::new(1.0, 9.8, 0.0); // length, gravity, damping (undamped)

    // Different energy levels
    let pendulum_ics = vec![
        (0.5, 0.0, "small oscillation"),
        (1.5, 0.0, "medium oscillation"),
        (3.0, 0.0, "near separatrix"),
        (0.0, 3.5, "rotation (whirling)"),
    ];

    println!("Pendulum trajectories:");
    for (theta0, omega0, desc) in pendulum_ics {
        let mut initial: Multivector<2, 0, 0> = Multivector::zero();
        initial.set(1, theta0);
        initial.set(2, omega0);

        let traj = solver.solve(&pendulum, initial, 0.0, 20.0, 2000).unwrap();

        // Compute energy
        let g_over_l = 9.8 / 1.0;
        let energy = 0.5 * omega0 * omega0 - g_over_l * theta0.cos();

        // Check if it's rotating (passes through θ = π)
        let mut max_theta: f64 = 0.0;
        for (_, state) in traj.iter() {
            max_theta = max_theta.max(state.get(1).abs());
        }

        let motion_type = if max_theta > std::f64::consts::PI {
            "rotating"
        } else {
            "oscillating"
        };

        println!(
            "  IC: (θ={:.1}, ω={:.1}) → E = {:+.2}, {} ({})",
            theta0, omega0, energy, motion_type, desc
        );
    }

    // ============================================================
    // Summary
    // ============================================================
    println!("\n=== Phase Portrait Summary ===\n");
    println!("The phase portraits reveal the qualitative behavior of each system:");
    println!("  - Van der Pol: Spiral convergence to limit cycle");
    println!("  - Duffing: Basins of attraction for two stable wells");
    println!("  - Pendulum: Oscillations, rotations, and separatrix");

    println!("\n=== Example Complete ===");
}
