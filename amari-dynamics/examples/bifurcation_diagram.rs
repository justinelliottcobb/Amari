//! Bifurcation Diagram Example
//!
//! This example demonstrates bifurcation analysis in dynamical systems.
//! We show how the qualitative behavior of systems changes as parameters vary.
//!
//! # Systems Demonstrated
//!
//! 1. **Logistic Map** - Classic period-doubling route to chaos
//! 2. **Saddle-Node Bifurcation** - Fixed point creation/annihilation
//! 3. **Supercritical Pitchfork** - Symmetry breaking
//!
//! # Run this example
//!
//! ```bash
//! cargo run --example bifurcation_diagram
//! ```

use amari_core::Multivector;
use amari_dynamics::{DiscreteMap, HenonMap};

fn main() {
    println!("=== Bifurcation Analysis ===\n");

    // ============================================================
    // 1. Logistic Map: x_{n+1} = r * x_n * (1 - x_n)
    // ============================================================
    println!("--- 1. Logistic Map Bifurcation ---\n");
    println!("Equation: x_{{n+1}} = r * x_n * (1 - x_n)");
    println!("Shows period-doubling cascade to chaos.\n");

    let r_values = [2.5, 3.0, 3.449, 3.5, 3.57, 3.8, 4.0];

    println!("r       | Behavior            | Attractor values");
    println!("--------|---------------------|------------------------------------------");

    for &r in &r_values {
        let mut x = 0.5; // Initial condition

        // Transient removal (500 iterations)
        for _ in 0..500 {
            x = r * x * (1.0 - x);
        }

        // Collect attractor points (256 iterations)
        let mut attractor: Vec<f64> = Vec::new();
        for _ in 0..256 {
            x = r * x * (1.0 - x);
            // Only keep unique values (with tolerance)
            let is_new = !attractor.iter().any(|&v| (v - x).abs() < 1e-6);
            if is_new && attractor.len() < 8 {
                attractor.push(x);
            }
        }

        attractor.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let behavior = match attractor.len() {
            1 => "Fixed point (period-1)",
            2 => "Period-2 cycle     ",
            4 => "Period-4 cycle     ",
            _ if attractor.len() >= 8 => "Chaos              ",
            _ => "Higher period      ",
        };

        let values: Vec<String> = attractor
            .iter()
            .take(4)
            .map(|v| format!("{:.4}", v))
            .collect();
        let suffix = if attractor.len() > 4 { "..." } else { "" };

        println!(
            "r={:.3} | {} | {}{}",
            r,
            behavior,
            values.join(", "),
            suffix
        );
    }

    // ============================================================
    // 2. Hénon Map Bifurcation
    // ============================================================
    println!("\n--- 2. Hénon Map Bifurcation ---\n");
    println!("Equations: x_{{n+1}} = 1 - a*x_n^2 + y_n, y_{{n+1}} = b*x_n");
    println!("A 2D map with strange attractor.\n");

    let a_values = [0.5, 1.0, 1.2, 1.3, 1.4];
    let b = 0.3;

    println!("a      | b   | Lyapunov estimate | Behavior");
    println!("-------|-----|-------------------|------------------");

    for &a in &a_values {
        let henon = HenonMap::new(a, b);

        let mut state: Multivector<2, 0, 0> = Multivector::zero();
        state.set(1, 0.5); // x
        state.set(2, 0.0); // y

        // Transient removal
        for _ in 0..500 {
            if let Ok(next) = henon.iterate(&state) {
                state = next;
            }
        }

        // Estimate Lyapunov exponent via finite differences
        let mut lyap_sum = 0.0;
        let _epsilon = 1e-8;
        let n_iter = 1000;

        for _ in 0..n_iter {
            let x = state.get(1);
            // Jacobian: dx'/dx = -2ax
            let derivative = (-2.0 * a * x).abs();
            if derivative > 0.0 {
                lyap_sum += derivative.ln();
            }

            if let Ok(next) = henon.iterate(&state) {
                state = next;
            }
        }

        let lyap = lyap_sum / n_iter as f64;

        let behavior = if lyap > 0.05 {
            "Chaotic"
        } else if lyap > -0.05 {
            "Edge of chaos"
        } else if state.get(1).abs() > 1e6 {
            "Escaping"
        } else {
            "Regular"
        };

        println!(
            "a={:.1} | {:.1} | λ ≈ {:+.4}         | {}",
            a, b, lyap, behavior
        );
    }

    // ============================================================
    // 3. Saddle-Node Bifurcation Example
    // ============================================================
    println!("\n--- 3. Saddle-Node Bifurcation ---\n");
    println!("Normal form: dx/dt = r + x²");
    println!("At r < 0: two fixed points exist");
    println!("At r = 0: they collide (saddle-node)");
    println!("At r > 0: no fixed points\n");

    let r_values_sn: [f64; 6] = [-0.25, -0.1, -0.01, 0.0, 0.01, 0.1];

    println!("r       | Fixed points");
    println!("--------|----------------------------------------");

    for &r in &r_values_sn {
        if r < 0.0 {
            let sqrt_neg_r = (-r).sqrt();
            println!(
                "r={:+.2} | x* = {:.4} (stable), x* = {:+.4} (unstable)",
                r, -sqrt_neg_r, sqrt_neg_r
            );
        } else if r.abs() < 1e-10 {
            println!("r= 0.00 | x* = 0.0000 (saddle-node bifurcation)");
        } else {
            println!("r={:+.2} | No fixed points (all trajectories escape)", r);
        }
    }

    // ============================================================
    // 4. Supercritical Pitchfork Bifurcation
    // ============================================================
    println!("\n--- 4. Supercritical Pitchfork Bifurcation ---\n");
    println!("Normal form: dx/dt = rx - x³");
    println!("At r < 0: one stable fixed point at x* = 0");
    println!("At r = 0: pitchfork bifurcation");
    println!("At r > 0: x* = 0 unstable, x* = ±√r stable\n");

    let r_values_pf: [f64; 6] = [-0.5, -0.1, 0.0, 0.1, 0.5, 1.0];

    println!("r       | Fixed points");
    println!("--------|----------------------------------------");

    for &r in &r_values_pf {
        if r < 0.0 {
            println!("r={:+.1} | x* = 0 (stable)", r);
        } else if r.abs() < 1e-10 {
            println!("r= 0.0 | x* = 0 (critical point - pitchfork)",);
        } else {
            let sqrt_r = r.sqrt();
            println!(
                "r={:+.1} | x* = 0 (unstable), x* = ±{:.4} (stable)",
                r, sqrt_r
            );
        }
    }

    // ============================================================
    // Summary
    // ============================================================
    println!("\n=== Bifurcation Summary ===\n");
    println!("Key concepts demonstrated:");
    println!("  - Period-doubling: Logistic map route to chaos");
    println!("  - Saddle-node: Creation/annihilation of fixed points");
    println!("  - Pitchfork: Symmetry breaking transition");
    println!("  - Strange attractor: Hénon map chaotic dynamics");

    println!("\n=== Example Complete ===");
}
