//! # Bifurcation Analysis Example
//!
//! Demonstrates bifurcation detection and diagram generation for dynamical systems.
//! Shows the period-doubling route to chaos in the logistic map.
//!
//! ## Mathematical Background
//!
//! A bifurcation occurs when a small change in parameter causes a qualitative
//! change in system behavior. Common types include:
//! - Saddle-node: Creation/destruction of fixed points
//! - Pitchfork: Symmetry breaking
//! - Hopf: Birth of limit cycles
//! - Period-doubling: Route to chaos
//!
//! Run with: `cargo run --bin bifurcation_analysis`

use amari_core::Multivector;
use amari_dynamics::{
    systems::{HenonMap, DuffingOscillator},
    solver::{RungeKutta4, ODESolver},
    bifurcation::{
        BifurcationDiagram, ContinuationConfig, BifurcationType,
        detect_bifurcation, BifurcationConfig,
    },
};

/// Simple logistic map for bifurcation demonstration
struct LogisticMap {
    r: f64,
}

impl LogisticMap {
    fn new(r: f64) -> Self {
        Self { r }
    }

    fn iterate(&self, x: f64) -> f64 {
        self.r * x * (1.0 - x)
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("═══════════════════════════════════════════════════════════════");
    println!("                 BIFURCATION ANALYSIS DEMO");
    println!("═══════════════════════════════════════════════════════════════\n");

    // =========================================================================
    // Part 1: Logistic Map Period-Doubling
    // =========================================================================
    println!("Part 1: Logistic Map - Period-Doubling Route to Chaos");
    println!("──────────────────────────────────────────────────────\n");

    println!("The logistic map: x_{n+1} = r * x_n * (1 - x_n)\n");

    // Generate bifurcation diagram data
    let r_min = 2.5;
    let r_max = 4.0;
    let num_r = 150;
    let transient = 500;
    let samples = 100;

    println!("Generating bifurcation diagram...");
    println!("  r ∈ [{}, {}]", r_min, r_max);
    println!("  {} parameter values", num_r);
    println!("  {} transient iterations discarded", transient);
    println!("  {} attractor samples per parameter\n", samples);

    // Compute bifurcation diagram manually for logistic map
    let dr = (r_max - r_min) / (num_r as f64);

    println!("Bifurcation structure:");
    println!("  r       | Attractor Type    | Representative Values");
    println!("  ────────┼───────────────────┼─────────────────────────────");

    let key_r_values = [2.8, 3.2, 3.5, 3.56, 3.57, 3.83, 4.0];

    for &r in &key_r_values {
        let map = LogisticMap::new(r);

        // Run transient
        let mut x = 0.5;
        for _ in 0..transient {
            x = map.iterate(x);
        }

        // Collect attractor samples
        let mut attractor: Vec<f64> = Vec::new();
        for _ in 0..samples {
            x = map.iterate(x);
            attractor.push(x);
        }

        // Remove duplicates (with tolerance)
        attractor.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mut unique: Vec<f64> = Vec::new();
        for val in &attractor {
            if unique.is_empty() || (val - unique.last().unwrap()).abs() > 0.001 {
                unique.push(*val);
            }
        }

        let attractor_type = match unique.len() {
            1 => "Fixed point",
            2 => "Period-2 cycle",
            4 => "Period-4 cycle",
            8 => "Period-8 cycle",
            _ if unique.len() > 20 => "Chaos",
            n => &format!("Period-{}", n),
        };

        let repr: String = unique.iter()
            .take(4)
            .map(|v| format!("{:.3}", v))
            .collect::<Vec<_>>()
            .join(", ");
        let suffix = if unique.len() > 4 { "..." } else { "" };

        println!("  {:7.3} | {:17} | {}{}", r, attractor_type, repr, suffix);
    }

    // =========================================================================
    // Part 2: Period-Doubling Cascade
    // =========================================================================
    println!("\n\nPart 2: Period-Doubling Cascade and Feigenbaum Constant");
    println!("────────────────────────────────────────────────────────\n");

    // Find bifurcation points
    let bifurcation_points = [
        (1, 3.0),
        (2, 3.44949),
        (4, 3.54409),
        (8, 3.5644),
        (16, 3.5688),
    ];

    println!("Period-doubling bifurcation points:");
    println!("  Period | r_n       | δ_n (ratio)");
    println!("  ───────┼───────────┼────────────");

    for i in 0..bifurcation_points.len() {
        let (period, r) = bifurcation_points[i];
        if i >= 2 {
            let (_, r_prev) = bifurcation_points[i - 1];
            let (_, r_prev2) = bifurcation_points[i - 2];
            let delta = (r_prev - r_prev2) / (r - r_prev);
            println!("  {:6} | {:9.5} | {:.4}", period, r, delta);
        } else {
            println!("  {:6} | {:9.5} | -", period, r);
        }
    }

    println!("\nFeigenbaum constant δ ≈ 4.6692...");
    println!("(Universal constant for period-doubling cascades)");

    // =========================================================================
    // Part 3: Henon Map Bifurcations
    // =========================================================================
    println!("\n\nPart 3: Henon Map");
    println!("─────────────────\n");

    println!("The Henon map: x_{n+1} = 1 - a*x_n² + y_n");
    println!("               y_{n+1} = b*x_n\n");

    // Classic parameters
    let henon = HenonMap::classic();  // a=1.4, b=0.3
    println!("Classic parameters: a=1.4, b=0.3");

    // Iterate to find attractor
    let mut x = 0.1;
    let mut y = 0.1;

    // Discard transient
    for _ in 0..1000 {
        let (nx, ny) = henon.iterate(x, y);
        x = nx;
        y = ny;
    }

    // Collect attractor points
    println!("\nStrange attractor sample points:");
    for i in 0..10 {
        let (nx, ny) = henon.iterate(x, y);
        x = nx;
        y = ny;
        println!("  ({:8.5}, {:8.5})", x, y);
    }

    // Scan a parameter
    println!("\nHenon map behavior vs parameter a (b=0.3 fixed):");
    println!("  a     | Behavior");
    println!("  ──────┼─────────────────────");

    for a in [0.5, 0.9, 1.0, 1.2, 1.4, 1.5] {
        let map = HenonMap::new(a, 0.3);
        let mut x = 0.1;
        let mut y = 0.1;

        // Check for divergence or bounded behavior
        let mut diverged = false;
        for _ in 0..1000 {
            let (nx, ny) = map.iterate(x, y);
            x = nx;
            y = ny;
            if x.abs() > 100.0 || y.abs() > 100.0 {
                diverged = true;
                break;
            }
        }

        if diverged {
            println!("  {:5.2} | Divergent (escapes to infinity)", a);
        } else {
            // Check attractor dimension
            let mut points: Vec<(f64, f64)> = Vec::new();
            for _ in 0..100 {
                let (nx, ny) = map.iterate(x, y);
                x = nx;
                y = ny;
                points.push((x, y));
            }

            // Simple period detection
            let first = points[0];
            let mut period = 0;
            for (i, p) in points.iter().enumerate().skip(1) {
                if (p.0 - first.0).abs() < 0.001 && (p.1 - first.1).abs() < 0.001 {
                    period = i;
                    break;
                }
            }

            if period > 0 && period < 50 {
                println!("  {:5.2} | Period-{} orbit", a, period);
            } else {
                println!("  {:5.2} | Strange attractor (chaotic)", a);
            }
        }
    }

    // =========================================================================
    // Part 4: Hopf Bifurcation in Duffing Oscillator
    // =========================================================================
    println!("\n\nPart 4: Duffing Oscillator");
    println!("──────────────────────────\n");

    println!("The Duffing equation: ẍ + δẋ + αx + βx³ = γ cos(ωt)");
    println!("Double-well potential with periodic forcing\n");

    let rk4 = RungeKutta4::new();

    // Unforced double-well
    let duffing_unforced = DuffingOscillator::new(1.0, -1.0, 0.1, 0.0, 1.0);

    // Find the two stable equilibria
    println!("Unforced system (γ=0): Two stable equilibria at x = ±1");

    let mut ic_left: Multivector<2, 0, 0> = Multivector::zero();
    ic_left.set(1, -0.5);
    ic_left.set(2, 0.0);

    let mut ic_right: Multivector<2, 0, 0> = Multivector::zero();
    ic_right.set(1, 0.5);
    ic_right.set(2, 0.0);

    let traj_left = rk4.solve(&duffing_unforced, ic_left, 0.0, 50.0, 5000)?;
    let traj_right = rk4.solve(&duffing_unforced, ic_right, 0.0, 50.0, 5000)?;

    if let (Some((_, fl)), Some((_, fr))) = (traj_left.final_state(), traj_right.final_state()) {
        println!("  From x₀=-0.5: converges to x ≈ {:.4}", fl.get(1));
        println!("  From x₀=+0.5: converges to x ≈ {:.4}", fr.get(1));
    }

    // Forced system - chaos
    println!("\nForced system: Behavior vs forcing amplitude γ");
    println!("  γ     | Behavior");
    println!("  ──────┼────────────────────────");

    for gamma in [0.1, 0.2, 0.3, 0.35, 0.4] {
        let duffing = DuffingOscillator::new(1.0, -1.0, 0.15, gamma, 1.0);

        let mut ic: Multivector<2, 0, 0> = Multivector::zero();
        ic.set(1, 0.1);
        ic.set(2, 0.0);

        let traj = rk4.solve(&duffing, ic, 0.0, 200.0, 20000)?;

        // Check if trajectory visits both wells
        let mut visited_left = false;
        let mut visited_right = false;
        let mut crossings = 0;

        let mut last_x = 0.0;
        for i in 10000..20000 {  // After transient
            if let Some((_, state)) = traj.get(i) {
                let x = state.get(1);
                if x < -0.5 { visited_left = true; }
                if x > 0.5 { visited_right = true; }
                if (last_x < 0.0 && x >= 0.0) || (last_x > 0.0 && x <= 0.0) {
                    crossings += 1;
                }
                last_x = x;
            }
        }

        let behavior = if visited_left && visited_right {
            if crossings > 100 {
                "Chaotic (inter-well hopping)"
            } else {
                "Periodic (double-well)"
            }
        } else {
            "Periodic (single well)"
        };

        println!("  {:5.2} | {}", gamma, behavior);
    }

    println!("\n═══════════════════════════════════════════════════════════════");
    println!("                    DEMO COMPLETE");
    println!("═══════════════════════════════════════════════════════════════");

    Ok(())
}
