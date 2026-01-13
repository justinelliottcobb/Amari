//! # Stability Analysis Example
//!
//! Demonstrates linearization and stability classification of fixed points
//! using eigenvalue analysis of the Jacobian matrix.
//!
//! ## Mathematical Background
//!
//! For a system dx/dt = f(x) near fixed point x*, stability is determined by
//! the Jacobian eigenvalues:
//!
//! | Type            | Eigenvalue Condition          | Behavior |
//! |-----------------|-------------------------------|----------|
//! | Stable Node     | λ₁, λ₂ < 0, real             | Decay    |
//! | Stable Spiral   | Re(λ) < 0, complex           | Damped   |
//! | Unstable Node   | λ₁, λ₂ > 0, real             | Growth   |
//! | Unstable Spiral | Re(λ) > 0, complex           | Amplified|
//! | Saddle          | λ₁ < 0 < λ₂                  | Mixed    |
//! | Center          | Re(λ) = 0, complex           | Neutral  |
//!
//! Run with: `cargo run --bin stability_analysis`

use amari_core::Multivector;
use amari_dynamics::{
    systems::{LorenzSystem, VanDerPolOscillator, DuffingOscillator},
    solver::{RungeKutta4, ODESolver},
    stability::{
        find_fixed_point, find_all_fixed_points, analyze_stability,
        FixedPointConfig, StabilityType, StabilityResult,
        compute_jacobian, DifferentiationConfig,
    },
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("═══════════════════════════════════════════════════════════════");
    println!("                  STABILITY ANALYSIS DEMO");
    println!("═══════════════════════════════════════════════════════════════\n");

    // =========================================================================
    // Part 1: Simple Linear System
    // =========================================================================
    println!("Part 1: Linear System Classification");
    println!("─────────────────────────────────────\n");

    println!("For 2D linear systems dx/dt = Ax, stability depends on:");
    println!("  Trace τ = a₁₁ + a₂₂ (sum of eigenvalues)");
    println!("  Determinant Δ = det(A) (product of eigenvalues)\n");

    println!("Classification regions:");
    println!("  τ < 0, Δ > 0, τ² > 4Δ : Stable node");
    println!("  τ < 0, Δ > 0, τ² < 4Δ : Stable spiral");
    println!("  τ > 0, Δ > 0, τ² > 4Δ : Unstable node");
    println!("  τ > 0, Δ > 0, τ² < 4Δ : Unstable spiral");
    println!("  Δ < 0                 : Saddle");
    println!("  τ = 0, Δ > 0          : Center");

    // =========================================================================
    // Part 2: Van der Pol Fixed Point
    // =========================================================================
    println!("\n\nPart 2: Van der Pol Oscillator Fixed Point");
    println!("───────────────────────────────────────────\n");

    for mu in [0.0, 0.5, 1.0, 2.0] {
        let vdp = VanDerPolOscillator::new(mu);
        println!("μ = {}:", mu);

        // Initial guess near origin
        let mut guess: Multivector<2, 0, 0> = Multivector::zero();
        guess.set(1, 0.1);
        guess.set(2, 0.1);

        let fp_config = FixedPointConfig {
            tolerance: 1e-10,
            max_iterations: 100,
            ..Default::default()
        };

        match find_fixed_point(&vdp, &guess, &fp_config) {
            Ok(result) if result.converged => {
                let fp = &result.point;
                println!("  Fixed point: ({:.6}, {:.6})", fp.get(1), fp.get(2));

                // Analyze stability
                match analyze_stability(&vdp, fp) {
                    Ok(stability) => {
                        println!("  Stability: {:?}", stability.stability_type);
                        println!("  Trace: {:.4}, Det: {:.4}",
                                 stability.trace, stability.determinant);

                        print!("  Eigenvalues: ");
                        for (i, (re, im)) in stability.eigenvalues.iter().enumerate() {
                            if i > 0 { print!(", "); }
                            if im.abs() < 1e-10 {
                                print!("{:.4}", re);
                            } else if *im > 0.0 {
                                print!("{:.4}+{:.4}i", re, im);
                            } else {
                                print!("{:.4}{:.4}i", re, im);
                            }
                        }
                        println!();

                        // Physical interpretation
                        match stability.stability_type {
                            StabilityType::UnstableSpiral => {
                                println!("  → Trajectories spiral away from origin");
                                println!("    (explains why system has stable limit cycle)");
                            }
                            StabilityType::Center => {
                                println!("  → Pure oscillation (μ=0 is linear oscillator)");
                            }
                            _ => {}
                        }
                    }
                    Err(e) => println!("  Stability analysis error: {}", e),
                }
            }
            Ok(_) => println!("  Fixed point search did not converge"),
            Err(e) => println!("  Error: {}", e),
        }
        println!();
    }

    // =========================================================================
    // Part 3: Lorenz System Fixed Points
    // =========================================================================
    println!("\nPart 3: Lorenz System Fixed Points");
    println!("───────────────────────────────────\n");

    let lorenz = LorenzSystem::classic();
    println!("Lorenz system: σ={}, ρ={}, β={:.4}\n", lorenz.sigma(), lorenz.rho(), lorenz.beta());

    // The Lorenz system has three fixed points:
    // 1. Origin (0, 0, 0)
    // 2. C+ = (√(β(ρ-1)), √(β(ρ-1)), ρ-1)
    // 3. C- = (-√(β(ρ-1)), -√(β(ρ-1)), ρ-1)

    let beta = lorenz.beta();
    let rho = lorenz.rho();
    let sigma = lorenz.sigma();

    let sqrt_val = (beta * (rho - 1.0)).sqrt();

    let fp_configs = vec![
        ("Origin", 0.0, 0.0, 0.0),
        ("C+", sqrt_val, sqrt_val, rho - 1.0),
        ("C-", -sqrt_val, -sqrt_val, rho - 1.0),
    ];

    for (name, x0, y0, z0) in fp_configs {
        println!("Fixed point {}: ({:.4}, {:.4}, {:.4})", name, x0, y0, z0);

        let mut fp: Multivector<3, 0, 0> = Multivector::zero();
        fp.set(1, x0);
        fp.set(2, y0);
        fp.set(4, z0);

        // Verify it's a fixed point
        match lorenz.vector_field(&fp) {
            Ok(vf) => {
                let residual = (vf.get(1).powi(2) + vf.get(2).powi(2) + vf.get(4).powi(2)).sqrt();
                println!("  Residual |f(x*)| = {:.2e}", residual);
            }
            Err(_) => {}
        }

        // Compute and analyze Jacobian
        let diff_config = DifferentiationConfig::default();
        match compute_jacobian(&lorenz, &fp, &diff_config) {
            Ok(jac) => {
                println!("  Jacobian matrix:");
                for i in 0..3 {
                    print!("    [");
                    for j in 0..3 {
                        print!("{:8.3}", jac[(i, j)]);
                    }
                    println!(" ]");
                }

                match analyze_stability(&lorenz, &fp) {
                    Ok(stability) => {
                        println!("  Stability: {:?}", stability.stability_type);
                        println!("  Eigenvalues:");
                        for (i, (re, im)) in stability.eigenvalues.iter().enumerate() {
                            if im.abs() < 1e-6 {
                                println!("    λ_{} = {:+.4}", i + 1, re);
                            } else {
                                println!("    λ_{} = {:+.4} ± {:.4}i", i + 1, re, im.abs());
                            }
                        }
                    }
                    Err(e) => println!("  Stability error: {}", e),
                }
            }
            Err(e) => println!("  Jacobian error: {}", e),
        }
        println!();
    }

    println!("Note: For ρ=28 (classic Lorenz), C+ and C- are unstable spirals.");
    println!("      This is why trajectories don't settle at fixed points but");
    println!("      instead form the strange attractor.");

    // =========================================================================
    // Part 4: Duffing Oscillator - Bistability
    // =========================================================================
    println!("\n\nPart 4: Duffing Oscillator - Bistability");
    println!("─────────────────────────────────────────\n");

    // Unforced Duffing: ẍ + δẋ - x + x³ = 0
    // Written as system: dx/dt = y, dy/dt = x - x³ - δy
    // Fixed points: (0,0), (1,0), (-1,0)

    let duffing = DuffingOscillator::new(1.0, -1.0, 0.1, 0.0, 1.0);
    println!("Unforced Duffing: α=1, β=-1, δ=0.1, γ=0 (no forcing)\n");

    let fixed_points_duffing = vec![
        ("Origin (unstable saddle)", 0.0, 0.0),
        ("Left well (stable)", -1.0, 0.0),
        ("Right well (stable)", 1.0, 0.0),
    ];

    for (name, x0, y0) in fixed_points_duffing {
        println!("{}:", name);

        let mut fp: Multivector<2, 0, 0> = Multivector::zero();
        fp.set(1, x0);
        fp.set(2, y0);

        match analyze_stability(&duffing, &fp) {
            Ok(stability) => {
                println!("  Position: ({:.1}, {:.1})", x0, y0);
                println!("  Type: {:?}", stability.stability_type);

                print!("  Eigenvalues: ");
                for (i, (re, im)) in stability.eigenvalues.iter().enumerate() {
                    if i > 0 { print!(", "); }
                    if im.abs() < 1e-10 {
                        print!("{:+.4}", re);
                    } else {
                        print!("{:+.4}±{:.4}i", re, im.abs());
                    }
                }
                println!();
            }
            Err(e) => println!("  Error: {}", e),
        }
        println!();
    }

    println!("Bistability: System has two stable equilibria (wells) separated");
    println!("by an unstable saddle point at the origin.");

    // =========================================================================
    // Part 5: Stability Type Summary
    // =========================================================================
    println!("\n\nPart 5: Stability Classification Reference");
    println!("───────────────────────────────────────────\n");

    println!("┌────────────────────┬──────────────────────────┬─────────────────┐");
    println!("│ Type               │ Eigenvalue Condition     │ Phase Portrait  │");
    println!("├────────────────────┼──────────────────────────┼─────────────────┤");
    println!("│ Stable Node        │ λ₁,λ₂ < 0, real          │ Straight lines  │");
    println!("│ Stable Spiral      │ Re(λ) < 0, Im(λ) ≠ 0     │ Inward spiral   │");
    println!("│ Unstable Node      │ λ₁,λ₂ > 0, real          │ Diverging lines │");
    println!("│ Unstable Spiral    │ Re(λ) > 0, Im(λ) ≠ 0     │ Outward spiral  │");
    println!("│ Saddle Point       │ λ₁ < 0 < λ₂              │ Hyperbolic      │");
    println!("│ Center             │ Re(λ) = 0, Im(λ) ≠ 0     │ Elliptic orbits │");
    println!("│ Degenerate         │ λ = 0 (at least one)     │ Line of fixed   │");
    println!("└────────────────────┴──────────────────────────┴─────────────────┘");

    println!("\n═══════════════════════════════════════════════════════════════");
    println!("                    DEMO COMPLETE");
    println!("═══════════════════════════════════════════════════════════════");

    Ok(())
}
