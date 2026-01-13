//! # Lyapunov Exponents Example
//!
//! Demonstrates computation of Lyapunov exponents for chaos detection and
//! characterization of strange attractors.
//!
//! ## Mathematical Background
//!
//! Lyapunov exponents measure the average rate of separation of infinitesimally
//! close trajectories:
//! ```text
//! λ = lim(t→∞) (1/t) ln(|δx(t)|/|δx(0)|)
//! ```
//!
//! For an n-dimensional system:
//! - λ₁ > 0: Chaos (exponential divergence)
//! - Sum(λᵢ) < 0: Dissipative (volume contracting)
//! - Kaplan-Yorke dimension: D_KY = k + Σᵢ₌₁ᵏ λᵢ / |λₖ₊₁|
//!
//! Run with: `cargo run --bin lyapunov_exponents`

use amari_core::Multivector;
use amari_dynamics::{
    systems::{LorenzSystem, RosslerSystem, DuffingOscillator, VanDerPolOscillator},
    solver::{RungeKutta4, ODESolver},
    lyapunov::{LyapunovConfig, compute_lyapunov_spectrum, LyapunovSpectrum},
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("═══════════════════════════════════════════════════════════════");
    println!("                  LYAPUNOV EXPONENTS DEMO");
    println!("═══════════════════════════════════════════════════════════════\n");

    let config = LyapunovConfig {
        num_steps: 15000,
        dt: 0.01,
        transient_steps: 2000,
        orthonormalization_steps: 10,
    };

    // =========================================================================
    // Part 1: Lorenz System - Classic Chaos
    // =========================================================================
    println!("Part 1: Lorenz System (Classic Chaotic Attractor)");
    println!("─────────────────────────────────────────────────\n");

    let lorenz = LorenzSystem::classic();
    println!("Parameters: σ={}, ρ={}, β={:.4}", lorenz.sigma(), lorenz.rho(), lorenz.beta());

    let mut ic_lorenz: Multivector<3, 0, 0> = Multivector::zero();
    ic_lorenz.set(1, 1.0);
    ic_lorenz.set(2, 1.0);
    ic_lorenz.set(4, 1.0);

    let spectrum_lorenz = compute_lyapunov_spectrum(&lorenz, &ic_lorenz, &config)?;

    print_spectrum("Lorenz", &spectrum_lorenz);

    // Known values for comparison
    println!("\n  Literature values: λ₁ ≈ +0.906, λ₂ ≈ 0, λ₃ ≈ -14.57");

    // =========================================================================
    // Part 2: Rossler System - Simpler Chaos
    // =========================================================================
    println!("\n\nPart 2: Rossler System (Simpler Chaotic Attractor)");
    println!("───────────────────────────────────────────────────\n");

    let rossler = RosslerSystem::classic();
    println!("Parameters: a={}, b={}, c={}", rossler.a(), rossler.b(), rossler.c());

    let mut ic_rossler: Multivector<3, 0, 0> = Multivector::zero();
    ic_rossler.set(1, 1.0);
    ic_rossler.set(2, 1.0);
    ic_rossler.set(4, 1.0);

    let spectrum_rossler = compute_lyapunov_spectrum(&rossler, &ic_rossler, &config)?;

    print_spectrum("Rossler", &spectrum_rossler);

    println!("\n  Literature values: λ₁ ≈ +0.07, λ₂ ≈ 0, λ₃ ≈ -5.4");

    // =========================================================================
    // Part 3: Van der Pol - Limit Cycle (Non-Chaotic)
    // =========================================================================
    println!("\n\nPart 3: Van der Pol Oscillator (Limit Cycle - Non-Chaotic)");
    println!("──────────────────────────────────────────────────────────\n");

    let vdp = VanDerPolOscillator::new(1.0);
    println!("Parameter: μ={}", 1.0);

    let mut ic_vdp: Multivector<2, 0, 0> = Multivector::zero();
    ic_vdp.set(1, 0.1);
    ic_vdp.set(2, 0.0);

    let config_2d = LyapunovConfig {
        num_steps: 10000,
        dt: 0.01,
        transient_steps: 2000,
        orthonormalization_steps: 10,
    };

    let spectrum_vdp = compute_lyapunov_spectrum(&vdp, &ic_vdp, &config_2d)?;

    print_spectrum("Van der Pol", &spectrum_vdp);

    println!("\n  For a stable limit cycle: λ₁ = 0 (neutral along cycle),");
    println!("                            λ₂ < 0 (stable transverse)");

    // =========================================================================
    // Part 4: Comparison Table
    // =========================================================================
    println!("\n\nPart 4: System Comparison");
    println!("─────────────────────────\n");

    println!("System           | λ₁       | λ₂       | λ₃       | Sum      | D_KY   | Type");
    println!("─────────────────┼──────────┼──────────┼──────────┼──────────┼────────┼────────────");

    // Lorenz
    let s = &spectrum_lorenz;
    println!("Lorenz           | {:+8.4} | {:+8.4} | {:+8.4} | {:+8.4} | {:6.3} | {}",
             s.exponents[0], s.exponents[1], s.exponents[2],
             s.sum(), s.kaplan_yorke_dimension(),
             classify_dynamics(&s));

    // Rossler
    let s = &spectrum_rossler;
    println!("Rossler          | {:+8.4} | {:+8.4} | {:+8.4} | {:+8.4} | {:6.3} | {}",
             s.exponents[0], s.exponents[1], s.exponents[2],
             s.sum(), s.kaplan_yorke_dimension(),
             classify_dynamics(&s));

    // Van der Pol (2D, pad with N/A)
    let s = &spectrum_vdp;
    println!("Van der Pol      | {:+8.4} | {:+8.4} |   N/A    | {:+8.4} | {:6.3} | {}",
             s.exponents[0], s.exponents[1],
             s.sum(), s.kaplan_yorke_dimension(),
             classify_dynamics(&s));

    // =========================================================================
    // Part 5: Lorenz Parameter Dependence
    // =========================================================================
    println!("\n\nPart 5: Lyapunov Exponents vs Parameter (Lorenz ρ)");
    println!("───────────────────────────────────────────────────\n");

    let short_config = LyapunovConfig {
        num_steps: 8000,
        dt: 0.01,
        transient_steps: 1000,
        orthonormalization_steps: 10,
    };

    println!("  ρ      | λ₁       | λ₂       | λ₃       | Dynamics");
    println!("  ───────┼──────────┼──────────┼──────────┼──────────────────");

    for rho in [10.0, 15.0, 20.0, 24.0, 24.74, 28.0, 100.0, 160.0] {
        let system = LorenzSystem::new(10.0, rho, 8.0/3.0);

        let mut ic: Multivector<3, 0, 0> = Multivector::zero();
        ic.set(1, 1.0);
        ic.set(2, 1.0);
        ic.set(4, 1.0);

        match compute_lyapunov_spectrum(&system, &ic, &short_config) {
            Ok(spec) => {
                let dynamics = classify_dynamics(&spec);
                println!("  {:6.1} | {:+8.4} | {:+8.4} | {:+8.4} | {}",
                         rho, spec.exponents[0], spec.exponents[1], spec.exponents[2],
                         dynamics);
            }
            Err(_) => {
                println!("  {:6.1} | (computation failed)", rho);
            }
        }
    }

    println!("\nNote: ρ ≈ 24.74 is the critical value for onset of chaos");

    // =========================================================================
    // Part 6: Interpretation Guide
    // =========================================================================
    println!("\n\nPart 6: Interpretation Guide");
    println!("────────────────────────────\n");

    println!("Lyapunov Exponent Interpretation:");
    println!("──────────────────────────────────");
    println!("  λ > 0  : Exponential divergence (chaos indicator)");
    println!("  λ = 0  : Neutral direction (flow along attractor)");
    println!("  λ < 0  : Exponential convergence (stable direction)");
    println!();
    println!("Attractor Classification:");
    println!("──────────────────────────");
    println!("  Fixed Point    : All λᵢ < 0");
    println!("  Limit Cycle    : λ₁ = 0, λᵢ < 0 for i > 1");
    println!("  Torus (2-freq) : λ₁ = λ₂ = 0, λᵢ < 0 for i > 2");
    println!("  Strange Attr.  : λ₁ > 0, Sum(λᵢ) < 0 (dissipative chaos)");
    println!();
    println!("Kaplan-Yorke Dimension:");
    println!("────────────────────────");
    println!("  D_KY = k + (λ₁ + ... + λₖ) / |λₖ₊₁|");
    println!("  where k is largest index with Σᵢ≤ₖ λᵢ ≥ 0");
    println!("  D_KY gives fractal dimension of strange attractor");

    println!("\n═══════════════════════════════════════════════════════════════");
    println!("                    DEMO COMPLETE");
    println!("═══════════════════════════════════════════════════════════════");

    Ok(())
}

fn print_spectrum(name: &str, spectrum: &LyapunovSpectrum) {
    println!("Lyapunov spectrum for {}:", name);
    for (i, &exp) in spectrum.exponents.iter().enumerate() {
        let sign = if exp > 0.0 { "+" } else { "" };
        println!("  λ_{} = {}{:.4}", i + 1, sign, exp);
    }

    println!("\n  Sum of exponents: {:.4}", spectrum.sum());
    println!("  Kaplan-Yorke dimension: {:.3}", spectrum.kaplan_yorke_dimension());

    if spectrum.is_chaotic() {
        println!("  → CHAOTIC (positive largest exponent)");
    } else {
        println!("  → REGULAR (non-chaotic)");
    }
}

fn classify_dynamics(spectrum: &LyapunovSpectrum) -> &'static str {
    let lambda1 = spectrum.exponents[0];

    if lambda1 > 0.01 {
        "Chaotic"
    } else if lambda1 > -0.01 {
        if spectrum.exponents.len() > 1 && spectrum.exponents[1] > -0.01 {
            "Quasi-periodic"
        } else {
            "Limit cycle"
        }
    } else {
        "Fixed point"
    }
}
