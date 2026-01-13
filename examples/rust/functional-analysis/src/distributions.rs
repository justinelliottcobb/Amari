//! # Distributions Example
//!
//! Demonstrates distribution theory (generalized functions).
//!
//! ## Mathematical Background
//!
//! Distributions generalize functions to include objects like:
//! - Dirac delta δ(x): ⟨δ, φ⟩ = φ(0)
//! - Derivatives of non-differentiable functions
//! - Fourier transforms of polynomials
//!
//! A distribution T is a continuous linear functional on test functions.
//!
//! Run with: `cargo run --bin distributions`

use amari_functional::{
    distributions::{
        Distribution, TestFunction, DiracDelta, HeavisideStep,
        PrincipalValue, RegularDistribution, SingularDistribution,
    },
    convolution::{Convolution, FundamentalSolution},
    fourier::{FourierTransform, TemperedDistribution},
    differential::{DistributionalDerivative, WeakDerivative},
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("═══════════════════════════════════════════════════════════════");
    println!("                    DISTRIBUTIONS DEMO");
    println!("═══════════════════════════════════════════════════════════════\n");

    // =========================================================================
    // Part 1: Test Functions
    // =========================================================================
    println!("Part 1: Test Functions");
    println!("──────────────────────\n");

    println!("Test functions φ ∈ D(ℝ):");
    println!("  - Smooth (C^∞)");
    println!("  - Compact support (zero outside bounded set)");
    println!("  - Example: bump functions");

    // Standard bump function
    let bump = TestFunction::bump(-1.0, 1.0)?;

    println!("\nBump function on [-1, 1]:");
    println!("  φ(x) = exp(-1/(1-x²)) for |x| < 1");
    println!("       = 0 otherwise");

    let sample_points = [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5];
    println!("\n  x    | φ(x)");
    println!("  ─────┼─────────");
    for x in sample_points {
        println!("  {:5.1} | {:.6}", x, bump.evaluate(x));
    }

    // Integral (should be normalized)
    let integral = bump.integrate()?;
    println!("\n  ∫φ(x) dx = {:.6}", integral);

    // =========================================================================
    // Part 2: Regular Distributions
    // =========================================================================
    println!("\n\nPart 2: Regular Distributions");
    println!("──────────────────────────────\n");

    println!("Regular distribution T_f from locally integrable f:");
    println!("  ⟨T_f, φ⟩ = ∫ f(x)φ(x) dx");

    // Distribution from f(x) = x²
    let f_squared = RegularDistribution::from_function(|x| x * x)?;

    let pairing1 = f_squared.apply(&bump)?;
    println!("\nFor f(x) = x² and bump function φ:");
    println!("  ⟨T_f, φ⟩ = ∫ x²φ(x) dx ≈ {:.6}", pairing1);

    // Distribution from f(x) = sin(x)
    let f_sin = RegularDistribution::from_function(|x| x.sin())?;
    let pairing2 = f_sin.apply(&bump)?;
    println!("\nFor f(x) = sin(x) and bump function φ:");
    println!("  ⟨T_f, φ⟩ = ∫ sin(x)φ(x) dx ≈ {:.6}", pairing2);

    // =========================================================================
    // Part 3: Dirac Delta
    // =========================================================================
    println!("\n\nPart 3: Dirac Delta Distribution");
    println!("──────────────────────────────────\n");

    println!("Dirac delta δ centered at x₀:");
    println!("  ⟨δ_{x₀}, φ⟩ = φ(x₀)");
    println!("  'Infinite spike at x₀, zero elsewhere'");

    let delta_0 = DiracDelta::at(0.0)?;
    let delta_half = DiracDelta::at(0.5)?;

    println!("\nApplying to bump function:");
    println!("  ⟨δ₀, φ⟩ = φ(0) = {:.6}", delta_0.apply(&bump)?);
    println!("  ⟨δ_{0.5}, φ⟩ = φ(0.5) = {:.6}", delta_half.apply(&bump)?);

    // Delta as limit
    println!("\nDelta as limit of regular functions:");
    println!("  δₙ(x) = n/√(2π) exp(-n²x²/2) → δ as n → ∞");

    let gaussian_approx = |n: f64, x: f64| -> f64 {
        n / (2.0 * std::f64::consts::PI).sqrt() * (-n * n * x * x / 2.0).exp()
    };

    println!("\n  n  | ⟨δₙ, φ⟩ (approximating φ(0))");
    println!("  ───┼───────────────────────────────");
    for n in [1, 10, 100, 1000] {
        let approx = RegularDistribution::from_function(|x| gaussian_approx(n as f64, x))?;
        let value = approx.apply(&bump)?;
        println!("  {:4}| {:.6}", n, value);
    }
    println!("\n  True value: φ(0) = {:.6}", bump.evaluate(0.0));

    // =========================================================================
    // Part 4: Heaviside Step Function
    // =========================================================================
    println!("\n\nPart 4: Heaviside Step Function");
    println!("────────────────────────────────\n");

    println!("Heaviside step H(x):");
    println!("  H(x) = 0 for x < 0");
    println!("       = 1 for x > 0");
    println!("  (value at 0 is convention-dependent)");

    let heaviside = HeavisideStep::new()?;

    println!("\nAs distribution:");
    println!("  ⟨H, φ⟩ = ∫₀^∞ φ(x) dx");

    let h_bump = heaviside.apply(&bump)?;
    println!("  ⟨H, φ⟩ = {:.6}", h_bump);
    println!("  (Half of ∫φ since φ is symmetric about 0)");

    // =========================================================================
    // Part 5: Distributional Derivatives
    // =========================================================================
    println!("\n\nPart 5: Distributional Derivatives");
    println!("───────────────────────────────────\n");

    println!("Derivative of distribution T:");
    println!("  ⟨T', φ⟩ = -⟨T, φ'⟩");
    println!("  (Integration by parts)");

    // Derivative of Heaviside = Dirac delta
    println!("\nH'(x) = δ(x):");
    let h_prime = heaviside.derivative()?;
    let delta_test = delta_0.apply(&bump)?;
    let h_prime_test = h_prime.apply(&bump)?;

    println!("  ⟨H', φ⟩ = {:.6}", h_prime_test);
    println!("  ⟨δ, φ⟩ = {:.6}", delta_test);
    println!("  Equal? {}", (h_prime_test - delta_test).abs() < 1e-6);

    // Derivative of delta
    println!("\nδ'(x) = -δ' (derivative of delta):");
    println!("  ⟨δ', φ⟩ = -⟨δ, φ'⟩ = -φ'(0)");

    let delta_prime = delta_0.derivative()?;
    let delta_prime_test = delta_prime.apply(&bump)?;
    let phi_prime_at_0 = bump.derivative()?.evaluate(0.0);

    println!("  ⟨δ', φ⟩ = {:.6}", delta_prime_test);
    println!("  -φ'(0) = {:.6}", -phi_prime_at_0);

    // Every distribution has derivatives
    println!("\nKey property: Every distribution is infinitely differentiable!");
    println!("  (Derivatives may be more singular)");

    // =========================================================================
    // Part 6: Weak Derivatives of Functions
    // =========================================================================
    println!("\n\nPart 6: Weak Derivatives");
    println!("────────────────────────\n");

    println!("A function f has weak derivative g if:");
    println!("  ∫ f·φ' dx = -∫ g·φ dx for all φ ∈ D");

    // |x| has weak derivative sgn(x)
    println!("\nf(x) = |x| has weak derivative g(x) = sgn(x)");
    println!("  (Classical derivative doesn't exist at 0)");

    let abs_fn = RegularDistribution::from_function(|x: f64| x.abs())?;
    let weak_deriv = abs_fn.derivative()?;

    // Test against bump function
    let weak_deriv_test = weak_deriv.apply(&bump)?;
    println!("\n  ⟨|x|', φ⟩ = {:.6}", weak_deriv_test);

    // =========================================================================
    // Part 7: Principal Value Distribution
    // =========================================================================
    println!("\n\nPart 7: Principal Value");
    println!("────────────────────────\n");

    println!("Principal value of 1/x:");
    println!("  ⟨PV(1/x), φ⟩ = lim_{ε→0} ∫_{|x|>ε} φ(x)/x dx");
    println!("  (Symmetric limit around singularity)");

    let pv_inv_x = PrincipalValue::inv_x()?;
    let pv_test = pv_inv_x.apply(&bump)?;

    println!("\n  ⟨PV(1/x), φ⟩ = {:.6}", pv_test);
    println!("  (Should be 0 for symmetric φ)");

    // Asymmetric test function
    let asymmetric_bump = TestFunction::asymmetric_bump(-1.0, 1.0)?;
    let pv_asym = pv_inv_x.apply(&asymmetric_bump)?;
    println!("\n  For asymmetric test function:");
    println!("  ⟨PV(1/x), ψ⟩ = {:.6}", pv_asym);

    // =========================================================================
    // Part 8: Convolution with Distributions
    // =========================================================================
    println!("\n\nPart 8: Convolution");
    println!("───────────────────\n");

    println!("Convolution with delta:");
    println!("  (f * δ)(x) = f(x)  (delta is identity)");
    println!("  (f * δ_a)(x) = f(x-a)  (shift by a)");

    // f * δ' = f'
    println!("\nConvolution with delta derivative:");
    println!("  (f * δ')(x) = f'(x)  (differentiation)");

    // Fundamental solutions
    println!("\nFundamental solutions of differential operators:");
    println!("  Lu = δ where L is differential operator");
    println!("  Example: (d²/dx² - k²)G = δ");
    println!("           G(x) = -e^{-k|x|} / (2k)");

    // =========================================================================
    // Part 9: Tempered Distributions and Fourier Transform
    // =========================================================================
    println!("\n\nPart 9: Tempered Distributions");
    println!("───────────────────────────────\n");

    println!("Tempered distributions S'(ℝ):");
    println!("  - Continuous functionals on Schwartz space S(ℝ)");
    println!("  - Schwartz functions: rapidly decreasing (all derivatives)");
    println!("  - Allows Fourier transform of distributions");

    println!("\nFourier transform of distributions:");
    println!("  ⟨F̂[T], φ⟩ = ⟨T, φ̂⟩");

    println!("\nKey transforms:");
    println!("  F̂[δ] = 1/(2π)^{1/2}  (constant)");
    println!("  F̂[1] = (2π)^{1/2} δ  (delta)");
    println!("  F̂[e^{iωx}] = (2π)^{1/2} δ(k-ω)  (shifted delta)");

    // =========================================================================
    // Part 10: Support of Distributions
    // =========================================================================
    println!("\n\nPart 10: Support of Distributions");
    println!("──────────────────────────────────\n");

    println!("Support of T = smallest closed set outside which T = 0");
    println!("  supp(δ_0) = {{0}}");
    println!("  supp(H) = [0, ∞)");
    println!("  supp(PV(1/x)) = ℝ");

    println!("\nSingular support = points where T is not smooth");
    println!("  sing supp(H) = {{0}}  (jump at 0)");
    println!("  sing supp(δ) = {{0}}  (point mass)");

    // =========================================================================
    // Part 11: Applications
    // =========================================================================
    println!("\n\nPart 11: Applications");
    println!("─────────────────────\n");

    println!("Physics:");
    println!("  - Point masses: m·δ(x - x₀)");
    println!("  - Point charges: q·δ(r - r₀)");
    println!("  - Green's functions: fundamental solutions");

    println!("\nPDEs:");
    println!("  - Weak solutions (Sobolev spaces)");
    println!("  - Heat kernel as fundamental solution");
    println!("  - Wave propagation");

    println!("\nSignal Processing:");
    println!("  - Impulse response = δ * system");
    println!("  - Sampling: f(t)·comb(t) = Σ f(n)·δ(t-n)");
    println!("  - Convolution theorems");

    println!("\nQuantum Mechanics:");
    println!("  - Position eigenstates |x⟩");
    println!("  - Momentum eigenstates |p⟩");
    println!("  - ⟨x|p⟩ = e^{ipx}/√(2π) (improper eigenfunction)");

    println!("\n═══════════════════════════════════════════════════════════════");
    println!("                    DEMO COMPLETE");
    println!("═══════════════════════════════════════════════════════════════");

    Ok(())
}
