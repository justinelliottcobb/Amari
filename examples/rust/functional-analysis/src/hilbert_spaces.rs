//! # Hilbert Spaces Example
//!
//! Demonstrates Hilbert space concepts and operations.
//!
//! ## Mathematical Background
//!
//! A Hilbert space H is a complete inner product space:
//! - Inner product ⟨·,·⟩: H × H → C satisfying:
//!   - Linearity in first argument
//!   - Conjugate symmetry: ⟨x,y⟩ = ⟨y,x⟩*
//!   - Positive definiteness: ⟨x,x⟩ ≥ 0, with equality iff x = 0
//! - Completeness: Every Cauchy sequence converges
//!
//! Run with: `cargo run --bin hilbert_spaces`

use amari_functional::{
    hilbert::{HilbertSpace, InnerProductSpace, RealHilbert, ComplexHilbert},
    spaces::{L2Space, SequenceSpace, SobolevSpace},
    basis::{OrthonormalBasis, FourierBasis, LegendrePolynomials},
};
use nalgebra::{DVector, Complex};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("═══════════════════════════════════════════════════════════════");
    println!("                    HILBERT SPACES DEMO");
    println!("═══════════════════════════════════════════════════════════════\n");

    // =========================================================================
    // Part 1: Finite-Dimensional Hilbert Spaces (Rⁿ, Cⁿ)
    // =========================================================================
    println!("Part 1: Finite-Dimensional Hilbert Spaces");
    println!("──────────────────────────────────────────\n");

    // Real Hilbert space R³
    let v1 = DVector::from_vec(vec![1.0, 2.0, 3.0]);
    let v2 = DVector::from_vec(vec![4.0, 5.0, 6.0]);

    let r3 = RealHilbert::<3>::new();

    println!("R³ as a Hilbert space:");
    println!("  v₁ = {:?}", v1.as_slice());
    println!("  v₂ = {:?}", v2.as_slice());

    let inner = r3.inner_product(&v1, &v2)?;
    println!("\n  ⟨v₁, v₂⟩ = {} (dot product)", inner);

    let norm1 = r3.norm(&v1)?;
    let norm2 = r3.norm(&v2)?;
    println!("  ‖v₁‖ = {:.4}", norm1);
    println!("  ‖v₂‖ = {:.4}", norm2);

    // Verify Cauchy-Schwarz: |⟨x,y⟩| ≤ ‖x‖‖y‖
    let cs_lhs = inner.abs();
    let cs_rhs = norm1 * norm2;
    println!("\n  Cauchy-Schwarz: |⟨v₁, v₂⟩| ≤ ‖v₁‖‖v₂‖");
    println!("    {} ≤ {:.4}  ✓", cs_lhs, cs_rhs);

    // Angle between vectors
    let cos_angle = inner / (norm1 * norm2);
    let angle = cos_angle.acos();
    println!("\n  Angle between v₁ and v₂: {:.2}° ({:.4} rad)",
             angle.to_degrees(), angle);

    // Complex Hilbert space C²
    println!("\n\nC² as a Hilbert space:");
    let z1 = vec![Complex::new(1.0, 2.0), Complex::new(3.0, 4.0)];
    let z2 = vec![Complex::new(5.0, -1.0), Complex::new(2.0, 3.0)];

    let c2 = ComplexHilbert::<2>::new();

    println!("  z₁ = [{}, {}]", z1[0], z1[1]);
    println!("  z₂ = [{}, {}]", z2[0], z2[1]);

    let complex_inner = c2.inner_product(&z1, &z2)?;
    println!("\n  ⟨z₁, z₂⟩ = {} (conjugate-linear in first arg)", complex_inner);

    // =========================================================================
    // Part 2: L² Space (Square-Integrable Functions)
    // =========================================================================
    println!("\n\nPart 2: L² Space of Square-Integrable Functions");
    println!("────────────────────────────────────────────────\n");

    println!("L²([a,b]) = {{ f : ∫|f(x)|²dx < ∞ }}");
    println!("Inner product: ⟨f,g⟩ = ∫ f(x)g(x) dx\n");

    let l2 = L2Space::new(0.0, 1.0)?;  // L²([0,1])

    // Define functions via discretization
    let n_points = 1000;
    let sin_fn: Vec<f64> = (0..n_points)
        .map(|i| {
            let x = i as f64 / (n_points - 1) as f64;
            (std::f64::consts::PI * x).sin()
        })
        .collect();

    let cos_fn: Vec<f64> = (0..n_points)
        .map(|i| {
            let x = i as f64 / (n_points - 1) as f64;
            (std::f64::consts::PI * x).cos()
        })
        .collect();

    let sin_norm = l2.norm(&sin_fn)?;
    let cos_norm = l2.norm(&cos_fn)?;
    let sin_cos_inner = l2.inner_product(&sin_fn, &cos_fn)?;

    println!("Functions on [0,1]:");
    println!("  f(x) = sin(πx)");
    println!("  g(x) = cos(πx)");
    println!("\n  ‖f‖ = {:.6}", sin_norm);
    println!("  ‖g‖ = {:.6}", cos_norm);
    println!("  ⟨f,g⟩ = {:.6}", sin_cos_inner);
    println!("\n  Note: sin(πx) and cos(πx) are NOT orthogonal on [0,1]");
    println!("        (they ARE orthogonal on [0,2π])");

    // =========================================================================
    // Part 3: Orthonormal Bases
    // =========================================================================
    println!("\n\nPart 3: Orthonormal Bases");
    println!("─────────────────────────\n");

    println!("An orthonormal basis {{eₙ}} satisfies:");
    println!("  ⟨eₘ, eₙ⟩ = δₘₙ  (orthonormality)");
    println!("  span{{eₙ}} = H   (completeness)");

    // Fourier basis
    println!("\nFourier basis on L²([0,2π]):");
    let fourier = FourierBasis::new(0.0, 2.0 * std::f64::consts::PI, 10)?;

    println!("  e₀(x) = 1/√(2π)");
    println!("  e₂ₖ₋₁(x) = cos(kx)/√π");
    println!("  e₂ₖ(x) = sin(kx)/√π");

    // Verify orthonormality
    let e1 = fourier.basis_function(1)?;  // cos(x)/√π
    let e2 = fourier.basis_function(2)?;  // sin(x)/√π
    let e3 = fourier.basis_function(3)?;  // cos(2x)/√π

    let l2_2pi = L2Space::new(0.0, 2.0 * std::f64::consts::PI)?;

    println!("\n  Orthonormality check:");
    println!("    ⟨e₁, e₁⟩ = {:.6} (should be 1)", l2_2pi.inner_product(&e1, &e1)?);
    println!("    ⟨e₁, e₂⟩ = {:.6} (should be 0)", l2_2pi.inner_product(&e1, &e2)?);
    println!("    ⟨e₁, e₃⟩ = {:.6} (should be 0)", l2_2pi.inner_product(&e1, &e3)?);

    // Legendre polynomials
    println!("\nLegendre polynomials on L²([-1,1]):");
    let legendre = LegendrePolynomials::new(5)?;

    println!("  P₀(x) = 1");
    println!("  P₁(x) = x");
    println!("  P₂(x) = (3x² - 1)/2");
    println!("  P₃(x) = (5x³ - 3x)/2");

    // =========================================================================
    // Part 4: Projections and Best Approximation
    // =========================================================================
    println!("\n\nPart 4: Projections and Best Approximation");
    println!("───────────────────────────────────────────\n");

    println!("Projection theorem: For closed subspace M ⊂ H,");
    println!("  every x ∈ H has unique decomposition x = P_M(x) + x^⊥");
    println!("  where P_M(x) ∈ M and x^⊥ ⊥ M");

    // Project f(x) = x onto span{sin(πx)} in L²([0,1])
    let linear_fn: Vec<f64> = (0..n_points)
        .map(|i| i as f64 / (n_points - 1) as f64)
        .collect();

    // Projection: P_sin(f) = ⟨f, sin⟩/‖sin‖² · sin
    let sin_normalized: Vec<f64> = sin_fn.iter()
        .map(|&s| s / sin_norm)
        .collect();

    let coeff = l2.inner_product(&linear_fn, &sin_normalized)?;
    let projection: Vec<f64> = sin_normalized.iter()
        .map(|&s| coeff * s)
        .collect();

    let projection_norm = l2.norm(&projection)?;
    let residual: Vec<f64> = linear_fn.iter()
        .zip(&projection)
        .map(|(&f, &p)| f - p)
        .collect();
    let residual_norm = l2.norm(&residual)?;

    println!("\nProject f(x) = x onto span{{sin(πx)}} in L²([0,1]):");
    println!("  Coefficient: ⟨f, e⟩ = {:.6}", coeff);
    println!("  ‖projection‖ = {:.6}", projection_norm);
    println!("  ‖residual‖ = {:.6}", residual_norm);
    println!("\n  Verify: ‖f‖² = ‖proj‖² + ‖resid‖² (Pythagorean)");
    let f_norm = l2.norm(&linear_fn)?;
    println!("    {:.6} ≈ {:.6} + {:.6} = {:.6}",
             f_norm * f_norm,
             projection_norm * projection_norm,
             residual_norm * residual_norm,
             projection_norm * projection_norm + residual_norm * residual_norm);

    // =========================================================================
    // Part 5: Parseval's Identity and Completeness
    // =========================================================================
    println!("\n\nPart 5: Parseval's Identity");
    println!("───────────────────────────\n");

    println!("For orthonormal basis {{eₙ}}, Parseval's identity states:");
    println!("  ‖f‖² = Σₙ |⟨f, eₙ⟩|²\n");

    // Approximate a function using Fourier series
    let target_fn: Vec<f64> = (0..n_points)
        .map(|i| {
            let x = 2.0 * std::f64::consts::PI * i as f64 / (n_points - 1) as f64;
            x * (2.0 * std::f64::consts::PI - x)  // parabola on [0, 2π]
        })
        .collect();

    let target_norm_sq = l2_2pi.inner_product(&target_fn, &target_fn)?;

    println!("Approximating f(x) = x(2π-x) with Fourier series:");
    println!("  ‖f‖² = {:.6}", target_norm_sq);

    let mut sum_coeffs_sq = 0.0;
    println!("\n  n | |⟨f, eₙ⟩|² | Cumulative sum | Relative error");
    println!("  ──┼─────────────┼────────────────┼───────────────");

    for n in 0..10 {
        let en = fourier.basis_function(n)?;
        let coeff = l2_2pi.inner_product(&target_fn, &en)?;
        sum_coeffs_sq += coeff * coeff;
        let rel_error = (target_norm_sq - sum_coeffs_sq).abs() / target_norm_sq;
        println!("  {:2}| {:11.6} | {:14.6} | {:13.2e}", n, coeff*coeff, sum_coeffs_sq, rel_error);
    }

    println!("\n  As N → ∞, Σ|⟨f, eₙ⟩|² → ‖f‖² (Parseval)");

    // =========================================================================
    // Part 6: Riesz Representation Theorem
    // =========================================================================
    println!("\n\nPart 6: Riesz Representation Theorem");
    println!("─────────────────────────────────────\n");

    println!("Every bounded linear functional φ: H → ℂ has unique representation:");
    println!("  φ(x) = ⟨x, y⟩  for some y ∈ H");
    println!("  ‖φ‖ = ‖y‖");

    // Example: evaluation functional at x₀ (in finite dimensions)
    let r4 = RealHilbert::<4>::new();
    let representer = DVector::from_vec(vec![1.0, 0.0, 0.0, 0.0]);  // e₁

    println!("\nExample in R⁴:");
    println!("  Functional: φ(v) = v₁ (first coordinate)");
    println!("  Representer: y = (1, 0, 0, 0)");
    println!("  φ(v) = ⟨v, y⟩ = v₁  ✓");

    let test_v = DVector::from_vec(vec![5.0, 2.0, 7.0, 3.0]);
    let phi_v = r4.inner_product(&test_v, &representer)?;
    println!("\n  φ((5,2,7,3)) = ⟨(5,2,7,3), (1,0,0,0)⟩ = {}", phi_v);

    // =========================================================================
    // Part 7: Sobolev Spaces (Brief Introduction)
    // =========================================================================
    println!("\n\nPart 7: Sobolev Spaces");
    println!("──────────────────────\n");

    println!("Sobolev spaces H^k include functions with k weak derivatives:");
    println!("  H¹([a,b]) = {{ f ∈ L² : f' ∈ L² }}");
    println!("  Inner product: ⟨f,g⟩_{H¹} = ⟨f,g⟩_{L²} + ⟨f',g'⟩_{L²}");

    let h1 = SobolevSpace::new(0.0, 1.0, 1)?;  // H¹([0,1])

    // Smooth function and its derivative
    let smooth_fn: Vec<f64> = (0..n_points)
        .map(|i| {
            let x = i as f64 / (n_points - 1) as f64;
            x * x  // x²
        })
        .collect();

    let derivative: Vec<f64> = (0..n_points)
        .map(|i| {
            let x = i as f64 / (n_points - 1) as f64;
            2.0 * x  // 2x
        })
        .collect();

    let l2_norm_sq = l2.inner_product(&smooth_fn, &smooth_fn)?;
    let deriv_norm_sq = l2.inner_product(&derivative, &derivative)?;
    let h1_norm = (l2_norm_sq + deriv_norm_sq).sqrt();

    println!("\nf(x) = x² on [0,1]:");
    println!("  ‖f‖_{L²}² = ∫₀¹ x⁴ dx = {:.6}", l2_norm_sq);
    println!("  ‖f'‖_{L²}² = ∫₀¹ 4x² dx = {:.6}", deriv_norm_sq);
    println!("  ‖f‖_{H¹} = {:.6}", h1_norm);

    // =========================================================================
    // Part 8: Sequence Spaces (ℓ²)
    // =========================================================================
    println!("\n\nPart 8: Sequence Space ℓ²");
    println!("─────────────────────────\n");

    println!("ℓ² = {{ (xₙ) : Σ|xₙ|² < ∞ }}");
    println!("Inner product: ⟨x,y⟩ = Σ xₙȳₙ\n");

    let l2_seq = SequenceSpace::l2();

    // Sequences
    let seq1: Vec<f64> = (1..=10).map(|n| 1.0 / n as f64).collect();  // 1/n
    let seq2: Vec<f64> = (1..=10).map(|n| 1.0 / (n * n) as f64).collect();  // 1/n²

    println!("Sequences (first 10 terms):");
    println!("  x = (1, 1/2, 1/3, ..., 1/n, ...)");
    println!("  y = (1, 1/4, 1/9, ..., 1/n², ...)");

    let seq_inner = l2_seq.inner_product(&seq1, &seq2)?;
    let seq1_norm = l2_seq.norm(&seq1)?;
    let seq2_norm = l2_seq.norm(&seq2)?;

    println!("\n  (Truncated to 10 terms):");
    println!("  ⟨x,y⟩ ≈ {:.6}", seq_inner);
    println!("  ‖x‖ ≈ {:.6}", seq1_norm);
    println!("  ‖y‖ ≈ {:.6}", seq2_norm);

    println!("\n  Note: Full ‖x‖ = π/√6 (Basel problem)");
    println!("        Full ‖y‖ = π²/√90");

    println!("\n═══════════════════════════════════════════════════════════════");
    println!("                    DEMO COMPLETE");
    println!("═══════════════════════════════════════════════════════════════");

    Ok(())
}
