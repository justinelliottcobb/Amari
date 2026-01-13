//! # Banach Spaces Example
//!
//! Demonstrates Banach space concepts beyond Hilbert spaces.
//!
//! ## Mathematical Background
//!
//! A Banach space is a complete normed vector space:
//! - Every Cauchy sequence converges
//! - Has a norm (but not necessarily inner product)
//! - Examples: Lᵖ spaces, C([a,b]), ℓᵖ sequences
//!
//! Run with: `cargo run --bin banach_spaces`

use amari_functional::{
    banach::{BanachSpace, NormedSpace, DualSpace},
    spaces::{LpSpace, ContinuousFunctions, SequenceLp, BoundedVariation},
    norms::{LpNorm, SupNorm, BVNorm},
    duality::{DualPairing, ReflexiveSpace},
    fixed_point::{BanachFixedPoint, Contraction},
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("═══════════════════════════════════════════════════════════════");
    println!("                    BANACH SPACES DEMO");
    println!("═══════════════════════════════════════════════════════════════\n");

    // =========================================================================
    // Part 1: Lᵖ Spaces
    // =========================================================================
    println!("Part 1: Lᵖ Spaces");
    println!("─────────────────\n");

    println!("Lᵖ([a,b]) = {{ f : ∫|f|ᵖ < ∞ }},  p ∈ [1,∞)");
    println!("  ‖f‖_p = (∫|f(x)|ᵖ dx)^(1/p)\n");

    let n_points = 1000;
    let test_fn: Vec<f64> = (0..n_points)
        .map(|i| {
            let x = i as f64 / (n_points - 1) as f64;
            x * x  // f(x) = x²
        })
        .collect();

    println!("For f(x) = x² on [0,1]:");

    for p in [1.0, 2.0, 3.0, 4.0, f64::INFINITY] {
        let lp = LpSpace::new(0.0, 1.0, p)?;
        let norm = lp.norm(&test_fn)?;

        if p.is_infinite() {
            println!("  ‖f‖_∞ = {:.6} (sup norm)", norm);
        } else {
            // Analytical: (∫₀¹ x^(2p) dx)^(1/p) = (1/(2p+1))^(1/p)
            let expected = (1.0 / (2.0 * p + 1.0)).powf(1.0 / p);
            println!("  ‖f‖_{} = {:.6} (expected: {:.6})", p as i32, norm, expected);
        }
    }

    // L∞ = essentially bounded functions
    println!("\nL^∞ space (essentially bounded):");
    let l_inf = LpSpace::l_infinity(0.0, 1.0)?;
    let bounded_fn: Vec<f64> = (0..n_points)
        .map(|i| {
            let x = i as f64 / (n_points - 1) as f64;
            x.sin()
        })
        .collect();
    let sup_norm = l_inf.norm(&bounded_fn)?;
    println!("  ‖sin(x)‖_∞ = {:.6} on [0,1]", sup_norm);

    // =========================================================================
    // Part 2: Sequence Spaces ℓᵖ
    // =========================================================================
    println!("\n\nPart 2: Sequence Spaces ℓᵖ");
    println!("──────────────────────────\n");

    println!("ℓᵖ = {{ (xₙ) : Σ|xₙ|ᵖ < ∞ }},  p ∈ [1,∞)");

    let seq: Vec<f64> = (1..=20).map(|n| 1.0 / n as f64).collect();
    println!("\nFor sequence (1/n):");

    for p in [1.0, 2.0, 3.0] {
        let lp_seq = SequenceLp::new(p)?;
        let norm = lp_seq.norm(&seq)?;

        // Note: sequence 1/n is in ℓᵖ only for p > 1
        let in_space = if p == 1.0 { "divergent" } else { "convergent" };
        println!("  ‖·‖_{} = {:.6} (truncated, series {})", p as i32, norm, in_space);
    }

    println!("\nℓ¹ ⊂ ℓ² ⊂ ℓ³ ⊂ ... ⊂ ℓ^∞");
    println!("  (Larger p allows more sequences)");

    // =========================================================================
    // Part 3: C([a,b]) - Continuous Functions
    // =========================================================================
    println!("\n\nPart 3: C([a,b]) - Continuous Functions");
    println!("────────────────────────────────────────\n");

    println!("C([a,b]) = continuous functions with sup norm");
    println!("  ‖f‖ = sup|f(x)| over x ∈ [a,b]");

    let c_space = ContinuousFunctions::new(0.0, 1.0)?;

    let f1: Vec<f64> = (0..n_points).map(|i| {
        let x = i as f64 / (n_points - 1) as f64;
        x
    }).collect();

    let f2: Vec<f64> = (0..n_points).map(|i| {
        let x = i as f64 / (n_points - 1) as f64;
        x * x
    }).collect();

    println!("\nf(x) = x on [0,1]:");
    println!("  ‖f‖ = {:.6}", c_space.norm(&f1)?);

    println!("\ng(x) = x² on [0,1]:");
    println!("  ‖g‖ = {:.6}", c_space.norm(&f2)?);

    // Distance in C space
    let distance = c_space.distance(&f1, &f2)?;
    println!("\n  d(f, g) = ‖f - g‖ = {:.6}", distance);
    println!("  (Maximum difference at x where |x - x²| is largest)");

    // =========================================================================
    // Part 4: Dual Spaces
    // =========================================================================
    println!("\n\nPart 4: Dual Spaces");
    println!("───────────────────\n");

    println!("Dual space X* = bounded linear functionals on X");
    println!("  φ: X → ℝ with |φ(x)| ≤ ‖φ‖·‖x‖");

    println!("\nClassic dualities:");
    println!("  (ℓ¹)* = ℓ^∞");
    println!("  (ℓᵖ)* = ℓᵍ  where 1/p + 1/q = 1 (1 < p < ∞)");
    println!("  (L^p)* = L^q  where 1/p + 1/q = 1 (1 < p < ∞)");
    println!("  (C([a,b]))* = signed Radon measures");

    // Dual pairing example
    let l2_seq = SequenceLp::new(2.0)?;
    let x_seq: Vec<f64> = (1..=10).map(|n| 1.0 / n as f64).collect();
    let y_seq: Vec<f64> = (1..=10).map(|n| 1.0 / (n * n) as f64).collect();

    let pairing = l2_seq.dual_pairing(&x_seq, &y_seq)?;
    println!("\nDual pairing ⟨x, y⟩ for x = (1/n), y = (1/n²):");
    println!("  ⟨x, y⟩ = Σ xₙyₙ = {:.6}", pairing);

    // =========================================================================
    // Part 5: Reflexive Spaces
    // =========================================================================
    println!("\n\nPart 5: Reflexive Spaces");
    println!("────────────────────────\n");

    println!("X is reflexive if X = X** (isometrically)");
    println!("  - Hilbert spaces are reflexive");
    println!("  - Lᵖ (1 < p < ∞) are reflexive");
    println!("  - L¹, L^∞, C([a,b]) are NOT reflexive");

    println!("\nReflexivity matters for:");
    println!("  - Weak compactness of unit ball");
    println!("  - Existence of minimizers");
    println!("  - Fixed point theorems");

    // =========================================================================
    // Part 6: Banach Fixed Point Theorem
    // =========================================================================
    println!("\n\nPart 6: Banach Fixed Point Theorem");
    println!("───────────────────────────────────\n");

    println!("If T: X → X is a contraction (‖Tx - Ty‖ ≤ q‖x - y‖, q < 1)");
    println!("on complete metric space X, then T has unique fixed point.");

    // Example: Solving f(x) = cos(x) as fixed point of T(x) = cos(x)
    println!("\nExample: Find fixed point of T(x) = cos(x)");
    println!("  (solving x = cos(x))");

    let mut x = 0.5;  // Initial guess
    println!("\n  Iteration | x_n       | T(x_n)    | |x_n - T(x_n)|");
    println!("  ──────────┼───────────┼───────────┼────────────────");

    for i in 0..10 {
        let tx = x.cos();
        let error = (x - tx).abs();
        println!("  {:9} | {:.7} | {:.7} | {:.2e}", i, x, tx, error);
        if error < 1e-10 {
            break;
        }
        x = tx;
    }

    println!("\n  Fixed point: x* ≈ {:.10}", x);
    println!("  (Dottie number)");

    // Contraction constant
    // |T'(x)| = |−sin(x)| ≤ sin(1) ≈ 0.84 < 1 for x near fixed point
    let q = 1.0f64.sin();
    println!("\n  Contraction constant q ≤ sin(1) ≈ {:.4}", q);
    println!("  Error bound: |x_n - x*| ≤ qⁿ/(1-q) · |x₁ - x₀|");

    // =========================================================================
    // Part 7: Bounded Variation
    // =========================================================================
    println!("\n\nPart 7: Functions of Bounded Variation");
    println!("───────────────────────────────────────\n");

    println!("BV([a,b]) = functions with bounded total variation");
    println!("  Var(f) = sup Σ|f(xᵢ) - f(xᵢ₋₁)| over partitions");

    let bv_space = BoundedVariation::new(0.0, 1.0)?;

    // Monotone function - variation = |f(b) - f(a)|
    let monotone: Vec<f64> = (0..n_points).map(|i| {
        i as f64 / (n_points - 1) as f64
    }).collect();

    // Oscillating function
    let oscillating: Vec<f64> = (0..n_points).map(|i| {
        let x = i as f64 / (n_points - 1) as f64;
        (10.0 * std::f64::consts::PI * x).sin()
    }).collect();

    let var_monotone = bv_space.variation(&monotone)?;
    let var_oscillating = bv_space.variation(&oscillating)?;

    println!("\nf(x) = x (monotone):");
    println!("  Var(f) = {:.6}", var_monotone);

    println!("\ng(x) = sin(10πx) (oscillating):");
    println!("  Var(g) = {:.6} (≈ 10 full oscillations × 2)", var_oscillating);

    // =========================================================================
    // Part 8: Embedding Theorems
    // =========================================================================
    println!("\n\nPart 8: Embedding Theorems");
    println!("──────────────────────────\n");

    println!("Space embeddings X ↪ Y (continuous injection):");
    println!("  - ℓᵖ ↪ ℓᵍ for p ≤ q");
    println!("  - L^q ↪ L^p for p ≤ q on bounded domains");
    println!("  - BV ↪ L^∞ ↪ Lᵖ for all p");
    println!("  - Sobolev: H¹ ↪ C([a,b]) in 1D");

    println!("\nCompact embeddings (bounded → precompact):");
    println!("  - Rellich-Kondrachov: H¹ ↪↪ L²");
    println!("  - Arzelà-Ascoli: equicontinuous families in C");

    // Demonstrate norm comparison
    println!("\nNorm comparison for f(x) = x on [0,1]:");
    let l1 = LpSpace::new(0.0, 1.0, 1.0)?;
    let l2 = LpSpace::new(0.0, 1.0, 2.0)?;
    let l4 = LpSpace::new(0.0, 1.0, 4.0)?;

    let norm_l1 = l1.norm(&monotone)?;
    let norm_l2 = l2.norm(&monotone)?;
    let norm_l4 = l4.norm(&monotone)?;
    let norm_sup = c_space.norm(&monotone)?;

    println!("  ‖f‖_1 = {:.6}", norm_l1);
    println!("  ‖f‖_2 = {:.6}", norm_l2);
    println!("  ‖f‖_4 = {:.6}", norm_l4);
    println!("  ‖f‖_∞ = {:.6}", norm_sup);
    println!("\n  ‖·‖_1 ≤ ‖·‖_2 ≤ ‖·‖_4 ≤ ‖·‖_∞ (on [0,1])");

    // =========================================================================
    // Part 9: Open Mapping and Closed Graph Theorems
    // =========================================================================
    println!("\n\nPart 9: Fundamental Theorems");
    println!("────────────────────────────\n");

    println!("Open Mapping Theorem:");
    println!("  If T: X → Y is bounded, linear, surjective,");
    println!("  and X, Y are Banach, then T maps open sets to open sets.");
    println!("  Consequence: If T is also injective, T⁻¹ is bounded.");

    println!("\nClosed Graph Theorem:");
    println!("  If T: X → Y is linear with closed graph,");
    println!("  and X, Y are Banach, then T is bounded.");
    println!("  (Graph closed: xₙ→x and Txₙ→y implies Tx=y)");

    println!("\nUniform Boundedness (Banach-Steinhaus):");
    println!("  If {Tₐ} are bounded operators with sup‖Tₐx‖ < ∞ for each x,");
    println!("  then sup‖Tₐ‖ < ∞.");

    // =========================================================================
    // Part 10: Hahn-Banach Theorem
    // =========================================================================
    println!("\n\nPart 10: Hahn-Banach Theorem");
    println!("────────────────────────────\n");

    println!("Hahn-Banach: Bounded linear functionals can be extended.");
    println!("  If φ: M → ℝ is bounded on subspace M ⊂ X,");
    println!("  there exists Φ: X → ℝ extending φ with ‖Φ‖ = ‖φ‖.");

    println!("\nConsequences:");
    println!("  1. X* separates points: φ(x) = 0 ∀φ implies x = 0");
    println!("  2. ‖x‖ = max{|φ(x)| : ‖φ‖ ≤ 1}");
    println!("  3. For any x₀ ≠ 0, exists φ with φ(x₀) = ‖x₀‖, ‖φ‖ = 1");

    println!("\n═══════════════════════════════════════════════════════════════");
    println!("                    DEMO COMPLETE");
    println!("═══════════════════════════════════════════════════════════════");

    Ok(())
}
