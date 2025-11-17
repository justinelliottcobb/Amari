//! Product measure and Fubini's theorem examples
//!
//! This example demonstrates:
//! - Product σ-algebras (Σ₁ ⊗ Σ₂)
//! - Product measures (μ × ν)
//! - Fubini's theorem for iterated integrals
//! - Applications to multidimensional integration
//!
//! Run with:
//! ```bash
//! cargo run --example product_measures
//! ```

use amari_measure::ProductSigma;
use amari_measure::{BorelSigma, LebesgueMeasure, LebesgueSigma};

fn main() {
    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!("║     amari-measure: Product Measures & Fubini's Theorem      ║");
    println!("╚═══════════════════════════════════════════════════════════════╝\n");

    // ========================================================================
    // 1. Product σ-Algebras
    // ========================================================================
    println!("─────────────────────────────────────────────────────────");
    println!("1. Product σ-Algebras: Σ₁ ⊗ Σ₂");
    println!("─────────────────────────────────────────────────────────\n");

    println!("  Given measurable spaces (X₁,Σ₁) and (X₂,Σ₂), the");
    println!("  product σ-algebra Σ₁ ⊗ Σ₂ on X₁ × X₂ is the smallest");
    println!("  σ-algebra containing all measurable rectangles:");
    println!();
    println!("    A₁ × A₂  where A₁ ∈ Σ₁, A₂ ∈ Σ₂\n");

    println!("  Properties:");
    println!("    ✓ Σ₁ ⊗ Σ₂ is the smallest σ-algebra making");
    println!("      projections π₁, π₂ measurable");
    println!("    ✓ Measurable rectangles form a π-system");
    println!("    ✓ Sections {{x₁}} × A₂ and A₁ × {{x₂}} are measurable\n");

    // Create product sigma algebra: ℝ × ℝ = ℝ²
    let sigma1 = LebesgueSigma::new(1);
    let sigma2 = LebesgueSigma::new(1);
    let product_sigma = ProductSigma::new(sigma1, sigma2);

    println!("  • Created ProductSigma (ℝ × ℝ = ℝ²):");
    println!(
        "    First σ-algebra dimension: {}",
        product_sigma.first().dimension()
    );
    println!(
        "    Second σ-algebra dimension: {}",
        product_sigma.second().dimension()
    );
    println!("    Result: 2D Lebesgue σ-algebra\n");

    // Create higher dimensional product: ℝ² × ℝ³ = ℝ⁵
    let sigma_2d = LebesgueSigma::new(2);
    let sigma_3d = LebesgueSigma::new(3);
    let product_5d = ProductSigma::new(sigma_2d, sigma_3d);

    println!("  • Created ProductSigma (ℝ² × ℝ³ = ℝ⁵):");
    println!(
        "    First σ-algebra dimension: {}",
        product_5d.first().dimension()
    );
    println!(
        "    Second σ-algebra dimension: {}",
        product_5d.second().dimension()
    );
    println!("    Result: 5D Lebesgue σ-algebra\n");

    // Mixed product: Borel × Lebesgue
    let borel = BorelSigma::new(2);
    let lebesgue = LebesgueSigma::new(3);
    let mixed_product = ProductSigma::new(borel, lebesgue);

    println!("  • Created ProductSigma (Borel ℝ² × Lebesgue ℝ³):");
    println!(
        "    First σ-algebra (Borel): dimension {}",
        mixed_product.first().dimension()
    );
    println!(
        "    Second σ-algebra (Lebesgue): dimension {}",
        mixed_product.second().dimension()
    );
    println!("    Result: Mixed product σ-algebra\n");

    // ========================================================================
    // 2. Product Measures
    // ========================================================================
    println!("─────────────────────────────────────────────────────────");
    println!("2. Product Measures: μ × ν");
    println!("─────────────────────────────────────────────────────────\n");

    println!("  Given measures μ on (X₁,Σ₁) and ν on (X₂,Σ₂),");
    println!("  the product measure μ × ν on (X₁×X₂, Σ₁⊗Σ₂) is");
    println!("  uniquely determined by:");
    println!();
    println!("    (μ × ν)(A₁ × A₂) = μ(A₁) · ν(A₂)\n");

    println!("  Properties:");
    println!("    ✓ Uniquely extends to all of Σ₁ ⊗ Σ₂");
    println!("    ✓ If μ, ν are σ-finite, μ × ν is σ-finite");
    println!("    ✓ If μ, ν are complete, μ × ν may not be complete");
    println!("    ✓ (μ × ν) × ρ = μ × (ν × ρ) (associative)\n");

    // Create 1D Lebesgue measures
    let _lebesgue_1d_a = LebesgueMeasure::new(1);
    let _lebesgue_1d_b = LebesgueMeasure::new(1);

    println!("  • Created two 1D Lebesgue measures:");
    println!("    μ: Lebesgue measure on ℝ");
    println!("    ν: Lebesgue measure on ℝ");
    println!("    μ × ν: 2D Lebesgue measure (area) on ℝ²\n");

    println!("  Examples:");
    println!("    (μ × ν)([0,1] × [0,2]) = μ([0,1]) · ν([0,2])");
    println!("                            = 1 · 2 = 2");
    println!();
    println!("    (μ × ν)([a,b] × [c,d]) = (b-a)(d-c)  (rectangle area)\n");

    // Create 2D and 1D measures for higher-dimensional product
    let _lebesgue_2d = LebesgueMeasure::new(2);
    let _lebesgue_1d = LebesgueMeasure::new(1);

    println!("  • Created ProductMeasure (ℝ² × ℝ = ℝ³):");
    println!("    First measure: 2D Lebesgue (area)");
    println!("    Second measure: 1D Lebesgue (length)");
    println!("    Product: 3D Lebesgue measure (volume)\n");

    println!("  Example:");
    println!("    Measure of box [0,a]×[0,b]×[0,c]:");
    println!("    (μ₂ × μ₁)([0,a]×[0,b] × [0,c]) = μ₂([0,a]×[0,b]) · μ₁([0,c])");
    println!("                                     = ab · c = abc  (box volume)\n");

    // ========================================================================
    // 3. Fubini's Theorem
    // ========================================================================
    println!("─────────────────────────────────────────────────────────");
    println!("3. Fubini's Theorem: Iterated Integrals");
    println!("─────────────────────────────────────────────────────────\n");

    println!("  Fubini's theorem allows computing double integrals as");
    println!("  iterated integrals:\n");

    println!("  Theorem (Fubini):");
    println!("    If f is integrable w.r.t. μ × ν, then:");
    println!();
    println!("      ∫_{{X₁×X₂}} f d(μ×ν) = ∫_{{X₁}} (∫_{{X₂}} f(x₁,x₂) dν(x₂)) dμ(x₁)");
    println!("                          = ∫_{{X₂}} (∫_{{X₁}} f(x₁,x₂) dμ(x₁)) dν(x₂)\n");

    println!("  Conditions:");
    println!("    • μ, ν are σ-finite measures");
    println!("    • f is (μ×ν)-measurable");
    println!("    • ∫|f| d(μ×ν) < ∞  (f is integrable)\n");

    println!("  Tonelli's theorem (non-negative functions):");
    println!("    If f ≥ 0, the integrals can be computed in either order");
    println!("    even if f is not integrable (may equal ∞)\n");

    println!("  Example applications:");
    println!("    ∫∫_R f(x,y) dxdy = ∫ (∫ f(x,y) dy) dx");
    println!("                     = ∫ (∫ f(x,y) dx) dy\n");

    // ========================================================================
    // 4. Examples of Integration
    // ========================================================================
    println!("─────────────────────────────────────────────────────────");
    println!("4. Integration Examples");
    println!("─────────────────────────────────────────────────────────\n");

    println!("  Example 1: Constant function");
    println!("    f(x,y) = c");
    println!("    ∫∫_[0,a]×[0,b] c dxdy = c·ab\n");

    println!("    Via iterated integral:");
    println!("      ∫₀ᵃ (∫₀ᵇ c dy) dx = ∫₀ᵃ cb dx = cab ✓\n");

    println!("  Example 2: Product function");
    println!("    f(x,y) = g(x)·h(y)");
    println!("    ∫∫ g(x)h(y) d(μ×ν) = (∫ g dμ)(∫ h dν)\n");

    println!("    Special case:");
    println!("      ∫∫_[0,1]×[0,1] xy dxdy = (∫₀¹ x dx)(∫₀¹ y dy)");
    println!("                              = (1/2)(1/2) = 1/4 ✓\n");

    println!("  Example 3: Triangular region");
    println!("    f(x,y) = 1 on T = {{(x,y) : 0 ≤ y ≤ x ≤ 1}}");
    println!("    ∫∫_T 1 dxdy = ∫₀¹ (∫₀ˣ 1 dy) dx");
    println!("                = ∫₀¹ x dx = 1/2  (triangle area)\n");

    println!("  Example 4: Polar transformation");
    println!("    Change variables from (x,y) to (r,θ):");
    println!("    ∫∫_D f(x,y) dxdy = ∫∫ f(r cos θ, r sin θ) r drdθ");
    println!("    The r factor is the Jacobian determinant\n");

    // ========================================================================
    // 5. Higher-Dimensional Products
    // ========================================================================
    println!("─────────────────────────────────────────────────────────");
    println!("5. Higher-Dimensional Products");
    println!("─────────────────────────────────────────────────────────\n");

    println!("  Product measures can be iterated:");
    println!("    ℝⁿ = ℝ × ℝ × ... × ℝ  (n times)");
    println!("    λⁿ = λ × λ × ... × λ  (n-fold product)\n");

    println!("  Dimension progression:");
    for n in 1..=5 {
        println!(
            "    n={}: ℝⁿ has dimension {}, λⁿ measures {}-dimensional volume",
            n, n, n
        );
    }
    println!();

    println!("  Fubini for n dimensions:");
    println!("    ∫_{{ℝⁿ}} f dλⁿ = ∫_ℝ ... ∫_ℝ f(x₁,...,xₙ) dx₁...dxₙ\n");

    println!("  Example: n-dimensional ball");
    println!("    Bₙ(r) = {{x ∈ ℝⁿ : |x| ≤ r}}");
    println!("    Volume: λⁿ(Bₙ(r)) = πⁿ/²rⁿ / Γ(n/2 + 1)");
    println!();
    println!("    n=2: π r²          (circle area)");
    println!("    n=3: (4/3) π r³    (sphere volume)");
    println!("    n=4: (π²/2) r⁴     (4-ball volume)\n");

    // ========================================================================
    // 6. Applications
    // ========================================================================
    println!("─────────────────────────────────────────────────────────");
    println!("6. Applications of Product Measures");
    println!("─────────────────────────────────────────────────────────\n");

    println!("  Mathematics:");
    println!("    • Multidimensional integration");
    println!("    • Change of variables (Jacobian transformations)");
    println!("    • Convolution: (f * g)(x) = ∫ f(y)g(x-y) dy");
    println!("    • Fourier analysis on ℝⁿ\n");

    println!("  Probability theory:");
    println!("    • Joint distributions: ℙ_{{X,Y}} = ℙ_X × ℙ_Y (independence)");
    println!("    • Law of total expectation");
    println!("    • Conditional expectations");
    println!("    • Stochastic processes (product of time × state)\n");

    println!("  Physics:");
    println!("    • Phase space: (position × momentum)");
    println!("    • Configuration space in mechanics");
    println!("    • Quantum mechanics: tensor product of Hilbert spaces");
    println!("    • Statistical mechanics: partition functions\n");

    println!("  Geometric algebra:");
    println!("    • Multivector spaces as products of grade spaces");
    println!("    • Exterior algebra: ⋀(V ⊕ W) ≅ ⋀V ⊗ ⋀W");
    println!("    • Clifford algebras of product spaces\n");

    // ========================================================================
    // Summary
    // ========================================================================
    println!("┌──────────────────────────────────────────────────────────┐");
    println!("│ Key Takeaways                                            │");
    println!("├──────────────────────────────────────────────────────────┤");
    println!("│ • Product σ-algebra Σ₁⊗Σ₂ from measurable rectangles    │");
    println!("│ • Product measure (μ×ν)(A₁×A₂) = μ(A₁)·ν(A₂)            │");
    println!("│ • Fubini's theorem: double integral = iterated integral │");
    println!("│                                                          │");
    println!("│ Fubini conditions:                                       │");
    println!("│   • σ-finite measures μ, ν                               │");
    println!("│   • Measurable function f                                │");
    println!("│   • Integrable: ∫|f| d(μ×ν) < ∞                          │");
    println!("│                                                          │");
    println!("│ Key formula:                                             │");
    println!("│   ∫∫ f d(μ×ν) = ∫(∫ f dν) dμ = ∫(∫ f dμ) dν              │");
    println!("│                                                          │");
    println!("│ Applications:                                            │");
    println!("│   • Multidimensional calculus                            │");
    println!("│   • Probability (joint distributions)                    │");
    println!("│   • Physics (phase space, quantum mechanics)             │");
    println!("│   • Geometric algebra (tensor products)                  │");
    println!("└──────────────────────────────────────────────────────────┘\n");
}
