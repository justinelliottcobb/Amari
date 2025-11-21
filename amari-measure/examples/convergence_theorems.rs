//! Convergence theorem examples
//!
//! This example demonstrates the fundamental convergence theorems
//! in Lebesgue integration theory:
//! - Monotone Convergence Theorem (MCT)
//! - Dominated Convergence Theorem (DCT)
//! - Fatou's Lemma
//!
//! Run with:
//! ```bash
//! cargo run --example convergence_theorems
//! ```

// Note: These types and functions are exported but not used in this example.
// The example focuses on conceptual demonstrations rather than actual integration computations.
#[allow(unused_imports)]
use amari_measure::{
    dominated_convergence, fatou_lemma, monotone_convergence, DominatedConvergenceResult,
    FatouResult, MonotoneConvergenceResult,
};

fn main() {
    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!("║       amari-measure: Convergence Theorem Examples           ║");
    println!("╚═══════════════════════════════════════════════════════════════╝\n");

    // ========================================================================
    // 1. Why Convergence Theorems Matter
    // ========================================================================
    println!("─────────────────────────────────────────────────────────");
    println!("1. Why Convergence Theorems Matter");
    println!("─────────────────────────────────────────────────────────\n");

    println!("  Convergence theorems answer the fundamental question:");
    println!("    When can we exchange limits and integrals?\n");

    println!("  The problem:");
    println!("    Given fₙ → f, does ∫fₙ → ∫f?\n");

    println!("  Classical counterexample:");
    println!("    fₙ(x) = n·χ_{{[0,1/n]}}(x)  on [0,1]");
    println!();
    println!("    • fₙ → 0 pointwise everywhere");
    println!("    • ∫₀¹ fₙ dx = n·(1/n) = 1  for all n");
    println!("    • ∫₀¹ 0 dx = 0");
    println!();
    println!("    So ∫fₙ = 1 ↛ 0 = ∫0  (limit fails!)\n");

    println!("  Convergence theorems provide sufficient conditions");
    println!("  under which the exchange is valid:\n");

    println!("    ✓ Monotone Convergence: fₙ ↗ f (non-negative, increasing)");
    println!("    ✓ Dominated Convergence: |fₙ| ≤ g (bounded by integrable g)");
    println!("    ✓ Fatou's Lemma: Lower bound (lim inf inequality)\n");

    // ========================================================================
    // 2. Monotone Convergence Theorem (MCT)
    // ========================================================================
    println!("─────────────────────────────────────────────────────────");
    println!("2. Monotone Convergence Theorem (Beppo Levi, 1906)");
    println!("─────────────────────────────────────────────────────────\n");

    println!("  Theorem:");
    println!("    If 0 ≤ f₁ ≤ f₂ ≤ f₃ ≤ ... and fₙ → f a.e.,");
    println!("    then:");
    println!("      lim_{{n→∞}} ∫fₙ dμ = ∫f dμ");
    println!();
    println!("  (Both sides may be infinite)\n");

    println!("  Conditions:");
    println!("    1. Non-negativity: fₙ ≥ 0 for all n");
    println!("    2. Monotonicity: fₙ ≤ fₙ₊₁ for all n");
    println!("    3. Pointwise convergence: fₙ(x) → f(x) a.e.\n");

    println!("  Key insight:");
    println!("    Monotonicity prevents mass from \"escaping to infinity\"");
    println!("    as in the counterexample above.\n");

    println!("  Applications:");
    println!("    • Series of non-negative functions:");
    println!("      ∫(∑ fₙ) dμ = ∑(∫ fₙ dμ)  (Tonelli)");
    println!();
    println!("    • Construction of Lebesgue integral:");
    println!("      ∫f dμ = sup{{ ∫s dμ : s simple, s ≤ f }}");
    println!();
    println!("    • Proving other convergence results\n");

    println!("  Example 1: Summing an infinite series");
    println!("    fₙ(x) = ∑_{{k=1}}^n x^k / k!  on [0,1]");
    println!("    f(x) = e^x - 1");
    println!();
    println!("    ∫₀¹ (∑_{{k=1}}^∞ x^k/k!) dx = ∑_{{k=1}}^∞ ∫₀¹ x^k/k! dx");
    println!("                                = ∑_{{k=1}}^∞ 1/(k!(k+1))");
    println!("                                = ∫₀¹ (e^x - 1) dx = e - 2\n");

    println!("  Example 2: Improper integrals");
    println!("    ∫₀^∞ f(x) dx = lim_{{n→∞}} ∫₀ⁿ f(x) dx");
    println!();
    println!("    Use fₙ = f·χ_{{[0,n]}} ↗ f (monotone increasing)\n");

    // ========================================================================
    // 3. Dominated Convergence Theorem (DCT)
    // ========================================================================
    println!("─────────────────────────────────────────────────────────");
    println!("3. Dominated Convergence Theorem (Lebesgue, 1904)");
    println!("─────────────────────────────────────────────────────────\n");

    println!("  Theorem:");
    println!("    If fₙ → f a.e., |fₙ| ≤ g, and ∫g dμ < ∞,");
    println!("    then:");
    println!("      lim_{{n→∞}} ∫fₙ dμ = ∫f dμ");
    println!();
    println!("  Moreover:");
    println!("    lim_{{n→∞}} ∫|fₙ - f| dμ = 0  (L¹ convergence)\n");

    println!("  Conditions:");
    println!("    1. Pointwise convergence: fₙ(x) → f(x) a.e.");
    println!("    2. Dominating function: |fₙ(x)| ≤ g(x) for all n, x");
    println!("    3. Integrable dominator: ∫g dμ < ∞\n");

    println!("  Key insight:");
    println!("    The dominating function g prevents mass from escaping");
    println!("    to infinity or concentrating at a point.\n");

    println!("  Applications:");
    println!("    • Differentiation under integral sign (Leibniz rule):");
    println!("      d/dx ∫f(x,y) dy = ∫(∂f/∂x)(x,y) dy");
    println!();
    println!("    • Fourier analysis (Riemann-Lebesgue lemma)");
    println!();
    println!("    • Probability: expectations of converging r.v.s");
    println!();
    println!("    • Quantum mechanics: time evolution of states\n");

    println!("  Example 1: Differentiation under the integral");
    println!("    F(t) = ∫₀^∞ e^{{-tx²}} dx  for t > 0");
    println!();
    println!("    F'(t) = d/dt ∫₀^∞ e^{{-tx²}} dx");
    println!("          = ∫₀^∞ -x²e^{{-tx²}} dx  (DCT with g(x) = x²e^{{-x²}})");
    println!();
    println!("    Dominating: |(-x²e^{{-tx²}})| ≤ x²e^{{-x²}} for t ≥ 1\n");

    println!("  Example 2: Fourier transform");
    println!("    f̂ₙ(ξ) = ∫ fₙ(x)e^{{-2πiξx}} dx");
    println!();
    println!("    If fₙ → f in L¹, then f̂ₙ → f̂ uniformly");
    println!("    (using |fₙ(x)e^{{-2πiξx}}| ≤ |fₙ(x)| and DCT)\n");

    // ========================================================================
    // 4. Fatou's Lemma
    // ========================================================================
    println!("─────────────────────────────────────────────────────────");
    println!("4. Fatou's Lemma (Pierre Fatou, 1906)");
    println!("─────────────────────────────────────────────────────────\n");

    println!("  Lemma:");
    println!("    If fₙ ≥ 0 for all n, then:");
    println!("      ∫ lim inf_{{n→∞}} fₙ dμ ≤ lim inf_{{n→∞}} ∫fₙ dμ\n");

    println!("  Conditions:");
    println!("    • Non-negativity: fₙ ≥ 0 for all n");
    println!("    • Measurability: fₙ measurable for all n\n");

    println!("  Note: Inequality can be strict!");
    println!("    fₙ = χ_{{[n, n+1]}}  on ℝ");
    println!("    • lim inf fₙ = 0  (no overlap for large n)");
    println!("    • ∫fₙ dx = 1 for all n");
    println!("    • ∫0 dx = 0 < 1 = lim inf ∫fₙ dx\n");

    println!("  Applications:");
    println!("    • Proving dominated convergence theorem");
    println!("    • Lower semicontinuity in calculus of variations");
    println!("    • Proving existence of minimizers");
    println!("    • Probability: Fatou's lemma for expectations\n");

    println!("  Example: Existence of minimizers");
    println!("    Minimize E[f] = ∫|∇f|² + V(x)|f|² dx");
    println!();
    println!("    If fₙ is minimizing sequence (E[fₙ] → inf E),");
    println!("    and fₙ ⇀ f weakly, then by Fatou:");
    println!("      E[f] ≤ lim inf E[fₙ] = inf E");
    println!();
    println!("    So f is a minimizer.\n");

    // ========================================================================
    // 5. Comparison of Theorems
    // ========================================================================
    println!("─────────────────────────────────────────────────────────");
    println!("5. Comparison of Convergence Theorems");
    println!("─────────────────────────────────────────────────────────\n");

    println!("┌────────────┬───────────────┬──────────────┬─────────────┐");
    println!("│ Theorem    │ Conditions    │ Conclusion   │ Strictness  │");
    println!("├────────────┼───────────────┼──────────────┼─────────────┤");
    println!("│ MCT        │ fₙ ≥ 0, fₙ ↗  │ ∫fₙ → ∫f     │ Equality    │");
    println!("│ DCT        │ |fₙ| ≤ g ∈ L¹ │ ∫fₙ → ∫f     │ Equality    │");
    println!("│ Fatou      │ fₙ ≥ 0        │ ∫lim inf ≤...│ Inequality  │");
    println!("└────────────┴───────────────┴──────────────┴─────────────┘\n");

    println!("  Relationships:");
    println!("    • MCT ⟹ DCT  (via Fatou)");
    println!("    • Fatou + monotonicity ⟹ MCT");
    println!("    • DCT is most commonly used in practice");
    println!("    • MCT is fundamental for construction\n");

    println!("  When to use which:");
    println!();
    println!("    Use MCT when:");
    println!("      ✓ Functions are non-negative");
    println!("      ✓ Sequence is monotone increasing");
    println!("      ✓ Example: infinite series ∑ aₙ\n");

    println!("    Use DCT when:");
    println!("      ✓ Functions can change sign");
    println!("      ✓ Bounded by integrable function");
    println!("      ✓ Example: differentiation under integral\n");

    println!("    Use Fatou when:");
    println!("      ✓ Only need lower bound (inequality)");
    println!("      ✓ No monotonicity or domination");
    println!("      ✓ Example: proving lower semicontinuity\n");

    // ========================================================================
    // 6. Common Mistakes
    // ========================================================================
    println!("─────────────────────────────────────────────────────────");
    println!("6. Common Mistakes and Pitfalls");
    println!("─────────────────────────────────────────────────────────\n");

    println!("  Mistake 1: Forgetting domination");
    println!("    fₙ(x) = n·χ_{{[0,1/n]}}(x)");
    println!("    • fₙ → 0 pointwise");
    println!("    • No dominating function! (fₙ unbounded)");
    println!("    • ∫fₙ = 1 ↛ 0 = ∫0  ✗\n");

    println!("  Mistake 2: Forgetting non-negativity in MCT");
    println!("    fₙ(x) = (-1)^n/n");
    println!("    • fₙ → 0");
    println!("    • Not non-negative or monotone");
    println!("    • Cannot use MCT directly ✗\n");

    println!("  Mistake 3: Using Fatou backwards");
    println!("    Fatou: ∫lim inf ≤ lim inf ∫");
    println!("    NOT: lim sup ∫ ≤ ∫lim sup  (this is false!)\n");

    println!("  Mistake 4: Forgetting \"almost everywhere\"");
    println!("    Convergence/domination need only hold a.e.");
    println!("    Sets of measure zero don't affect integrals\n");

    // ========================================================================
    // Summary
    // ========================================================================
    println!("┌──────────────────────────────────────────────────────────┐");
    println!("│ Key Takeaways                                            │");
    println!("├──────────────────────────────────────────────────────────┤");
    println!("│ Convergence theorems let us exchange limits & integrals │");
    println!("│                                                          │");
    println!("│ Monotone Convergence (MCT):                              │");
    println!("│   • 0 ≤ f₁ ≤ f₂ ≤ ... → f                                │");
    println!("│   • ∫fₙ → ∫f  (equality)                                 │");
    println!("│   • Use: infinite series, constructive proofs            │");
    println!("│                                                          │");
    println!("│ Dominated Convergence (DCT):                             │");
    println!("│   • fₙ → f a.e., |fₙ| ≤ g ∈ L¹                           │");
    println!("│   • ∫fₙ → ∫f  (equality + L¹ convergence)                │");
    println!("│   • Use: differentiation under ∫, Fourier analysis       │");
    println!("│                                                          │");
    println!("│ Fatou's Lemma:                                           │");
    println!("│   • fₙ ≥ 0                                                │");
    println!("│   • ∫lim inf fₙ ≤ lim inf ∫fₙ  (inequality!)             │");
    println!("│   • Use: lower semicontinuity, minimization              │");
    println!("│                                                          │");
    println!("│ Power of Lebesgue theory:                                │");
    println!("│   • Handle discontinuous functions                       │");
    println!("│   • Robust limit theorems                                │");
    println!("│   • Foundation for modern analysis                       │");
    println!("└──────────────────────────────────────────────────────────┘\n");

    println!("  Further reading:");
    println!("    • Uniform integrability (generalized DCT)");
    println!("    • Vitali convergence theorem");
    println!("    • Egorov's theorem (a.e. → uniform on large sets)");
    println!("    • Lusin's theorem (measurable ≈ continuous)\n");
}
