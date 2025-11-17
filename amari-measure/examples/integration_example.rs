//! Lebesgue integration examples
//!
//! This example demonstrates integration of measurable functions
//! with respect to measures, including:
//! - Simple functions and step functions
//! - Integration of general measurable functions
//! - Properties of the Lebesgue integral
//! - Comparison with Riemann integration
//!
//! Run with:
//! ```bash
//! cargo run --example integration_example
//! ```

use amari_measure::LebesgueMeasure;

fn main() {
    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!("║        amari-measure: Lebesgue Integration Examples         ║");
    println!("╚═══════════════════════════════════════════════════════════════╝\n");

    // ========================================================================
    // 1. What is Lebesgue Integration?
    // ========================================================================
    println!("─────────────────────────────────────────────────────────");
    println!("1. Lebesgue Integration: Integration w.r.t. Measures");
    println!("─────────────────────────────────────────────────────────\n");

    println!("  The Lebesgue integral generalizes the Riemann integral by");
    println!("  integrating with respect to a measure μ:");
    println!();
    println!("    ∫ f dμ = lim ∫ sₙ dμ");
    println!();
    println!("  where sₙ are simple functions approximating f.\n");

    println!("  Key differences from Riemann integration:");
    println!("    ✓ Partitions the range (y-axis) instead of domain (x-axis)");
    println!("    ✓ Handles discontinuous functions (e.g., Dirichlet function)");
    println!("    ✓ Integrates over arbitrary measurable sets");
    println!("    ✓ Powerful limit theorems (monotone, dominated convergence)\n");

    println!("  Properties:");
    println!("    • Linearity: ∫(af + bg)dμ = a∫f dμ + b∫g dμ");
    println!("    • Monotonicity: f ≤ g ⟹ ∫f dμ ≤ ∫g dμ");
    println!("    • Countable additivity: ∫_{{⋃Aₙ}} f dμ = ∑ ∫_{{Aₙ}} f dμ");
    println!("    • Absolute continuity: |∫f dμ| ≤ ∫|f| dμ\n");

    // ========================================================================
    // 2. Simple Functions
    // ========================================================================
    println!("─────────────────────────────────────────────────────────");
    println!("2. Simple Functions: Building Blocks of Integration");
    println!("─────────────────────────────────────────────────────────\n");

    println!("  A simple function has the form:");
    println!("    s(x) = ∑ᵢ aᵢ · χ_{{Aᵢ}}(x)");
    println!();
    println!("  where:");
    println!("    • aᵢ are constants (finitely many)");
    println!("    • χ_{{Aᵢ}} are characteristic (indicator) functions");
    println!("    • Aᵢ are measurable sets (disjoint partition)\n");

    println!("  Integration of simple functions:");
    println!("    ∫ s dμ = ∑ᵢ aᵢ · μ(Aᵢ)");
    println!();
    println!("  This is the definition from which all Lebesgue integration derives.\n");

    println!("  Example: Step function on [0,1]");
    println!("    s(x) = {{ 1 if x ∈ [0, 0.5)");
    println!("           {{ 2 if x ∈ [0.5, 1]");
    println!();
    println!("    ∫ s dλ = 1·λ([0,0.5)) + 2·λ([0.5,1])");
    println!("           = 1·0.5 + 2·0.5");
    println!("           = 1.5\n");

    // ========================================================================
    // 3. Measurable Functions
    // ========================================================================
    println!("─────────────────────────────────────────────────────────");
    println!("3. Measurable Functions");
    println!("─────────────────────────────────────────────────────────\n");

    println!("  A function f: X → ℝ is measurable if:");
    println!("    f⁻¹((a, ∞)) ∈ Σ  for all a ∈ ℝ");
    println!();
    println!("  Equivalently:");
    println!("    • Preimages of Borel sets are measurable");
    println!("    • Can be approximated by simple functions\n");

    println!("  Examples of measurable functions:");
    println!("    ✓ Continuous functions (always measurable)");
    println!("    ✓ Monotone functions");
    println!("    ✓ Limits of measurable functions");
    println!("    ✓ Characteristic functions of measurable sets");
    println!("    ✓ Compositions of measurable functions\n");

    println!("  Non-measurable functions:");
    println!("    ✗ Functions with non-measurable level sets");
    println!("    ✗ Vitali's non-measurable set indicator");
    println!("      (requires Axiom of Choice)\n");

    // ========================================================================
    // 4. Integration Process
    // ========================================================================
    println!("─────────────────────────────────────────────────────────");
    println!("4. The Integration Process");
    println!("─────────────────────────────────────────────────────────\n");

    println!("  Lebesgue integration proceeds in stages:\n");

    println!("  Stage 1: Simple functions");
    println!("    ∫ s dμ = ∑ᵢ aᵢ · μ(Aᵢ)");
    println!("    Direct definition, always finite or +∞\n");

    println!("  Stage 2: Non-negative functions");
    println!("    ∫ f dμ = sup {{ ∫ s dμ : s simple, s ≤ f }}");
    println!("    Supremum over simple function approximations\n");

    println!("  Stage 3: General functions");
    println!("    f = f⁺ - f⁻  where f⁺ = max(f,0), f⁻ = max(-f,0)");
    println!("    ∫ f dμ = ∫ f⁺ dμ - ∫ f⁻ dμ");
    println!("    Provided at least one integral is finite\n");

    println!("  Stage 4: Complex/multivector valued");
    println!("    ∫ f dμ = ∫ Re(f) dμ + i·∫ Im(f) dμ");
    println!("    Component-wise integration\n");

    // ========================================================================
    // 5. Integration Examples (Conceptual)
    // ========================================================================
    println!("─────────────────────────────────────────────────────────");
    println!("5. Integration Examples");
    println!("─────────────────────────────────────────────────────────\n");

    // Create Lebesgue measure on ℝ
    let lebesgue_1d = LebesgueMeasure::new(1);
    println!("  • Created 1D Lebesgue measure (length on ℝ)");
    println!("    Dimension: {}\n", lebesgue_1d.dimension());

    println!("  Example 1: Constant function");
    println!("    f(x) = c  for all x");
    println!("    ∫_[a,b] c dλ = c · λ([a,b]) = c(b-a)");
    println!("    e.g., ∫_[0,1] 5 dλ = 5·1 = 5\n");

    println!("  Example 2: Linear function");
    println!("    f(x) = x");
    println!("    ∫_[0,1] x dλ = 1/2  (same as Riemann integral)");
    println!("    ∫_[0,b] x dλ = b²/2\n");

    println!("  Example 3: Indicator function");
    println!("    f(x) = χ_A(x) = {{ 1 if x ∈ A");
    println!("                    {{ 0 if x ∉ A");
    println!("    ∫ χ_A dμ = μ(A)");
    println!("    e.g., ∫ χ_[0,0.5] dλ = λ([0,0.5]) = 0.5\n");

    println!("  Example 4: Dirichlet function (non-Riemann integrable!)");
    println!("    f(x) = {{ 1 if x ∈ ℚ (rational)");
    println!("           {{ 0 if x ∈ ℝ∖ℚ (irrational)");
    println!("    ∫ f dλ = λ(ℚ) = 0  (rationals have measure zero)");
    println!("    Not Riemann integrable, but Lebesgue integrable!\n");

    // Create 2D Lebesgue measure
    let lebesgue_2d = LebesgueMeasure::new(2);
    println!("  • Created 2D Lebesgue measure (area on ℝ²)");
    println!("    Dimension: {}\n", lebesgue_2d.dimension());

    println!("  Example 5: Integration over rectangles");
    println!("    f(x,y) = xy");
    println!("    ∫_[0,a]×[0,b] xy dλ² = (a²/2)(b²/2) = a²b²/4");
    println!("    e.g., ∫_[0,1]×[0,1] xy dλ² = 1/4\n");

    // ========================================================================
    // 6. Comparison: Riemann vs Lebesgue
    // ========================================================================
    println!("─────────────────────────────────────────────────────────");
    println!("6. Riemann vs Lebesgue Integration");
    println!("─────────────────────────────────────────────────────────\n");

    println!("┌───────────────┬─────────────────┬──────────────────────┐");
    println!("│ Property      │ Riemann         │ Lebesgue             │");
    println!("├───────────────┼─────────────────┼──────────────────────┤");
    println!("│ Partition     │ Domain (x-axis) │ Range (y-axis)       │");
    println!("│ Functions     │ Continuous      │ Measurable           │");
    println!("│ Discontinuity │ Limited         │ Arbitrary (a.e.)     │");
    println!("│ Null sets     │ No special role │ Fundamental          │");
    println!("│ Limit thms    │ Weak            │ Strong (MCT, DCT)    │");
    println!("│ Completeness  │ Not complete    │ Complete space       │");
    println!("└───────────────┴─────────────────┴──────────────────────┘\n");

    println!("  Key advantage of Lebesgue:");
    println!("    If fₙ → f and fₙ Riemann integrable, does ∫fₙ → ∫f?");
    println!("      • Riemann: Only under strong conditions");
    println!("      • Lebesgue: Yes (dominated convergence theorem)\n");

    // ========================================================================
    // 7. Integration Properties
    // ========================================================================
    println!("─────────────────────────────────────────────────────────");
    println!("7. Properties of Lebesgue Integration");
    println!("─────────────────────────────────────────────────────────\n");

    println!("  1. Linearity:");
    println!("     ∫(af + bg) dμ = a∫f dμ + b∫g dμ\n");

    println!("  2. Monotonicity:");
    println!("     f ≤ g a.e. ⟹ ∫f dμ ≤ ∫g dμ\n");

    println!("  3. Triangle inequality:");
    println!("     |∫f dμ| ≤ ∫|f| dμ\n");

    println!("  4. Dominated convergence:");
    println!("     If fₙ → f a.e. and |fₙ| ≤ g with ∫g dμ < ∞,");
    println!("     then ∫fₙ dμ → ∫f dμ\n");

    println!("  5. Monotone convergence:");
    println!("     If 0 ≤ f₁ ≤ f₂ ≤ ... and fₙ → f,");
    println!("     then ∫fₙ dμ → ∫f dμ\n");

    println!("  6. Fatou's lemma:");
    println!("     ∫ lim inf fₙ dμ ≤ lim inf ∫fₙ dμ\n");

    // ========================================================================
    // Summary
    // ========================================================================
    println!("┌──────────────────────────────────────────────────────────┐");
    println!("│ Key Takeaways                                            │");
    println!("├──────────────────────────────────────────────────────────┤");
    println!("│ • Lebesgue integral extends Riemann to more functions   │");
    println!("│ • Built from simple functions via approximation         │");
    println!("│ • Handles discontinuous functions (Dirichlet function)   │");
    println!("│ • Powerful limit theorems for convergence               │");
    println!("│ • Integration w.r.t. arbitrary measures (not just dx)   │");
    println!("│ • Foundation for probability theory, quantum mechanics   │");
    println!("│                                                          │");
    println!("│ Three stages:                                            │");
    println!("│   1. Simple functions: ∫s dμ = ∑ aᵢμ(Aᵢ)                │");
    println!("│   2. Non-negative: supremum over simple functions       │");
    println!("│   3. General: f = f⁺ - f⁻                               │");
    println!("│                                                          │");
    println!("│ Almost everywhere (a.e.):                                │");
    println!("│   • Property holds except on set of measure zero        │");
    println!("│   • f = g a.e. ⟹ ∫f dμ = ∫g dμ                          │");
    println!("└──────────────────────────────────────────────────────────┘\n");
}
