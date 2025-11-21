//! Basic measure examples
//!
//! This example demonstrates the fundamental measure types in amari-measure:
//! - Lebesgue measure (volume in ℝⁿ)
//! - Counting measure (cardinality on discrete sets)
//! - Dirac measure (point masses)
//! - Probability measure (normalized measures)
//!
//! Run with:
//! ```bash
//! cargo run --example basic_measures
//! ```

use amari_measure::{CountingMeasure, DiracMeasure, LebesgueMeasure, ProbabilityMeasure};

fn main() {
    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!("║         amari-measure: Basic Measure Examples                ║");
    println!("╚═══════════════════════════════════════════════════════════════╝\n");

    // ========================================================================
    // 1. Lebesgue Measure - Volume in ℝⁿ
    // ========================================================================
    println!("─────────────────────────────────────────────────────────");
    println!("1. Lebesgue Measure (Volume in ℝⁿ)");
    println!("─────────────────────────────────────────────────────────\n");

    // 1-dimensional Lebesgue measure (length on the real line)
    let lebesgue_1d = LebesgueMeasure::new(1);
    println!("  • Created 1D Lebesgue measure (length)");
    println!("    Dimension: {}", lebesgue_1d.dimension());
    println!("    Measures intervals [a,b] with length b-a\n");

    // 2-dimensional Lebesgue measure (area in the plane)
    let lebesgue_2d = LebesgueMeasure::new(2);
    println!("  • Created 2D Lebesgue measure (area)");
    println!("    Dimension: {}", lebesgue_2d.dimension());
    println!("    Measures regions in ℝ² with area\n");

    // 3-dimensional Lebesgue measure (volume in space)
    let lebesgue_3d = LebesgueMeasure::new(3);
    println!("  • Created 3D Lebesgue measure (volume)");
    println!("    Dimension: {}", lebesgue_3d.dimension());
    println!("    Measures regions in ℝ³ with volume\n");

    println!("  Properties of Lebesgue measure:");
    println!("    ✓ Translation invariant: μ(A + x) = μ(A)");
    println!("    ✓ σ-finite: ℝⁿ = ⋃ Bₙ(0) with μ(Bₙ(0)) < ∞");
    println!("    ✓ Complete: Subsets of null sets are measurable");
    println!("    ✓ Regular: Can approximate measurable sets with open/closed sets\n");

    // ========================================================================
    // 2. Counting Measure - Cardinality on Discrete Sets
    // ========================================================================
    println!("─────────────────────────────────────────────────────────");
    println!("2. Counting Measure (Cardinality)");
    println!("─────────────────────────────────────────────────────────\n");

    // Counting measure assigns the cardinality to each set
    let _counting = CountingMeasure::new();
    println!("  • Created counting measure");
    println!("    μ(A) = |A| (number of elements in A)\n");

    println!("  Properties of counting measure:");
    println!("    ✓ Defined on any set (uses power set σ-algebra)");
    println!("    ✓ Discrete: Only assigns integer values");
    println!("    ✓ Finite on finite sets, infinite on infinite sets");
    println!("    ✓ Makes every function from a discrete space measurable\n");

    println!("  Example applications:");
    println!("    • Discrete probability (uniform distribution on finite sets)");
    println!("    • Combinatorics (counting configurations)");
    println!("    • Graph theory (vertex/edge counting)");
    println!("    • Integer lattices (lattice point problems)\n");

    // ========================================================================
    // 3. Dirac Measure - Point Masses
    // ========================================================================
    println!("─────────────────────────────────────────────────────────");
    println!("3. Dirac Measure (Point Masses)");
    println!("─────────────────────────────────────────────────────────\n");

    // Dirac measure - conceptual example
    let _dirac_origin = DiracMeasure::new(1);
    println!("  • Dirac measure δ₀ at origin");
    println!("    δ₀(A) = 1 if 0 ∈ A, else 0");
    println!("    Measures on 1D space (ℝ)\n");

    // Dirac measure in 3D space
    let _dirac_3d = DiracMeasure::new(3);
    println!("  • Dirac measure in ℝ³");
    println!("    Concentrates all mass at a single point");
    println!("    Measures on 3D space\n");

    println!("  Properties of Dirac measure:");
    println!("    ✓ Atomic: All mass concentrated at one point");
    println!("    ✓ Probability measure: δₓ(ℝⁿ) = 1");
    println!("    ✓ Singular w.r.t. Lebesgue: No density dδₓ/dλ");
    println!("    ✓ Models deterministic outcomes (zero variance)\n");

    println!("  Example applications:");
    println!("    • Deterministic probability (sure events)");
    println!("    • Initial conditions in dynamical systems");
    println!("    • Point sources in physics (charges, masses)");
    println!("    • Discrete approximations of continuous distributions\n");

    // ========================================================================
    // 4. Probability Measure - Normalized Measures
    // ========================================================================
    println!("─────────────────────────────────────────────────────────");
    println!("4. Probability Measure (ℙ(Ω) = 1)");
    println!("─────────────────────────────────────────────────────────\n");

    // Create a probability measure on a 2D space - conceptual example
    let _prob = ProbabilityMeasure::<f64>::new(1.0);
    println!("  • Probability measure on ℝ²");
    println!("    ℙ(Ω) = 1 (total probability is 1)");
    println!("    Measures on 2D space\n");

    println!("  Properties of probability measures:");
    println!("    ✓ Non-negative: ℙ(A) ≥ 0 for all events A");
    println!("    ✓ Normalized: ℙ(Ω) = 1");
    println!("    ✓ Countably additive: ℙ(⋃ Aₙ) = ∑ ℙ(Aₙ) for disjoint Aₙ");
    println!("    ✓ Monotone: A ⊆ B ⟹ ℙ(A) ≤ ℙ(B)\n");

    println!("  Example applications:");
    println!("    • Probability theory and statistics");
    println!("    • Random variables and expectations");
    println!("    • Stochastic processes");
    println!("    • Machine learning (probabilistic models)\n");

    // ========================================================================
    // Comparison Summary
    // ========================================================================
    println!("─────────────────────────────────────────────────────────");
    println!("Measure Comparison Summary");
    println!("─────────────────────────────────────────────────────────\n");

    println!("┌─────────────┬────────────┬────────────┬─────────────────┐");
    println!("│ Measure     │ Finiteness │ Sign       │ Completeness    │");
    println!("├─────────────┼────────────┼────────────┼─────────────────┤");
    println!("│ Lebesgue    │ σ-finite   │ Unsigned   │ Complete        │");
    println!("│ Counting    │ Infinite*  │ Unsigned   │ Complete        │");
    println!("│ Dirac       │ Finite     │ Unsigned   │ Complete        │");
    println!("│ Probability │ Finite     │ Unsigned   │ (varies)        │");
    println!("└─────────────┴────────────┴────────────┴─────────────────┘");
    println!("  * Finite on finite sets, infinite on infinite sets\n");

    println!("┌──────────────────────────────────────────────────────────┐");
    println!("│ Key Takeaways                                            │");
    println!("├──────────────────────────────────────────────────────────┤");
    println!("│ • Lebesgue: Natural notion of volume/area/length        │");
    println!("│ • Counting: Discrete measure for finite sets            │");
    println!("│ • Dirac: Atomic measure at single points                │");
    println!("│ • Probability: Normalized for probability theory        │");
    println!("│                                                          │");
    println!("│ All measures satisfy:                                    │");
    println!("│   1. μ(∅) = 0                                            │");
    println!("│   2. μ(⋃ Aₙ) = ∑ μ(Aₙ) for disjoint Aₙ                 │");
    println!("└──────────────────────────────────────────────────────────┘\n");
}
