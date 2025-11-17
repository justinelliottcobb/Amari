//! Geometric measure examples
//!
//! This example demonstrates multivector-valued measures on Clifford algebras.
//! Geometric measures extend classical real-valued measures to assign multivectors
//! (elements of Cl(p,q,r)) to measurable sets.
//!
//! Run with:
//! ```bash
//! cargo run --example geometric_measures
//! ```

use amari_measure::{geometric_lebesgue_measure, GeometricDensity, GeometricMeasure};
use amari_measure::{BorelSigma, LebesgueSigma};
use amari_measure::{Complete, Incomplete, SigmaFinite, Unsigned};

fn main() {
    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!("║      amari-measure: Geometric Measure Examples              ║");
    println!("╚═══════════════════════════════════════════════════════════════╝\n");

    // ========================================================================
    // 1. What is a Geometric Measure?
    // ========================================================================
    println!("─────────────────────────────────────────────────────────");
    println!("1. Geometric Measures: Multivector-Valued Measures");
    println!("─────────────────────────────────────────────────────────\n");

    println!("  A geometric measure μ: Σ → Cl(p,q,r) assigns multivectors");
    println!("  to measurable sets instead of real numbers:");
    println!();
    println!("    μ(A) = μ₀(A) + μ₁(A)e₁ + μ₂(A)e₂ + ... + μ₁₂(A)e₁e₂ + ...");
    println!();
    println!("  where each coefficient μᵢ(A) is a real-valued measure.\n");

    println!("  Grade decomposition:");
    println!("    μ = μ⟨0⟩ + μ⟨1⟩ + μ⟨2⟩ + ... + μ⟨n⟩");
    println!();
    println!("  where μ⟨k⟩ measures the grade-k component.\n");

    println!("  Properties:");
    println!("    ✓ Countably additive: μ(⋃ Aₙ) = ∑ μ(Aₙ) (as multivector sum)");
    println!("    ✓ Grade structure: 2ⁿ components for n-dimensional space");
    println!("    ✓ Extends classical measures: μ⟨0⟩ is a real measure");
    println!("    ✓ Geometric interpretation: Each grade has physical meaning\n");

    // ========================================================================
    // 2. Creating Geometric Measures
    // ========================================================================
    println!("─────────────────────────────────────────────────────────");
    println!("2. Creating Geometric Measures");
    println!("─────────────────────────────────────────────────────────\n");

    // 2D geometric measure (plane)
    println!("  • 2D Geometric Measure (Plane):");
    let mu_2d = geometric_lebesgue_measure(2);
    println!("    Dimension: {}", mu_2d.dimension());
    println!("    Grades: {} (0, 1, 2)", mu_2d.num_grades());
    println!(
        "    Components: {} (scalar, e₁, e₂, e₁e₂)",
        mu_2d.num_components()
    );
    println!();
    println!("    Physical interpretation:");
    println!("      • Grade 0 (scalar): Area measure");
    println!("      • Grade 1 (vectors): Linear density/flux");
    println!("      • Grade 2 (bivector): Oriented area element\n");

    // 3D geometric measure (space)
    println!("  • 3D Geometric Measure (Space):");
    let mu_3d = geometric_lebesgue_measure(3);
    println!("    Dimension: {}", mu_3d.dimension());
    println!("    Grades: {} (0, 1, 2, 3)", mu_3d.num_grades());
    println!("    Components: {} (1 + 3 + 3 + 1)", mu_3d.num_components());
    println!();
    println!("    Physical interpretation:");
    println!("      • Grade 0: Volume measure");
    println!("      • Grade 1: Vector density");
    println!("      • Grade 2: Bivector (oriented area)");
    println!("      • Grade 3: Trivector (oriented volume/pseudoscalar)\n");

    // Higher dimensional example
    println!("  • 4D Geometric Measure (Spacetime):");
    let mu_4d = geometric_lebesgue_measure(4);
    println!("    Dimension: {}", mu_4d.dimension());
    println!("    Grades: {} (0, 1, 2, 3, 4)", mu_4d.num_grades());
    println!(
        "    Components: {} (1 + 4 + 6 + 4 + 1)",
        mu_4d.num_components()
    );
    println!("    Formula: 2^n = 2^4 = 16 components\n");

    // ========================================================================
    // 3. Geometric Measure with Different σ-Algebras
    // ========================================================================
    println!("─────────────────────────────────────────────────────────");
    println!("3. Geometric Measures with Different σ-Algebras");
    println!("─────────────────────────────────────────────────────────\n");

    // Borel geometric measure (incomplete)
    let borel_sigma = BorelSigma::new(2);
    let _mu_borel: GeometricMeasure<_, SigmaFinite, Unsigned, Incomplete> =
        GeometricMeasure::new(borel_sigma, 2);
    println!("  • Borel Geometric Measure:");
    println!("    σ-algebra: Borel (smallest containing open sets)");
    println!("    Completeness: Incomplete (not all null subsets)");
    println!("    Use case: Continuous functions, theoretical analysis\n");

    // Lebesgue geometric measure (complete)
    let lebesgue_sigma = LebesgueSigma::new(2);
    let _mu_lebesgue: GeometricMeasure<_, SigmaFinite, Unsigned, Complete> =
        GeometricMeasure::new(lebesgue_sigma, 2);
    println!("  • Lebesgue Geometric Measure:");
    println!("    σ-algebra: Lebesgue (completion of Borel)");
    println!("    Completeness: Complete (all null subsets measurable)");
    println!("    Use case: Integration, practical computations\n");

    println!("  Relationship:");
    println!("    Lebesgue σ-algebra = Borel σ-algebra + null set completions");
    println!("    Every Lebesgue measurable function is a.e. equal to a Borel function\n");

    // ========================================================================
    // 4. Geometric Densities
    // ========================================================================
    println!("─────────────────────────────────────────────────────────");
    println!("4. Geometric Densities: ρ: ℝⁿ → Cl(p,q,r)");
    println!("─────────────────────────────────────────────────────────\n");

    println!("  A geometric density defines a measure via integration:");
    println!("    μ(A) = ∫_A ρ(x) dλ(x)");
    println!();
    println!("  where ρ(x) is a multivector-valued function.\n");

    // Create geometric density
    let density_2d = GeometricDensity::new(2);
    println!("  • 2D Geometric Density:");
    println!("    Domain dimension: {}", density_2d.dimension());
    println!(
        "    Reference measure: Lebesgue (dimension {})",
        density_2d.reference_measure().dimension()
    );
    println!();
    println!("    Each component ρᵢ(x) is a real-valued density");
    println!("    Integration performed component-wise\n");

    let density_3d = GeometricDensity::new(3);
    println!("  • 3D Geometric Density:");
    println!("    Domain dimension: {}", density_3d.dimension());
    println!(
        "    Reference measure: Lebesgue (dimension {})",
        density_3d.reference_measure().dimension()
    );
    println!();
    println!("    Example applications:");
    println!("      • Electromagnetic field distributions");
    println!("      • Quantum mechanical probability currents");
    println!("      • Fluid flow with vorticity\n");

    // ========================================================================
    // 5. Component Structure
    // ========================================================================
    println!("─────────────────────────────────────────────────────────");
    println!("5. Component Structure by Dimension");
    println!("─────────────────────────────────────────────────────────\n");

    println!("┌───────┬────────┬────────────────────────────────────────┐");
    println!("│  Dim  │ Grades │ Components (by grade)                  │");
    println!("├───────┼────────┼────────────────────────────────────────┤");

    for dim in 0..=4 {
        let mu = geometric_lebesgue_measure(dim);
        let components_str = match dim {
            0 => "1".to_string(),
            1 => "1 + 1".to_string(),
            2 => "1 + 2 + 1".to_string(),
            3 => "1 + 3 + 3 + 1".to_string(),
            4 => "1 + 4 + 6 + 4 + 1".to_string(),
            _ => format!("2^{}", dim),
        };
        println!(
            "│   {}   │   {}    │ {} = {} total",
            dim,
            mu.num_grades(),
            components_str,
            mu.num_components()
        );
    }
    println!("└───────┴────────┴────────────────────────────────────────┘");
    println!();
    println!("  Formula: For dimension n, total components = 2^n");
    println!("  Binomial expansion: (1+1)^n = ∑ C(n,k) from k=0 to n\n");

    // ========================================================================
    // 6. Applications
    // ========================================================================
    println!("─────────────────────────────────────────────────────────");
    println!("6. Applications of Geometric Measures");
    println!("─────────────────────────────────────────────────────────\n");

    println!("  Physics:");
    println!("    • Electromagnetic field theory (F = E + I*B)");
    println!("    • Quantum mechanics (spinor fields, probability currents)");
    println!("    • General relativity (stress-energy tensor)");
    println!("    • Fluid dynamics (velocity + vorticity fields)\n");

    println!("  Engineering:");
    println!("    • Computer graphics (oriented primitives)");
    println!("    • Robotics (rigid body transformations)");
    println!("    • Signal processing (multidimensional signals)");
    println!("    • Computer vision (image moments, shape descriptors)\n");

    println!("  Mathematics:");
    println!("    • Differential geometry (differential forms as measures)");
    println!("    • Lie theory (invariant measures on Lie groups)");
    println!("    • Harmonic analysis (Clifford-valued functions)");
    println!("    • Stochastic geometry (random sets with orientation)\n");

    // ========================================================================
    // Summary
    // ========================================================================
    println!("┌──────────────────────────────────────────────────────────┐");
    println!("│ Key Takeaways                                            │");
    println!("├──────────────────────────────────────────────────────────┤");
    println!("│ • Geometric measures assign multivectors to sets         │");
    println!("│ • Each grade component is a separate real measure        │");
    println!("│ • 2^n total components for n-dimensional space           │");
    println!("│ • Extends classical measure theory to geometric algebra  │");
    println!("│ • Physical interpretation varies by grade:               │");
    println!("│   - Grade 0: Scalar density/mass                         │");
    println!("│   - Grade 1: Vector field/flux                           │");
    println!("│   - Grade 2: Bivector/circulation                        │");
    println!("│   - Grade n: Pseudoscalar/oriented volume                │");
    println!("│                                                          │");
    println!("│ Next steps:                                              │");
    println!("│ • Integration of multivector-valued functions            │");
    println!("│ • Radon-Nikodym derivatives for geometric densities      │");
    println!("│ • Product measures on product algebras                   │");
    println!("└──────────────────────────────────────────────────────────┘\n");
}
