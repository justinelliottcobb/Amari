//! # Spectral Theory Example
//!
//! Demonstrates spectral analysis of operators.
//!
//! ## Mathematical Background
//!
//! The spectrum σ(T) of operator T contains all λ where (T - λI)⁻¹ doesn't exist:
//! - Point spectrum σ_p: eigenvalues (T - λI not injective)
//! - Continuous spectrum σ_c: range dense but not closed
//! - Residual spectrum σ_r: range not dense
//!
//! Spectral theorem: Self-adjoint operators have diagonal representation.
//!
//! Run with: `cargo run --bin spectral_theory`

use amari_functional::{
    spectral::{
        Spectrum, SpectralDecomposition, SpectralMeasure,
        PointSpectrum, ContinuousSpectrum, ResidualSpectrum,
    },
    operator::{LinearOperator, SelfAdjoint, CompactOperator},
    matrix::MatrixOperator,
    functional_calculus::FunctionalCalculus,
    hilbert::RealHilbert,
};
use nalgebra::{DMatrix, DVector};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("═══════════════════════════════════════════════════════════════");
    println!("                    SPECTRAL THEORY DEMO");
    println!("═══════════════════════════════════════════════════════════════\n");

    // =========================================================================
    // Part 1: Eigenvalue Problems
    // =========================================================================
    println!("Part 1: Eigenvalue Problems");
    println!("───────────────────────────\n");

    println!("Eigenvalue equation: Tx = λx");
    println!("  - λ is eigenvalue");
    println!("  - x ≠ 0 is eigenvector");
    println!("  - Eigenspace: all eigenvectors for λ, plus 0");

    let matrix = DMatrix::from_row_slice(3, 3, &[
        4.0, 1.0, 1.0,
        1.0, 4.0, 1.0,
        1.0, 1.0, 4.0,
    ]);

    let op = MatrixOperator::new(matrix.clone())?;

    println!("\nSymmetric matrix A:");
    println!("  [4  1  1]");
    println!("  [1  4  1]");
    println!("  [1  1  4]");

    let (eigenvalues, eigenvectors) = op.eigen_decomposition()?;

    println!("\nEigenvalues and eigenvectors:");
    for i in 0..eigenvalues.len() {
        let ev = &eigenvectors[i];
        println!("  λ_{} = {:.4}", i, eigenvalues[i]);
        println!("    v_{} = ({:.4}, {:.4}, {:.4})", i, ev[0], ev[1], ev[2]);
    }

    // Verify Av = λv
    println!("\nVerification Av = λv:");
    for i in 0..eigenvalues.len() {
        let v = &eigenvectors[i];
        let av = op.apply(v)?;
        let lambda_v: Vec<f64> = v.iter().map(|x| eigenvalues[i] * x).collect();
        let error: f64 = av.iter().zip(&lambda_v).map(|(a, b)| (a - b).powi(2)).sum::<f64>().sqrt();
        println!("  ‖Av_{} - λ_{}v_{}‖ = {:.2e}", i, i, i, error);
    }

    // =========================================================================
    // Part 2: Spectral Properties of Self-Adjoint Operators
    // =========================================================================
    println!("\n\nPart 2: Self-Adjoint Operator Spectrum");
    println!("───────────────────────────────────────\n");

    println!("For self-adjoint operators T = T*:");
    println!("  1. All eigenvalues are real");
    println!("  2. Eigenvectors for distinct eigenvalues are orthogonal");
    println!("  3. No residual spectrum");
    println!("  4. Spectrum ⊂ [m, M] where m = inf⟨Tx,x⟩, M = sup⟨Tx,x⟩");

    // Verify orthogonality
    let hilbert = RealHilbert::<3>::new();
    println!("\nOrthogonality of eigenvectors:");
    for i in 0..3 {
        for j in (i+1)..3 {
            let inner = hilbert.inner_product(&eigenvectors[i], &eigenvectors[j])?;
            println!("  ⟨v_{}, v_{}⟩ = {:.6}", i, j, inner);
        }
    }

    // Rayleigh quotient
    println!("\nRayleigh quotient R(x) = ⟨Ax,x⟩/⟨x,x⟩:");
    println!("  min R(x) = λ_min, max R(x) = λ_max");

    let test_vectors = vec![
        DVector::from_vec(vec![1.0, 0.0, 0.0]),
        DVector::from_vec(vec![1.0, 1.0, 1.0]),
        DVector::from_vec(vec![1.0, -1.0, 0.0]),
    ];

    for v in &test_vectors {
        let av = op.apply(v)?;
        let numerator = hilbert.inner_product(&av, v)?;
        let denominator = hilbert.inner_product(v, v)?;
        let rayleigh = numerator / denominator;
        println!("  R({:?}) = {:.4}", v.as_slice(), rayleigh);
    }

    // =========================================================================
    // Part 3: Spectral Theorem (Finite-Dimensional)
    // =========================================================================
    println!("\n\nPart 3: Spectral Theorem");
    println!("────────────────────────\n");

    println!("Spectral theorem for self-adjoint operators:");
    println!("  A = Σ λᵢ Pᵢ  (spectral decomposition)");
    println!("  where Pᵢ is projection onto eigenspace of λᵢ");

    let decomp = SpectralDecomposition::compute(&op)?;

    println!("\nSpectral decomposition of A:");
    println!("  A = λ₀P₀ + λ₁P₁ + λ₂P₂");
    println!("  where:");
    for (i, (lambda, proj)) in decomp.components().iter().enumerate() {
        println!("    λ_{} = {:.4}, P_{} = v_{}v_{}ᵀ", i, lambda, i, i, i);
    }

    // Reconstruct matrix
    let reconstructed = decomp.reconstruct()?;
    let recon_error = (&matrix - &reconstructed).norm();
    println!("\n  Reconstruction error ‖A - ΣλᵢPᵢ‖ = {:.2e}", recon_error);

    // =========================================================================
    // Part 4: Functional Calculus
    // =========================================================================
    println!("\n\nPart 4: Functional Calculus");
    println!("───────────────────────────\n");

    println!("For self-adjoint A = Σλᵢ Pᵢ, we can define f(A):");
    println!("  f(A) = Σ f(λᵢ) Pᵢ");

    let fc = FunctionalCalculus::new(&op)?;

    // Square root
    println!("\nSquare root: √A");
    let sqrt_a = fc.apply(|x| x.sqrt())?;
    let sqrt_eigs = sqrt_a.eigenvalues()?;
    println!("  Eigenvalues of √A:");
    for (i, ev) in sqrt_eigs.iter().enumerate() {
        println!("    {:.4} (expected: √{:.4} = {:.4})", ev, eigenvalues[i], eigenvalues[i].sqrt());
    }

    // Verify (√A)² = A
    let sqrt_sq = sqrt_a.compose(&sqrt_a)?;
    let test = DVector::from_vec(vec![1.0, 2.0, 3.0]);
    let a_test = op.apply(&test)?;
    let sqrt_sq_test = sqrt_sq.apply(&test)?;
    let error: f64 = a_test.iter().zip(&sqrt_sq_test).map(|(a, b)| (a - b).powi(2)).sum::<f64>().sqrt();
    println!("\n  Verification: ‖(√A)²x - Ax‖ = {:.2e}", error);

    // Exponential
    println!("\nMatrix exponential: exp(A)");
    let exp_a = fc.apply(|x| x.exp())?;
    let exp_eigs = exp_a.eigenvalues()?;
    println!("  Eigenvalues of exp(A):");
    for (i, ev) in exp_eigs.iter().enumerate() {
        println!("    {:.4} (expected: exp({:.4}) = {:.4})", ev, eigenvalues[i], eigenvalues[i].exp());
    }

    // Inverse
    println!("\nInverse: A⁻¹");
    let inv_a = fc.apply(|x| 1.0 / x)?;
    let inv_eigs = inv_a.eigenvalues()?;
    println!("  Eigenvalues of A⁻¹:");
    for (i, ev) in inv_eigs.iter().enumerate() {
        println!("    {:.4} (expected: 1/{:.4} = {:.4})", ev, eigenvalues[i], 1.0 / eigenvalues[i]);
    }

    // =========================================================================
    // Part 5: Compact Operator Spectrum
    // =========================================================================
    println!("\n\nPart 5: Compact Operator Spectrum");
    println!("──────────────────────────────────\n");

    println!("For compact operators on infinite-dimensional spaces:");
    println!("  - 0 is always in spectrum (may or may not be eigenvalue)");
    println!("  - Non-zero spectrum consists only of eigenvalues");
    println!("  - Each non-zero eigenvalue has finite multiplicity");
    println!("  - Eigenvalues can only accumulate at 0");

    // Simulate compact operator (diagonal with eigenvalues → 0)
    let n = 10;
    let mut diag_entries = vec![0.0; n * n];
    for i in 0..n {
        diag_entries[i * n + i] = 1.0 / (i + 1) as f64;  // 1, 1/2, 1/3, ...
    }
    let compact_matrix = DMatrix::from_vec(n, n, diag_entries);
    let compact_op = MatrixOperator::new(compact_matrix)?;

    println!("\nDiagonal operator with eigenvalues 1, 1/2, 1/3, ..., 1/10:");
    let compact_eigs = compact_op.eigenvalues()?;
    println!("  Eigenvalues: {:?}", &compact_eigs[..5]);
    println!("  ...");
    println!("  (accumulating at 0)");

    // =========================================================================
    // Part 6: Spectrum of Non-Self-Adjoint Operators
    // =========================================================================
    println!("\n\nPart 6: Non-Self-Adjoint Operators");
    println!("───────────────────────────────────\n");

    println!("Non-self-adjoint operators can have:");
    println!("  - Complex eigenvalues");
    println!("  - Non-orthogonal eigenvectors");
    println!("  - Residual spectrum");

    // Rotation + scaling (not self-adjoint)
    let non_sa = DMatrix::from_row_slice(2, 2, &[
        0.0, -1.0,
        1.0,  0.0,
    ]);
    let non_sa_op = MatrixOperator::new(non_sa)?;

    println!("\n90° rotation matrix:");
    println!("  [0  -1]");
    println!("  [1   0]");

    let complex_eigs = non_sa_op.complex_eigenvalues()?;
    println!("\n  Complex eigenvalues:");
    for (i, (re, im)) in complex_eigs.iter().enumerate() {
        println!("    λ_{} = {} + {}i", i, re, im);
    }
    println!("  (Pure imaginary: rotation has no real eigenvalues)");

    // =========================================================================
    // Part 7: Spectral Radius
    // =========================================================================
    println!("\n\nPart 7: Spectral Radius");
    println!("────────────────────────\n");

    println!("Spectral radius: ρ(T) = max |λ| over λ ∈ σ(T)");
    println!("Key properties:");
    println!("  - ρ(T) ≤ ‖T‖ for any operator norm");
    println!("  - For self-adjoint: ρ(T) = ‖T‖");
    println!("  - lim ‖Tⁿ‖^(1/n) = ρ(T) (Gelfand formula)");

    let spectral_radius = op.spectral_radius()?;
    let op_norm = op.norm()?;

    println!("\nFor our symmetric matrix A:");
    println!("  ρ(A) = {:.6}", spectral_radius);
    println!("  ‖A‖ = {:.6}", op_norm);
    println!("  ρ(A) = ‖A‖? {} (expected for self-adjoint)",
             (spectral_radius - op_norm).abs() < 1e-10);

    // Gelfand formula demonstration
    println!("\nGelfand formula: ρ(A) = lim ‖Aⁿ‖^(1/n)");
    let mut an = op.clone();
    for n in 1..=5 {
        let an_norm = an.norm()?;
        let estimate = an_norm.powf(1.0 / n as f64);
        println!("  n={}: ‖A^{}‖^(1/{}) = {:.6}", n, n, n, estimate);
        if n < 5 {
            an = an.compose(&op)?;
        }
    }

    // =========================================================================
    // Part 8: Perturbation Theory
    // =========================================================================
    println!("\n\nPart 8: Perturbation Theory");
    println!("───────────────────────────\n");

    println!("How do eigenvalues change under perturbation?");
    println!("  A(ε) = A + εB");

    let perturbation = DMatrix::from_row_slice(3, 3, &[
        0.1, 0.0, 0.0,
        0.0, 0.1, 0.0,
        0.0, 0.0, 0.1,
    ]);

    println!("\nSmall diagonal perturbation B = 0.1·I");

    let epsilons = [0.0, 0.1, 0.2, 0.5];
    println!("\n  ε    | λ₀      | λ₁      | λ₂");
    println!("  ─────┼─────────┼─────────┼─────────");

    for eps in epsilons {
        let perturbed = &matrix + eps * &perturbation;
        let perturbed_op = MatrixOperator::new(perturbed)?;
        let perturbed_eigs = perturbed_op.eigenvalues()?;
        println!("  {:.1}  | {:.5} | {:.5} | {:.5}",
                 eps, perturbed_eigs[0], perturbed_eigs[1], perturbed_eigs[2]);
    }

    println!("\n  First-order perturbation: λ(ε) ≈ λ(0) + ε⟨Bv, v⟩");
    println!("  (Valid when eigenvalues are well-separated)");

    // =========================================================================
    // Part 9: Spectral Gap
    // =========================================================================
    println!("\n\nPart 9: Spectral Gap");
    println!("────────────────────\n");

    println!("Spectral gap = difference between largest and second-largest eigenvalues");
    println!("  (or smallest non-zero and zero for stochastic matrices)");

    let sorted_eigs = {
        let mut e = eigenvalues.clone();
        e.sort_by(|a, b| b.partial_cmp(a).unwrap());
        e
    };

    let gap = sorted_eigs[0] - sorted_eigs[1];
    println!("\nFor our matrix:");
    println!("  Sorted eigenvalues: {:.4}, {:.4}, {:.4}",
             sorted_eigs[0], sorted_eigs[1], sorted_eigs[2]);
    println!("  Spectral gap: {:.4}", gap);

    println!("\nImportance:");
    println!("  - Large gap → fast convergence of power iteration");
    println!("  - Large gap → stability under perturbation");
    println!("  - In Markov chains: gap determines mixing time");

    println!("\n═══════════════════════════════════════════════════════════════");
    println!("                    DEMO COMPLETE");
    println!("═══════════════════════════════════════════════════════════════");

    Ok(())
}
