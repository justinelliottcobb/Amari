//! # Operators Example
//!
//! Demonstrates bounded linear operators on Hilbert and Banach spaces.
//!
//! ## Mathematical Background
//!
//! A bounded linear operator T: H₁ → H₂ satisfies:
//! - Linearity: T(αx + βy) = αT(x) + βT(y)
//! - Boundedness: ‖Tx‖ ≤ M‖x‖ for some M ≥ 0
//!
//! The operator norm is ‖T‖ = sup{‖Tx‖ : ‖x‖ = 1}
//!
//! Run with: `cargo run --bin operators`

use amari_functional::{
    operator::{
        LinearOperator, BoundedOperator, CompactOperator,
        AdjointOperator, SelfAdjoint, UnitaryOperator,
    },
    matrix::{MatrixOperator, TridiagonalOperator},
    integral::{IntegralOperator, KernelFunction},
    differential::{DifferentialOperator, LaplacianOperator},
    hilbert::{RealHilbert, L2Space},
};
use nalgebra::{DMatrix, DVector};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("═══════════════════════════════════════════════════════════════");
    println!("                    OPERATORS DEMO");
    println!("═══════════════════════════════════════════════════════════════\n");

    // =========================================================================
    // Part 1: Matrix Operators
    // =========================================================================
    println!("Part 1: Matrix Operators");
    println!("────────────────────────\n");

    // Simple 3x3 matrix as operator on R³
    let matrix = DMatrix::from_row_slice(3, 3, &[
        2.0, 1.0, 0.0,
        1.0, 3.0, 1.0,
        0.0, 1.0, 2.0,
    ]);

    let op = MatrixOperator::new(matrix.clone())?;

    println!("Matrix A:");
    println!("  [2  1  0]");
    println!("  [1  3  1]");
    println!("  [0  1  2]");

    // Apply operator
    let x = DVector::from_vec(vec![1.0, 2.0, 3.0]);
    let ax = op.apply(&x)?;

    println!("\nApplying A to x = (1, 2, 3):");
    println!("  Ax = ({}, {}, {})", ax[0], ax[1], ax[2]);

    // Operator norm
    let op_norm = op.norm()?;
    println!("\nOperator norm ‖A‖ = {:.6}", op_norm);
    println!("  (Largest singular value)");

    // Verify boundedness
    let x_norm = x.norm();
    let ax_norm = ax.norm();
    println!("\n  ‖x‖ = {:.6}", x_norm);
    println!("  ‖Ax‖ = {:.6}", ax_norm);
    println!("  ‖Ax‖/‖x‖ = {:.6} ≤ ‖A‖ = {:.6}  ✓", ax_norm/x_norm, op_norm);

    // =========================================================================
    // Part 2: Adjoint Operators
    // =========================================================================
    println!("\n\nPart 2: Adjoint Operators");
    println!("─────────────────────────\n");

    println!("The adjoint T* satisfies: ⟨Tx, y⟩ = ⟨x, T*y⟩");
    println!("For matrices: A* = Aᵀ (real) or A* = A† (complex)");

    let adjoint = op.adjoint()?;

    // For this symmetric matrix, A = A*
    let y = DVector::from_vec(vec![2.0, 1.0, 1.0]);
    let a_star_y = adjoint.apply(&y)?;

    println!("\nFor our symmetric matrix A = Aᵀ:");
    println!("  A*y = ({}, {}, {})", a_star_y[0], a_star_y[1], a_star_y[2]);

    // Verify adjoint property
    let hilbert = RealHilbert::<3>::new();
    let lhs = hilbert.inner_product(&ax, &y)?;
    let rhs = hilbert.inner_product(&x, &a_star_y)?;

    println!("\n  ⟨Ax, y⟩ = {:.6}", lhs);
    println!("  ⟨x, A*y⟩ = {:.6}", rhs);
    println!("  Equal? {}  ✓", (lhs - rhs).abs() < 1e-10);

    // Self-adjoint operators
    println!("\nSelf-adjoint operators (A = A*):");
    println!("  - Real symmetric matrices");
    println!("  - Have real eigenvalues");
    println!("  - Eigenvectors are orthogonal");

    let is_self_adjoint = op.is_self_adjoint()?;
    println!("\n  Is our matrix self-adjoint? {}", is_self_adjoint);

    // =========================================================================
    // Part 3: Unitary and Orthogonal Operators
    // =========================================================================
    println!("\n\nPart 3: Unitary Operators");
    println!("─────────────────────────\n");

    println!("Unitary operator U: U*U = UU* = I");
    println!("  - Preserves inner products: ⟨Ux, Uy⟩ = ⟨x, y⟩");
    println!("  - Preserves norms: ‖Ux‖ = ‖x‖");

    // Rotation matrix (2D)
    let theta = std::f64::consts::PI / 4.0;  // 45°
    let rotation = DMatrix::from_row_slice(2, 2, &[
        theta.cos(), -theta.sin(),
        theta.sin(), theta.cos(),
    ]);

    let rot_op = MatrixOperator::new(rotation)?;

    println!("\nRotation by 45°:");
    println!("  R = [cos θ  -sin θ]");
    println!("      [sin θ   cos θ]");

    let v = DVector::from_vec(vec![1.0, 0.0]);
    let rv = rot_op.apply(&v)?;

    println!("\n  R(1, 0) = ({:.4}, {:.4})", rv[0], rv[1]);
    println!("  ‖v‖ = {:.4}, ‖Rv‖ = {:.4} (preserved)", v.norm(), rv.norm());

    let is_unitary = rot_op.is_unitary()?;
    println!("\n  Is rotation unitary? {}", is_unitary);

    // =========================================================================
    // Part 4: Compact Operators
    // =========================================================================
    println!("\n\nPart 4: Compact Operators");
    println!("─────────────────────────\n");

    println!("A compact operator maps bounded sets to precompact sets.");
    println!("Key properties:");
    println!("  - Every compact operator on infinite-dim space has 0 in spectrum");
    println!("  - Non-zero eigenvalues have finite multiplicity");
    println!("  - Non-zero eigenvalues can only accumulate at 0");

    // Integral operator with continuous kernel is compact
    println!("\nExample: Volterra integral operator");
    println!("  (Tf)(x) = ∫₀ˣ f(t) dt");

    let kernel = |x: f64, t: f64| if t <= x { 1.0 } else { 0.0 };
    let volterra = IntegralOperator::new(0.0, 1.0, kernel)?;

    let f: Vec<f64> = (0..100)
        .map(|i| i as f64 / 99.0)  // f(x) = x
        .collect();

    let tf = volterra.apply(&f)?;

    println!("\n  For f(x) = x:");
    println!("    (Tf)(x) = ∫₀ˣ t dt = x²/2");
    println!("    Computed (Tf)(0.5) ≈ {:.6}", tf[50]);
    println!("    Expected: 0.5²/2 = 0.125");

    // Hilbert-Schmidt operators
    println!("\nHilbert-Schmidt operators:");
    println!("  Integral operators with L² kernel:");
    println!("  ∫∫|K(x,t)|² dx dt < ∞");
    println!("  All Hilbert-Schmidt operators are compact.");

    // =========================================================================
    // Part 5: Differential Operators
    // =========================================================================
    println!("\n\nPart 5: Differential Operators");
    println!("───────────────────────────────\n");

    println!("Differential operators are typically unbounded.");
    println!("They are defined on dense subspaces.");

    // First derivative
    let derivative = DifferentialOperator::first_derivative(0.0, 1.0, 100)?;

    let sin_fn: Vec<f64> = (0..100)
        .map(|i| {
            let x = i as f64 / 99.0;
            (std::f64::consts::PI * x).sin()
        })
        .collect();

    let cos_fn = derivative.apply(&sin_fn)?;

    println!("First derivative operator D = d/dx:");
    println!("  (Df)(x) = f'(x)");
    println!("\n  For f(x) = sin(πx):");
    println!("    f'(x) = π cos(πx)");
    println!("    Computed f'(0.5) ≈ {:.6}", cos_fn[50]);
    println!("    Expected: π·cos(π/2) = 0");

    // Laplacian
    println!("\nLaplacian operator Δ = ∂²/∂x²:");
    let laplacian = LaplacianOperator::new_1d(0.0, 1.0, 100)?;

    let laplacian_sin = laplacian.apply(&sin_fn)?;

    println!("  For f(x) = sin(πx):");
    println!("    Δf(x) = -π² sin(πx)");
    println!("    Computed Δf(0.5) ≈ {:.6}", laplacian_sin[50]);
    println!("    Expected: -π² ≈ {:.6}", -std::f64::consts::PI.powi(2));

    // =========================================================================
    // Part 6: Spectrum Preview
    // =========================================================================
    println!("\n\nPart 6: Operator Spectrum (Preview)");
    println!("────────────────────────────────────\n");

    println!("The spectrum σ(T) consists of λ where (T - λI) is not invertible:");
    println!("  - Point spectrum: eigenvalues");
    println!("  - Continuous spectrum: λ where range is dense but not closed");
    println!("  - Residual spectrum: λ where range is not dense");

    // Eigenvalues of our symmetric matrix
    let eigenvalues = op.eigenvalues()?;
    println!("\nEigenvalues of our symmetric matrix:");
    for (i, ev) in eigenvalues.iter().enumerate() {
        println!("  λ_{} = {:.6}", i, ev);
    }

    println!("\nFor self-adjoint operators:");
    println!("  - Spectrum is real");
    println!("  - No residual spectrum");
    println!("  - Spectral theorem gives decomposition");

    // =========================================================================
    // Part 7: Operator Algebra
    // =========================================================================
    println!("\n\nPart 7: Operator Algebra");
    println!("────────────────────────\n");

    println!("Operators form an algebra under:");
    println!("  - Addition: (S + T)x = Sx + Tx");
    println!("  - Scalar multiplication: (αT)x = α(Tx)");
    println!("  - Composition: (ST)x = S(Tx)");

    let b_matrix = DMatrix::from_row_slice(3, 3, &[
        1.0, 0.0, 1.0,
        0.0, 2.0, 0.0,
        1.0, 0.0, 1.0,
    ]);
    let op_b = MatrixOperator::new(b_matrix)?;

    // Sum
    let sum_op = op.add(&op_b)?;
    let sum_result = sum_op.apply(&x)?;
    println!("\n(A + B)x = ({:.1}, {:.1}, {:.1})", sum_result[0], sum_result[1], sum_result[2]);

    // Composition
    let comp_op = op.compose(&op_b)?;
    let comp_result = comp_op.apply(&x)?;
    println!("(AB)x = ({:.1}, {:.1}, {:.1})", comp_result[0], comp_result[1], comp_result[2]);

    // Commutator
    let ab = op.compose(&op_b)?;
    let ba = op_b.compose(&op)?;
    let ab_x = ab.apply(&x)?;
    let ba_x = ba.apply(&x)?;
    let commutator: Vec<f64> = ab_x.iter().zip(&ba_x).map(|(a, b)| a - b).collect();
    println!("\n[A,B]x = ABx - BAx = ({:.1}, {:.1}, {:.1})",
             commutator[0], commutator[1], commutator[2]);

    let matrices_commute = commutator.iter().all(|&c| c.abs() < 1e-10);
    println!("Do A and B commute? {}", matrices_commute);

    // =========================================================================
    // Part 8: Tridiagonal Operators (Efficient)
    // =========================================================================
    println!("\n\nPart 8: Tridiagonal Operators");
    println!("──────────────────────────────\n");

    println!("Many discretized differential operators are tridiagonal:");
    println!("  Ax_i = a_i x_{i-1} + b_i x_i + c_i x_{i+1}");

    let n = 5;
    let sub_diag = vec![-1.0; n-1];  // a_i = -1
    let main_diag = vec![2.0; n];     // b_i = 2
    let super_diag = vec![-1.0; n-1]; // c_i = -1

    let tridiag = TridiagonalOperator::new(sub_diag, main_diag, super_diag)?;

    println!("\nDiscrete Laplacian (1D, Dirichlet BC):");
    println!("  [ 2 -1  0  0  0]");
    println!("  [-1  2 -1  0  0]");
    println!("  [ 0 -1  2 -1  0]");
    println!("  [ 0  0 -1  2 -1]");
    println!("  [ 0  0  0 -1  2]");

    let test_vec = DVector::from_vec(vec![1.0, 0.0, 0.0, 0.0, 1.0]);
    let result = tridiag.apply(&test_vec)?;
    println!("\n  Applied to (1, 0, 0, 0, 1): ({}, {}, {}, {}, {})",
             result[0], result[1], result[2], result[3], result[4]);

    // Eigenvalues of discrete Laplacian
    let tridiag_eigs = tridiag.eigenvalues()?;
    println!("\n  Eigenvalues:");
    for (i, ev) in tridiag_eigs.iter().enumerate() {
        let expected = 4.0 * ((i + 1) as f64 * std::f64::consts::PI / (2.0 * (n + 1) as f64)).sin().powi(2);
        println!("    λ_{} = {:.6} (expected: {:.6})", i, ev, expected);
    }

    println!("\n═══════════════════════════════════════════════════════════════");
    println!("                    DEMO COMPLETE");
    println!("═══════════════════════════════════════════════════════════════");

    Ok(())
}
