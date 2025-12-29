import init, {
  WasmHilbertSpace,
  WasmMatrixOperator,
  WasmSpectralDecomposition,
  WasmSobolevSpace,
  powerMethod,
  inverseIteration,
  computeEigenvalues
} from '@justinelliottcobb/amari-wasm';

/**
 * Functional Analysis Examples
 *
 * This example demonstrates the functional analysis capabilities of Amari v0.15.0:
 * - Hilbert spaces over Clifford algebras
 * - Linear operators and matrix representations
 * - Spectral decomposition and eigenvalue computation
 * - Sobolev spaces with weak derivatives
 */
async function runFunctionalAnalysisExample() {
  console.log('📐 Functional Analysis Examples');
  console.log('================================');
  console.log('Hilbert Spaces, Operators, and Spectral Theory');

  // Initialize the WASM module
  await init();

  // 1. HILBERT SPACE OPERATIONS
  console.log('\n1. HILBERT SPACE - Cl(2,0,0) ≅ ℝ⁴');
  console.log('   The Clifford algebra Cl(2,0,0) as a 4-dimensional Hilbert space...');

  const space = new WasmHilbertSpace();

  console.log(`   Dimension: ${space.dimension()}`);
  console.log(`   Signature (p, q, r): [${space.signature().join(', ')}]`);

  // Create vectors in the Hilbert space
  // Basis: 1, e₁, e₂, e₁₂
  const x = [1.0, 2.0, 0.0, 0.0];  // 1 + 2e₁
  const y = [0.0, 0.0, 3.0, 4.0];  // 3e₂ + 4e₁₂
  const z = [1.0, 1.0, 1.0, 1.0];  // All components

  console.log(`\n   Vector x = 1 + 2e₁: [${x.join(', ')}]`);
  console.log(`   Vector y = 3e₂ + 4e₁₂: [${y.join(', ')}]`);
  console.log(`   Vector z = 1 + e₁ + e₂ + e₁₂: [${z.join(', ')}]`);

  // Inner products
  const ip_xy = space.innerProduct(x, y);
  const ip_xz = space.innerProduct(x, z);
  const ip_zz = space.innerProduct(z, z);

  console.log(`\n   Inner Products:`);
  console.log(`   ⟨x, y⟩ = ${ip_xy.toFixed(4)} (orthogonal since 0)`);
  console.log(`   ⟨x, z⟩ = ${ip_xz.toFixed(4)}`);
  console.log(`   ⟨z, z⟩ = ${ip_zz.toFixed(4)} = ||z||²`);

  // Norms and distances
  const norm_x = space.norm(x);
  const norm_y = space.norm(y);
  const dist_xy = space.distance(x, y);

  console.log(`\n   Norms:`);
  console.log(`   ||x|| = ${norm_x.toFixed(4)}`);
  console.log(`   ||y|| = ${norm_y.toFixed(4)}`);
  console.log(`   d(x,y) = ||x - y|| = ${dist_xy.toFixed(4)}`);

  // Orthogonality check
  const orthogonal = space.isOrthogonal(x, y, 1e-10);
  console.log(`\n   x ⟂ y? ${orthogonal}`);

  // Normalization
  const z_normalized = space.normalize(z);
  const norm_z_normalized = space.norm(z_normalized);
  console.log(`   Normalized z: [${z_normalized.map(v => v.toFixed(4)).join(', ')}]`);
  console.log(`   ||normalized z|| = ${norm_z_normalized.toFixed(4)}`);

  // Projection
  const proj_x_onto_z = space.project(x, z);
  console.log(`\n   Projection of x onto z:`);
  console.log(`   proj_z(x) = [${proj_x_onto_z.map(v => v.toFixed(4)).join(', ')}]`);

  // 2. MATRIX OPERATORS
  console.log('\n2. MATRIX OPERATORS - Bounded Linear Operators');
  console.log('   Working with 4×4 matrices on the Hilbert space...');

  // Identity operator
  const identity = WasmMatrixOperator.identity();
  console.log(`\n   Identity operator I:`);
  console.log(`   Trace(I) = ${identity.trace()}`);
  console.log(`   ||I|| = ${identity.operatorNorm().toFixed(4)}`);

  // Diagonal operator
  const diag = WasmMatrixOperator.diagonal([4.0, 3.0, 2.0, 1.0]);
  console.log(`\n   Diagonal operator D = diag(4, 3, 2, 1):`);
  console.log(`   Trace(D) = ${diag.trace()}`);
  console.log(`   ||D|| = ${diag.operatorNorm().toFixed(4)}`);
  console.log(`   D is symmetric: ${diag.isSymmetric(1e-10)}`);

  // Apply operator to vector
  const Dx = diag.apply([1.0, 1.0, 1.0, 1.0]);
  console.log(`   D · [1,1,1,1] = [${Dx.map(v => v.toFixed(1)).join(', ')}]`);

  // Scaling operator
  const scaled = WasmMatrixOperator.scaling(2.0);
  console.log(`\n   Scaling operator 2I:`);
  console.log(`   Trace(2I) = ${scaled.trace()}`);

  // Operator composition (matrix multiplication)
  const composed = diag.compose(scaled);
  console.log(`\n   Composition D ∘ (2I) = 2D:`);
  console.log(`   Trace = ${composed.trace()}`);

  // Transpose
  const asymmetric_entries = [
    1, 2, 0, 0,
    0, 1, 2, 0,
    0, 0, 1, 2,
    0, 0, 0, 1
  ];
  const asymmetric = new WasmMatrixOperator(asymmetric_entries);
  const transposed = asymmetric.transpose();
  console.log(`\n   Upper triangular matrix A and Aᵀ:`);
  console.log(`   A is symmetric: ${asymmetric.isSymmetric(1e-10)}`);
  console.log(`   Trace(A) = Trace(Aᵀ) = ${asymmetric.trace()}`);

  // 3. SPECTRAL DECOMPOSITION
  console.log('\n3. SPECTRAL DECOMPOSITION - Eigenvalue Analysis');
  console.log('   Decomposing symmetric operators using the spectral theorem...');

  // Create a symmetric positive-definite matrix
  const symmetric = WasmMatrixOperator.diagonal([4.0, 3.0, 2.0, 1.0]);

  // Compute spectral decomposition
  const decomp = WasmSpectralDecomposition.compute(symmetric, 100, 1e-10);

  console.log(`\n   Spectral decomposition of D = diag(4, 3, 2, 1):`);
  console.log(`   Eigenvalues: [${decomp.eigenvalues().map(v => v.toFixed(4)).join(', ')}]`);
  console.log(`   Complete decomposition: ${decomp.isComplete()}`);
  console.log(`   Spectral radius ρ(D) = ${decomp.spectralRadius().toFixed(4)}`);

  const condNum = decomp.conditionNumber();
  if (condNum !== undefined) {
    console.log(`   Condition number κ(D) = ${condNum.toFixed(4)}`);
  }

  console.log(`\n   Positivity checks:`);
  console.log(`   Positive definite: ${decomp.isPositiveDefinite()}`);
  console.log(`   Positive semi-definite: ${decomp.isPositiveSemidefinite()}`);

  // Apply the reconstructed operator
  const test_vec = [1.0, 0.0, 0.0, 0.0];
  const applied = decomp.apply(test_vec);
  console.log(`\n   Verify: D·e₀ = [${applied.map(v => v.toFixed(4)).join(', ')}]`);
  console.log(`   Expected: [4, 0, 0, 0]`);

  // Functional calculus: apply f(λ) = √λ
  console.log(`\n   Functional calculus - computing √D:`);
  const sqrt_applied = decomp.applyFunction(
    (lambda: number) => Math.sqrt(lambda),
    [1.0, 0.0, 0.0, 0.0]
  );
  console.log(`   √D·e₀ = [${sqrt_applied.map(v => v.toFixed(4)).join(', ')}]`);
  console.log(`   Expected: [2, 0, 0, 0] (since √4 = 2)`);

  // Compute exp(D) using functional calculus
  const exp_applied = decomp.applyFunction(
    (lambda: number) => Math.exp(lambda),
    [1.0, 0.0, 0.0, 0.0]
  );
  console.log(`   exp(D)·e₀ = [${exp_applied.map(v => v.toFixed(4)).join(', ')}]`);
  console.log(`   Expected: [${Math.exp(4).toFixed(4)}, 0, 0, 0]`);

  // 4. EIGENVALUE ALGORITHMS
  console.log('\n4. EIGENVALUE ALGORITHMS - Numerical Methods');
  console.log('   Power method, inverse iteration, and Jacobi algorithm...');

  // Power method: find dominant eigenvalue
  const power_result = powerMethod(symmetric, undefined, 100, 1e-10);
  console.log(`\n   Power method (dominant eigenvalue):`);
  console.log(`   λ₁ = ${power_result[0].toFixed(4)}`);
  console.log(`   v₁ = [${power_result.slice(1).map(v => v.toFixed(4)).join(', ')}]`);

  // Inverse iteration: find eigenvalue near a shift
  const inverse_result = inverseIteration(symmetric, 1.5, undefined, 100, 1e-10);
  console.log(`\n   Inverse iteration (eigenvalue near 1.5):`);
  console.log(`   λ ≈ ${inverse_result[0].toFixed(4)}`);

  // All eigenvalues using Jacobi algorithm
  const all_eigenvalues = computeEigenvalues(symmetric, 100, 1e-10);
  console.log(`\n   All eigenvalues (Jacobi algorithm):`);
  console.log(`   [${all_eigenvalues.map(v => v.toFixed(4)).join(', ')}]`);

  // 5. SOBOLEV SPACES
  console.log('\n5. SOBOLEV SPACES - Function Spaces with Derivatives');
  console.log('   Working with H¹([0,1]) - functions with L² derivatives...');

  const h1 = WasmSobolevSpace.h1UnitInterval();
  h1.setQuadraturePoints(64);

  console.log(`\n   H¹ Sobolev space properties:`);
  console.log(`   Order k = ${h1.order()}`);
  console.log(`   Domain = [${h1.bounds().join(', ')}]`);
  console.log(`   Poincaré constant ≈ ${h1.poincareConstant().toFixed(6)}`);
  console.log(`   (Theory: C = 1/π ≈ 0.318310)`);

  // Define functions for norm computation
  // f(x) = x (identity function)
  const f = (x: number) => x;
  const df = (_x: number) => 1.0;  // f'(x) = 1

  // Convert to JS functions for WASM
  const f_js = new Function('x', 'return x') as any;
  const df_js = new Function('x', 'return 1.0') as any;

  console.log(`\n   Function f(x) = x on [0, 1]:`);

  const l2_norm = h1.l2Norm(f_js);
  console.log(`   ||f||_{L²} = ${l2_norm.toFixed(4)}`);
  console.log(`   (Theory: ∫₀¹ x² dx = 1/3, so ||f||_{L²} = √(1/3) ≈ 0.5774)`);

  const h1_seminorm = h1.h1Seminorm(df_js);
  console.log(`\n   |f|_{H¹} = ||f'||_{L²} = ${h1_seminorm.toFixed(4)}`);
  console.log(`   (Theory: ∫₀¹ 1² dx = 1, so |f|_{H¹} = 1.0)`);

  const h1_norm = h1.h1Norm(f_js, df_js);
  console.log(`\n   ||f||_{H¹} = √(||f||²_{L²} + ||f'||²_{L²}) = ${h1_norm.toFixed(4)}`);
  console.log(`   (Theory: √(1/3 + 1) = √(4/3) ≈ 1.1547)`);

  // L² inner product example
  // g(x) = 1 - x
  const g_js = new Function('x', 'return 1 - x') as any;
  const inner = h1.l2InnerProduct(f_js, g_js);
  console.log(`\n   ⟨f, g⟩_{L²} where f(x) = x, g(x) = 1-x:`);
  console.log(`   ∫₀¹ x(1-x) dx = ${inner.toFixed(4)}`);
  console.log(`   (Theory: 1/2 - 1/3 = 1/6 ≈ 0.1667)`);

  // H² space
  const h2 = WasmSobolevSpace.h2UnitInterval();
  console.log(`\n   H² Sobolev space:`);
  console.log(`   Order k = ${h2.order()}`);

  // 6. APPLICATION: Quantum Mechanics
  console.log('\n6. APPLICATION - Quantum Mechanics');
  console.log('   Modeling a quantum observable as a self-adjoint operator...');

  // Create a simple Hamiltonian (energy operator)
  // H = diagonal matrix representing energy levels
  const hamiltonian = WasmMatrixOperator.diagonal([0.0, 1.0, 2.0, 3.0]);

  console.log(`\n   Hamiltonian H with energy levels [0, 1, 2, 3]:`);
  console.log(`   H is self-adjoint (symmetric): ${hamiltonian.isSymmetric(1e-10)}`);

  // Initial state: superposition
  const initial_state = space.normalize([1.0, 1.0, 0.0, 0.0]);
  console.log(`   Initial state |ψ⟩ = [${initial_state.map(v => v.toFixed(4)).join(', ')}]`);

  // Expected energy: ⟨ψ|H|ψ⟩
  const H_psi = hamiltonian.apply(initial_state);
  const expected_energy = space.innerProduct(initial_state, H_psi);
  console.log(`   Expected energy ⟨ψ|H|ψ⟩ = ${expected_energy.toFixed(4)}`);

  // Time evolution: e^{-iHt} (approximated as exp(-Ht) for real case)
  const H_decomp = WasmSpectralDecomposition.compute(hamiltonian, 100, 1e-10);
  const t = 1.0;  // time
  const evolved = H_decomp.applyFunction(
    (E: number) => Math.exp(-E * t),
    initial_state
  );
  console.log(`   Time evolution e^{-Ht}|ψ⟩ at t=1:`);
  console.log(`   [${evolved.map(v => v.toFixed(4)).join(', ')}]`);

  // 7. PERFORMANCE BENCHMARK
  console.log('\n7. PERFORMANCE BENCHMARK');
  console.log('   Running batch operations...');

  const iterations = 100;
  const start = performance.now();

  for (let i = 0; i < iterations; i++) {
    // Create and apply operators
    const A = WasmMatrixOperator.diagonal([
      1.0 + i * 0.01,
      2.0 + i * 0.01,
      3.0 + i * 0.01,
      4.0 + i * 0.01
    ]);
    const v = [1.0, 0.5, 0.25, 0.125];
    const result = A.apply(v);

    // Compute norm
    space.norm(result);
  }

  const end = performance.now();
  console.log(`   ${iterations} operator applications: ${(end - start).toFixed(2)}ms`);
  console.log(`   Average: ${((end - start) / iterations).toFixed(3)}ms per operation`);

  console.log('\n✅ Functional analysis example completed!');
  console.log('\nKey concepts demonstrated:');
  console.log('• Hilbert space Cl(2,0,0) with inner product ⟨·,·⟩');
  console.log('• Bounded linear operators as matrices');
  console.log('• Spectral theorem for self-adjoint operators');
  console.log('• Functional calculus f(A) for operator functions');
  console.log('• Sobolev spaces H^k with weak derivatives');
  console.log('• Applications to quantum mechanics');
}

// Run the example
runFunctionalAnalysisExample().catch(console.error);
