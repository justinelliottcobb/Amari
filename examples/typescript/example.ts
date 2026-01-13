/**
 * Amari Mathematical Computing Library - TypeScript Example v0.17.0
 *
 * This example demonstrates how to use the Amari WASM library in TypeScript
 * for geometric algebra, dual numbers, and numerical computing.
 */

import init, {
  WasmMultivector,
  WasmRotor,
  WasmDual,
  WasmDynamicalSystem,
  WasmTopology,
} from '@justinelliottcobb/amari-wasm';

// Common multivector coefficient arrays (8 coefficients for 3D Clifford algebra)
// Basis elements: 1, e1, e2, e3, e12, e13, e23, e123
const UNIT_X_VECTOR = new Float64Array([0, 1, 0, 0, 0, 0, 0, 0]); // e1
const UNIT_Y_VECTOR = new Float64Array([0, 0, 1, 0, 0, 0, 0, 0]); // e2
const UNIT_Z_VECTOR = new Float64Array([0, 0, 0, 1, 0, 0, 0, 0]); // e3
const UNIT_SCALAR = new Float64Array([1, 0, 0, 0, 0, 0, 0, 0]);   // 1

async function main() {
  // Initialize the WASM module
  await init();

  console.log('Amari Mathematical Computing Library - TypeScript Example v0.17.0\n');

  // ============================================
  // 1. Geometric Algebra Operations
  // ============================================
  console.log('=== Geometric Algebra ===\n');

  // Create basis vectors
  const e1 = WasmMultivector.basis_vector(0);
  const e2 = WasmMultivector.basis_vector(1);
  const e3 = WasmMultivector.basis_vector(2);

  console.log('Basis vectors:');
  console.log(`  e1 = ${e1.to_string()}`);
  console.log(`  e2 = ${e2.to_string()}`);
  console.log(`  e3 = ${e3.to_string()}`);

  // Geometric product: e1 * e2 = e12 (bivector)
  const e12 = e1.geometric_product(e2);
  console.log(`\nGeometric product: e1 * e2 = ${e12.to_string()}`);

  // e1 * e1 = 1 (self-product of unit vector)
  const e1_squared = e1.geometric_product(e1);
  console.log(`Self-product: e1 * e1 = ${e1_squared.to_string()}`);

  // Outer product: e1 ∧ e2 = e12
  const outer = e1.outer_product(e2);
  console.log(`Outer product: e1 ∧ e2 = ${outer.to_string()}`);

  // Inner product: e1 · e2 = 0 (orthogonal)
  const inner = e1.inner_product(e2);
  console.log(`Inner product: e1 · e2 = ${inner.to_string()} (orthogonal)`);

  // Inner product: e1 · e1 = 1
  const inner_self = e1.inner_product(e1);
  console.log(`Inner product: e1 · e1 = ${inner_self.to_string()}`);

  // ============================================
  // 2. Rotor Operations (3D Rotations)
  // ============================================
  console.log('\n=== Rotor Rotations ===\n');

  // Create a rotor for 90-degree rotation around z-axis
  const angle = Math.PI / 2;
  const rotor = WasmRotor.from_axis_angle(e3, angle);

  // Rotate the x-axis vector (e1)
  const rotated = rotor.rotate_vector(e1);

  console.log(`Rotating e1 by 90° around z-axis:`);
  console.log(`  Original: ${e1.to_string()}`);
  console.log(`  Rotated:  ${rotated.to_string()}`);
  console.log(`  (Expected: e2)`);

  // Compose rotations
  const rotor2 = WasmRotor.from_axis_angle(e1, Math.PI / 4);
  const composed = rotor.compose(rotor2);
  console.log(`\nRotor composition (first z, then x):`);
  console.log(`  Composed rotor: ${composed.to_string()}`);

  // ============================================
  // 3. Dual Numbers (Automatic Differentiation)
  // ============================================
  console.log('\n=== Dual Numbers for Automatic Differentiation ===\n');

  // f(x) = x^2 + 2x + 1 at x = 3
  // f'(x) = 2x + 2, so f'(3) = 8
  const x = WasmDual.variable(3.0);
  const two = WasmDual.constant(2.0);
  const one = WasmDual.constant(1.0);

  const x_squared = x.mul(x);
  const two_x = two.mul(x);
  const sum1 = x_squared.add(two_x);
  const result = sum1.add(one);

  console.log('f(x) = x² + 2x + 1');
  console.log(`f(3) = ${result.real()} (expected: 16)`);
  console.log(`f\'(3) = ${result.dual()} (expected: 8)`);

  // Trigonometric derivatives
  console.log('\nTrigonometric functions:');
  const y = WasmDual.variable(Math.PI / 4);
  const sin_y = y.sin();
  const cos_y = y.cos();

  console.log(`sin(π/4) = ${sin_y.real().toFixed(6)}, d/dx = ${sin_y.dual().toFixed(6)}`);
  console.log(`cos(π/4) = ${cos_y.real().toFixed(6)}, d/dx = ${cos_y.dual().toFixed(6)}`);
  console.log('(Derivative of sin is cos, derivative of cos is -sin)');

  // Chain rule example
  console.log('\nChain rule: f(x) = exp(sin(x)) at x = 0');
  const z = WasmDual.variable(0.0);
  const sin_z = z.sin();
  const exp_sin = sin_z.exp();
  console.log(`f(0) = ${exp_sin.real().toFixed(6)} (expected: 1)`);
  console.log(`f\'(0) = ${exp_sin.dual().toFixed(6)} (expected: 1)`);

  // ============================================
  // 4. Dynamical Systems (NEW in v0.17.0)
  // ============================================
  console.log('\n=== Dynamical Systems ===\n');

  try {
    // Create Lorenz system
    const lorenz = WasmDynamicalSystem.lorenz(10.0, 28.0, 8.0/3.0);
    console.log('Lorenz system created (σ=10, ρ=28, β=8/3)');

    // Initial condition
    const initial = new Float64Array([1.0, 0.0, 0.0]);

    // Integrate trajectory
    const trajectory = lorenz.integrate(initial, 0.0, 10.0, 0.01);
    console.log(`Trajectory computed: ${trajectory.length} points`);

    // Get final state
    const final_state = trajectory[trajectory.length - 1];
    console.log(`Final state: (${final_state[0].toFixed(4)}, ${final_state[1].toFixed(4)}, ${final_state[2].toFixed(4)})`);

    // Compute Lyapunov exponent (indicates chaos)
    const lyapunov = lorenz.largest_lyapunov_exponent(initial, 1000);
    console.log(`Largest Lyapunov exponent: ${lyapunov.toFixed(4)} (positive = chaotic)`);

    lorenz.free();
  } catch (e) {
    console.log('Dynamical systems module not available in this build');
  }

  // ============================================
  // 5. Topology (NEW in v0.17.0)
  // ============================================
  console.log('\n=== Computational Topology ===\n');

  try {
    // Create a simple simplicial complex (triangle)
    const complex = WasmTopology.create_simplex([0, 1, 2]);
    console.log('Triangle simplicial complex:');
    console.log(`  f-vector: ${complex.f_vector()}`);
    console.log(`  Euler characteristic: ${complex.euler_characteristic()}`);

    // Compute Betti numbers
    const betti = complex.betti_numbers();
    console.log(`  β₀ = ${betti[0]} (connected components)`);
    console.log(`  β₁ = ${betti[1]} (loops)`);

    complex.free();

    // Persistent homology example
    console.log('\nPersistent homology:');
    const points = new Float64Array([
      0.0, 0.0,
      1.0, 0.0,
      0.5, 0.866,
      0.5, 0.289
    ]);

    const persistence = WasmTopology.rips_persistence(points, 2.0);
    console.log(`  Persistence pairs computed: ${persistence.num_pairs()}`);

    persistence.free();
  } catch (e) {
    console.log('Topology module not available in this build');
  }

  // ============================================
  // 6. Performance Benchmark
  // ============================================
  console.log('\n=== Performance Benchmark ===\n');

  const iterations = 10000;
  const startTime = performance.now();

  for (let i = 0; i < iterations; i++) {
    const a = WasmMultivector.random();
    const b = WasmMultivector.random();
    const result = a.geometric_product(b);
    result.free();
    a.free();
    b.free();
  }

  const endTime = performance.now();
  const totalTime = endTime - startTime;
  const opsPerSecond = (iterations / (totalTime / 1000)).toFixed(0);

  console.log(`Geometric products: ${iterations} operations`);
  console.log(`Total time: ${totalTime.toFixed(2)}ms`);
  console.log(`Throughput: ${opsPerSecond} ops/sec`);

  // ============================================
  // 7. Memory Management
  // ============================================
  console.log('\n=== Memory Management ===\n');
  console.log('WASM objects must be freed to avoid memory leaks:');
  console.log('  multivector.free()');
  console.log('  rotor.free()');
  console.log('  dual.free()');
  console.log('');
  console.log('Use try/finally blocks for safe cleanup.');

  // Clean up WASM objects
  e1.free();
  e2.free();
  e3.free();
  e12.free();
  e1_squared.free();
  outer.free();
  inner.free();
  inner_self.free();
  rotor.free();
  rotor2.free();
  composed.free();
  rotated.free();
  x.free();
  two.free();
  one.free();
  x_squared.free();
  two_x.free();
  sum1.free();
  result.free();
  y.free();
  sin_y.free();
  cos_y.free();
  z.free();
  sin_z.free();
  exp_sin.free();

  console.log('\nExample completed successfully!');
}

// Run the example
main().catch(console.error);
