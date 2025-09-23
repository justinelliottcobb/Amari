/**
 * Amari Mathematical Computing Library - TypeScript Example
 *
 * This example demonstrates how to use the Amari WASM library in TypeScript
 * for geometric algebra, tropical algebra, and cellular automata.
 */

import init, {
  WasmMultivector,
  WasmRotor,
  WasmGeometricCA,
  create_bivector,
  geometric_product,
  tropical_add,
  tropical_multiply
} from '@amari/core';

// Common multivector coefficient arrays (8 coefficients for 3D Clifford algebra)
// Basis elements: 1, e1, e2, e3, e12, e13, e23, e123
const UNIT_X_VECTOR = new Float64Array([0, 1, 0, 0, 0, 0, 0, 0]); // e1
const UNIT_Y_VECTOR = new Float64Array([0, 0, 1, 0, 0, 0, 0, 0]); // e2
const UNIT_Z_VECTOR = new Float64Array([0, 0, 0, 1, 0, 0, 0, 0]); // e3
const UNIT_SCALAR = new Float64Array([1, 0, 0, 0, 0, 0, 0, 0]);   // 1

// Example vectors for demonstrations
const SAMPLE_VECTOR_1 = new Float64Array([1, 2, 3, 0, 0, 0, 0, 0]); // 1 + 2e1 + 3e2
const SAMPLE_VECTOR_2 = new Float64Array([4, 5, 6, 0, 0, 0, 0, 0]); // 4 + 5e1 + 6e2

async function main() {
  // Initialize the WASM module
  await init();

  console.log('🚀 Amari Mathematical Computing Library - TypeScript Example\n');

  // ============================================
  // 1. Geometric Algebra Operations
  // ============================================
  console.log('📐 Geometric Algebra:');

  // Create basis vectors
  const e1 = WasmMultivector.basis_vector(0);
  const e2 = WasmMultivector.basis_vector(1);
  const e3 = WasmMultivector.basis_vector(2);

  // Geometric product
  const v1 = WasmMultivector.from_coefficients(SAMPLE_VECTOR_1);
  const v2 = WasmMultivector.from_coefficients(SAMPLE_VECTOR_2);
  const product = v1.geometric_product(v2);

  console.log(`v1 = ${v1.to_string()}`);
  console.log(`v2 = ${v2.to_string()}`);
  console.log(`v1 * v2 = ${product.to_string()}\n`);

  // ============================================
  // 2. Rotor Operations (3D Rotations)
  // ============================================
  console.log('🔄 Rotor Operations:');

  // Create a rotor for 90-degree rotation around z-axis
  const angle = Math.PI / 2;
  const axis = e3;
  const rotor = WasmRotor.from_axis_angle(axis, angle);

  // Apply rotation to a vector (unit scalar in this case)
  const vector = WasmMultivector.from_coefficients(UNIT_SCALAR);
  const rotated = rotor.rotate_vector(vector);

  console.log(`Original vector: ${vector.to_string()}`);
  console.log(`Rotated vector: ${rotated.to_string()}`);
  console.log(`Rotation angle: ${angle} radians\n`);

  // ============================================
  // 3. Tropical Algebra Operations
  // ============================================
  console.log('🌴 Tropical Algebra:');

  // In tropical algebra: addition = max, multiplication = addition
  const a = 5.0;
  const b = 3.0;
  const c = 7.0;

  const tropicalSum = tropical_add(tropical_add(a, b), c);
  const tropicalProduct = tropical_multiply(tropical_multiply(a, b), c);

  console.log(`Tropical sum (max): ${a} ⊕ ${b} ⊕ ${c} = ${tropicalSum}`);
  console.log(`Tropical product (add): ${a} ⊗ ${b} ⊗ ${c} = ${tropicalProduct}\n`);

  // ============================================
  // 4. Cellular Automata with Geometric States
  // ============================================
  console.log('🔲 Geometric Cellular Automata:');

  // Create a 50x50 geometric CA
  const ca = WasmGeometricCA.new(50, 50);

  // Set initial configuration (glider pattern with multivector states)
  ca.set_cell(25, 25, WasmMultivector.basis_vector(0));
  ca.set_cell(26, 25, WasmMultivector.basis_vector(1));
  ca.set_cell(27, 25, WasmMultivector.basis_vector(2));
  ca.set_cell(27, 26, WasmMultivector.basis_vector(0));
  ca.set_cell(26, 27, WasmMultivector.basis_vector(1));

  // Evolve the CA for 10 steps
  for (let i = 0; i < 10; i++) {
    ca.step();
  }

  console.log(`CA evolved for 10 steps`);
  console.log(`Generation: ${ca.generation()}`);
  console.log(`Grid dimensions: ${ca.width()} x ${ca.height()}\n`);

  // ============================================
  // 5. Performance Metrics
  // ============================================
  console.log('⚡ Performance Metrics:');

  const iterations = 10000;
  const startTime = performance.now();

  for (let i = 0; i < iterations; i++) {
    const a = WasmMultivector.random();
    const b = WasmMultivector.random();
    const result = a.geometric_product(b);
    result.free(); // Clean up WASM memory
    a.free();
    b.free();
  }

  const endTime = performance.now();
  const totalTime = endTime - startTime;
  const opsPerSecond = (iterations / (totalTime / 1000)).toFixed(0);

  console.log(`Performed ${iterations} geometric products in ${totalTime.toFixed(2)}ms`);
  console.log(`Operations per second: ${opsPerSecond}\n`);

  // Clean up WASM objects
  e1.free();
  e2.free();
  e3.free();
  v1.free();
  v2.free();
  product.free();
  rotor.free();
  vector.free();
  rotated.free();
  ca.free();

  console.log('✅ Example completed successfully!');
}

// Run the example
main().catch(console.error);