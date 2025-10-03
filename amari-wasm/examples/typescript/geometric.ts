import init, { WasmMultivector, WasmRotor } from '@justinelliottcobb/amari-wasm';

async function runGeometricAlgebraExample() {
  console.log('🔢 Geometric Algebra Examples');
  console.log('============================');

  // Initialize the WASM module
  await init();

  // Basic multivector operations
  console.log('\n1. Basic Multivector Operations:');
  const e1 = WasmMultivector.basis_vector(0);
  const e2 = WasmMultivector.basis_vector(1);
  const e3 = WasmMultivector.basis_vector(2);

  console.log(`e1: ${e1.to_string()}`);
  console.log(`e2: ${e2.to_string()}`);
  console.log(`e3: ${e3.to_string()}`);

  // Geometric product (creates bivectors)
  const e12 = e1.geometric_product(e2);
  const e23 = e2.geometric_product(e3);
  const e31 = e3.geometric_product(e1);

  console.log(`\n2. Geometric Products (Bivectors):`);
  console.log(`e1 ∧ e2 = ${e12.to_string()}`);
  console.log(`e2 ∧ e3 = ${e23.to_string()}`);
  console.log(`e3 ∧ e1 = ${e31.to_string()}`);

  // 3D vector creation and operations
  console.log('\n3. 3D Vector Operations:');
  const vectorA = WasmMultivector.from_coefficients(
    new Float64Array([0, 1, 0, 0, 0, 0, 0, 0]) // x-component
  );
  const vectorB = WasmMultivector.from_coefficients(
    new Float64Array([0, 0, 1, 0, 0, 0, 0, 0]) // y-component
  );

  const sum = vectorA.add(vectorB);
  const cross = vectorA.wedge_product(vectorB);
  const dot = vectorA.inner_product(vectorB);

  console.log(`Vector A: ${vectorA.to_string()}`);
  console.log(`Vector B: ${vectorB.to_string()}`);
  console.log(`A + B: ${sum.to_string()}`);
  console.log(`A ∧ B (cross): ${cross.to_string()}`);
  console.log(`A · B (dot): ${dot}`);

  // Rotor-based rotations
  console.log('\n4. 3D Rotations with Rotors:');

  // 90-degree rotation around z-axis
  const zAxis = WasmMultivector.basis_vector(2);
  const rotor90 = WasmRotor.from_axis_angle(zAxis, Math.PI / 2);

  // Rotate x-unit vector
  const xVector = WasmMultivector.from_coefficients(
    new Float64Array([0, 1, 0, 0, 0, 0, 0, 0])
  );
  const rotatedX = rotor90.rotate_vector(xVector);

  console.log(`Original vector: ${xVector.to_string()}`);
  console.log(`After 90° rotation around z: ${rotatedX.to_string()}`);

  // Compose rotations
  const rotor45 = WasmRotor.from_axis_angle(zAxis, Math.PI / 4);
  const composedRotor = rotor90.compose(rotor45);
  const doubleRotated = composedRotor.rotate_vector(xVector);

  console.log(`After composed 135° rotation: ${doubleRotated.to_string()}`);

  // Electromagnetic field example
  console.log('\n5. Electromagnetic Field (Physics Application):');

  // Electric field in x-direction
  const electricField = WasmMultivector.from_coefficients(
    new Float64Array([0, 1, 0, 0, 0, 0, 0, 0])
  );

  // Magnetic field in y-direction
  const magneticField = WasmMultivector.from_coefficients(
    new Float64Array([0, 0, 1, 0, 0, 0, 0, 0])
  );

  // Electromagnetic field bivector
  const emField = electricField.wedge_product(magneticField);
  console.log(`Electric field: ${electricField.to_string()}`);
  console.log(`Magnetic field: ${magneticField.to_string()}`);
  console.log(`EM field bivector: ${emField.to_string()}`);

  // Clean up WASM memory
  e1.free();
  e2.free();
  e3.free();
  e12.free();
  e23.free();
  e31.free();
  vectorA.free();
  vectorB.free();
  sum.free();
  cross.free();
  zAxis.free();
  rotor90.free();
  rotor45.free();
  composedRotor.free();
  xVector.free();
  rotatedX.free();
  doubleRotated.free();
  electricField.free();
  magneticField.free();
  emField.free();

  console.log('\n✅ Geometric algebra example completed!');
}

// Run the example
runGeometricAlgebraExample().catch(console.error);