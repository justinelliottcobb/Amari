/**
 * Examples demonstrating the Amari library usage
 */

import { initAmari, GA, Multivector, Rotor, MultivectorBuilder, BasisBlade } from './index.js';

/**
 * Basic geometric algebra operations example
 */
export async function basicExample(): Promise<void> {
  await initAmari();
  
  console.log('=== Basic Geometric Algebra Example ===');
  
  // Create basis vectors
  const e1 = GA.e1();
  const e2 = GA.e2();
  const e3 = GA.e3();
  
  console.log('Basis vectors:');
  console.log(`e1 = ${e1.toString()}`);
  console.log(`e2 = ${e2.toString()}`);
  console.log(`e3 = ${e3.toString()}`);
  
  // Test orthonormality
  console.log('\nOrthonormality:');
  console.log(`e1 · e1 = ${e1.scalarProduct(e1)}`);
  console.log(`e1 · e2 = ${e1.scalarProduct(e2)}`);
  
  // Create bivector
  const e12 = e1.outerProduct(e2);
  console.log(`\ne1 ∧ e2 = ${e12.toString()}`);
  
  // Test anticommutivity
  const e21 = e2.outerProduct(e1);
  console.log(`e2 ∧ e1 = ${e21.toString()}`);
  console.log(`Anticommutative: ${e12.add(e21).norm() < 1e-10}`);
}

/**
 * Rotation example using rotors
 */
export async function rotationExample(): Promise<void> {
  await initAmari();
  
  console.log('\n=== Rotation Example ===');
  
  const e1 = GA.e1();
  const e2 = GA.e2();
  const e12 = e1.outerProduct(e2);
  
  // Create 90-degree rotation in xy-plane
  const angle = Math.PI / 2;
  const rotor = Rotor.fromBivector(e12, angle);
  
  console.log(`90° rotation rotor: ${rotor.apply(GA.scalar(1)).toString()}`);
  
  // Rotate e1 -> should become e2
  const rotatedE1 = rotor.apply(e1);
  console.log(`R·e1·R† = ${rotatedE1.toString()}`);
  console.log(`Error from e2: ${rotatedE1.sub(e2).norm()}`);
  
  // Rotate e2 -> should become -e1
  const rotatedE2 = rotor.apply(e2);
  console.log(`R·e2·R† = ${rotatedE2.toString()}`);
  console.log(`Error from -e1: ${rotatedE2.add(e1).norm()}`);
}

/**
 * Builder pattern example
 */
export async function builderExample(): Promise<void> {
  await initAmari();
  
  console.log('\n=== Builder Pattern Example ===');
  
  // Create a complex multivector using the builder
  const mv = GA.builder()
    .scalar(1.0)
    .e1(2.0)
    .e2(3.0)
    .e12(0.5)
    .build();
  
  console.log(`Complex multivector: ${mv.toString()}`);
  console.log(`Norm: ${mv.norm()}`);
  console.log(`Normalized: ${mv.normalize().toString()}`);
  
  // Demonstrate component access
  console.log('\nComponent access:');
  console.log(`Scalar part: ${mv.scalar}`);
  console.log(`e1 part: ${mv.e1}`);
  console.log(`e2 part: ${mv.e2}`);
  console.log(`e12 part: ${mv.e12}`);
}

/**
 * Vector operations example
 */
export async function vectorExample(): Promise<void> {
  await initAmari();
  
  console.log('\n=== Vector Operations Example ===');
  
  // Create two vectors
  const v1 = GA.builder().e1(3).e2(4).build();
  const v2 = GA.builder().e1(1).e2(2).build();
  
  console.log(`v1 = ${v1.toString()}`);
  console.log(`v2 = ${v2.toString()}`);
  
  // Dot product (inner product of vectors gives scalar)
  const dot = v1.innerProduct(v2).scalar;
  console.log(`v1 · v2 = ${dot}`);
  
  // Cross product in 3D (using outer product and dualization)
  const v3d1 = GA.builder().e1(1).e2(0).e3(0).build();
  const v3d2 = GA.builder().e1(0).e2(1).e3(0).build();
  const cross = v3d1.outerProduct(v3d2);
  console.log(`e1 × e2 = ${cross.toString()}`);
  
  // Magnitude
  console.log(`|v1| = ${v1.norm()}`);
  console.log(`|v2| = ${v2.norm()}`);
}

/**
 * Run all examples
 */
export async function runAllExamples(): Promise<void> {
  try {
    await basicExample();
    await rotationExample();
    await builderExample();
    await vectorExample();
    console.log('\n=== All examples completed successfully! ===');
  } catch (error) {
    console.error('Example failed:', error);
  }
}

// If running directly in Node.js
if (typeof window === 'undefined' && typeof module !== 'undefined') {
  runAllExamples();
}