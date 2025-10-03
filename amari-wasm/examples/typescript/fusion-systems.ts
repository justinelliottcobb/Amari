import init, {
  WasmTropicalDualClifford,
  FusionUtils
} from '@justinelliottcobb/amari-wasm';

async function runFusionSystemsExample() {
  console.log('🔲 Fusion Systems Examples');
  console.log('===========================');

  // Initialize the WASM module
  await init();

  // Basic fusion system operations
  console.log('\n1. Tropical-Dual-Clifford Fusion:');
  console.log('   Combining tropical algebra + automatic differentiation + geometric algebra');

  // Create fusion objects with different component values
  const fusionA = WasmTropicalDualClifford.new(
    2.0,   // tropical value
    3.0,   // dual real part
    1.0,   // dual infinitesimal part
    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] // Clifford coefficients (scalar + 7 basis elements)
  );

  const fusionB = WasmTropicalDualClifford.new(
    1.5,   // tropical value
    2.0,   // dual real part
    0.5,   // dual infinitesimal part
    [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] // Clifford coefficients (e1 basis vector)
  );

  console.log(`   Fusion A: tropical=${fusionA.tropicalValue()}, dual=(${fusionA.dualReal()}, ${fusionA.dualInfinitesimal()})`);
  console.log(`   Fusion B: tropical=${fusionB.tropicalValue()}, dual=(${fusionB.dualReal()}, ${fusionB.dualInfinitesimal()})`);

  // Tropical operations (max-plus algebra)
  console.log('\n2. Tropical Operations in Fusion:');

  const tropicalSum = fusionA.tropicalAdd(fusionB);
  const tropicalProduct = fusionA.tropicalMultiply(fusionB);

  console.log(`   A ⊕ B (tropical): ${tropicalSum.tropicalValue()}`);
  console.log(`   A ⊗ B (tropical): ${tropicalProduct.tropicalValue()}`);

  // Dual number operations (automatic differentiation)
  console.log('\n3. Automatic Differentiation in Fusion:');

  const dualSum = fusionA.dualAdd(fusionB);
  const dualProduct = fusionA.dualMultiply(fusionB);

  console.log(`   A + B (dual): (${dualSum.dualReal()}, ${dualSum.dualInfinitesimal()})`);
  console.log(`   A × B (dual): (${dualProduct.dualReal()}, ${dualProduct.dualInfinitesimal()})`);

  // Clifford algebra operations (geometric algebra)
  console.log('\n4. Geometric Algebra in Fusion:');

  const cliffordProduct = fusionA.cliffordProduct(fusionB);
  const cliffordWedge = fusionA.cliffordWedge(fusionB);

  console.log(`   A * B (geometric product): ${cliffordProduct.cliffordToString()}`);
  console.log(`   A ∧ B (wedge product): ${cliffordWedge.cliffordToString()}`);

  // Combined fusion operations
  console.log('\n5. Full Fusion Operations:');

  const fullFusion = fusionA.fusionMultiply(fusionB);
  console.log(`   Full fusion A ⊛ B:`);
  console.log(`     Tropical: ${fullFusion.tropicalValue()}`);
  console.log(`     Dual: (${fullFusion.dualReal()}, ${fullFusion.dualInfinitesimal()})`);
  console.log(`     Clifford: ${fullFusion.cliffordToString()}`);

  // Neural network attention mechanism
  console.log('\n6. Attention Mechanism Simulation:');

  // Create query, key, value vectors in fusion space
  const query = WasmTropicalDualClifford.new(
    1.0, 2.0, 1.0,
    [0.5, 0.3, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0]
  );

  const key1 = WasmTropicalDualClifford.new(
    0.8, 1.5, 0.5,
    [0.4, 0.4, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0]
  );

  const key2 = WasmTropicalDualClifford.new(
    1.2, 1.8, 0.7,
    [0.3, 0.5, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0]
  );

  // Compute attention scores using fusion operations
  const attention1 = query.fusionMultiply(key1);
  const attention2 = query.fusionMultiply(key2);

  console.log(`   Query vector: tropical=${query.tropicalValue()}`);
  console.log(`   Key 1 attention: ${attention1.tropicalValue()}`);
  console.log(`   Key 2 attention: ${attention2.tropicalValue()}`);

  // Determine which key has higher attention
  const maxAttention = Math.max(attention1.tropicalValue(), attention2.tropicalValue());
  console.log(`   Maximum attention: ${maxAttention}`);

  // Batch processing for efficiency
  console.log('\n7. Batch Processing:');

  const batchInputs = [
    { tropical: 1.0, dualReal: 1.0, dualInf: 0.5, clifford: [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] },
    { tropical: 1.5, dualReal: 1.5, dualInf: 0.7, clifford: [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] },
    { tropical: 2.0, dualReal: 2.0, dualInf: 1.0, clifford: [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0] }
  ];

  console.log('   Processing batch of fusion vectors...');

  const batchResults = [];
  for (let i = 0; i < batchInputs.length; i++) {
    const input = batchInputs[i];
    const fusionInput = WasmTropicalDualClifford.new(
      input.tropical,
      input.dualReal,
      input.dualInf,
      input.clifford
    );

    const result = fusionInput.fusionMultiply(fusionA);
    batchResults.push({
      tropical: result.tropicalValue(),
      dual: [result.dualReal(), result.dualInfinitesimal()],
      index: i
    });

    console.log(`   Batch ${i + 1}: tropical=${result.tropicalValue().toFixed(3)}, dual=[${result.dualReal().toFixed(3)}, ${result.dualInfinitesimal().toFixed(3)}]`);

    fusionInput.free();
    result.free();
  }

  // Optimization in fusion space
  console.log('\n8. Optimization in Fusion Space:');

  console.log('   Gradient descent using dual components for derivatives...');

  let optimPoint = WasmTropicalDualClifford.new(
    1.0, 0.0, 1.0, // Start with dual infinitesimal = 1 for gradient
    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
  );

  for (let iter = 0; iter < 5; iter++) {
    // Simulate objective function: f(x) = x² in dual space
    const objective = optimPoint.dualMultiply(optimPoint);
    const gradient = objective.dualInfinitesimal(); // df/dx
    const currentValue = objective.dualReal();

    console.log(`   Iteration ${iter}: value=${currentValue.toFixed(4)}, gradient=${gradient.toFixed(4)}`);

    // Update using gradient descent
    const learningRate = 0.1;
    const newDualReal = optimPoint.dualReal() - learningRate * gradient;

    const newPoint = WasmTropicalDualClifford.new(
      optimPoint.tropicalValue(),
      newDualReal,
      1.0, // Keep infinitesimal for gradient computation
      [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    );

    optimPoint.free();
    optimPoint = newPoint;
    objective.free();

    if (Math.abs(gradient) < 1e-6) {
      console.log(`   Converged at x = ${optimPoint.dualReal()}`);
      break;
    }
  }

  // Geometric transformations in fusion space
  console.log('\n9. Geometric Transformations:');

  // Create a rotor-like element in Clifford algebra component
  const rotorFusion = WasmTropicalDualClifford.new(
    1.0, 1.0, 0.0,
    [Math.cos(Math.PI/4), 0.0, 0.0, 0.0, 0.0, 0.0, Math.sin(Math.PI/4), 0.0] // e12 bivector rotor
  );

  const vectorFusion = WasmTropicalDualClifford.new(
    1.0, 1.0, 0.0,
    [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] // e1 vector
  );

  // Apply rotation in Clifford space while preserving tropical and dual components
  const rotated = rotorFusion.cliffordProduct(vectorFusion);
  console.log(`   Original vector: ${vectorFusion.cliffordToString()}`);
  console.log(`   Rotated vector: ${rotated.cliffordToString()}`);

  // Utility function demonstration
  console.log('\n10. Fusion Utilities:');

  const utilityResult = FusionUtils.combineAll([1.0, 2.0, 1.5], [1.0, 1.5, 1.2], [0.5, 0.7, 0.3]);
  console.log(`   Combined fusion result: ${utilityResult.tropicalValue()}`);

  // Clean up WASM memory
  fusionA.free();
  fusionB.free();
  tropicalSum.free();
  tropicalProduct.free();
  dualSum.free();
  dualProduct.free();
  cliffordProduct.free();
  cliffordWedge.free();
  fullFusion.free();
  query.free();
  key1.free();
  key2.free();
  attention1.free();
  attention2.free();
  optimPoint.free();
  rotorFusion.free();
  vectorFusion.free();
  rotated.free();
  utilityResult.free();

  console.log('\n✅ Fusion systems example completed!');
}

// Run the example
runFusionSystemsExample().catch(console.error);