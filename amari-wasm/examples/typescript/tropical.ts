import init, {
  WasmTropicalNumber,
  WasmTropicalVector,
  WasmTropicalMatrix,
  tropical_add,
  tropical_multiply
} from '@justinelliottcobb/amari-wasm';

async function runTropicalAlgebraExample() {
  console.log('🌴 Tropical Algebra Examples');
  console.log('============================');

  // Initialize the WASM module
  await init();

  // Basic tropical operations
  console.log('\n1. Basic Tropical Operations:');
  console.log('   Tropical addition = max(a, b)');
  console.log('   Tropical multiplication = a + b');

  const a = 5.0;
  const b = 3.0;
  const tropAdd = tropical_add(a, b);
  const tropMult = tropical_multiply(a, b);

  console.log(`   ${a} ⊕ ${b} = max(${a}, ${b}) = ${tropAdd}`);
  console.log(`   ${a} ⊗ ${b} = ${a} + ${b} = ${tropMult}`);

  // Tropical numbers
  console.log('\n2. Tropical Number Objects:');
  const tropA = WasmTropicalNumber.new(7.0);
  const tropB = WasmTropicalNumber.new(4.0);
  const tropC = WasmTropicalNumber.new(-2.0);

  console.log(`   A = ${tropA.value()}`);
  console.log(`   B = ${tropB.value()}`);
  console.log(`   C = ${tropC.value()}`);

  const sum = tropA.tropical_add(tropB);
  const product = tropA.tropical_multiply(tropB);
  const power = tropA.tropical_power(3);

  console.log(`   A ⊕ B = ${sum.value()}`);
  console.log(`   A ⊗ B = ${product.value()}`);
  console.log(`   A^3 = ${power.value()}`);

  // Tropical vectors for neural networks
  console.log('\n3. Tropical Vectors (Neural Network Layers):');
  const inputVector = WasmTropicalVector.from_array([2.0, -1.0, 3.5, 0.0]);
  console.log(`   Input: [${inputVector.to_array().join(', ')}]`);

  // Simulate weighted connections (tropical matrix-vector multiplication)
  const weights = WasmTropicalMatrix.from_nested_array([
    [1.0, 0.5, -0.5, 2.0],
    [-1.0, 2.0, 1.5, 0.0],
    [0.0, -2.0, 1.0, 1.5]
  ]);

  console.log('   Weight matrix:');
  const weightArray = weights.to_nested_array();
  weightArray.forEach((row, i) => {
    console.log(`     [${row.join(', ')}]`);
  });

  const output = weights.tropical_multiply_vector(inputVector);
  console.log(`   Output: [${output.to_array().join(', ')}]`);

  // Shortest path example
  console.log('\n4. Shortest Path Problem:');
  console.log('   Tropical matrix powers give shortest paths');

  // Distance matrix (3x3 graph)
  const distanceMatrix = WasmTropicalMatrix.from_nested_array([
    [0.0, 2.0, 5.0],
    [3.0, 0.0, 1.0],
    [1.0, 4.0, 0.0]
  ]);

  console.log('   Distance matrix:');
  const distArray = distanceMatrix.to_nested_array();
  distArray.forEach((row, i) => {
    console.log(`     [${row.join(', ')}]`);
  });

  // Compute shortest paths via tropical matrix multiplication
  const paths2 = distanceMatrix.tropical_multiply(distanceMatrix);
  const paths3 = paths2.tropical_multiply(distanceMatrix);

  console.log('\n   2-step shortest paths:');
  const paths2Array = paths2.to_nested_array();
  paths2Array.forEach((row, i) => {
    console.log(`     [${row.join(', ')}]`);
  });

  console.log('\n   3-step shortest paths:');
  const paths3Array = paths3.to_nested_array();
  paths3Array.forEach((row, i) => {
    console.log(`     [${row.join(', ')}]`);
  });

  // Tropical neural network simulation
  console.log('\n5. Tropical Neural Network Simulation:');

  const layer1Input = WasmTropicalVector.from_array([1.0, -0.5, 2.0]);
  const layer1Weights = WasmTropicalMatrix.from_nested_array([
    [0.5, -1.0, 1.5],
    [2.0, 0.0, -0.5],
    [-1.0, 1.0, 0.8],
    [0.3, -2.0, 2.5]
  ]);

  const layer1Output = layer1Weights.tropical_multiply_vector(layer1Input);
  console.log(`   Layer 1 input: [${layer1Input.to_array().join(', ')}]`);
  console.log(`   Layer 1 output: [${layer1Output.to_array().join(', ')}]`);

  // Second layer
  const layer2Weights = WasmTropicalMatrix.from_nested_array([
    [1.0, -0.5, 0.8, 0.0],
    [0.5, 2.0, -1.0, 1.5]
  ]);

  const finalOutput = layer2Weights.tropical_multiply_vector(layer1Output);
  console.log(`   Final output: [${finalOutput.to_array().join(', ')}]`);

  // Batch operations for efficiency
  console.log('\n6. Batch Operations:');
  const batchInput = [
    [1.0, 0.0, -1.0],
    [2.0, -0.5, 1.5],
    [-1.0, 1.0, 0.0]
  ];

  console.log('   Processing batch of 3 samples...');
  batchInput.forEach((sample, i) => {
    const sampleVector = WasmTropicalVector.from_array(sample);
    const result = layer1Weights.tropical_multiply_vector(sampleVector);
    console.log(`   Sample ${i + 1}: [${sample.join(', ')}] → [${result.to_array().join(', ')}]`);
    sampleVector.free();
    result.free();
  });

  // Clean up WASM memory
  tropA.free();
  tropB.free();
  tropC.free();
  sum.free();
  product.free();
  power.free();
  inputVector.free();
  weights.free();
  output.free();
  distanceMatrix.free();
  paths2.free();
  paths3.free();
  layer1Input.free();
  layer1Weights.free();
  layer1Output.free();
  layer2Weights.free();
  finalOutput.free();

  console.log('\n✅ Tropical algebra example completed!');
}

// Run the example
runTropicalAlgebraExample().catch(console.error);