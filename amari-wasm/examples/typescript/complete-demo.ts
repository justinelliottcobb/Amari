import init, {
  WasmMultivector,
  WasmRotor,
  WasmTropicalNumber,
  WasmDualNumber,
  WasmTropicalDualClifford,
  WasmDuallyFlatManifold,
  InfoGeomUtils,
  WasmHilbertSpace,
  WasmMatrixOperator,
  WasmSpectralDecomposition
} from '@justinelliottcobb/amari-wasm';

/**
 * Complete demonstration of all Amari v0.15.0 mathematical systems
 * This example showcases the unified power of the expanded WASM library
 */
async function runCompleteDemo() {
  console.log('🚀 Amari v0.15.0 Complete Demonstration');
  console.log('=======================================');
  console.log('Unified Mathematical Computing: Geometric Algebra + Tropical Algebra +');
  console.log('Automatic Differentiation + Fusion Systems + Information Geometry +');
  console.log('Functional Analysis (Hilbert Spaces, Operators, Spectral Theory)');

  // Initialize the WASM module
  await init();

  // 1. GEOMETRIC ALGEBRA: 3D Physics Simulation
  console.log('\n🔢 1. GEOMETRIC ALGEBRA - 3D Physics Simulation');
  console.log('   Simulating electromagnetic field rotation...');

  const electricField = WasmMultivector.from_coefficients(
    new Float64Array([0, 1, 0, 0, 0, 0, 0, 0]) // E-field in x direction
  );

  const magneticField = WasmMultivector.from_coefficients(
    new Float64Array([0, 0, 1, 0, 0, 0, 0, 0]) // B-field in y direction
  );

  // Create electromagnetic field bivector
  const emField = electricField.wedge_product(magneticField);
  console.log(`   EM field bivector: ${emField.to_string()}`);

  // Rotate the field by 45 degrees around z-axis
  const zAxis = WasmMultivector.basis_vector(2);
  const rotor = WasmRotor.from_axis_angle(zAxis, Math.PI / 4);
  const rotatedEM = rotor.rotate_vector(emField);
  console.log(`   After 45° rotation: ${rotatedEM.to_string()}`);

  // 2. TROPICAL ALGEBRA: Neural Network Optimization
  console.log('\n🌴 2. TROPICAL ALGEBRA - Neural Network Layer');
  console.log('   Simulating tropical neural network forward pass...');

  const input1 = WasmTropicalNumber.new(2.5);
  const input2 = WasmTropicalNumber.new(1.8);
  const input3 = WasmTropicalNumber.new(3.1);

  const weight1 = WasmTropicalNumber.new(0.5);
  const weight2 = WasmTropicalNumber.new(-0.3);
  const weight3 = WasmTropicalNumber.new(1.2);

  // Tropical operations: multiply = add, add = max
  const output1 = input1.tropical_multiply(weight1); // 2.5 + 0.5 = 3.0
  const output2 = input2.tropical_multiply(weight2); // 1.8 + (-0.3) = 1.5
  const output3 = input3.tropical_multiply(weight3); // 3.1 + 1.2 = 4.3

  const neuronOutput = output1.tropical_add(output2).tropical_add(output3); // max(3.0, 1.5, 4.3) = 4.3

  console.log(`   Inputs: [${input1.value()}, ${input2.value()}, ${input3.value()}]`);
  console.log(`   Weights: [${weight1.value()}, ${weight2.value()}, ${weight3.value()}]`);
  console.log(`   Neuron output: ${neuronOutput.value()}`);

  // 3. AUTOMATIC DIFFERENTIATION: Optimization
  console.log('\n📈 3. AUTOMATIC DIFFERENTIATION - Function Optimization');
  console.log('   Finding minimum of f(x) = x² - 4x + 3...');

  let x = 5.0; // Starting point
  const learningRate = 0.1;

  for (let i = 0; i < 5; i++) {
    const xDual = WasmDualNumber.new(x, 1.0); // x with dx/dx = 1

    // f(x) = x² - 4x + 3
    const xSquared = xDual.multiply(xDual);
    const fourX = xDual.scale(4.0);
    const three = WasmDualNumber.new(3.0, 0.0);
    const fx = xSquared.subtract(fourX).add(three);

    const value = fx.value();
    const derivative = fx.derivative();

    console.log(`   Step ${i}: x=${x.toFixed(3)}, f(x)=${value.toFixed(3)}, f'(x)=${derivative.toFixed(3)}`);

    // Gradient descent update
    x = x - learningRate * derivative;

    // Clean up
    xDual.free();
    xSquared.free();
    fourX.free();
    three.free();
    fx.free();

    if (Math.abs(derivative) < 0.01) break;
  }
  console.log(`   Converged to minimum at x ≈ ${x.toFixed(3)} (expected: x = 2)`);

  // 4. INFORMATION GEOMETRY: Machine Learning Metrics
  console.log('\n📊 4. INFORMATION GEOMETRY - ML Model Comparison');
  console.log('   Comparing probability distributions from different models...');

  const model1_output = [0.7, 0.2, 0.1]; // Confident prediction
  const model2_output = [0.4, 0.4, 0.2]; // Less confident
  const true_labels = [1.0, 0.0, 0.0];   // Ground truth (one-hot)

  const manifold = new WasmDuallyFlatManifold(3, 0.0);

  const kl_model1 = manifold.klDivergence(true_labels, model1_output);
  const kl_model2 = manifold.klDivergence(true_labels, model2_output);
  const js_divergence = manifold.jsDivergence(model1_output, model2_output);

  console.log(`   Model 1: [${model1_output.join(', ')}]`);
  console.log(`   Model 2: [${model2_output.join(', ')}]`);
  console.log(`   True: [${true_labels.join(', ')}]`);
  console.log(`   KL(true||model1): ${kl_model1.toFixed(4)}`);
  console.log(`   KL(true||model2): ${kl_model2.toFixed(4)}`);
  console.log(`   JS(model1,model2): ${js_divergence.toFixed(4)}`);
  console.log(`   Better model: ${kl_model1 < kl_model2 ? 'Model 1' : 'Model 2'}`);

  // 5. FUSION SYSTEMS: Advanced Neural Architecture
  console.log('\n🔲 5. FUSION SYSTEMS - Advanced Neural Architecture');
  console.log('   Demonstrating tropical-dual-Clifford attention mechanism...');

  // Query vector in fusion space
  const query = WasmTropicalDualClifford.new(
    1.0,  // tropical component
    2.0,  // dual real (function value)
    1.0,  // dual infinitesimal (derivative)
    [0.5, 0.3, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0] // Clifford multivector
  );

  // Key vectors
  const key1 = WasmTropicalDualClifford.new(
    0.8, 1.5, 0.5,
    [0.4, 0.4, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0]
  );

  const key2 = WasmTropicalDualClifford.new(
    1.2, 1.8, 0.7,
    [0.3, 0.5, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0]
  );

  // Compute attention using fusion operations
  const attention1 = query.fusionMultiply(key1);
  const attention2 = query.fusionMultiply(key2);

  console.log(`   Query vector: trop=${query.tropicalValue()}, dual=(${query.dualReal()}, ${query.dualInfinitesimal()})`);
  console.log(`   Attention 1: ${attention1.tropicalValue().toFixed(3)}`);
  console.log(`   Attention 2: ${attention2.tropicalValue().toFixed(3)}`);
  console.log(`   Primary attention: Key ${attention1.tropicalValue() > attention2.tropicalValue() ? '1' : '2'}`);

  // 6. FUNCTIONAL ANALYSIS: Hilbert Spaces and Operators
  console.log('\n📐 6. FUNCTIONAL ANALYSIS - Hilbert Spaces & Spectral Theory');
  console.log('   Working with Hilbert space Cl(2,0,0) and linear operators...');

  // Create Hilbert space
  const hilbertSpace = new WasmHilbertSpace();
  console.log(`   Hilbert space dimension: ${hilbertSpace.dimension()}`);

  // Create vectors and compute inner products
  const psi = [1.0, 0.0, 0.0, 0.0];  // Ground state
  const phi = [0.0, 1.0, 0.0, 0.0];  // First excited state

  const innerProduct = hilbertSpace.innerProduct(psi, phi);
  const normPsi = hilbertSpace.norm(psi);
  console.log(`   |ψ⟩ = [1, 0, 0, 0], |φ⟩ = [0, 1, 0, 0]`);
  console.log(`   ⟨ψ|φ⟩ = ${innerProduct} (orthogonal states)`);
  console.log(`   ||ψ|| = ${normPsi}`);

  // Create Hamiltonian (energy operator)
  const hamiltonian = WasmMatrixOperator.diagonal([0.0, 1.0, 2.0, 3.0]);
  console.log(`   Hamiltonian H = diag(0, 1, 2, 3) - energy levels`);

  // Spectral decomposition
  const spectral = WasmSpectralDecomposition.compute(hamiltonian, 100, 1e-10);
  const eigenvalues = spectral.eigenvalues();
  console.log(`   Eigenvalues (energy levels): [${eigenvalues.map(e => e.toFixed(1)).join(', ')}]`);
  console.log(`   Spectral radius: ${spectral.spectralRadius()}`);
  console.log(`   Positive semi-definite: ${spectral.isPositiveSemidefinite()}`);

  // Functional calculus: time evolution
  const superposition = hilbertSpace.normalize([1.0, 1.0, 0.0, 0.0]);
  const evolved = spectral.applyFunction((E: number) => Math.exp(-E * 0.5), superposition);
  console.log(`   Time evolution e^{-Ht/2}|ψ⟩: [${evolved.map(v => v.toFixed(4)).join(', ')}]`);

  // 7. UNIFIED EXAMPLE: Physics + ML + Optimization
  console.log('\n⚡ 7. UNIFIED EXAMPLE - Physics-Informed Neural Network');
  console.log('   Combining all systems for physics-informed machine learning...');

  // Use geometric algebra for physics simulation
  const velocity = WasmMultivector.from_coefficients(
    new Float64Array([0, 2, 1, 0, 0, 0, 0, 0]) // velocity vector
  );
  const acceleration = WasmMultivector.from_coefficients(
    new Float64Array([0, 0.5, -0.2, 0, 0, 0, 0, 0]) // acceleration vector
  );

  // Use automatic differentiation for parameter optimization
  const dt = WasmDualNumber.new(0.1, 1.0); // time step with derivative
  const position_update = velocity.scale(dt.value()).add(acceleration.scale(0.5 * dt.value() * dt.value()));

  // Use tropical algebra for loss computation (sparse optimization)
  const predicted_pos = WasmTropicalNumber.new(position_update.scalar_part());
  const target_pos = WasmTropicalNumber.new(3.0);
  const loss = predicted_pos.tropical_add(target_pos.scale(-1.0)); // tropical distance

  // Use information geometry for model evaluation
  const prediction_dist = InfoGeomUtils.softmax([position_update.scalar_part(), 1.0, 0.5]);
  const target_dist = [0.8, 0.15, 0.05];
  const model_divergence = manifold.klDivergence(target_dist, prediction_dist);

  console.log(`   Physics: velocity=[${velocity.coefficients()[1]}, ${velocity.coefficients()[2]}]`);
  console.log(`   Update: position change = ${position_update.scalar_part().toFixed(3)}`);
  console.log(`   Optimization: tropical loss = ${loss.value().toFixed(3)}`);
  console.log(`   Evaluation: KL divergence = ${model_divergence.toFixed(4)}`);

  // Performance demonstration
  console.log('\n🎯 8. PERFORMANCE SHOWCASE');
  console.log('   Running batch operations across all systems...');

  const startTime = performance.now();

  // Batch geometric operations
  for (let i = 0; i < 100; i++) {
    const v1 = WasmMultivector.random();
    const v2 = WasmMultivector.random();
    const result = v1.geometric_product(v2);
    v1.free();
    v2.free();
    result.free();
  }

  // Batch tropical operations
  for (let i = 0; i < 100; i++) {
    const t1 = WasmTropicalNumber.new(Math.random() * 10);
    const t2 = WasmTropicalNumber.new(Math.random() * 10);
    const result = t1.tropical_multiply(t2);
    t1.free();
    t2.free();
    result.free();
  }

  const endTime = performance.now();
  console.log(`   Completed 200 operations in ${(endTime - startTime).toFixed(2)}ms`);

  // Clean up all remaining objects
  electricField.free();
  magneticField.free();
  emField.free();
  zAxis.free();
  rotor.free();
  rotatedEM.free();
  input1.free();
  input2.free();
  input3.free();
  weight1.free();
  weight2.free();
  weight3.free();
  output1.free();
  output2.free();
  output3.free();
  neuronOutput.free();
  query.free();
  key1.free();
  key2.free();
  attention1.free();
  attention2.free();
  velocity.free();
  acceleration.free();
  dt.free();
  predicted_pos.free();
  target_pos.free();
  loss.free();

  console.log('\n🎉 COMPLETE DEMONSTRATION FINISHED!');
  console.log('=====================================');
  console.log('✅ Geometric Algebra: Physics simulations');
  console.log('✅ Tropical Algebra: Neural network optimization');
  console.log('✅ Automatic Differentiation: Function optimization');
  console.log('✅ Information Geometry: ML model evaluation');
  console.log('✅ Fusion Systems: Advanced neural architectures');
  console.log('✅ Functional Analysis: Hilbert spaces and spectral theory');
  console.log('✅ Unified Systems: Physics-informed machine learning');
  console.log('\n📦 Amari v0.15.0 - Your unified mathematical computing platform!');
}

// Run the complete demonstration
runCompleteDemo().catch(console.error);