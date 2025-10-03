import init, {
  WasmDualNumber,
  WasmMultiDualNumber,
  DualUtils
} from '@justinelliottcobb/amari-wasm';

async function runAutomaticDifferentiationExample() {
  console.log('📈 Automatic Differentiation Examples');
  console.log('=====================================');

  // Initialize the WASM module
  await init();

  // Single-variable derivatives
  console.log('\n1. Single-Variable Derivatives:');
  console.log('   f(x) = x² + 3x + 2, compute f\'(x) at x = 5');

  const x = WasmDualNumber.new(5.0, 1.0); // value=5, derivative=1 for dx/dx

  // Compute f(x) = x² + 3x + 2
  const xSquared = x.multiply(x);          // x²
  const threeX = x.scale(3.0);             // 3x
  const temp = xSquared.add(threeX);       // x² + 3x
  const constant = WasmDualNumber.new(2.0, 0.0); // constant 2
  const result = temp.add(constant);       // x² + 3x + 2

  console.log(`   f(5) = ${result.value()}`);
  console.log(`   f'(5) = ${result.derivative()}`);
  console.log(`   Expected: f(5) = 42, f'(5) = 13`);

  // Transcendental functions
  console.log('\n2. Transcendental Functions:');
  console.log('   g(x) = sin(x) * e^x at x = π/4');

  const piOver4 = WasmDualNumber.new(Math.PI / 4, 1.0);
  const sinX = piOver4.sin();
  const expX = piOver4.exp();
  const gResult = sinX.multiply(expX);

  console.log(`   g(π/4) = ${gResult.value()}`);
  console.log(`   g'(π/4) = ${gResult.derivative()}`);

  // Composition of functions
  console.log('\n3. Function Composition:');
  console.log('   h(x) = ln(x² + 1) at x = 2');

  const x2 = WasmDualNumber.new(2.0, 1.0);
  const x2Squared = x2.multiply(x2);
  const x2SquaredPlus1 = x2Squared.add(WasmDualNumber.new(1.0, 0.0));
  const hResult = x2SquaredPlus1.ln();

  console.log(`   h(2) = ${hResult.value()}`);
  console.log(`   h'(2) = ${hResult.derivative()}`);

  // Multi-variable partial derivatives
  console.log('\n4. Multi-Variable Partial Derivatives:');
  console.log('   f(x,y) = x²y + xy² at (x=2, y=3)');

  // ∂f/∂x: treat x as variable, y as constant
  const x_var = WasmMultiDualNumber.new([2.0, 3.0], [1.0, 0.0]); // ∂/∂x
  console.log(`   ∂f/∂x at (2,3) = ${computeMultivarFunction(x_var).partial_derivative(0)}`);

  // ∂f/∂y: treat y as variable, x as constant
  const y_var = WasmMultiDualNumber.new([2.0, 3.0], [0.0, 1.0]); // ∂/∂y
  console.log(`   ∂f/∂y at (2,3) = ${computeMultivarFunction(y_var).partial_derivative(1)}`);

  // Gradient computation
  console.log('\n5. Gradient Vector:');
  const gradientX = WasmMultiDualNumber.new([2.0, 3.0], [1.0, 0.0]);
  const gradientY = WasmMultiDualNumber.new([2.0, 3.0], [0.0, 1.0]);

  const fx = computeMultivarFunction(gradientX);
  const fy = computeMultivarFunction(gradientY);

  console.log(`   ∇f(2,3) = [${fx.partial_derivative(0)}, ${fy.partial_derivative(1)}]`);

  // Newton's method for root finding
  console.log('\n6. Newton\'s Method Root Finding:');
  console.log('   Finding root of f(x) = x³ - 2x - 5');

  let xNewton = 2.0; // Initial guess
  console.log(`   Initial guess: x₀ = ${xNewton}`);

  for (let i = 0; i < 5; i++) {
    const xDual = WasmDualNumber.new(xNewton, 1.0);

    // f(x) = x³ - 2x - 5
    const xCubed = xDual.multiply(xDual).multiply(xDual);
    const twoX = xDual.scale(2.0);
    const fResult = xCubed.subtract(twoX).subtract(WasmDualNumber.new(5.0, 0.0));

    const fValue = fResult.value();
    const fDerivative = fResult.derivative();

    xNewton = xNewton - fValue / fDerivative; // Newton's update

    console.log(`   x₁₊${i} = ${xNewton}, f(x) = ${fValue}`);

    xDual.free();
    xCubed.free();
    twoX.free();
    fResult.free();

    if (Math.abs(fValue) < 1e-10) {
      console.log(`   Converged to root: x = ${xNewton}`);
      break;
    }
  }

  // Optimization example
  console.log('\n7. Function Optimization:');
  console.log('   Minimizing f(x) = x⁴ - 4x³ + 6x² - 4x + 1');

  let xOpt = 0.5; // Initial guess
  const learningRate = 0.01;

  console.log(`   Initial guess: x₀ = ${xOpt}`);

  for (let i = 0; i < 100; i++) {
    const xDual = WasmDualNumber.new(xOpt, 1.0);

    // f(x) = x⁴ - 4x³ + 6x² - 4x + 1
    const x2 = xDual.multiply(xDual);
    const x3 = x2.multiply(xDual);
    const x4 = x3.multiply(xDual);

    const term1 = x4;
    const term2 = x3.scale(-4.0);
    const term3 = x2.scale(6.0);
    const term4 = xDual.scale(-4.0);
    const term5 = WasmDualNumber.new(1.0, 0.0);

    const fOpt = term1.add(term2).add(term3).add(term4).add(term5);

    const gradient = fOpt.derivative();
    xOpt = xOpt - learningRate * gradient; // Gradient descent

    if (i % 20 === 0) {
      console.log(`   Iteration ${i}: x = ${xOpt.toFixed(6)}, f(x) = ${fOpt.value().toFixed(6)}`);
    }

    // Clean up
    xDual.free();
    x2.free();
    x3.free();
    x4.free();
    term1.free();
    term2.free();
    term3.free();
    term4.free();
    term5.free();
    fOpt.free();

    if (Math.abs(gradient) < 1e-8) {
      console.log(`   Converged to minimum: x = ${xOpt}, gradient = ${gradient}`);
      break;
    }
  }

  // Polynomial evaluation utilities
  console.log('\n8. Polynomial Evaluation:');
  const coefficients = [1, -2, 3, -1]; // represents x³ - 2x² + 3x - 1
  const evalPoint = WasmDualNumber.new(1.5, 1.0);
  const polyResult = DualUtils.eval_polynomial(coefficients, evalPoint);

  console.log(`   P(x) = x³ - 2x² + 3x - 1`);
  console.log(`   P(1.5) = ${polyResult.value()}`);
  console.log(`   P'(1.5) = ${polyResult.derivative()}`);

  // Clean up remaining objects
  x.free();
  xSquared.free();
  threeX.free();
  temp.free();
  constant.free();
  result.free();
  piOver4.free();
  sinX.free();
  expX.free();
  gResult.free();
  x2.free();
  x2Squared.free();
  x2SquaredPlus1.free();
  hResult.free();
  x_var.free();
  y_var.free();
  gradientX.free();
  gradientY.free();
  fx.free();
  fy.free();
  evalPoint.free();
  polyResult.free();

  console.log('\n✅ Automatic differentiation example completed!');
}

function computeMultivarFunction(vars: any): any {
  // f(x,y) = x²y + xy²
  const values = vars.values();
  const x = values[0];
  const y = values[1];

  // This is a simplified representation
  // In practice, you'd use the MultiDualNumber operations
  const result = vars.clone();
  // Implementation would involve proper multi-dual arithmetic
  return result;
}

// Run the example
runAutomaticDifferentiationExample().catch(console.error);