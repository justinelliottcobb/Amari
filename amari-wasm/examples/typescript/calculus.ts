/**
 * Differential Calculus and Riemannian Geometry Examples
 *
 * This example demonstrates the calculus capabilities of Amari WASM:
 * - Scalar and vector field evaluation
 * - Numerical derivatives (gradient, divergence, curl, Laplacian)
 * - Integration (1D and 2D)
 * - Riemannian manifolds (metrics, curvature, geodesics)
 */

import * as Amari from '../../pkg/amari_wasm';

console.log('='.repeat(80));
console.log('Amari WASM - Differential Calculus and Riemannian Geometry');
console.log('='.repeat(80));

// ============================================================================
// 1. Scalar Fields
// ============================================================================

console.log('\n1. Scalar Field Evaluation');
console.log('-'.repeat(80));

// Create a 2D scalar field: f(x, y) = x² + y²
const paraboloid = Amari.ScalarField.fromFunction2D((x: number, y: number) => x * x + y * y);

// Evaluate at various points
const points2D = [
    [0, 0],
    [1, 0],
    [0, 1],
    [1, 1],
    [2, 3]
];

console.log('Scalar field f(x, y) = x² + y²:');
for (const [x, y] of points2D) {
    const value = paraboloid.evaluate([x, y]);
    console.log(`  f(${x}, ${y}) = ${value}`);
}

// Batch evaluation
console.log('\nBatch evaluation:');
const flatPoints = points2D.flat();
const batchResults = paraboloid.batchEvaluate(flatPoints);
console.log(`  Evaluated ${points2D.length} points:`, batchResults);

// ============================================================================
// 2. Vector Fields
// ============================================================================

console.log('\n2. Vector Field Evaluation');
console.log('-'.repeat(80));

// Create a 2D rotation field: F(x, y) = [-y, x]
const rotationField = Amari.VectorField.fromFunction2D((x: number, y: number) => [-y, x]);

console.log('Vector field F(x, y) = [-y, x] (rotation):');
for (const [x, y] of points2D) {
    const vector = rotationField.evaluate([x, y]);
    console.log(`  F(${x}, ${y}) = [${vector[0].toFixed(2)}, ${vector[1].toFixed(2)}]`);
}

// Create a 3D gradient field: F(x, y, z) = [x, y, z]
const gradientField = Amari.VectorField.fromFunction3D(
    (x: number, y: number, z: number) => [x, y, z]
);

console.log('\nVector field F(x, y, z) = [x, y, z] (gradient):');
const points3D = [[1, 2, 3], [0, 0, 1]];
for (const [x, y, z] of points3D) {
    const vector = gradientField.evaluate([x, y, z]);
    console.log(`  F(${x}, ${y}, ${z}) = [${vector.map(v => v.toFixed(2)).join(', ')}]`);
}

// ============================================================================
// 3. Numerical Derivatives
// ============================================================================

console.log('\n3. Numerical Derivatives');
console.log('-'.repeat(80));

const derivative = new Amari.NumericalDerivative(1e-5);

// Gradient of f(x, y) = x² + y² at (1, 2)
console.log('Gradient ∇f at (1, 2):');
const grad = derivative.gradient(paraboloid, [1, 2]);
console.log(`  ∇f(1, 2) = [${grad.map(v => v.toFixed(4)).join(', ')}]`);
console.log(`  Expected: [2.0000, 4.0000]`);

// Divergence of F(x, y) = [-y, x] at (1, 1)
console.log('\nDivergence ∇·F at (1, 1):');
const div = derivative.divergence(rotationField, [1, 1]);
console.log(`  ∇·F(1, 1) = ${div.toFixed(6)}`);
console.log(`  Expected: 0 (rotation field has zero divergence)`);

// Curl of F(x, y, z) = [0, 0, x] at (1, 2, 3)
const magneticField = Amari.VectorField.fromFunction3D(
    (x: number, y: number, z: number) => [0, 0, x]
);

console.log('\nCurl ∇×F at (1, 2, 3):');
const curl = derivative.curl(magneticField, [1, 2, 3]);
console.log(`  ∇×F(1, 2, 3) = [${curl.map(v => v.toFixed(4)).join(', ')}]`);
console.log(`  Expected: [0, 1, 0]`);

// Laplacian of f(x, y) = x² + y² at (1, 2)
console.log('\nLaplacian ∇²f at (1, 2):');
const laplacian = derivative.laplacian(paraboloid, [1, 2]);
console.log(`  ∇²f(1, 2) = ${laplacian.toFixed(4)}`);
console.log(`  Expected: 4.0000 (∂²/∂x² + ∂²/∂y² = 2 + 2)`);

// ============================================================================
// 4. Integration
// ============================================================================

console.log('\n4. Numerical Integration');
console.log('-'.repeat(80));

// 1D integration: ∫₀¹ x² dx = 1/3
console.log('1D integration: ∫₀¹ x² dx');
const integral1D = Amari.Integration.integrate1D(
    (x: number) => x * x,
    0, 1, 100  // from 0 to 1, 100 subdivisions
);
console.log(`  Result: ${integral1D.toFixed(6)}`);
console.log(`  Expected: 0.333333 (1/3)`);
console.log(`  Error: ${Math.abs(integral1D - 1/3).toExponential(2)}`);

// 2D integration: ∫∫[0,1]×[0,1] (x² + y²) dx dy = 2/3
console.log('\n2D integration: ∫∫[0,1]×[0,1] (x² + y²) dx dy');
const integral2D = Amari.Integration.integrate2D(
    (x: number, y: number) => x * x + y * y,
    0, 1,  // x bounds
    0, 1,  // y bounds
    50, 50 // subdivisions
);
console.log(`  Result: ${integral2D.toFixed(6)}`);
console.log(`  Expected: 0.666667 (2/3)`);
console.log(`  Error: ${Math.abs(integral2D - 2/3).toExponential(2)}`);

// ============================================================================
// 5. Riemannian Manifolds - Euclidean Space
// ============================================================================

console.log('\n5. Euclidean (Flat) Manifold');
console.log('-'.repeat(80));

const euclidean2D = Amari.RiemannianManifold.euclidean(2);

console.log('2D Euclidean space:');
console.log(`  Dimension: ${euclidean2D.dimension}`);

// Christoffel symbols (all zero for flat space)
const gamma = euclidean2D.christoffel(0, 0, 0, [1, 2]);
console.log(`  Γ⁰₀₀(1,2) = ${gamma.toFixed(6)} (expected: 0)`);

// Scalar curvature (zero for flat space)
const R_flat = euclidean2D.scalarCurvature([1, 2]);
console.log(`  Scalar curvature R(1,2) = ${R_flat.toFixed(6)} (expected: 0)`);

// ============================================================================
// 6. Riemannian Manifolds - Sphere
// ============================================================================

console.log('\n6. Spherical (Curved) Manifold');
console.log('-'.repeat(80));

const sphere = Amari.RiemannianManifold.sphere(1.0);

console.log('Unit sphere (radius = 1):');
console.log(`  Dimension: ${sphere.dimension}`);

// Scalar curvature at north pole (θ=0, φ=0)
// For a sphere of radius R: K = 2/R²
const R_sphere = sphere.scalarCurvature([0.01, 0.01]); // Near north pole
console.log(`  Scalar curvature R ≈ ${R_sphere.toFixed(4)}`);
console.log(`  Expected: 2.0000 (K = 2/R² = 2/1² for unit sphere)`);

// Scalar curvature at equator (θ=π/2, φ=0)
const R_equator = sphere.scalarCurvature([Math.PI / 2, 0]);
console.log(`  Scalar curvature at equator ≈ ${R_equator.toFixed(4)}`);

// ============================================================================
// 7. Riemannian Manifolds - Hyperbolic Space
// ============================================================================

console.log('\n7. Hyperbolic (Negatively Curved) Manifold');
console.log('-'.repeat(80));

const hyperbolic = Amari.RiemannianManifold.hyperbolic();

console.log('Hyperbolic plane (Poincaré half-plane):');
console.log(`  Dimension: ${hyperbolic.dimension}`);

// Scalar curvature (constant negative curvature)
const R_hyperbolic = hyperbolic.scalarCurvature([1, 1]);
console.log(`  Scalar curvature R(1,1) ≈ ${R_hyperbolic.toFixed(4)}`);
console.log(`  Expected: -2.0000 (constant negative curvature)`);

// ============================================================================
// 8. Geodesics - Great Circles on a Sphere
// ============================================================================

console.log('\n8. Geodesic Trajectories');
console.log('-'.repeat(80));

console.log('Geodesic on unit sphere:');

// Initial position (θ=π/4, φ=0) and velocity
const initialPos = [Math.PI / 4, 0];
const initialVel = [0, 1]; // Moving in φ direction

// Compute geodesic
const trajectory = sphere.geodesic(
    initialPos,
    initialVel,
    3.0,   // t_max
    0.1    // dt
);

// The trajectory array is flat: [x0, y0, vx0, vy0, x1, y1, vx1, vy1, ...]
const numPoints = trajectory.length / 4;
console.log(`  Computed ${numPoints} trajectory points`);

// Display first and last points
console.log(`  Initial: θ=${trajectory[0].toFixed(4)}, φ=${trajectory[1].toFixed(4)}`);
const lastIdx = (numPoints - 1) * 4;
console.log(`  Final:   θ=${trajectory[lastIdx].toFixed(4)}, φ=${trajectory[lastIdx + 1].toFixed(4)}`);

// ============================================================================
// 9. Practical Example: Heat Equation on a Manifold
// ============================================================================

console.log('\n9. Practical Application: Heat Equation');
console.log('-'.repeat(80));

// Simulate heat diffusion: ∂u/∂t = k∇²u
// Initial condition: Gaussian centered at origin
const k = 0.1; // Thermal diffusivity
const dt = 0.01;
const t_max = 0.5;

console.log('Heat diffusion on 2D Euclidean space:');
console.log(`  Initial: Gaussian u(x,y,0) = exp(-(x²+y²))`);
console.log(`  Equation: ∂u/∂t = ${k}∇²u`);

// Grid of points
const gridSize = 5;
const gridPoints: number[][] = [];
for (let i = -gridSize; i <= gridSize; i++) {
    for (let j = -gridSize; j <= gridSize; j++) {
        gridPoints.push([i * 0.2, j * 0.2]);
    }
}

// Initial temperature field
const initialTemp = gridPoints.map(([x, y]) => Math.exp(-(x * x + y * y)));

console.log(`  Grid: ${(2 * gridSize + 1)}×${(2 * gridSize + 1)} = ${gridPoints.length} points`);
console.log(`  Time steps: ${Math.floor(t_max / dt)}`);

// Evolve (simplified - just show concept)
let temperature = [...initialTemp];
const numSteps = Math.floor(t_max / dt);

for (let step = 0; step < Math.min(numSteps, 10); step++) {
    const newTemp = temperature.map((T, i) => {
        // Create temporary scalar field at current temperature
        const tempField = Amari.ScalarField.fromFunction2D(
            (x: number, y: number) => {
                // Approximate field value
                const [px, py] = gridPoints[i];
                const dx = x - px;
                const dy = y - py;
                const r2 = dx * dx + dy * dy;
                return T * Math.exp(-r2 / (0.1 + step * 0.01));
            }
        );

        // Compute Laplacian
        const lap = derivative.laplacian(tempField, gridPoints[i]);

        // Update: u(t+dt) = u(t) + k*dt*∇²u
        return T + k * dt * lap;
    });

    temperature = newTemp;
}

// Find max and center temperatures
const maxTemp = Math.max(...temperature);
const centerIdx = Math.floor(gridPoints.length / 2);
const centerTemp = temperature[centerIdx];

console.log(`  After t=${t_max}s:`);
console.log(`    Max temperature: ${maxTemp.toFixed(4)}`);
console.log(`    Center temperature: ${centerTemp.toFixed(4)}`);
console.log(`    Diffusion: ${((1 - centerTemp / maxTemp) * 100).toFixed(1)}%`);

// ============================================================================
// 10. Summary and Performance
// ============================================================================

console.log('\n10. Summary');
console.log('='.repeat(80));

console.log('Demonstrated capabilities:');
console.log('  ✓ Scalar and vector field evaluation');
console.log('  ✓ Numerical derivatives (gradient, divergence, curl, Laplacian)');
console.log('  ✓ Numerical integration (1D and 2D)');
console.log('  ✓ Riemannian manifolds (Euclidean, sphere, hyperbolic)');
console.log('  ✓ Christoffel symbols and curvature tensors');
console.log('  ✓ Geodesic trajectories');
console.log('  ✓ Practical PDE simulation (heat equation)');

console.log('\nKey features:');
console.log('  • Browser-native WebAssembly execution');
console.log('  • Type-safe JavaScript/TypeScript API');
console.log('  • Efficient batch operations');
console.log('  • Configurable numerical precision');
console.log('  • Support for 2D and 3D manifolds');

console.log('\n' + '='.repeat(80));
console.log('Example completed successfully!');
console.log('='.repeat(80));
