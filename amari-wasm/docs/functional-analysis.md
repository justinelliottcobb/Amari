# Functional Analysis (amari-functional)

*Added in v0.15.0*

Hilbert spaces, linear operators, spectral decomposition, and Sobolev spaces on multivector spaces.

## Quick Start

```typescript
import init, {
  WasmHilbertSpace,
  WasmMatrixOperator,
  WasmSpectralDecomposition,
  WasmSobolevSpace,
  powerMethod,
  computeEigenvalues
} from '@justinelliottcobb/amari-wasm';

async function functionalDemo() {
  await init();

  // Create a Hilbert space Cl(2,0,0) = R^4
  const hilbert = new WasmHilbertSpace();
  console.log(`Dimension: ${hilbert.dimension()}`); // 4

  // Create multivectors from coefficients [scalar, e1, e2, e12]
  const x = hilbert.fromCoefficients([1.0, 2.0, 3.0, 4.0]);
  const y = hilbert.fromCoefficients([0.5, 1.5, 2.5, 3.5]);

  // Inner product and norm
  console.log(`<x, y> = ${hilbert.innerProduct(x, y)}`);
  console.log(`||x|| = ${hilbert.norm(x)}`);

  // Create a symmetric matrix operator (4x4 row-major)
  const A = new WasmMatrixOperator([
    4, 1, 0, 0,
    1, 3, 1, 0,
    0, 1, 2, 1,
    0, 0, 1, 1
  ]);

  // Spectral decomposition
  const decomp = WasmSpectralDecomposition.compute(A, 100, 1e-10);
  console.log(`Eigenvalues: ${decomp.eigenvalues()}`);
  console.log(`Condition number: ${decomp.conditionNumber()}`);

  // Functional calculus: exp(A)x
  const expAx = decomp.applyFunction((lambda) => Math.exp(lambda), x);

  // Sobolev spaces for PDE analysis
  const h1 = new WasmSobolevSpace(1, 0.0, 1.0); // H^1([0,1])
  const f = (x) => Math.sin(Math.PI * x);
  const df = (x) => Math.PI * Math.cos(Math.PI * x);
  console.log(`||sin(pi*x)||_{H^1} = ${h1.h1Norm(f, df)}`);

  // Clean up
  hilbert.free(); A.free(); decomp.free(); h1.free();
}

functionalDemo();
```

## API Reference

### WasmHilbertSpace

- `new()`: Create Hilbert space Cl(2,0,0) = R^4
- `dimension()`: Get space dimension
- `signature()`: Get Clifford algebra signature [p, q, r]
- `fromCoefficients(coeffs)`: Create multivector from coefficients
- `innerProduct(x, y)`: Compute inner product
- `norm(x)`: Compute norm
- `distance(x, y)`: Compute distance
- `normalize(x)`: Normalize to unit length
- `project(x, y)`: Orthogonal projection of x onto y
- `isOrthogonal(x, y, tol)`: Check orthogonality

### WasmMatrixOperator

- `new(entries)`: Create from 16 entries (4x4 row-major)
- `identity()`: Create identity operator
- `zero()`: Create zero operator
- `diagonal(entries)`: Create diagonal matrix
- `scaling(lambda)`: Create lambda*I
- `apply(x)`: Apply operator T(x)
- `operatorNorm()`: Compute operator norm
- `isSymmetric(tol)`: Check symmetry
- `add(other)`: Add operators
- `compose(other)`: Compose operators (matrix multiply)
- `scale(lambda)`: Scale by scalar
- `transpose()`: Compute transpose
- `trace()`: Compute trace

### WasmSpectralDecomposition

- `compute(matrix, maxIter, tol)`: Compute eigenvalue decomposition
- `eigenvalues()`: Get eigenvalues
- `eigenvectors()`: Get eigenvectors (flattened)
- `isComplete()`: Check if decomposition is complete
- `spectralRadius()`: Get largest |eigenvalue|
- `conditionNumber()`: Get condition number
- `isPositiveDefinite()`: Check positive definiteness
- `apply(x)`: Apply reconstructed operator
- `applyFunction(f, x)`: Functional calculus f(T)x

### WasmSobolevSpace

- `new(order, lower, upper)`: Create H^k([a,b])
- `h1UnitInterval()`: Create H^1([0,1])
- `h2UnitInterval()`: Create H^2([0,1])
- `poincareConstant()`: Get Poincare constant
- `h1Norm(f, df)`: Compute H^1 norm
- `h1Seminorm(df)`: Compute H^1 seminorm
- `l2Norm(f)`: Compute L^2 norm
- `l2InnerProduct(f, g)`: Compute L^2 inner product

### Standalone Functions

- `powerMethod(matrix, initial, maxIter, tol)`: Dominant eigenvalue
- `inverseIteration(matrix, shift, initial, maxIter, tol)`: Eigenvalue near shift
- `computeEigenvalues(matrix, maxIter, tol)`: All eigenvalues

## Use Cases

- **Spectral Methods**: Eigenvalue problems and spectral decomposition
- **PDE Analysis**: Sobolev space norms for weak solutions
- **Quantum Mechanics**: Hilbert space operators
