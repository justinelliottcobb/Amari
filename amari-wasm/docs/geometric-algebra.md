# Geometric Algebra (amari-core)

Geometric algebra extends linear algebra with the geometric product, enabling intuitive representation of rotations, reflections, and other transformations.

## Quick Start

```typescript
import init, { WasmMultivector, WasmRotor } from '@justinelliottcobb/amari-wasm';

async function main() {
  await init();

  // Create basis vectors
  const e1 = WasmMultivector.basis_vector(0);
  const e2 = WasmMultivector.basis_vector(1);

  // Compute geometric product
  const product = e1.geometric_product(e2);
  console.log(product.to_string()); // e12 (bivector)

  // Create a rotor for 90-degree rotation
  const rotor = WasmRotor.from_axis_angle(
    WasmMultivector.basis_vector(2), // z-axis
    Math.PI / 2
  );

  // Rotate a vector
  const vector = WasmMultivector.from_coefficients(
    new Float64Array([1, 0, 0, 0, 0, 0, 0, 0])
  );
  const rotated = rotor.rotate_vector(vector);

  // Clean up WASM memory
  e1.free();
  e2.free();
  product.free();
  rotor.free();
  vector.free();
  rotated.free();
}

main();
```

## Concepts

The WASM bindings expose Cl(3,0,0) — the 3D Euclidean Clifford algebra with 8 basis elements (2^3):

- **Scalar** (grade 0): 1 component
- **Vectors** (grade 1): e1, e2, e3
- **Bivectors** (grade 2): e12, e13, e23
- **Trivector** (grade 3): e123

Basis blade indexing uses bitset representation: e1=1, e2=2, e12=3, e3=4, e13=5, e23=6, e123=7.

## API Reference

### Multivector Operations

- `WasmMultivector.basis_vector(index)`: Create a basis vector (0, 1, or 2)
- `WasmMultivector.scalar(value)`: Create a scalar multivector
- `WasmMultivector.from_coefficients(array)`: Create from 8 coefficients
- `geometric_product(other)`: Compute geometric product
- `wedge_product(other)`: Compute wedge (outer) product
- `inner_product(other)`: Compute inner product
- `add(other)`: Add multivectors
- `grade(k)`: Extract grade-k component
- `norm()`: Compute magnitude
- `normalize()`: Normalize to unit length
- `inverse()`: Compute multiplicative inverse
- `reverse()`: Compute reverse (reversion)
- `exp()`: Compute exponential

### Rotor Operations

- `WasmRotor.from_axis_angle(axis, angle)`: Create rotation rotor
- `WasmRotor.from_bivector(bivector, angle)`: Create from bivector
- `rotate_vector(vector)`: Apply rotation to vector
- `compose(other)`: Compose rotations
- `inverse()`: Compute inverse rotation

### Batch Operations

- `BatchOperations.add(a, b)`: Batch add multivector arrays
- `BatchOperations.geometric_product(a, b)`: Batch geometric products

## Use Cases

- **Computer Graphics**: 3D rotations and transformations using rotors
- **Physics Simulations**: Electromagnetic fields and relativistic calculations
- **Robotics**: Rigid body kinematics and dynamics
- **Game Development**: Efficient spatial transformations
