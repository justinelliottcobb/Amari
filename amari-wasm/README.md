# @amari/core

🚀 **Advanced Mathematical Computing Library for JavaScript/TypeScript**

[![npm version](https://badge.fury.io/js/%40amari%2Fcore.svg)](https://www.npmjs.com/package/@amari/core)
[![CI](https://github.com/justinelliottcobb/Amari/actions/workflows/ci.yml/badge.svg)](https://github.com/justinelliottcobb/Amari/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Amari is a high-performance mathematical computing library that brings advanced algebraic structures to JavaScript/TypeScript through WebAssembly. It combines geometric algebra, tropical algebra, dual number automatic differentiation, and cellular automata in a unified framework.

## ✨ Features

- **🔢 Geometric Algebra (Clifford Algebra)**: Multivectors, rotors, and geometric products for 3D rotations and spatial transformations
- **🌴 Tropical Algebra**: Max-plus semiring operations for optimization and neural network applications
- **📈 Automatic Differentiation**: Forward-mode AD with dual numbers for exact derivatives
- **🔲 Cellular Automata**: Geometric cellular automata with multivector states
- **⚡ WebGPU Acceleration**: Optional GPU acceleration for large-scale operations
- **🦀 Pure Rust Implementation**: Memory-safe, high-performance core with WASM bindings
- **📦 TypeScript Support**: Full TypeScript definitions included

## 📦 Installation

```bash
npm install @amari/core
```

Or with yarn:

```bash
yarn add @amari/core
```

## 🚀 Quick Start

```typescript
import init, { WasmMultivector, WasmRotor } from '@amari/core';

async function main() {
  // Initialize the WASM module
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

## 📚 Core Concepts

### Geometric Algebra

Geometric algebra extends linear algebra with the geometric product, enabling intuitive representation of rotations, reflections, and other transformations:

```typescript
// Multivector operations
const v1 = WasmMultivector.from_coefficients(coeffs);
const v2 = WasmMultivector.random();

const sum = v1.add(v2);
const product = v1.geometric_product(v2);
const wedge = v1.wedge_product(v2);
const inner = v1.inner_product(v2);
```

### Tropical Algebra

Tropical algebra replaces addition with max and multiplication with addition, useful for optimization:

```typescript
import { tropical_add, tropical_multiply } from '@amari/core';

// Tropical operations: add = max, multiply = add
const a = 5.0, b = 3.0;
const trop_sum = tropical_add(a, b); // max(5, 3) = 5
const trop_prod = tropical_multiply(a, b); // 5 + 3 = 8
```

### Cellular Automata

Create and evolve cellular automata with geometric algebra states:

```typescript
const ca = WasmGeometricCA.new(100, 100);

// Set initial configuration
ca.set_cell(50, 50, WasmMultivector.basis_vector(0));

// Evolve the system
for (let i = 0; i < 100; i++) {
  ca.step();
}

console.log(`Generation: ${ca.generation()}`);
```

## 🎯 Use Cases

- **Computer Graphics**: 3D rotations and transformations using rotors
- **Physics Simulations**: Geometric algebra for electromagnetic fields
- **Machine Learning**: Tropical neural networks and automatic differentiation
- **Optimization**: Tropical algebra for shortest path and scheduling problems
- **Scientific Computing**: High-performance mathematical operations
- **Game Development**: Efficient spatial transformations and physics

## 🔧 API Reference

### Multivector Operations

- `WasmMultivector.basis_vector(index)`: Create a basis vector
- `WasmMultivector.scalar(value)`: Create a scalar multivector
- `WasmMultivector.from_coefficients(array)`: Create from coefficients
- `geometric_product(a, b)`: Compute geometric product
- `wedge_product(a, b)`: Compute wedge (outer) product
- `inner_product(a, b)`: Compute inner product

### Rotor Operations

- `WasmRotor.from_axis_angle(axis, angle)`: Create rotation rotor
- `WasmRotor.from_bivector(bivector, angle)`: Create from bivector
- `rotate_vector(vector)`: Apply rotation to vector
- `compose(other)`: Compose rotations

### Tropical Operations

- `tropical_add(a, b)`: Tropical addition (max)
- `tropical_multiply(a, b)`: Tropical multiplication (addition)
- `tropical_power(base, exp)`: Tropical exponentiation

## 🔍 Examples

Check out the [examples directory](https://github.com/justinelliottcobb/Amari/tree/master/examples) for more detailed usage:

- [Basic Geometric Algebra](https://github.com/justinelliottcobb/Amari/blob/master/examples/typescript/geometric.ts)
- [3D Rotations with Rotors](https://github.com/justinelliottcobb/Amari/blob/master/examples/typescript/rotations.ts)
- [Tropical Neural Networks](https://github.com/justinelliottcobb/Amari/blob/master/examples/typescript/tropical.ts)
- [Cellular Automata](https://github.com/justinelliottcobb/Amari/blob/master/examples/typescript/cellular.ts)

## 🏗️ Building from Source

```bash
# Clone the repository
git clone https://github.com/justinelliottcobb/Amari.git
cd Amari/amari-wasm

# Install dependencies
npm install

# Build WASM module
npm run build

# Run tests
npm test
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. See [CONTRIBUTING.md](https://github.com/justinelliottcobb/Amari/blob/master/CONTRIBUTING.md) for details.

## 📄 License

MIT License - see [LICENSE](https://github.com/justinelliottcobb/Amari/blob/master/LICENSE) for details.

## 🙏 Acknowledgments

- Built with Rust and wasm-bindgen
- Inspired by geometric algebra libraries like GAViewer and Ganja.js
- Tropical algebra concepts from discrete mathematics

## 📬 Contact

- GitHub: [@justinelliottcobb](https://github.com/justinelliottcobb)
- NPM: [@amari/core](https://www.npmjs.com/package/@amari/core)

---

**Made with ❤️ and 🦀 by the Amari team**