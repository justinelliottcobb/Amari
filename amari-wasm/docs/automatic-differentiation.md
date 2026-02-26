# Automatic Differentiation (amari-dual)

Forward-mode automatic differentiation using dual numbers for exact derivatives.

## Quick Start

```typescript
import { WasmDualNumber } from '@justinelliottcobb/amari-wasm';

// Create dual number x = 3.0 + 1.0ε (value 3, derivative seed 1)
const x = new WasmDualNumber(3.0, 1.0);

// Compute f(x) = x^2 + sin(x)
const x_squared = x.mul(x);
const sin_x = x.sin();
const result = x_squared.add(sin_x);

console.log(`f(3) = ${result.value()}`);      // 9 + sin(3)
console.log(`f'(3) = ${result.derivative()}`); // 6 + cos(3)
```

## API Reference

### WasmDualNumber

- `WasmDualNumber(value, derivative)`: Create dual number
- `value()`: Get real part
- `derivative()`: Get derivative part
- `add(other)`: Addition
- `sub(other)`: Subtraction
- `mul(other)`: Multiplication (product rule)
- `div(other)`: Division (quotient rule)
- `pow(n)`: Power (chain rule)
- `exp()`: Exponential
- `ln()`: Natural logarithm
- `sqrt()`: Square root
- `sin()`: Sine
- `cos()`: Cosine
- `abs()`: Absolute value
- `sigmoid()`: Sigmoid activation
- `relu()`: ReLU activation

### WasmMultiDualNumber

Multi-variable forward-mode AD for computing gradients.

## Use Cases

- **Machine Learning**: Computing gradients for backpropagation
- **Optimization**: Gradient-based optimization methods
- **Scientific Computing**: Sensitivity analysis
