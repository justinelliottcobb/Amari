# Amari v0.4.0 Examples

This directory contains comprehensive examples showcasing all mathematical systems in the Amari WebAssembly library.

## 🚀 Quick Start

```bash
# Install dependencies
npm install @justinelliottcobb/amari-wasm

# Run any example
npx ts-node examples/typescript/complete-demo.ts
```

## 📚 Example Files

### 1. Complete Demonstration (`complete-demo.ts`)
**🎯 Start here!** A comprehensive showcase that demonstrates all five mathematical systems working together:
- Physics simulation with geometric algebra
- Neural network optimization with tropical algebra
- Function optimization with automatic differentiation
- Machine learning evaluation with information geometry
- Advanced architectures with fusion systems

### 2. Geometric Algebra (`geometric.ts`)
**🔢 3D Mathematics & Physics**
- Basic multivector operations
- Geometric products and bivectors
- 3D rotations using rotors
- Electromagnetic field simulations
- Vector algebra in geometric algebra framework

### 3. Tropical Algebra (`tropical.ts`)
**🌴 Neural Networks & Optimization**
- Tropical arithmetic (max-plus algebra)
- Neural network layer simulation
- Shortest path problems
- Matrix-vector operations
- Batch processing for machine learning

### 4. Automatic Differentiation (`automatic-diff.ts`)
**📈 Optimization & Gradients**
- Single and multi-variable derivatives
- Function composition
- Newton's method for root finding
- Gradient descent optimization
- Polynomial evaluation

### 5. Information Geometry (`information-geometry.ts`)
**📊 Statistical Analysis & ML Metrics**
- Probability distribution analysis
- KL divergence and JS divergence
- Fisher information matrices
- Entropy and cross-entropy
- Mutual information
- Alpha connections

### 6. Fusion Systems (`fusion-systems.ts`)
**🔲 Advanced Neural Architectures**
- Tropical-dual-Clifford fusion operations
- Attention mechanisms
- Multi-modal neural networks
- Geometric transformations in fusion space
- Batch processing in fusion systems

## 🎯 Use Cases by Domain

### 🎮 Game Development
```typescript
// 3D rotations and transformations
const rotor = WasmRotor.from_axis_angle(axis, angle);
const rotated = rotor.rotate_vector(gameObject);
```

### 🧠 Machine Learning
```typescript
// Tropical neural networks
const output = tropicalMatrix.tropical_multiply_vector(input);

// Automatic differentiation for training
const loss = computeLoss(WasmDualNumber.new(weight, 1.0));
const gradient = loss.derivative();
```

### 🔬 Scientific Computing
```typescript
// Electromagnetic field calculations
const emField = electricField.wedge_product(magneticField);

// Statistical manifold analysis
const divergence = manifold.klDivergence(distribution1, distribution2);
```

### 🎨 Computer Graphics
```typescript
// Efficient spatial transformations
const transformed = rotor.rotate_vector(vertex);
const projected = camera.apply_projection(transformed);
```

### 🤖 AI Research
```typescript
// Fusion systems for multimodal learning
const attention = query.fusionMultiply(key);
const output = attention.apply_to_value(value);
```

## 📖 Learning Path

1. **Beginners**: Start with `complete-demo.ts` to see all systems in action
2. **Geometric Algebra**: Focus on `geometric.ts` for 3D mathematics
3. **Machine Learning**: Explore `tropical.ts` and `automatic-diff.ts`
4. **Statistics**: Study `information-geometry.ts` for statistical analysis
5. **Advanced**: Dive into `fusion-systems.ts` for cutting-edge architectures

## 🔧 API Patterns

All examples follow consistent patterns:

### Initialization
```typescript
import init, { WasmClass } from '@justinelliottcobb/amari-wasm';
await init(); // Always initialize WASM first
```

### Memory Management
```typescript
const object = WasmClass.new(params);
// ... use object
object.free(); // Always free WASM objects
```

### Error Handling
```typescript
try {
  const result = object.compute(params);
  console.log(result);
} catch (error) {
  console.error('Computation failed:', error);
}
```

## 🎛️ Configuration

### TypeScript Setup
```json
{
  "compilerOptions": {
    "target": "ES2020",
    "module": "ESNext",
    "moduleResolution": "node",
    "allowSyntheticDefaultImports": true
  }
}
```

### Web Environment
```html
<script type="module">
  import init from '@justinelliottcobb/amari-wasm';
  await init();
  // Your code here
</script>
```

### Node.js Environment
```javascript
import init from '@justinelliottcobb/amari-wasm';
// Works directly with ES modules
```

## 🚀 Performance Tips

1. **Batch Operations**: Process multiple items together when possible
2. **Memory Management**: Always call `.free()` on WASM objects
3. **Reuse Objects**: Create objects once and reuse them in loops
4. **Web Workers**: Use Web Workers for heavy computations
5. **SIMD**: Enable SIMD in your build environment for extra performance

## 🐛 Troubleshooting

### Common Issues

**WASM not initialized**
```typescript
// ❌ Wrong
const obj = WasmClass.new();

// ✅ Correct
await init();
const obj = WasmClass.new();
```

**Memory leaks**
```typescript
// ❌ Wrong
const result = obj1.multiply(obj2);
return result; // obj1, obj2, result not freed

// ✅ Correct
const result = obj1.multiply(obj2);
const value = result.value();
obj1.free();
obj2.free();
result.free();
return value;
```

**Type errors**
```typescript
// ❌ Wrong
const arr = [1, 2, 3];
obj.set_coefficients(arr);

// ✅ Correct
const arr = new Float64Array([1, 2, 3]);
obj.set_coefficients(arr);
```

## 📝 Contributing

Want to add more examples? Follow these guidelines:

1. **Clear Documentation**: Explain what the example demonstrates
2. **Memory Safety**: Always free WASM objects
3. **Error Handling**: Include try-catch blocks
4. **Performance**: Show efficient patterns
5. **Real-world Use**: Connect to practical applications

## 🔗 Additional Resources

- [Amari Core Documentation](https://github.com/justinelliottcobb/Amari/docs)
- [Geometric Algebra Tutorial](https://bivector.net/)
- [Tropical Algebra in ML](https://arxiv.org/abs/1805.07966)
- [Information Geometry](https://www.springer.com/gp/book/9781402083617)

---

**Happy Computing with Amari v0.4.0! 🚀**