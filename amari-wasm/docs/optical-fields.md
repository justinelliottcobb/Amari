# Optical Field Operations (amari-holographic)

*Added in v0.15.1*

GA-native Lee hologram encoding for DMD displays and VSA-based optical processing.

## Quick Start

```typescript
import init, {
  WasmOpticalRotorField,
  WasmBinaryHologram,
  WasmGeometricLeeEncoder,
  WasmOpticalFieldAlgebra,
  WasmOpticalCodebook,
  WasmTropicalOpticalAlgebra
} from '@justinelliottcobb/amari-wasm';

async function opticalDemo() {
  await init();

  // Create optical rotor fields (phase + amplitude on a grid)
  const field1 = WasmOpticalRotorField.random(256, 256, 12345n);
  const uniform = WasmOpticalRotorField.uniform(0.0, 0.5, 256, 256);

  // Lee hologram encoding for DMD display
  const encoder = WasmGeometricLeeEncoder.withFrequency(256, 256, 0.25);
  const hologram = encoder.encode(uniform);

  console.log(`Fill factor: ${hologram.fillFactor()}`);
  console.log(`Efficiency: ${encoder.theoreticalEfficiency(uniform)}`);

  // Get binary data for hardware interface
  const binaryData = hologram.asBytes();

  // VSA operations on optical fields
  const algebra = new WasmOpticalFieldAlgebra(256, 256);
  const bound = algebra.bind(field1, uniform);
  const similarity = algebra.similarity(field1, field1); // 1.0

  // Seed-based symbol codebook
  const codebook = new WasmOpticalCodebook(64, 64, 42n);
  codebook.register("AGENT");
  codebook.register("ACTION");
  const agentField = codebook.get("AGENT");

  // Clean up WASM memory
  field1.free(); uniform.free(); hologram.free();
  bound.free(); agentField.free();
}

opticalDemo();
```

## API Reference

### WasmOpticalRotorField

- `random(width, height, seed)`: Create random phase field
- `uniform(phase, amplitude, width, height)`: Uniform field
- `identity(width, height)`: Identity field (phase = 0)
- `fromPhase(phases, width, height)`: Create from phase array
- `phaseAt(x, y)`: Get phase at point (radians)
- `amplitudeAt(x, y)`: Get amplitude at point
- `totalEnergy()`: Sum of squared amplitudes
- `normalized()`: Normalized copy (energy = 1)

### WasmGeometricLeeEncoder

- `withFrequency(width, height, frequency)`: Create horizontal carrier encoder
- `new(width, height, frequency, angle)`: Create with angled carrier
- `encode(field)`: Encode to binary hologram
- `modulate(field)`: Get modulated field before thresholding
- `theoreticalEfficiency(field)`: Compute diffraction efficiency

### WasmBinaryHologram

- `get(x, y)`: Get pixel value
- `set(x, y, value)`: Set pixel value
- `fillFactor()`: Fraction of "on" pixels
- `hammingDistance(other)`: Compute Hamming distance
- `asBytes()`: Get packed binary data
- `inverted()`: Create inverted copy

### WasmOpticalFieldAlgebra

- `bind(a, b)`: Rotor multiplication (phase addition)
- `unbind(key, bound)`: Retrieve associated field
- `bundle(fields, weights)`: Weighted superposition
- `bundleUniform(fields)`: Equal-weight bundle
- `similarity(a, b)`: Normalized inner product
- `inverse(field)`: Phase negation
- `scale(field, factor)`: Amplitude scaling
- `addPhase(field, phase)`: Add constant phase

### WasmOpticalCodebook

- `new(width, height, baseSeed)`: Create codebook
- `register(symbol)`: Register symbol with auto-seed
- `get(symbol)`: Get or generate field for symbol
- `contains(symbol)`: Check if symbol is registered
- `symbols()`: Get all registered symbol names

### WasmTropicalOpticalAlgebra

- `tropicalAdd(a, b)`: Pointwise minimum phase magnitude
- `tropicalMax(a, b)`: Pointwise maximum phase magnitude
- `tropicalMul(a, b)`: Binding (phase addition)
- `softTropicalAdd(a, b, beta)`: Soft minimum with temperature
- `phaseDistance(a, b)`: Sum of absolute phase differences
- `attractorConverge(initial, attractors, maxIter, tol)`: Attractor dynamics

## Use Cases

- **Holographic Displays**: Lee hologram encoding for DMD and SLM devices
- **Optical Computing**: Phase-encoded VSA operations for optical neural networks
- **Symbol Processing**: Codebook-based optical symbol manipulation
