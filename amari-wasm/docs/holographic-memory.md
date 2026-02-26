# Holographic Memory and Fusion (amari-fusion, amari-holographic)

*Added in v0.12.3*

Vector Symbolic Architecture (VSA) for associative memory with binding and bundling operations, plus TropicalDualClifford fusion for LLM evaluation.

## Quick Start

```typescript
import init, {
  WasmTropicalDualClifford,
  WasmHolographicMemory,
  WasmResonator
} from '@justinelliottcobb/amari-wasm';

async function holographicDemo() {
  await init();

  // Create random vectors for keys and values
  const key1 = WasmTropicalDualClifford.randomVector();
  const value1 = WasmTropicalDualClifford.randomVector();
  const key2 = WasmTropicalDualClifford.randomVector();
  const value2 = WasmTropicalDualClifford.randomVector();

  // Create holographic memory
  const memory = new WasmHolographicMemory();

  // Store associations
  memory.store(key1, value1);
  memory.store(key2, value2);

  // Retrieve with a key
  const result = memory.retrieve(key1);
  console.log(`Confidence: ${result.confidence()}`);
  console.log(`Similarity to original: ${result.value().similarity(value1)}`);

  // Check capacity
  const info = memory.capacityInfo();
  console.log(`Items stored: ${info.itemCount}`);
  console.log(`Estimated SNR: ${info.estimatedSnr}`);

  // Binding operations (key * value)
  const bound = key1.bind(value1);
  const unbound = bound.unbind(key1); // Recovers value1

  // Similarity computation
  const sim = key1.similarity(key2);
  console.log(`Key similarity: ${sim}`);

  // Resonator cleanup for noisy inputs
  const codebook = [key1, key2, value1, value2];
  const resonator = WasmResonator.new(codebook);
  const noisyInput = key1; // Add noise in practice
  const cleaned = resonator.cleanup(noisyInput);
  console.log(`Best match index: ${cleaned.bestMatchIndex()}`);

  // Clean up WASM memory
  key1.free(); value1.free(); key2.free(); value2.free();
  memory.free(); bound.free(); unbound.free(); resonator.free();
}

holographicDemo();
```

## API Reference

### TropicalDualClifford Operations

- `bind(other)`: Binding operation using geometric product
- `unbind(other)`: Inverse binding for retrieval
- `bundle(other, beta)`: Bundling operation for superposition
- `similarity(other)`: Compute normalized similarity
- `bindingIdentity()`: Get the identity element for binding
- `bindingInverse()`: Compute approximate inverse
- `randomVector()`: Create a random unit vector
- `normalizeToUnit()`: Normalize to unit magnitude

### HolographicMemory

- `store(key, value)`: Store a key-value association
- `storeBatch(pairs)`: Store multiple associations efficiently
- `retrieve(key)`: Retrieve value associated with key
- `capacityInfo()`: Get storage statistics (item count, SNR, capacity)
- `clear()`: Clear all stored associations

### Resonator

- `new(codebook)`: Create resonator with clean reference vectors
- `cleanup(input)`: Clean up noisy input to nearest codebook entry
- `cleanupWithIterations(input, maxIter)`: Iterative cleanup

## Use Cases

- **Symbolic AI**: Associative reasoning and concept binding
- **Cognitive Architectures**: Brain-inspired memory systems for AI agents
- **Embedding Retrieval**: Content-addressable semantic search
