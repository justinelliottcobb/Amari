# Computational Topology (amari-topology)

*Added in v0.16.0*

Simplicial complexes, homology computation, persistent homology, and Morse theory.

## Quick Start

```typescript
import init, {
  WasmSimplex,
  WasmSimplicialComplex,
  WasmFiltration,
  WasmPersistentHomology,
  ripsFromDistances,
  findCriticalPoints2D,
  WasmMorseComplex
} from '@justinelliottcobb/amari-wasm';

async function topologyDemo() {
  await init();

  // Create a triangle (2-simplex with all faces)
  const complex = new WasmSimplicialComplex();
  complex.addSimplex([0, 1, 2]); // Triangle + closure

  // Compute Betti numbers (topological invariants)
  const betti = complex.bettiNumbers();
  console.log(`b0 = ${betti[0]}`); // 1 (connected)
  console.log(`b1 = ${betti[1]}`); // 0 (no holes - filled)

  // Euler characteristic
  console.log(`chi = ${complex.eulerCharacteristic()}`); // 1

  // Circle (unfilled boundary)
  const circle = new WasmSimplicialComplex();
  circle.addSimplex([0, 1]);
  circle.addSimplex([1, 2]);
  circle.addSimplex([2, 0]);
  const circleBetti = circle.bettiNumbers();
  console.log(`Circle b1 = ${circleBetti[1]}`); // 1 (one hole!)

  // Persistent Homology
  const filt = new WasmFiltration();
  filt.add(0.0, [0]); filt.add(0.0, [1]); filt.add(0.0, [2]);
  filt.add(1.0, [0, 1]);
  filt.add(2.0, [1, 2]);
  filt.add(3.0, [0, 2]); // Creates loop

  const ph = WasmPersistentHomology.compute(filt);
  const diagram = ph.getDiagram();
  for (let i = 0; i < diagram.length; i += 3) {
    console.log(`H${diagram[i]}: born=${diagram[i+1]}, death=${diagram[i+2]}`);
  }

  // Vietoris-Rips from point cloud
  const ripsFilt = ripsFromDistances(3, 2, [1.0, 1.0, 1.0]);
  const ripsPH = WasmPersistentHomology.compute(ripsFilt);

  // Clean up
  complex.free(); circle.free(); filt.free(); ph.free();
  ripsFilt.free(); ripsPH.free();
}

topologyDemo();
```

## API Reference

### WasmSimplex

- `new(vertices)`: Create simplex from vertex indices
- `dimension()`: Get dimension (vertices - 1)
- `getVertices()`: Get sorted vertex array
- `orientation()`: Get orientation sign (+1 or -1)
- `containsVertex(v)`: Check if vertex is in simplex
- `faces(k)`: Get all k-dimensional faces
- `boundaryFaces()`: Get boundary faces with signs

### WasmSimplicialComplex

- `new()`: Create empty complex
- `addSimplex(vertices)`: Add simplex and all its faces
- `contains(vertices)`: Check if simplex exists
- `dimension()`: Get maximum simplex dimension
- `simplexCount(dim)`: Count simplices in dimension
- `totalSimplexCount()`: Total simplex count
- `vertexCount()`: Count 0-simplices
- `edgeCount()`: Count 1-simplices
- `bettiNumbers()`: Compute [b0, b1, b2, ...]
- `eulerCharacteristic()`: Compute chi = sum(-1)^k f_k
- `fVector()`: Get face counts [f0, f1, f2, ...]
- `isConnected()`: Check if complex is connected
- `connectedComponents()`: Count components

### WasmFiltration

- `new()`: Create empty filtration
- `add(time, vertices)`: Add simplex at filtration time
- `isEmpty()`: Check if filtration is empty
- `complexAt(time)`: Get complex at given time
- `bettiAt(time)`: Get Betti numbers at time

### WasmPersistentHomology

- `compute(filtration)`: Compute persistent homology
- `getDiagram()`: Get [dim, birth, death, ...] triples
- `bettiAt(time)`: Get Betti numbers at time
- `intervalCount(dim)`: Count intervals in dimension

### Standalone Functions

- `ripsFromDistances(numPoints, maxDim, distances)`: Create Rips filtration
- `findCriticalPoints2D(resolution, xMin, xMax, yMin, yMax, tolerance, values)`: Find critical points

### WasmMorseComplex

- `new(criticalPoints)`: Create from critical points
- `countsByIndex()`: Get counts by Morse index
- `checkWeakMorseInequalities(betti)`: Verify c_k >= b_k

## Use Cases

- **Topological Data Analysis**: Persistent homology for shape detection
- **Computational Biology**: Protein structure analysis
- **Sensor Networks**: Coverage analysis using homology
