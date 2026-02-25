# Enumerative Geometry (amari-enumerative)

*Added in v0.18.1*

WDVV curve counting, equivariant localization, matroids, CSM classes, operadic composition, and stability conditions.

## API Reference

### Curve Counting

- `WasmWDVVEngine.new()`: Create WDVV engine for rational curve counting
- `WasmWDVVEngine.rationalCurveCount(degree)`: Compute N_d (rational curves through 3d-1 points)
- `WasmWDVVEngine.requiredPointCount(degree, genus)`: Required point count (3d+g-1)
- `WasmWDVVEngine.getTable()`: Get table of computed curve counts
- `WasmWDVVEngine.p1xp1Count(a, b)`: Rational curves on P^1 x P^1 of bidegree (a,b)
- `WasmWDVVEngine.p3Count(degree)`: Rational curves in P^3

### Equivariant Localization

- `WasmEquivariantLocalizer.new(k, n)`: Create localizer for Gr(k,n)
- `WasmEquivariantLocalizer.fixedPointCount()`: Count T-fixed points (= C(n,k))
- `WasmEquivariantLocalizer.localizedIntersection(classes)`: Intersection via localization

### Matroids

- `WasmMatroid.uniform(k, n)`: Create uniform matroid U_{k,n}
- `WasmMatroid.getRank()`: Get matroid rank
- `WasmMatroid.getNumBases()`: Get number of bases
- `WasmMatroid.dual()`: Compute dual matroid
- `WasmMatroid.deleteElement(e)`: Delete element
- `WasmMatroid.contractElement(e)`: Contract element

### CSM Classes

- `WasmCSMClass.ofSchubertCell(partition, k, n)`: CSM class of Schubert cell
- `WasmCSMClass.ofSchubertVariety(partition, k, n)`: CSM class of Schubert variety
- `WasmCSMClass.eulerCharacteristic()`: Get Euler characteristic

### Stability Conditions

- `WasmStabilityCondition.new(k, n, trust)`: Create stability condition
- `WasmStabilityCondition.phase(class)`: Compute phase of a class
- `WasmStabilityCondition.stableCount(namespace)`: Count stable objects

### Wall Crossing

- `WasmWallCrossingEngine.new(k, n)`: Create wall-crossing engine
- `WasmWallCrossingEngine.computeWalls(namespace)`: Find walls
- `WasmWallCrossingEngine.stableCountAt(namespace, trust)`: Stable count at trust level
- `WasmWallCrossingEngine.phaseDiagram(namespace)`: Generate phase diagram

### Operadic Composition

- `WasmComposableNamespace.new(namespace)`: Create composable namespace
- `WasmComposableNamespace.markOutput(capId)`: Mark output interface
- `WasmComposableNamespace.markInput(capId)`: Mark input interface
- `composeNamespaces(outer, inner)`: Compose along matching interfaces
- `compositionMultiplicity(outer, inner)`: Intersection number of interfaces
- `interfacesCompatible(outer, inner)`: Check interface compatibility

## Use Cases

- **Algebraic Geometry**: Rational curve counting, Schubert calculus, Gromov-Witten invariants
- **Combinatorics**: Matroid operations, Littlewood-Richardson coefficients
- **Access Control**: Geometric namespace/capability systems for secure multi-agent coordination
