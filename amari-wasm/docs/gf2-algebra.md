# GF(2) Algebra & Coding Theory (amari-core, amari-enumerative)

*Added in v0.19.0*

Finite field GF(2) arithmetic, linear algebra, binary Clifford algebra, coding theory, matroid representability, and enumerative combinatorics over finite fields.

## Overview

The GF(2) WASM bindings provide:
- **Linear algebra over GF(2)**: bit-packed vectors and matrices with Gaussian elimination, rank, null space, and linear system solving
- **Binary Clifford algebra**: Cl(N,R;F₂) with geometric, outer, and inner products
- **Coding theory**: Binary linear codes (Hamming, Reed-Muller, Golay), weight enumerators, MacWilliams duality, and coding bounds
- **Grassmannian combinatorics**: Gaussian binomials, subspace enumeration, Schubert cells
- **Finite field point counting**: Grassmannian and Schubert variety point counts over F_q
- **Matroid representability**: Testing representability over GF(2), GF(3), GF(q), with Fano matroid construction
- **Kazhdan-Lusztig polynomials**: KL and Z-polynomials for matroids

## Quick Start

```typescript
import init, {
  WasmGF2Vector,
  WasmGF2Matrix,
  WasmBinaryMultivector,
  WasmBinaryCode,
  GF2Grassmannian,
  GF2FiniteField,
  GF2Representability,
  GF2KazhdanLusztig
} from '@justinelliottcobb/amari-wasm';

async function gf2Demo() {
  await init();

  // --- GF(2) Linear Algebra ---

  // Create vectors over GF(2)
  const v1 = WasmGF2Vector.fromBits(new Uint8Array([1, 0, 1, 1]));
  const v2 = WasmGF2Vector.fromBits(new Uint8Array([1, 1, 0, 1]));
  console.log(`v1 weight: ${v1.weight()}`);           // 3
  console.log(`Hamming distance: ${v1.hammingDistance(v2)}`); // 2
  console.log(`dot product: ${v1.dot(v2)}`);           // 0

  const sum = v1.add(v2); // XOR: [0, 1, 1, 0]
  console.log(`v1 + v2 = ${sum.toString()}`);

  // Create and manipulate matrices over GF(2)
  const matrix = WasmGF2Matrix.fromRows(
    new Uint8Array([1, 0, 1, 0, 1, 1]), 2, 3
  );
  console.log(`rank: ${matrix.rank()}`);              // 2

  // Solve Ax = b
  const b = WasmGF2Vector.fromBits(new Uint8Array([1, 1]));
  const x = matrix.solve(b);
  if (x) console.log(`solution: ${x.toString()}`);

  // Null space basis
  const nullSpace = matrix.nullSpace(); // [[1, 1, 1]]

  // --- Binary Clifford Algebra ---

  // Cl(3,0; F₂): 3 non-degenerate generators
  const e1 = WasmBinaryMultivector.basisVector(3, 0, 0);
  const e2 = WasmBinaryMultivector.basisVector(3, 0, 1);
  const e12 = e1.geometricProduct(e2);
  console.log(`e1 * e2 = ${e12.toString()}`);         // e12

  // Over GF(2): eᵢ² = 1
  const e1sq = e1.geometricProduct(e1);
  console.log(`e1² = ${e1sq.toString()}`);             // 1

  // Degenerate algebra Cl(2,1; F₂): e3² = 0
  const e3 = WasmBinaryMultivector.basisVector(2, 1, 2);
  const e3sq = e3.geometricProduct(e3);
  console.log(`e3² = ${e3sq.toString()}`);             // 0

  // --- Coding Theory ---

  // Hamming code [7, 4, 3]
  const hamming = WasmBinaryCode.hammingCode(3);
  console.log(`parameters: [${hamming.parameters()}]`); // [7, 4, 3]

  // Encode a message
  const message = new Uint8Array([1, 0, 1, 0]);
  const codeword = hamming.encode(message);
  console.log(`codeword: [${codeword}]`);

  // Verify: syndrome of valid codeword is zero
  const syn = hamming.syndrome(codeword);
  console.log(`syndrome: [${syn}]`);                   // all zeros

  // Weight distribution
  const weights = hamming.weightDistribution();
  console.log(`weight distribution: [${weights}]`);

  // Other standard codes
  const golay = WasmBinaryCode.extendedGolayCode();    // [24, 12, 8]
  const rm = WasmBinaryCode.reedMullerCode(1, 3);      // RM(1,3) = [8, 4, 4]
  const simplex = WasmBinaryCode.simplexCode(3);        // [7, 3, 4]

  // Coding bounds
  console.log(`Singleton bound: ${WasmBinaryCode.singletonBound(7, 4)}`);
  console.log(`Hamming bound: ${WasmBinaryCode.hammingBound(7, 4)}`);

  // --- Grassmannian Combinatorics ---

  // Gaussian binomial [4 choose 2]_2 = 35
  console.log(GF2Grassmannian.gaussianBinomial(4, 2, 2)); // 35

  // Number of lines in PG(2, F₂) = 7
  console.log(GF2Grassmannian.binaryGrassmannianSize(1, 3)); // 7

  // Enumerate all 1-dimensional subspaces of F₂³
  const subspaces = GF2Grassmannian.enumerateSubspaces(1, 3);
  console.log(`${subspaces.length} subspaces`);        // 7

  // --- Finite Field Point Counting ---

  // |Gr(2, 4; F₃)| (Grassmannian over F₃)
  console.log(GF2FiniteField.grassmannianPoints(2, 4, 3)); // 130

  // Poincaré polynomial of Gr(2, 4)
  const poincare = GF2FiniteField.grassmannianPoincarePolynomial(2, 4);

  // --- Matroid Representability ---

  // Fano matroid (7-point Fano plane, rank 3)
  const fano = GF2Representability.fanoMatroid();

  // Test binary representability
  const result = GF2Representability.isBinary(fano);
  console.log(`Fano is binary: ${result.status}`);     // "representable"
  if (result.matrix) {
    console.log(`Representation matrix:`, result.matrix);
  }

  // Fano matroid is binary but NOT ternary
  const ternaryResult = GF2Representability.isTernary(fano);
  console.log(`Fano is ternary: ${ternaryResult.status}`); // "not_representable"

  // --- Kazhdan-Lusztig Polynomials ---

  // KL polynomial of a uniform matroid
  // (Construct a matroid via the enumerative module's WasmMatroid)
  const klCoeffs = GF2KazhdanLusztig.klPolynomial(fano);
  console.log(`KL polynomial: [${klCoeffs}]`);

  // Non-negativity conjecture check
  console.log(`KL non-negative: ${GF2KazhdanLusztig.klIsNonNegative(fano)}`);

  // Clean up WASM memory
  v1.free(); v2.free(); sum.free();
  matrix.free(); b.free();
  e1.free(); e2.free(); e12.free();
  hamming.free(); golay.free();
  fano.free();
}

gf2Demo();
```

## API Reference

### WasmGF2Vector

Bit-packed vectors in F₂ⁿ with XOR/AND arithmetic.

- `new(dim)`: Create zero vector
- `fromBits(bits)`: Create from Uint8Array of 0/1 values
- `get(i)` / `set(i, value)`: Element access (0 or 1)
- `dim()`: Vector dimension
- `isZero()`: Check if zero
- `weight()`: Hamming weight (number of 1s)
- `hammingDistance(other)`: Hamming distance
- `dot(other)`: Dot product (returns 0 or 1)
- `add(other)`: XOR addition
- `toBits()`: Extract as Uint8Array
- `toString()`: String representation

### WasmGF2Matrix

Matrices over GF(2) with Gaussian elimination and linear algebra.

- `new(nrows, ncols)`: Create zero matrix
- `identity(n)`: Create identity matrix
- `fromRows(data, nrows, ncols)`: Create from flat row-major Uint8Array
- `get(row, col)` / `set(row, col, value)`: Element access
- `nrows()` / `ncols()`: Dimensions
- `mulVec(v)`: Matrix-vector product
- `mulMat(other)`: Matrix-matrix product
- `transpose()`: Transpose
- `rank()`: Rank (number of linearly independent rows)
- `determinant()`: Determinant (square matrices only, returns 0 or 1)
- `nullSpace()`: Null space basis as array of arrays
- `columnSpace()`: Column space basis as array of arrays
- `solve(b)`: Solve Ax = b (returns null if no solution)
- `reducedRowEchelon()`: In-place RREF, returns pivot indices
- `toFlatArray()`: Row-major flat representation
- `toString()`: String representation

### WasmBinaryMultivector

Clifford algebra Cl(N, R; F₂) with runtime signature dispatch.

Supported signatures: Cl(2,0), Cl(3,0), Cl(4,0), Cl(2,1), Cl(3,1).

- `new(n, r)`: Create zero multivector
- `one(n, r)`: Scalar identity
- `basisVector(n, r, i)`: Basis vector e_{i+1} (0-indexed)
- `basisBlade(n, r, index)`: Basis blade by index
- `fromBits(bits, n, r)`: From coefficient array
- `get(index)` / `set(index, value)`: Coefficient access
- `geometricProduct(other)`: Clifford product
- `outerProduct(other)`: Wedge product
- `innerProduct(other)`: Left contraction
- `add(other)`: XOR addition
- `gradeProjection(grade)`: Keep only blades of given grade
- `reverse()`: Reversal operator
- `isZero()`: Check if zero
- `weight()`: Number of nonzero coefficients
- `grade()`: Highest grade present
- `basisCount()`: Total basis blades (2^(N+R))
- `n` / `r`: Signature parameters (getters)
- `toString()`: String representation

### WasmBinaryCode

Binary linear codes [n, k, d] with encoding, error detection, and weight analysis.

**Constructors:**
- `fromGenerator(data, nrows, ncols)`: From generator matrix (flat Uint8Array)
- `fromParityCheck(data, nrows, ncols)`: From parity check matrix
- `hammingCode(r)`: Hamming code [2^r-1, 2^r-r-1, 3]
- `simplexCode(r)`: Simplex code [2^r-1, r, 2^(r-1)]
- `reedMullerCode(r, m)`: Reed-Muller code RM(r, m)
- `extendedGolayCode()`: Extended Golay code [24, 12, 8]

**Properties:**
- `length()`: Code length n
- `dimension()`: Code dimension k
- `minimumDistance()`: Minimum distance d
- `parameters()`: Returns [n, k, d]
- `isSelfDual()`: Check if C = C⊥

**Operations:**
- `encode(message)`: Encode k-bit message to n-bit codeword
- `syndrome(received)`: Compute error syndrome
- `dual()`: Dual code C⊥
- `weightEnumerator()`: Weight enumerator polynomial coefficients
- `weightDistribution()`: Weight distribution [A_0, ..., A_n]
- `generatorMatrix()`: Generator matrix as WasmGF2Matrix

**Coding Bounds (static):**
- `singletonBound(n, k)`: d ≤ n - k + 1
- `hammingBound(n, k)`: Sphere-packing bound (packing radius)
- `plotkinBound(n, k)`: Returns -1 if not applicable
- `gilbertVarshamovBound(n, k)`: Existence bound

### GF2Grassmannian

Grassmannian combinatorics (static methods).

- `gaussianBinomial(n, k, q)`: Gaussian binomial coefficient [n choose k]_q
- `binaryGrassmannianSize(k, n)`: |Gr(k, n; F₂)|
- `enumerateSubspaces(k, n)`: All k-dimensional subspaces of F₂ⁿ (n ≤ 20)
- `schubertCellOf(matrix)`: Schubert cell partition of an RREF subspace
- `schubertCellSize(partition)`: Size of a Schubert cell (2^|λ|)

### GF2FiniteField

Finite field point counting for algebraic varieties.

- `grassmannianPoints(k, n, q)`: |Gr(k, n; F_q)|
- `schubertCellPoints(partition, q)`: |C_λ(F_q)| = q^|λ|
- `schubertVarietyPoints(partition, k, n, q)`: |X_λ(F_q)|
- `grassmannianPoincarePolynomial(k, n)`: Poincaré polynomial coefficients
- `schubertPoincarePolynomial(partition, k, n)`: Schubert variety Poincaré polynomial

### GF2Representability

Matroid representability testing over finite fields.

- `isBinary(matroid)`: Test GF(2) representability → `{status, matrix?}`
- `isTernary(matroid)`: Test GF(3) representability → `{status, matrix?}`
- `isRepresentableOverGFq(matroid, q)`: Test GF(q) representability
- `isRegular(matroid)`: Representable over every field
- `hasMinor(matroid, minor)`: Minor containment check
- `fanoMatroid()`: Construct the Fano matroid (PG(2,2))
- `dualFanoMatroid()`: Construct the dual Fano matroid
- `columnMatroid(matrix)`: Column matroid of a GF(2) matrix
- `standardRepresentation(matroid)`: Find a GF(2) representation matrix

**Return format for representability tests:**
```typescript
{ status: "representable", matrix: [[0,1,...], ...] }
{ status: "not_representable" }
{ status: "inconclusive", reason: "..." }
```

### GF2KazhdanLusztig

Kazhdan-Lusztig polynomials and related invariants for matroids.

- `klPolynomial(matroid)`: KL polynomial coefficients [a_0, a_1, ...]
- `zPolynomial(matroid)`: Z-polynomial coefficients
- `inverseKlPolynomial(matroid)`: Inverse KL polynomial
- `klIsNonNegative(matroid)`: Check non-negativity of coefficients
- `schubertKlPolynomial(partition, k, n)`: KL polynomial for Schubert variety in Gr(k,n)

## Use Cases

- **Error-Correcting Codes**: Construct and analyze binary linear codes for communication systems
- **Cryptography**: GF(2) matrix operations for linear cryptanalysis and stream ciphers
- **Combinatorial Optimization**: Matroid theory for greedy algorithm correctness
- **Algebraic Geometry**: Grassmannian point counting and Schubert calculus over finite fields
- **Topological Combinatorics**: Kazhdan-Lusztig polynomials for matroid invariants
- **Digital Circuit Design**: GF(2) arithmetic maps directly to XOR/AND gate networks
- **Quantum Error Correction**: Stabilizer codes built from GF(2) matrices
