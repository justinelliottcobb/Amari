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

/**
 * GF(2) Algebra & Coding Theory Examples
 *
 * This example demonstrates:
 * - Linear algebra over GF(2) (vectors, matrices, Gaussian elimination)
 * - Binary Clifford algebra Cl(N,R; F_2)
 * - Binary linear codes (Hamming, Reed-Muller, Golay)
 * - Grassmannian combinatorics and finite field point counting
 * - Matroid representability testing
 * - Kazhdan-Lusztig polynomials for matroids
 */
async function runGF2AlgebraExample() {
  console.log('🔢 GF(2) Algebra & Coding Theory Examples');
  console.log('==========================================');

  // Initialize the WASM module
  await init();

  // ========================================================================
  // 1. GF(2) Vectors
  // ========================================================================

  console.log('\n1. GF(2) Vector Arithmetic:');

  const v1 = WasmGF2Vector.fromBits(new Uint8Array([1, 0, 1, 1]));
  const v2 = WasmGF2Vector.fromBits(new Uint8Array([1, 1, 0, 1]));

  console.log(`   v1 = ${v1.toString()}`);
  console.log(`   v2 = ${v2.toString()}`);
  console.log(`   v1 weight: ${v1.weight()}`);
  console.log(`   v2 weight: ${v2.weight()}`);
  console.log(`   Hamming distance: ${v1.hammingDistance(v2)}`);
  console.log(`   Dot product: ${v1.dot(v2)}`);

  const sum = v1.add(v2);
  console.log(`   v1 + v2 (XOR): ${sum.toString()}`);

  // ========================================================================
  // 2. GF(2) Matrices
  // ========================================================================

  console.log('\n2. GF(2) Matrix Operations:');

  // 2x3 matrix over GF(2)
  const matrix = WasmGF2Matrix.fromRows(
    new Uint8Array([1, 0, 1, 0, 1, 1]), 2, 3
  );
  console.log(`   Matrix (2x3):\n${matrix.toString()}`);
  console.log(`   Rank: ${matrix.rank()}`);

  // Identity matrix
  const eye = WasmGF2Matrix.identity(3);
  console.log(`   Identity (3x3):\n${eye.toString()}`);
  console.log(`   Determinant: ${eye.determinant()}`);

  // Solve Ax = b
  const b = WasmGF2Vector.fromBits(new Uint8Array([1, 1]));
  const x = matrix.solve(b);
  if (x) {
    console.log(`   Solve Ax = [1,1]: x = ${x.toString()}`);
    x.free();
  }

  // Null space
  const nullBasis = matrix.nullSpace();
  console.log(`   Null space basis: ${JSON.stringify(nullBasis)}`);

  // Transpose
  const mt = matrix.transpose();
  console.log(`   Transpose (3x2):\n${mt.toString()}`);

  // ========================================================================
  // 3. Binary Clifford Algebra
  // ========================================================================

  console.log('\n3. Binary Clifford Algebra Cl(N,R; F_2):');

  // Cl(3,0; F_2): 3 non-degenerate generators, e_i^2 = 1
  const e1 = WasmBinaryMultivector.basisVector(3, 0, 0);
  const e2 = WasmBinaryMultivector.basisVector(3, 0, 1);
  const e3 = WasmBinaryMultivector.basisVector(3, 0, 2);

  console.log(`   Cl(3,0; F_2) - 8-dimensional algebra`);
  console.log(`   e1 = ${e1.toString()}`);
  console.log(`   e2 = ${e2.toString()}`);

  // Geometric product
  const e12 = e1.geometricProduct(e2);
  console.log(`   e1 * e2 = ${e12.toString()}`);

  // e_i^2 = 1 in non-degenerate case
  const e1sq = e1.geometricProduct(e1);
  console.log(`   e1^2 = ${e1sq.toString()} (= 1 for non-degenerate)`);

  // Outer product (wedge)
  const e1_wedge_e2 = e1.outerProduct(e2);
  console.log(`   e1 ^ e2 = ${e1_wedge_e2.toString()}`);

  // Trivector
  const e123 = e12.geometricProduct(e3);
  console.log(`   e1 * e2 * e3 = ${e123.toString()} (pseudoscalar)`);

  // Grade projection
  const mixed = e1.add(e12);
  const grade1 = mixed.gradeProjection(1);
  const grade2 = mixed.gradeProjection(2);
  console.log(`   Mixed: ${mixed.toString()}`);
  console.log(`   Grade-1 part: ${grade1.toString()}`);
  console.log(`   Grade-2 part: ${grade2.toString()}`);

  // Degenerate algebra Cl(2,1; F_2): e3^2 = 0
  console.log(`\n   Cl(2,1; F_2) - degenerate generator:`);
  const e3_degen = WasmBinaryMultivector.basisVector(2, 1, 2);
  const e3_sq = e3_degen.geometricProduct(e3_degen);
  console.log(`   e3^2 = ${e3_sq.toString()} (= 0 for degenerate)`);

  // ========================================================================
  // 4. Binary Linear Codes
  // ========================================================================

  console.log('\n4. Binary Linear Codes:');

  // Hamming code [7, 4, 3]
  const hamming = WasmBinaryCode.hammingCode(3);
  const params = hamming.parameters();
  console.log(`   Hamming code: [${params}]`);
  console.log(`   Length n=${hamming.length()}, Dimension k=${hamming.dimension()}, Distance d=${hamming.minimumDistance()}`);

  // Encode a message
  const message = new Uint8Array([1, 0, 1, 0]);
  const codeword = hamming.encode(message);
  console.log(`   Message: [${message}]`);
  console.log(`   Codeword: [${codeword}]`);

  // Verify: syndrome of valid codeword is zero
  const syn = hamming.syndrome(codeword);
  console.log(`   Syndrome (should be zero): [${syn}]`);

  // Introduce an error and check syndrome
  const received = new Uint8Array(codeword);
  received[2] ^= 1; // flip bit 2
  const errSyn = hamming.syndrome(received);
  console.log(`   After flipping bit 2: syndrome = [${errSyn}]`);

  // Weight distribution
  const weights = hamming.weightDistribution();
  console.log(`   Weight distribution: [${weights}]`);

  // Self-duality check
  console.log(`   Self-dual: ${hamming.isSelfDual()}`);

  // Dual code
  const dual = hamming.dual();
  const dualParams = dual.parameters();
  console.log(`   Dual code: [${dualParams}]`);

  // Other standard codes
  console.log('\n   Standard codes:');

  const golay = WasmBinaryCode.extendedGolayCode();
  console.log(`   Extended Golay: [${golay.parameters()}], self-dual: ${golay.isSelfDual()}`);

  const rm = WasmBinaryCode.reedMullerCode(1, 3);
  console.log(`   Reed-Muller RM(1,3): [${rm.parameters()}]`);

  const simplex = WasmBinaryCode.simplexCode(3);
  console.log(`   Simplex(3): [${simplex.parameters()}]`);

  // Coding bounds
  console.log('\n   Coding bounds for [7, 4]:');
  console.log(`   Singleton bound: ${WasmBinaryCode.singletonBound(7, 4)}`);
  console.log(`   Hamming bound (packing radius): ${WasmBinaryCode.hammingBound(7, 4)}`);
  console.log(`   Plotkin bound: ${WasmBinaryCode.plotkinBound(7, 4)}`);
  console.log(`   Gilbert-Varshamov bound: ${WasmBinaryCode.gilbertVarshamovBound(7, 4)}`);

  // ========================================================================
  // 5. Grassmannian Combinatorics
  // ========================================================================

  console.log('\n5. Grassmannian Combinatorics:');

  // Gaussian binomial coefficient
  const gb = GF2Grassmannian.gaussianBinomial(4, 2, 2);
  console.log(`   [4 choose 2]_2 = ${gb} (Gaussian binomial)`);

  // Size of Grassmannian Gr(k, n; F_2)
  const gr12 = GF2Grassmannian.binaryGrassmannianSize(1, 3);
  console.log(`   |Gr(1, 3; F_2)| = ${gr12} (lines in PG(2, F_2) = Fano plane)`);

  const gr24 = GF2Grassmannian.binaryGrassmannianSize(2, 4);
  console.log(`   |Gr(2, 4; F_2)| = ${gr24} (planes in PG(3, F_2))`);

  // Enumerate subspaces
  const subspaces = GF2Grassmannian.enumerateSubspaces(1, 3);
  console.log(`   All 1-dimensional subspaces of F_2^3: ${subspaces.length} subspaces`);

  // ========================================================================
  // 6. Finite Field Point Counting
  // ========================================================================

  console.log('\n6. Finite Field Point Counting:');

  // Grassmannian over various fields
  const gr24_f2 = GF2FiniteField.grassmannianPoints(2, 4, 2);
  const gr24_f3 = GF2FiniteField.grassmannianPoints(2, 4, 3);
  const gr24_f5 = GF2FiniteField.grassmannianPoints(2, 4, 5);
  console.log(`   |Gr(2, 4; F_2)| = ${gr24_f2}`);
  console.log(`   |Gr(2, 4; F_3)| = ${gr24_f3}`);
  console.log(`   |Gr(2, 4; F_5)| = ${gr24_f5}`);

  // Poincare polynomial
  const poincare = GF2FiniteField.grassmannianPoincarePolynomial(2, 4);
  console.log(`   Poincare polynomial of Gr(2,4): [${poincare}]`);

  // Schubert cell points
  const cellPts = GF2FiniteField.schubertCellPoints([1, 0], 2);
  console.log(`   |C_{(1,0)}(F_2)| = ${cellPts} (= q^1 = 2)`);

  // ========================================================================
  // 7. Matroid Representability
  // ========================================================================

  console.log('\n7. Matroid Representability:');

  // Fano matroid (PG(2,2)) — the prototypical binary matroid
  const fano = GF2Representability.fanoMatroid();
  console.log(`   Fano matroid constructed`);

  // Binary representability
  const binaryResult = GF2Representability.isBinary(fano);
  console.log(`   Fano is binary: ${binaryResult.status}`);
  if (binaryResult.matrix) {
    console.log(`   Representation matrix available`);
  }

  // Ternary representability (Fano is NOT ternary)
  const ternaryResult = GF2Representability.isTernary(fano);
  console.log(`   Fano is ternary: ${ternaryResult.status}`);

  // Dual Fano matroid
  const dualFano = GF2Representability.dualFanoMatroid();
  const dualBinary = GF2Representability.isBinary(dualFano);
  console.log(`   Dual Fano is binary: ${dualBinary.status}`);

  // ========================================================================
  // 8. Kazhdan-Lusztig Polynomials
  // ========================================================================

  console.log('\n8. Kazhdan-Lusztig Polynomials:');

  // KL polynomial of the Fano matroid
  const klCoeffs = GF2KazhdanLusztig.klPolynomial(fano);
  console.log(`   KL polynomial of Fano matroid: [${klCoeffs}]`);

  // Non-negativity check (a key conjecture, proved by Elias-Williamson)
  const klNonNeg = GF2KazhdanLusztig.klIsNonNegative(fano);
  console.log(`   KL coefficients non-negative: ${klNonNeg}`);

  // Z-polynomial
  const zCoeffs = GF2KazhdanLusztig.zPolynomial(fano);
  console.log(`   Z-polynomial of Fano matroid: [${zCoeffs}]`);

  // ========================================================================
  // Clean up WASM memory
  // ========================================================================

  v1.free(); v2.free(); sum.free();
  matrix.free(); eye.free(); b.free(); mt.free();
  e1.free(); e2.free(); e3.free();
  e12.free(); e1sq.free(); e1_wedge_e2.free(); e123.free();
  mixed.free(); grade1.free(); grade2.free();
  e3_degen.free(); e3_sq.free();
  hamming.free(); dual.free();
  golay.free(); rm.free(); simplex.free();
  fano.free(); dualFano.free();

  console.log('\n✅ GF(2) algebra example completed!');
}

// Run the example
runGF2AlgebraExample().catch(console.error);
