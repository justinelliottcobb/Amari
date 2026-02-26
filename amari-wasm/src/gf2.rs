//! GF(2) algebra WASM bindings — finite field arithmetic, linear algebra,
//! binary Clifford algebra, coding theory, and enumerative combinatorics.
//!
//! Exposes amari-core GF(2) types and amari-enumerative GF(2) extensions
//! for use from JavaScript/TypeScript via WebAssembly.

use wasm_bindgen::prelude::*;

use amari_core::gf2::{
    binary_grassmannian_size, enumerate_subspaces, gaussian_binomial, schubert_cell_of,
    schubert_cell_size, BinaryMultivector, GF2Matrix, GF2Vector, GF2,
};

#[cfg(test)]
use amari_enumerative::Matroid;
use amari_enumerative::{
    finite_field, kazhdan_lusztig, representability, weight_enumerator, Partition,
};

use crate::enumerative::WasmMatroid;

// ─── WasmGF2Vector ───

/// A vector over GF(2), stored as bit-packed u64 words.
///
/// All arithmetic is modular over the field with two elements:
/// addition is XOR, multiplication is AND.
#[wasm_bindgen]
pub struct WasmGF2Vector {
    inner: GF2Vector,
}

#[wasm_bindgen]
impl WasmGF2Vector {
    /// Create a zero vector of the given dimension.
    #[wasm_bindgen(constructor)]
    pub fn new(dim: usize) -> Self {
        Self {
            inner: GF2Vector::zero(dim),
        }
    }

    /// Create a vector from an array of 0/1 values.
    #[wasm_bindgen(js_name = fromBits)]
    pub fn from_bits(bits: &[u8]) -> Self {
        Self {
            inner: GF2Vector::from_bits(bits),
        }
    }

    /// Get the element at index i (returns 0 or 1).
    pub fn get(&self, i: usize) -> u8 {
        self.inner.get(i).value()
    }

    /// Set the element at index i (value should be 0 or 1).
    pub fn set(&mut self, i: usize, value: u8) {
        self.inner.set(i, GF2::new(value));
    }

    /// Dimension of the vector.
    pub fn dim(&self) -> usize {
        self.inner.dim()
    }

    /// Whether this is the zero vector.
    #[wasm_bindgen(js_name = isZero)]
    pub fn is_zero(&self) -> bool {
        self.inner.is_zero()
    }

    /// Hamming weight (number of 1s).
    pub fn weight(&self) -> usize {
        self.inner.weight()
    }

    /// Hamming distance to another vector.
    #[wasm_bindgen(js_name = hammingDistance)]
    pub fn hamming_distance(&self, other: &WasmGF2Vector) -> usize {
        self.inner.hamming_distance(&other.inner)
    }

    /// Dot product (returns 0 or 1).
    pub fn dot(&self, other: &WasmGF2Vector) -> u8 {
        self.inner.dot(&other.inner).value()
    }

    /// Addition (XOR).
    pub fn add(&self, other: &WasmGF2Vector) -> WasmGF2Vector {
        WasmGF2Vector {
            inner: self.inner.add(&other.inner),
        }
    }

    /// Extract all bits as a u8 array.
    #[wasm_bindgen(js_name = toBits)]
    pub fn to_bits(&self) -> Vec<u8> {
        (0..self.inner.dim())
            .map(|i| self.inner.get(i).value())
            .collect()
    }

    /// String representation.
    #[wasm_bindgen(js_name = toString)]
    pub fn to_string_js(&self) -> String {
        format!("{}", self.inner)
    }
}

// ─── WasmGF2Matrix ───

/// A matrix over GF(2), stored as row vectors of bit-packed words.
///
/// Supports Gaussian elimination, rank computation, null space extraction,
/// and linear system solving — all via bitwise operations.
#[wasm_bindgen]
pub struct WasmGF2Matrix {
    inner: GF2Matrix,
}

#[wasm_bindgen]
impl WasmGF2Matrix {
    /// Create a zero matrix.
    #[wasm_bindgen(constructor)]
    pub fn new(nrows: usize, ncols: usize) -> Self {
        Self {
            inner: GF2Matrix::zero(nrows, ncols),
        }
    }

    /// Create an identity matrix.
    pub fn identity(n: usize) -> Self {
        Self {
            inner: GF2Matrix::identity(n),
        }
    }

    /// Create from a flat row-major array of 0/1 values.
    ///
    /// The array length must equal nrows * ncols.
    #[wasm_bindgen(js_name = fromRows)]
    pub fn from_rows(data: &[u8], nrows: usize, ncols: usize) -> Result<WasmGF2Matrix, JsValue> {
        if data.len() != nrows * ncols {
            return Err(JsValue::from_str(&format!(
                "data length {} != nrows * ncols ({})",
                data.len(),
                nrows * ncols
            )));
        }
        let rows: Vec<GF2Vector> = (0..nrows)
            .map(|i| {
                let row_data = &data[i * ncols..(i + 1) * ncols];
                GF2Vector::from_bits(row_data)
            })
            .collect();
        Ok(Self {
            inner: GF2Matrix::from_rows(rows),
        })
    }

    /// Get element at (row, col), returns 0 or 1.
    pub fn get(&self, row: usize, col: usize) -> u8 {
        self.inner.get(row, col).value()
    }

    /// Set element at (row, col), value should be 0 or 1.
    pub fn set(&mut self, row: usize, col: usize, value: u8) {
        self.inner.set(row, col, GF2::new(value));
    }

    /// Number of rows.
    pub fn nrows(&self) -> usize {
        self.inner.nrows()
    }

    /// Number of columns.
    pub fn ncols(&self) -> usize {
        self.inner.ncols()
    }

    /// Matrix-vector product over GF(2).
    #[wasm_bindgen(js_name = mulVec)]
    pub fn mul_vec(&self, v: &WasmGF2Vector) -> WasmGF2Vector {
        WasmGF2Vector {
            inner: self.inner.mul_vec(&v.inner),
        }
    }

    /// Matrix-matrix product over GF(2).
    #[wasm_bindgen(js_name = mulMat)]
    pub fn mul_mat(&self, other: &WasmGF2Matrix) -> WasmGF2Matrix {
        WasmGF2Matrix {
            inner: self.inner.mul_mat(&other.inner),
        }
    }

    /// Transpose.
    pub fn transpose(&self) -> WasmGF2Matrix {
        WasmGF2Matrix {
            inner: self.inner.transpose(),
        }
    }

    /// Rank (number of linearly independent rows).
    pub fn rank(&self) -> usize {
        self.inner.rank()
    }

    /// Determinant (square matrices only). Returns 0 or 1.
    pub fn determinant(&self) -> Result<u8, JsValue> {
        self.inner
            .determinant()
            .map(|d| d.value())
            .map_err(|e| JsValue::from_str(&format!("{}", e)))
    }

    /// Null space basis vectors as nested arrays via serde.
    ///
    /// Returns a JS array of arrays, where each inner array is a basis vector
    /// represented as 0/1 values.
    #[wasm_bindgen(js_name = nullSpace)]
    pub fn null_space(&self) -> JsValue {
        let vecs = self.inner.null_space();
        let result: Vec<Vec<u8>> = vecs
            .iter()
            .map(|v| (0..v.dim()).map(|i| v.get(i).value()).collect())
            .collect();
        serde_wasm_bindgen::to_value(&result).unwrap_or(JsValue::NULL)
    }

    /// Column space basis vectors as nested arrays via serde.
    #[wasm_bindgen(js_name = columnSpace)]
    pub fn column_space(&self) -> JsValue {
        let vecs = self.inner.column_space();
        let result: Vec<Vec<u8>> = vecs
            .iter()
            .map(|v| (0..v.dim()).map(|i| v.get(i).value()).collect())
            .collect();
        serde_wasm_bindgen::to_value(&result).unwrap_or(JsValue::NULL)
    }

    /// Solve Ax = b over GF(2). Returns null if no solution exists.
    pub fn solve(&self, b: &WasmGF2Vector) -> Option<WasmGF2Vector> {
        self.inner
            .solve(&b.inner)
            .map(|v| WasmGF2Vector { inner: v })
    }

    /// Reduced row echelon form (in-place). Returns pivot column indices.
    #[wasm_bindgen(js_name = reducedRowEchelon)]
    pub fn reduced_row_echelon(&mut self) -> Vec<usize> {
        self.inner.reduced_row_echelon()
    }

    /// Flat row-major representation as 0/1 values.
    #[wasm_bindgen(js_name = toFlatArray)]
    pub fn to_flat_array(&self) -> Vec<u8> {
        let mut result = Vec::with_capacity(self.inner.nrows() * self.inner.ncols());
        for i in 0..self.inner.nrows() {
            for j in 0..self.inner.ncols() {
                result.push(self.inner.get(i, j).value());
            }
        }
        result
    }

    /// String representation.
    #[wasm_bindgen(js_name = toString)]
    pub fn to_string_js(&self) -> String {
        format!("{}", self.inner)
    }
}

// ─── WasmBinaryMultivector ───

/// Internal enum for dispatching BinaryMultivector operations across
/// supported algebra signatures. wasm_bindgen cannot handle const generics,
/// so we dispatch at runtime.
enum BinaryMvInner {
    Cl2(BinaryMultivector<2, 0>),
    Cl3(BinaryMultivector<3, 0>),
    Cl4(BinaryMultivector<4, 0>),
    Cl21(BinaryMultivector<2, 1>),
    Cl31(BinaryMultivector<3, 1>),
}

/// Helper macro to dispatch a method call across all BinaryMvInner variants.
macro_rules! dispatch {
    ($self:expr, $method:ident $(, $arg:expr)*) => {
        match $self {
            BinaryMvInner::Cl2(mv) => BinaryMvInner::Cl2(mv.$method($($arg),*)),
            BinaryMvInner::Cl3(mv) => BinaryMvInner::Cl3(mv.$method($($arg),*)),
            BinaryMvInner::Cl4(mv) => BinaryMvInner::Cl4(mv.$method($($arg),*)),
            BinaryMvInner::Cl21(mv) => BinaryMvInner::Cl21(mv.$method($($arg),*)),
            BinaryMvInner::Cl31(mv) => BinaryMvInner::Cl31(mv.$method($($arg),*)),
        }
    };
}

/// Helper macro to dispatch a binary operation (self op other) where both must
/// have the same variant.
macro_rules! dispatch_binary {
    ($self:expr, $other:expr, $method:ident) => {
        match ($self, $other) {
            (BinaryMvInner::Cl2(a), BinaryMvInner::Cl2(b)) => BinaryMvInner::Cl2(a.$method(b)),
            (BinaryMvInner::Cl3(a), BinaryMvInner::Cl3(b)) => BinaryMvInner::Cl3(a.$method(b)),
            (BinaryMvInner::Cl4(a), BinaryMvInner::Cl4(b)) => BinaryMvInner::Cl4(a.$method(b)),
            (BinaryMvInner::Cl21(a), BinaryMvInner::Cl21(b)) => BinaryMvInner::Cl21(a.$method(b)),
            (BinaryMvInner::Cl31(a), BinaryMvInner::Cl31(b)) => BinaryMvInner::Cl31(a.$method(b)),
            _ => return Err(JsValue::from_str("signature mismatch")),
        }
    };
}

/// Helper macro to dispatch a query (returns a value, not a new multivector).
macro_rules! dispatch_query {
    ($self:expr, $method:ident $(, $arg:expr)*) => {
        match $self {
            BinaryMvInner::Cl2(mv) => mv.$method($($arg),*),
            BinaryMvInner::Cl3(mv) => mv.$method($($arg),*),
            BinaryMvInner::Cl4(mv) => mv.$method($($arg),*),
            BinaryMvInner::Cl21(mv) => mv.$method($($arg),*),
            BinaryMvInner::Cl31(mv) => mv.$method($($arg),*),
        }
    };
}

impl BinaryMvInner {
    fn get(&self, index: usize) -> GF2 {
        dispatch_query!(self, get, index)
    }

    fn set(&mut self, index: usize, value: GF2) {
        match self {
            BinaryMvInner::Cl2(mv) => mv.set(index, value),
            BinaryMvInner::Cl3(mv) => mv.set(index, value),
            BinaryMvInner::Cl4(mv) => mv.set(index, value),
            BinaryMvInner::Cl21(mv) => mv.set(index, value),
            BinaryMvInner::Cl31(mv) => mv.set(index, value),
        }
    }

    fn is_zero(&self) -> bool {
        dispatch_query!(self, is_zero)
    }

    fn weight(&self) -> usize {
        dispatch_query!(self, weight)
    }

    fn grade(&self) -> usize {
        dispatch_query!(self, grade)
    }

    fn basis_count(&self) -> usize {
        match self {
            BinaryMvInner::Cl2(_) => BinaryMultivector::<2, 0>::BASIS_COUNT,
            BinaryMvInner::Cl3(_) => BinaryMultivector::<3, 0>::BASIS_COUNT,
            BinaryMvInner::Cl4(_) => BinaryMultivector::<4, 0>::BASIS_COUNT,
            BinaryMvInner::Cl21(_) => BinaryMultivector::<2, 1>::BASIS_COUNT,
            BinaryMvInner::Cl31(_) => BinaryMultivector::<3, 1>::BASIS_COUNT,
        }
    }
}

impl std::fmt::Display for BinaryMvInner {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BinaryMvInner::Cl2(mv) => write!(f, "{}", mv),
            BinaryMvInner::Cl3(mv) => write!(f, "{}", mv),
            BinaryMvInner::Cl4(mv) => write!(f, "{}", mv),
            BinaryMvInner::Cl21(mv) => write!(f, "{}", mv),
            BinaryMvInner::Cl31(mv) => write!(f, "{}", mv),
        }
    }
}

fn create_zero(n: usize, r: usize) -> Result<BinaryMvInner, JsValue> {
    match (n, r) {
        (2, 0) => Ok(BinaryMvInner::Cl2(BinaryMultivector::<2, 0>::zero())),
        (3, 0) => Ok(BinaryMvInner::Cl3(BinaryMultivector::<3, 0>::zero())),
        (4, 0) => Ok(BinaryMvInner::Cl4(BinaryMultivector::<4, 0>::zero())),
        (2, 1) => Ok(BinaryMvInner::Cl21(BinaryMultivector::<2, 1>::zero())),
        (3, 1) => Ok(BinaryMvInner::Cl31(BinaryMultivector::<3, 1>::zero())),
        _ => Err(JsValue::from_str(&format!(
            "unsupported signature Cl({},{};F₂). Supported: (2,0), (3,0), (4,0), (2,1), (3,1)",
            n, r
        ))),
    }
}

fn create_one(n: usize, r: usize) -> Result<BinaryMvInner, JsValue> {
    match (n, r) {
        (2, 0) => Ok(BinaryMvInner::Cl2(BinaryMultivector::<2, 0>::one())),
        (3, 0) => Ok(BinaryMvInner::Cl3(BinaryMultivector::<3, 0>::one())),
        (4, 0) => Ok(BinaryMvInner::Cl4(BinaryMultivector::<4, 0>::one())),
        (2, 1) => Ok(BinaryMvInner::Cl21(BinaryMultivector::<2, 1>::one())),
        (3, 1) => Ok(BinaryMvInner::Cl31(BinaryMultivector::<3, 1>::one())),
        _ => Err(JsValue::from_str("unsupported signature")),
    }
}

fn create_basis_vector(n: usize, r: usize, i: usize) -> Result<BinaryMvInner, JsValue> {
    match (n, r) {
        (2, 0) => Ok(BinaryMvInner::Cl2(BinaryMultivector::<2, 0>::basis_vector(
            i,
        ))),
        (3, 0) => Ok(BinaryMvInner::Cl3(BinaryMultivector::<3, 0>::basis_vector(
            i,
        ))),
        (4, 0) => Ok(BinaryMvInner::Cl4(BinaryMultivector::<4, 0>::basis_vector(
            i,
        ))),
        (2, 1) => Ok(BinaryMvInner::Cl21(
            BinaryMultivector::<2, 1>::basis_vector(i),
        )),
        (3, 1) => Ok(BinaryMvInner::Cl31(
            BinaryMultivector::<3, 1>::basis_vector(i),
        )),
        _ => Err(JsValue::from_str("unsupported signature")),
    }
}

fn create_basis_blade(n: usize, r: usize, index: usize) -> Result<BinaryMvInner, JsValue> {
    match (n, r) {
        (2, 0) => Ok(BinaryMvInner::Cl2(BinaryMultivector::<2, 0>::basis_blade(
            index,
        ))),
        (3, 0) => Ok(BinaryMvInner::Cl3(BinaryMultivector::<3, 0>::basis_blade(
            index,
        ))),
        (4, 0) => Ok(BinaryMvInner::Cl4(BinaryMultivector::<4, 0>::basis_blade(
            index,
        ))),
        (2, 1) => Ok(BinaryMvInner::Cl21(BinaryMultivector::<2, 1>::basis_blade(
            index,
        ))),
        (3, 1) => Ok(BinaryMvInner::Cl31(BinaryMultivector::<3, 1>::basis_blade(
            index,
        ))),
        _ => Err(JsValue::from_str("unsupported signature")),
    }
}

fn create_from_bits(n: usize, r: usize, bits: &[u8]) -> Result<BinaryMvInner, JsValue> {
    match (n, r) {
        (2, 0) => Ok(BinaryMvInner::Cl2(BinaryMultivector::<2, 0>::from_bits(
            bits,
        ))),
        (3, 0) => Ok(BinaryMvInner::Cl3(BinaryMultivector::<3, 0>::from_bits(
            bits,
        ))),
        (4, 0) => Ok(BinaryMvInner::Cl4(BinaryMultivector::<4, 0>::from_bits(
            bits,
        ))),
        (2, 1) => Ok(BinaryMvInner::Cl21(BinaryMultivector::<2, 1>::from_bits(
            bits,
        ))),
        (3, 1) => Ok(BinaryMvInner::Cl31(BinaryMultivector::<3, 1>::from_bits(
            bits,
        ))),
        _ => Err(JsValue::from_str("unsupported signature")),
    }
}

/// A multivector over GF(2) in the Clifford algebra Cl(N, R; F₂).
///
/// N non-degenerate generators (eᵢ² = 1) and R degenerate generators (eⱼ² = 0).
/// Supported signatures: Cl(2,0), Cl(3,0), Cl(4,0), Cl(2,1), Cl(3,1).
#[wasm_bindgen]
pub struct WasmBinaryMultivector {
    inner: BinaryMvInner,
    n: usize,
    r: usize,
}

#[wasm_bindgen]
impl WasmBinaryMultivector {
    /// Create a zero multivector with the given signature.
    #[wasm_bindgen(constructor)]
    pub fn new(n: usize, r: usize) -> Result<WasmBinaryMultivector, JsValue> {
        let inner = create_zero(n, r)?;
        Ok(Self { inner, n, r })
    }

    /// Create the scalar identity (1).
    pub fn one(n: usize, r: usize) -> Result<WasmBinaryMultivector, JsValue> {
        let inner = create_one(n, r)?;
        Ok(Self { inner, n, r })
    }

    /// Create a basis vector e_{i+1} (0-indexed).
    #[wasm_bindgen(js_name = basisVector)]
    pub fn basis_vector(n: usize, r: usize, i: usize) -> Result<WasmBinaryMultivector, JsValue> {
        let inner = create_basis_vector(n, r, i)?;
        Ok(Self { inner, n, r })
    }

    /// Create a single basis blade by index.
    #[wasm_bindgen(js_name = basisBlade)]
    pub fn basis_blade(n: usize, r: usize, index: usize) -> Result<WasmBinaryMultivector, JsValue> {
        let inner = create_basis_blade(n, r, index)?;
        Ok(Self { inner, n, r })
    }

    /// Create from a coefficient array of 0/1 values (one per basis blade).
    #[wasm_bindgen(js_name = fromBits)]
    pub fn from_bits(bits: &[u8], n: usize, r: usize) -> Result<WasmBinaryMultivector, JsValue> {
        let inner = create_from_bits(n, r, bits)?;
        Ok(Self { inner, n, r })
    }

    /// Get coefficient of basis blade at index (returns 0 or 1).
    pub fn get(&self, index: usize) -> u8 {
        self.inner.get(index).value()
    }

    /// Set coefficient of basis blade at index (value should be 0 or 1).
    pub fn set(&mut self, index: usize, value: u8) {
        self.inner.set(index, GF2::new(value));
    }

    /// Geometric product (Clifford product) over GF(2).
    #[wasm_bindgen(js_name = geometricProduct)]
    pub fn geometric_product(
        &self,
        other: &WasmBinaryMultivector,
    ) -> Result<WasmBinaryMultivector, JsValue> {
        let result = dispatch_binary!(&self.inner, &other.inner, geometric_product);
        Ok(Self {
            inner: result,
            n: self.n,
            r: self.r,
        })
    }

    /// Outer (wedge) product over GF(2).
    #[wasm_bindgen(js_name = outerProduct)]
    pub fn outer_product(
        &self,
        other: &WasmBinaryMultivector,
    ) -> Result<WasmBinaryMultivector, JsValue> {
        let result = dispatch_binary!(&self.inner, &other.inner, outer_product);
        Ok(Self {
            inner: result,
            n: self.n,
            r: self.r,
        })
    }

    /// Inner product (left contraction) over GF(2).
    #[wasm_bindgen(js_name = innerProduct)]
    pub fn inner_product(
        &self,
        other: &WasmBinaryMultivector,
    ) -> Result<WasmBinaryMultivector, JsValue> {
        let result = dispatch_binary!(&self.inner, &other.inner, inner_product);
        Ok(Self {
            inner: result,
            n: self.n,
            r: self.r,
        })
    }

    /// Addition (XOR of coefficients).
    pub fn add(&self, other: &WasmBinaryMultivector) -> Result<WasmBinaryMultivector, JsValue> {
        let result = dispatch_binary!(&self.inner, &other.inner, gf2_add);
        Ok(Self {
            inner: result,
            n: self.n,
            r: self.r,
        })
    }

    /// Grade projection: keep only blades of the given grade.
    #[wasm_bindgen(js_name = gradeProjection)]
    pub fn grade_projection(&self, grade: usize) -> WasmBinaryMultivector {
        let result = dispatch!(&self.inner, grade_projection, grade);
        Self {
            inner: result,
            n: self.n,
            r: self.r,
        }
    }

    /// Reverse operator.
    pub fn reverse(&self) -> WasmBinaryMultivector {
        let result = dispatch!(&self.inner, reverse);
        Self {
            inner: result,
            n: self.n,
            r: self.r,
        }
    }

    /// Whether this is the zero multivector.
    #[wasm_bindgen(js_name = isZero)]
    pub fn is_zero(&self) -> bool {
        self.inner.is_zero()
    }

    /// Number of nonzero coefficients.
    pub fn weight(&self) -> usize {
        self.inner.weight()
    }

    /// Highest grade present.
    pub fn grade(&self) -> usize {
        self.inner.grade()
    }

    /// Total number of basis blades (2^(N+R)).
    #[wasm_bindgen(js_name = basisCount)]
    pub fn basis_count(&self) -> usize {
        self.inner.basis_count()
    }

    /// Number of non-degenerate generators (N).
    #[wasm_bindgen(getter, js_name = n)]
    pub fn n_generators(&self) -> usize {
        self.n
    }

    /// Number of degenerate generators (R).
    #[wasm_bindgen(getter, js_name = r)]
    pub fn r_generators(&self) -> usize {
        self.r
    }

    /// String representation.
    #[wasm_bindgen(js_name = toString)]
    pub fn to_string_js(&self) -> String {
        self.inner.to_string()
    }
}

// ─── GF2Grassmannian ───

/// Grassmannian combinatorics over GF(2).
///
/// Static utility methods for Gaussian binomial coefficients,
/// binary Grassmannian sizes, and subspace enumeration.
#[wasm_bindgen]
pub struct GF2Grassmannian;

#[wasm_bindgen]
impl GF2Grassmannian {
    /// Gaussian binomial coefficient [n choose k]_q.
    #[wasm_bindgen(js_name = gaussianBinomial)]
    pub fn gaussian_binomial_js(n: usize, k: usize, q: u64) -> u64 {
        gaussian_binomial(n, k, q)
    }

    /// Number of k-dimensional subspaces of F₂ⁿ.
    #[wasm_bindgen(js_name = binaryGrassmannianSize)]
    pub fn binary_grassmannian_size_js(k: usize, n: usize) -> u64 {
        binary_grassmannian_size(k, n)
    }

    /// Enumerate all k-dimensional subspaces of F₂ⁿ.
    ///
    /// Returns a JS array of matrices (each in RREF), where each matrix
    /// is represented as a flat array of 0/1 values with [k, n] dimensions.
    /// Panics if n > 20 (too many subspaces).
    #[wasm_bindgen(js_name = enumerateSubspaces)]
    pub fn enumerate_subspaces_js(k: usize, n: usize) -> JsValue {
        let subspaces = enumerate_subspaces(k, n);
        let result: Vec<Vec<Vec<u8>>> = subspaces
            .iter()
            .map(|m| {
                (0..m.nrows())
                    .map(|i| (0..m.ncols()).map(|j| m.get(i, j).value()).collect())
                    .collect()
            })
            .collect();
        serde_wasm_bindgen::to_value(&result).unwrap_or(JsValue::NULL)
    }

    /// Schubert cell partition of a subspace (given as RREF matrix).
    #[wasm_bindgen(js_name = schubertCellOf)]
    pub fn schubert_cell_of_js(matrix: &WasmGF2Matrix) -> Vec<usize> {
        schubert_cell_of(&matrix.inner)
    }

    /// Size of a Schubert cell given its partition: 2^|λ|.
    #[wasm_bindgen(js_name = schubertCellSize)]
    pub fn schubert_cell_size_js(partition: &[usize]) -> u64 {
        schubert_cell_size(partition)
    }
}

// ─── WasmBinaryCode ───

/// A binary linear code [n, k, d] over GF(2).
///
/// Supports code construction, encoding/decoding, weight analysis,
/// and standard coding theory bounds.
#[wasm_bindgen]
pub struct WasmBinaryCode {
    inner: weight_enumerator::BinaryCode,
}

#[wasm_bindgen]
impl WasmBinaryCode {
    /// Create from a generator matrix (flat row-major 0/1 array with dimensions).
    #[wasm_bindgen(js_name = fromGenerator)]
    pub fn from_generator(
        data: &[u8],
        nrows: usize,
        ncols: usize,
    ) -> Result<WasmBinaryCode, JsValue> {
        let matrix = flat_to_gf2_matrix(data, nrows, ncols)?;
        weight_enumerator::BinaryCode::from_generator(matrix)
            .map(|code| Self { inner: code })
            .map_err(|e| JsValue::from_str(&format!("{}", e)))
    }

    /// Create from a parity check matrix (flat row-major 0/1 array with dimensions).
    #[wasm_bindgen(js_name = fromParityCheck)]
    pub fn from_parity_check(
        data: &[u8],
        nrows: usize,
        ncols: usize,
    ) -> Result<WasmBinaryCode, JsValue> {
        let matrix = flat_to_gf2_matrix(data, nrows, ncols)?;
        weight_enumerator::BinaryCode::from_parity_check(matrix)
            .map(|code| Self { inner: code })
            .map_err(|e| JsValue::from_str(&format!("{}", e)))
    }

    /// Hamming code [2^r - 1, 2^r - r - 1, 3].
    #[wasm_bindgen(js_name = hammingCode)]
    pub fn hamming_code(r: usize) -> WasmBinaryCode {
        Self {
            inner: weight_enumerator::hamming_code(r),
        }
    }

    /// Simplex code [2^r - 1, r, 2^(r-1)].
    #[wasm_bindgen(js_name = simplexCode)]
    pub fn simplex_code(r: usize) -> WasmBinaryCode {
        Self {
            inner: weight_enumerator::simplex_code(r),
        }
    }

    /// Reed-Muller code RM(r, m).
    #[wasm_bindgen(js_name = reedMullerCode)]
    pub fn reed_muller_code(r: usize, m: usize) -> Result<WasmBinaryCode, JsValue> {
        weight_enumerator::reed_muller_code(r, m)
            .map(|code| Self { inner: code })
            .map_err(|e| JsValue::from_str(&format!("{}", e)))
    }

    /// Extended Golay code [24, 12, 8].
    #[wasm_bindgen(js_name = extendedGolayCode)]
    pub fn extended_golay_code() -> WasmBinaryCode {
        Self {
            inner: weight_enumerator::extended_golay_code(),
        }
    }

    /// Code length n.
    pub fn length(&self) -> usize {
        self.inner.length()
    }

    /// Code dimension k.
    pub fn dimension(&self) -> usize {
        self.inner.dimension()
    }

    /// Minimum distance d.
    #[wasm_bindgen(js_name = minimumDistance)]
    pub fn minimum_distance(&self) -> usize {
        self.inner.minimum_distance()
    }

    /// Code parameters [n, k, d].
    pub fn parameters(&self) -> Vec<usize> {
        let (n, k, d) = self.inner.parameters();
        vec![n, k, d]
    }

    /// Whether the code is self-dual (C = C⊥).
    #[wasm_bindgen(js_name = isSelfDual)]
    pub fn is_self_dual(&self) -> bool {
        self.inner.is_self_dual()
    }

    /// Encode a message vector (k bits as 0/1 array) into a codeword (n bits).
    pub fn encode(&self, message: &[u8]) -> Vec<u8> {
        let msg = GF2Vector::from_bits(message);
        let codeword = self.inner.encode(&msg);
        (0..codeword.dim())
            .map(|i| codeword.get(i).value())
            .collect()
    }

    /// Compute syndrome of a received word (n bits as 0/1 array).
    pub fn syndrome(&self, received: &[u8]) -> Vec<u8> {
        let recv = GF2Vector::from_bits(received);
        let syn = self.inner.syndrome(&recv);
        (0..syn.dim()).map(|i| syn.get(i).value()).collect()
    }

    /// Dual code C⊥.
    pub fn dual(&self) -> WasmBinaryCode {
        Self {
            inner: self.inner.dual(),
        }
    }

    /// Weight enumerator polynomial coefficients [A_0, A_1, ..., A_n].
    #[wasm_bindgen(js_name = weightEnumerator)]
    pub fn weight_enumerator(&self) -> Vec<f64> {
        self.inner
            .weight_enumerator()
            .iter()
            .map(|&x| x as f64)
            .collect()
    }

    /// Weight distribution [A_0, A_1, ..., A_n].
    #[wasm_bindgen(js_name = weightDistribution)]
    pub fn weight_distribution(&self) -> Vec<f64> {
        self.inner
            .weight_distribution()
            .iter()
            .map(|&x| x as f64)
            .collect()
    }

    /// Generator matrix as flat row-major 0/1 array.
    #[wasm_bindgen(js_name = generatorMatrix)]
    pub fn generator_matrix(&self) -> WasmGF2Matrix {
        WasmGF2Matrix {
            inner: self.inner.generator_matrix().clone(),
        }
    }

    // ─── Static bounds ───

    /// Singleton bound: d ≤ n - k + 1.
    #[wasm_bindgen(js_name = singletonBound)]
    pub fn singleton_bound(n: usize, k: usize) -> usize {
        weight_enumerator::singleton_bound(n, k)
    }

    /// Hamming (sphere-packing) bound.
    #[wasm_bindgen(js_name = hammingBound)]
    pub fn hamming_bound(n: usize, k: usize) -> usize {
        weight_enumerator::hamming_bound(n, k)
    }

    /// Plotkin bound (returns -1 if not applicable).
    #[wasm_bindgen(js_name = plotkinBound)]
    pub fn plotkin_bound(n: usize, k: usize) -> i32 {
        weight_enumerator::plotkin_bound(n, k)
            .map(|v| v as i32)
            .unwrap_or(-1)
    }

    /// Gilbert-Varshamov bound.
    #[wasm_bindgen(js_name = gilbertVarshamovBound)]
    pub fn gilbert_varshamov_bound(n: usize, k: usize) -> usize {
        weight_enumerator::gilbert_varshamov_bound(n, k)
    }
}

// ─── GF2FiniteField ───

/// Finite field point counting for Grassmannians and Schubert varieties.
#[wasm_bindgen]
pub struct GF2FiniteField;

#[wasm_bindgen]
impl GF2FiniteField {
    /// Number of F_q-rational points of the Grassmannian Gr(k, n).
    #[wasm_bindgen(js_name = grassmannianPoints)]
    pub fn grassmannian_points(k: usize, n: usize, q: u64) -> f64 {
        finite_field::grassmannian_points(k, n, q) as f64
    }

    /// Number of F_q-rational points of a Schubert cell C_λ.
    #[wasm_bindgen(js_name = schubertCellPoints)]
    pub fn schubert_cell_points(partition: &[usize], q: u64) -> f64 {
        let p = Partition::new(partition.to_vec());
        finite_field::schubert_cell_points(&p, q) as f64
    }

    /// Number of F_q-rational points of a Schubert variety X_λ in Gr(k, n).
    #[wasm_bindgen(js_name = schubertVarietyPoints)]
    pub fn schubert_variety_points(partition: &[usize], k: usize, n: usize, q: u64) -> f64 {
        let p = Partition::new(partition.to_vec());
        finite_field::schubert_variety_points(&p, (k, n), q) as f64
    }

    /// Poincaré polynomial of the Grassmannian Gr(k, n).
    /// Returns coefficients [a_0, a_1, ..., a_d] where P(t) = Σ a_i t^i.
    #[wasm_bindgen(js_name = grassmannianPoincarePolynomial)]
    pub fn grassmannian_poincare_polynomial(k: usize, n: usize) -> Vec<f64> {
        finite_field::grassmannian_poincare_polynomial(k, n)
            .iter()
            .map(|&x| x as f64)
            .collect()
    }

    /// Poincaré polynomial of a Schubert variety X_λ in Gr(k, n).
    #[wasm_bindgen(js_name = schubertPoincarePolynomial)]
    pub fn schubert_poincare_polynomial(partition: &[usize], k: usize, n: usize) -> Vec<f64> {
        let p = Partition::new(partition.to_vec());
        finite_field::schubert_poincare_polynomial(&p, (k, n))
            .iter()
            .map(|&x| x as f64)
            .collect()
    }
}

// ─── GF2Representability ───

/// Matroid representability testing over finite fields.
///
/// Tests whether a matroid can be represented as a column matroid
/// of a matrix over GF(2), GF(3), or GF(q).
#[wasm_bindgen]
pub struct GF2Representability;

#[wasm_bindgen]
impl GF2Representability {
    /// Test if a matroid is binary (representable over GF(2)).
    ///
    /// Returns a JS object: `{status: "representable"|"not_representable"|"inconclusive", matrix?: [[0,1,...],...]}`
    #[wasm_bindgen(js_name = isBinary)]
    pub fn is_binary(matroid: &WasmMatroid) -> JsValue {
        representability_result_to_js(representability::is_binary(&matroid.inner))
    }

    /// Test if a matroid is ternary (representable over GF(3)).
    #[wasm_bindgen(js_name = isTernary)]
    pub fn is_ternary(matroid: &WasmMatroid) -> JsValue {
        representability_result_to_js(representability::is_ternary(&matroid.inner))
    }

    /// Test if a matroid is regular (representable over every field).
    #[wasm_bindgen(js_name = isRegular)]
    pub fn is_regular(matroid: &WasmMatroid) -> bool {
        representability::is_regular(&matroid.inner)
    }

    /// Test if a matroid is representable over GF(q).
    #[wasm_bindgen(js_name = isRepresentableOverGFq)]
    pub fn is_representable_over_gfq(matroid: &WasmMatroid, q: u64) -> JsValue {
        representability_result_to_js(representability::is_representable_over_gfq(
            &matroid.inner,
            q,
        ))
    }

    /// Check if a matroid contains a given minor.
    #[wasm_bindgen(js_name = hasMinor)]
    pub fn has_minor(matroid: &WasmMatroid, minor: &WasmMatroid) -> bool {
        representability::has_minor(&matroid.inner, &minor.inner)
    }

    /// Construct the Fano matroid (the matroid of the Fano plane, PG(2,2)).
    #[wasm_bindgen(js_name = fanoMatroid)]
    pub fn fano_matroid() -> WasmMatroid {
        WasmMatroid {
            inner: representability::fano_matroid(),
        }
    }

    /// Construct the dual Fano matroid.
    #[wasm_bindgen(js_name = dualFanoMatroid)]
    pub fn dual_fano_matroid() -> WasmMatroid {
        WasmMatroid {
            inner: representability::dual_fano_matroid(),
        }
    }

    /// Construct the column matroid of a GF(2) matrix.
    #[wasm_bindgen(js_name = columnMatroid)]
    pub fn column_matroid(matrix: &WasmGF2Matrix) -> WasmMatroid {
        WasmMatroid {
            inner: representability::column_matroid(&matrix.inner),
        }
    }

    /// Find a standard GF(2) representation matrix for a matroid, if one exists.
    #[wasm_bindgen(js_name = standardRepresentation)]
    pub fn standard_representation(matroid: &WasmMatroid) -> Option<WasmGF2Matrix> {
        representability::standard_representation(&matroid.inner)
            .map(|m| WasmGF2Matrix { inner: m })
    }
}

/// Convert a RepresentabilityResult to a JS object.
fn representability_result_to_js(result: representability::RepresentabilityResult) -> JsValue {
    match result {
        representability::RepresentabilityResult::Representable(matrix) => {
            let matrix_data: Vec<Vec<u8>> = (0..matrix.nrows())
                .map(|i| {
                    (0..matrix.ncols())
                        .map(|j| matrix.get(i, j).value())
                        .collect()
                })
                .collect();
            let obj = js_sys::Object::new();
            js_sys::Reflect::set(
                &obj,
                &JsValue::from_str("status"),
                &JsValue::from_str("representable"),
            )
            .ok();
            js_sys::Reflect::set(
                &obj,
                &JsValue::from_str("matrix"),
                &serde_wasm_bindgen::to_value(&matrix_data).unwrap_or(JsValue::NULL),
            )
            .ok();
            obj.into()
        }
        representability::RepresentabilityResult::NotRepresentable => {
            let obj = js_sys::Object::new();
            js_sys::Reflect::set(
                &obj,
                &JsValue::from_str("status"),
                &JsValue::from_str("not_representable"),
            )
            .ok();
            obj.into()
        }
        representability::RepresentabilityResult::Inconclusive { reason } => {
            let obj = js_sys::Object::new();
            js_sys::Reflect::set(
                &obj,
                &JsValue::from_str("status"),
                &JsValue::from_str("inconclusive"),
            )
            .ok();
            js_sys::Reflect::set(
                &obj,
                &JsValue::from_str("reason"),
                &JsValue::from_str(&reason),
            )
            .ok();
            obj.into()
        }
    }
}

// ─── GF2KazhdanLusztig ───

/// Kazhdan-Lusztig polynomials and related invariants for matroids.
#[wasm_bindgen]
pub struct GF2KazhdanLusztig;

#[wasm_bindgen]
impl GF2KazhdanLusztig {
    /// Kazhdan-Lusztig polynomial of a matroid.
    /// Returns coefficients [a_0, a_1, ...] where P(t) = Σ a_i t^i.
    #[wasm_bindgen(js_name = klPolynomial)]
    pub fn kl_polynomial(matroid: &WasmMatroid) -> Vec<f64> {
        kazhdan_lusztig::kl_polynomial(&matroid.inner)
            .iter()
            .map(|&x| x as f64)
            .collect()
    }

    /// Z-polynomial of a matroid.
    #[wasm_bindgen(js_name = zPolynomial)]
    pub fn z_polynomial(matroid: &WasmMatroid) -> Vec<f64> {
        kazhdan_lusztig::z_polynomial(&matroid.inner)
            .iter()
            .map(|&x| x as f64)
            .collect()
    }

    /// Inverse Kazhdan-Lusztig polynomial.
    #[wasm_bindgen(js_name = inverseKlPolynomial)]
    pub fn inverse_kl_polynomial(matroid: &WasmMatroid) -> Vec<f64> {
        kazhdan_lusztig::inverse_kl_polynomial(&matroid.inner)
            .iter()
            .map(|&x| x as f64)
            .collect()
    }

    /// Check whether the KL polynomial has non-negative coefficients.
    #[wasm_bindgen(js_name = klIsNonNegative)]
    pub fn kl_is_non_negative(matroid: &WasmMatroid) -> bool {
        kazhdan_lusztig::kl_is_non_negative(&matroid.inner)
    }

    /// KL polynomial for a Schubert variety in Gr(k, n).
    #[wasm_bindgen(js_name = schubertKlPolynomial)]
    pub fn schubert_kl_polynomial(partition: &[usize], k: usize, n: usize) -> Vec<f64> {
        let p = Partition::new(partition.to_vec());
        kazhdan_lusztig::schubert_kl_polynomial(&p, (k, n))
            .iter()
            .map(|&x| x as f64)
            .collect()
    }
}

// ─── Helpers ───

/// Convert a flat row-major u8 array to a GF2Matrix.
fn flat_to_gf2_matrix(data: &[u8], nrows: usize, ncols: usize) -> Result<GF2Matrix, JsValue> {
    if data.len() != nrows * ncols {
        return Err(JsValue::from_str(&format!(
            "data length {} != nrows * ncols ({})",
            data.len(),
            nrows * ncols
        )));
    }
    let rows: Vec<GF2Vector> = (0..nrows)
        .map(|i| GF2Vector::from_bits(&data[i * ncols..(i + 1) * ncols]))
        .collect();
    Ok(GF2Matrix::from_rows(rows))
}

// ─── Tests ───

#[cfg(test)]
mod tests {
    use super::*;

    // ─── WasmGF2Vector tests ───

    #[test]
    fn test_gf2_vector_creation() {
        let v = WasmGF2Vector::new(4);
        assert_eq!(v.dim(), 4);
        assert!(v.is_zero());
    }

    #[test]
    fn test_gf2_vector_from_bits() {
        let v = WasmGF2Vector::from_bits(&[1, 0, 1, 1]);
        assert_eq!(v.dim(), 4);
        assert_eq!(v.get(0), 1);
        assert_eq!(v.get(1), 0);
        assert_eq!(v.get(2), 1);
        assert_eq!(v.get(3), 1);
        assert_eq!(v.weight(), 3);
    }

    #[test]
    fn test_gf2_vector_add_dot() {
        let a = WasmGF2Vector::from_bits(&[1, 0, 1]);
        let b = WasmGF2Vector::from_bits(&[1, 1, 0]);
        let sum = a.add(&b);
        assert_eq!(sum.to_bits(), vec![0, 1, 1]); // XOR
        assert_eq!(a.dot(&b), 1); // 1*1 + 0*1 + 1*0 = 1
    }

    #[test]
    fn test_gf2_vector_hamming() {
        let a = WasmGF2Vector::from_bits(&[1, 0, 1, 0]);
        let b = WasmGF2Vector::from_bits(&[0, 0, 1, 1]);
        assert_eq!(a.hamming_distance(&b), 2);
    }

    // ─── WasmGF2Matrix tests ───

    #[test]
    fn test_gf2_matrix_identity() {
        let id = WasmGF2Matrix::identity(3);
        assert_eq!(id.nrows(), 3);
        assert_eq!(id.ncols(), 3);
        assert_eq!(id.rank(), 3);
        assert_eq!(id.determinant().unwrap(), 1);
    }

    #[test]
    fn test_gf2_matrix_mul_vec() {
        // [[1,0,1],[0,1,1]] * [1,1,0] = [1, 1]
        let m = WasmGF2Matrix::from_rows(&[1, 0, 1, 0, 1, 1], 2, 3).unwrap();
        let v = WasmGF2Vector::from_bits(&[1, 1, 0]);
        let result = m.mul_vec(&v);
        assert_eq!(result.to_bits(), vec![1, 1]);
    }

    #[test]
    fn test_gf2_matrix_solve() {
        let id = WasmGF2Matrix::identity(2);
        let b = WasmGF2Vector::from_bits(&[1, 1]);
        let x = id.solve(&b).unwrap();
        assert_eq!(x.to_bits(), vec![1, 1]);
    }

    #[test]
    fn test_gf2_matrix_null_space() {
        // [[1,0,1],[0,1,1]] has null space {[1,1,1]}
        // Test via the underlying GF2Matrix directly (serde_wasm_bindgen requires wasm target)
        let m = WasmGF2Matrix::from_rows(&[1, 0, 1, 0, 1, 1], 2, 3).unwrap();
        let ns = m.inner.null_space();
        assert_eq!(ns.len(), 1);
        let product = m.inner.mul_vec(&ns[0]);
        assert!(product.is_zero());
    }

    #[test]
    fn test_gf2_matrix_transpose_roundtrip() {
        let m = WasmGF2Matrix::from_rows(&[1, 0, 1, 0, 1, 0], 2, 3).unwrap();
        let tt = m.transpose().transpose();
        assert_eq!(m.to_flat_array(), tt.to_flat_array());
    }

    // ─── WasmBinaryMultivector tests ───

    #[test]
    fn test_binary_mv_basis_vector_square() {
        // In Cl(3,0;F₂), eᵢ² = 1
        let e1 = WasmBinaryMultivector::basis_vector(3, 0, 0).unwrap();
        let sq = e1.geometric_product(&e1).unwrap();
        let one = WasmBinaryMultivector::one(3, 0).unwrap();
        assert_eq!(sq.get(0), one.get(0)); // scalar = 1
        assert_eq!(sq.weight(), 1);
    }

    #[test]
    fn test_binary_mv_degenerate_square() {
        // In Cl(2,1;F₂), e3² = 0 (degenerate)
        let e3 = WasmBinaryMultivector::basis_vector(2, 1, 2).unwrap();
        let sq = e3.geometric_product(&e3).unwrap();
        assert!(sq.is_zero());
    }

    #[test]
    fn test_binary_mv_grade_projection() {
        // 1 + e1 + e12
        let mut mv = WasmBinaryMultivector::new(3, 0).unwrap();
        mv.set(0, 1); // scalar
        mv.set(1, 1); // e1
        mv.set(3, 1); // e12
        let grade0 = mv.grade_projection(0);
        assert_eq!(grade0.get(0), 1);
        assert_eq!(grade0.weight(), 1);
    }

    // Note: signature mismatch test omitted — JsValue panics on non-wasm32 targets,
    // making Err(JsValue) untestable in native tests. The dispatch_binary! macro
    // returns Err for mismatched variants; this is validated by the WASM runtime.

    // ─── GF2Grassmannian tests ───

    #[test]
    fn test_gaussian_binomial() {
        // [3 choose 1]_2 = (2^3 - 1)/(2^1 - 1) = 7
        assert_eq!(GF2Grassmannian::gaussian_binomial_js(3, 1, 2), 7);
        // [4 choose 2]_2 = 35
        assert_eq!(GF2Grassmannian::gaussian_binomial_js(4, 2, 2), 35);
    }

    #[test]
    fn test_binary_grassmannian_size() {
        // Gr(1, 3; F₂) has 7 points (projective plane PG(2,2))
        assert_eq!(GF2Grassmannian::binary_grassmannian_size_js(1, 3), 7);
    }

    // ─── WasmBinaryCode tests ───

    #[test]
    fn test_hamming_code_params() {
        let code = WasmBinaryCode::hamming_code(3);
        assert_eq!(code.length(), 7);
        assert_eq!(code.dimension(), 4);
        assert_eq!(code.minimum_distance(), 3);
        assert_eq!(code.parameters(), vec![7, 4, 3]);
    }

    #[test]
    fn test_hamming_code_encode_syndrome() {
        let code = WasmBinaryCode::hamming_code(3);
        let msg = vec![1u8, 0, 1, 0]; // 4-bit message
        let codeword = code.encode(&msg);
        assert_eq!(codeword.len(), 7);
        // Syndrome of valid codeword should be zero
        let syn = code.syndrome(&codeword);
        assert!(syn.iter().all(|&b| b == 0));
    }

    #[test]
    fn test_binary_code_weight_distribution() {
        let code = WasmBinaryCode::hamming_code(3);
        let wd = code.weight_distribution();
        assert_eq!(wd.len(), 8); // weights 0..7
        assert_eq!(wd[0], 1.0); // zero codeword
        assert_eq!(wd.iter().sum::<f64>(), 16.0); // 2^4 = 16 codewords total
    }

    #[test]
    fn test_coding_bounds() {
        // [7, 4] code
        assert_eq!(WasmBinaryCode::singleton_bound(7, 4), 4); // n-k+1
                                                              // hamming_bound returns the packing radius t (max correctable errors)
        let hb = WasmBinaryCode::hamming_bound(7, 4);
        assert_eq!(hb, 1); // [7,4,3] Hamming code: t=1
    }

    // ─── GF2FiniteField tests ───

    #[test]
    fn test_grassmannian_points() {
        // |Gr(1, 3; F₂)| = 7
        assert_eq!(GF2FiniteField::grassmannian_points(1, 3, 2), 7.0);
    }

    // ─── GF2Representability tests ───

    #[test]
    fn test_fano_matroid_construction() {
        let fano = GF2Representability::fano_matroid();
        // Fano matroid has 7 elements and rank 3
        assert_eq!(fano.inner.ground_set_size, 7);
        assert_eq!(fano.inner.rank, 3);
    }

    // ─── GF2KazhdanLusztig tests ───

    #[test]
    fn test_kl_polynomial_uniform() {
        // KL polynomial of U_{2,4} starts with 1
        let u24 = WasmMatroid {
            inner: Matroid::uniform(2, 4),
        };
        let kl = GF2KazhdanLusztig::kl_polynomial(&u24);
        assert!(!kl.is_empty());
        assert_eq!(kl[0], 1.0); // constant term is always 1
    }

    #[test]
    fn test_kl_non_negative() {
        let u23 = WasmMatroid {
            inner: Matroid::uniform(2, 3),
        };
        assert!(GF2KazhdanLusztig::kl_is_non_negative(&u23));
    }
}
