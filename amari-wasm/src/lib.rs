//! WASM bindings for the Amari mathematical computing library
//!
//! This module provides WebAssembly bindings for:
//! - **Geometric algebra** (amari-core) - Multivectors, rotors, projections
//! - **Tropical algebra** (amari-tropical) - Optimization via max-plus operations
//! - **Automatic differentiation** (amari-dual) - Forward-mode AD for ML
//! - **Differential calculus** (amari-calculus) - Manifolds, Riemannian geometry
//! - **Measure theory** (amari-measure) - Lebesgue integration, Monte Carlo
//! - **Fusion systems** (amari-fusion) - TropicalDualClifford for LLM evaluation
//! - **Information geometry** (amari-info-geom) - Fisher metrics, statistical manifolds
//! - **Holographic memory** (amari-holographic) - Vector Symbolic Architectures (v0.12.3+)
//!
//! # Holographic Memory (Vector Symbolic Architectures)
//!
//! The fusion module exposes holographic memory operations for storing and
//! retrieving associations in high-dimensional distributed representations:
//!
//! ```javascript
//! import { WasmHolographicMemory, WasmResonator, initHolographic } from 'amari-wasm';
//!
//! initHolographic();
//!
//! // Create holographic memory (256-dimensional ProductClifford algebra)
//! const memory = new WasmHolographicMemory();
//!
//! // Generate random keys and values
//! const key = WasmHolographicMemory.randomVersor(2);  // Product of 2 vectors
//! const value = WasmHolographicMemory.randomVersor(2);
//!
//! // Store and retrieve
//! memory.store(key, value);
//! const retrieved = memory.retrieve(key);
//!
//! // Check capacity
//! console.log(`Items: ${memory.itemCount()} / ${memory.theoreticalCapacity()}`);
//! console.log(`Near capacity: ${memory.isNearCapacity()}`);
//!
//! // Create resonator for cleanup
//! const codebook = [key1, key2, key3].flat();  // Flattened codebook
//! const resonator = new WasmResonator(codebook);
//! const cleanedUp = resonator.cleanupWithInfo(noisyInput);
//! ```
//!
//! For TropicalDualClifford binding operations:
//!
//! ```javascript
//! import { WasmTropicalDualClifford } from 'amari-wasm';
//!
//! const key = WasmTropicalDualClifford.randomVector();
//! const value = WasmTropicalDualClifford.randomVector();
//!
//! // Binding operations
//! const bound = key.bind(value);      // Create association
//! const retrieved = key.unbind(bound); // Retrieve value
//! const bundled = key.bundle(value, 1.0); // Superposition
//!
//! // Similarity
//! const sim = key.similarity(value);
//! const cliffordSim = key.cliffordSimilarity(value);
//! ```

use amari_core::{rotor::Rotor, Bivector, Multivector};
use std::cell::RefCell;
use wasm_bindgen::prelude::*;

// Optional modules - some enabled for expanded WASM functionality
pub mod automata; // Enabled for v0.9.4 - Cellular automata, inverse design, self-assembly for web
pub mod calculus; // Enabled for v0.11.0 - Differential calculus, manifolds, and Riemannian geometry for web
pub mod dual; // Enabled for v0.9.3 - automatic differentiation for machine learning in web
pub mod enumerative; // Enabled for v0.9.4 - Enumerative geometry and intersection theory for web
pub mod fusion; // Enabled for v0.9.4 - TropicalDualClifford system for LLM evaluation in web
pub mod info_geom; // Enabled for v0.9.4 - Information geometry and statistical manifolds for web
pub mod measure; // Enabled for v0.10.0 - Measure theory and Lebesgue integration for web
pub mod network; // Enabled for v0.9.4 - Geometric network analysis for web
pub mod optimization; // Enabled for v0.9.7 - Advanced optimization algorithms for web
pub mod relativistic;
pub mod tropical; // Enabled for v0.9.3 - critical for optimization algorithms in web

/// Number of coefficients in a 3D Clifford algebra multivector (2^3 = 8)
/// Basis elements: 1, e1, e2, e3, e12, e13, e23, e123
const MULTIVECTOR_COEFFICIENTS: usize = 8;

/// Console logging utility
#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console)]
    fn log(s: &str);
}

macro_rules! console_log {
    ($($t:tt)*) => (log(&format_args!($($t)*).to_string()))
}

/// WASM wrapper for Multivector with TypedArray support
#[wasm_bindgen]
pub struct WasmMultivector {
    inner: Multivector<3, 0, 0>, // Default to 3D Euclidean for now
}

impl Default for WasmMultivector {
    fn default() -> Self {
        Self::new()
    }
}

#[wasm_bindgen]
impl WasmMultivector {
    /// Create a new zero multivector
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            inner: Multivector::zero(),
        }
    }

    /// Create from a Float64Array of coefficients
    #[wasm_bindgen(js_name = fromCoefficients)]
    pub fn from_coefficients(coefficients: &[f64]) -> Result<WasmMultivector, JsValue> {
        if coefficients.len() != MULTIVECTOR_COEFFICIENTS {
            return Err(JsValue::from_str(
                "3D Clifford algebra requires exactly 8 coefficients",
            ));
        }

        Ok(Self {
            inner: Multivector::from_coefficients(coefficients.to_vec()),
        })
    }

    /// Create a scalar multivector
    #[wasm_bindgen(js_name = scalar)]
    pub fn scalar(value: f64) -> Self {
        Self {
            inner: Multivector::scalar(value),
        }
    }

    /// Create a basis vector (0-indexed)
    #[wasm_bindgen(js_name = basisVector)]
    pub fn basis_vector(index: usize) -> Result<WasmMultivector, JsValue> {
        if index >= 3 {
            return Err(JsValue::from_str(
                "Basis vector index must be 0, 1, or 2 for 3D",
            ));
        }

        Ok(Self {
            inner: Multivector::basis_vector(index),
        })
    }

    /// Get coefficients as a Float64Array
    #[wasm_bindgen(js_name = getCoefficients)]
    pub fn get_coefficients(&self) -> Vec<f64> {
        let mut coeffs = vec![0.0; MULTIVECTOR_COEFFICIENTS];
        #[allow(clippy::needless_range_loop)]
        for i in 0..MULTIVECTOR_COEFFICIENTS {
            coeffs[i] = self.inner.get(i);
        }
        coeffs
    }

    /// Get a specific coefficient
    #[wasm_bindgen(js_name = getCoefficient)]
    pub fn get_coefficient(&self, index: usize) -> f64 {
        self.inner.get(index)
    }

    /// Set a specific coefficient
    #[wasm_bindgen(js_name = setCoefficient)]
    pub fn set_coefficient(&mut self, index: usize, value: f64) {
        self.inner.set(index, value);
    }

    /// Geometric product
    #[wasm_bindgen(js_name = geometricProduct)]
    pub fn geometric_product(&self, other: &WasmMultivector) -> WasmMultivector {
        Self {
            inner: self.inner.geometric_product(&other.inner),
        }
    }

    /// Inner product (dot product for vectors)
    #[wasm_bindgen(js_name = innerProduct)]
    pub fn inner_product(&self, other: &WasmMultivector) -> WasmMultivector {
        Self {
            inner: self.inner.inner_product(&other.inner),
        }
    }

    /// Outer product (wedge product)
    #[wasm_bindgen(js_name = outerProduct)]
    pub fn outer_product(&self, other: &WasmMultivector) -> WasmMultivector {
        Self {
            inner: self.inner.outer_product(&other.inner),
        }
    }

    /// Scalar product
    #[wasm_bindgen(js_name = scalarProduct)]
    pub fn scalar_product(&self, other: &WasmMultivector) -> f64 {
        self.inner.scalar_product(&other.inner)
    }

    /// Reverse
    pub fn reverse(&self) -> WasmMultivector {
        Self {
            inner: self.inner.reverse(),
        }
    }

    /// Grade projection
    #[wasm_bindgen(js_name = gradeProjection)]
    pub fn grade_projection(&self, grade: usize) -> WasmMultivector {
        Self {
            inner: self.inner.grade_projection(grade),
        }
    }

    /// Exponential (for bivectors to create rotors)
    pub fn exp(&self) -> WasmMultivector {
        Self {
            inner: self.inner.exp(),
        }
    }

    /// Compute magnitude
    pub fn magnitude(&self) -> f64 {
        self.inner.magnitude()
    }

    /// Compute norm (alias for magnitude, maintained for compatibility)
    pub fn norm(&self) -> f64 {
        self.magnitude()
    }

    /// Normalize
    pub fn normalize(&self) -> Result<WasmMultivector, JsValue> {
        self.inner
            .normalize()
            .map(|mv| Self { inner: mv })
            .ok_or_else(|| JsValue::from_str("Cannot normalize zero multivector"))
    }

    /// Compute inverse
    pub fn inverse(&self) -> Result<WasmMultivector, JsValue> {
        self.inner
            .inverse()
            .map(|mv| Self { inner: mv })
            .ok_or_else(|| JsValue::from_str("Multivector is not invertible"))
    }

    /// Add two multivectors
    pub fn add(&self, other: &WasmMultivector) -> WasmMultivector {
        Self {
            inner: &self.inner + &other.inner,
        }
    }

    /// Subtract two multivectors
    pub fn sub(&self, other: &WasmMultivector) -> WasmMultivector {
        Self {
            inner: &self.inner - &other.inner,
        }
    }

    /// Scale by a scalar
    pub fn scale(&self, scalar: f64) -> WasmMultivector {
        Self {
            inner: &self.inner * scalar,
        }
    }
}

/// Batch operations for performance
#[wasm_bindgen]
pub struct BatchOperations;

#[wasm_bindgen]
impl BatchOperations {
    /// Batch geometric product: compute a[i] * b[i] for all i
    /// Optimized for WebAssembly performance with reduced allocations
    #[wasm_bindgen(js_name = batchGeometricProduct)]
    pub fn batch_geometric_product(a_batch: &[f64], b_batch: &[f64]) -> Result<Vec<f64>, JsValue> {
        let batch_size = a_batch.len() / MULTIVECTOR_COEFFICIENTS;

        if !a_batch.len().is_multiple_of(MULTIVECTOR_COEFFICIENTS)
            || !b_batch.len().is_multiple_of(MULTIVECTOR_COEFFICIENTS)
        {
            return Err(JsValue::from_str(
                "Batch arrays must have length divisible by multivector coefficients",
            ));
        }

        if a_batch.len() != b_batch.len() {
            return Err(JsValue::from_str("Batch arrays must have the same length"));
        }

        // Pre-allocate result vector to avoid repeated allocations
        let mut result = vec![0.0; a_batch.len()];

        // Use optimized batch processing with minimal allocations
        for i in 0..batch_size {
            let start = i * MULTIVECTOR_COEFFICIENTS;

            // Direct coefficient access without intermediate vector allocation
            let a_coeffs = &a_batch[start..start + MULTIVECTOR_COEFFICIENTS];
            let b_coeffs = &b_batch[start..start + MULTIVECTOR_COEFFICIENTS];

            // Inline geometric product computation for WASM optimization
            Self::geometric_product_hot_path(
                a_coeffs,
                b_coeffs,
                &mut result[start..start + MULTIVECTOR_COEFFICIENTS],
            );
        }

        Ok(result)
    }

    /// Hot path optimized geometric product for WASM
    /// Computes geometric product directly on coefficient slices
    #[inline(always)]
    fn geometric_product_hot_path(a: &[f64], b: &[f64], result: &mut [f64]) {
        // Manually unrolled geometric product for 3D Euclidean space
        // Based on multiplication table for Cl(3,0,0)

        // Clear result
        result.fill(0.0);

        // Scalar * all
        let a0 = a[0];
        if a0 != 0.0 {
            for i in 0..8 {
                result[i] += a0 * b[i];
            }
        }

        // e1 products (index 1)
        let a1 = a[1];
        if a1 != 0.0 {
            result[0] += a1 * b[1]; // e1 * 1 = e1, 1 * e1 = e1
            result[1] += a1 * b[0]; // 1 * e1 = e1
            result[2] += a1 * b[3]; // e1 * e2 = e12
            result[3] += a1 * b[2]; // e1 * e12 = e2
            result[4] += a1 * b[5]; // e1 * e3 = e13
            result[5] += a1 * b[4]; // e1 * e13 = e3
            result[6] -= a1 * b[7]; // e1 * e23 = -e123
            result[7] -= a1 * b[6]; // e1 * e123 = -e23
        }

        // e2 products (index 2)
        let a2 = a[2];
        if a2 != 0.0 {
            result[0] += a2 * b[2]; // e2 * 1 = e2
            result[1] -= a2 * b[3]; // e2 * e1 = -e12
            result[2] += a2 * b[0]; // 1 * e2 = e2
            result[3] -= a2 * b[1]; // e2 * e12 = -e1
            result[4] += a2 * b[6]; // e2 * e3 = e23
            result[5] += a2 * b[7]; // e2 * e13 = e123
            result[6] += a2 * b[4]; // e2 * e23 = e3
            result[7] += a2 * b[5]; // e2 * e123 = e13
        }

        // e3 products (index 4)
        let a4 = a[4];
        if a4 != 0.0 {
            result[0] += a4 * b[4]; // e3 * 1 = e3
            result[1] -= a4 * b[5]; // e3 * e1 = -e13
            result[2] -= a4 * b[6]; // e3 * e2 = -e23
            result[3] -= a4 * b[7]; // e3 * e12 = -e123
            result[4] += a4 * b[0]; // 1 * e3 = e3
            result[5] -= a4 * b[1]; // e3 * e13 = -e1
            result[6] -= a4 * b[2]; // e3 * e23 = -e2
            result[7] -= a4 * b[3]; // e3 * e123 = -e12
        }

        // e12 products (index 3)
        let a3 = a[3];
        if a3 != 0.0 {
            result[0] -= a3 * b[3]; // e12 * 1 = e12, e12^2 = -1
            result[1] += a3 * b[2]; // e12 * e1 = e2
            result[2] -= a3 * b[1]; // e12 * e2 = -e1
            result[3] += a3 * b[0]; // 1 * e12 = e12
            result[4] += a3 * b[7]; // e12 * e3 = e123
            result[5] -= a3 * b[6]; // e12 * e13 = -e23
            result[6] += a3 * b[5]; // e12 * e23 = e13
            result[7] += a3 * b[4]; // e12 * e123 = e3
        }

        // e13 products (index 5)
        let a5 = a[5];
        if a5 != 0.0 {
            result[0] -= a5 * b[5]; // e13^2 = -1
            result[1] += a5 * b[4]; // e13 * e1 = e3
            result[2] -= a5 * b[7]; // e13 * e2 = -e123
            result[3] += a5 * b[6]; // e13 * e12 = e23
            result[4] -= a5 * b[1]; // e13 * e3 = -e1
            result[5] += a5 * b[0]; // 1 * e13 = e13
            result[6] -= a5 * b[3]; // e13 * e23 = -e12
            result[7] -= a5 * b[2]; // e13 * e123 = -e2
        }

        // e23 products (index 6)
        let a6 = a[6];
        if a6 != 0.0 {
            result[0] -= a6 * b[6]; // e23^2 = -1
            result[1] += a6 * b[7]; // e23 * e1 = e123
            result[2] += a6 * b[4]; // e23 * e2 = e3
            result[3] -= a6 * b[5]; // e23 * e12 = -e13
            result[4] -= a6 * b[2]; // e23 * e3 = -e2
            result[5] += a6 * b[3]; // e23 * e13 = e12
            result[6] += a6 * b[0]; // 1 * e23 = e23
            result[7] += a6 * b[1]; // e23 * e123 = e1
        }

        // e123 products (index 7)
        let a7 = a[7];
        if a7 != 0.0 {
            result[0] -= a7 * b[7]; // e123^2 = -1
            result[1] -= a7 * b[6]; // e123 * e1 = -e23
            result[2] += a7 * b[5]; // e123 * e2 = e13
            result[3] -= a7 * b[4]; // e123 * e12 = -e3
            result[4] += a7 * b[3]; // e123 * e3 = e12
            result[5] -= a7 * b[2]; // e123 * e13 = -e2
            result[6] += a7 * b[1]; // e123 * e23 = e1
            result[7] += a7 * b[0]; // 1 * e123 = e123
        }
    }

    /// Batch addition
    #[wasm_bindgen(js_name = batchAdd)]
    pub fn batch_add(a_batch: &[f64], b_batch: &[f64]) -> Result<Vec<f64>, JsValue> {
        if a_batch.len() != b_batch.len() {
            return Err(JsValue::from_str("Batch arrays must have the same length"));
        }

        let mut result = Vec::with_capacity(a_batch.len());
        for i in 0..a_batch.len() {
            result.push(a_batch[i] + b_batch[i]);
        }

        Ok(result)
    }
}

/// Rotor operations for WASM
#[wasm_bindgen]
pub struct WasmRotor {
    inner: Rotor<3, 0, 0>,
}

#[wasm_bindgen]
impl WasmRotor {
    /// Create a rotor from a bivector and angle
    #[wasm_bindgen(js_name = fromBivector)]
    pub fn from_bivector(bivector: &WasmMultivector, angle: f64) -> WasmRotor {
        // Convert the WasmMultivector to a Bivector wrapper for type safety.
        // The Bivector type ensures only grade-2 components are used, providing
        // compile-time guarantees that the rotor is constructed from valid bivector data.
        let biv = Bivector::from_multivector(&bivector.inner);
        Self {
            inner: Rotor::from_bivector(&biv, angle),
        }
    }

    /// Apply rotor to a multivector
    pub fn apply(&self, mv: &WasmMultivector) -> WasmMultivector {
        WasmMultivector {
            inner: self.inner.apply(&mv.inner),
        }
    }

    /// Compose two rotors
    pub fn compose(&self, other: &WasmRotor) -> WasmRotor {
        Self {
            inner: self.inner.compose(&other.inner),
        }
    }

    /// Get inverse rotor
    pub fn inverse(&self) -> WasmRotor {
        Self {
            inner: self.inner.inverse(),
        }
    }
}

/// Memory pool for reducing allocation overhead in WASM
struct MemoryPool {
    coefficient_buffers: Vec<Vec<f64>>,
}

impl MemoryPool {
    fn new() -> Self {
        Self {
            coefficient_buffers: Vec::new(),
        }
    }

    fn get_buffer(&mut self) -> Vec<f64> {
        self.coefficient_buffers
            .pop()
            .unwrap_or_else(|| Vec::with_capacity(MULTIVECTOR_COEFFICIENTS))
    }

    #[allow(dead_code)]
    fn return_buffer(&mut self, mut buffer: Vec<f64>) {
        if buffer.capacity() >= MULTIVECTOR_COEFFICIENTS {
            buffer.clear();
            if self.coefficient_buffers.len() < 16 {
                // Limit pool size
                self.coefficient_buffers.push(buffer);
            }
        }
    }
}

thread_local! {
    static MEMORY_POOL: RefCell<MemoryPool> = RefCell::new(MemoryPool::new());
}

/// High-performance WASM operations with memory pooling
#[wasm_bindgen]
pub struct PerformanceOperations;

#[wasm_bindgen]
impl PerformanceOperations {
    /// Fast geometric product for hot paths with memory pooling
    #[wasm_bindgen(js_name = fastGeometricProduct)]
    pub fn fast_geometric_product(lhs: &[f64], rhs: &[f64]) -> Vec<f64> {
        if lhs.len() != MULTIVECTOR_COEFFICIENTS || rhs.len() != MULTIVECTOR_COEFFICIENTS {
            return vec![0.0; MULTIVECTOR_COEFFICIENTS];
        }

        MEMORY_POOL.with(|pool| {
            let mut pool = pool.borrow_mut();
            let mut result = pool.get_buffer();
            result.resize(MULTIVECTOR_COEFFICIENTS, 0.0);

            BatchOperations::geometric_product_hot_path(lhs, rhs, &mut result);

            // Don't return buffer to pool since we're returning it to JS
            result
        })
    }

    /// Optimized vector operations for 3D space
    #[wasm_bindgen(js_name = vectorCrossProduct)]
    pub fn vector_cross_product(v1: &[f64], v2: &[f64]) -> Vec<f64> {
        if v1.len() < 3 || v2.len() < 3 {
            return vec![0.0; 3];
        }

        vec![
            v1[1] * v2[2] - v1[2] * v2[1],
            v1[2] * v2[0] - v1[0] * v2[2],
            v1[0] * v2[1] - v1[1] * v2[0],
        ]
    }

    /// Optimized vector dot product
    #[wasm_bindgen(js_name = vectorDotProduct)]
    pub fn vector_dot_product(v1: &[f64], v2: &[f64]) -> f64 {
        let len = v1.len().min(v2.len());
        let mut result = 0.0;
        for i in 0..len {
            result += v1[i] * v2[i];
        }
        result
    }

    /// Batch normalize vectors for efficiency
    #[wasm_bindgen(js_name = batchNormalize)]
    pub fn batch_normalize(vectors: &[f64], vector_size: usize) -> Vec<f64> {
        let num_vectors = vectors.len() / vector_size;
        let mut result = Vec::with_capacity(vectors.len());

        for i in 0..num_vectors {
            let start = i * vector_size;
            let end = start + vector_size;
            let vector = &vectors[start..end];

            // Calculate magnitude
            let mag_sq: f64 = vector.iter().map(|x| x * x).sum();
            let mag = mag_sq.sqrt();

            if mag > 1e-14 {
                let inv_mag = 1.0 / mag;
                for &component in vector {
                    result.push(component * inv_mag);
                }
            } else {
                result.extend(vec![0.0; vector_size]);
            }
        }

        result
    }
}

/// Initialize the WASM module
#[wasm_bindgen(start)]
pub fn init() {
    // Initialize console error panic hook
    std::panic::set_hook(Box::new(console_error_panic_hook::hook));

    console_log!("Amari WASM module initialized with complete mathematical computing support");
}

#[cfg(test)]
mod tests {
    use super::*;
    use wasm_bindgen_test::*;

    #[wasm_bindgen_test]
    #[allow(dead_code)]
    fn test_basic_operations() {
        let e1 = WasmMultivector::basis_vector(0).unwrap();
        let e2 = WasmMultivector::basis_vector(1).unwrap();

        let e12 = e1.outer_product(&e2);
        assert_eq!(e12.get_coefficient(3), 1.0);
    }
}
