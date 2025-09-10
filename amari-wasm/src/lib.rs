//! WASM bindings for the Amari geometric algebra library

use wasm_bindgen::prelude::*;
use amari_core::{Multivector, rotor::Rotor};

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
        if coefficients.len() != 8 {
            return Err(JsValue::from_str("3D Clifford algebra requires exactly 8 coefficients"));
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
            return Err(JsValue::from_str("Basis vector index must be 0, 1, or 2 for 3D"));
        }
        
        Ok(Self {
            inner: Multivector::basis_vector(index),
        })
    }
    
    /// Get coefficients as a Float64Array
    #[wasm_bindgen(js_name = getCoefficients)]
    pub fn get_coefficients(&self) -> Vec<f64> {
        let mut coeffs = vec![0.0; 8];
        for i in 0..8 {
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
    
    /// Compute norm
    pub fn norm(&self) -> f64 {
        self.inner.norm()
    }
    
    /// Normalize
    pub fn normalize(&self) -> Result<WasmMultivector, JsValue> {
        self.inner.normalize()
            .map(|mv| Self { inner: mv })
            .ok_or_else(|| JsValue::from_str("Cannot normalize zero multivector"))
    }
    
    /// Compute inverse
    pub fn inverse(&self) -> Result<WasmMultivector, JsValue> {
        self.inner.inverse()
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
    #[wasm_bindgen(js_name = batchGeometricProduct)]
    pub fn batch_geometric_product(a_batch: &[f64], b_batch: &[f64]) -> Result<Vec<f64>, JsValue> {
        let batch_size = a_batch.len() / 8;
        
        if a_batch.len() % 8 != 0 || b_batch.len() % 8 != 0 {
            return Err(JsValue::from_str("Batch arrays must have length divisible by 8"));
        }
        
        if a_batch.len() != b_batch.len() {
            return Err(JsValue::from_str("Batch arrays must have the same length"));
        }
        
        let mut result = Vec::with_capacity(a_batch.len());
        
        for i in 0..batch_size {
            let start = i * 8;
            let end = start + 8;
            
            let a = Multivector::<3, 0, 0>::from_coefficients(a_batch[start..end].to_vec());
            let b = Multivector::<3, 0, 0>::from_coefficients(b_batch[start..end].to_vec());
            let product = a.geometric_product(&b);
            
            for j in 0..8 {
                result.push(product.get(j));
            }
        }
        
        Ok(result)
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
        Self {
            inner: Rotor::from_bivector(&bivector.inner, angle),
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

/// Initialize the WASM module
#[wasm_bindgen(start)]
pub fn init() {
    console_log!("Amari WASM module initialized");
}

#[cfg(test)]
mod tests {
    use super::*;
    use wasm_bindgen_test::*;
    
    #[wasm_bindgen_test]
    fn test_basic_operations() {
        let e1 = WasmMultivector::basis_vector(0).unwrap();
        let e2 = WasmMultivector::basis_vector(1).unwrap();
        
        let e12 = e1.outer_product(&e2);
        assert_eq!(e12.get_coefficient(3), 1.0);
    }
}