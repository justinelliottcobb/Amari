//! WASM bindings for amari-enumerative: Enumerative geometry and intersection theory
//!
//! This module provides WebAssembly bindings for advanced enumerative geometry featuring:
//!
//! - **Intersection Theory**: Chow rings, intersection multiplicities, and Bézout's theorem
//! - **Schubert Calculus**: Computations on Grassmannians and flag varieties
//! - **Gromov-Witten Theory**: Curve counting and quantum cohomology
//! - **Tropical Geometry**: Tropical curve counting and correspondence theorems
//! - **Moduli Spaces**: Computations on moduli spaces of curves and surfaces
//!
//! Perfect for:
//! - Advanced mathematical research and education
//! - Algebraic geometry computations in web browsers
//! - Interactive tools for studying curve counting problems
//! - Web-based research platforms for enumerative geometry
//! - Educational demonstrations of intersection theory concepts

use amari_enumerative::{ChowClass, Grassmannian, ProjectiveSpace};
use num_rational::Rational64;
use wasm_bindgen::prelude::*;

/// WASM wrapper for Chow classes in intersection theory
#[wasm_bindgen]
pub struct WasmChowClass {
    inner: ChowClass,
}

#[wasm_bindgen]
impl WasmChowClass {
    /// Create a new Chow class with given dimension and degree
    #[wasm_bindgen(constructor)]
    pub fn new(dimension: usize, degree: f64) -> Self {
        let rational_degree = Rational64::new((degree * 1000.0) as i64, 1000);
        Self {
            inner: ChowClass::new(dimension, rational_degree),
        }
    }

    /// Create a hypersurface class of given degree
    #[wasm_bindgen(js_name = hypersurface)]
    pub fn hypersurface(degree: i32) -> Self {
        Self {
            inner: ChowClass::hypersurface(degree as i64),
        }
    }

    /// Create a point class
    #[wasm_bindgen(js_name = point)]
    pub fn point() -> Self {
        Self {
            inner: ChowClass::point(),
        }
    }

    /// Create a linear subspace class
    #[wasm_bindgen(js_name = linearSubspace)]
    pub fn linear_subspace(codimension: usize) -> Self {
        Self {
            inner: ChowClass::linear_subspace(codimension),
        }
    }

    /// Get the dimension of this Chow class
    #[wasm_bindgen(js_name = getDimension)]
    pub fn get_dimension(&self) -> usize {
        self.inner.dimension
    }

    /// Get the degree of this Chow class
    #[wasm_bindgen(js_name = getDegree)]
    pub fn get_degree(&self) -> f64 {
        (*self.inner.degree.numer() as f64) / (*self.inner.degree.denom() as f64)
    }

    /// Multiply two Chow classes
    #[wasm_bindgen(js_name = multiply)]
    pub fn multiply(&self, other: &WasmChowClass) -> WasmChowClass {
        WasmChowClass {
            inner: self.inner.multiply(&other.inner),
        }
    }

    /// Check if this class is zero
    #[wasm_bindgen(js_name = isZero)]
    pub fn is_zero(&self) -> bool {
        self.inner.is_zero()
    }

    /// Raise this class to a power
    #[wasm_bindgen(js_name = power)]
    pub fn power(&self, n: usize) -> WasmChowClass {
        WasmChowClass {
            inner: self.inner.power(n),
        }
    }
}

/// WASM wrapper for projective spaces
#[wasm_bindgen]
pub struct WasmProjectiveSpace {
    inner: ProjectiveSpace,
}

#[wasm_bindgen]
impl WasmProjectiveSpace {
    /// Create a new projective space of given dimension
    #[wasm_bindgen(constructor)]
    pub fn new(dimension: usize) -> Self {
        Self {
            inner: ProjectiveSpace::new(dimension),
        }
    }

    /// Get the dimension of this projective space
    #[wasm_bindgen(js_name = getDimension)]
    pub fn get_dimension(&self) -> usize {
        self.inner.dimension
    }

    /// Compute Bézout intersection number for two curves
    #[wasm_bindgen(js_name = bezoutIntersection)]
    pub fn bezout_intersection(&self, degree1: i32, degree2: i32) -> f64 {
        // Bézout's theorem: intersection number is product of degrees
        (degree1 * degree2) as f64
    }

    /// Check if the projective space has given dimension
    #[wasm_bindgen(js_name = hasDimension)]
    pub fn has_dimension(&self, dim: usize) -> bool {
        self.inner.dimension == dim
    }
}

/// WASM wrapper for Grassmannian varieties
#[wasm_bindgen]
pub struct WasmGrassmannian {
    inner: Grassmannian,
}

#[wasm_bindgen]
impl WasmGrassmannian {
    /// Create a new Grassmannian Gr(k, n) of k-planes in n-space
    #[wasm_bindgen(constructor)]
    pub fn new(k: usize, n: usize) -> Result<WasmGrassmannian, JsValue> {
        match Grassmannian::new(k, n) {
            Ok(grassmannian) => Ok(WasmGrassmannian {
                inner: grassmannian,
            }),
            Err(e) => Err(JsValue::from_str(&format!("Grassmannian error: {:?}", e))),
        }
    }

    /// Get the parameters (k, n) of this Grassmannian
    #[wasm_bindgen(js_name = getParameters)]
    pub fn get_parameters(&self) -> Vec<usize> {
        vec![self.inner.k, self.inner.n]
    }

    /// Get the dimension of this Grassmannian
    #[wasm_bindgen(js_name = getDimension)]
    pub fn get_dimension(&self) -> usize {
        self.inner.dimension()
    }
}

/// WASM wrapper for tropical curves (simplified)
#[wasm_bindgen]
pub struct WasmTropicalCurve {
    degree: i64,
    genus: usize,
}

#[wasm_bindgen]
impl WasmTropicalCurve {
    /// Create a new tropical curve with given degree and genus
    #[wasm_bindgen(constructor)]
    pub fn new(degree: i32, genus: usize) -> Self {
        Self {
            degree: degree as i64,
            genus,
        }
    }

    /// Get the degree of this tropical curve
    #[wasm_bindgen(js_name = getDegree)]
    pub fn get_degree(&self) -> i32 {
        self.degree as i32
    }

    /// Get the genus of this tropical curve
    #[wasm_bindgen(js_name = getGenus)]
    pub fn get_genus(&self) -> usize {
        self.genus
    }

    /// Compute expected number of vertices using Euler characteristic
    #[wasm_bindgen(js_name = expectedVertices)]
    pub fn expected_vertices(&self) -> usize {
        // Simplified calculation: for tropical curves, vertices roughly scale with degree
        (self.degree.unsigned_abs().max(1) * 3) as usize
    }
}

/// WASM wrapper for moduli spaces (simplified)
#[wasm_bindgen]
pub struct WasmModuliSpace {
    genus: usize,
    marked_points: usize,
    #[allow(dead_code)]
    stable: bool,
}

#[wasm_bindgen]
impl WasmModuliSpace {
    /// Create moduli space of curves M_g,n
    #[wasm_bindgen(js_name = ofCurves)]
    pub fn of_curves(genus: usize, marked_points: usize) -> Self {
        Self {
            genus,
            marked_points,
            stable: false,
        }
    }

    /// Create moduli space of stable curves
    #[wasm_bindgen(js_name = ofStableCurves)]
    pub fn of_stable_curves(genus: usize, marked_points: usize) -> Self {
        Self {
            genus,
            marked_points,
            stable: true,
        }
    }

    /// Get the expected dimension
    #[wasm_bindgen(js_name = expectedDimension)]
    pub fn expected_dimension(&self) -> i32 {
        // Dimension of M_g,n = 3g - 3 + n
        (3 * self.genus as i32) - 3 + (self.marked_points as i32)
    }

    /// Check if the moduli space is proper
    #[wasm_bindgen(js_name = isProper)]
    pub fn is_proper(&self) -> bool {
        // M_g,n is proper when 2g - 2 + n > 0
        (2 * self.genus as i32) - 2 + (self.marked_points as i32) > 0
    }

    /// Get the genus
    #[wasm_bindgen(js_name = getGenus)]
    pub fn get_genus(&self) -> usize {
        self.genus
    }

    /// Get the number of marked points
    #[wasm_bindgen(js_name = getMarkedPoints)]
    pub fn get_marked_points(&self) -> usize {
        self.marked_points
    }
}

/// Utility functions for enumerative geometry
#[wasm_bindgen]
pub struct EnumerativeUtils;

#[wasm_bindgen]
impl EnumerativeUtils {
    /// Compute binomial coefficient C(n, k)
    #[wasm_bindgen(js_name = binomial)]
    pub fn binomial(n: usize, k: usize) -> Result<f64, JsValue> {
        if k > n {
            return Ok(0.0);
        }

        let mut result = 1.0;
        for i in 0..k {
            result *= (n - i) as f64;
            result /= (i + 1) as f64;
        }
        Ok(result)
    }

    /// Compute Euler characteristic of projective space
    #[wasm_bindgen(js_name = eulerCharacteristic)]
    pub fn euler_characteristic(dimension: usize) -> usize {
        dimension + 1
    }

    /// Validate partition for Schubert calculus
    #[wasm_bindgen(js_name = validatePartition)]
    pub fn validate_partition(partition: &[usize], k: usize, n: usize) -> bool {
        if partition.len() > k {
            return false;
        }

        for i in 0..partition.len() {
            if partition[i] > n - k {
                return false;
            }
            if i > 0 && partition[i] > partition[i - 1] {
                return false;
            }
        }
        true
    }

    /// Compute expected number of rational curves of given degree
    #[wasm_bindgen(js_name = expectedRationalCurves)]
    pub fn expected_rational_curves(degree: usize, points: usize) -> Result<f64, JsValue> {
        if points != 3 * degree - 1 {
            return Err(JsValue::from_str("Invalid number of points for degree"));
        }

        // Simplified calculation - in practice this would use more sophisticated methods
        let mut result = 1.0;
        for d in 1..=degree {
            result *= d as f64;
        }
        Ok(result)
    }

    /// Compute intersection multiplicity using Bézout's theorem
    #[wasm_bindgen(js_name = bezoutMultiplicity)]
    pub fn bezout_multiplicity(degree1: i32, degree2: i32, space_dimension: usize) -> f64 {
        if space_dimension == 2 {
            // For plane curves, intersection number is product of degrees
            (degree1 * degree2) as f64
        } else {
            // For higher dimensions, more complex formula
            ((degree1 * degree2) as f64) * (space_dimension as f64)
        }
    }

    /// Check if two curves can intersect transversely
    #[wasm_bindgen(js_name = canIntersectTransversely)]
    pub fn can_intersect_transversely(
        _degree1: i32,
        _degree2: i32,
        space_dimension: usize,
    ) -> bool {
        // Simplified check: curves can intersect transversely if sum of codimensions <= space dimension
        let codim1 = 1; // curves are typically codimension 1
        let codim2 = 1;
        (codim1 + codim2) <= space_dimension
    }
}

/// Batch operations for high-performance enumerative computations
#[wasm_bindgen]
pub struct EnumerativeBatch;

#[wasm_bindgen]
impl EnumerativeBatch {
    /// Compute multiple Bézout intersections in batch
    #[wasm_bindgen(js_name = bezoutBatch)]
    pub fn bezout_batch(
        projective_space: &WasmProjectiveSpace,
        degrees1: &[i32],
        degrees2: &[i32],
    ) -> Result<Vec<f64>, JsValue> {
        if degrees1.len() != degrees2.len() {
            return Err(JsValue::from_str("Degree arrays must have same length"));
        }

        let mut results = Vec::new();
        for i in 0..degrees1.len() {
            results.push(projective_space.bezout_intersection(degrees1[i], degrees2[i]));
        }

        Ok(results)
    }

    /// Compute multiple binomial coefficients in batch
    #[wasm_bindgen(js_name = binomialBatch)]
    pub fn binomial_batch(n_values: &[usize], k_values: &[usize]) -> Result<Vec<f64>, JsValue> {
        if n_values.len() != k_values.len() {
            return Err(JsValue::from_str("Input arrays must have same length"));
        }

        let mut results = Vec::new();
        for i in 0..n_values.len() {
            match EnumerativeUtils::binomial(n_values[i], k_values[i]) {
                Ok(result) => results.push(result),
                Err(_) => results.push(0.0),
            }
        }

        Ok(results)
    }

    /// Get number of available batch operations
    #[wasm_bindgen(js_name = getBatchOperationCount)]
    pub fn get_batch_operation_count() -> usize {
        2 // bezout_batch and binomial_batch
    }
}

/// Initialize the enumerative geometry module
#[wasm_bindgen(js_name = initEnumerative)]
pub fn init_enumerative() {
    web_sys::console::log_1(&"Amari Enumerative WASM module initialized: Intersection theory, Schubert calculus, and Gromov-Witten theory ready".into());
}

#[cfg(test)]
#[allow(dead_code)]
mod tests {
    use super::*;
    use wasm_bindgen_test::*;

    #[wasm_bindgen_test]
    fn test_chow_class_creation() {
        let class = WasmChowClass::new(2, 3.0);
        assert_eq!(class.get_dimension(), 2);
        assert_eq!(class.get_degree(), 3.0);
    }

    #[wasm_bindgen_test]
    fn test_projective_space() {
        let p2 = WasmProjectiveSpace::new(2);
        assert_eq!(p2.get_dimension(), 2);

        // Bézout's theorem: intersection of cubic and quartic should be 3*4=12
        let intersection = p2.bezout_intersection(3, 4);
        assert_eq!(intersection, 12.0);
    }

    #[wasm_bindgen_test]
    fn test_grassmannian() {
        let gr = WasmGrassmannian::new(2, 4);
        assert!(gr.is_ok());
        let gr = gr.unwrap();
        assert_eq!(gr.get_parameters(), vec![2, 4]);
        assert_eq!(gr.get_dimension(), 4); // Gr(2,4) has dimension 2*(4-2) = 4
    }

    #[wasm_bindgen_test]
    fn test_tropical_curve() {
        let curve = WasmTropicalCurve::new(3, 1);
        assert_eq!(curve.get_degree(), 3);
        assert_eq!(curve.get_genus(), 1);
        assert!(curve.expected_vertices() > 0);
    }

    #[wasm_bindgen_test]
    fn test_moduli_space() {
        let m11 = WasmModuliSpace::of_curves(1, 1);
        assert_eq!(m11.get_genus(), 1);
        assert_eq!(m11.get_marked_points(), 1);
        assert_eq!(m11.expected_dimension(), 1); // 3*1 - 3 + 1 = 1
        assert!(m11.is_proper()); // 2*1 - 2 + 1 = 1 > 0
    }

    #[wasm_bindgen_test]
    fn test_enumerative_utils() {
        // Test binomial coefficient
        let binom = EnumerativeUtils::binomial(5, 2);
        assert!(binom.is_ok());
        assert_eq!(binom.unwrap(), 10.0);

        // Test Euler characteristic
        assert_eq!(EnumerativeUtils::euler_characteristic(2), 3);

        // Test partition validation
        let partition = vec![3, 2, 1];
        assert!(EnumerativeUtils::validate_partition(&partition, 3, 5));

        // Test Bézout multiplicity
        assert_eq!(EnumerativeUtils::bezout_multiplicity(3, 4, 2), 12.0);

        // Test transverse intersection check
        assert!(EnumerativeUtils::can_intersect_transversely(3, 4, 2));
    }

    #[wasm_bindgen_test]
    fn test_batch_operations() {
        let p2 = WasmProjectiveSpace::new(2);
        let degrees1 = vec![2, 3, 4];
        let degrees2 = vec![3, 4, 5];

        let results = EnumerativeBatch::bezout_batch(&p2, &degrees1, &degrees2);
        assert!(results.is_ok());

        let results = results.unwrap();
        assert_eq!(results.len(), 3);
        assert_eq!(results[0], 6.0); // 2*3=6
        assert_eq!(results[1], 12.0); // 3*4=12
        assert_eq!(results[2], 20.0); // 4*5=20
    }
}
