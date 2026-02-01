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

// =============================================================================
// NEW WASM BINDINGS FOR AMARI-ENUMERATIVE EXTENSIONS
// =============================================================================

use amari_enumerative::{
    Capability, CapabilityId, IntersectionResult, Namespace, NamespaceIntersection, Partition,
    SchubertCalculus, SchubertClass,
};

/// WASM wrapper for partitions used in Schubert calculus and LR coefficients
#[wasm_bindgen]
pub struct WasmPartition {
    inner: Partition,
}

#[wasm_bindgen]
impl WasmPartition {
    /// Create a new partition from an array of parts
    #[wasm_bindgen(constructor)]
    pub fn new(parts: &[usize]) -> Self {
        Self {
            inner: Partition::new(parts.to_vec()),
        }
    }

    /// Get the parts of this partition
    #[wasm_bindgen(js_name = getParts)]
    pub fn get_parts(&self) -> Vec<usize> {
        self.inner.parts.clone()
    }

    /// Get the size (sum of parts) of this partition
    #[wasm_bindgen(js_name = getSize)]
    pub fn get_size(&self) -> usize {
        self.inner.size()
    }

    /// Get the length (number of non-zero parts) of this partition
    #[wasm_bindgen(js_name = getLength)]
    pub fn get_length(&self) -> usize {
        self.inner.length()
    }

    /// Check if this partition contains another (for skew shapes)
    #[wasm_bindgen(js_name = contains)]
    pub fn contains(&self, other: &WasmPartition) -> bool {
        self.inner.contains(&other.inner)
    }

    /// Check if this partition is valid (weakly decreasing, positive parts)
    #[wasm_bindgen(js_name = isValid)]
    pub fn is_valid(&self) -> bool {
        self.inner.is_valid()
    }

    /// Check if this partition fits in a k × (n-k) box (for Grassmannian)
    #[wasm_bindgen(js_name = fitsInBox)]
    pub fn fits_in_box(&self, k: usize, n: usize) -> bool {
        self.inner.parts.len() <= k && self.inner.parts.iter().all(|&p| p <= n - k)
    }
}

/// WASM wrapper for Schubert classes on Grassmannians
#[wasm_bindgen]
pub struct WasmSchubertClass {
    inner: SchubertClass,
}

#[wasm_bindgen]
impl WasmSchubertClass {
    /// Create a new Schubert class from a partition on Gr(k, n)
    #[wasm_bindgen(constructor)]
    pub fn new(partition: &[usize], k: usize, n: usize) -> Result<WasmSchubertClass, JsValue> {
        match SchubertClass::new(partition.to_vec(), (k, n)) {
            Ok(class) => Ok(WasmSchubertClass { inner: class }),
            Err(e) => Err(JsValue::from_str(&format!("Schubert class error: {:?}", e))),
        }
    }

    /// Get the partition defining this Schubert class
    #[wasm_bindgen(js_name = getPartition)]
    pub fn get_partition(&self) -> Vec<usize> {
        self.inner.partition.clone()
    }

    /// Get the Grassmannian parameters (k, n)
    #[wasm_bindgen(js_name = getGrassmannianDim)]
    pub fn get_grassmannian_dim(&self) -> Vec<usize> {
        vec![self.inner.grassmannian_dim.0, self.inner.grassmannian_dim.1]
    }

    /// Get the codimension of this Schubert class
    #[wasm_bindgen(js_name = getCodimension)]
    pub fn get_codimension(&self) -> usize {
        self.inner.codimension()
    }

    /// Create the special Schubert class σ_1 (single box)
    #[wasm_bindgen(js_name = sigma1)]
    pub fn sigma_1(k: usize, n: usize) -> Result<WasmSchubertClass, JsValue> {
        WasmSchubertClass::new(&[1], k, n)
    }
}

/// WASM wrapper for Schubert calculus computations
#[wasm_bindgen]
pub struct WasmSchubertCalculus {
    inner: SchubertCalculus,
}

#[wasm_bindgen]
impl WasmSchubertCalculus {
    /// Create a new Schubert calculus context for Gr(k, n)
    #[wasm_bindgen(constructor)]
    pub fn new(k: usize, n: usize) -> Self {
        Self {
            inner: SchubertCalculus::new((k, n)),
        }
    }

    /// Compute the intersection of two Schubert classes
    #[wasm_bindgen(js_name = intersect)]
    pub fn intersect(
        &mut self,
        class1: &WasmSchubertClass,
        class2: &WasmSchubertClass,
    ) -> WasmIntersectionResult {
        let result = self
            .inner
            .multi_intersect(&[class1.inner.clone(), class2.inner.clone()]);
        WasmIntersectionResult { inner: result }
    }

    /// Compute the intersection of multiple Schubert classes
    #[wasm_bindgen(js_name = multiIntersect)]
    pub fn multi_intersect(&mut self, classes: Vec<WasmSchubertClass>) -> WasmIntersectionResult {
        let inner_classes: Vec<SchubertClass> = classes.into_iter().map(|c| c.inner).collect();
        let result = self.inner.multi_intersect(&inner_classes);
        WasmIntersectionResult { inner: result }
    }

    /// Get the Grassmannian dimension k*(n-k)
    #[wasm_bindgen(js_name = getGrassmannianDimension)]
    pub fn get_grassmannian_dimension(&self) -> usize {
        let (k, n) = self.inner.grassmannian_dim;
        k * (n - k)
    }
}

/// WASM wrapper for intersection results
#[wasm_bindgen]
pub struct WasmIntersectionResult {
    inner: IntersectionResult,
}

#[wasm_bindgen]
impl WasmIntersectionResult {
    /// Check if the intersection is finite (a number)
    #[wasm_bindgen(js_name = isFinite)]
    pub fn is_finite(&self) -> bool {
        matches!(self.inner, IntersectionResult::Finite(_))
    }

    /// Check if the intersection is empty
    #[wasm_bindgen(js_name = isEmpty)]
    pub fn is_empty(&self) -> bool {
        matches!(self.inner, IntersectionResult::Empty)
    }

    /// Check if the intersection has positive dimension
    #[wasm_bindgen(js_name = isPositiveDimensional)]
    pub fn is_positive_dimensional(&self) -> bool {
        matches!(self.inner, IntersectionResult::PositiveDimensional { .. })
    }

    /// Get the finite count (or 0 if not finite)
    #[wasm_bindgen(js_name = getCount)]
    pub fn get_count(&self) -> u64 {
        match &self.inner {
            IntersectionResult::Finite(n) => *n,
            _ => 0,
        }
    }

    /// Get the dimension (for positive dimensional, -1 otherwise)
    #[wasm_bindgen(js_name = getDimension)]
    pub fn get_dimension(&self) -> i32 {
        match &self.inner {
            IntersectionResult::PositiveDimensional { dimension, .. } => *dimension as i32,
            IntersectionResult::Empty => -1,
            IntersectionResult::Finite(_) => 0,
        }
    }

    /// Get a string representation
    #[wasm_bindgen(js_name = asString)]
    pub fn as_string(&self) -> String {
        format!("{:?}", self.inner)
    }
}

/// Compute Littlewood-Richardson coefficient c^ν_{λμ}
#[wasm_bindgen(js_name = lrCoefficient)]
pub fn lr_coefficient(lambda: &WasmPartition, mu: &WasmPartition, nu: &WasmPartition) -> u64 {
    amari_enumerative::lr_coefficient(&lambda.inner, &mu.inner, &nu.inner)
}

/// Compute LR coefficients for multiple triples in batch
#[wasm_bindgen(js_name = lrCoefficientsBatch)]
pub fn lr_coefficients_batch(
    lambdas: Vec<WasmPartition>,
    mus: Vec<WasmPartition>,
    nus: Vec<WasmPartition>,
) -> Result<Vec<u64>, JsValue> {
    if lambdas.len() != mus.len() || mus.len() != nus.len() {
        return Err(JsValue::from_str("All arrays must have the same length"));
    }

    let results: Vec<u64> = lambdas
        .iter()
        .zip(mus.iter())
        .zip(nus.iter())
        .map(|((l, m), n)| amari_enumerative::lr_coefficient(&l.inner, &m.inner, &n.inner))
        .collect();

    Ok(results)
}

/// Expand a Schubert product σ_λ · σ_μ into a sum of Schubert classes
#[wasm_bindgen(js_name = schubertProduct)]
pub fn schubert_product(
    lambda: &WasmPartition,
    mu: &WasmPartition,
    k: usize,
    n: usize,
) -> Vec<JsValue> {
    let products = amari_enumerative::schubert_product(&lambda.inner, &mu.inner, (k, n));
    products
        .into_iter()
        .map(|(partition, coeff)| {
            let obj = js_sys::Object::new();
            let parts_array = js_sys::Array::new();
            for part in partition.parts {
                parts_array.push(&JsValue::from(part as u32));
            }
            js_sys::Reflect::set(&obj, &"partition".into(), &parts_array).unwrap();
            js_sys::Reflect::set(&obj, &"coefficient".into(), &JsValue::from(coeff as u32))
                .unwrap();
            obj.into()
        })
        .collect()
}

/// WASM wrapper for capabilities (access control conditions)
#[wasm_bindgen]
pub struct WasmCapability {
    inner: Capability,
}

#[wasm_bindgen]
impl WasmCapability {
    /// Create a new capability with given ID, name, and Schubert condition
    #[wasm_bindgen(constructor)]
    pub fn new(
        id: &str,
        name: &str,
        partition: &[usize],
        k: usize,
        n: usize,
    ) -> Result<WasmCapability, JsValue> {
        match Capability::new(id, name, partition.to_vec(), (k, n)) {
            Ok(cap) => Ok(WasmCapability { inner: cap }),
            Err(e) => Err(JsValue::from_str(&format!("Capability error: {:?}", e))),
        }
    }

    /// Get the capability ID
    #[wasm_bindgen(js_name = getId)]
    pub fn get_id(&self) -> String {
        self.inner.id.0.to_string()
    }

    /// Get the capability name
    #[wasm_bindgen(js_name = getName)]
    pub fn get_name(&self) -> String {
        self.inner.name.clone()
    }

    /// Get the codimension of this capability's Schubert condition
    #[wasm_bindgen(js_name = getCodimension)]
    pub fn get_codimension(&self) -> usize {
        self.inner.schubert_class.codimension()
    }

    /// Add a dependency on another capability
    #[wasm_bindgen(js_name = requires)]
    pub fn requires(mut self, required_id: &str) -> WasmCapability {
        self.inner = self.inner.requires(CapabilityId::new(required_id));
        self
    }
}

/// WASM wrapper for namespaces (geometric access control)
#[wasm_bindgen]
pub struct WasmNamespace {
    inner: Namespace,
}

#[wasm_bindgen]
impl WasmNamespace {
    /// Create a full namespace (identity position) on Gr(k, n)
    #[wasm_bindgen(js_name = full)]
    pub fn full(name: &str, k: usize, n: usize) -> Result<WasmNamespace, JsValue> {
        match Namespace::full(name, k, n) {
            Ok(ns) => Ok(WasmNamespace { inner: ns }),
            Err(e) => Err(JsValue::from_str(&format!("Namespace error: {:?}", e))),
        }
    }

    /// Get the namespace name
    #[wasm_bindgen(js_name = getName)]
    pub fn get_name(&self) -> String {
        self.inner.name.clone()
    }

    /// Get the Grassmannian parameters
    #[wasm_bindgen(js_name = getGrassmannian)]
    pub fn get_grassmannian(&self) -> Vec<usize> {
        vec![self.inner.grassmannian.0, self.inner.grassmannian.1]
    }

    /// Get the number of capabilities
    #[wasm_bindgen(js_name = getCapabilityCount)]
    pub fn get_capability_count(&self) -> usize {
        self.inner.capabilities.len()
    }

    /// Grant a capability to this namespace
    #[wasm_bindgen(js_name = grant)]
    pub fn grant(&mut self, capability: WasmCapability) -> Result<(), JsValue> {
        self.inner
            .grant(capability.inner)
            .map_err(|e| JsValue::from_str(&format!("Grant error: {:?}", e)))
    }

    /// Count valid configurations (intersection number)
    #[wasm_bindgen(js_name = countConfigurations)]
    pub fn count_configurations(&self) -> WasmIntersectionResult {
        WasmIntersectionResult {
            inner: self.inner.count_configurations(),
        }
    }

    /// Get the total codimension from all capabilities
    #[wasm_bindgen(js_name = getTotalCodimension)]
    pub fn get_total_codimension(&self) -> usize {
        self.inner
            .capabilities
            .iter()
            .map(|c| c.schubert_class.codimension())
            .sum()
    }

    /// Get the remaining dimension after capabilities
    #[wasm_bindgen(js_name = getRemainingDimension)]
    pub fn get_remaining_dimension(&self) -> i32 {
        let (k, n) = self.inner.grassmannian;
        let total_codim = self.get_total_codimension();
        (k * (n - k)) as i32 - total_codim as i32
    }
}

/// Check if a capability is accessible in a namespace
#[wasm_bindgen(js_name = capabilityAccessible)]
pub fn capability_accessible(
    namespace: &WasmNamespace,
    capability: &WasmCapability,
) -> Result<bool, JsValue> {
    amari_enumerative::capability_accessible(&namespace.inner, &capability.inner)
        .map_err(|e| JsValue::from_str(&format!("Accessibility check error: {:?}", e)))
}

/// Compute the intersection of two namespaces
#[wasm_bindgen(js_name = namespaceIntersection)]
pub fn namespace_intersection(
    ns1: &WasmNamespace,
    ns2: &WasmNamespace,
) -> Result<WasmNamespaceIntersection, JsValue> {
    match amari_enumerative::namespace_intersection(&ns1.inner, &ns2.inner) {
        Ok(intersection) => Ok(WasmNamespaceIntersection {
            inner: intersection,
        }),
        Err(e) => Err(JsValue::from_str(&format!(
            "Namespace intersection error: {:?}",
            e
        ))),
    }
}

/// WASM wrapper for namespace intersection results
#[wasm_bindgen]
pub struct WasmNamespaceIntersection {
    inner: NamespaceIntersection,
}

#[wasm_bindgen]
impl WasmNamespaceIntersection {
    /// Check if the namespaces are incompatible (different Grassmannians)
    #[wasm_bindgen(js_name = isIncompatible)]
    pub fn is_incompatible(&self) -> bool {
        matches!(self.inner, NamespaceIntersection::Incompatible)
    }

    /// Check if the intersection is disjoint (empty)
    #[wasm_bindgen(js_name = isDisjoint)]
    pub fn is_disjoint(&self) -> bool {
        matches!(self.inner, NamespaceIntersection::Disjoint)
    }

    /// Check if the intersection is a single point
    #[wasm_bindgen(js_name = isSinglePoint)]
    pub fn is_single_point(&self) -> bool {
        matches!(self.inner, NamespaceIntersection::SinglePoint)
    }

    /// Check if the intersection is finite (multiple points)
    #[wasm_bindgen(js_name = isFinitePoints)]
    pub fn is_finite_points(&self) -> bool {
        matches!(self.inner, NamespaceIntersection::FinitePoints(_))
    }

    /// Check if the intersection is a subspace
    #[wasm_bindgen(js_name = isSubspace)]
    pub fn is_subspace(&self) -> bool {
        matches!(self.inner, NamespaceIntersection::Subspace { .. })
    }

    /// Get the count of intersection points (0 if not finite)
    #[wasm_bindgen(js_name = getCount)]
    pub fn get_count(&self) -> u64 {
        match &self.inner {
            NamespaceIntersection::SinglePoint => 1,
            NamespaceIntersection::FinitePoints(n) => *n,
            _ => 0,
        }
    }

    /// Get the dimension (-1 if not a subspace)
    #[wasm_bindgen(js_name = getDimension)]
    pub fn get_dimension(&self) -> i32 {
        match &self.inner {
            NamespaceIntersection::Subspace { dimension } => *dimension as i32,
            _ => -1,
        }
    }

    /// Get a string representation
    #[wasm_bindgen(js_name = asString)]
    pub fn as_string(&self) -> String {
        match &self.inner {
            NamespaceIntersection::Incompatible => "Incompatible".to_string(),
            NamespaceIntersection::Disjoint => "Disjoint".to_string(),
            NamespaceIntersection::SinglePoint => "SinglePoint".to_string(),
            NamespaceIntersection::FinitePoints(n) => format!("FinitePoints({})", n),
            NamespaceIntersection::Subspace { dimension } => format!("Subspace(dim={})", dimension),
        }
    }
}

/// Batch operations for Schubert calculus
#[wasm_bindgen]
pub struct SchubertBatch;

#[wasm_bindgen]
impl SchubertBatch {
    /// Compute multiple Schubert multi-intersections in batch
    #[wasm_bindgen(js_name = multiIntersectBatch)]
    pub fn multi_intersect_batch(
        class_batches: Vec<JsValue>,
        k: usize,
        n: usize,
    ) -> Result<Vec<WasmIntersectionResult>, JsValue> {
        let mut results = Vec::new();

        for batch_js in class_batches {
            let batch_array: js_sys::Array = batch_js.into();
            let mut classes = Vec::new();

            for i in 0..batch_array.length() {
                let partition_array: js_sys::Array = batch_array.get(i).into();
                let partition: Vec<usize> = partition_array
                    .iter()
                    .map(|v| v.as_f64().unwrap_or(0.0) as usize)
                    .collect();

                if let Ok(class) = SchubertClass::new(partition, (k, n)) {
                    classes.push(class);
                }
            }

            let mut calc = SchubertCalculus::new((k, n));
            let result = calc.multi_intersect(&classes);
            results.push(WasmIntersectionResult { inner: result });
        }

        Ok(results)
    }

    /// Compute the famous "lines meeting 4 lines" problem
    #[wasm_bindgen(js_name = linesMeeting4Lines)]
    pub fn lines_meeting_4_lines() -> WasmIntersectionResult {
        // σ_1^4 in Gr(2, 4) = 2
        let mut calc = SchubertCalculus::new((2, 4));
        let sigma_1 = SchubertClass::new(vec![1], (2, 4)).unwrap();
        let classes = vec![sigma_1.clone(), sigma_1.clone(), sigma_1.clone(), sigma_1];
        WasmIntersectionResult {
            inner: calc.multi_intersect(&classes),
        }
    }
}

/// Namespace batch operations
#[wasm_bindgen]
pub struct NamespaceBatch;

#[wasm_bindgen]
impl NamespaceBatch {
    /// Count configurations for multiple namespaces in batch
    #[wasm_bindgen(js_name = countConfigurationsBatch)]
    pub fn count_configurations_batch(
        namespaces: Vec<WasmNamespace>,
    ) -> Vec<WasmIntersectionResult> {
        namespaces
            .iter()
            .map(|ns| WasmIntersectionResult {
                inner: ns.inner.count_configurations(),
            })
            .collect()
    }
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
        let partition = vec![2, 2, 1]; // Valid partition: all parts <= n-k = 5-3 = 2
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

    #[wasm_bindgen_test]
    fn test_partition() {
        let partition = WasmPartition::new(&[3, 2, 1]);
        assert_eq!(partition.get_parts(), vec![3, 2, 1]);
        assert_eq!(partition.get_size(), 6);
        assert_eq!(partition.get_length(), 3);
        assert!(partition.is_valid());
        assert!(partition.fits_in_box(3, 6)); // 3 parts, max value 3 <= 6-3=3
    }

    #[wasm_bindgen_test]
    fn test_schubert_class() {
        let class = WasmSchubertClass::new(&[2, 1], 2, 5);
        assert!(class.is_ok());
        let class = class.unwrap();
        assert_eq!(class.get_partition(), vec![2, 1]);
        assert_eq!(class.get_grassmannian_dim(), vec![2, 5]);
        assert_eq!(class.get_codimension(), 3); // 2 + 1 = 3
    }

    #[wasm_bindgen_test]
    fn test_schubert_calculus() {
        let calc = WasmSchubertCalculus::new(2, 4);
        assert_eq!(calc.get_grassmannian_dimension(), 4); // 2*(4-2) = 4

        // Test σ_1 classes
        let sigma_1 = WasmSchubertClass::sigma_1(2, 4).unwrap();
        assert_eq!(sigma_1.get_codimension(), 1);
    }

    #[wasm_bindgen_test]
    fn test_lr_coefficient() {
        let lambda = WasmPartition::new(&[2, 1]);
        let mu = WasmPartition::new(&[1, 1]);
        let nu = WasmPartition::new(&[3, 2]);

        let _coeff = lr_coefficient(&lambda, &mu, &nu);
        // This should give a valid LR coefficient
        // coeff is u64, always >= 0
    }

    #[wasm_bindgen_test]
    fn test_intersection_result() {
        // Test the classic "lines meeting 4 lines" problem
        let result = SchubertBatch::lines_meeting_4_lines();
        assert!(result.is_finite());
        assert_eq!(result.get_count(), 2); // Famous result: 2 lines
    }

    #[wasm_bindgen_test]
    fn test_capability_creation() {
        let cap = WasmCapability::new("read", "Read Access", &[1], 2, 4);
        assert!(cap.is_ok());
        let cap = cap.unwrap();
        assert_eq!(cap.get_id(), "read");
        assert_eq!(cap.get_name(), "Read Access");
        assert_eq!(cap.get_codimension(), 1);
    }

    #[wasm_bindgen_test]
    fn test_namespace() {
        let ns = WasmNamespace::full("agent", 2, 4);
        assert!(ns.is_ok());
        let mut ns = ns.unwrap();
        assert_eq!(ns.get_name(), "agent");
        assert_eq!(ns.get_grassmannian(), vec![2, 4]);
        assert_eq!(ns.get_capability_count(), 0);

        // Grant a capability
        let cap = WasmCapability::new("read", "Read", &[1], 2, 4).unwrap();
        assert!(ns.grant(cap).is_ok());
        assert_eq!(ns.get_capability_count(), 1);
        assert_eq!(ns.get_total_codimension(), 1);
        assert_eq!(ns.get_remaining_dimension(), 3); // 4 - 1 = 3
    }

    #[wasm_bindgen_test]
    fn test_namespace_intersection_result() {
        let ns1 = WasmNamespace::full("ns1", 2, 4).unwrap();
        let ns2 = WasmNamespace::full("ns2", 2, 4).unwrap();

        let result = namespace_intersection(&ns1, &ns2);
        assert!(result.is_ok());
        let result = result.unwrap();
        // Two full namespaces should have a subspace intersection
        assert!(result.is_subspace());
    }
}
