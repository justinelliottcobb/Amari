//! WASM bindings for computational topology (amari-topology).
//!
//! Exposes simplicial complexes, homology computation, persistent homology,
//! and Morse theory for web applications.
//!
//! # Example
//!
//! ```javascript
//! import {
//!   WasmSimplex,
//!   WasmSimplicialComplex,
//!   WasmFiltration,
//!   WasmPersistentHomology
//! } from '@justinelliottcobb/amari-wasm';
//!
//! // Create a triangle
//! const complex = new WasmSimplicialComplex();
//! complex.addSimplex([0, 1, 2]);
//!
//! // Compute Betti numbers
//! const betti = complex.bettiNumbers();
//! console.log(`β₀ = ${betti[0]}, β₁ = ${betti[1]}`);
//! ```

use wasm_bindgen::prelude::*;

use amari_topology::{
    find_critical_points_grid, CriticalType, Filtration, PersistentHomology, Simplex,
    SimplicialComplex,
};

// ============================================================================
// WasmSimplex
// ============================================================================

/// A simplex (vertex, edge, triangle, tetrahedron, etc.) for WASM.
#[wasm_bindgen]
pub struct WasmSimplex {
    inner: Simplex,
}

#[wasm_bindgen]
impl WasmSimplex {
    /// Create a new simplex from vertex indices.
    #[wasm_bindgen(constructor)]
    pub fn new(vertices: Vec<usize>) -> Self {
        Self {
            inner: Simplex::new(vertices),
        }
    }

    /// Get the dimension of this simplex.
    ///
    /// - 0-simplex: point (1 vertex)
    /// - 1-simplex: edge (2 vertices)
    /// - 2-simplex: triangle (3 vertices)
    /// - 3-simplex: tetrahedron (4 vertices)
    #[wasm_bindgen]
    pub fn dimension(&self) -> usize {
        self.inner.dimension()
    }

    /// Get the vertices of this simplex (in sorted order).
    #[wasm_bindgen(js_name = getVertices)]
    pub fn get_vertices(&self) -> Vec<usize> {
        self.inner.vertices().to_vec()
    }

    /// Get the orientation (+1 or -1).
    #[wasm_bindgen]
    pub fn orientation(&self) -> i8 {
        self.inner.orientation()
    }

    /// Check if this simplex contains a vertex.
    #[wasm_bindgen(js_name = containsVertex)]
    pub fn contains_vertex(&self, vertex: usize) -> bool {
        self.inner.contains_vertex(vertex)
    }

    /// Get all k-faces of this simplex.
    #[wasm_bindgen]
    pub fn faces(&self, k: usize) -> Vec<WasmSimplex> {
        self.inner
            .faces(k)
            .into_iter()
            .map(|s| WasmSimplex { inner: s })
            .collect()
    }

    /// Get the boundary faces with orientations.
    /// Returns pairs of (simplex, sign).
    #[wasm_bindgen(js_name = boundaryFaces)]
    pub fn boundary_faces(&self) -> Vec<WasmBoundaryFace> {
        self.inner
            .boundary_faces()
            .into_iter()
            .map(|(s, sign)| WasmBoundaryFace {
                simplex: WasmSimplex { inner: s },
                sign,
            })
            .collect()
    }

    /// Display string representation.
    #[wasm_bindgen(js_name = toString)]
    pub fn to_string_js(&self) -> String {
        format!("{}", self.inner)
    }
}

/// A boundary face with its orientation sign.
#[wasm_bindgen]
pub struct WasmBoundaryFace {
    simplex: WasmSimplex,
    sign: i8,
}

#[wasm_bindgen]
impl WasmBoundaryFace {
    /// Get the face simplex.
    #[wasm_bindgen(getter)]
    pub fn simplex(&self) -> WasmSimplex {
        WasmSimplex {
            inner: self.simplex.inner.clone(),
        }
    }

    /// Get the orientation sign (+1 or -1).
    #[wasm_bindgen(getter)]
    pub fn sign(&self) -> i8 {
        self.sign
    }
}

// ============================================================================
// WasmSimplicialComplex
// ============================================================================

/// A simplicial complex for WASM.
#[wasm_bindgen]
pub struct WasmSimplicialComplex {
    inner: SimplicialComplex,
}

impl Default for WasmSimplicialComplex {
    fn default() -> Self {
        Self::new()
    }
}

#[wasm_bindgen]
impl WasmSimplicialComplex {
    /// Create an empty simplicial complex.
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            inner: SimplicialComplex::new(),
        }
    }

    /// Add a simplex (and all its faces) to the complex.
    #[wasm_bindgen(js_name = addSimplex)]
    pub fn add_simplex(&mut self, vertices: Vec<usize>) {
        self.inner.add_simplex(Simplex::new(vertices));
    }

    /// Add a simplex object.
    #[wasm_bindgen(js_name = addSimplexObject)]
    pub fn add_simplex_object(&mut self, simplex: &WasmSimplex) {
        self.inner.add_simplex(simplex.inner.clone());
    }

    /// Check if the complex contains a simplex with given vertices.
    #[wasm_bindgen]
    pub fn contains(&self, vertices: Vec<usize>) -> bool {
        self.inner.contains(&Simplex::new(vertices))
    }

    /// Get the maximum dimension of any simplex.
    #[wasm_bindgen]
    pub fn dimension(&self) -> usize {
        self.inner.dimension()
    }

    /// Count simplices of a given dimension.
    #[wasm_bindgen(js_name = simplexCount)]
    pub fn simplex_count(&self, dim: usize) -> usize {
        self.inner.simplex_count(dim)
    }

    /// Total number of simplices.
    #[wasm_bindgen(js_name = totalSimplexCount)]
    pub fn total_simplex_count(&self) -> usize {
        self.inner.total_simplex_count()
    }

    /// Number of vertices (0-simplices).
    #[wasm_bindgen(js_name = vertexCount)]
    pub fn vertex_count(&self) -> usize {
        self.inner.vertex_count()
    }

    /// Number of edges (1-simplices).
    #[wasm_bindgen(js_name = edgeCount)]
    pub fn edge_count(&self) -> usize {
        self.inner.edge_count()
    }

    /// Compute the Betti numbers β₀, β₁, β₂, ...
    ///
    /// - β₀ = number of connected components
    /// - β₁ = number of 1D holes (loops)
    /// - β₂ = number of 2D voids (cavities)
    #[wasm_bindgen(js_name = bettiNumbers)]
    pub fn betti_numbers(&self) -> Vec<usize> {
        self.inner.betti_numbers()
    }

    /// Compute the Euler characteristic χ = V - E + F - ...
    #[wasm_bindgen(js_name = eulerCharacteristic)]
    pub fn euler_characteristic(&self) -> i32 {
        self.inner.euler_characteristic() as i32
    }

    /// Get the f-vector (face counts by dimension).
    #[wasm_bindgen(js_name = fVector)]
    pub fn f_vector(&self) -> Vec<usize> {
        self.inner.f_vector()
    }

    /// Check if the complex is connected.
    #[wasm_bindgen(js_name = isConnected)]
    pub fn is_connected(&self) -> bool {
        self.inner.is_connected()
    }

    /// Count the number of connected components.
    #[wasm_bindgen(js_name = connectedComponents)]
    pub fn connected_components(&self) -> usize {
        self.inner.connected_components()
    }
}

// ============================================================================
// WasmFiltration
// ============================================================================

/// A filtration of simplicial complexes for persistent homology.
#[wasm_bindgen]
pub struct WasmFiltration {
    inner: Filtration,
}

impl Default for WasmFiltration {
    fn default() -> Self {
        Self::new()
    }
}

#[wasm_bindgen]
impl WasmFiltration {
    /// Create an empty filtration.
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            inner: Filtration::new(),
        }
    }

    /// Add a simplex at a given filtration time.
    #[wasm_bindgen]
    pub fn add(&mut self, time: f64, vertices: Vec<usize>) {
        self.inner.add(time, Simplex::new(vertices));
    }

    /// Check if the filtration is empty.
    #[wasm_bindgen(js_name = isEmpty)]
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Get the complex at a specific filtration time.
    #[wasm_bindgen(js_name = complexAt)]
    pub fn complex_at(&mut self, time: f64) -> WasmSimplicialComplex {
        WasmSimplicialComplex {
            inner: self.inner.complex_at(time),
        }
    }

    /// Get the Betti numbers at a specific time.
    #[wasm_bindgen(js_name = bettiAt)]
    pub fn betti_at(&mut self, time: f64) -> Vec<usize> {
        self.inner.complex_at(time).betti_numbers()
    }
}

/// Create a Vietoris-Rips filtration from pairwise distances.
///
/// - `num_points`: Number of points
/// - `max_dim`: Maximum simplex dimension to include
/// - `distances`: Flat array of pairwise distances (upper triangular, row-major)
#[wasm_bindgen(js_name = ripsFromDistances)]
pub fn rips_from_distances(
    num_points: usize,
    max_dim: usize,
    distances: Vec<f64>,
) -> WasmFiltration {
    // Build distance lookup from flat array
    let distance = |i: usize, j: usize| -> f64 {
        if i == j {
            return 0.0;
        }
        let (min_idx, max_idx) = if i < j { (i, j) } else { (j, i) };
        // Index in upper triangular matrix (row-major)
        let idx = min_idx * (2 * num_points - min_idx - 1) / 2 + (max_idx - min_idx - 1);
        distances.get(idx).copied().unwrap_or(f64::INFINITY)
    };

    WasmFiltration {
        inner: amari_topology::rips_filtration(num_points, max_dim, distance),
    }
}

// ============================================================================
// WasmPersistentHomology
// ============================================================================

/// Result of persistent homology computation.
#[wasm_bindgen]
pub struct WasmPersistentHomology {
    inner: PersistentHomology,
}

#[wasm_bindgen]
impl WasmPersistentHomology {
    /// Compute persistent homology from a filtration.
    #[wasm_bindgen]
    pub fn compute(filtration: &mut WasmFiltration) -> Result<WasmPersistentHomology, JsValue> {
        PersistentHomology::compute(&mut filtration.inner)
            .map(|ph| WasmPersistentHomology { inner: ph })
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Get the persistence diagram as an array of [dimension, birth, death] triples.
    #[wasm_bindgen(js_name = getDiagram)]
    pub fn get_diagram(&self) -> Vec<f64> {
        let diagram = self.inner.diagram();
        let mut result = Vec::new();

        for interval in diagram.intervals() {
            result.push(interval.dimension as f64);
            result.push(interval.birth);
            result.push(interval.death.unwrap_or(f64::INFINITY));
        }

        result
    }

    /// Get the Betti numbers at a specific filtration time.
    #[wasm_bindgen(js_name = bettiAt)]
    pub fn betti_at(&self, time: f64) -> Vec<usize> {
        self.inner.diagram().betti_at(time)
    }

    /// Get the number of intervals in dimension k.
    #[wasm_bindgen(js_name = intervalCount)]
    pub fn interval_count(&self, dim: usize) -> usize {
        self.inner
            .diagram()
            .intervals()
            .iter()
            .filter(|i| i.dimension == dim)
            .count()
    }
}

// ============================================================================
// Morse Theory
// ============================================================================

/// A critical point from Morse theory analysis.
#[wasm_bindgen]
pub struct WasmCriticalPoint {
    position: Vec<f64>,
    value: f64,
    critical_type: String,
    index: usize,
}

#[wasm_bindgen]
impl WasmCriticalPoint {
    /// Get the position coordinates.
    #[wasm_bindgen(getter)]
    pub fn position(&self) -> Vec<f64> {
        self.position.clone()
    }

    /// Get the function value at this critical point.
    #[wasm_bindgen(getter)]
    pub fn value(&self) -> f64 {
        self.value
    }

    /// Get the type of critical point: "minimum", "saddle", or "maximum".
    #[wasm_bindgen(getter, js_name = criticalType)]
    pub fn critical_type(&self) -> String {
        self.critical_type.clone()
    }

    /// Get the Morse index (number of negative eigenvalues of Hessian).
    #[wasm_bindgen(getter)]
    pub fn index(&self) -> usize {
        self.index
    }
}

/// Find critical points of a 2D function on a grid.
///
/// - `resolution`: Grid resolution in each dimension
/// - `x_min`, `x_max`, `y_min`, `y_max`: Bounding box
/// - `tolerance`: Gradient norm threshold for critical points
/// - `evaluator`: Function that takes (x, y) and returns f(x, y)
#[wasm_bindgen(js_name = findCriticalPoints2D)]
pub fn find_critical_points_2d(
    resolution: usize,
    x_min: f64,
    x_max: f64,
    y_min: f64,
    y_max: f64,
    tolerance: f64,
    values: Vec<f64>,
) -> Result<Vec<WasmCriticalPoint>, JsValue> {
    // Validate input size
    let expected_size = (resolution + 1) * (resolution + 1);
    if values.len() != expected_size {
        return Err(JsValue::from_str(&format!(
            "Expected {} values for resolution {}, got {}",
            expected_size,
            resolution,
            values.len()
        )));
    }

    // Create function from precomputed values
    let dx = (x_max - x_min) / resolution as f64;
    let dy = (y_max - y_min) / resolution as f64;

    let f = |x: f64, y: f64| -> f64 {
        // Find nearest grid indices
        let i = ((x - x_min) / dx).round() as usize;
        let j = ((y - y_min) / dy).round() as usize;
        let i = i.min(resolution);
        let j = j.min(resolution);
        values[i * (resolution + 1) + j]
    };

    let bounds = [(x_min, x_max), (y_min, y_max)];

    match find_critical_points_grid(f, &bounds, resolution, tolerance) {
        Ok(cps) => Ok(cps
            .into_iter()
            .map(|cp| WasmCriticalPoint {
                position: cp.position.to_vec(),
                value: cp.value,
                critical_type: match cp.critical_type {
                    CriticalType::Minimum => "minimum".to_string(),
                    CriticalType::Maximum => "maximum".to_string(),
                    CriticalType::Saddle(_) => "saddle".to_string(),
                    CriticalType::Degenerate => "degenerate".to_string(),
                },
                index: cp.index,
            })
            .collect()),
        Err(e) => Err(JsValue::from_str(&e.to_string())),
    }
}

/// Morse complex result for counting critical points by index.
#[wasm_bindgen]
pub struct WasmMorseComplex {
    counts: Vec<usize>,
}

#[wasm_bindgen]
impl WasmMorseComplex {
    /// Create from critical points.
    #[wasm_bindgen(constructor)]
    pub fn new(critical_points: Vec<WasmCriticalPoint>) -> Self {
        let mut counts = Vec::new();

        for cp in critical_points {
            while counts.len() <= cp.index {
                counts.push(0);
            }
            counts[cp.index] += 1;
        }

        Self { counts }
    }

    /// Get counts by Morse index.
    #[wasm_bindgen(js_name = countsByIndex)]
    pub fn counts_by_index(&self) -> Vec<usize> {
        self.counts.clone()
    }

    /// Check if weak Morse inequalities hold: c_k >= beta_k.
    #[wasm_bindgen(js_name = checkWeakMorseInequalities)]
    pub fn check_weak_morse_inequalities(&self, betti: Vec<usize>) -> bool {
        self.counts.iter().zip(betti.iter()).all(|(&c, &b)| c >= b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use wasm_bindgen_test::*;

    #[test]
    #[wasm_bindgen_test]
    fn test_simplex() {
        let s = WasmSimplex::new(vec![0, 1, 2]);
        assert_eq!(s.dimension(), 2);
        assert_eq!(s.get_vertices(), vec![0, 1, 2]);
    }

    #[test]
    #[wasm_bindgen_test]
    fn test_complex_betti() {
        let mut complex = WasmSimplicialComplex::new();
        complex.add_simplex(vec![0, 1, 2]);

        let betti = complex.betti_numbers();
        assert_eq!(betti[0], 1); // 1 component
        assert_eq!(betti[1], 0); // No holes (filled triangle)
    }

    #[test]
    #[wasm_bindgen_test]
    fn test_circle_betti() {
        let mut complex = WasmSimplicialComplex::new();
        complex.add_simplex(vec![0, 1]);
        complex.add_simplex(vec![1, 2]);
        complex.add_simplex(vec![2, 0]);

        let betti = complex.betti_numbers();
        assert_eq!(betti[0], 1); // 1 component
        assert_eq!(betti[1], 1); // 1 hole
    }
}
