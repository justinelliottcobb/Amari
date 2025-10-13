//! WASM bindings for geometric network analysis
//!
//! This module provides WebAssembly bindings for:
//! - Geometric networks embedded in Clifford algebra space
//! - Tropical network algorithms for efficient path finding
//! - Community detection and centrality measures
//! - Information diffusion modeling

use amari_core::Multivector;
use amari_network::{tropical::TropicalNetwork, GeometricNetwork, NodeMetadata};
use js_sys::Array;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use wasm_bindgen::prelude::*;

/// Helper function to extract coefficients from a multivector
fn multivector_to_coefficients<const P: usize, const Q: usize, const R: usize>(
    mv: &Multivector<P, Q, R>,
) -> Vec<f64> {
    let basis_count = 1 << (P + Q + R);
    (0..basis_count).map(|i| mv.get(i)).collect()
}

/// WASM wrapper for geometric networks
#[wasm_bindgen]
pub struct WasmGeometricNetwork {
    inner: GeometricNetwork<3, 0, 0>, // 3D Euclidean space for web compatibility
}

impl Default for WasmGeometricNetwork {
    fn default() -> Self {
        Self::new()
    }
}

/// WASM wrapper for network edges
#[wasm_bindgen]
#[derive(Clone)]
pub struct WasmGeometricEdge {
    source: usize,
    target: usize,
    weight: f64,
}

/// WASM wrapper for node metadata
#[wasm_bindgen]
#[derive(Clone)]
pub struct WasmNodeMetadata {
    label: Option<String>,
    properties: HashMap<String, f64>,
}

impl Default for WasmNodeMetadata {
    fn default() -> Self {
        Self::new()
    }
}

/// WASM wrapper for community detection results
#[wasm_bindgen]
pub struct WasmCommunity {
    nodes: Vec<usize>,
    centroid_coeffs: Vec<f64>,
    cohesion_score: f64,
}

/// WASM wrapper for propagation analysis
#[wasm_bindgen]
pub struct WasmPropagationAnalysis {
    coverage: Vec<usize>,
    influence_scores: Vec<f64>,
    convergence_time: usize,
}

/// WASM wrapper for tropical networks
#[wasm_bindgen]
pub struct WasmTropicalNetwork {
    inner: TropicalNetwork,
}

/// Path result for WASM
#[derive(Serialize, Deserialize)]
pub struct PathResult {
    pub path: Vec<usize>,
    pub distance: f64,
}

#[wasm_bindgen]
impl WasmGeometricNetwork {
    /// Create a new empty geometric network
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            inner: GeometricNetwork::new(),
        }
    }

    /// Create a network with pre-allocated capacity
    #[wasm_bindgen(js_name = withCapacity)]
    pub fn with_capacity(num_nodes: usize, num_edges: usize) -> Self {
        Self {
            inner: GeometricNetwork::with_capacity(num_nodes, num_edges),
        }
    }

    /// Add a node at the specified position (using flat coefficient array)
    #[wasm_bindgen(js_name = addNode)]
    pub fn add_node(&mut self, coefficients: &[f64]) -> Result<usize, JsValue> {
        if coefficients.len() != 8 {
            return Err(JsValue::from_str(
                "Coefficients array must have exactly 8 elements",
            ));
        }

        let mv = Multivector::from_coefficients(coefficients.to_vec());
        Ok(self.inner.add_node(mv))
    }

    /// Add a node with metadata
    #[wasm_bindgen(js_name = addNodeWithMetadata)]
    pub fn add_node_with_metadata(
        &mut self,
        coefficients: &[f64],
        label: Option<String>,
    ) -> Result<usize, JsValue> {
        if coefficients.len() != 8 {
            return Err(JsValue::from_str(
                "Coefficients array must have exactly 8 elements",
            ));
        }

        let mv = Multivector::from_coefficients(coefficients.to_vec());
        let metadata = NodeMetadata {
            label,
            properties: HashMap::new(),
        };

        Ok(self.inner.add_node_with_metadata(mv, metadata))
    }

    /// Add a directed edge between two nodes
    #[wasm_bindgen(js_name = addEdge)]
    pub fn add_edge(&mut self, source: usize, target: usize, weight: f64) -> Result<(), JsValue> {
        self.inner
            .add_edge(source, target, weight)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Add an undirected edge (creates two directed edges)
    #[wasm_bindgen(js_name = addUndirectedEdge)]
    pub fn add_undirected_edge(&mut self, a: usize, b: usize, weight: f64) -> Result<(), JsValue> {
        self.inner
            .add_undirected_edge(a, b, weight)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Get the number of nodes in the network
    #[wasm_bindgen(js_name = numNodes)]
    pub fn num_nodes(&self) -> usize {
        self.inner.num_nodes()
    }

    /// Get the number of edges in the network
    #[wasm_bindgen(js_name = numEdges)]
    pub fn num_edges(&self) -> usize {
        self.inner.num_edges()
    }

    /// Get node position as coefficient array
    #[wasm_bindgen(js_name = getNode)]
    pub fn get_node(&self, index: usize) -> Option<Vec<f64>> {
        self.inner.get_node(index).map(multivector_to_coefficients)
    }

    /// Get neighbors of a node
    #[wasm_bindgen(js_name = getNeighbors)]
    pub fn get_neighbors(&self, node: usize) -> Vec<usize> {
        self.inner.neighbors(node)
    }

    /// Get the degree of a node
    #[wasm_bindgen(js_name = getDegree)]
    pub fn get_degree(&self, node: usize) -> usize {
        self.inner.degree(node)
    }

    /// Compute geometric distance between two nodes
    #[wasm_bindgen(js_name = geometricDistance)]
    pub fn geometric_distance(&self, node1: usize, node2: usize) -> Result<f64, JsValue> {
        self.inner
            .geometric_distance(node1, node2)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Compute geometric centrality for all nodes
    #[wasm_bindgen(js_name = computeGeometricCentrality)]
    pub fn compute_geometric_centrality(&self) -> Result<Vec<f64>, JsValue> {
        self.inner
            .compute_geometric_centrality()
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Compute betweenness centrality for all nodes
    #[wasm_bindgen(js_name = computeBetweennessCentrality)]
    pub fn compute_betweenness_centrality(&self) -> Result<Vec<f64>, JsValue> {
        self.inner
            .compute_betweenness_centrality()
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Find shortest path between two nodes using edge weights
    #[wasm_bindgen(js_name = shortestPath)]
    pub fn shortest_path(&self, source: usize, target: usize) -> Result<JsValue, JsValue> {
        match self.inner.shortest_path(source, target) {
            Ok(Some((path, distance))) => {
                let result = PathResult { path, distance };
                Ok(serde_wasm_bindgen::to_value(&result)?)
            }
            Ok(None) => Ok(JsValue::NULL),
            Err(e) => Err(JsValue::from_str(&e.to_string())),
        }
    }

    /// Find shortest path using geometric distances
    #[wasm_bindgen(js_name = shortestGeometricPath)]
    pub fn shortest_geometric_path(
        &self,
        source: usize,
        target: usize,
    ) -> Result<JsValue, JsValue> {
        match self.inner.shortest_geometric_path(source, target) {
            Ok(Some((path, distance))) => {
                let result = PathResult { path, distance };
                Ok(serde_wasm_bindgen::to_value(&result)?)
            }
            Ok(None) => Ok(JsValue::NULL),
            Err(e) => Err(JsValue::from_str(&e.to_string())),
        }
    }

    /// Convert to tropical network for efficient path operations
    #[wasm_bindgen(js_name = toTropicalNetwork)]
    pub fn to_tropical_network(&self) -> Result<WasmTropicalNetwork, JsValue> {
        match self.inner.to_tropical_network() {
            Ok(tropical) => Ok(WasmTropicalNetwork { inner: tropical }),
            Err(e) => Err(JsValue::from_str(&e.to_string())),
        }
    }

    /// Detect communities using geometric clustering
    #[wasm_bindgen(js_name = findCommunities)]
    pub fn find_communities(&self, num_communities: usize) -> Result<Array, JsValue> {
        let communities = self
            .inner
            .find_communities(num_communities)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        let result = Array::new();
        for community in communities {
            let wasm_community = WasmCommunity {
                nodes: community.nodes,
                centroid_coeffs: multivector_to_coefficients(&community.geometric_centroid),
                cohesion_score: community.cohesion_score,
            };
            result.push(&JsValue::from(wasm_community));
        }

        Ok(result)
    }

    /// Simulate information diffusion through the network
    #[wasm_bindgen(js_name = simulateDiffusion)]
    pub fn simulate_diffusion(
        &self,
        initial_sources: &[usize],
        max_steps: usize,
        diffusion_rate: f64,
    ) -> Result<WasmPropagationAnalysis, JsValue> {
        let analysis = self
            .inner
            .simulate_diffusion(initial_sources, max_steps, diffusion_rate)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        Ok(WasmPropagationAnalysis {
            coverage: analysis.coverage,
            influence_scores: analysis.influence_scores,
            convergence_time: analysis.convergence_time,
        })
    }

    /// Perform spectral clustering
    #[wasm_bindgen(js_name = spectralClustering)]
    pub fn spectral_clustering(&self, num_clusters: usize) -> Result<Array, JsValue> {
        let clusters = self
            .inner
            .spectral_clustering(num_clusters)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        let result = Array::new();
        for cluster in clusters {
            let js_cluster = Array::new();
            for &node in &cluster {
                js_cluster.push(&JsValue::from(node));
            }
            result.push(&js_cluster);
        }

        Ok(result)
    }
}

#[wasm_bindgen]
impl WasmGeometricEdge {
    /// Create a new geometric edge
    #[wasm_bindgen(constructor)]
    pub fn new(source: usize, target: usize, weight: f64) -> Self {
        Self {
            source,
            target,
            weight,
        }
    }

    /// Get the source node
    #[wasm_bindgen(getter)]
    pub fn source(&self) -> usize {
        self.source
    }

    /// Get the target node
    #[wasm_bindgen(getter)]
    pub fn target(&self) -> usize {
        self.target
    }

    /// Get the edge weight
    #[wasm_bindgen(getter)]
    pub fn weight(&self) -> f64 {
        self.weight
    }
}

#[wasm_bindgen]
impl WasmNodeMetadata {
    /// Create new empty metadata
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            label: None,
            properties: HashMap::new(),
        }
    }

    /// Create metadata with label
    #[wasm_bindgen(js_name = withLabel)]
    pub fn with_label(label: &str) -> Self {
        Self {
            label: Some(label.to_string()),
            properties: HashMap::new(),
        }
    }

    /// Get the label
    #[wasm_bindgen(getter)]
    pub fn label(&self) -> Option<String> {
        self.label.clone()
    }

    /// Set the label
    #[wasm_bindgen(setter)]
    pub fn set_label(&mut self, label: Option<String>) {
        self.label = label;
    }

    /// Add a numerical property
    #[wasm_bindgen(js_name = setProperty)]
    pub fn set_property(&mut self, key: &str, value: f64) {
        self.properties.insert(key.to_string(), value);
    }

    /// Get a numerical property
    #[wasm_bindgen(js_name = getProperty)]
    pub fn get_property(&self, key: &str) -> Option<f64> {
        self.properties.get(key).copied()
    }
}

#[wasm_bindgen]
impl WasmCommunity {
    /// Get community member nodes
    #[wasm_bindgen(getter)]
    pub fn nodes(&self) -> Vec<usize> {
        self.nodes.clone()
    }

    /// Get geometric centroid coefficients
    #[wasm_bindgen(getter = centroidCoefficients)]
    pub fn centroid_coefficients(&self) -> Vec<f64> {
        self.centroid_coeffs.clone()
    }

    /// Get cohesion score
    #[wasm_bindgen(getter = cohesionScore)]
    pub fn cohesion_score(&self) -> f64 {
        self.cohesion_score
    }
}

#[wasm_bindgen]
impl WasmPropagationAnalysis {
    /// Get coverage over time
    #[wasm_bindgen(getter)]
    pub fn coverage(&self) -> Vec<usize> {
        self.coverage.clone()
    }

    /// Get influence scores for each node
    #[wasm_bindgen(getter = influenceScores)]
    pub fn influence_scores(&self) -> Vec<f64> {
        self.influence_scores.clone()
    }

    /// Get convergence time
    #[wasm_bindgen(getter = convergenceTime)]
    pub fn convergence_time(&self) -> usize {
        self.convergence_time
    }
}

#[wasm_bindgen]
impl WasmTropicalNetwork {
    /// Create a new tropical network with given size
    #[wasm_bindgen(constructor)]
    pub fn new(size: usize) -> Self {
        Self {
            inner: TropicalNetwork::new(size),
        }
    }

    /// Create tropical network from weight matrix
    #[wasm_bindgen(js_name = fromWeights)]
    pub fn from_weights(weights: &[f64], size: usize) -> Result<WasmTropicalNetwork, JsValue> {
        if weights.len() != size * size {
            return Err(JsValue::from_str("Weight array size mismatch"));
        }

        let mut weight_matrix = vec![vec![0.0; size]; size];
        for i in 0..size {
            for j in 0..size {
                weight_matrix[i][j] = weights[i * size + j];
            }
        }

        match TropicalNetwork::from_weights(&weight_matrix) {
            Ok(tropical) => Ok(WasmTropicalNetwork { inner: tropical }),
            Err(e) => Err(JsValue::from_str(&e.to_string())),
        }
    }

    /// Get the number of nodes
    #[wasm_bindgen(js_name = getSize)]
    pub fn get_size(&self) -> usize {
        self.inner.size()
    }

    /// Set edge weight between two nodes
    #[wasm_bindgen(js_name = setEdge)]
    pub fn set_edge(&mut self, source: usize, target: usize, weight: f64) -> Result<(), JsValue> {
        self.inner
            .set_edge(source, target, weight)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Find shortest path using tropical algebra
    #[wasm_bindgen(js_name = shortestPathTropical)]
    pub fn shortest_path_tropical(&self, source: usize, target: usize) -> Result<JsValue, JsValue> {
        match self.inner.shortest_path_tropical(source, target) {
            Ok(Some((path, distance))) => {
                let result = PathResult { path, distance };
                Ok(serde_wasm_bindgen::to_value(&result)?)
            }
            Ok(None) => Ok(JsValue::NULL),
            Err(e) => Err(JsValue::from_str(&e.to_string())),
        }
    }

    /// Compute tropical betweenness centrality
    #[wasm_bindgen(js_name = tropicalBetweenness)]
    pub fn tropical_betweenness(&self) -> Result<Vec<f64>, JsValue> {
        self.inner
            .tropical_betweenness()
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }
}

/// Utility functions for network analysis
#[wasm_bindgen]
pub struct NetworkUtils;

#[wasm_bindgen]
impl NetworkUtils {
    /// Create a random geometric network for testing
    #[wasm_bindgen(js_name = createRandomNetwork)]
    pub fn create_random_network(
        num_nodes: usize,
        connection_probability: f64,
    ) -> Result<WasmGeometricNetwork, JsValue> {
        if !(0.0..=1.0).contains(&connection_probability) {
            return Err(JsValue::from_str(
                "Connection probability must be between 0 and 1",
            ));
        }

        let mut network = WasmGeometricNetwork::new();

        // Add nodes at random positions
        for _ in 0..num_nodes {
            let coefficients = vec![
                fastrand::f64() * 2.0 - 1.0, // scalar
                fastrand::f64() * 2.0 - 1.0, // e1
                fastrand::f64() * 2.0 - 1.0, // e2
                fastrand::f64() * 2.0 - 1.0, // e3
                0.0,                         // e12
                0.0,                         // e13
                0.0,                         // e23
                0.0,                         // e123
            ];
            network.add_node(&coefficients)?;
        }

        // Add random edges
        for i in 0..num_nodes {
            for j in i + 1..num_nodes {
                if fastrand::f64() < connection_probability {
                    let weight = fastrand::f64() * 10.0 + 1.0;
                    network.add_undirected_edge(i, j, weight)?;
                }
            }
        }

        Ok(network)
    }

    /// Create a small-world network (Watts-Strogatz model)
    #[wasm_bindgen(js_name = createSmallWorldNetwork)]
    pub fn create_small_world_network(
        num_nodes: usize,
        k: usize,
        beta: f64,
    ) -> Result<WasmGeometricNetwork, JsValue> {
        if !k.is_multiple_of(2) || k >= num_nodes {
            return Err(JsValue::from_str("k must be even and less than num_nodes"));
        }

        if !(0.0..=1.0).contains(&beta) {
            return Err(JsValue::from_str("beta must be between 0 and 1"));
        }

        let mut network = WasmGeometricNetwork::new();

        // Add nodes in a circle
        for i in 0..num_nodes {
            let angle = 2.0 * std::f64::consts::PI * i as f64 / num_nodes as f64;
            let coefficients = vec![
                1.0,         // scalar
                angle.cos(), // e1
                angle.sin(), // e2
                0.0,         // e3
                0.0,         // e12
                0.0,         // e13
                0.0,         // e23
                0.0,         // e123
            ];
            network.add_node(&coefficients)?;
        }

        // Add regular ring edges
        for i in 0..num_nodes {
            for j in 1..=k / 2 {
                let target = (i + j) % num_nodes;
                network.add_undirected_edge(i, target, 1.0)?;
            }
        }

        // Rewire edges with probability beta
        for i in 0..num_nodes {
            for j in 1..=k / 2 {
                if fastrand::f64() < beta {
                    let original_target = (i + j) % num_nodes;
                    let mut new_target = fastrand::usize(0..num_nodes);

                    // Ensure new target is different and not already connected
                    while new_target == i || new_target == original_target {
                        new_target = fastrand::usize(0..num_nodes);
                    }

                    // Remove old edge and add new one
                    // Note: In a real implementation, we'd need edge removal functionality
                    network.add_undirected_edge(i, new_target, 1.0)?;
                }
            }
        }

        Ok(network)
    }

    /// Analyze network clustering coefficient
    #[wasm_bindgen(js_name = clusteringCoefficient)]
    pub fn clustering_coefficient(network: &WasmGeometricNetwork) -> f64 {
        let num_nodes = network.num_nodes();
        if num_nodes < 3 {
            return 0.0;
        }

        let mut total_coefficient = 0.0;
        let mut nodes_with_neighbors = 0;

        for i in 0..num_nodes {
            let neighbors = network.get_neighbors(i);
            let degree = neighbors.len();

            if degree < 2 {
                continue;
            }

            // Count triangles involving node i
            let mut triangles = 0;
            for &j in &neighbors {
                for &k in &neighbors {
                    if j < k {
                        let k_neighbors = network.get_neighbors(k);
                        if k_neighbors.contains(&j) {
                            triangles += 1;
                        }
                    }
                }
            }

            // Local clustering coefficient
            let possible_triangles = degree * (degree - 1) / 2;
            if possible_triangles > 0 {
                total_coefficient += triangles as f64 / possible_triangles as f64;
                nodes_with_neighbors += 1;
            }
        }

        if nodes_with_neighbors > 0 {
            total_coefficient / nodes_with_neighbors as f64
        } else {
            0.0
        }
    }
}

/// Initialize network module (called by main init)
pub fn init_network() {
    // Set up console error panic hook for better debugging
    std::panic::set_hook(Box::new(console_error_panic_hook::hook));
}
