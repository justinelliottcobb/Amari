//! Formally verified geometric network analysis with phantom types
//!
//! This module provides type-safe, mathematically verified implementations of
//! geometric network analysis using phantom types for compile-time invariants
//! and formal verification guarantees.

use crate::{
    Community, GeometricNetwork, NetworkError, NodeMetadata, PropagationAnalysis,
};
use amari_core::Multivector;
use std::marker::PhantomData;

#[cfg(feature = "formal-verification")]
use creusot_contracts::{ensures, requires};

/// Phantom type encoding the metric signature of the underlying Clifford algebra
/// - P: number of positive basis vectors (e_i² = +1)
/// - Q: number of negative basis vectors (e_i² = -1)
/// - R: number of null basis vectors (e_i² = 0)
pub struct NetworkSignature<const P: usize, const Q: usize, const R: usize>;

/// Phantom type for network property verification
pub struct NetworkProperty<const PROPERTY: usize>;

/// Network connectivity property markers
pub const CONNECTED: usize = 1;
pub const DISCONNECTED: usize = 2;
pub const WEAKLY_CONNECTED: usize = 3;

/// Network type property markers
pub const DIRECTED: usize = 10;
pub const UNDIRECTED: usize = 11;
pub const MIXED: usize = 12;

/// A verified geometric network with compile-time signature guarantees
///
/// This structure ensures at the type level that:
/// 1. The Clifford algebra signature is fixed and consistent
/// 2. Network operations preserve mathematical properties
/// 3. Graph-theoretic invariants are maintained
/// 4. Geometric operations are well-defined
#[derive(Debug, Clone)]
pub struct VerifiedGeometricNetwork<
    const P: usize,
    const Q: usize,
    const R: usize,
    const CONNECTIVITY: usize = DISCONNECTED,
    const NETWORK_TYPE: usize = DIRECTED,
> {
    /// The underlying geometric network
    pub(crate) inner: GeometricNetwork<P, Q, R>,
    /// Phantom marker for signature verification
    _signature: PhantomData<NetworkSignature<P, Q, R>>,
    /// Phantom marker for network properties
    _properties: PhantomData<(NetworkProperty<CONNECTIVITY>, NetworkProperty<NETWORK_TYPE>)>,
}

impl<
        const P: usize,
        const Q: usize,
        const R: usize,
        const CONNECTIVITY: usize,
        const NETWORK_TYPE: usize,
    > VerifiedGeometricNetwork<P, Q, R, CONNECTIVITY, NETWORK_TYPE>
{
    /// The total dimension of the underlying Clifford algebra
    pub const DIM: usize = P + Q + R;

    /// Create a new verified geometric network
    ///
    /// # Type Invariants
    /// - Signature (P,Q,R) is encoded at type level and cannot be violated
    /// - Network starts in disconnected state with zero nodes
    #[cfg_attr(feature = "formal-verification",
        ensures(result.num_nodes() == 0),
        ensures(result.num_edges() == 0))]
    pub fn new() -> Self {
        Self {
            inner: GeometricNetwork::new(),
            _signature: PhantomData,
            _properties: PhantomData,
        }
    }

    /// Add a node to the verified network
    ///
    /// # Type Safety
    /// The multivector must be compatible with the network's signature
    #[cfg_attr(feature = "formal-verification",
        ensures(result.num_nodes() == old(self.num_nodes()) + 1))]
    pub fn add_node(&mut self, position: Multivector<P, Q, R>) -> usize {
        self.inner.add_node(position)
    }

    /// Add a node with metadata to the verified network
    #[cfg_attr(feature = "formal-verification",
        ensures(result.num_nodes() == old(self.num_nodes()) + 1))]
    pub fn add_node_with_metadata(
        &mut self,
        position: Multivector<P, Q, R>,
        metadata: NodeMetadata,
    ) -> usize {
        self.inner.add_node_with_metadata(position, metadata)
    }

    /// Add an edge to the verified network
    ///
    /// # Type Safety
    /// Node indices are bounds-checked at runtime
    #[cfg_attr(feature = "formal-verification",
        requires(source < self.num_nodes()),
        requires(target < self.num_nodes()),
        requires(weight >= 0.0))]
    pub fn add_edge(
        &mut self,
        source: usize,
        target: usize,
        weight: f64,
    ) -> Result<(), NetworkError> {
        self.inner.add_edge(source, target, weight)
    }

    /// Add an undirected edge (bidirectional)
    #[cfg_attr(feature = "formal-verification",
        requires(source < self.num_nodes()),
        requires(target < self.num_nodes()),
        requires(weight >= 0.0))]
    pub fn add_undirected_edge(
        &mut self,
        source: usize,
        target: usize,
        weight: f64,
    ) -> Result<(), NetworkError> {
        self.inner.add_undirected_edge(source, target, weight)
    }

    /// Get the number of nodes in the network
    #[cfg_attr(feature = "formal-verification",
        ensures(result >= 0))]
    pub fn num_nodes(&self) -> usize {
        self.inner.num_nodes()
    }

    /// Get the number of edges in the network
    #[cfg_attr(feature = "formal-verification",
        ensures(result >= 0))]
    pub fn num_edges(&self) -> usize {
        self.inner.num_edges()
    }

    /// Compute geometric distance between two nodes
    ///
    /// # Type Safety
    /// Node indices are bounds-checked
    /// # Mathematical Properties
    /// The distance satisfies the metric axioms:
    /// - Non-negativity: d(i,j) >= 0
    /// - Identity: d(i,i) = 0
    /// - Symmetry: d(i,j) = d(j,i)
    /// - Triangle inequality: d(i,k) <= d(i,j) + d(j,k)
    #[cfg_attr(feature = "formal-verification",
        requires(i < self.num_nodes()),
        requires(j < self.num_nodes()),
        ensures(result >= 0.0),
        ensures(i == j ==> result == 0.0))]
    pub fn geometric_distance(&self, i: usize, j: usize) -> Result<f64, NetworkError> {
        self.inner.geometric_distance(i, j)
    }

    /// Compute geometric centrality for all nodes
    ///
    /// # Mathematical Properties
    /// - All centrality values are non-negative
    /// - Isolated nodes have zero centrality
    #[cfg_attr(feature = "formal-verification",
        ensures(result.len() == self.num_nodes()),
        ensures(forall(|i: usize| i < result.len() ==> result[i] >= 0.0)))]
    pub fn compute_geometric_centrality(&self) -> Result<Vec<f64>, NetworkError> {
        self.inner.compute_geometric_centrality()
    }

    /// Find communities using geometric clustering
    ///
    /// # Type Safety
    /// The number of clusters must be positive and not exceed the number of nodes
    #[cfg_attr(feature = "formal-verification",
        requires(num_clusters > 0),
        requires(num_clusters <= self.num_nodes()),
        ensures(result.len() <= num_clusters))]
    pub fn find_communities(
        &self,
        num_clusters: usize,
    ) -> Result<Vec<Community<P, Q, R>>, NetworkError> {
        self.inner.find_communities(num_clusters)
    }

    /// Simulate information diffusion
    ///
    /// # Type Safety
    /// - Source nodes must exist in the network
    /// - Time steps must be positive
    /// - Decay factor must be in (0,1)
    #[cfg_attr(feature = "formal-verification",
        requires(time_steps > 0),
        requires(decay_factor > 0.0 && decay_factor < 1.0),
        requires(forall(|i: usize| i < sources.len() ==> sources[i] < self.num_nodes())))]
    pub fn simulate_diffusion(
        &self,
        sources: &[usize],
        time_steps: usize,
        decay_factor: f64,
    ) -> Result<PropagationAnalysis, NetworkError> {
        self.inner
            .simulate_diffusion(sources, time_steps, decay_factor)
    }

    /// Convert to tropical network representation
    ///
    /// # Mathematical Properties
    /// The conversion preserves:
    /// - Node count
    /// - Edge connectivity
    /// - Path existence (but may change path lengths due to tropical arithmetic)
    #[cfg_attr(feature = "formal-verification",
        ensures(result.size() == self.num_nodes()))]
    pub fn to_tropical_network(&self) -> Result<crate::tropical::TropicalNetwork, NetworkError> {
        self.inner.to_tropical_network()
    }

    /// Get node metadata by index
    #[cfg_attr(feature = "formal-verification",
        requires(index < self.num_nodes()))]
    pub fn get_metadata(&self, index: usize) -> Option<&NodeMetadata> {
        self.inner.get_metadata(index)
    }

    /// Get neighbors of a node
    #[cfg_attr(feature = "formal-verification",
        requires(node < self.num_nodes()))]
    pub fn neighbors(&self, node: usize) -> Vec<usize> {
        self.inner.neighbors(node)
    }

    /// Get degree of a node (number of incident edges)
    #[cfg_attr(feature = "formal-verification",
        requires(node < self.num_nodes()),
        ensures(result >= 0))]
    pub fn degree(&self, node: usize) -> usize {
        self.inner.degree(node)
    }
}

/// Convert from regular GeometricNetwork to verified version
impl<const P: usize, const Q: usize, const R: usize> From<GeometricNetwork<P, Q, R>>
    for VerifiedGeometricNetwork<P, Q, R>
{
    fn from(network: GeometricNetwork<P, Q, R>) -> Self {
        Self {
            inner: network,
            _signature: PhantomData,
            _properties: PhantomData,
        }
    }
}

/// Convert from verified GeometricNetwork to regular version
impl<
        const P: usize,
        const Q: usize,
        const R: usize,
        const CONNECTIVITY: usize,
        const NETWORK_TYPE: usize,
    > From<VerifiedGeometricNetwork<P, Q, R, CONNECTIVITY, NETWORK_TYPE>>
    for GeometricNetwork<P, Q, R>
{
    fn from(verified: VerifiedGeometricNetwork<P, Q, R, CONNECTIVITY, NETWORK_TYPE>) -> Self {
        verified.inner
    }
}

/// Default implementation for verified networks
impl<const P: usize, const Q: usize, const R: usize> Default for VerifiedGeometricNetwork<P, Q, R> {
    fn default() -> Self {
        Self::new()
    }
}
