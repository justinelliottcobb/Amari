//! Tropical network analysis using max-plus algebra
//!
//! This module provides efficient graph algorithms using tropical (max-plus)
//! algebra, where addition becomes max and multiplication becomes addition.
//! This is particularly useful for shortest path computations.

use crate::{NetworkError, NetworkResult};
use amari_tropical::TropicalNumber;

/// Network using tropical algebra for efficient path operations
///
/// Represents a weighted graph using tropical numbers, enabling efficient
/// computation of shortest paths via tropical matrix operations.
#[derive(Clone, Debug)]
pub struct TropicalNetwork {
    /// Adjacency matrix using tropical numbers
    adjacency: Vec<Vec<TropicalNumber<f64>>>,
    /// Number of nodes in the network
    size: usize,
}

impl TropicalNetwork {
    /// Create a new tropical network with given size
    ///
    /// Initializes all distances to tropical zero (negative infinity),
    /// representing no connection between nodes.
    pub fn new(size: usize) -> Self {
        let adjacency = vec![vec![TropicalNumber::zero(); size]; size];
        Self { adjacency, size }
    }

    /// Create tropical network from weight matrix
    ///
    /// Converts a standard weighted adjacency matrix to tropical representation.
    /// Infinite weights are treated as no connection (tropical zero).
    pub fn from_weights(weights: &[Vec<f64>]) -> NetworkResult<Self> {
        if weights.is_empty() {
            return Err(NetworkError::EmptyNetwork);
        }

        let size = weights.len();
        for row in weights {
            if row.len() != size {
                return Err(NetworkError::DimensionMismatch {
                    expected: size,
                    got: row.len(),
                });
            }
        }

        let mut adjacency = vec![vec![TropicalNumber::zero(); size]; size];

        for i in 0..size {
            for j in 0..size {
                if i == j {
                    // Distance to self is tropical one (additive identity = 0)
                    adjacency[i][j] = TropicalNumber::tropical_one();
                } else if weights[i][j].is_finite() && weights[i][j] > 0.0 {
                    adjacency[i][j] = TropicalNumber::new(weights[i][j]);
                }
                // else remains tropical zero (no connection)
            }
        }

        Ok(Self { adjacency, size })
    }

    /// Get the number of nodes
    pub fn size(&self) -> usize {
        self.size
    }

    /// Set edge weight between two nodes
    pub fn set_edge(&mut self, source: usize, target: usize, weight: f64) -> NetworkResult<()> {
        if source >= self.size {
            return Err(NetworkError::NodeIndexOutOfBounds(source));
        }
        if target >= self.size {
            return Err(NetworkError::NodeIndexOutOfBounds(target));
        }

        self.adjacency[source][target] = if weight.is_finite() && weight > 0.0 {
            TropicalNumber::new(weight)
        } else {
            TropicalNumber::zero()
        };

        Ok(())
    }

    /// Get edge weight between two nodes
    pub fn get_edge(&self, source: usize, target: usize) -> NetworkResult<TropicalNumber<f64>> {
        if source >= self.size {
            return Err(NetworkError::NodeIndexOutOfBounds(source));
        }
        if target >= self.size {
            return Err(NetworkError::NodeIndexOutOfBounds(target));
        }

        Ok(self.adjacency[source][target])
    }

    /// Find shortest path using tropical algebra
    ///
    /// Uses tropical matrix operations to compute shortest path distance
    /// and reconstruct the actual path.
    pub fn shortest_path_tropical(
        &self,
        source: usize,
        target: usize,
    ) -> NetworkResult<Option<(Vec<usize>, f64)>> {
        if source >= self.size {
            return Err(NetworkError::NodeIndexOutOfBounds(source));
        }
        if target >= self.size {
            return Err(NetworkError::NodeIndexOutOfBounds(target));
        }

        // Compute all-pairs shortest paths
        let distances = self.all_pairs_shortest_paths()?;

        let distance = distances[source][target];
        if distance.is_zero() {
            return Ok(None); // No path exists
        }

        // Reconstruct path using parent tracking
        let path = self.reconstruct_path(source, target, &distances)?;
        Ok(Some((path, distance.value())))
    }

    /// Compute all-pairs shortest paths using tropical matrix powers
    ///
    /// Uses the tropical semiring where:
    /// - Addition is max (∨)
    /// - Multiplication is addition (+)
    /// - Zero is -∞ (no connection)
    /// - One is 0 (distance to self)
    pub fn all_pairs_shortest_paths(&self) -> NetworkResult<Vec<Vec<TropicalNumber<f64>>>> {
        let mut distances = self.adjacency.clone();

        // Floyd-Warshall algorithm in tropical semiring
        for k in 0..self.size {
            for i in 0..self.size {
                for j in 0..self.size {
                    let path_through_k = distances[i][k].tropical_add(distances[k][j]);
                    distances[i][j] = distances[i][j].tropical_add(path_through_k);
                }
            }
        }

        Ok(distances)
    }

    /// Compute network betweenness centrality using tropical algebra
    ///
    /// Measures how often each node lies on shortest paths between
    /// other pairs of nodes.
    pub fn tropical_betweenness(&self) -> NetworkResult<Vec<f64>> {
        let distances = self.all_pairs_shortest_paths()?;
        let mut betweenness = vec![0.0; self.size];

        for s in 0..self.size {
            for t in 0..self.size {
                if s == t {
                    continue;
                }

                let st_distance = distances[s][t];
                if st_distance.is_zero() {
                    continue; // No path from s to t
                }

                // Count nodes on shortest paths from s to t
                for v in 0..self.size {
                    if v == s || v == t {
                        continue;
                    }

                    let sv_distance = distances[s][v];
                    let vt_distance = distances[v][t];

                    if !sv_distance.is_zero() && !vt_distance.is_zero() {
                        let path_through_v = sv_distance.tropical_add(vt_distance);

                        // Check if path through v is a shortest path
                        if (path_through_v.value() - st_distance.value()).abs() < 1e-10 {
                            betweenness[v] += 1.0;
                        }
                    }
                }
            }
        }

        // Normalize by number of node pairs
        let normalization = ((self.size - 1) * (self.size - 2)) as f64;
        if normalization > 0.0 {
            for value in &mut betweenness {
                *value /= normalization;
            }
        }

        Ok(betweenness)
    }

    /// Reconstruct shortest path from distance matrix
    fn reconstruct_path(
        &self,
        source: usize,
        target: usize,
        distances: &[Vec<TropicalNumber<f64>>],
    ) -> NetworkResult<Vec<usize>> {
        if source == target {
            return Ok(vec![source]);
        }

        let target_distance = distances[source][target];
        if target_distance.is_zero() {
            return Err(NetworkError::ComputationError(
                "No path exists between nodes".to_string(),
            ));
        }

        let mut path = vec![source];
        let mut current = source;

        while current != target {
            let mut found_next = false;

            for next in 0..self.size {
                if next == current {
                    continue;
                }

                let remaining_distance = distances[next][target];
                if remaining_distance.is_zero() {
                    continue;
                }

                let edge_weight = self.adjacency[current][next];
                if edge_weight.is_zero() {
                    continue;
                }

                let total_distance = edge_weight.tropical_add(remaining_distance);

                // Check if this is on a shortest path
                if (total_distance.value() - distances[current][target].value()).abs() < 1e-10 {
                    path.push(next);
                    current = next;
                    found_next = true;
                    break;
                }
            }

            if !found_next {
                return Err(NetworkError::ComputationError(
                    "Failed to reconstruct path".to_string(),
                ));
            }
        }

        Ok(path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_tropical_network_creation() {
        let network = TropicalNetwork::new(3);
        assert_eq!(network.size(), 3);
    }

    #[test]
    fn test_simple_shortest_path() {
        let weights = vec![
            vec![0.0, 1.0, f64::INFINITY],
            vec![f64::INFINITY, 0.0, 1.0],
            vec![f64::INFINITY, f64::INFINITY, 0.0],
        ];

        let network = TropicalNetwork::from_weights(&weights).unwrap();
        let result = network.shortest_path_tropical(0, 2).unwrap();

        assert!(result.is_some());
        let (path, distance) = result.unwrap();
        assert_eq!(path, vec![0, 1, 2]);
        assert_relative_eq!(distance, 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_no_path() {
        let weights = vec![vec![0.0, 1.0], vec![f64::INFINITY, 0.0]];

        let network = TropicalNetwork::from_weights(&weights).unwrap();
        let result = network.shortest_path_tropical(1, 0).unwrap();

        assert!(result.is_none());
    }

    #[test]
    fn test_betweenness_centrality() {
        // Linear chain: 0 -> 1 -> 2
        let weights = vec![
            vec![0.0, 1.0, f64::INFINITY],
            vec![f64::INFINITY, 0.0, 1.0],
            vec![f64::INFINITY, f64::INFINITY, 0.0],
        ];

        let network = TropicalNetwork::from_weights(&weights).unwrap();
        let betweenness = network.tropical_betweenness().unwrap();

        // Node 1 should have highest betweenness (lies on path 0->2)
        assert!(betweenness[1] > betweenness[0]);
        assert!(betweenness[1] > betweenness[2]);
    }
}
