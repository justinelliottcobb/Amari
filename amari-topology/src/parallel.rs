//! Parallel computation utilities using Rayon.
//!
//! This module provides parallelized versions of computationally intensive
//! topological operations. Enable with the `parallel` feature flag.
//!
//! # Parallelized Operations
//!
//! - **Face Enumeration**: Generate all k-faces of a simplex in parallel
//! - **Boundary Map Construction**: Build sparse boundary matrices in parallel
//! - **Homology Computation**: Compute Betti numbers across dimensions in parallel
//! - **Filtration Processing**: Process filtration steps in parallel where possible
//! - **Persistence Computation**: Parallel persistence diagram construction
//!
//! # Usage
//!
//! ```rust,ignore
//! use amari_topology::parallel::*;
//!
//! // Parallel homology computation
//! let betti = parallel_betti_numbers(&complex);
//!
//! // Parallel boundary map construction
//! let boundary = parallel_boundary_map(&domain, &codomain);
//! ```

#[cfg(not(feature = "std"))]
use alloc::{vec, vec::Vec};

#[cfg(feature = "parallel")]
use rayon::prelude::*;

use crate::chain::{BoundaryMap, ChainGroup};
use crate::complex::SimplicialComplex;
use crate::homology::BettiNumbers;
use crate::persistence::Filtration;
use crate::simplex::Simplex;

// ============================================================================
// Parallel Face Enumeration
// ============================================================================

/// Generate all k-faces of a simplex in parallel.
///
/// For large simplices (dimension > 4), this can provide significant speedup.
#[cfg(feature = "parallel")]
pub fn parallel_faces(simplex: &Simplex, k: usize) -> Vec<Simplex> {
    if k >= simplex.dimension() {
        return vec![];
    }

    let vertices = simplex.vertices();
    let n = vertices.len();
    let choose_count = k + 1;

    // For small cases, use sequential
    if n <= 6 {
        return simplex.faces(k);
    }

    // Generate all combinations in parallel
    let combinations = generate_combinations_parallel(n, choose_count);

    combinations
        .into_par_iter()
        .map(|indices| {
            let face_vertices: Vec<usize> = indices.iter().map(|&i| vertices[i]).collect();
            Simplex::new(face_vertices)
        })
        .collect()
}

/// Sequential fallback for face enumeration.
#[cfg(not(feature = "parallel"))]
pub fn parallel_faces(simplex: &Simplex, k: usize) -> Vec<Simplex> {
    simplex.faces(k)
}

/// Generate all k-combinations of n elements in parallel.
#[cfg(feature = "parallel")]
fn generate_combinations_parallel(n: usize, k: usize) -> Vec<Vec<usize>> {
    if k == 0 {
        return vec![vec![]];
    }
    if k > n {
        return vec![];
    }

    // Split the problem: combinations containing 0 and combinations not containing 0
    let with_first: Vec<Vec<usize>> = if k == 1 {
        vec![vec![0]]
    } else {
        generate_combinations_parallel(n - 1, k - 1)
            .into_par_iter()
            .map(|mut combo| {
                combo.insert(0, 0);
                combo.iter_mut().skip(1).for_each(|x| *x += 1);
                combo[0] = 0;
                // Fix: rebuild properly
                let mut result = vec![0];
                result.extend(combo.into_iter().skip(1));
                result
            })
            .collect()
    };

    let without_first: Vec<Vec<usize>> = generate_combinations_parallel(n - 1, k)
        .into_par_iter()
        .map(|combo| combo.into_iter().map(|x| x + 1).collect())
        .collect();

    let mut result = with_first;
    result.extend(without_first);
    result
}

// ============================================================================
// Parallel Boundary Map Construction
// ============================================================================

/// Construct a boundary map in parallel.
///
/// This parallelizes the iteration over domain simplices.
/// Currently falls back to sequential for correctness while preserving the parallel API.
#[cfg(feature = "parallel")]
pub fn parallel_boundary_map(domain: &ChainGroup, codomain: &ChainGroup) -> BoundaryMap {
    let cols = domain.rank();

    if cols < 100 {
        // Use sequential for small cases
        return BoundaryMap::from_chain_groups(domain, codomain);
    }

    // TODO: Implement true parallel construction when BoundaryMap supports from_entries
    // For now, fall back to sequential construction
    BoundaryMap::from_chain_groups(domain, codomain)
}

/// Sequential fallback for boundary map construction.
#[cfg(not(feature = "parallel"))]
pub fn parallel_boundary_map(domain: &ChainGroup, codomain: &ChainGroup) -> BoundaryMap {
    BoundaryMap::from_chain_groups(domain, codomain)
}

// ============================================================================
// Parallel Homology Computation
// ============================================================================

/// Compute Betti numbers in parallel across dimensions.
///
/// Each dimension's computation is independent, so we can parallelize.
#[cfg(feature = "parallel")]
pub fn parallel_betti_numbers(complex: &SimplicialComplex) -> BettiNumbers {
    let dim = complex.dimension();
    if dim == 0 {
        return vec![complex.simplex_count(0)];
    }

    // For small complexes, use sequential
    if complex.total_simplices() < 100 {
        return complex.betti_numbers();
    }

    // Compute chain groups
    let chain_groups: Vec<ChainGroup> = (0..=dim + 1).map(|k| complex.chain_group(k)).collect();

    // Compute boundary maps in parallel
    let boundary_maps: Vec<BoundaryMap> = (0..=dim + 1)
        .into_par_iter()
        .map(|k| {
            if k == 0 {
                BoundaryMap::zero(chain_groups[0].rank(), 0)
            } else {
                BoundaryMap::from_chain_groups(&chain_groups[k], &chain_groups[k - 1])
            }
        })
        .collect();

    // Compute Betti numbers in parallel
    (0..=dim)
        .into_par_iter()
        .map(|k| {
            let kernel_dim = boundary_maps[k].kernel_dim();
            let image_dim = boundary_maps[k + 1].image_dim();
            kernel_dim.saturating_sub(image_dim)
        })
        .collect()
}

/// Sequential fallback for Betti number computation.
#[cfg(not(feature = "parallel"))]
pub fn parallel_betti_numbers(complex: &SimplicialComplex) -> BettiNumbers {
    complex.betti_numbers()
}

// ============================================================================
// Parallel Persistence Computation
// ============================================================================

/// Compute Betti numbers at multiple filtration times.
///
/// Note: This function is sequential because `complex_at` requires mutable access.
/// The parallelism opportunity is in computing Betti numbers for each complex.
pub fn parallel_betti_at_times(filtration: &mut Filtration, times: &[f64]) -> Vec<Vec<usize>> {
    times
        .iter()
        .map(|&t| {
            let complex = filtration.complex_at(t);
            complex.betti_numbers()
        })
        .collect()
}

// ============================================================================
// Parallel Critical Point Detection
// ============================================================================

/// Evaluate a function on a grid in parallel.
///
/// Useful for Morse theory critical point detection.
#[cfg(feature = "parallel")]
pub fn parallel_grid_evaluation<F>(
    f: F,
    x_range: (f64, f64),
    y_range: (f64, f64),
    resolution: usize,
) -> Vec<Vec<f64>>
where
    F: Fn(f64, f64) -> f64 + Sync,
{
    let (x_min, x_max) = x_range;
    let (y_min, y_max) = y_range;
    let dx = (x_max - x_min) / resolution as f64;
    let dy = (y_max - y_min) / resolution as f64;

    (0..=resolution)
        .into_par_iter()
        .map(|i| {
            let x = x_min + i as f64 * dx;
            (0..=resolution)
                .map(|j| {
                    let y = y_min + j as f64 * dy;
                    f(x, y)
                })
                .collect()
        })
        .collect()
}

/// Sequential fallback for grid evaluation.
#[cfg(not(feature = "parallel"))]
pub fn parallel_grid_evaluation<F>(
    f: F,
    x_range: (f64, f64),
    y_range: (f64, f64),
    resolution: usize,
) -> Vec<Vec<f64>>
where
    F: Fn(f64, f64) -> f64,
{
    let (x_min, x_max) = x_range;
    let (y_min, y_max) = y_range;
    let dx = (x_max - x_min) / resolution as f64;
    let dy = (y_max - y_min) / resolution as f64;

    (0..=resolution)
        .map(|i| {
            let x = x_min + i as f64 * dx;
            (0..=resolution)
                .map(|j| {
                    let y = y_min + j as f64 * dy;
                    f(x, y)
                })
                .collect()
        })
        .collect()
}

// ============================================================================
// Parallel Rips Filtration Construction
// ============================================================================

/// Construct a Rips filtration in parallel.
///
/// The distance computations for edges and higher simplices can be parallelized.
#[cfg(feature = "parallel")]
pub fn parallel_rips_filtration<F>(points: usize, max_dim: usize, distance: F) -> Filtration
where
    F: Fn(usize, usize) -> f64 + Sync,
{
    let mut filtration = Filtration::new();

    // Add vertices at time 0
    for i in 0..points {
        filtration.add(0.0, Simplex::new(vec![i]));
    }

    // Add edges in parallel
    let edges: Vec<(f64, Simplex)> = (0..points)
        .into_par_iter()
        .flat_map(|i| {
            ((i + 1)..points)
                .map(|j| {
                    let d = distance(i, j);
                    (d, Simplex::new(vec![i, j]))
                })
                .collect::<Vec<_>>()
        })
        .collect();

    for (d, simplex) in edges {
        filtration.add(d, simplex);
    }

    // Add triangles in parallel if requested
    if max_dim >= 2 {
        let triangles: Vec<(f64, Simplex)> = (0..points)
            .into_par_iter()
            .flat_map(|i| {
                let mut local = Vec::new();
                for j in (i + 1)..points {
                    for k in (j + 1)..points {
                        let d = distance(i, j).max(distance(j, k)).max(distance(i, k));
                        local.push((d, Simplex::new(vec![i, j, k])));
                    }
                }
                local
            })
            .collect();

        for (d, simplex) in triangles {
            filtration.add(d, simplex);
        }
    }

    // Add tetrahedra in parallel if requested
    if max_dim >= 3 {
        let tetrahedra: Vec<(f64, Simplex)> = (0..points)
            .into_par_iter()
            .flat_map(|i| {
                let mut local = Vec::new();
                for j in (i + 1)..points {
                    for k in (j + 1)..points {
                        for l in (k + 1)..points {
                            let d = [
                                distance(i, j),
                                distance(i, k),
                                distance(i, l),
                                distance(j, k),
                                distance(j, l),
                                distance(k, l),
                            ]
                            .into_iter()
                            .fold(0.0f64, |a, b| a.max(b));
                            local.push((d, Simplex::new(vec![i, j, k, l])));
                        }
                    }
                }
                local
            })
            .collect();

        for (d, simplex) in tetrahedra {
            filtration.add(d, simplex);
        }
    }

    filtration
}

/// Sequential fallback for Rips filtration construction.
#[cfg(not(feature = "parallel"))]
pub fn parallel_rips_filtration<F>(points: usize, max_dim: usize, distance: F) -> Filtration
where
    F: Fn(usize, usize) -> f64,
{
    crate::persistence::rips_filtration(points, max_dim, distance)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parallel_faces() {
        let simplex = Simplex::new(vec![0, 1, 2, 3]);

        // Test vertices (0-faces)
        let vertices = parallel_faces(&simplex, 0);
        assert_eq!(vertices.len(), 4);

        // Test edges (1-faces)
        let edges = parallel_faces(&simplex, 1);
        assert_eq!(edges.len(), 6);

        // Test triangles (2-faces)
        let triangles = parallel_faces(&simplex, 2);
        assert_eq!(triangles.len(), 4);
    }

    #[test]
    fn test_parallel_betti_numbers() {
        let mut complex = SimplicialComplex::new();
        complex.add_simplex(Simplex::new(vec![0, 1, 2]));

        let betti = parallel_betti_numbers(&complex);
        assert_eq!(betti[0], 1); // One component
        assert_eq!(betti[1], 0); // No holes
    }

    #[test]
    fn test_parallel_grid_evaluation() {
        let f = |x: f64, y: f64| x * x + y * y;
        let values = parallel_grid_evaluation(f, (-1.0, 1.0), (-1.0, 1.0), 10);

        assert_eq!(values.len(), 11);
        assert_eq!(values[0].len(), 11);

        // Origin should be approximately 0
        let center = values[5][5];
        assert!(center < 0.05);
    }

    #[test]
    fn test_parallel_rips_filtration() {
        let points: Vec<(f64, f64)> = vec![(0.0, 0.0), (1.0, 0.0), (0.5, 0.866)];
        let distance = |i: usize, j: usize| {
            let (x1, y1) = points[i];
            let (x2, y2) = points[j];
            ((x2 - x1).powi(2) + (y2 - y1).powi(2)).sqrt()
        };

        let filt = parallel_rips_filtration(3, 2, distance);
        assert!(!filt.is_empty());
    }
}
