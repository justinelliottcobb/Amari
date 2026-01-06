//! Persistent homology for topological data analysis.
//!
//! Persistent homology tracks how topological features (connected components,
//! loops, voids) appear and disappear as we vary a parameter (typically a
//! distance threshold or function level set).

#[cfg(not(feature = "std"))]
use alloc::{vec, vec::Vec};

use crate::complex::SimplicialComplex;
use crate::simplex::Simplex;
use crate::{Result, TopologyError};

/// A filtration is a nested sequence of simplicial complexes.
///
/// K_0 ⊆ K_1 ⊆ ... ⊆ K_n
///
/// Each simplex has a "birth time" indicating when it enters the filtration.
#[derive(Clone, Debug)]
pub struct Filtration {
    /// Simplices ordered by birth time
    simplices: Vec<(f64, Simplex)>,
    /// Whether the filtration is already sorted
    sorted: bool,
}

impl Filtration {
    /// Create an empty filtration.
    pub fn new() -> Self {
        Self {
            simplices: Vec::new(),
            sorted: true,
        }
    }

    /// Add a simplex at the given time.
    pub fn add(&mut self, time: f64, simplex: Simplex) {
        self.simplices.push((time, simplex));
        self.sorted = false;
    }

    /// Ensure the filtration is sorted by time.
    fn ensure_sorted(&mut self) {
        if !self.sorted {
            // Sort by time, then by dimension (lower dimension first)
            self.simplices.sort_by(|a, b| {
                a.0.partial_cmp(&b.0)
                    .unwrap_or(core::cmp::Ordering::Equal)
                    .then_with(|| a.1.dimension().cmp(&b.1.dimension()))
            });
            self.sorted = true;
        }
    }

    /// Get the complex at a given time threshold.
    pub fn complex_at(&mut self, time: f64) -> SimplicialComplex {
        self.ensure_sorted();

        let mut complex = SimplicialComplex::new();
        for (t, simplex) in &self.simplices {
            if *t <= time {
                complex.add_simplex(simplex.clone());
            }
        }
        complex
    }

    /// Get all unique filtration times.
    pub fn times(&mut self) -> Vec<f64> {
        self.ensure_sorted();

        let mut times: Vec<f64> = self.simplices.iter().map(|(t, _)| *t).collect();
        times.dedup();
        times
    }

    /// Validate the filtration (faces must appear before the simplex).
    pub fn validate(&mut self) -> Result<()> {
        self.ensure_sorted();

        for (t, simplex) in &self.simplices {
            for k in 0..simplex.dimension() {
                for face in simplex.faces(k) {
                    // Find the birth time of this face
                    let face_time = self
                        .simplices
                        .iter()
                        .find(|(_, s)| s == &face)
                        .map(|(ft, _)| *ft);

                    match face_time {
                        Some(ft) if ft > *t => {
                            return Err(TopologyError::InvalidFiltration(format!(
                                "Face {:?} born at {} after simplex {:?} at {}",
                                face.vertices(),
                                ft,
                                simplex.vertices(),
                                t
                            )));
                        }
                        None => {
                            return Err(TopologyError::InvalidFiltration(format!(
                                "Face {:?} not in filtration",
                                face.vertices()
                            )));
                        }
                        _ => {}
                    }
                }
            }
        }

        Ok(())
    }

    /// Number of simplices in the filtration.
    pub fn len(&self) -> usize {
        self.simplices.len()
    }

    /// Check if filtration is empty.
    pub fn is_empty(&self) -> bool {
        self.simplices.is_empty()
    }
}

impl Default for Filtration {
    fn default() -> Self {
        Self::new()
    }
}

/// An interval in a persistence barcode.
#[derive(Clone, Debug, PartialEq)]
pub struct BarcodeInterval {
    /// Dimension of the feature (0 = component, 1 = loop, etc.)
    pub dimension: usize,
    /// Birth time
    pub birth: f64,
    /// Death time (infinity for features that never die)
    pub death: Option<f64>,
}

impl BarcodeInterval {
    /// Create a new barcode interval.
    pub fn new(dimension: usize, birth: f64, death: Option<f64>) -> Self {
        Self {
            dimension,
            birth,
            death,
        }
    }

    /// The lifetime (persistence) of this feature.
    pub fn persistence(&self) -> f64 {
        match self.death {
            Some(d) => d - self.birth,
            None => f64::INFINITY,
        }
    }

    /// Check if the feature is still alive at the given time.
    pub fn is_alive_at(&self, time: f64) -> bool {
        time >= self.birth && self.death.map_or(true, |d| time < d)
    }

    /// Check if this is an essential feature (never dies).
    pub fn is_essential(&self) -> bool {
        self.death.is_none()
    }
}

/// A persistence diagram represents birth-death pairs.
#[derive(Clone, Debug, Default)]
pub struct PersistenceDiagram {
    /// Intervals for each dimension
    intervals: Vec<BarcodeInterval>,
}

impl PersistenceDiagram {
    /// Create an empty diagram.
    pub fn new() -> Self {
        Self {
            intervals: Vec::new(),
        }
    }

    /// Add an interval to the diagram.
    pub fn add(&mut self, interval: BarcodeInterval) {
        self.intervals.push(interval);
    }

    /// Get all intervals.
    pub fn intervals(&self) -> &[BarcodeInterval] {
        &self.intervals
    }

    /// Get intervals of a specific dimension.
    pub fn intervals_dim(&self, dim: usize) -> Vec<&BarcodeInterval> {
        self.intervals
            .iter()
            .filter(|i| i.dimension == dim)
            .collect()
    }

    /// Total number of intervals.
    pub fn len(&self) -> usize {
        self.intervals.len()
    }

    /// Check if diagram is empty.
    pub fn is_empty(&self) -> bool {
        self.intervals.is_empty()
    }

    /// Betti numbers at a given time.
    pub fn betti_at(&self, time: f64) -> Vec<usize> {
        let max_dim = self
            .intervals
            .iter()
            .map(|i| i.dimension)
            .max()
            .unwrap_or(0);
        let mut betti = vec![0usize; max_dim + 1];

        for interval in &self.intervals {
            if interval.is_alive_at(time) {
                betti[interval.dimension] += 1;
            }
        }

        betti
    }

    /// Get the most persistent features in each dimension.
    pub fn most_persistent(&self, dim: usize, count: usize) -> Vec<&BarcodeInterval> {
        let mut intervals: Vec<_> = self.intervals_dim(dim);
        intervals.sort_by(|a, b| {
            b.persistence()
                .partial_cmp(&a.persistence())
                .unwrap_or(core::cmp::Ordering::Equal)
        });
        intervals.into_iter().take(count).collect()
    }
}

/// Compute persistent homology of a filtration.
#[derive(Clone, Debug)]
pub struct PersistentHomology {
    /// The computed persistence diagram
    diagram: PersistenceDiagram,
}

impl PersistentHomology {
    /// Compute persistent homology using the standard algorithm.
    ///
    /// This implements the reduction algorithm for persistence.
    pub fn compute(filtration: &mut Filtration) -> Result<Self> {
        filtration.ensure_sorted();

        if filtration.is_empty() {
            return Ok(Self {
                diagram: PersistenceDiagram::new(),
            });
        }

        // Use the standard persistence algorithm
        // For simplicity, we track connected components and use incremental homology

        let mut diagram = PersistenceDiagram::new();

        // Track birth times and use union-find for H_0
        let times = filtration.times();
        if times.is_empty() {
            return Ok(Self { diagram });
        }

        // For each time step, compute homology and track changes
        let mut prev_betti = vec![0usize; 10]; // Support up to dimension 9

        for &t in &times {
            let complex = filtration.complex_at(t);
            let curr_betti = complex.betti_numbers();

            // Extend prev_betti if needed
            while prev_betti.len() < curr_betti.len() {
                prev_betti.push(0);
            }

            for (dim, (&curr, &prev)) in curr_betti.iter().zip(prev_betti.iter()).enumerate() {
                if curr > prev {
                    // New features born
                    for _ in 0..(curr - prev) {
                        diagram.add(BarcodeInterval::new(dim, t, None));
                    }
                } else if curr < prev {
                    // Features died - find the oldest alive ones to kill
                    let to_kill = prev - curr;
                    let mut killed = 0;
                    for interval in diagram.intervals.iter_mut().rev() {
                        if interval.dimension == dim && interval.death.is_none() {
                            interval.death = Some(t);
                            killed += 1;
                            if killed >= to_kill {
                                break;
                            }
                        }
                    }
                }
            }

            prev_betti = curr_betti;
        }

        Ok(Self { diagram })
    }

    /// Get the persistence diagram.
    pub fn diagram(&self) -> &PersistenceDiagram {
        &self.diagram
    }

    /// Get Betti numbers at a specific filtration time.
    pub fn betti_at(&self, time: f64) -> Vec<usize> {
        self.diagram.betti_at(time)
    }
}

/// Create a Rips filtration from points with a distance function.
///
/// The Rips complex at scale ε contains a simplex [v0, ..., vk] if
/// all pairwise distances d(vi, vj) ≤ ε.
pub fn rips_filtration<F>(points: usize, max_dim: usize, distance: F) -> Filtration
where
    F: Fn(usize, usize) -> f64,
{
    let mut filtration = Filtration::new();

    // Add vertices at time 0
    for i in 0..points {
        filtration.add(0.0, Simplex::new(vec![i]));
    }

    // Add edges at their distance
    for i in 0..points {
        for j in (i + 1)..points {
            let d = distance(i, j);
            filtration.add(d, Simplex::new(vec![i, j]));
        }
    }

    // Add higher simplices (simplified: just check max edge length)
    if max_dim >= 2 {
        for i in 0..points {
            for j in (i + 1)..points {
                for k in (j + 1)..points {
                    let d = distance(i, j).max(distance(j, k)).max(distance(i, k));
                    filtration.add(d, Simplex::new(vec![i, j, k]));
                }
            }
        }
    }

    if max_dim >= 3 {
        for i in 0..points {
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
                        filtration.add(d, Simplex::new(vec![i, j, k, l]));
                    }
                }
            }
        }
    }

    filtration
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_filtration() {
        let mut filt = Filtration::new();
        assert!(filt.times().is_empty());
        assert!(filt.is_empty());
    }

    #[test]
    fn test_filtration_ordering() {
        let mut filt = Filtration::new();
        filt.add(1.0, Simplex::new(vec![0, 1]));
        filt.add(0.0, Simplex::new(vec![0]));
        filt.add(0.0, Simplex::new(vec![1]));

        let times = filt.times();
        assert_eq!(times, vec![0.0, 1.0]);
    }

    #[test]
    fn test_barcode_interval() {
        let finite = BarcodeInterval::new(0, 0.0, Some(1.0));
        assert_eq!(finite.persistence(), 1.0);
        assert!(!finite.is_essential());
        assert!(finite.is_alive_at(0.5));
        assert!(!finite.is_alive_at(1.5));

        let essential = BarcodeInterval::new(0, 0.0, None);
        assert_eq!(essential.persistence(), f64::INFINITY);
        assert!(essential.is_essential());
    }

    #[test]
    fn test_growing_complex_persistence() {
        // Start with 3 separate points, then connect them
        let mut filt = Filtration::new();
        filt.add(0.0, Simplex::new(vec![0]));
        filt.add(0.0, Simplex::new(vec![1]));
        filt.add(0.0, Simplex::new(vec![2]));
        filt.add(1.0, Simplex::new(vec![0, 1]));
        filt.add(2.0, Simplex::new(vec![1, 2]));
        filt.add(3.0, Simplex::new(vec![0, 2])); // Creates a loop

        let ph = PersistentHomology::compute(&mut filt).unwrap();
        let diagram = ph.diagram();

        // At t=0: 3 components (β_0 = 3)
        let betti_0 = diagram.betti_at(0.0);
        assert_eq!(betti_0[0], 3);

        // At t=1.5: 2 components (one merged)
        let betti_1 = diagram.betti_at(1.5);
        assert_eq!(betti_1[0], 2);

        // At t=2.5: 1 component
        let betti_2 = diagram.betti_at(2.5);
        assert_eq!(betti_2[0], 1);
    }

    #[test]
    fn test_rips_filtration() {
        // 3 points forming an equilateral triangle
        let points: Vec<(f64, f64)> = vec![(0.0, 0.0), (1.0, 0.0), (0.5, 0.866)];
        let distance = |i: usize, j: usize| -> f64 {
            let (x1, y1) = points[i];
            let (x2, y2) = points[j];
            ((x2 - x1).powi(2) + (y2 - y1).powi(2)).sqrt()
        };

        let filt = rips_filtration(3, 2, distance);
        assert!(!filt.is_empty());
    }
}
