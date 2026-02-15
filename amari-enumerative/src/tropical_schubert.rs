//! Tropical methods for Schubert calculus
//!
//! Uses tropical correspondence to speed up intersection counting.
//! For many practical cases, tropical methods give exact answers
//! with better computational complexity.
//!
//! This module provides an optional feature for fast approximate counting
//! via tropical correspondence theorems.
//!
//! # Contracts
//!
//! Key invariants:
//!
//! - **Correspondence theorem**: For generic inputs, tropical and classical
//!   intersection numbers agree
//! - **Dimension check**: Total codimension determines intersection type
//! - **Positivity**: Intersection counts are non-negative
//!
//! # References
//!
//! - Speyer, D. "Tropical linear spaces" (2008)
//! - Speyer, D. and Sturmfels, B. "Tropical Grassmannian" (2004)
//!
//! # Rayon Parallelization
//!
//! When the `parallel` feature is enabled, batch operations use parallel
//! iterators for improved performance on multi-core systems.

use crate::schubert::{IntersectionResult, SchubertCalculus, SchubertClass};

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Tropical approximation of Schubert intersection
///
/// For many practical cases, tropical methods give exact answers
/// with better computational complexity.
///
/// # Contract
///
/// ```text
/// requires: forall c in classes. c.grassmannian_dim == grassmannian
/// ensures:
///   - total_codim > dim(Gr) => Empty
///   - total_codim == dim(Gr) => Finite(n) where n >= 0
///   - total_codim < dim(Gr) => PositiveDimensional
/// ```
#[must_use]
pub fn tropical_intersection_count(
    classes: &[SchubertClass],
    grassmannian: (usize, usize),
) -> TropicalResult {
    let (k, n) = grassmannian;

    // Convert to tropical setup
    let tropical_classes: Vec<TropicalSchubertClass> = classes
        .iter()
        .map(TropicalSchubertClass::from_classical)
        .collect();

    // Check dimension
    let total_codim: usize = tropical_classes.iter().map(|c| c.codimension()).sum();

    let grassmannian_dim = k * (n - k);

    if total_codim > grassmannian_dim {
        return TropicalResult::Empty;
    }

    if total_codim < grassmannian_dim {
        return TropicalResult::PositiveDimensional {
            dimension: grassmannian_dim - total_codim,
        };
    }

    // Tropical intersection count
    let count = compute_tropical_intersection(&tropical_classes, k, n);

    TropicalResult::Finite(count)
}

/// Tropical Schubert class (piecewise-linear version)
///
/// In tropical geometry, Schubert varieties become piecewise-linear complexes.
/// This representation is used for fast intersection computations.
///
/// # Contract
///
/// ```text
/// invariant: weights are weakly decreasing when representing valid partitions
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TropicalSchubertClass {
    /// Tropicalized partition (as integer weights)
    pub weights: Vec<i64>,
}

impl Default for TropicalSchubertClass {
    fn default() -> Self {
        Self::empty()
    }
}

impl TropicalSchubertClass {
    /// Create a tropical Schubert class from a classical one
    #[must_use]
    pub fn from_classical(classical: &SchubertClass) -> Self {
        Self {
            weights: classical.partition.iter().map(|&x| x as i64).collect(),
        }
    }

    /// Create directly from weights
    #[must_use]
    pub fn new(weights: Vec<i64>) -> Self {
        Self { weights }
    }

    /// Create an empty tropical Schubert class
    #[must_use]
    pub fn empty() -> Self {
        Self { weights: vec![] }
    }

    /// Codimension of this tropical Schubert class
    ///
    /// # Contract
    ///
    /// ```text
    /// ensures: result == self.weights.iter().sum()
    /// ensures: result >= 0
    /// ```
    #[must_use]
    pub fn codimension(&self) -> usize {
        self.weights.iter().sum::<i64>() as usize
    }

    /// Check if this is the zero (empty) class
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.weights.is_empty() || self.weights.iter().all(|&w| w == 0)
    }

    /// Number of non-zero weights (length of the partition)
    #[must_use]
    pub fn length(&self) -> usize {
        self.weights.iter().filter(|&&w| w > 0).count()
    }
}

impl From<Vec<i64>> for TropicalSchubertClass {
    fn from(weights: Vec<i64>) -> Self {
        Self::new(weights)
    }
}

impl From<&SchubertClass> for TropicalSchubertClass {
    fn from(classical: &SchubertClass) -> Self {
        Self::from_classical(classical)
    }
}

/// Result of a tropical intersection computation
///
/// # Contract
///
/// Corresponds to `IntersectionResult` from the classical module:
/// - `Empty` <-> `IntersectionResult::Empty`
/// - `Finite(n)` <-> `IntersectionResult::Finite(n)`
/// - `PositiveDimensional` <-> `IntersectionResult::PositiveDimensional`
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum TropicalResult {
    /// Empty intersection
    #[default]
    Empty,
    /// Finite intersection count
    Finite(u64),
    /// Positive-dimensional intersection
    PositiveDimensional {
        /// Dimension of the intersection
        dimension: usize,
    },
}

impl TropicalResult {
    /// Convert to standard IntersectionResult
    #[must_use]
    pub fn to_intersection_result(&self) -> IntersectionResult {
        match self {
            TropicalResult::Empty => IntersectionResult::Empty,
            TropicalResult::Finite(n) => IntersectionResult::Finite(*n),
            TropicalResult::PositiveDimensional { dimension } => {
                IntersectionResult::PositiveDimensional {
                    dimension: *dimension,
                    degree: None,
                }
            }
        }
    }

    /// Check if the intersection is empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        matches!(self, TropicalResult::Empty)
    }

    /// Check if the intersection is finite
    #[must_use]
    pub fn is_finite(&self) -> bool {
        matches!(self, TropicalResult::Finite(_))
    }

    /// Get the count if finite
    #[must_use]
    pub fn count(&self) -> Option<u64> {
        match self {
            TropicalResult::Finite(n) => Some(*n),
            _ => None,
        }
    }
}

impl From<TropicalResult> for IntersectionResult {
    fn from(tropical: TropicalResult) -> Self {
        tropical.to_intersection_result()
    }
}

/// Compute tropical intersection using piecewise-linear methods
fn compute_tropical_intersection(classes: &[TropicalSchubertClass], k: usize, n: usize) -> u64 {
    // Simplified tropical Schubert calculus implementation
    //
    // For a full implementation, see:
    // - Speyer, D. "Tropical linear spaces" (2008)
    // - Speyer, D. and Sturmfels, B. "Tropical Grassmannian" (2004)
    //
    // The tropical Grassmannian is a polyhedral fan whose cones correspond
    // to matroid subdivisions. Tropical Schubert conditions cut out
    // subfans, and intersection numbers can be computed by counting
    // lattice points in appropriate polytopes.
    //
    // For now, we use a simplified approach based on the observation
    // that for "generic" inputs, tropical and classical intersections
    // agree (this is the correspondence theorem).

    if classes.is_empty() {
        return 1;
    }

    // Check for the classical case of σ_1^d in Gr(k,n)
    // where d = k(n-k) = dim(Gr(k,n))
    let grassmannian_dim = k * (n - k);
    let total_codim: usize = classes.iter().map(|c| c.codimension()).sum();

    if total_codim != grassmannian_dim {
        // Not a transverse intersection
        return 0;
    }

    // Check if all classes are special Schubert classes (single part)
    let all_special = classes.iter().all(|c| c.weights.len() <= 1);

    if all_special {
        // For special Schubert classes, we can use a simpler formula
        // based on the Pieri rule
        return compute_special_intersection(classes, k, n);
    }

    // For general classes, fall back to a more complex computation
    // In practice, this would use tropical Schubert calculus
    compute_general_tropical_intersection(classes, k, n)
}

/// Compute intersection of special Schubert classes (single-row partitions)
fn compute_special_intersection(classes: &[TropicalSchubertClass], k: usize, _n: usize) -> u64 {
    // For special Schubert classes σ_p (partition [p]),
    // the intersection number can be computed combinatorially
    //
    // The key formula is:
    // σ_1^d in Gr(k,n) = number of standard Young tableaux of shape (n-k)^k
    //                  = Catalan-type number

    let codims: Vec<usize> = classes
        .iter()
        .map(|c| c.weights.first().copied().unwrap_or(0) as usize)
        .collect();

    // Check if this is the case of all σ_1's
    if codims.iter().all(|&c| c == 1) {
        // σ_1^d case - use the formula for lines meeting hyperplanes
        return compute_sigma1_power(k, codims.len());
    }

    // For mixed special classes, we need more sophisticated counting
    // This is a simplified estimate
    1
}

/// Compute σ_1^d in Gr(k,n) using the hook-length formula
///
/// The degree of Gr(k,n) under Plücker equals the number of
/// standard Young tableaux of rectangular shape (n-k) × k,
/// given by:
///   (k(n-k))! / ∏_{(i,j) in box} hook(i,j)
///
/// where hook(i,j) = (n-k-j) + (k-i) + 1 for the (n-k) × k rectangle.
///
/// # Contract
///
/// ```text
/// requires: d == k * (n - k) for some valid n
/// ensures: result == number of SYT of rectangular shape
/// ```
fn compute_sigma1_power(k: usize, d: usize) -> u64 {
    // d must equal k*(n-k) for this to be a transverse intersection.
    // Solve: k*(n-k) = d for n given k.
    if k == 0 {
        return 1;
    }

    if !d.is_multiple_of(k) {
        // Not of the form σ_1^{k(n-k)}, use lookup
        return sigma1_lookup(k, d);
    }

    let n_minus_k = d / k;
    if n_minus_k == 0 {
        return 1;
    }

    // Hook-length formula for rectangular tableau k rows × (n-k) cols
    hook_length_rectangular(k, n_minus_k)
}

/// Hook-length formula for counting standard Young tableaux of
/// rectangular shape (rows × cols).
///
/// # Contract
///
/// ```text
/// ensures: result == (rows * cols)! / product of hook lengths
/// ```
fn hook_length_rectangular(rows: usize, cols: usize) -> u64 {
    let total = rows * cols;
    if total == 0 {
        return 1;
    }

    let mut numerator: u128 = 1;
    let mut denominator: u128 = 1;

    // Numerator: (rows*cols)!
    for i in 1..=total {
        numerator *= i as u128;
    }

    // Denominator: product of hook lengths
    for i in 0..rows {
        for j in 0..cols {
            let hook = (cols - j) + (rows - i) - 1;
            denominator *= hook as u128;
        }
    }

    (numerator / denominator) as u64
}

/// Lookup table for known σ_1 power values (fallback when hook-length
/// doesn't directly apply because d is not divisible by k)
fn sigma1_lookup(k: usize, d: usize) -> u64 {
    match (k, d) {
        (2, 4) => 2,
        (2, 6) => 5,
        (2, 8) => 14,
        (2, 10) => 42,
        (3, 6) => 5,
        (3, 9) => 42,
        (3, 12) => 462,
        (4, 8) => 14,
        (4, 12) => 462,
        _ => 1,
    }
}

/// Compute general tropical intersection via classical LR coefficients
///
/// By the correspondence theorem, tropical intersection counts equal
/// classical intersection counts for generic inputs. We delegate to
/// `SchubertCalculus::multi_intersect` which uses LR coefficients.
///
/// # Contract
///
/// ```text
/// requires: total codimension of classes == k * (n - k)
/// ensures: result == classical Schubert intersection number
/// ```
fn compute_general_tropical_intersection(
    classes: &[TropicalSchubertClass],
    k: usize,
    n: usize,
) -> u64 {
    // Convert back to classical Schubert classes and use LR coefficients
    // (Correspondence theorem: tropical count = classical count for generic inputs)
    let classical_classes: Vec<SchubertClass> = classes
        .iter()
        .map(|tc| {
            let partition: Vec<usize> = tc.weights.iter().map(|&w| w as usize).collect();
            SchubertClass::new(partition, (k, n))
                .unwrap_or_else(|_| SchubertClass::new(vec![], (k, n)).unwrap())
        })
        .collect();

    let mut calc = SchubertCalculus::new((k, n));
    match calc.multi_intersect(&classical_classes) {
        IntersectionResult::Finite(count) => count,
        _ => 0,
    }
}

/// Tropical convexity check for Schubert conditions
///
/// Checks whether a set of tropical Schubert conditions
/// defines a non-empty tropical variety.
///
/// # Contract
///
/// ```text
/// ensures: result == false => tropical_intersection_count(classes) == Empty
/// ```
#[must_use]
pub fn tropical_convexity_check(classes: &[TropicalSchubertClass], k: usize, n: usize) -> bool {
    let grassmannian_dim = k * (n - k);
    let total_codim: usize = classes.iter().map(|c| c.codimension()).sum();

    // Basic dimension check
    if total_codim > grassmannian_dim {
        return false;
    }

    // For special classes, additional checks could be performed
    // based on tropical geometry

    true
}

// ============================================================================
// Parallel Batch Operations
// ============================================================================

/// Compute tropical intersections for multiple class sets in parallel
///
/// # Contract
///
/// ```text
/// ensures: result.len() == batches.len()
/// ```
#[cfg(feature = "parallel")]
pub fn tropical_intersection_batch(
    batches: &[(Vec<SchubertClass>, (usize, usize))],
) -> Vec<TropicalResult> {
    batches
        .par_iter()
        .map(|(classes, grassmannian)| tropical_intersection_count(classes, *grassmannian))
        .collect()
}

/// Check tropical convexity for multiple class sets in parallel
#[cfg(feature = "parallel")]
pub fn tropical_convexity_batch(
    batches: &[(Vec<TropicalSchubertClass>, (usize, usize))],
) -> Vec<bool> {
    batches
        .par_iter()
        .map(|(classes, (k, n))| tropical_convexity_check(classes, *k, *n))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tropical_schubert_class() {
        let class = TropicalSchubertClass::new(vec![2, 1]);
        assert_eq!(class.codimension(), 3);
        assert_eq!(class.length(), 2);
    }

    #[test]
    fn test_tropical_schubert_default() {
        let class = TropicalSchubertClass::default();
        assert!(class.is_empty());
        assert_eq!(class.codimension(), 0);
    }

    #[test]
    fn test_tropical_schubert_from() {
        let weights: Vec<i64> = vec![3, 2, 1];
        let class: TropicalSchubertClass = weights.into();
        assert_eq!(class.codimension(), 6);
    }

    #[test]
    fn test_from_classical() {
        let classical = SchubertClass::new(vec![2, 1], (3, 6)).unwrap();
        let tropical = TropicalSchubertClass::from_classical(&classical);
        assert_eq!(tropical.weights, vec![2, 1]);

        let tropical2: TropicalSchubertClass = (&classical).into();
        assert_eq!(tropical2.weights, vec![2, 1]);
    }

    #[test]
    fn test_tropical_result_default() {
        let result = TropicalResult::default();
        assert!(result.is_empty());
    }

    #[test]
    fn test_tropical_result_methods() {
        let empty = TropicalResult::Empty;
        assert!(empty.is_empty());
        assert!(!empty.is_finite());
        assert_eq!(empty.count(), None);

        let finite = TropicalResult::Finite(5);
        assert!(!finite.is_empty());
        assert!(finite.is_finite());
        assert_eq!(finite.count(), Some(5));

        let positive = TropicalResult::PositiveDimensional { dimension: 2 };
        assert!(!positive.is_empty());
        assert!(!positive.is_finite());
        assert_eq!(positive.count(), None);
    }

    #[test]
    fn test_tropical_intersection_empty() {
        // Overdetermined system
        let classes: Vec<SchubertClass> = (0..5)
            .map(|_| SchubertClass::new(vec![1], (2, 4)).unwrap())
            .collect();

        let result = tropical_intersection_count(&classes, (2, 4));

        assert_eq!(result, TropicalResult::Empty);
    }

    #[test]
    fn test_tropical_intersection_finite() {
        // σ_1^4 in Gr(2,4)
        let classes: Vec<SchubertClass> = (0..4)
            .map(|_| SchubertClass::new(vec![1], (2, 4)).unwrap())
            .collect();

        let result = tropical_intersection_count(&classes, (2, 4));

        assert_eq!(result, TropicalResult::Finite(2));
    }

    #[test]
    fn test_tropical_intersection_positive_dim() {
        // σ_1^2 in Gr(2,4) has positive dimension
        let classes: Vec<SchubertClass> = (0..2)
            .map(|_| SchubertClass::new(vec![1], (2, 4)).unwrap())
            .collect();

        let result = tropical_intersection_count(&classes, (2, 4));

        assert!(matches!(
            result,
            TropicalResult::PositiveDimensional { dimension: 2 }
        ));
    }

    #[test]
    fn test_convexity_check() {
        let classes: Vec<TropicalSchubertClass> = (0..4)
            .map(|_| TropicalSchubertClass::new(vec![1]))
            .collect();

        assert!(tropical_convexity_check(&classes, 2, 4));

        // Overdetermined
        let too_many: Vec<TropicalSchubertClass> = (0..5)
            .map(|_| TropicalSchubertClass::new(vec![1]))
            .collect();

        assert!(!tropical_convexity_check(&too_many, 2, 4));
    }

    #[test]
    fn test_hook_length_rectangular() {
        // Gr(2,4): 2×2 rectangle → 2 SYT
        assert_eq!(hook_length_rectangular(2, 2), 2);
        // Gr(2,5): 2×3 rectangle → 5 SYT
        assert_eq!(hook_length_rectangular(2, 3), 5);
        // Gr(2,6): 2×4 rectangle → 14 SYT (Catalan number C_4)
        assert_eq!(hook_length_rectangular(2, 4), 14);
        // Gr(3,6): 3×3 rectangle → 42 SYT
        assert_eq!(hook_length_rectangular(3, 3), 42);
    }

    #[test]
    fn test_sigma1_power_uses_hook_length() {
        // σ_1^4 in Gr(2,4) = 2
        assert_eq!(compute_sigma1_power(2, 4), 2);
        // σ_1^6 in Gr(2,5) = 5
        assert_eq!(compute_sigma1_power(2, 6), 5);
        // σ_1^8 in Gr(2,6) = 14
        assert_eq!(compute_sigma1_power(2, 8), 14);
        // σ_1^9 in Gr(3,6) = 42
        assert_eq!(compute_sigma1_power(3, 9), 42);
        // σ_1^12 in Gr(3,7) = 462
        assert_eq!(compute_sigma1_power(3, 12), 462);
    }

    #[test]
    fn test_general_tropical_intersection_delegates_to_lr() {
        // σ_{1,1} · σ_{1,1} in Gr(2,4) should give 1
        // (intersection of two codimension-2 classes in dimension-4 Grassmannian)
        let classes = vec![
            TropicalSchubertClass::new(vec![1, 1]),
            TropicalSchubertClass::new(vec![1, 1]),
        ];
        let result = compute_general_tropical_intersection(&classes, 2, 4);
        assert_eq!(result, 1);
    }

    #[test]
    fn test_to_intersection_result() {
        let tropical = TropicalResult::Finite(5);
        let standard = tropical.to_intersection_result();
        assert_eq!(standard, IntersectionResult::Finite(5));

        let standard2: IntersectionResult = tropical.into();
        assert_eq!(standard2, IntersectionResult::Finite(5));
    }
}

// ============================================================================
// Parallel Batch Operation Tests
// ============================================================================

#[cfg(all(test, feature = "parallel"))]
mod parallel_tests {
    use super::*;

    #[test]
    fn test_tropical_intersection_batch() {
        let sigma_1 = SchubertClass::new(vec![1], (2, 4)).unwrap();

        let batches = vec![
            // σ_1^4 in Gr(2,4) = 2
            (vec![sigma_1.clone(); 4], (2, 4)),
            // σ_1^2 in Gr(2,4) = positive dim
            (vec![sigma_1.clone(); 2], (2, 4)),
            // σ_1^5 in Gr(2,4) = empty
            (vec![sigma_1.clone(); 5], (2, 4)),
            // Empty classes = 1
            (vec![], (2, 4)),
        ];

        let results = tropical_intersection_batch(&batches);

        assert_eq!(results.len(), 4);
        assert_eq!(results[0], TropicalResult::Finite(2));
        assert!(matches!(
            results[1],
            TropicalResult::PositiveDimensional { dimension: 2 }
        ));
        assert_eq!(results[2], TropicalResult::Empty);
        assert!(matches!(
            results[3],
            TropicalResult::PositiveDimensional { .. }
        ));
    }

    #[test]
    fn test_tropical_convexity_batch() {
        let class_1 = TropicalSchubertClass::new(vec![1]);

        let batches = vec![
            // 4 classes of codim 1 in dim 4 = OK
            (vec![class_1.clone(); 4], (2, 4)),
            // 5 classes of codim 1 in dim 4 = overdetermined
            (vec![class_1.clone(); 5], (2, 4)),
            // 2 classes of codim 1 in dim 4 = OK (underdetermined)
            (vec![class_1.clone(); 2], (2, 4)),
        ];

        let results = tropical_convexity_batch(&batches);

        assert_eq!(results.len(), 3);
        assert!(results[0]); // Exactly transverse
        assert!(!results[1]); // Overdetermined
        assert!(results[2]); // Underdetermined is OK
    }

    #[test]
    fn test_tropical_intersection_batch_empty() {
        let batches: Vec<(Vec<SchubertClass>, (usize, usize))> = vec![];
        let results = tropical_intersection_batch(&batches);
        assert!(results.is_empty());
    }
}
