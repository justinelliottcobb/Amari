//! Schubert calculus on Grassmannians and flag varieties
//!
//! This module implements Schubert classes and their intersection theory
//! on Grassmannians and flag varieties.
//!
//! # Contracts
//!
//! The key mathematical invariants maintained:
//!
//! - **Grassmannian bounds**: Partitions fit in the k × (n-k) box
//! - **Intersection dimension**: codim(σ_λ ∩ σ_μ) = codim(σ_λ) + codim(σ_μ) (generically)
//! - **Transversality**: When ∑ codim = dim(Gr), intersection is finite
//! - **Commutativity**: σ_λ · σ_μ = σ_μ · σ_λ
//!
//! # Rayon Parallelization
//!
//! When the `parallel` feature is enabled, computationally intensive operations
//! use parallel iterators for improved performance on multi-core systems.

use crate::littlewood_richardson::{lr_coefficient, schubert_product, Partition};
use crate::{ChowClass, EnumerativeError, EnumerativeResult};
use num_rational::Rational64;
use std::collections::{BTreeMap, HashMap};

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Result of a Schubert intersection computation
///
/// # Contracts
///
/// - `Empty`: Returned when total codimension exceeds Grassmannian dimension
/// - `Finite(n)`: Returned when total codimension equals Grassmannian dimension
/// - `PositiveDimensional`: Returned when total codimension is less than Grassmannian dimension
///
/// # Invariant
///
/// ```text
/// requires: total_codim = sum of codimensions of input classes
/// ensures:
///   - total_codim > dim(Gr) => Empty
///   - total_codim == dim(Gr) => Finite(n) where n >= 0
///   - total_codim < dim(Gr) => PositiveDimensional { dimension: dim(Gr) - total_codim }
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum IntersectionResult {
    /// Empty intersection (overdetermined)
    #[default]
    Empty,
    /// Finite number of points
    Finite(u64),
    /// Positive-dimensional intersection
    PositiveDimensional {
        /// Dimension of the intersection
        dimension: usize,
        /// Degree (if computable)
        degree: Option<u64>,
    },
}

/// Schubert class indexed by a Young diagram/partition
///
/// # Contracts
///
/// - Each partition entry must be ≤ n-k
/// - Partition entries should be weakly decreasing (though we allow flexibility)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SchubertClass {
    /// The partition indexing this Schubert class
    pub partition: Vec<usize>,
    /// Dimension of the underlying Grassmannian
    pub grassmannian_dim: (usize, usize), // (k, n) for Gr(k, n)
}

impl SchubertClass {
    /// Create a new Schubert class
    ///
    /// # Contract
    ///
    /// ```text
    /// requires: forall i. partition[i] <= n - k
    /// ensures: result.codimension() == partition.iter().sum()
    /// ```
    ///
    /// # Errors
    ///
    /// Returns `SchubertError` if any partition entry exceeds n-k.
    pub fn new(partition: Vec<usize>, grassmannian_dim: (usize, usize)) -> EnumerativeResult<Self> {
        let (k, n) = grassmannian_dim;

        // Validate partition - each part must be ≤ n-k
        for &part in &partition {
            if part > n - k {
                return Err(EnumerativeError::SchubertError(format!(
                    "Partition entry {} exceeds n-k = {}",
                    part,
                    n - k
                )));
            }
        }

        Ok(Self {
            partition,
            grassmannian_dim,
        })
    }

    /// Create from a Partition type
    pub fn from_partition(
        partition: Partition,
        grassmannian_dim: (usize, usize),
    ) -> EnumerativeResult<Self> {
        Self::new(partition.parts, grassmannian_dim)
    }

    /// Convert to a Partition type
    #[must_use]
    pub fn to_partition(&self) -> Partition {
        Partition::new(self.partition.clone())
    }

    /// Convert to a Chow class
    #[must_use]
    pub fn to_chow_class(&self) -> ChowClass {
        let codimension = self.partition.iter().sum::<usize>();
        let degree = Rational64::from(1);

        ChowClass::new(codimension, degree)
    }

    /// Compute the dimension of this Schubert variety
    ///
    /// # Contract
    ///
    /// ```text
    /// ensures: result == k * (n - k) - self.codimension()
    /// ensures: result <= k * (n - k)
    /// ```
    #[must_use]
    pub fn dimension(&self) -> usize {
        let (k, n) = self.grassmannian_dim;
        let total_dim = k * (n - k);
        let codim = self.partition.iter().sum::<usize>();
        total_dim - codim
    }

    /// Compute the codimension of this Schubert variety
    ///
    /// # Contract
    ///
    /// ```text
    /// ensures: result == partition.iter().sum()
    /// ```
    #[must_use]
    pub fn codimension(&self) -> usize {
        self.partition.iter().sum()
    }

    /// Raise Schubert class to a power (repeated intersection)
    #[must_use]
    pub fn power(&self, exponent: usize) -> SchubertClass {
        // Simplified - real power requires sophisticated Schubert calculus
        let mut new_partition = self.partition.clone();
        for _ in 1..exponent {
            if !new_partition.is_empty() {
                new_partition[0] += 1;
            } else {
                new_partition.push(1);
            }
        }

        SchubertClass {
            partition: new_partition,
            grassmannian_dim: self.grassmannian_dim,
        }
    }

    /// Giambelli determinant formula
    pub fn giambelli_determinant(
        partition: &[usize],
        grassmannian_dim: (usize, usize),
    ) -> EnumerativeResult<Self> {
        Self::new(partition.to_vec(), grassmannian_dim)
    }
}

/// Schubert calculus engine
#[derive(Debug)]
pub struct SchubertCalculus {
    /// The underlying Grassmannian
    pub grassmannian_dim: (usize, usize),
    /// Cache for computed intersection numbers
    intersection_cache: HashMap<(Vec<usize>, Vec<usize>), Rational64>,
    /// Cache for LR coefficients
    lr_cache: BTreeMap<(Partition, Partition, Partition), u64>,
}

impl Default for SchubertCalculus {
    fn default() -> Self {
        Self::new((2, 4)) // Default to Gr(2,4)
    }
}

impl SchubertCalculus {
    /// Create a new Schubert calculus engine
    #[must_use]
    pub fn new(grassmannian_dim: (usize, usize)) -> Self {
        Self {
            grassmannian_dim,
            intersection_cache: HashMap::new(),
            lr_cache: BTreeMap::new(),
        }
    }

    /// Get the dimension of the Grassmannian
    ///
    /// # Contract
    ///
    /// ```text
    /// ensures: result == k * (n - k)
    /// ```
    #[must_use]
    pub fn grassmannian_dimension(&self) -> usize {
        let (k, n) = self.grassmannian_dim;
        k * (n - k)
    }

    /// Compute intersection number of two Schubert classes
    ///
    /// # Contract
    ///
    /// ```text
    /// requires: class1.grassmannian_dim == class2.grassmannian_dim == self.grassmannian_dim
    /// ensures: result >= 0
    /// ensures: class1.dimension() + class2.dimension() != self.grassmannian_dimension()
    ///          => result == 0
    /// ```
    pub fn intersection_number(
        &mut self,
        class1: &SchubertClass,
        class2: &SchubertClass,
    ) -> EnumerativeResult<Rational64> {
        // Check cache first
        let key = (class1.partition.clone(), class2.partition.clone());
        if let Some(&cached) = self.intersection_cache.get(&key) {
            return Ok(cached);
        }

        // Use actual Schubert calculus for two classes
        let result = if class1.dimension() + class2.dimension() == self.grassmannian_dimension() {
            // Transverse intersection - compute via LR coefficients
            let p1 = class1.to_partition();
            let p2 = class2.to_partition();
            let (k, n) = self.grassmannian_dim;
            let fundamental = Partition::new(vec![n - k; k]);

            let coeff = lr_coefficient(&p1, &p2, &fundamental);
            Rational64::from(coeff as i64)
        } else {
            Rational64::from(0)
        };

        // Cache the result
        self.intersection_cache.insert(key, result);
        Ok(result)
    }

    /// Intersect multiple Schubert classes
    ///
    /// Given classes σ_{λ_1}, ..., σ_{λ_m}, compute their intersection number
    /// in the Grassmannian Gr(k, n).
    ///
    /// # Contract
    ///
    /// ```text
    /// requires: forall c in classes. c.grassmannian_dim == self.grassmannian_dim
    /// ensures:
    ///   - sum(c.codimension() for c in classes) > dim(Gr) => Empty
    ///   - sum(c.codimension() for c in classes) == dim(Gr) => Finite(n)
    ///   - sum(c.codimension() for c in classes) < dim(Gr) => PositiveDimensional
    /// ```
    pub fn multi_intersect(&mut self, classes: &[SchubertClass]) -> IntersectionResult {
        if classes.is_empty() {
            return IntersectionResult::PositiveDimensional {
                dimension: self.grassmannian_dimension(),
                degree: Some(1),
            };
        }

        let grassmannian_dim = self.grassmannian_dimension();

        // Total codimension
        let total_codim: usize = classes.iter().map(|c| c.codimension()).sum();

        match total_codim.cmp(&grassmannian_dim) {
            std::cmp::Ordering::Greater => IntersectionResult::Empty,
            std::cmp::Ordering::Less => {
                let remaining_dim = grassmannian_dim - total_codim;
                IntersectionResult::PositiveDimensional {
                    dimension: remaining_dim,
                    degree: self.compute_degree_if_easy(classes),
                }
            }
            std::cmp::Ordering::Equal => {
                // Transverse intersection
                let count = self.compute_transverse_intersection(classes);
                IntersectionResult::Finite(count)
            }
        }
    }

    /// Compute intersection number when codimensions sum to Grassmannian dimension
    fn compute_transverse_intersection(&mut self, classes: &[SchubertClass]) -> u64 {
        if classes.is_empty() {
            return 1;
        }

        if classes.len() == 1 {
            // Single class at top dimension
            let (k, n) = self.grassmannian_dim;
            let fundamental = vec![n - k; k];
            if classes[0].partition == fundamental {
                return 1;
            } else {
                return 0;
            }
        }

        // Convert to partitions and iteratively multiply
        let partitions: Vec<Partition> = classes.iter().map(|c| c.to_partition()).collect();

        self.multiply_partitions(&partitions)
    }

    /// Multiply partitions and extract fundamental class coefficient
    ///
    /// # Rayon Parallelization
    ///
    /// When many partitions need to be multiplied and the intermediate
    /// products generate many terms, parallel computation can speed this up.
    fn multiply_partitions(&mut self, partitions: &[Partition]) -> u64 {
        let (k, n) = self.grassmannian_dim;

        // Start with first partition
        let mut current: BTreeMap<Partition, u64> = BTreeMap::new();
        current.insert(partitions[0].clone(), 1);

        // Iteratively multiply
        for partition in &partitions[1..] {
            let next = self.multiply_step(&current, partition, k, n);
            current = next;
        }

        // Extract coefficient of fundamental class
        let fundamental = Partition::new(vec![n - k; k]);
        current.get(&fundamental).copied().unwrap_or(0)
    }

    /// Single multiplication step (can be parallelized)
    #[cfg(feature = "parallel")]
    fn multiply_step(
        &self,
        current: &BTreeMap<Partition, u64>,
        partition: &Partition,
        k: usize,
        n: usize,
    ) -> BTreeMap<Partition, u64> {
        // Parallel version: collect into pairs and merge
        let pairs: Vec<_> = current.iter().collect();

        let partial_results: Vec<BTreeMap<Partition, u64>> = pairs
            .par_iter()
            .map(|(nu, coeff)| {
                let products = schubert_product(nu, partition, (k, n));
                let mut local: BTreeMap<Partition, u64> = BTreeMap::new();
                for (rho, lr_coeff) in products {
                    *local.entry(rho).or_insert(0) += **coeff * lr_coeff;
                }
                local
            })
            .collect();

        // Merge all partial results
        let mut next: BTreeMap<Partition, u64> = BTreeMap::new();
        for partial in partial_results {
            for (rho, coeff) in partial {
                *next.entry(rho).or_insert(0) += coeff;
            }
        }
        next
    }

    /// Single multiplication step (sequential version)
    #[cfg(not(feature = "parallel"))]
    fn multiply_step(
        &self,
        current: &BTreeMap<Partition, u64>,
        partition: &Partition,
        k: usize,
        n: usize,
    ) -> BTreeMap<Partition, u64> {
        let mut next: BTreeMap<Partition, u64> = BTreeMap::new();

        for (nu, coeff) in current {
            let products = schubert_product(nu, partition, (k, n));
            for (rho, lr_coeff) in products {
                *next.entry(rho).or_insert(0) += *coeff * lr_coeff;
            }
        }

        next
    }

    fn compute_degree_if_easy(&self, _classes: &[SchubertClass]) -> Option<u64> {
        // Degree computation for positive-dimensional intersection
        // is more complex; return None for now
        None
    }

    /// Get or compute LR coefficient with caching
    ///
    /// # Contract
    ///
    /// ```text
    /// ensures: result == lr_coefficient(lambda, mu, nu)
    /// ensures: lr_cached(lambda, mu, nu) == lr_cached(mu, lambda, nu)  // symmetry
    /// ```
    pub fn lr_cached(&mut self, lambda: &Partition, mu: &Partition, nu: &Partition) -> u64 {
        // Normalize key (LR coefficients are symmetric in λ, μ)
        let (a, b) = if lambda <= mu {
            (lambda.clone(), mu.clone())
        } else {
            (mu.clone(), lambda.clone())
        };

        let key = (a, b, nu.clone());

        if let Some(&cached) = self.lr_cache.get(&key) {
            return cached;
        }

        let result = lr_coefficient(lambda, mu, nu);
        self.lr_cache.insert(key, result);
        result
    }

    /// Expand product of two Schubert classes
    ///
    /// # Contract
    ///
    /// ```text
    /// requires: class1.grassmannian_dim == class2.grassmannian_dim == self.grassmannian_dim
    /// ensures: forall (c, coeff) in result. coeff > 0
    /// ensures: product(class1, class2) == product(class2, class1)  // commutativity
    /// ```
    #[must_use]
    pub fn product(
        &mut self,
        class1: &SchubertClass,
        class2: &SchubertClass,
    ) -> Vec<(SchubertClass, u64)> {
        let p1 = class1.to_partition();
        let p2 = class2.to_partition();

        let products = schubert_product(&p1, &p2, self.grassmannian_dim);

        products
            .into_iter()
            .filter_map(|(partition, coeff)| {
                SchubertClass::new(partition.parts, self.grassmannian_dim)
                    .ok()
                    .map(|class| (class, coeff))
            })
            .collect()
    }

    /// Multiply two Schubert classes using Pieri's rule (simplified)
    pub fn pieri_multiply(
        &self,
        schubert_class: &SchubertClass,
        special_class: usize,
    ) -> EnumerativeResult<Vec<SchubertClass>> {
        // Simplified Pieri rule - adds horizontal strips
        let mut results = Vec::new();
        let (k, n) = self.grassmannian_dim;

        // Option 1: Add to the first row (if it exists)
        if !schubert_class.partition.is_empty() {
            let mut new_partition = schubert_class.partition.clone();
            new_partition[0] += special_class;

            // Check if this partition is valid
            if new_partition[0] <= n - k {
                if let Ok(new_class) = SchubertClass::new(new_partition, self.grassmannian_dim) {
                    results.push(new_class);
                }
            }
        }

        // Option 2: Add a new row
        let mut new_partition = schubert_class.partition.clone();
        new_partition.push(special_class);

        // Check if this partition is valid
        if special_class <= n - k {
            if let Ok(new_class) = SchubertClass::new(new_partition, self.grassmannian_dim) {
                results.push(new_class);
            }
        }

        // If no valid results, return the original approach
        if results.is_empty() {
            let mut new_partition = schubert_class.partition.clone();
            if !new_partition.is_empty() {
                new_partition[0] += special_class;
            } else {
                new_partition.push(special_class);
            }
            results.push(SchubertClass::new(new_partition, self.grassmannian_dim)?);
        }

        Ok(results)
    }
}

/// Flag variety F(n1, n2, ..., nk; n)
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FlagVariety {
    /// Dimensions of the flags
    pub flag_dims: Vec<usize>,
    /// Ambient dimension
    pub ambient_dim: usize,
}

impl FlagVariety {
    /// Create a new flag variety
    ///
    /// # Contract
    ///
    /// ```text
    /// requires: flag_dims is strictly increasing
    /// requires: flag_dims.last() < ambient_dim
    /// ensures: result.dimension() >= 0
    /// ```
    pub fn new(flag_dims: Vec<usize>, ambient_dim: usize) -> EnumerativeResult<Self> {
        // Validate that flag dimensions are increasing
        for i in 1..flag_dims.len() {
            if flag_dims[i] <= flag_dims[i - 1] {
                return Err(EnumerativeError::SchubertError(
                    "Flag dimensions must be strictly increasing".to_string(),
                ));
            }
        }

        if flag_dims.last().copied().unwrap_or(0) >= ambient_dim {
            return Err(EnumerativeError::SchubertError(
                "Largest flag dimension must be less than ambient dimension".to_string(),
            ));
        }

        Ok(Self {
            flag_dims,
            ambient_dim,
        })
    }

    /// Compute the dimension of the flag variety
    #[must_use]
    pub fn dimension(&self) -> usize {
        let mut dim = 0;
        let mut prev_dim = 0;

        for &flag_dim in &self.flag_dims {
            dim += (flag_dim - prev_dim) * (self.ambient_dim - prev_dim);
            prev_dim = flag_dim;
        }

        dim
    }
}

// ============================================================================
// Parallel Batch Operations
// ============================================================================

/// Compute multiple Schubert intersections in parallel
///
/// # Contract
///
/// ```text
/// requires: forall batch in batches. batch.classes all have same grassmannian_dim
/// ensures: result.len() == batches.len()
/// ```
#[cfg(feature = "parallel")]
pub fn multi_intersect_batch(
    batches: &[(Vec<SchubertClass>, (usize, usize))],
) -> Vec<IntersectionResult> {
    batches
        .par_iter()
        .map(|(classes, grassmannian_dim)| {
            let mut calc = SchubertCalculus::new(*grassmannian_dim);
            calc.multi_intersect(classes)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_schubert_class_creation() {
        let class = SchubertClass::new(vec![2, 1], (3, 6)).unwrap();
        assert_eq!(class.partition, vec![2, 1]);
        assert_eq!(class.codimension(), 3);
    }

    #[test]
    fn test_intersection_result_default() {
        let result = IntersectionResult::default();
        assert_eq!(result, IntersectionResult::Empty);
    }

    #[test]
    fn test_schubert_calculus_default() {
        let calc = SchubertCalculus::default();
        assert_eq!(calc.grassmannian_dim, (2, 4));
    }

    #[test]
    fn test_multi_intersect_four_lines() {
        // Classic: how many lines meet 4 general lines in P³?
        // This is σ_1^4 in Gr(2,4) = 2
        let mut calc = SchubertCalculus::new((2, 4));
        let sigma_1 = SchubertClass::new(vec![1], (2, 4)).unwrap();

        let classes = vec![
            sigma_1.clone(),
            sigma_1.clone(),
            sigma_1.clone(),
            sigma_1.clone(),
        ];

        let result = calc.multi_intersect(&classes);
        assert_eq!(result, IntersectionResult::Finite(2));
    }

    #[test]
    fn test_multi_intersect_underdetermined() {
        let mut calc = SchubertCalculus::new((2, 4));
        let sigma_1 = SchubertClass::new(vec![1], (2, 4)).unwrap();

        // Only 2 conditions in Gr(2,4) which has dimension 4
        let classes = vec![sigma_1.clone(), sigma_1.clone()];

        let result = calc.multi_intersect(&classes);
        assert!(matches!(
            result,
            IntersectionResult::PositiveDimensional { dimension: 2, .. }
        ));
    }

    #[test]
    fn test_multi_intersect_overdetermined() {
        let mut calc = SchubertCalculus::new((2, 4));
        let sigma_1 = SchubertClass::new(vec![1], (2, 4)).unwrap();

        // 5 conditions exceeds dimension 4
        let classes = vec![
            sigma_1.clone(),
            sigma_1.clone(),
            sigma_1.clone(),
            sigma_1.clone(),
            sigma_1.clone(),
        ];

        let result = calc.multi_intersect(&classes);
        assert_eq!(result, IntersectionResult::Empty);
    }

    #[test]
    fn test_product() {
        let mut calc = SchubertCalculus::new((2, 4));
        let sigma_1 = SchubertClass::new(vec![1], (2, 4)).unwrap();

        let products = calc.product(&sigma_1, &sigma_1);

        // σ_1 · σ_1 = σ_2 + σ_{1,1}
        assert_eq!(products.len(), 2);

        let partitions: Vec<Vec<usize>> =
            products.iter().map(|(c, _)| c.partition.clone()).collect();
        assert!(partitions.contains(&vec![2]));
        assert!(partitions.contains(&vec![1, 1]));
    }

    #[test]
    fn test_partition_conversion() {
        let class = SchubertClass::new(vec![3, 2, 1], (4, 8)).unwrap();
        let partition = class.to_partition();
        assert_eq!(partition.parts, vec![3, 2, 1]);

        let class2 = SchubertClass::from_partition(partition, (4, 8)).unwrap();
        assert_eq!(class2.partition, vec![3, 2, 1]);
    }

    #[test]
    fn test_flag_variety() {
        let flag = FlagVariety::new(vec![1, 2], 4).unwrap();
        assert!(flag.dimension() > 0);
    }
}

// ============================================================================
// Parallel Batch Operation Tests
// ============================================================================

#[cfg(all(test, feature = "parallel"))]
mod parallel_tests {
    use super::*;

    #[test]
    fn test_multi_intersect_batch() {
        // Test multiple intersection problems in parallel
        let sigma_1_gr24 = SchubertClass::new(vec![1], (2, 4)).unwrap();
        let sigma_1_gr25 = SchubertClass::new(vec![1], (2, 5)).unwrap();

        let batches = vec![
            // σ_1^4 in Gr(2,4) = 2
            (vec![sigma_1_gr24.clone(); 4], (2, 4)),
            // σ_1^6 in Gr(2,5) should be finite
            (vec![sigma_1_gr25.clone(); 6], (2, 5)),
            // Overdetermined: 5 conditions in dim 4
            (vec![sigma_1_gr24.clone(); 5], (2, 4)),
            // Underdetermined: 2 conditions in dim 4
            (vec![sigma_1_gr24.clone(); 2], (2, 4)),
        ];

        let results = multi_intersect_batch(&batches);

        assert_eq!(results.len(), 4);
        assert_eq!(results[0], IntersectionResult::Finite(2));
        assert!(matches!(results[1], IntersectionResult::Finite(_)));
        assert_eq!(results[2], IntersectionResult::Empty);
        assert!(matches!(
            results[3],
            IntersectionResult::PositiveDimensional { dimension: 2, .. }
        ));
    }

    #[test]
    fn test_multi_intersect_batch_empty() {
        let batches: Vec<(Vec<SchubertClass>, (usize, usize))> = vec![];
        let results = multi_intersect_batch(&batches);
        assert!(results.is_empty());
    }

    #[test]
    fn test_multi_intersect_batch_single() {
        let sigma_1 = SchubertClass::new(vec![1], (2, 4)).unwrap();
        let batches = vec![(vec![sigma_1; 4], (2, 4))];

        let results = multi_intersect_batch(&batches);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0], IntersectionResult::Finite(2));
    }
}
