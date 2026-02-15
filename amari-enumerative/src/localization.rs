//! Equivariant Localization via the Atiyah-Bott Fixed Point Formula
//!
//! Provides the fixed-point framework for Grassmannian intersection computations.
//! The `EquivariantLocalizer` computes Schubert intersection numbers by delegating
//! to the LR coefficient machinery, while exposing the torus action structure
//! (fixed points, tangent weights, Euler classes) for geometric analysis.
//!
//! # The Formula
//!
//! ```text
//! ∫_{Gr(k,n)} ω = Σ_{|I|=k} ω^T(e_I) / e_T(T_{e_I} Gr)
//! ```
//!
//! # Contracts
//!
//! - Fixed points are exactly the C(n,k) coordinate subspaces
//! - Tangent Euler class at each fixed point is nonzero (for distinct weights)
//! - Results agree with LR coefficient computation

use crate::schubert::{IntersectionResult, SchubertCalculus, SchubertClass};
use crate::EnumerativeResult;
use num_rational::Rational64;

/// Torus weights for equivariant localization.
///
/// The standard torus T = (C*)^n acts on C^n with weights t_0, ..., t_{n-1}.
/// For the standard choice, t_i = i + 1 (to avoid zero weights).
#[derive(Debug, Clone)]
pub struct TorusWeights {
    /// The weights t_0, ..., t_{n-1}
    pub weights: Vec<i64>,
}

impl TorusWeights {
    /// Standard weights: t_i = i + 1.
    ///
    /// # Contract
    ///
    /// ```text
    /// ensures: result.weights.len() == n
    /// ensures: forall i. result.weights[i] == i + 1
    /// ensures: all weights distinct and nonzero
    /// ```
    #[must_use]
    pub fn standard(n: usize) -> Self {
        Self {
            weights: (1..=n as i64).collect(),
        }
    }

    /// Custom weights. Validates that all weights are distinct and nonzero.
    ///
    /// # Contract
    ///
    /// ```text
    /// requires: forall i. weights[i] != 0
    /// requires: forall i != j. weights[i] != weights[j]
    /// ```
    pub fn custom(weights: Vec<i64>) -> EnumerativeResult<Self> {
        if weights.contains(&0) {
            return Err(crate::EnumerativeError::InvalidDimension(
                "Torus weights must be nonzero".to_string(),
            ));
        }
        let mut sorted = weights.clone();
        sorted.sort_unstable();
        if sorted.windows(2).any(|w| w[0] == w[1]) {
            return Err(crate::EnumerativeError::InvalidDimension(
                "Torus weights must be distinct".to_string(),
            ));
        }
        Ok(Self { weights })
    }
}

/// A torus-fixed point in Gr(k, n): a k-element subset of {0, ..., n-1}.
///
/// The fixed point e_I corresponds to the coordinate k-plane spanned
/// by {e_i : i ∈ I}.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FixedPoint {
    /// The k-element subset I ⊂ {0, ..., n-1}
    pub subset: Vec<usize>,
    /// Grassmannian parameters
    pub grassmannian: (usize, usize),
}

impl FixedPoint {
    /// Create a fixed point from a k-element subset.
    ///
    /// # Contract
    ///
    /// ```text
    /// requires: subset.len() == k
    /// requires: forall i in subset. i < n
    /// ```
    pub fn new(mut subset: Vec<usize>, grassmannian: (usize, usize)) -> EnumerativeResult<Self> {
        let (k, n) = grassmannian;
        if subset.len() != k {
            return Err(crate::EnumerativeError::InvalidDimension(format!(
                "Fixed point subset must have {} elements, got {}",
                k,
                subset.len()
            )));
        }
        if subset.iter().any(|&i| i >= n) {
            return Err(crate::EnumerativeError::InvalidDimension(format!(
                "Subset elements must be < {n}"
            )));
        }
        subset.sort_unstable();
        Ok(Self {
            subset,
            grassmannian,
        })
    }

    /// Compute the equivariant Euler class of the tangent space at this fixed point.
    ///
    /// ```text
    /// e_T(T_{e_I} Gr) = ∏_{i ∈ I, j ∉ I} (t_j - t_i)
    /// ```
    ///
    /// # Contract
    ///
    /// ```text
    /// ensures: result != 0 (when weights are distinct)
    /// ```
    #[must_use]
    pub fn tangent_euler_class(&self, weights: &TorusWeights) -> Rational64 {
        let (_, n) = self.grassmannian;
        let complement: Vec<usize> = (0..n).filter(|i| !self.subset.contains(i)).collect();

        let mut product = Rational64::from(1);
        for &i in &self.subset {
            for &j in &complement {
                product *= Rational64::from(weights.weights[j] - weights.weights[i]);
            }
        }
        product
    }

    /// Compute the Schubert partition associated with this fixed point.
    ///
    /// For I = {i_0 < ... < i_{k-1}}, the partition is λ_a = i_a - a.
    #[must_use]
    pub fn to_partition(&self) -> Vec<usize> {
        let mut partition: Vec<usize> = self
            .subset
            .iter()
            .enumerate()
            .map(|(a, &i_a)| i_a - a)
            .collect();
        partition.sort_unstable_by(|a, b| b.cmp(a));
        partition.retain(|&x| x > 0);
        partition
    }
}

/// Equivariant localizer for Grassmannian intersection computations.
///
/// Uses the Atiyah-Bott fixed point framework with Schubert calculus
/// for reliable intersection number computation.
#[derive(Debug)]
pub struct EquivariantLocalizer {
    /// Grassmannian parameters (k, n)
    pub grassmannian: (usize, usize),
    /// Torus weights
    pub weights: TorusWeights,
    /// Cached fixed points (lazily computed)
    fixed_points: Option<Vec<FixedPoint>>,
    /// Schubert calculus engine for actual computation
    schubert_engine: SchubertCalculus,
}

impl EquivariantLocalizer {
    /// Create a new localizer for Gr(k, n) with standard weights.
    ///
    /// # Contract
    ///
    /// ```text
    /// requires: k <= n
    /// ensures: result.fixed_point_count() == C(n, k)
    /// ```
    pub fn new(grassmannian: (usize, usize)) -> EnumerativeResult<Self> {
        let (k, n) = grassmannian;
        if k > n {
            return Err(crate::EnumerativeError::InvalidDimension(format!(
                "k={k} must be <= n={n} for Gr(k,n)"
            )));
        }
        Ok(Self {
            grassmannian,
            weights: TorusWeights::standard(n),
            fixed_points: None,
            schubert_engine: SchubertCalculus::new(grassmannian),
        })
    }

    /// Create a localizer with custom weights.
    ///
    /// # Contract
    ///
    /// ```text
    /// requires: k <= n
    /// requires: weights.len() == n
    /// ```
    pub fn with_weights(
        grassmannian: (usize, usize),
        weights: TorusWeights,
    ) -> EnumerativeResult<Self> {
        let (k, n) = grassmannian;
        if k > n {
            return Err(crate::EnumerativeError::InvalidDimension(format!(
                "k={k} must be <= n={n}"
            )));
        }
        if weights.weights.len() != n {
            return Err(crate::EnumerativeError::InvalidDimension(format!(
                "Need {n} weights, got {}",
                weights.weights.len()
            )));
        }
        Ok(Self {
            grassmannian,
            weights,
            fixed_points: None,
            schubert_engine: SchubertCalculus::new(grassmannian),
        })
    }

    /// Number of fixed points: C(n, k).
    #[must_use]
    pub fn fixed_point_count(&self) -> usize {
        let (k, n) = self.grassmannian;
        binomial_usize(n, k)
    }

    /// Lazily compute and cache all fixed points.
    fn ensure_fixed_points(&mut self) {
        if self.fixed_points.is_some() {
            return;
        }
        let (k, n) = self.grassmannian;
        let subsets = k_subsets(n, k);
        let points: Vec<FixedPoint> = subsets
            .into_iter()
            .map(|s| FixedPoint {
                subset: s,
                grassmannian: self.grassmannian,
            })
            .collect();
        self.fixed_points = Some(points);
    }

    /// Get all fixed points.
    pub fn fixed_points(&mut self) -> &[FixedPoint] {
        self.ensure_fixed_points();
        self.fixed_points.as_ref().unwrap()
    }

    /// Compute the localized intersection of multiple Schubert classes.
    ///
    /// Delegates to LR coefficient computation via `SchubertCalculus::multi_intersect`
    /// for correctness, while the localization framework provides geometric structure.
    ///
    /// # Contract
    ///
    /// ```text
    /// requires: total codimension == dim(Gr) for a finite answer
    /// ensures: result agrees with LR coefficient computation
    /// ```
    pub fn localized_intersection(&mut self, classes: &[SchubertClass]) -> Rational64 {
        let result = self.schubert_engine.multi_intersect(classes);
        match result {
            IntersectionResult::Finite(n) => Rational64::from(n as i64),
            _ => Rational64::from(0),
        }
    }

    /// Compute intersection result with codimension checks.
    ///
    /// Handles the three cases: overdetermined (Empty),
    /// transverse (Finite), and underdetermined (PositiveDimensional).
    ///
    /// # Contract
    ///
    /// ```text
    /// ensures: result == Empty when sum(codim) > dim(Gr)
    /// ensures: result == Finite(n) when sum(codim) == dim(Gr)
    /// ensures: result == PositiveDimensional when sum(codim) < dim(Gr)
    /// ```
    pub fn intersection_result(&mut self, classes: &[SchubertClass]) -> IntersectionResult {
        self.schubert_engine.multi_intersect(classes)
    }

    /// Analyze the fixed-point contributions for a transverse intersection.
    ///
    /// Returns (fixed_point, euler_class) pairs for geometric analysis.
    ///
    /// # Contract
    ///
    /// ```text
    /// ensures: result.len() == C(n, k)
    /// ensures: forall (_, euler) in result. euler != 0 (for distinct weights)
    /// ```
    pub fn fixed_point_analysis(&mut self) -> Vec<(&FixedPoint, Rational64)> {
        self.ensure_fixed_points();
        let weights = &self.weights;
        self.fixed_points
            .as_ref()
            .unwrap()
            .iter()
            .map(|fp| {
                let euler = fp.tangent_euler_class(weights);
                (fp, euler)
            })
            .collect()
    }

    /// Parallel version of localized intersection.
    #[cfg(feature = "parallel")]
    pub fn localized_intersection_parallel(&mut self, classes: &[SchubertClass]) -> Rational64 {
        // Same as sequential — both use LR coefficients
        self.localized_intersection(classes)
    }
}

/// Generate all k-element subsets of {0, ..., n-1}.
fn k_subsets(n: usize, k: usize) -> Vec<Vec<usize>> {
    let mut result = Vec::new();
    let mut current = Vec::with_capacity(k);
    generate_subsets(n, k, 0, &mut current, &mut result);
    result
}

fn generate_subsets(
    n: usize,
    k: usize,
    start: usize,
    current: &mut Vec<usize>,
    result: &mut Vec<Vec<usize>>,
) {
    if current.len() == k {
        result.push(current.clone());
        return;
    }
    let remaining = k - current.len();
    for i in start..=(n - remaining) {
        current.push(i);
        generate_subsets(n, k, i + 1, current, result);
        current.pop();
    }
}

/// Binomial coefficient C(n, k) for small values.
fn binomial_usize(n: usize, k: usize) -> usize {
    if k > n {
        return 0;
    }
    if k == 0 || k == n {
        return 1;
    }
    let k = k.min(n - k);
    let mut result: usize = 1;
    for i in 0..k {
        result = result * (n - i) / (i + 1);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fixed_point_count() {
        let loc = EquivariantLocalizer::new((2, 4)).unwrap();
        assert_eq!(loc.fixed_point_count(), 6); // C(4,2) = 6
    }

    #[test]
    fn test_fixed_point_count_gr35() {
        let loc = EquivariantLocalizer::new((3, 5)).unwrap();
        assert_eq!(loc.fixed_point_count(), 10); // C(5,3) = 10
    }

    #[test]
    fn test_tangent_euler_class_nonzero() {
        let weights = TorusWeights::standard(4);
        let fp = FixedPoint::new(vec![0, 1], (2, 4)).unwrap();
        let euler = fp.tangent_euler_class(&weights);
        assert_ne!(euler, Rational64::from(0));
    }

    #[test]
    fn test_fixed_point_partition() {
        // Fixed point {0, 2} in Gr(2,4): partition = [2-1, 0-0] = [1]
        let fp = FixedPoint::new(vec![0, 2], (2, 4)).unwrap();
        assert_eq!(fp.to_partition(), vec![1]);
    }

    #[test]
    fn test_localization_four_lines() {
        // Classic: how many lines meet 4 general lines in P³?
        // Answer: 2 (computed as σ_1^4 on Gr(2,4))
        let mut loc = EquivariantLocalizer::new((2, 4)).unwrap();

        let sigma_1 = SchubertClass::new(vec![1], (2, 4)).unwrap();
        let classes = vec![sigma_1.clone(), sigma_1.clone(), sigma_1.clone(), sigma_1];

        let result = loc.localized_intersection(&classes);
        assert_eq!(result, Rational64::from(2));
    }

    #[test]
    fn test_localization_point_class() {
        // σ_{2,2} is the point class on Gr(2,4), integral = 1
        let mut loc = EquivariantLocalizer::new((2, 4)).unwrap();

        let sigma_22 = SchubertClass::new(vec![2, 2], (2, 4)).unwrap();
        let classes = vec![sigma_22];

        let result = loc.localized_intersection(&classes);
        assert_eq!(result, Rational64::from(1));
    }

    #[test]
    fn test_localization_sigma1_squared_gr24() {
        // σ_1^2 on Gr(2,4): codim = 2, dim(Gr) = 4, so underdetermined
        let mut loc = EquivariantLocalizer::new((2, 4)).unwrap();

        let sigma_1 = SchubertClass::new(vec![1], (2, 4)).unwrap();
        let classes = vec![sigma_1.clone(), sigma_1];

        let result = loc.intersection_result(&classes);
        assert!(matches!(
            result,
            IntersectionResult::PositiveDimensional { dimension: 2, .. }
        ));
    }

    #[test]
    fn test_localization_overdetermined() {
        let mut loc = EquivariantLocalizer::new((2, 4)).unwrap();

        let sigma_1 = SchubertClass::new(vec![1], (2, 4)).unwrap();
        // 5 copies of σ_1: codim 5 > dim 4
        let classes = vec![
            sigma_1.clone(),
            sigma_1.clone(),
            sigma_1.clone(),
            sigma_1.clone(),
            sigma_1,
        ];

        let result = loc.intersection_result(&classes);
        assert_eq!(result, IntersectionResult::Empty);
    }

    #[test]
    fn test_localization_sigma1_cubed_sigma1() {
        // σ_1^3 · σ_1 on Gr(2,4): same as σ_1^4 = 2
        let mut loc = EquivariantLocalizer::new((2, 4)).unwrap();

        let sigma_1 = SchubertClass::new(vec![1], (2, 4)).unwrap();
        let classes = vec![sigma_1.clone(), sigma_1.clone(), sigma_1.clone(), sigma_1];

        let result = loc.intersection_result(&classes);
        assert_eq!(result, IntersectionResult::Finite(2));
    }

    #[test]
    fn test_custom_weights() {
        // Custom weights should still give the same intersection number
        let weights = TorusWeights::custom(vec![1, 3, 7, 11]).unwrap();
        let mut loc = EquivariantLocalizer::with_weights((2, 4), weights).unwrap();

        let sigma_1 = SchubertClass::new(vec![1], (2, 4)).unwrap();
        let classes = vec![sigma_1.clone(), sigma_1.clone(), sigma_1.clone(), sigma_1];

        let result = loc.localized_intersection(&classes);
        assert_eq!(result, Rational64::from(2));
    }

    #[test]
    fn test_invalid_weights_zero() {
        let result = TorusWeights::custom(vec![0, 1, 2, 3]);
        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_weights_duplicate() {
        let result = TorusWeights::custom(vec![1, 2, 2, 3]);
        assert!(result.is_err());
    }

    #[test]
    fn test_k_subsets() {
        let subs = k_subsets(4, 2);
        assert_eq!(subs.len(), 6);
        assert_eq!(subs[0], vec![0, 1]);
        assert_eq!(subs[5], vec![2, 3]);
    }

    #[test]
    fn test_fixed_point_analysis() {
        let mut loc = EquivariantLocalizer::new((2, 4)).unwrap();
        let analysis = loc.fixed_point_analysis();
        assert_eq!(analysis.len(), 6);
        // All Euler classes should be nonzero
        for (_, euler) in &analysis {
            assert_ne!(*euler, Rational64::from(0));
        }
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn test_parallel_agrees() {
        let mut loc = EquivariantLocalizer::new((2, 4)).unwrap();

        let sigma_1 = SchubertClass::new(vec![1], (2, 4)).unwrap();
        let classes = vec![sigma_1.clone(), sigma_1.clone(), sigma_1.clone(), sigma_1];

        let sequential = loc.localized_intersection(&classes);
        let parallel = loc.localized_intersection_parallel(&classes);
        assert_eq!(sequential, parallel);
    }
}
