Implement amari-measure v0.10.0
Pre-Prompt Context
[Include the full Amari Development Pre-Prompt Template]
Crate Overview
Crate Name: amari-measure
Version: 0.10.0
Purpose: Measure-theoretic foundations for geometric algebra - rigorous integration, probability measures, and analysis on multivector spaces
Core Capabilities:

Geometric measures (multivector-valued measures)
Lebesgue integration of multivector fields
Radon-Nikodym derivatives and densities
Pushforward and pullback of measures
Product measures and Fubini's theorem
Signed and complex measures
Convergence theorems (monotone, dominated, Fatou)

Mathematical Foundation
Measure Theory Basics
A measure μ on a σ-algebra Σ is a function μ: Σ → [0, ∞] satisfying:

μ(∅) = 0
Countable additivity: μ(⋃ᵢ Aᵢ) = Σᵢ μ(Aᵢ) for disjoint {Aᵢ}

Geometric Measures
A geometric measure is a multivector-valued measure:
μ: Σ → Cl(p,q,r)
This extends real-valued measures to geometric algebra, enabling:

Vector-valued densities (e.g., velocity distributions)
Bivector-valued measures (e.g., oriented area elements)
Full multivector integration

Lebesgue Integration
For measurable function f and measure μ:
∫ f dμ = sup { ∫ s dμ : s simple, s ≤ f }
For multivector-valued functions F: X → Cl(p,q,r):
∫ F dμ = ∫ (scalar part) dμ + ∫ (vector part) dμ + ∫ (bivector part) dμ + ...
Radon-Nikodym Theorem
If ν ≪ μ (ν absolutely continuous with respect to μ), then:
dν/dμ exists and ν(A) = ∫_A (dν/dμ) dμ
For geometric measures, the derivative dν/dμ is a multivector-valued density.
Crate Structure
amari-measure/
├── Cargo.toml
├── src/
│   ├── lib.rs                      # Public API, module docs
│   ├── error.rs                    # Error types
│   ├── sigma_algebra.rs            # σ-algebras and measurable sets
│   ├── measure.rs                  # Measure trait and implementations
│   ├── geometric_measure.rs        # Multivector-valued measures
│   ├── integration.rs              # Lebesgue integration
│   ├── density.rs                  # Radon-Nikodym derivatives
│   ├── pushforward.rs              # Pushforward and pullback
│   ├── product.rs                  # Product measures
│   ├── convergence.rs              # Convergence theorems
│   ├── signed_measure.rs           # Signed and complex measures
│   └── phantom.rs                  # Phantom types for measure properties
├── tests/
│   ├── measure_tests.rs
│   ├── integration_tests.rs
│   ├── density_tests.rs
│   ├── convergence_tests.rs
│   └── property_tests.rs
├── benches/
│   └── integration_benchmarks.rs
└── examples/
    ├── lebesgue_integration.rs
    ├── probability_measure.rs
    ├── geometric_density.rs
    └── fubini_theorem.rs
Core Type Definitions
Phantom Types for Measure Properties
rust// src/phantom.rs
use std::marker::PhantomData;

/// Measure is finite (μ(X) < ∞)
pub struct Finite;

/// Measure is σ-finite (X = ⋃ᵢ Aᵢ with μ(Aᵢ) < ∞)
pub struct SigmaFinite;

/// Measure is infinite
pub struct Infinite;

/// Measure is a probability measure (μ(X) = 1)
pub struct Probability;

/// Measure is signed (can be negative)
pub struct Signed;

/// Measure is unsigned (always non-negative)
pub struct Unsigned;

/// Measure is complete (all subsets of null sets are measurable)
pub struct Complete;

/// Measure is incomplete
pub struct Incomplete;

/// Measure type with compile-time properties
pub struct MeasureType
    Finiteness = SigmaFinite,
    Signedness = Unsigned,
    Completeness = Incomplete,
> {
    _finiteness: PhantomData<Finiteness>,
    _signedness: PhantomData<Signedness>,
    _completeness: PhantomData<Completeness>,
}
Error Types
rust// src/error.rs
use thiserror::Error;

#[derive(Error, Debug)]
pub enum MeasureError {
    #[error("Set is not measurable: {0}")]
    NotMeasurable(String),
    
    #[error("Measure is not finite: {0}")]
    NotFinite(String),
    
    #[error("Measure is not σ-finite")]
    NotSigmaFinite,
    
    #[error("Function is not integrable: {0}")]
    NotIntegrable(String),
    
    #[error("Measure {target} is not absolutely continuous with respect to {reference}")]
    NotAbsolutelyContinuous { target: String, reference: String },
    
    #[error("Integral does not converge: {0}")]
    NonConvergent(String),
    
    #[error("Dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },
    
    #[error("Invalid measure: {0}")]
    InvalidMeasure(String),
    
    #[error("Integration with Amari core failed: {0}")]
    AmariIntegrationError(#[from] amari_core::AmariError),
}

pub type Result<T> = std::result::Result<T, MeasureError>;
Implementation Requirements
1. σ-Algebras (src/sigma_algebra.rs)
rustuse std::collections::HashSet;
use std::hash::Hash;
use creusot_contracts::*;

/// A σ-algebra of measurable sets
///
/// A collection Σ of subsets of X satisfying:
/// 1. X ∈ Σ
/// 2. A ∈ Σ ⟹ Aᶜ ∈ Σ
/// 3. {Aᵢ} ⊆ Σ ⟹ ⋃ᵢ Aᵢ ∈ Σ
pub trait SigmaAlgebra<T: Eq + Hash> {
    /// Check if a set is measurable
    fn is_measurable(&self, set: &HashSet<T>) -> bool;
    
    /// The entire space X
    fn whole_space(&self) -> HashSet<T>;
    
    /// Complement of a measurable set
    fn complement(&self, set: &HashSet<T>) -> Result<HashSet<T>>;
    
    /// Countable union of measurable sets
    fn countable_union(&self, sets: &[HashSet<T>]) -> Result<HashSet<T>>;
    
    /// Countable intersection of measurable sets
    fn countable_intersection(&self, sets: &[HashSet<T>]) -> Result<HashSet<T>>;
}

/// Borel σ-algebra on ℝⁿ
#[derive(Clone, Debug)]
pub struct BorelSigmaAlgebra {
    /// Dimension of the space
    dimension: usize,
    
    /// Open sets that generate the σ-algebra
    generating_sets: Vec<BorelSet>,
}

/// A Borel set in ℝⁿ
#[derive(Clone, Debug)]
pub enum BorelSet {
    /// Empty set
    Empty,
    
    /// Entire space
    WholeSpace { dimension: usize },
    
    /// Open interval (a, b)
    OpenInterval { dimension: usize, lower: Vec<f64>, upper: Vec<f64> },
    
    /// Closed interval [a, b]
    ClosedInterval { dimension: usize, lower: Vec<f64>, upper: Vec<f64> },
    
    /// Half-open interval [a, b)
    HalfOpenInterval { dimension: usize, lower: Vec<f64>, upper: Vec<f64> },
    
    /// Complement of a Borel set
    Complement(Box<BorelSet>),
    
    /// Countable union
    CountableUnion(Vec<BorelSet>),
    
    /// Countable intersection
    CountableIntersection(Vec<BorelSet>),
}

impl BorelSigmaAlgebra {
    /// Create Borel σ-algebra on ℝⁿ
    pub fn new(dimension: usize) -> Self {
        Self {
            dimension,
            generating_sets: Vec::new(),
        }
    }
    
    /// Check if a point is in a Borel set
    pub fn contains_point(&self, set: &BorelSet, point: &[f64]) -> bool {
        match set {
            BorelSet::Empty => false,
            BorelSet::WholeSpace { .. } => true,
            BorelSet::OpenInterval { lower, upper, .. } => {
                point.iter().zip(lower.iter().zip(upper.iter()))
                    .all(|(p, (l, u))| p > l && p < u)
            },
            BorelSet::ClosedInterval { lower, upper, .. } => {
                point.iter().zip(lower.iter().zip(upper.iter()))
                    .all(|(p, (l, u))| p >= l && p <= u)
            },
            BorelSet::HalfOpenInterval { lower, upper, .. } => {
                point.iter().zip(lower.iter().zip(upper.iter()))
                    .all(|(p, (l, u))| p >= l && p < u)
            },
            BorelSet::Complement(inner) => !self.contains_point(inner, point),
            BorelSet::CountableUnion(sets) => {
                sets.iter().any(|s| self.contains_point(s, point))
            },
            BorelSet::CountableIntersection(sets) => {
                sets.iter().all(|s| self.contains_point(s, point))
            },
        }
    }
}

/// Product σ-algebra: Σ₁ ⊗ Σ₂
pub struct ProductSigmaAlgebra<T1: Eq + Hash, T2: Eq + Hash> {
    first: Box<dyn SigmaAlgebra<T1>>,
    second: Box<dyn SigmaAlgebra<T2>>,
}

impl<T1: Eq + Hash, T2: Eq + Hash> ProductSigmaAlgebra<T1, T2> {
    pub fn new(
        first: Box<dyn SigmaAlgebra<T1>>,
        second: Box<dyn SigmaAlgebra<T2>>,
    ) -> Self {
        Self { first, second }
    }
}
2. Measure Trait (src/measure.rs)
rustuse super::sigma_algebra::*;
use std::collections::HashSet;
use std::hash::Hash;
use creusot_contracts::*;

/// A measure on a σ-algebra
///
/// μ: Σ → [0, ∞] satisfying:
/// 1. μ(∅) = 0
/// 2. μ(⋃ᵢ Aᵢ) = Σᵢ μ(Aᵢ) for disjoint {Aᵢ}
pub trait Measure<T: Eq + Hash> {
    /// The σ-algebra this measure is defined on
    fn sigma_algebra(&self) -> &dyn SigmaAlgebra<T>;
    
    /// Measure of a set
    ///
    /// # Properties
    ///
    /// - μ(∅) = 0
    /// - μ(A) ≥ 0 for all A
    /// - μ(⋃ᵢ Aᵢ) = Σᵢ μ(Aᵢ) for disjoint Aᵢ
    #[requires(self.sigma_algebra().is_measurable(set))]
    #[ensures(result.is_ok() ==> result.unwrap() >= 0.0)]
    fn measure(&self, set: &HashSet<T>) -> Result<f64>;
    
    /// Measure of empty set is zero
    #[ensures(result == 0.0)]
    fn measure_empty(&self) -> f64 {
        0.0
    }
    
    /// Monotonicity: A ⊆ B ⟹ μ(A) ≤ μ(B)
    fn is_monotone(&self, subset: &HashSet<T>, superset: &HashSet<T>) -> bool {
        if !subset.is_subset(superset) {
            return true; // Vacuously true
        }
        match (self.measure(subset), self.measure(superset)) {
            (Ok(mu_a), Ok(mu_b)) => mu_a <= mu_b,
            _ => false,
        }
    }
    
    /// Check if measure is finite
    fn is_finite(&self) -> bool;
    
    /// Check if measure is σ-finite
    fn is_sigma_finite(&self) -> bool;
    
    /// Check if measure is probability measure (μ(X) = 1)
    fn is_probability_measure(&self) -> bool;
}

/// Lebesgue measure on ℝⁿ
#[derive(Clone, Debug)]
pub struct LebesgueMeasure {
    /// Dimension of the space
    dimension: usize,
    
    /// Borel σ-algebra
    sigma_algebra: BorelSigmaAlgebra,
}

impl LebesgueMeasure {
    /// Create Lebesgue measure on ℝⁿ
    pub fn new(dimension: usize) -> Self {
        Self {
            dimension,
            sigma_algebra: BorelSigmaAlgebra::new(dimension),
        }
    }
    
    /// Measure of a Borel set
    ///
    /// For intervals [a, b], this is the volume: ∏ᵢ (bᵢ - aᵢ)
    pub fn measure_borel_set(&self, set: &BorelSet) -> Result<f64> {
        match set {
            BorelSet::Empty => Ok(0.0),
            BorelSet::WholeSpace { .. } => Ok(f64::INFINITY),
            BorelSet::OpenInterval { lower, upper, .. } |
            BorelSet::ClosedInterval { lower, upper, .. } |
            BorelSet::HalfOpenInterval { lower, upper, .. } => {
                if lower.len() != self.dimension || upper.len() != self.dimension {
                    return Err(MeasureError::DimensionMismatch {
                        expected: self.dimension,
                        got: lower.len(),
                    });
                }
                
                let volume = lower.iter().zip(upper.iter())
                    .map(|(l, u)| u - l)
                    .product();
                
                Ok(volume)
            },
            BorelSet::Complement(inner) => {
                let inner_measure = self.measure_borel_set(inner)?;
                if inner_measure.is_infinite() {
                    Ok(0.0) // Measure of complement of infinite set can be 0
                } else {
                    Ok(f64::INFINITY) // Complement typically has infinite measure
                }
            },
            BorelSet::CountableUnion(sets) => {
                // Check if sets are disjoint (simplified - should be more rigorous)
                let total: f64 = sets.iter()
                    .map(|s| self.measure_borel_set(s).unwrap_or(0.0))
                    .sum();
                Ok(total)
            },
            BorelSet::CountableIntersection(sets) => {
                // Use inclusion-exclusion or direct computation
                // Simplified: take minimum
                sets.iter()
                    .map(|s| self.measure_borel_set(s))
                    .try_fold(f64::INFINITY, |acc, r| r.map(|m| acc.min(m)))
            },
        }
    }
}

/// Counting measure: μ(A) = |A| (cardinality)
#[derive(Clone, Debug)]
pub struct CountingMeasure<T: Eq + Hash> {
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Eq + Hash> CountingMeasure<T> {
    pub fn new() -> Self {
        Self {
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<T: Eq + Hash> Measure<T> for CountingMeasure<T> {
    fn sigma_algebra(&self) -> &dyn SigmaAlgebra<T> {
        unimplemented!("Counting measure uses power set σ-algebra")
    }
    
    fn measure(&self, set: &HashSet<T>) -> Result<f64> {
        Ok(set.len() as f64)
    }
    
    fn is_finite(&self) -> bool {
        false
    }
    
    fn is_sigma_finite(&self) -> bool {
        true
    }
    
    fn is_probability_measure(&self) -> bool {
        false
    }
}

/// Dirac measure: δₓ(A) = 1 if x ∈ A, 0 otherwise
#[derive(Clone, Debug)]
pub struct DiracMeasure<T: Eq + Hash + Clone> {
    point: T,
}

impl<T: Eq + Hash + Clone> DiracMeasure<T> {
    pub fn new(point: T) -> Self {
        Self { point }
    }
}

impl<T: Eq + Hash + Clone> Measure<T> for DiracMeasure<T> {
    fn sigma_algebra(&self) -> &dyn SigmaAlgebra<T> {
        unimplemented!("Dirac measure uses power set σ-algebra")
    }
    
    fn measure(&self, set: &HashSet<T>) -> Result<f64> {
        Ok(if set.contains(&self.point) { 1.0 } else { 0.0 })
    }
    
    fn is_finite(&self) -> bool {
        true
    }
    
    fn is_sigma_finite(&self) -> bool {
        true
    }
    
    fn is_probability_measure(&self) -> bool {
        true
    }
}
3. Geometric Measures (src/geometric_measure.rs)
rustuse amari_core::Multivector;
use super::measure::*;
use super::sigma_algebra::*;
use creusot_contracts::*;

/// A multivector-valued measure
///
/// μ: Σ → Cl(p,q,r)
///
/// This generalizes real-valued measures to geometric algebra.
/// Each grade component is a separate measure.
#[derive(Clone, Debug)]
pub struct GeometricMeasure<const P: usize, const Q: usize, const R: usize> {
    /// Measure for each grade
    grade_measures: Vec<Box<dyn Measure<Vec<f64>>>>,
    
    /// σ-algebra
    sigma_algebra: BorelSigmaAlgebra,
    
    /// Dimension of base space
    dimension: usize,
}

impl<const P: usize, const Q: usize, const R: usize> GeometricMeasure<P, Q, R> {
    /// Create geometric measure from grade components
    pub fn new(
        grade_measures: Vec<Box<dyn Measure<Vec<f64>>>>,
        dimension: usize,
    ) -> Result<Self> {
        let expected_grades = 2_usize.pow((P + Q + R) as u32);
        
        if grade_measures.len() != expected_grades {
            return Err(MeasureError::DimensionMismatch {
                expected: expected_grades,
                got: grade_measures.len(),
            });
        }
        
        Ok(Self {
            grade_measures,
            sigma_algebra: BorelSigmaAlgebra::new(dimension),
            dimension,
        })
    }
    
    /// Measure of a Borel set (returns multivector)
    ///
    /// μ(A) = μ₀(A) + μ₁(A)e₁ + μ₂(A)e₂ + ... + μ₁₂(A)e₁e₂ + ...
    pub fn measure_set(&self, set: &BorelSet) -> Result<Multivector<P, Q, R>> {
        let mut components = Vec::new();
        
        // Convert BorelSet to HashSet (simplified)
        // In practice, would need proper measurable set representation
        let measurable_set = self.borel_to_measurable(set)?;
        
        for grade_measure in &self.grade_measures {
            let component_measure = grade_measure.measure(&measurable_set)?;
            components.push(component_measure);
        }
        
        Multivector::from_components(&components)
            .map_err(|e| MeasureError::AmariIntegrationError(e))
    }
    
    /// Convert Borel set to measurable set (helper)
    fn borel_to_measurable(&self, _set: &BorelSet) -> Result<HashSet<Vec<f64>>> {
        // Simplified: would need proper discretization or symbolic representation
        Ok(HashSet::new())
    }
    
    /// Create geometric Lebesgue measure
    ///
    /// Each grade component uses standard Lebesgue measure
    pub fn lebesgue(dimension: usize) -> Result<Self> {
        let expected_grades = 2_usize.pow(dimension as u32);
        let mut grade_measures: Vec<Box<dyn Measure<Vec<f64>>>> = Vec::new();
        
        for _ in 0..expected_grades {
            // Each grade gets Lebesgue measure (simplified)
            // In practice, would need proper implementation
            grade_measures.push(Box::new(CountingMeasure::new()));
        }
        
        Self::new(grade_measures, dimension)
    }
    
    /// Create probability measure over multivectors
    ///
    /// Total measure is 1: ∫ dμ = 1 (as multivector)
    pub fn probability(
        density: impl Fn(&[f64]) -> Multivector<P, Q, R> + 'static,
        dimension: usize,
    ) -> Result<Self> {
        // Would integrate density to verify normalization
        // Then create measure with this density
        unimplemented!("Probability measure construction")
    }
}

/// Geometric density function
///
/// A multivector-valued function ρ: ℝⁿ → Cl(p,q,r) such that:
/// μ(A) = ∫_A ρ(x) dλ(x)
///
/// where λ is a reference measure (usually Lebesgue)
#[derive(Clone)]
pub struct GeometricDensity<const P: usize, const Q: usize, const R: usize> {
    /// Density function
    density: Box<dyn Fn(&[f64]) -> Multivector<P, Q, R>>,
    
    /// Reference measure
    reference_measure: LebesgueMeasure,
}

impl<const P: usize, const Q: usize, const R: usize> GeometricDensity<P, Q, R> {
    /// Create geometric density
    pub fn new(
        density: impl Fn(&[f64]) -> Multivector<P, Q, R> + 'static,
        dimension: usize,
    ) -> Self {
        Self {
            density: Box::new(density),
            reference_measure: LebesgueMeasure::new(dimension),
        }
    }
    
    /// Evaluate density at a point
    pub fn evaluate(&self, point: &[f64]) -> Multivector<P, Q, R> {
        (self.density)(point)
    }
    
    /// Check if density is normalized (for probability densities)
    ///
    /// ∫ ρ(x) dλ(x) should equal 1 (as multivector)
    pub fn is_normalized(&self, domain: &BorelSet) -> Result<bool> {
        // Would perform numerical integration
        unimplemented!("Normalization check")
    }
}
4. Lebesgue Integration (src/integration.rs)
rustuse amari_core::Multivector;
use super::measure::*;
use super::geometric_measure::*;
use super::sigma_algebra::*;
use creusot_contracts::*;

/// Lebesgue integrator for real-valued functions
pub struct LebesgueIntegrator {
    /// Tolerance for numerical integration
    tolerance: f64,
    
    /// Maximum subdivisions
    max_subdivisions: usize,
}

impl LebesgueIntegrator {
    pub fn new(tolerance: f64, max_subdivisions: usize) -> Self {
        Self {
            tolerance,
            max_subdivisions,
        }
    }
    
    /// Integrate f with respect to measure μ over set A
    ///
    /// ∫_A f dμ
    ///
    /// # Properties
    ///
    /// - Linearity: ∫(af + bg)dμ = a∫f dμ + b∫g dμ
    /// - Monotonicity: f ≤ g ⟹ ∫f dμ ≤ ∫g dμ
    /// - Additivity: ∫_{A∪B} f dμ = ∫_A f dμ + ∫_B f dμ (A,B disjoint)
    #[requires(tolerance > 0.0)]
    pub fn integrate<F>(
        &self,
        function: F,
        measure: &LebesgueMeasure,
        domain: &BorelSet,
    ) -> Result<f64>
    where
        F: Fn(&[f64]) -> f64,
    {
        // Use adaptive quadrature for numerical integration
        self.adaptive_integrate(&function, measure, domain, 0)
    }
    
    /// Adaptive integration using subdivision
    fn adaptive_integrate<F>(
        &self,
        function: &F,
        measure: &LebesgueMeasure,
        domain: &BorelSet,
        depth: usize,
    ) -> Result<f64>
    where
        F: Fn(&[f64]) -> f64,
    {
        if depth > self.max_subdivisions {
            return Err(MeasureError::NonConvergent(
                format!("Integration did not converge after {} subdivisions", depth)
            ));
        }
        
        match domain {
            BorelSet::OpenInterval { lower, upper, .. } |
            BorelSet::ClosedInterval { lower, upper, .. } |
            BorelSet::HalfOpenInterval { lower, upper, .. } => {
                // Simpson's rule or similar quadrature
                self.simpson_rule(function, lower, upper)
            },
            BorelSet::CountableUnion(sets) => {
                // Integrate over each set and sum
                let mut total = 0.0;
                for set in sets {
                    total += self.adaptive_integrate(function, measure, set, depth + 1)?;
                }
                Ok(total)
            },
            _ => Err(MeasureError::NotIntegrable(
                "Unsupported domain type".to_string()
            )),
        }
    }
    
    /// Simpson's rule for integration
    fn simpson_rule<F>(&self, function: &F, lower: &[f64], upper: &[f64]) -> Result<f64>
    where
        F: Fn(&[f64]) -> f64,
    {
        if lower.len() != upper.len() {
            return Err(MeasureError::DimensionMismatch {
                expected: lower.len(),
                got: upper.len(),
            });
        }
        
        let dim = lower.len();
        
        // For multidimensional integration, use product of 1D Simpson's rules
        // Simplified implementation
        let n_points = 100; // Number of sample points
        let mut sum = 0.0;
        let volume = lower.iter().zip(upper.iter())
            .map(|(l, u)| u - l)
            .product::<f64>();
        
        // Monte Carlo sampling (simplified - should use proper quadrature)
        use rand::Rng;
        let mut rng = rand::thread_rng();
        
        for _ in 0..n_points {
            let point: Vec<f64> = lower.iter().zip(upper.iter())
                .map(|(l, u)| l + rng.gen::<f64>() * (u - l))
                .collect();
            
            sum += function(&point);
        }
        
        Ok(sum * volume / n_points as f64)
    }
    
    /// Monotone convergence theorem
    ///
    /// If {fₙ} is increasing and fₙ → f, then:
    /// ∫ f dμ = lim_{n→∞} ∫ fₙ dμ
    pub fn monotone_convergence<F>(
        &self,
        sequence: &[F],
        measure: &LebesgueMeasure,
        domain: &BorelSet,
    ) -> Result<f64>
    where
        F: Fn(&[f64]) -> f64,
    {
        if sequence.is_empty() {
            return Ok(0.0);
        }
        
        let mut integrals = Vec::new();
        
        for f in sequence {
            let integral = self.integrate(f, measure, domain)?;
            integrals.push(integral);
        }
        
        // Check monotonicity (simplified)
        for i in 1..integrals.len() {
            if integrals[i] < integrals[i-1] - self.tolerance {
                return Err(MeasureError::InvalidMeasure(
                    "Sequence is not monotone increasing".to_string()
                ));
            }
        }
        
        // Return limit (last value)
        Ok(*integrals.last().unwrap())
    }
    
    /// Dominated convergence theorem
    ///
    /// If fₙ → f pointwise and |fₙ| ≤ g where ∫g dμ < ∞, then:
    /// ∫ f dμ = lim_{n→∞} ∫ fₙ dμ
    pub fn dominated_convergence<F, G>(
        &self,
        sequence: &[F],
        dominator: G,
        measure: &LebesgueMeasure,
        domain: &BorelSet,
    ) -> Result<f64>
    where
        F: Fn(&[f64]) -> f64,
        G: Fn(&[f64]) -> f64,
    {
        // Check dominator is integrable
        let dominator_integral = self.integrate(&dominator, measure, domain)?;
        if dominator_integral.is_infinite() {
            return Err(MeasureError::NotIntegrable(
                "Dominator is not integrable".to_string()
            ));
        }
        
        // Compute limit of integrals
        let mut integrals = Vec::new();
        for f in sequence {
            let integral = self.integrate(f, measure, domain)?;
            integrals.push(integral);
        }
        
        // Check convergence
        if integrals.len() < 2 {
            return Ok(integrals.first().copied().unwrap_or(0.0));
        }
        
        let last = integrals.last().unwrap();
        let second_last = integrals[integrals.len() - 2];
        
        if (last - second_last).abs() > self.tolerance {
            return Err(MeasureError::NonConvergent(
                "Sequence did not converge".to_string()
            ));
        }
        
        Ok(*last)
    }
}

/// Geometric integrator for multivector-valued functions
pub struct GeometricIntegrator<const P: usize, const Q: usize, const R: usize> {
    /// Base integrator for real-valued functions
    base_integrator: LebesgueIntegrator,
}

impl<const P: usize, const Q: usize, const R: usize> GeometricIntegrator<P, Q, R> {
    pub fn new(tolerance: f64, max_subdivisions: usize) -> Self {
        Self {
            base_integrator: LebesgueIntegrator::new(tolerance, max_subdivisions),
        }
    }
    
    /// Integrate multivector-valued function
    ///
    /// ∫ F dμ where F: ℝⁿ → Cl(p,q,r)
    ///
    /// This integrates each component separately:
    /// ∫ F dμ = (∫ F₀ dμ) + (∫ F₁ dμ)e₁ + (∫ F₂ dμ)e₂ + ...
    pub fn integrate<F>(
        &self,
        function: F,
        measure: &GeometricMeasure<P, Q, R>,
        domain: &BorelSet,
    ) -> Result<Multivector<P, Q, R>>
    where
        F: Fn(&[f64]) -> Multivector<P, Q, R>,
    {
        let reference_measure = LebesgueMeasure::new(measure.dimension);
        let expected_components = 2_usize.pow((P + Q + R) as u32);
        let mut integrated_components = Vec::with_capacity(expected_components);
        
        // Integrate each component
        for i in 0..expected_components {
            let component_function = |x: &[f64]| {
                function(x).component(i)
            };
            
            let component_integral = self.base_integrator.integrate(
                component_function,
                &reference_measure,
                domain,
            )?;
            
            integrated_components.push(component_integral);
        }
        
        Multivector::from_components(&integrated_components)
            .map_err(|e| MeasureError::AmariIntegrationError(e))
    }
    
    /// Integrate with respect to geometric density
    ///
    /// ∫ F·ρ dλ where ρ is geometric density and λ is Lebesgue measure
    pub fn integrate_with_density<F>(
        &self,
        function: F,
        density: &GeometricDensity<P, Q, R>,
        domain: &BorelSet,
    ) -> Result<Multivector<P, Q, R>>
    where
        F: Fn(&[f64]) -> Multivector<P, Q, R>,
    {
        // Integrate F(x)·ρ(x)
        let combined_function = |x: &[f64]| {
            function(x) * density.evaluate(x)
        };
        
        let lebesgue = LebesgueMeasure::new(density.reference_measure.dimension);
        let geometric_measure = GeometricMeasure::lebesgue(lebesgue.dimension)?;
        
        self.integrate(combined_function, &geometric_measure, domain)
    }
}
5. Radon-Nikodym Derivatives (src/density.rs)
rustuse amari_core::Multivector;
use super::measure::*;
use super::geometric_measure::*;
use creusot_contracts::*;

/// Radon-Nikodym derivative computer
///
/// Computes dν/dμ when ν ≪ μ (ν absolutely continuous w.r.t. μ)
pub struct RadonNikodymComputer {
    /// Tolerance for convergence
    tolerance: f64,
}

impl RadonNikodymComputer {
    pub fn new(tolerance: f64) -> Self {
        Self { tolerance }
    }
    
    /// Check if ν is absolutely continuous with respect to μ
    ///
    /// ν ≪ μ ⟺ ∀A: μ(A) = 0 ⟹ ν(A) = 0
    pub fn is_absolutely_continuous<T: Eq + Hash>(
        &self,
        nu: &dyn Measure<T>,
        mu: &dyn Measure<T>,
        test_sets: &[HashSet<T>],
    ) -> bool {
        for set in test_sets {
            if let (Ok(mu_measure), Ok(nu_measure)) = (mu.measure(set), nu.measure(set)) {
                if mu_measure.abs() < self.tolerance && nu_measure.abs() > self.tolerance {
                    return false;
                }
            }
        }
        true
    }
    
    /// Compute Radon-Nikodym derivative dν/dμ
    ///
    /// Returns density ρ such that ν(A) = ∫_A ρ dμ
    ///
    /// # Theorem (Radon-Nikodym)
    ///
    /// If μ, ν are σ-finite measures and ν ≪ μ, then there exists
    /// a measurable function f ≥ 0 such that:
    ///
    /// ν(A) = ∫_A f dμ for all measurable A
    ///
    /// Moreover, f is unique μ-almost everywhere.
    pub fn compute_derivative(
        &self,
        nu: &LebesgueMeasure,
        mu: &LebesgueMeasure,
        point: &[f64],
    ) -> Result<f64> {
        // Radon-Nikodym derivative at a point (when it exists)
        // dν/dμ(x) = lim_{r→0} ν(B_r(x)) / μ(B_r(x))
        
        if !nu.is_sigma_finite() || !mu.is_sigma_finite() {
            return Err(MeasureError::NotSigmaFinite);
        }
        
        // Compute limit using shrinking balls
        let radii = vec![0.1, 0.01, 0.001, 0.0001];
        let mut ratios = Vec::new();
        
        for radius in radii {
            let ball = self.ball_around(point, radius);
            
            let nu_measure = nu.measure_borel_set(&ball)?;
            let mu_measure = mu.measure_borel_set(&ball)?;
            
            if mu_measure.abs() < self.tolerance {
                return Err(MeasureError::NotAbsolutelyContinuous {
                    target: "nu".to_string(),
                    reference: "mu".to_string(),
                });
            }
            
            ratios.push(nu_measure / mu_measure);
        }
        
        // Check convergence
        if ratios.len() >= 2 {
            let last_diff = (ratios[ratios.len() - 1] - ratios[ratios.len() - 2]).abs();
            if last_diff > self.tolerance {
                return Err(MeasureError::NonConvergent(
                    "Radon-Nikodym derivative did not converge".to_string()
                ));
            }
        }
        
        Ok(*ratios.last().unwrap())
    }
    
    /// Create ball around point (helper)
    fn ball_around(&self, center: &[f64], radius: f64) -> BorelSet {
        let lower: Vec<f64> = center.iter().map(|x| x - radius).collect();
        let upper: Vec<f64> = center.iter().map(|x| x + radius).collect();
        
        BorelSet::OpenInterval {
            dimension: center.len(),
            lower,
            upper,
        }
    }
    
    /// Compute geometric Radon-Nikodym derivative
    ///
    /// For geometric measures ν, μ: Σ → Cl(p,q,r)
    /// Computes dν/dμ: X → Cl(p,q,r)
    pub fn compute_geometric_derivative<const P: usize, const Q: usize, const R: usize>(
        &self,
        nu: &GeometricMeasure<P, Q, R>,
        mu: &GeometricMeasure<P, Q, R>,
        point: &[f64],
    ) -> Result<Multivector<P, Q, R>> {
        let expected_components = 2_usize.pow((P + Q + R) as u32);
        let mut derivative_components = Vec::with_capacity(expected_components);
        
        // Compute derivative for each grade component
        for i in 0..expected_components {
            // Extract i-th grade measure from nu and mu
            // Compute ordinary Radon-Nikodym derivative
            // This is simplified - would need proper grade extraction
            
            let derivative = 1.0; // Placeholder
            derivative_components.push(derivative);
        }
        
        Multivector::from_components(&derivative_components)
            .map_err(|e| MeasureError::AmariIntegrationError(e))
    }
}

/// Lebesgue decomposition
///
/// Every measure ν can be uniquely decomposed as:
/// ν = ν_ac + ν_s
///
/// where ν_ac ≪ μ (absolutely continuous) and ν_s ⊥ μ (singular)
pub struct LebesgueDecomposition {
    /// Absolutely continuous part
    pub absolutely_continuous: Box<dyn Measure<Vec<f64>>>,
    
    /// Singular part
    pub singular: Box<dyn Measure<Vec<f64>>>,
}

impl LebesgueDecomposition {
    /// Compute Lebesgue decomposition
    pub fn decompose(
        nu: &LebesgueMeasure,
        mu: &LebesgueMeasure,
    ) -> Result<Self> {
        // Hahn decomposition followed by absolute value
        // Simplified implementation
        unimplemented!("Lebesgue decomposition")
    }
}
6. Pushforward and Pullback (src/pushforward.rs)
rustuse super::measure::*;
use super::geometric_measure::*;
use amari_core::Multivector;
use creusot_contracts::*;

/// Pushforward of a measure under a map
///
/// Given μ: Σ_X → [0,∞] and f: X → Y,
/// the pushforward f₊μ: Σ_Y → [0,∞] is defined by:
///
/// (f₊μ)(B) = μ(f⁻¹(B))
pub struct Pushforward {
    /// The map f: X → Y
    map: Box<dyn Fn(&[f64]) -> Vec<f64>>,
    
    /// Original measure μ on X
    source_measure: LebesgueMeasure,
}

impl Pushforward {
    pub fn new(
        map: impl Fn(&[f64]) -> Vec<f64> + 'static,
        source_measure: LebesgueMeasure,
    ) -> Self {
        Self {
            map: Box::new(map),
            source_measure,
        }
    }
    
    /// Compute pushforward measure of a set
    ///
    /// (f₊μ)(B) = μ(f⁻¹(B))
    pub fn pushforward_measure(&self, target_set: &BorelSet) -> Result<f64> {
        // Compute preimage f⁻¹(B)
        let preimage = self.compute_preimage(target_set)?;
        
        // Measure the preimage
        self.source_measure.measure_borel_set(&preimage)
    }
    
    /// Compute preimage f⁻¹(B)
    fn compute_preimage(&self, target_set: &BorelSet) -> Result<BorelSet> {
        // This is non-trivial in general
        // For simple cases (linear maps), can compute explicitly
        // For general maps, need numerical approximation
        unimplemented!("Preimage computation")
    }
    
    /// Compute pushforward density using change of variables
    ///
    /// If μ has density ρ, then f₊μ has density:
    /// (f₊ρ)(y) = ρ(f⁻¹(y)) · |det(Df⁻¹(y))|
    ///
    /// where Df⁻¹ is the Jacobian of the inverse
    pub fn pushforward_density(&self, point: &[f64]) -> Result<f64> {
        // Compute inverse map and Jacobian
        // This requires the map to be invertible
        unimplemented!("Pushforward density via change of variables")
    }
}

/// Pullback of a measure under a map
///
/// Given ν: Σ_Y → [0,∞] and f: X → Y,
/// the pullback f*ν: Σ_X → [0,∞] is defined by:
///
/// (f*ν)(A) = ν(f(A))
pub struct Pullback {
    /// The map f: X → Y
    map: Box<dyn Fn(&[f64]) -> Vec<f64>>,
    
    /// Target measure ν on Y
    target_measure: LebesgueMeasure,
}

impl Pullback {
    pub fn new(
        map: impl Fn(&[f64]) -> Vec<f64> + 'static,
        target_measure: LebesgueMeasure,
    ) -> Self {
        Self {
            map: Box::new(map),
            target_measure,
        }
    }
    
    /// Compute pullback measure of a set
    ///
    /// (f*ν)(A) = ν(f(A))
    pub fn pullback_measure(&self, source_set: &BorelSet) -> Result<f64> {
        // Compute image f(A)
        let image = self.compute_image(source_set)?;
        
        // Measure the image
        self.target_measure.measure_borel_set(&image)
    }
    
    /// Compute image f(A)
    fn compute_image(&self, source_set: &BorelSet) -> Result<BorelSet> {
        // This requires sampling or symbolic computation
        unimplemented!("Image computation")
    }
}

/// Geometric pushforward for multivector-valued measures
pub struct GeometricPushforward<const P: usize, const Q: usize, const R: usize> {
    /// Geometric map (multivector → multivector)
    map: Box<dyn Fn(&Multivector<P, Q, R>) -> Multivector<P, Q, R>>,
    
    /// Source geometric measure
    source_measure: GeometricMeasure<P, Q, R>,
}

impl<const P: usize, const Q: usize, const R: usize> GeometricPushforward<P, Q, R> {
    pub fn new(
        map: impl Fn(&Multivector<P, Q, R>) -> Multivector<P, Q, R> + 'static,
        source_measure: GeometricMeasure<P, Q, R>,
    ) -> Self {
        Self {
            map: Box::new(map),
            source_measure,
        }
    }
    
    /// Pushforward geometric measure
    pub fn pushforward(&self, target_set: &BorelSet) -> Result<Multivector<P, Q, R>> {
        // Similar to scalar case but returns multivector
        unimplemented!("Geometric pushforward")
    }
}
7. Product Measures (src/product.rs)
rustuse super::measure::*;
use super::sigma_algebra::*;
use super::integration::*;
use creusot_contracts::*;

/// Product measure μ ⊗ ν on X × Y
///
/// (μ ⊗ ν)(A × B) = μ(A) · ν(B)
pub struct ProductMeasure {
    /// First measure μ on X
    first: LebesgueMeasure,
    
    /// Second measure ν on Y
    second: LebesgueMeasure,
    
    /// Dimension of product space
    product_dimension: usize,
}

impl ProductMeasure {
    /// Create product measure
    pub fn new(first: LebesgueMeasure, second: LebesgueMeasure) -> Self {
        let product_dimension = first.dimension + second.dimension;
        Self {
            first,
            second,
            product_dimension,
        }
    }
    
    /// Measure of product set A × B
    ///
    /// (μ ⊗ ν)(A × B) = μ(A) · ν(B)
    pub fn measure_product(
        &self,
        first_set: &BorelSet,
        second_set: &BorelSet,
    ) -> Result<f64> {
        let mu_a = self.first.measure_borel_set(first_set)?;
        let nu_b = self.second.measure_borel_set(second_set)?;
        
        Ok(mu_a * nu_b)
    }
    
    /// Fubini's theorem: swap order of integration
    ///
    /// ∫∫ f(x,y) d(μ⊗ν)(x,y) = ∫[∫ f(x,y) dν(y)] dμ(x)
    ///                        = ∫[∫ f(x,y) dμ(x)] dν(y)
    ///
    /// # Theorem (Fubini-Tonelli)
    ///
    /// If f ≥ 0 or ∫|f| d(μ⊗ν) < ∞, then the iterated integrals
    /// exist and are equal to the double integral.
    pub fn fubini_swap<F>(
        &self,
        function: F,
        first_domain: &BorelSet,
        second_domain: &BorelSet,
    ) -> Result<(f64, f64)>
    where
        F: Fn(&[f64], &[f64]) -> f64,
    {
        let integrator = LebesgueIntegrator::new(1e-6, 1000);
        
        // Order 1: ∫[∫ f(x,y) dν(y)] dμ(x)
        let order1 = {
            let outer_function = |x: &[f64]| {
                let inner_function = |y: &[f64]| function(x, y);
                integrator.integrate(inner_function, &self.second, second_domain)
                    .unwrap_or(0.0)
            };
            integrator.integrate(outer_function, &self.first, first_domain)?
        };
        
        // Order 2: ∫[∫ f(x,y) dμ(x)] dν(y)
        let order2 = {
            let outer_function = |y: &[f64]| {
                let inner_function = |x: &[f64]| function(x, y);
                integrator.integrate(inner_function, &self.first, first_domain)
                    .unwrap_or(0.0)
            };
            integrator.integrate(outer_function, &self.second, second_domain)?
        };
        
        Ok((order1, order2))
    }
    
    /// Verify Fubini's theorem holds
    pub fn verify_fubini<F>(
        &self,
        function: F,
        first_domain: &BorelSet,
        second_domain: &BorelSet,
        tolerance: f64,
    ) -> Result<bool>
    where
        F: Fn(&[f64], &[f64]) -> f64,
    {
        let (order1, order2) = self.fubini_swap(function, first_domain, second_domain)?;
        Ok((order1 - order2).abs() < tolerance)
    }
}

/// Marginal measures from product measure
///
/// Given (μ ⊗ ν) on X × Y, compute:
/// - Marginal on X: μ(A) = (μ⊗ν)(A × Y)
/// - Marginal on Y: ν(B) = (μ⊗ν)(X × B)
pub struct MarginalMeasures {
    product: ProductMeasure,
}

impl MarginalMeasures {
    pub fn new(product: ProductMeasure) -> Self {
        Self { product }
    }
    
    /// First marginal (project onto X)
    pub fn first_marginal(&self, set: &BorelSet) -> Result<f64> {
        // Integrate over second component
        let whole_second = BorelSet::WholeSpace {
            dimension: self.product.second.dimension,
        };
        
        self.product.measure_product(set, &whole_second)
    }
    
    /// Second marginal (project onto Y)
    pub fn second_marginal(&self, set: &BorelSet) -> Result<f64> {
        // Integrate over first component
        let whole_first = BorelSet::WholeSpace {
            dimension: self.product.first.dimension,
        };
        
        self.product.measure_product(&whole_first, set)
    }
}
8. Convergence Theorems (src/convergence.rs)
rustuse super::measure::*;
use super::integration::*;
use super::sigma_algebra::*;
use creusot_contracts::*;

/// Convergence theorem verifier
pub struct ConvergenceVerifier {
    tolerance: f64,
}

impl ConvergenceVerifier {
    pub fn new(tolerance: f64) -> Self {
        Self { tolerance }
    }
    
    /// Verify monotone convergence theorem
    ///
    /// # Theorem (Monotone Convergence / Beppo Levi)
    ///
    /// If 0 ≤ f₁ ≤ f₂ ≤ ... and fₙ → f pointwise, then:
    /// lim_{n→∞} ∫ fₙ dμ = ∫ f dμ
    pub fn verify_monotone_convergence<F>(
        &self,
        sequence: &[F],
        limit: &F,
        measure: &LebesgueMeasure,
        domain: &BorelSet,
    ) -> Result<bool>
    where
        F: Fn(&[f64]) -> f64,
    {
        let integrator = LebesgueIntegrator::new(self.tolerance, 1000);
        
        // Compute ∫ fₙ dμ for each n
        let mut sequence_integrals = Vec::new();
        for f in sequence {
            let integral = integrator.integrate(f, measure, domain)?;
            sequence_integrals.push(integral);
        }
        
        // Verify monotonicity
        for i in 1..sequence_integrals.len() {
            if sequence_integrals[i] < sequence_integrals[i-1] - self.tolerance {
                return Ok(false); // Not monotone
            }
        }
        
        // Compute ∫ f dμ
        let limit_integral = integrator.integrate(limit, measure, domain)?;
        
        // Check if lim ∫ fₙ dμ = ∫ f dμ
        let sequence_limit = *sequence_integrals.last().unwrap();
        Ok((sequence_limit - limit_integral).abs() < self.tolerance)
    }
    
    /// Verify dominated convergence theorem
    ///
    /// # Theorem (Dominated Convergence / Lebesgue)
    ///
    /// If fₙ → f pointwise, |fₙ| ≤ g for all n, and ∫ g dμ < ∞, then:
    /// lim_{n→∞} ∫ fₙ dμ = ∫ f dμ
    pub fn verify_dominated_convergence<F, G>(
        &self,
        sequence: &[F],
        limit: &F,
        dominator: &G,
        measure: &LebesgueMeasure,
        domain: &BorelSet,
    ) -> Result<bool>
    where
        F: Fn(&[f64]) -> f64,
        G: Fn(&[f64]) -> f64,
    {
        let integrator = LebesgueIntegrator::new(self.tolerance, 1000);
        
        // Verify dominator is integrable
        let dominator_integral = integrator.integrate(dominator, measure, domain)?;
        if dominator_integral.is_infinite() {
            return Err(MeasureError::NotIntegrable(
                "Dominator is not integrable".to_string()
            ));
        }
        
        // Compute sequence integrals
        let mut sequence_integrals = Vec::new();
        for f in sequence {
            let integral = integrator.integrate(f, measure, domain)?;
            sequence_integrals.push(integral);
        }
        
        // Compute limit integral
        let limit_integral = integrator.integrate(limit, measure, domain)?;
        
        // Check convergence
        let sequence_limit = *sequence_integrals.last().unwrap();
        Ok((sequence_limit - limit_integral).abs() < self.tolerance)
    }
    
    /// Fatou's lemma
    ///
    /// # Theorem (Fatou)
    ///
    /// If fₙ ≥ 0 for all n, then:
    /// ∫ (lim inf fₙ) dμ ≤ lim inf (∫ fₙ dμ)
    pub fn verify_fatou_lemma<F>(
        &self,
        sequence: &[F],
        measure: &LebesgueMeasure,
        domain: &BorelSet,
    ) -> Result<bool>
    where
        F: Fn(&[f64]) -> f64,
    {
        let integrator = LebesgueIntegrator::new(self.tolerance, 1000);
        
        // Compute ∫ fₙ dμ
        let mut sequence_integrals = Vec::new();
        for f in sequence {
            let integral = integrator.integrate(f, measure, domain)?;
            sequence_integrals.push(integral);
        }
        
        // Compute lim inf of integrals
        let lim_inf_integrals = sequence_integrals.iter()
            .copied()
            .fold(f64::INFINITY, f64::min);
        
        // Compute lim inf of sequence (pointwise)
        // This is simplified - would need proper pointwise computation
        let lim_inf_function = |x: &[f64]| {
            let values: Vec<f64> = sequence.iter().map(|f| f(x)).collect();
            values.iter().copied().fold(f64::INFINITY, f64::min)
        };
        
        // Compute ∫ (lim inf fₙ) dμ
        let integral_of_lim_inf = integrator.integrate(
            lim_inf_function,
            measure,
            domain,
        )?;
        
        // Verify Fatou's inequality
        Ok(integral_of_lim_inf <= lim_inf_integrals + self.tolerance)
    }
}
9. Comprehensive Testing
rust// tests/measure_tests.rs
use amari_measure::*;
use approx::assert_relative_eq;

#[test]
fn test_lebesgue_measure_interval() {
    let measure = LebesgueMeasure::new(1);
    
    let interval = BorelSet::ClosedInterval {
        dimension: 1,
        lower: vec![0.0],
        upper: vec![1.0],
    };
    
    let result = measure.measure_borel_set(&interval).unwrap();
    assert_relative_eq!(result, 1.0, epsilon = 1e-10);
}

#[test]
fn test_lebesgue_measure_rectangle() {
    let measure = LebesgueMeasure::new(2);
    
    let rectangle = BorelSet::ClosedInterval {
        dimension: 2,
        lower: vec![0.0, 0.0],
        upper: vec![2.0, 3.0],
    };
    
    let result = measure.measure_borel_set(&rectangle).unwrap();
    assert_relative_eq!(result, 6.0, epsilon = 1e-10);
}

#[test]
fn test_dirac_measure() {
    let measure = DiracMeasure::new(42);
    
    let mut set1 = HashSet::new();
    set1.insert(42);
    assert_relative_eq!(measure.measure(&set1).unwrap(), 1.0);
    
    let mut set2 = HashSet::new();
    set2.insert(43);
    assert_relative_eq!(measure.measure(&set2).unwrap(), 0.0);
}

// tests/integration_tests.rs
#[test]
fn test_integrate_constant() {
    let integrator = LebesgueIntegrator::new(1e-6, 1000);
    let measure = LebesgueMeasure::new(1);
    
    let domain = BorelSet::ClosedInterval {
        dimension: 1,
        lower: vec![0.0],
        upper: vec![1.0],
    };
    
    let constant_function = |_x: &[f64]| 5.0;
    
    let result = integrator.integrate(constant_function, &measure, &domain).unwrap();
    assert_relative_eq!(result, 5.0, epsilon = 1e-3);
}

#[test]
fn test_integrate_linear() {
    let integrator = LebesgueIntegrator::new(1e-6, 1000);
    let measure = LebesgueMeasure::new(1);
    
    let domain = BorelSet::RetryJEContinuerust    let domain = BorelSet::ClosedInterval {
        dimension: 1,
        lower: vec![0.0],
        upper: vec![1.0],
    };
    
    let linear_function = |x: &[f64]| x[0];
    
    let result = integrator.integrate(linear_function, &measure, &domain).unwrap();
    // ∫₀¹ x dx = 1/2
    assert_relative_eq!(result, 0.5, epsilon = 1e-2);
}

#[test]
fn test_integrate_quadratic() {
    let integrator = LebesgueIntegrator::new(1e-6, 1000);
    let measure = LebesgueMeasure::new(1);
    
    let domain = BorelSet::ClosedInterval {
        dimension: 1,
        lower: vec![0.0],
        upper: vec![1.0],
    };
    
    let quadratic_function = |x: &[f64]| x[0] * x[0];
    
    let result = integrator.integrate(quadratic_function, &measure, &domain).unwrap();
    // ∫₀¹ x² dx = 1/3
    assert_relative_eq!(result, 1.0/3.0, epsilon = 1e-2);
}

// tests/density_tests.rs
#[test]
fn test_radon_nikodym_simple() {
    let computer = RadonNikodymComputer::new(1e-6);
    
    let mu = LebesgueMeasure::new(1);
    let nu = LebesgueMeasure::new(1); // Same measure
    
    let point = vec![0.5];
    let derivative = computer.compute_derivative(&nu, &mu, &point).unwrap();
    
    // dν/dμ = 1 when ν = μ
    assert_relative_eq!(derivative, 1.0, epsilon = 1e-2);
}

// tests/convergence_tests.rs
#[test]
fn test_monotone_convergence() {
    let verifier = ConvergenceVerifier::new(1e-6);
    let measure = LebesgueMeasure::new(1);
    
    let domain = BorelSet::ClosedInterval {
        dimension: 1,
        lower: vec![0.0],
        upper: vec![1.0],
    };
    
    // Sequence: fₙ(x) = min(n·x, 1)
    let f1 = |x: &[f64]| (1.0 * x[0]).min(1.0);
    let f2 = |x: &[f64]| (2.0 * x[0]).min(1.0);
    let f3 = |x: &[f64]| (3.0 * x[0]).min(1.0);
    let f4 = |x: &[f64]| (4.0 * x[0]).min(1.0);
    
    let sequence = vec![f1, f2, f3, f4];
    
    // Limit: f(x) = 1
    let limit = |_x: &[f64]| 1.0;
    
    let result = verifier.verify_monotone_convergence(
        &sequence,
        &limit,
        &measure,
        &domain,
    ).unwrap();
    
    assert!(result);
}

#[test]
fn test_dominated_convergence() {
    let verifier = ConvergenceVerifier::new(1e-6);
    let measure = LebesgueMeasure::new(1);
    
    let domain = BorelSet::ClosedInterval {
        dimension: 1,
        lower: vec![0.0],
        upper: vec![1.0],
    };
    
    // Sequence: fₙ(x) = x^n
    let f1 = |x: &[f64]| x[0].powi(1);
    let f2 = |x: &[f64]| x[0].powi(2);
    let f3 = |x: &[f64]| x[0].powi(3);
    let f4 = |x: &[f64]| x[0].powi(4);
    
    let sequence = vec![f1, f2, f3, f4];
    
    // Limit: f(x) = 0 for x ∈ [0,1)
    let limit = |x: &[f64]| if x[0] < 1.0 { 0.0 } else { 1.0 };
    
    // Dominator: g(x) = 1
    let dominator = |_x: &[f64]| 1.0;
    
    let result = verifier.verify_dominated_convergence(
        &sequence,
        &limit,
        &dominator,
        &measure,
        &domain,
    ).unwrap();
    
    assert!(result);
}

// tests/product_tests.rs
#[test]
fn test_product_measure() {
    let mu = LebesgueMeasure::new(1);
    let nu = LebesgueMeasure::new(1);
    let product = ProductMeasure::new(mu, nu);
    
    let set_x = BorelSet::ClosedInterval {
        dimension: 1,
        lower: vec![0.0],
        upper: vec![2.0],
    };
    
    let set_y = BorelSet::ClosedInterval {
        dimension: 1,
        lower: vec![0.0],
        upper: vec![3.0],
    };
    
    let result = product.measure_product(&set_x, &set_y).unwrap();
    assert_relative_eq!(result, 6.0, epsilon = 1e-10);
}

#[test]
fn test_fubini_theorem() {
    let mu = LebesgueMeasure::new(1);
    let nu = LebesgueMeasure::new(1);
    let product = ProductMeasure::new(mu, nu);
    
    let domain_x = BorelSet::ClosedInterval {
        dimension: 1,
        lower: vec![0.0],
        upper: vec![1.0],
    };
    
    let domain_y = BorelSet::ClosedInterval {
        dimension: 1,
        lower: vec![0.0],
        upper: vec![1.0],
    };
    
    // f(x,y) = x + y
    let function = |x: &[f64], y: &[f64]| x[0] + y[0];
    
    let verified = product.verify_fubini(
        function,
        &domain_x,
        &domain_y,
        1e-2,
    ).unwrap();
    
    assert!(verified);
}

// tests/geometric_measure_tests.rs
use amari_core::Multivector;

#[test]
fn test_geometric_lebesgue_measure() {
    let measure = GeometricMeasure::<3, 0, 0>::lebesgue(3).unwrap();
    // Basic construction test
    assert_eq!(measure.dimension, 3);
}

#[test]
fn test_geometric_density() {
    let density = GeometricDensity::<2, 0, 0>::new(
        |x| {
            let scalar = x[0] + x[1];
            Multivector::from_components(&vec![scalar, 0.0, 0.0, 0.0]).unwrap()
        },
        2,
    );
    
    let point = vec![1.0, 2.0];
    let result = density.evaluate(&point);
    
    assert_relative_eq!(result.component(0), 3.0, epsilon = 1e-10);
}

#[test]
fn test_geometric_integration() {
    let integrator = GeometricIntegrator::<2, 0, 0>::new(1e-6, 1000);
    let measure = GeometricMeasure::<2, 0, 0>::lebesgue(2).unwrap();
    
    let domain = BorelSet::ClosedInterval {
        dimension: 2,
        lower: vec![0.0, 0.0],
        upper: vec![1.0, 1.0],
    };
    
    // Constant multivector field
    let field = |_x: &[f64]| {
        Multivector::from_components(&vec![1.0, 0.0, 0.0, 0.0]).unwrap()
    };
    
    let result = integrator.integrate(field, &measure, &domain).unwrap();
    
    // Should integrate to 1.0 in scalar component (area = 1)
    assert_relative_eq!(result.component(0), 1.0, epsilon = 1e-2);
}
10. Property-Based Tests
rust// tests/property_tests.rs
use proptest::prelude::*;
use amari_measure::*;

proptest! {
    #[test]
    fn prop_measure_monotonicity(
        a_lower in 0.0f64..1.0,
        a_upper in 1.0f64..2.0,
        b_upper in 2.0f64..3.0,
    ) {
        let measure = LebesgueMeasure::new(1);
        
        let set_a = BorelSet::ClosedInterval {
            dimension: 1,
            lower: vec![a_lower],
            upper: vec![a_upper],
        };
        
        let set_b = BorelSet::ClosedInterval {
            dimension: 1,
            lower: vec![a_lower],
            upper: vec![b_upper],
        };
        
        let mu_a = measure.measure_borel_set(&set_a).unwrap();
        let mu_b = measure.measure_borel_set(&set_b).unwrap();
        
        // A ⊆ B ⟹ μ(A) ≤ μ(B)
        prop_assert!(mu_a <= mu_b + 1e-10);
    }
    
    #[test]
    fn prop_measure_additivity(
        split_point in 0.1f64..0.9,
    ) {
        let measure = LebesgueMeasure::new(1);
        
        let set_total = BorelSet::ClosedInterval {
            dimension: 1,
            lower: vec![0.0],
            upper: vec![1.0],
        };
        
        let set_left = BorelSet::ClosedInterval {
            dimension: 1,
            lower: vec![0.0],
            upper: vec![split_point],
        };
        
        let set_right = BorelSet::ClosedInterval {
            dimension: 1,
            lower: vec![split_point],
            upper: vec![1.0],
        };
        
        let mu_total = measure.measure_borel_set(&set_total).unwrap();
        let mu_left = measure.measure_borel_set(&set_left).unwrap();
        let mu_right = measure.measure_borel_set(&set_right).unwrap();
        
        // μ(A ∪ B) = μ(A) + μ(B) for disjoint A, B
        prop_assert!((mu_total - (mu_left + mu_right)).abs() < 1e-10);
    }
    
    #[test]
    fn prop_integral_linearity(
        a in -10.0f64..10.0,
        b in -10.0f64..10.0,
    ) {
        let integrator = LebesgueIntegrator::new(1e-6, 1000);
        let measure = LebesgueMeasure::new(1);
        
        let domain = BorelSet::ClosedInterval {
            dimension: 1,
            lower: vec![0.0],
            upper: vec![1.0],
        };
        
        let f = |x: &[f64]| x[0];
        let g = |x: &[f64]| x[0] * x[0];
        let combined = |x: &[f64]| a * f(x) + b * g(x);
        
        let int_f = integrator.integrate(f, &measure, &domain).unwrap();
        let int_g = integrator.integrate(g, &measure, &domain).unwrap();
        let int_combined = integrator.integrate(combined, &measure, &domain).unwrap();
        
        // ∫(af + bg) = a∫f + b∫g
        prop_assert!((int_combined - (a * int_f + b * int_g)).abs() < 1e-2);
    }
    
    #[test]
    fn prop_radon_nikodym_identity(
        point in prop::collection::vec(0.0f64..1.0, 1..3),
    ) {
        let computer = RadonNikodymComputer::new(1e-6);
        let mu = LebesgueMeasure::new(point.len());
        
        // dμ/dμ should be 1
        let derivative = computer.compute_derivative(&mu, &mu, &point);
        
        if let Ok(d) = derivative {
            prop_assert!((d - 1.0).abs() < 1e-1);
        }
    }
}
11. Benchmarks
rust// benches/integration_benchmarks.rs
use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use amari_measure::*;

fn bench_lebesgue_integration(c: &mut Criterion) {
    let mut group = c.benchmark_group("lebesgue_integration");
    
    for dim in [1, 2, 3, 4].iter() {
        group.bench_with_input(
            BenchmarkId::from_parameter(dim),
            dim,
            |b, &dim| {
                let integrator = LebesgueIntegrator::new(1e-3, 100);
                let measure = LebesgueMeasure::new(dim);
                
                let lower = vec![0.0; dim];
                let upper = vec![1.0; dim];
                let domain = BorelSet::ClosedInterval {
                    dimension: dim,
                    lower,
                    upper,
                };
                
                let function = |x: &[f64]| x.iter().sum::<f64>();
                
                b.iter(|| {
                    integrator.integrate(
                        black_box(&function),
                        black_box(&measure),
                        black_box(&domain),
                    )
                });
            },
        );
    }
    
    group.finish();
}

fn bench_geometric_integration(c: &mut Criterion) {
    let mut group = c.benchmark_group("geometric_integration");
    
    for dim in [2, 3, 4].iter() {
        group.bench_with_input(
            BenchmarkId::from_parameter(dim),
            dim,
            |b, &dim| {
                use amari_core::Multivector;
                
                let integrator = GeometricIntegrator::<3, 0, 0>::new(1e-3, 100);
                let measure = GeometricMeasure::<3, 0, 0>::lebesgue(*dim).unwrap();
                
                let lower = vec![0.0; *dim];
                let upper = vec![1.0; *dim];
                let domain = BorelSet::ClosedInterval {
                    dimension: *dim,
                    lower,
                    upper,
                };
                
                let field = |x: &[f64]| {
                    let scalar = x.iter().sum::<f64>();
                    Multivector::from_components(&vec![scalar, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]).unwrap()
                };
                
                b.iter(|| {
                    integrator.integrate(
                        black_box(&field),
                        black_box(&measure),
                        black_box(&domain),
                    )
                });
            },
        );
    }
    
    group.finish();
}

fn bench_radon_nikodym(c: &mut Criterion) {
    let computer = RadonNikodymComputer::new(1e-6);
    let mu = LebesgueMeasure::new(2);
    let nu = LebesgueMeasure::new(2);
    
    c.bench_function("radon_nikodym_2d", |b| {
        let point = vec![0.5, 0.5];
        b.iter(|| {
            computer.compute_derivative(
                black_box(&nu),
                black_box(&mu),
                black_box(&point),
            )
        });
    });
}

criterion_group!(
    benches,
    bench_lebesgue_integration,
    bench_geometric_integration,
    bench_radon_nikodym
);
criterion_main!(benches);
12. Examples
rust// examples/lebesgue_integration.rs
//! Basic Lebesgue integration examples

use amari_measure::*;

fn main() -> Result<()> {
    println!("=== Lebesgue Integration Examples ===\n");
    
    // Example 1: Integrate constant function
    {
        println!("Example 1: ∫₀¹ 5 dx");
        
        let integrator = LebesgueIntegrator::new(1e-6, 1000);
        let measure = LebesgueMeasure::new(1);
        
        let domain = BorelSet::ClosedInterval {
            dimension: 1,
            lower: vec![0.0],
            upper: vec![1.0],
        };
        
        let f = |_x: &[f64]| 5.0;
        let result = integrator.integrate(f, &measure, &domain)?;
        
        println!("Result: {:.6}", result);
        println!("Expected: 5.0\n");
    }
    
    // Example 2: Integrate polynomial
    {
        println!("Example 2: ∫₀¹ x² dx");
        
        let integrator = LebesgueIntegrator::new(1e-6, 1000);
        let measure = LebesgueMeasure::new(1);
        
        let domain = BorelSet::ClosedInterval {
            dimension: 1,
            lower: vec![0.0],
            upper: vec![1.0],
        };
        
        let f = |x: &[f64]| x[0].powi(2);
        let result = integrator.integrate(f, &measure, &domain)?;
        
        println!("Result: {:.6}", result);
        println!("Expected: 0.333333\n");
    }
    
    // Example 3: Multidimensional integration
    {
        println!("Example 3: ∫∫ (x+y) dxdy over [0,1]×[0,1]");
        
        let integrator = LebesgueIntegrator::new(1e-6, 1000);
        let measure = LebesgueMeasure::new(2);
        
        let domain = BorelSet::ClosedInterval {
            dimension: 2,
            lower: vec![0.0, 0.0],
            upper: vec![1.0, 1.0],
        };
        
        let f = |x: &[f64]| x[0] + x[1];
        let result = integrator.integrate(f, &measure, &domain)?;
        
        println!("Result: {:.6}", result);
        println!("Expected: 1.0\n");
    }
    
    Ok(())
}

// examples/probability_measure.rs
//! Probability measure examples

use amari_measure::*;

fn main() -> Result<()> {
    println!("=== Probability Measure Examples ===\n");
    
    // Example 1: Dirac measure (point mass)
    {
        println!("Example 1: Dirac measure at x=5");
        
        let measure = DiracMeasure::new(5);
        
        let mut set1 = std::collections::HashSet::new();
        set1.insert(5);
        
        let mut set2 = std::collections::HashSet::new();
        set2.insert(3);
        
        println!("δ₅({{5}}) = {:.6}", measure.measure(&set1)?);
        println!("δ₅({{3}}) = {:.6}", measure.measure(&set2)?);
        println!();
    }
    
    // Example 2: Uniform distribution
    {
        println!("Example 2: Uniform distribution on [0,1]");
        
        let integrator = LebesgueIntegrator::new(1e-6, 1000);
        let measure = LebesgueMeasure::new(1);
        
        let domain = BorelSet::ClosedInterval {
            dimension: 1,
            lower: vec![0.0],
            upper: vec![1.0],
        };
        
        // Density is constant: f(x) = 1
        let density = |_x: &[f64]| 1.0;
        
        // Integrate to get probabilities
        let prob_first_half = {
            let half_domain = BorelSet::ClosedInterval {
                dimension: 1,
                lower: vec![0.0],
                upper: vec![0.5],
            };
            integrator.integrate(density, &measure, &half_domain)?
        };
        
        println!("P(X ∈ [0, 0.5]) = {:.6}", prob_first_half);
        println!("Expected: 0.5\n");
    }
    
    Ok(())
}

// examples/geometric_density.rs
//! Geometric density examples

use amari_measure::*;
use amari_core::Multivector;

fn main() -> Result<()> {
    println!("=== Geometric Density Examples ===\n");
    
    // Example 1: Vector-valued density
    {
        println!("Example 1: Vector-valued density ρ(x,y) = xe₁ + ye₂");
        
        let density = GeometricDensity::<2, 0, 0>::new(
            |x| {
                // Scalar: 0, e1: x[0], e2: x[1], e12: 0
                Multivector::from_components(&vec![
                    0.0,           // scalar
                    x[0],          // e1
                    x[1],          // e2
                    0.0,           // e12
                ]).unwrap()
            },
            2,
        );
        
        let point = vec![1.0, 2.0];
        let value = density.evaluate(&point);
        
        println!("ρ(1, 2) = {:.3}e₁ + {:.3}e₂", value.component(1), value.component(2));
        println!();
    }
    
    // Example 2: Bivector-valued density
    {
        println!("Example 2: Bivector density for oriented area");
        
        let density = GeometricDensity::<2, 0, 0>::new(
            |x| {
                let bivector_component = x[0] * x[1];
                Multivector::from_components(&vec![
                    0.0,               // scalar
                    0.0,               // e1
                    0.0,               // e2
                    bivector_component, // e12
                ]).unwrap()
            },
            2,
        );
        
        let point = vec![3.0, 4.0];
        let value = density.evaluate(&point);
        
        println!("ρ(3, 4) = {:.3}e₁₂", value.component(3));
        println!();
    }
    
    Ok(())
}

// examples/fubini_theorem.rs
//! Fubini's theorem demonstration

use amari_measure::*;

fn main() -> Result<()> {
    println!("=== Fubini's Theorem ===\n");
    
    let mu = LebesgueMeasure::new(1);
    let nu = LebesgueMeasure::new(1);
    let product = ProductMeasure::new(mu, nu);
    
    let domain_x = BorelSet::ClosedInterval {
        dimension: 1,
        lower: vec![0.0],
        upper: vec![1.0],
    };
    
    let domain_y = BorelSet::ClosedInterval {
        dimension: 1,
        lower: vec![0.0],
        upper: vec![1.0],
    };
    
    // Test several functions
    let functions: Vec<(&str, Box<dyn Fn(&[f64], &[f64]) -> f64>)> = vec![
        ("x + y", Box::new(|x, y| x[0] + y[0])),
        ("x * y", Box::new(|x, y| x[0] * y[0])),
        ("x² + y²", Box::new(|x, y| x[0].powi(2) + y[0].powi(2))),
    ];
    
    for (name, f) in functions.iter() {
        println!("Function: f(x,y) = {}", name);
        
        let (order1, order2) = product.fubini_swap(&**f, &domain_x, &domain_y)?;
        
        println!("  ∫∫ f(x,y) dydx = {:.6}", order1);
        println!("  ∫∫ f(x,y) dxdy = {:.6}", order2);
        println!("  Difference: {:.9}", (order1 - order2).abs());
        println!();
    }
    
    Ok(())
}
13. Cargo.toml
toml[package]
name = "amari-measure"
version = "0.10.0"
edition = "2021"
rust-version = "1.75"
authors = ["Justin Restivo <your.email@example.com>"]
description = "Measure-theoretic foundations for geometric algebra - Lebesgue integration and probability"
license = "MIT OR Apache-2.0"
repository = "https://github.com/username/Amari"
keywords = ["measure-theory", "integration", "lebesgue", "geometric-algebra", "probability"]
categories = ["mathematics", "science"]

[dependencies]
amari-core = { path = "../amari-core", version = "0.9.0" }
thiserror = "1.0"
rand = "0.8"

# Optional dependencies
serde = { version = "1.0", features = ["derive"], optional = true }
rayon = { version = "1.10", optional = true }

[dev-dependencies]
approx = "0.5"
proptest = "1.4"
criterion = "0.5"
creusot-contracts = "0.2"

[features]
default = []
serde = ["dep:serde", "amari-core/serde"]
parallel = ["dep:rayon"]

[[bench]]
name = "integration_benchmarks"
harness = false

[package.metadata.docs.rs]
all-features = true
rustdoc-args = ["--cfg", "docsrs"]
14. Documentation (lib.rs header)
rust// src/lib.rs
//! # amari-measure
//!
//! Measure-theoretic foundations for geometric algebra, providing rigorous
//! integration theory and probability measures over multivector spaces.
//!
//! ## Mathematical Background
//!
//! ### Measure Theory
//!
//! A **measure** is a function μ: Σ → [0, ∞] defined on a σ-algebra Σ that satisfies:
//! 1. μ(∅) = 0 (null empty set)
//! 2. Countable additivity: μ(⋃ᵢ Aᵢ) = Σᵢ μ(Aᵢ) for disjoint sets
//!
//! ### Lebesgue Integration
//!
//! The Lebesgue integral extends the Riemann integral by:
//! - Integrating over arbitrary measurable sets (not just intervals)
//! - Handling more general functions (measurable functions)
//! - Providing powerful convergence theorems
//!
//! For a measurable function f and measure μ:
//! ```text
//! ∫ f dμ = sup { ∫ s dμ : s simple, 0 ≤ s ≤ f }
//! ```
//!
//! ### Geometric Measures
//!
//! This crate extends real-valued measures to **multivector-valued measures**:
//! ```text
//! μ: Σ → Cl(p,q,r)
//! ```
//!
//! This enables:
//! - Vector-valued densities (velocity distributions, force fields)
//! - Bivector-valued measures (oriented area elements, electromagnetic fields)
//! - Full multivector integration for geometric algebra applications
//!
//! ## Key Theorems
//!
//! ### Monotone Convergence Theorem (Beppo Levi)
//! If 0 ≤ f₁ ≤ f₂ ≤ ... and fₙ → f pointwise, then:
//! ```text
//! lim_{n→∞} ∫ fₙ dμ = ∫ f dμ
//! ```
//!
//! ### Dominated Convergence Theorem (Lebesgue)
//! If fₙ → f pointwise, |fₙ| ≤ g for all n, and ∫ g dμ < ∞, then:
//! ```text
//! lim_{n→∞} ∫ fₙ dμ = ∫ f dμ
//! ```
//!
//! ### Radon-Nikodym Theorem
//! If μ, ν are σ-finite and ν ≪ μ, there exists f ≥ 0 such that:
//! ```text
//! ν(A) = ∫_A f dμ
//! ```
//!
//! ### Fubini's Theorem
//! For product measure μ ⊗ ν:
//! ```text
//! ∫∫ f d(μ⊗ν) = ∫[∫ f dν] dμ = ∫[∫ f dμ] dν
//! ```
//!
//! ## Usage Example
//!
//! ```rust
//! use amari_measure::*;
//!
//! # fn main() -> Result<()> {
//! // Create Lebesgue measure on ℝ
//! let measure = LebesgueMeasure::new(1);
//!
//! // Define domain [0, 1]
//! let domain = BorelSet::ClosedInterval {
//!     dimension: 1,
//!     lower: vec![0.0],
//!     upper: vec![1.0],
//! };
//!
//! // Integrate f(x) = x²
//! let integrator = LebesgueIntegrator::new(1e-6, 1000);
//! let f = |x: &[f64]| x[0].powi(2);
//! let result = integrator.integrate(f, &measure, &domain)?;
//!
//! // Result should be 1/3
//! assert!((result - 1.0/3.0).abs() < 1e-2);
//! # Ok(())
//! # }
//! ```
//!
//! ## Integration with Amari
//!
//! This crate provides the measure-theoretic foundations for:
//! - **amari-probabilistic**: Probability theory over multivectors
//! - **amari-info-geom**: Information geometry with proper measures
//! - **amari-calculus**: Geometric calculus requiring integration
//! - **amari-pde**: PDE theory with weak solutions
//!
//! ## Performance Characteristics
//!
//! - Integration: O(n^d) where d is dimension (curse of dimensionality)
//! - Radon-Nikodym derivatives: O(1/ε²) for accuracy ε
//! - Product measures: O(mn) for sets of size m, n
//! - Geometric measures: O(2^n) where n is GA dimension
//!
//! ## References
//!
//! - Rudin, W. "Real and Complex Analysis" (1987)
//! - Folland, G. "Real Analysis" (1999)
//! - Cohn, D. "Measure Theory" (2013)
//! - Billingsley, P. "Probability and Measure" (2012)

pub mod error;
pub mod sigma_algebra;
pub mod measure;
pub mod geometric_measure;
pub mod integration;
pub mod density;
pub mod pushforward;
pub mod product;
pub mod convergence;
pub mod phantom;

pub use error::{MeasureError, Result};
pub use sigma_algebra::*;
pub use measure::*;
pub use geometric_measure::*;
pub use integration::*;
pub use density::*;
pub use pushforward::*;
pub use product::*;
pub use convergence::*;
pub use phantom::*;
Implementation Priorities

σ-algebras and basic measures (foundation)
Lebesgue integration (core functionality)
Geometric measures (GA extension)
Radon-Nikodym derivatives (densities)
Product measures and Fubini (multidimensional)
Convergence theorems (theoretical completeness)
Pushforward/pullback (advanced features)
Tests and examples (verification)

Success Criteria

 All measure axioms verified with Creusot contracts
 Convergence theorems correctly implemented
 Integration accurate to specified tolerance
 Geometric measures integrate with amari-core
 Property-based tests verify measure theory axioms
 Examples demonstrate all key theorems
 Performance acceptable for practical use
 Documentation includes mathematical background
 All clippy lints pass (pedantic + nursery)
 Ready for amari-probabilistic and amari-calculus

Notes

This crate provides the analytical foundations for Amari
Complements the algebraic foundations in amari-core
Enables rigorous probability theory in amari-probabilistic
Essential for differential geometry in amari-calculus
Numerical integration is approximate - tolerances matter
Creusot verification focuses on measure axioms, not numerical accuracy
Start simple (1D intervals), extend to general sets gradually