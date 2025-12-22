//! Phantom type system for distribution properties
//!
//! This module provides zero-cost phantom types that encode distribution properties
//! at compile time, enabling type-safe probabilistic operations on geometric algebra.
//!
//! # Design Philosophy
//!
//! Following the Amari phantom types methodology:
//! - Zero runtime cost (all markers are zero-sized types)
//! - Compile-time verification of distribution properties
//! - Type-level encoding of mathematical structure
//! - Prevention of invalid probabilistic operations
//!
//! # Property Categories
//!
//! ## Support Properties
//!
//! - `Bounded`: Distribution has bounded support (compact)
//! - `Unbounded`: Distribution has unbounded support
//! - `Discrete`: Distribution is over a discrete set
//! - `Continuous`: Distribution has continuous density
//!
//! ## Moment Properties
//!
//! - `FiniteMoments<N>`: Distribution has finite moments up to order N
//! - `LightTailed`: Exponentially bounded tails
//! - `HeavyTailed`: Polynomial-bounded tails
//!
//! ## Geometric Properties
//!
//! - `GradeHomogeneous`: Distribution is over a single grade
//! - `GradeHeterogeneous`: Distribution spans multiple grades
//! - `RotorValued`: Distribution is over the rotor subgroup

use core::marker::PhantomData;

/// Marker trait for all distribution property types
///
/// This trait is sealed and cannot be implemented outside this module.
pub trait DistributionProperty: private::Sealed {}

/// Marker trait for support properties
pub trait SupportProperty: DistributionProperty {}

/// Marker trait for moment properties
pub trait MomentProperty: DistributionProperty {}

/// Marker trait for geometric properties
pub trait GeometricProperty: DistributionProperty {}

// Sealed trait pattern
mod private {
    pub trait Sealed {}
}

// ============================================================================
// Support Properties
// ============================================================================

/// Marker type for bounded support distributions
///
/// Distribution support is contained in a compact subset of the multivector space.
///
/// # Examples
///
/// - Uniform distribution on a hypercube
/// - Truncated Gaussian
/// - Distribution on unit rotors (SO(n) subgroup)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Bounded;

impl DistributionProperty for Bounded {}
impl SupportProperty for Bounded {}
impl private::Sealed for Bounded {}

/// Marker type for unbounded support distributions
///
/// Distribution support extends to infinity in some directions.
///
/// # Examples
///
/// - Gaussian distribution on multivector space
/// - Cauchy distribution
/// - Lévy stable distributions
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Unbounded;

impl DistributionProperty for Unbounded {}
impl SupportProperty for Unbounded {}
impl private::Sealed for Unbounded {}

/// Marker type for discrete distributions
///
/// Distribution assigns probability to a countable set of points.
///
/// # Examples
///
/// - Categorical distribution over basis blades
/// - Poisson-like distribution on integer coefficients
/// - Multinomial distribution on grades
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Discrete;

impl DistributionProperty for Discrete {}
impl SupportProperty for Discrete {}
impl private::Sealed for Discrete {}

/// Marker type for continuous distributions
///
/// Distribution has a density with respect to Lebesgue measure.
///
/// # Examples
///
/// - Gaussian on multivector coefficients
/// - Beta distribution on normalized rotors
/// - Wishart-like distribution on bivector covariance
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Continuous;

impl DistributionProperty for Continuous {}
impl SupportProperty for Continuous {}
impl private::Sealed for Continuous {}

// ============================================================================
// Moment Properties
// ============================================================================

/// Marker type for distributions with finite moments
///
/// Type parameter N indicates the highest finite moment.
///
/// # Examples
///
/// - `FiniteMoments<2>`: Distribution has finite mean and variance
/// - `FiniteMoments<4>`: Has finite kurtosis
/// - All bounded distributions have all moments finite
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FiniteMoments<const N: usize>;

impl<const N: usize> DistributionProperty for FiniteMoments<N> {}
impl<const N: usize> MomentProperty for FiniteMoments<N> {}
impl<const N: usize> private::Sealed for FiniteMoments<N> {}

/// Marker type for light-tailed distributions
///
/// Tails decay at least exponentially: P(|X| > t) ≤ C·exp(-λt)
///
/// # Examples
///
/// - Gaussian distribution
/// - Exponential distribution
/// - Gamma distribution
///
/// # Properties
///
/// - All moments are finite
/// - MGF exists in a neighborhood of 0
/// - Sub-Gaussian or sub-exponential concentration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct LightTailed;

impl DistributionProperty for LightTailed {}
impl MomentProperty for LightTailed {}
impl private::Sealed for LightTailed {}

/// Marker type for heavy-tailed distributions
///
/// Tails decay polynomially: P(|X| > t) ~ C·t^(-α)
///
/// # Examples
///
/// - Cauchy distribution (no finite moments)
/// - Pareto distribution
/// - Student's t-distribution
///
/// # Properties
///
/// - May have infinite moments
/// - MGF does not exist
/// - Weaker concentration inequalities
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct HeavyTailed;

impl DistributionProperty for HeavyTailed {}
impl MomentProperty for HeavyTailed {}
impl private::Sealed for HeavyTailed {}

// ============================================================================
// Geometric Properties
// ============================================================================

/// Marker type for grade-homogeneous distributions
///
/// Distribution is concentrated on a single grade of the Clifford algebra.
///
/// # Examples
///
/// - Distribution on vectors only (grade 1)
/// - Distribution on bivectors only (grade 2)
/// - Scalar-valued random variables (grade 0)
///
/// # Type Parameter
///
/// - `G`: The grade of the distribution (0, 1, 2, ...)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GradeHomogeneous<const G: usize>;

impl<const G: usize> DistributionProperty for GradeHomogeneous<G> {}
impl<const G: usize> GeometricProperty for GradeHomogeneous<G> {}
impl<const G: usize> private::Sealed for GradeHomogeneous<G> {}

/// Marker type for grade-heterogeneous distributions
///
/// Distribution spans multiple grades of the Clifford algebra.
///
/// # Examples
///
/// - Full multivector distribution (all grades)
/// - Even-grade distribution (scalars + bivectors + ...)
/// - Rotor distribution (even subalgebra)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GradeHeterogeneous;

impl DistributionProperty for GradeHeterogeneous {}
impl GeometricProperty for GradeHeterogeneous {}
impl private::Sealed for GradeHeterogeneous {}

/// Marker type for rotor-valued distributions
///
/// Distribution is over the spin group (normalized even-grade elements).
///
/// # Examples
///
/// - Uniform distribution on SO(3) via unit quaternions
/// - von Mises-Fisher on rotation axes with concentration on angle
/// - Bingham distribution on rotation matrices
///
/// # Properties
///
/// - Elements satisfy R·R̃ = 1 (unit norm)
/// - Even-grade only (scalar + bivector + ...)
/// - Forms a Lie group under geometric product
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RotorValued;

impl DistributionProperty for RotorValued {}
impl GeometricProperty for RotorValued {}
impl private::Sealed for RotorValued {}

/// Marker type for versor-valued distributions
///
/// Distribution is over versors (products of vectors).
///
/// # Properties
///
/// - Includes both rotors (even) and reflections (odd)
/// - Elements are invertible
/// - Grade-mixing under geometric product
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct VersorValued;

impl DistributionProperty for VersorValued {}
impl GeometricProperty for VersorValued {}
impl private::Sealed for VersorValued {}

// ============================================================================
// Type-Level Property Combinations
// ============================================================================

/// Distribution over bounded support with all moments finite
#[allow(dead_code)]
pub type CompactDistribution = (Bounded, LightTailed, GradeHeterogeneous);

/// Standard Gaussian-like distribution on multivector space
#[allow(dead_code)]
pub type GaussianLike = (Unbounded, LightTailed, GradeHeterogeneous);

/// Distribution on the rotation group
#[allow(dead_code)]
pub type RotationDistribution = (Bounded, LightTailed, RotorValued);

/// Heavy-tailed distribution (Cauchy-like)
#[allow(dead_code)]
pub type CauchyLike = (Unbounded, HeavyTailed, GradeHeterogeneous);

// ============================================================================
// Property Marker Holder
// ============================================================================

/// Zero-cost container for distribution property markers
///
/// Used to attach compile-time properties to distributions without runtime cost.
#[derive(Debug, Clone, Copy)]
pub struct PropertyMarker<S, M, G> {
    _support: PhantomData<S>,
    _moments: PhantomData<M>,
    _geometric: PhantomData<G>,
}

impl<S, M, G> PropertyMarker<S, M, G> {
    /// Create a new property marker
    pub const fn new() -> Self {
        Self {
            _support: PhantomData,
            _moments: PhantomData,
            _geometric: PhantomData,
        }
    }
}

impl<S, M, G> Default for PropertyMarker<S, M, G> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::mem::size_of;

    #[test]
    fn test_phantom_types_are_zero_sized() {
        assert_eq!(size_of::<Bounded>(), 0);
        assert_eq!(size_of::<Unbounded>(), 0);
        assert_eq!(size_of::<Discrete>(), 0);
        assert_eq!(size_of::<Continuous>(), 0);
        assert_eq!(size_of::<LightTailed>(), 0);
        assert_eq!(size_of::<HeavyTailed>(), 0);
        assert_eq!(size_of::<FiniteMoments<2>>(), 0);
        assert_eq!(size_of::<GradeHomogeneous<1>>(), 0);
        assert_eq!(size_of::<GradeHeterogeneous>(), 0);
        assert_eq!(size_of::<RotorValued>(), 0);
        assert_eq!(size_of::<VersorValued>(), 0);
    }

    #[test]
    fn test_property_marker_is_zero_sized() {
        type GaussianMarker = PropertyMarker<Unbounded, LightTailed, GradeHeterogeneous>;
        assert_eq!(size_of::<GaussianMarker>(), 0);

        type RotorMarker = PropertyMarker<Bounded, LightTailed, RotorValued>;
        assert_eq!(size_of::<RotorMarker>(), 0);
    }

    #[test]
    fn test_distribution_property_trait() {
        fn is_distribution_property<T: DistributionProperty>() {}

        is_distribution_property::<Bounded>();
        is_distribution_property::<Unbounded>();
        is_distribution_property::<LightTailed>();
        is_distribution_property::<HeavyTailed>();
        is_distribution_property::<GradeHomogeneous<0>>();
        is_distribution_property::<RotorValued>();
    }

    #[test]
    fn test_support_property_trait() {
        fn is_support_property<T: SupportProperty>() {}

        is_support_property::<Bounded>();
        is_support_property::<Unbounded>();
        is_support_property::<Discrete>();
        is_support_property::<Continuous>();
    }

    #[test]
    fn test_geometric_property_trait() {
        fn is_geometric_property<T: GeometricProperty>() {}

        is_geometric_property::<GradeHomogeneous<1>>();
        is_geometric_property::<GradeHomogeneous<2>>();
        is_geometric_property::<GradeHeterogeneous>();
        is_geometric_property::<RotorValued>();
        is_geometric_property::<VersorValued>();
    }
}
