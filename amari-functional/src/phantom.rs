//! Phantom types for compile-time verification of functional analysis properties.
//!
//! This module provides zero-cost phantom types that encode mathematical properties
//! of function spaces, operators, and spectral characteristics at the type level.
//!
//! # Space Completeness
//!
//! - [`Complete`]: The space is complete (every Cauchy sequence converges)
//! - [`PreHilbert`]: Inner product space, not necessarily complete
//!
//! # Operator Properties
//!
//! - [`Bounded`]: The operator is bounded (continuous)
//! - [`Compact`]: The operator is compact (maps bounded sets to precompact sets)
//! - [`SelfAdjoint`]: The operator equals its adjoint
//! - [`Normal`]: The operator commutes with its adjoint
//! - [`Unitary`]: The operator preserves inner products
//!
//! # Spectral Properties
//!
//! - [`DiscreteSpectrum`]: Spectrum consists of isolated eigenvalues
//! - [`ContinuousSpectrum`]: Spectrum contains continuous parts
//! - [`PurePointSpectrum`]: All spectrum is point spectrum

use core::marker::PhantomData;

// Sealed trait pattern to prevent external implementations
mod private {
    pub trait Sealed {}
}

// ============================================================================
// Space Completeness Properties
// ============================================================================

/// Marker trait for space completeness properties.
pub trait CompletenessProperty: private::Sealed + Clone + Copy + Default + 'static {}

/// Complete space: every Cauchy sequence converges within the space.
///
/// This is the defining property of Banach and Hilbert spaces.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Complete;

impl private::Sealed for Complete {}
impl CompletenessProperty for Complete {}

/// Pre-Hilbert space: inner product defined but not necessarily complete.
///
/// Completion of a PreHilbert space yields a Hilbert space.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct PreHilbert;

impl private::Sealed for PreHilbert {}
impl CompletenessProperty for PreHilbert {}

// ============================================================================
// Operator Properties
// ============================================================================

/// Marker trait for operator boundedness properties.
pub trait BoundednessProperty: private::Sealed + Clone + Copy + Default + 'static {}

/// Bounded operator: ||Tx|| ≤ M||x|| for some M and all x.
///
/// Bounded operators are exactly the continuous linear operators.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Bounded;

impl private::Sealed for Bounded {}
impl BoundednessProperty for Bounded {}

/// Unbounded operator: no finite bound exists.
///
/// Examples include differential operators on L² spaces.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Unbounded;

impl private::Sealed for Unbounded {}
impl BoundednessProperty for Unbounded {}

/// Marker trait for operator compactness properties.
pub trait CompactnessProperty: private::Sealed + Clone + Copy + Default + 'static {}

/// Compact operator: maps bounded sets to precompact sets.
///
/// Compact operators have discrete spectrum (except possibly 0).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Compact;

impl private::Sealed for Compact {}
impl CompactnessProperty for Compact {}

/// Non-compact operator.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct NonCompact;

impl private::Sealed for NonCompact {}
impl CompactnessProperty for NonCompact {}

/// Marker trait for operator symmetry properties.
pub trait SymmetryProperty: private::Sealed + Clone + Copy + Default + 'static {}

/// Self-adjoint operator: T = T*.
///
/// Self-adjoint operators have real spectrum and orthogonal eigenvectors.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct SelfAdjoint;

impl private::Sealed for SelfAdjoint {}
impl SymmetryProperty for SelfAdjoint {}

/// Normal operator: TT* = T*T.
///
/// Normal operators can be diagonalized by unitary transformation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Normal;

impl private::Sealed for Normal {}
impl SymmetryProperty for Normal {}

/// Unitary operator: T*T = TT* = I.
///
/// Unitary operators preserve inner products and norms.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Unitary;

impl private::Sealed for Unitary {}
impl SymmetryProperty for Unitary {}

/// General operator with no special symmetry properties.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct General;

impl private::Sealed for General {}
impl SymmetryProperty for General {}

// ============================================================================
// Spectral Properties
// ============================================================================

/// Marker trait for spectral properties.
pub trait SpectralProperty: private::Sealed + Clone + Copy + Default + 'static {}

/// Discrete spectrum: consists only of isolated eigenvalues.
///
/// Compact operators on infinite-dimensional spaces have discrete spectrum.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct DiscreteSpectrum;

impl private::Sealed for DiscreteSpectrum {}
impl SpectralProperty for DiscreteSpectrum {}

/// Continuous spectrum: contains intervals or continuous parts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct ContinuousSpectrum;

impl private::Sealed for ContinuousSpectrum {}
impl SpectralProperty for ContinuousSpectrum {}

/// Pure point spectrum: all spectral values are eigenvalues.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct PurePointSpectrum;

impl private::Sealed for PurePointSpectrum {}
impl SpectralProperty for PurePointSpectrum {}

/// Mixed spectrum: combination of discrete and continuous parts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct MixedSpectrum;

impl private::Sealed for MixedSpectrum {}
impl SpectralProperty for MixedSpectrum {}

// ============================================================================
// Sobolev Space Properties
// ============================================================================

/// Marker trait for Sobolev regularity.
pub trait RegularityProperty: private::Sealed + Clone + Copy + Default + 'static {}

/// L² regularity (k=0 Sobolev space).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct L2Regularity;

impl private::Sealed for L2Regularity {}
impl RegularityProperty for L2Regularity {}

/// H¹ regularity (k=1 Sobolev space).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct H1Regularity;

impl private::Sealed for H1Regularity {}
impl RegularityProperty for H1Regularity {}

/// H² regularity (k=2 Sobolev space).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct H2Regularity;

impl private::Sealed for H2Regularity {}
impl RegularityProperty for H2Regularity {}

/// General Hᵏ regularity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct HkRegularity<const K: usize>;

impl<const K: usize> private::Sealed for HkRegularity<K> {}
impl<const K: usize> RegularityProperty for HkRegularity<K> {}

// ============================================================================
// Fredholm Properties
// ============================================================================

/// Marker trait for Fredholm operator properties.
pub trait FredholmProperty: private::Sealed + Clone + Copy + Default + 'static {}

/// Fredholm operator: finite-dimensional kernel and cokernel.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Fredholm;

impl private::Sealed for Fredholm {}
impl FredholmProperty for Fredholm {}

/// Semi-Fredholm operator: either kernel or cokernel is finite-dimensional.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct SemiFredholm;

impl private::Sealed for SemiFredholm {}
impl FredholmProperty for SemiFredholm {}

/// Not Fredholm.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct NotFredholm;

impl private::Sealed for NotFredholm {}
impl FredholmProperty for NotFredholm {}

// ============================================================================
// Type Aliases for Common Combinations
// ============================================================================

/// Properties of a compact self-adjoint operator.
pub type CompactSelfAdjointOperator = (Compact, SelfAdjoint, DiscreteSpectrum);

/// Properties of a unitary operator.
pub type UnitaryOperator = (Bounded, Unitary, PurePointSpectrum);

/// Properties of a Hilbert-Schmidt operator.
pub type HilbertSchmidtOperator = (Compact, General, DiscreteSpectrum);

/// Standard L² Hilbert space properties.
pub type L2SpaceProperties = (Complete, L2Regularity);

/// Standard H¹ Sobolev space properties.
pub type H1SpaceProperties = (Complete, H1Regularity);

// ============================================================================
// Phantom Data Wrapper
// ============================================================================

/// Zero-cost wrapper for phantom type properties.
///
/// This struct has no runtime cost and is used to carry type-level
/// information about space and operator properties.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Properties<T>(PhantomData<T>);

impl<T> Properties<T> {
    /// Create a new properties marker.
    pub const fn new() -> Self {
        Self(PhantomData)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::mem::size_of;

    #[test]
    fn test_phantom_types_are_zero_sized() {
        assert_eq!(size_of::<Complete>(), 0);
        assert_eq!(size_of::<PreHilbert>(), 0);
        assert_eq!(size_of::<Bounded>(), 0);
        assert_eq!(size_of::<Compact>(), 0);
        assert_eq!(size_of::<SelfAdjoint>(), 0);
        assert_eq!(size_of::<Normal>(), 0);
        assert_eq!(size_of::<Unitary>(), 0);
        assert_eq!(size_of::<DiscreteSpectrum>(), 0);
        assert_eq!(size_of::<Fredholm>(), 0);
        assert_eq!(size_of::<L2Regularity>(), 0);
        assert_eq!(size_of::<HkRegularity<5>>(), 0);
        assert_eq!(size_of::<Properties<CompactSelfAdjointOperator>>(), 0);
    }

    #[test]
    fn test_phantom_types_implement_traits() {
        fn assert_completeness<T: CompletenessProperty>() {}
        fn assert_boundedness<T: BoundednessProperty>() {}
        fn assert_compactness<T: CompactnessProperty>() {}
        fn assert_symmetry<T: SymmetryProperty>() {}
        fn assert_spectral<T: SpectralProperty>() {}
        fn assert_regularity<T: RegularityProperty>() {}
        fn assert_fredholm<T: FredholmProperty>() {}

        assert_completeness::<Complete>();
        assert_completeness::<PreHilbert>();

        assert_boundedness::<Bounded>();
        assert_boundedness::<Unbounded>();

        assert_compactness::<Compact>();
        assert_compactness::<NonCompact>();

        assert_symmetry::<SelfAdjoint>();
        assert_symmetry::<Normal>();
        assert_symmetry::<Unitary>();
        assert_symmetry::<General>();

        assert_spectral::<DiscreteSpectrum>();
        assert_spectral::<ContinuousSpectrum>();
        assert_spectral::<PurePointSpectrum>();
        assert_spectral::<MixedSpectrum>();

        assert_regularity::<L2Regularity>();
        assert_regularity::<H1Regularity>();
        assert_regularity::<H2Regularity>();
        assert_regularity::<HkRegularity<3>>();

        assert_fredholm::<Fredholm>();
        assert_fredholm::<SemiFredholm>();
        assert_fredholm::<NotFredholm>();
    }

    #[test]
    fn test_phantom_types_are_copy() {
        let c1 = Complete;
        let c2 = c1;
        assert_eq!(c1, c2);

        let b1 = Bounded;
        let b2 = b1;
        assert_eq!(b1, b2);
    }

    #[test]
    fn test_properties_wrapper() {
        let props: Properties<CompactSelfAdjointOperator> = Properties::new();
        assert_eq!(size_of_val(&props), 0);
    }
}
