//! Phantom types for compile-time verification of enumerative geometry properties.
//!
//! This module provides zero-cost phantom types that encode mathematical properties
//! of partitions, tableaux, and Schubert classes at the type level.
//!
//! # Partition Validity
//!
//! - [`ValidPartition`]: Partition has been validated (weakly decreasing, positive parts)
//! - [`UnvalidatedPartition`]: Partition has not been validated
//!
//! # Tableau Properties
//!
//! - [`Semistandard`]: Tableau satisfies semistandard conditions
//! - [`LatticeWord`]: Tableau satisfies lattice word condition
//!
//! # Capability States
//!
//! - [`Granted`]: Capability has been granted to a namespace
//! - [`Pending`]: Capability is pending grant approval
//!
//! # Intersection States
//!
//! - [`Transverse`]: Intersection is transverse (codimensions sum to dimension)
//! - [`Excess`]: Intersection is overdetermined

use core::marker::PhantomData;

// Sealed trait pattern to prevent external implementations
mod private {
    pub trait Sealed {}
}

// ============================================================================
// Partition Validity Properties
// ============================================================================

/// Marker trait for partition validity states.
pub trait PartitionValidity: private::Sealed + Clone + Copy + Default + 'static {}

/// Valid partition: weakly decreasing positive integers.
///
/// A partition λ = (λ_1, λ_2, ..., λ_k) where λ_1 ≥ λ_2 ≥ ... ≥ λ_k > 0.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct ValidPartition;

impl private::Sealed for ValidPartition {}
impl PartitionValidity for ValidPartition {}

/// Unvalidated partition: may not satisfy partition constraints.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct UnvalidatedPartition;

impl private::Sealed for UnvalidatedPartition {}
impl PartitionValidity for UnvalidatedPartition {}

// ============================================================================
// Tableau Properties
// ============================================================================

/// Marker trait for tableau validity states.
pub trait TableauValidity: private::Sealed + Clone + Copy + Default + 'static {}

/// Semistandard tableau: rows weakly increasing, columns strictly increasing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Semistandard;

impl private::Sealed for Semistandard {}
impl TableauValidity for Semistandard {}

/// Tableau satisfies the lattice word (Yamanouchi) condition.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct LatticeWord;

impl private::Sealed for LatticeWord {}
impl TableauValidity for LatticeWord {}

/// Unverified tableau: validity not yet checked.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct UnverifiedTableau;

impl private::Sealed for UnverifiedTableau {}
impl TableauValidity for UnverifiedTableau {}

// ============================================================================
// Capability Grant States
// ============================================================================

/// Marker trait for capability grant states.
pub trait GrantState: private::Sealed + Clone + Copy + Default + 'static {}

/// Capability has been granted and is active.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Granted;

impl private::Sealed for Granted {}
impl GrantState for Granted {}

/// Capability is pending grant (dependencies not yet satisfied).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Pending;

impl private::Sealed for Pending {}
impl GrantState for Pending {}

/// Capability is revoked.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Revoked;

impl private::Sealed for Revoked {}
impl GrantState for Revoked {}

// ============================================================================
// Intersection Properties
// ============================================================================

/// Marker trait for intersection dimension properties.
pub trait IntersectionDimension: private::Sealed + Clone + Copy + Default + 'static {}

/// Transverse intersection: codimensions sum to ambient dimension.
///
/// The expected number of intersection points is finite.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Transverse;

impl private::Sealed for Transverse {}
impl IntersectionDimension for Transverse {}

/// Excess intersection: codimensions sum exceeds ambient dimension.
///
/// The intersection is generically empty.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Excess;

impl private::Sealed for Excess {}
impl IntersectionDimension for Excess {}

/// Deficient intersection: codimensions sum is less than ambient dimension.
///
/// The intersection has positive dimension.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Deficient;

impl private::Sealed for Deficient {}
impl IntersectionDimension for Deficient {}

/// Unknown intersection dimension (not yet computed).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct UnknownDimension;

impl private::Sealed for UnknownDimension {}
impl IntersectionDimension for UnknownDimension {}

// ============================================================================
// Grassmannian Containment
// ============================================================================

/// Marker trait for partition containment in a Grassmannian box.
pub trait BoxContainment: private::Sealed + Clone + Copy + Default + 'static {}

/// Partition fits in k × (n-k) box.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct FitsInBox;

impl private::Sealed for FitsInBox {}
impl BoxContainment for FitsInBox {}

/// Partition may not fit in box (not verified).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct UnverifiedBox;

impl private::Sealed for UnverifiedBox {}
impl BoxContainment for UnverifiedBox {}

// ============================================================================
// Type Aliases for Common Combinations
// ============================================================================

/// A valid LR tableau (semistandard + lattice word condition).
pub type ValidLRTableau = (Semistandard, LatticeWord);

/// A Schubert class verified to fit in a Grassmannian.
pub type ValidSchubertClass = (ValidPartition, FitsInBox);

/// Properties wrapper for zero-cost phantom data.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Properties<T>(PhantomData<T>);

impl<T> Properties<T> {
    /// Create a new properties marker.
    #[must_use]
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
        assert_eq!(size_of::<ValidPartition>(), 0);
        assert_eq!(size_of::<UnvalidatedPartition>(), 0);
        assert_eq!(size_of::<Semistandard>(), 0);
        assert_eq!(size_of::<LatticeWord>(), 0);
        assert_eq!(size_of::<Granted>(), 0);
        assert_eq!(size_of::<Pending>(), 0);
        assert_eq!(size_of::<Transverse>(), 0);
        assert_eq!(size_of::<FitsInBox>(), 0);
        assert_eq!(size_of::<Properties<ValidLRTableau>>(), 0);
    }

    #[test]
    fn test_phantom_types_implement_traits() {
        fn assert_partition_validity<T: PartitionValidity>() {}
        fn assert_tableau_validity<T: TableauValidity>() {}
        fn assert_grant_state<T: GrantState>() {}
        fn assert_intersection_dim<T: IntersectionDimension>() {}
        fn assert_box_containment<T: BoxContainment>() {}

        assert_partition_validity::<ValidPartition>();
        assert_partition_validity::<UnvalidatedPartition>();

        assert_tableau_validity::<Semistandard>();
        assert_tableau_validity::<LatticeWord>();
        assert_tableau_validity::<UnverifiedTableau>();

        assert_grant_state::<Granted>();
        assert_grant_state::<Pending>();
        assert_grant_state::<Revoked>();

        assert_intersection_dim::<Transverse>();
        assert_intersection_dim::<Excess>();
        assert_intersection_dim::<Deficient>();

        assert_box_containment::<FitsInBox>();
        assert_box_containment::<UnverifiedBox>();
    }

    #[test]
    fn test_phantom_types_are_copy() {
        let v1 = ValidPartition;
        let v2 = v1;
        assert_eq!(v1, v2);

        let s1 = Semistandard;
        let s2 = s1;
        assert_eq!(s1, s2);
    }
}
