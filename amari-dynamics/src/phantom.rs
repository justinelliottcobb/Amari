//! Phantom types for compile-time verification of dynamical system properties
//!
//! This module provides zero-cost type markers using the sealed trait pattern
//! to enable compile-time verification of dynamical system properties such as:
//!
//! - Time dependence (autonomous vs non-autonomous)
//! - Flow continuity (continuous vs discrete)
//! - Stability (verified stable, unstable, or unknown)
//! - Chaos (verified chaotic, regular, or unknown)
//!
//! # Example
//!
//! ```ignore
//! use amari_dynamics::phantom::*;
//!
//! // A system verified to be autonomous and stable
//! let system: TypedSystem<MySystem, Autonomous, ContinuousTime, Stable, Regular> = ...;
//!
//! // Functions can require specific properties
//! fn analyze_stable<S, T, C, Ch>(sys: &TypedSystem<S, T, C, Stable, Ch>) { ... }
//! ```

use core::marker::PhantomData;

// ============================================================================
// Sealed Trait Pattern
// ============================================================================

mod private {
    /// Sealed trait to prevent external implementations
    pub trait Sealed {}
}

// ============================================================================
// Time Dependence Properties
// ============================================================================

/// Marker trait for time dependence of dynamical systems
pub trait TimeDependence: private::Sealed + Clone + Copy + Default + 'static {
    /// Returns true if the system is autonomous (time-independent)
    fn is_autonomous() -> bool;
}

/// Autonomous system: dx/dt = f(x)
///
/// The vector field does not explicitly depend on time.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Autonomous;

impl private::Sealed for Autonomous {}
impl TimeDependence for Autonomous {
    fn is_autonomous() -> bool {
        true
    }
}

/// Non-autonomous system: dx/dt = f(x, t)
///
/// The vector field explicitly depends on time.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct NonAutonomous;

impl private::Sealed for NonAutonomous {}
impl TimeDependence for NonAutonomous {
    fn is_autonomous() -> bool {
        false
    }
}

// ============================================================================
// Flow Continuity Properties
// ============================================================================

/// Marker trait for flow continuity (continuous vs discrete time)
pub trait ContinuityProperty: private::Sealed + Clone + Copy + Default + 'static {
    /// Returns true if the system evolves in continuous time
    fn is_continuous() -> bool;
}

/// Continuous-time dynamical system: dx/dt = f(x)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct ContinuousTime;

impl private::Sealed for ContinuousTime {}
impl ContinuityProperty for ContinuousTime {
    fn is_continuous() -> bool {
        true
    }
}

/// Discrete-time dynamical system (map): x_{n+1} = f(x_n)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct DiscreteTime;

impl private::Sealed for DiscreteTime {}
impl ContinuityProperty for DiscreteTime {
    fn is_continuous() -> bool {
        false
    }
}

// ============================================================================
// Stability Properties
// ============================================================================

/// Marker trait for stability verification status
pub trait StabilityProperty: private::Sealed + Clone + Copy + Default + 'static {
    /// Returns true if stability has been verified
    fn is_verified() -> bool;

    /// Returns Some(true) if stable, Some(false) if unstable, None if unknown
    fn stability_status() -> Option<bool>;
}

/// System verified to be stable (all eigenvalues have negative real parts)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Stable;

impl private::Sealed for Stable {}
impl StabilityProperty for Stable {
    fn is_verified() -> bool {
        true
    }

    fn stability_status() -> Option<bool> {
        Some(true)
    }
}

/// System verified to be unstable (at least one eigenvalue has positive real part)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Unstable;

impl private::Sealed for Unstable {}
impl StabilityProperty for Unstable {
    fn is_verified() -> bool {
        true
    }

    fn stability_status() -> Option<bool> {
        Some(false)
    }
}

/// Stability has not been verified
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct UnknownStability;

impl private::Sealed for UnknownStability {}
impl StabilityProperty for UnknownStability {
    fn is_verified() -> bool {
        false
    }

    fn stability_status() -> Option<bool> {
        None
    }
}

// ============================================================================
// Chaos Properties
// ============================================================================

/// Marker trait for chaos verification status
pub trait ChaosProperty: private::Sealed + Clone + Copy + Default + 'static {
    /// Returns true if chaos status has been verified
    fn is_verified() -> bool;

    /// Returns Some(true) if chaotic, Some(false) if regular, None if unknown
    fn chaos_status() -> Option<bool>;
}

/// System verified to be chaotic (positive maximal Lyapunov exponent)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Chaotic;

impl private::Sealed for Chaotic {}
impl ChaosProperty for Chaotic {
    fn is_verified() -> bool {
        true
    }

    fn chaos_status() -> Option<bool> {
        Some(true)
    }
}

/// System verified to be regular (non-chaotic, non-positive Lyapunov exponents)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Regular;

impl private::Sealed for Regular {}
impl ChaosProperty for Regular {
    fn is_verified() -> bool {
        true
    }

    fn chaos_status() -> Option<bool> {
        Some(false)
    }
}

/// Chaos status has not been verified
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct UnknownChaos;

impl private::Sealed for UnknownChaos {}
impl ChaosProperty for UnknownChaos {
    fn is_verified() -> bool {
        false
    }

    fn chaos_status() -> Option<bool> {
        None
    }
}

// ============================================================================
// Dimension Properties
// ============================================================================

/// Marker trait for system dimension categories
pub trait DimensionCategory: private::Sealed + Clone + Copy + Default + 'static {
    /// Returns the dimension if known at compile time
    fn dimension() -> Option<usize>;
}

/// One-dimensional system (scalar ODE)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct OneDimensional;

impl private::Sealed for OneDimensional {}
impl DimensionCategory for OneDimensional {
    fn dimension() -> Option<usize> {
        Some(1)
    }
}

/// Two-dimensional system (planar dynamical system)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct TwoDimensional;

impl private::Sealed for TwoDimensional {}
impl DimensionCategory for TwoDimensional {
    fn dimension() -> Option<usize> {
        Some(2)
    }
}

/// Three-dimensional system
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct ThreeDimensional;

impl private::Sealed for ThreeDimensional {}
impl DimensionCategory for ThreeDimensional {
    fn dimension() -> Option<usize> {
        Some(3)
    }
}

/// Higher-dimensional system (dimension > 3 or unknown)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct HighDimensional;

impl private::Sealed for HighDimensional {}
impl DimensionCategory for HighDimensional {
    fn dimension() -> Option<usize> {
        None
    }
}

// ============================================================================
// Reversibility Properties
// ============================================================================

/// Marker trait for system reversibility
pub trait ReversibilityProperty: private::Sealed + Clone + Copy + Default + 'static {
    /// Returns true if the system is time-reversible
    fn is_reversible() -> bool;
}

/// System is time-reversible (Hamiltonian systems, etc.)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Reversible;

impl private::Sealed for Reversible {}
impl ReversibilityProperty for Reversible {
    fn is_reversible() -> bool {
        true
    }
}

/// System is not time-reversible (dissipative systems)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Irreversible;

impl private::Sealed for Irreversible {}
impl ReversibilityProperty for Irreversible {
    fn is_reversible() -> bool {
        false
    }
}

/// Reversibility is unknown
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct UnknownReversibility;

impl private::Sealed for UnknownReversibility {}
impl ReversibilityProperty for UnknownReversibility {
    fn is_reversible() -> bool {
        false // Conservative default
    }
}

// ============================================================================
// Typed System Wrapper
// ============================================================================

/// Typed dynamical system with compile-time property verification
///
/// This wrapper attaches phantom type markers to a dynamical system,
/// enabling compile-time verification of system properties.
///
/// # Type Parameters
///
/// - `S`: The underlying system type
/// - `Time`: Time dependence marker ([`Autonomous`] or [`NonAutonomous`])
/// - `Cont`: Continuity marker ([`ContinuousTime`] or [`DiscreteTime`])
/// - `Stab`: Stability marker ([`Stable`], [`Unstable`], or [`UnknownStability`])
/// - `Chaos`: Chaos marker ([`Chaotic`], [`Regular`], or [`UnknownChaos`])
#[derive(Debug, Clone)]
pub struct TypedSystem<S, Time, Cont, Stab, Chaos>
where
    Time: TimeDependence,
    Cont: ContinuityProperty,
    Stab: StabilityProperty,
    Chaos: ChaosProperty,
{
    /// The underlying dynamical system
    pub system: S,
    _time: PhantomData<Time>,
    _continuity: PhantomData<Cont>,
    _stability: PhantomData<Stab>,
    _chaos: PhantomData<Chaos>,
}

impl<S, Time, Cont, Stab, Chaos> TypedSystem<S, Time, Cont, Stab, Chaos>
where
    Time: TimeDependence,
    Cont: ContinuityProperty,
    Stab: StabilityProperty,
    Chaos: ChaosProperty,
{
    /// Create a new typed system wrapper
    pub fn new(system: S) -> Self {
        Self {
            system,
            _time: PhantomData,
            _continuity: PhantomData,
            _stability: PhantomData,
            _chaos: PhantomData,
        }
    }

    /// Get a reference to the underlying system
    pub fn inner(&self) -> &S {
        &self.system
    }

    /// Get a mutable reference to the underlying system
    pub fn inner_mut(&mut self) -> &mut S {
        &mut self.system
    }

    /// Consume the wrapper and return the underlying system
    pub fn into_inner(self) -> S {
        self.system
    }

    /// Check if the system is autonomous
    pub fn is_autonomous(&self) -> bool {
        Time::is_autonomous()
    }

    /// Check if the system is continuous-time
    pub fn is_continuous(&self) -> bool {
        Cont::is_continuous()
    }

    /// Check if stability has been verified
    pub fn stability_verified(&self) -> bool {
        Stab::is_verified()
    }

    /// Check if chaos status has been verified
    pub fn chaos_verified(&self) -> bool {
        Chaos::is_verified()
    }
}

// ============================================================================
// Type Aliases for Common System Types
// ============================================================================

/// Autonomous continuous-time system with unknown properties
pub type AutonomousContinuousSystem<S> =
    TypedSystem<S, Autonomous, ContinuousTime, UnknownStability, UnknownChaos>;

/// Autonomous discrete-time map with unknown properties
pub type AutonomousDiscreteMap<S> =
    TypedSystem<S, Autonomous, DiscreteTime, UnknownStability, UnknownChaos>;

/// System verified to be stable and non-chaotic
pub type VerifiedStableSystem<S> = TypedSystem<S, Autonomous, ContinuousTime, Stable, Regular>;

/// System verified to be chaotic
pub type VerifiedChaoticSystem<S> = TypedSystem<S, Autonomous, ContinuousTime, Unstable, Chaotic>;

/// Non-autonomous continuous-time system
pub type NonAutonomousContinuousSystem<S> =
    TypedSystem<S, NonAutonomous, ContinuousTime, UnknownStability, UnknownChaos>;

// ============================================================================
// Properties Wrapper (for grouping multiple properties)
// ============================================================================

/// Zero-sized wrapper for grouping phantom type properties
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Properties<T>(PhantomData<T>);

impl<T> Properties<T> {
    /// Create a new properties wrapper
    pub const fn new() -> Self {
        Self(PhantomData)
    }
}

/// Common property combinations for stable continuous systems
pub type StableContinuousProperties = (Autonomous, ContinuousTime, Stable, Regular);
/// Common property combinations for chaotic continuous systems
pub type ChaoticContinuousProperties = (Autonomous, ContinuousTime, Unstable, Chaotic);
/// Common property combinations for unverified continuous systems
pub type UnverifiedContinuousProperties =
    (Autonomous, ContinuousTime, UnknownStability, UnknownChaos);

#[cfg(test)]
mod tests {
    use super::*;
    use core::mem::size_of;

    #[test]
    fn test_phantom_types_are_zero_sized() {
        assert_eq!(size_of::<Autonomous>(), 0);
        assert_eq!(size_of::<NonAutonomous>(), 0);
        assert_eq!(size_of::<ContinuousTime>(), 0);
        assert_eq!(size_of::<DiscreteTime>(), 0);
        assert_eq!(size_of::<Stable>(), 0);
        assert_eq!(size_of::<Unstable>(), 0);
        assert_eq!(size_of::<UnknownStability>(), 0);
        assert_eq!(size_of::<Chaotic>(), 0);
        assert_eq!(size_of::<Regular>(), 0);
        assert_eq!(size_of::<UnknownChaos>(), 0);
        assert_eq!(size_of::<Properties<StableContinuousProperties>>(), 0);
    }

    #[test]
    fn test_time_dependence() {
        assert!(Autonomous::is_autonomous());
        assert!(!NonAutonomous::is_autonomous());
    }

    #[test]
    fn test_continuity() {
        assert!(ContinuousTime::is_continuous());
        assert!(!DiscreteTime::is_continuous());
    }

    #[test]
    fn test_stability_status() {
        assert_eq!(Stable::stability_status(), Some(true));
        assert_eq!(Unstable::stability_status(), Some(false));
        assert_eq!(UnknownStability::stability_status(), None);
    }

    #[test]
    fn test_chaos_status() {
        assert_eq!(Chaotic::chaos_status(), Some(true));
        assert_eq!(Regular::chaos_status(), Some(false));
        assert_eq!(UnknownChaos::chaos_status(), None);
    }

    #[test]
    fn test_dimension_category() {
        assert_eq!(OneDimensional::dimension(), Some(1));
        assert_eq!(TwoDimensional::dimension(), Some(2));
        assert_eq!(ThreeDimensional::dimension(), Some(3));
        assert_eq!(HighDimensional::dimension(), None);
    }

    #[test]
    fn test_typed_system() {
        struct DummySystem;

        let typed: AutonomousContinuousSystem<DummySystem> = TypedSystem::new(DummySystem);

        assert!(typed.is_autonomous());
        assert!(typed.is_continuous());
        assert!(!typed.stability_verified());
        assert!(!typed.chaos_verified());
    }

    #[test]
    fn test_typed_system_verified() {
        struct DummySystem;

        let typed: VerifiedStableSystem<DummySystem> = TypedSystem::new(DummySystem);

        assert!(typed.is_autonomous());
        assert!(typed.is_continuous());
        assert!(typed.stability_verified());
        assert!(typed.chaos_verified());
    }
}
