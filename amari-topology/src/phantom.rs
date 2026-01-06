//! Phantom types for compile-time verification of topological properties.
//!
//! This module provides zero-cost phantom types that encode topological properties
//! at compile time, enabling the type system to enforce mathematical invariants
//! and prevent invalid operations.
//!
//! # Design Philosophy
//!
//! Following the Amari phantom types methodology:
//! - Zero runtime cost (all markers are zero-sized types)
//! - Compile-time verification of topological properties
//! - Type-level encoding of mathematical structure
//! - Prevention of invalid topological operations
//!
//! # Property Categories
//!
//! ## Orientation Properties
//!
//! - [`Oriented`]: Simplex/complex has consistent orientation
//! - [`Unoriented`]: No orientation tracking
//!
//! ## Coefficient Ring Properties
//!
//! - [`IntegerCoefficients`]: Homology over ℤ
//! - [`Mod2Coefficients`]: Homology over ℤ/2ℤ (useful for non-orientable spaces)
//! - [`RealCoefficients`]: Homology over ℝ
//!
//! ## Complex Properties
//!
//! - [`Closed`]: Complex has no boundary (∂K = ∅)
//! - [`WithBoundary`]: Complex may have boundary
//! - [`Connected`]: Complex is path-connected
//! - [`Disconnected`]: Complex may have multiple components
//!
//! ## Filtration Properties
//!
//! - [`Validated`]: Filtration has been validated (faces precede simplices)
//! - [`Unvalidated`]: Filtration validity not yet checked

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use core::marker::PhantomData;

// Sealed trait pattern to prevent external implementations
mod private {
    pub trait Sealed {}
}

// ============================================================================
// Orientation Properties
// ============================================================================

/// Marker trait for orientation properties.
pub trait OrientationProperty: private::Sealed + Clone + Copy + Default + 'static {}

/// Oriented simplex/complex: consistent orientation is tracked.
///
/// For simplices, this means the ordering of vertices determines a sign.
/// For complexes, adjacent simplices have compatible orientations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Oriented;

impl private::Sealed for Oriented {}
impl OrientationProperty for Oriented {}

/// Unoriented simplex/complex: no orientation tracking.
///
/// Useful when orientation doesn't matter or for ℤ/2ℤ coefficients.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Unoriented;

impl private::Sealed for Unoriented {}
impl OrientationProperty for Unoriented {}

// ============================================================================
// Coefficient Ring Properties
// ============================================================================

/// Marker trait for homology coefficient rings.
pub trait CoefficientRing: private::Sealed + Clone + Copy + Default + 'static {
    /// The scalar type for this coefficient ring.
    type Scalar: Clone + Default;

    /// Zero element of the ring.
    fn zero() -> Self::Scalar;

    /// One element of the ring.
    fn one() -> Self::Scalar;

    /// Negate an element.
    fn negate(x: Self::Scalar) -> Self::Scalar;

    /// Add two elements.
    fn add(x: Self::Scalar, y: Self::Scalar) -> Self::Scalar;

    /// Multiply two elements.
    fn mul(x: Self::Scalar, y: Self::Scalar) -> Self::Scalar;
}

/// Integer coefficients ℤ for standard homology.
///
/// Most commonly used coefficient ring. Captures torsion information.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct IntegerCoefficients;

impl private::Sealed for IntegerCoefficients {}
impl CoefficientRing for IntegerCoefficients {
    type Scalar = i64;

    #[inline]
    fn zero() -> i64 {
        0
    }

    #[inline]
    fn one() -> i64 {
        1
    }

    #[inline]
    fn negate(x: i64) -> i64 {
        -x
    }

    #[inline]
    fn add(x: i64, y: i64) -> i64 {
        x.wrapping_add(y)
    }

    #[inline]
    fn mul(x: i64, y: i64) -> i64 {
        x.wrapping_mul(y)
    }
}

/// Mod 2 coefficients ℤ/2ℤ for homology.
///
/// Useful for:
/// - Non-orientable manifolds
/// - Simpler boundary computations (no sign tracking)
/// - Persistent homology with field coefficients
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Mod2Coefficients;

impl private::Sealed for Mod2Coefficients {}
impl CoefficientRing for Mod2Coefficients {
    type Scalar = u8;

    #[inline]
    fn zero() -> u8 {
        0
    }

    #[inline]
    fn one() -> u8 {
        1
    }

    #[inline]
    fn negate(x: u8) -> u8 {
        x
    } // -1 ≡ 1 (mod 2)

    #[inline]
    fn add(x: u8, y: u8) -> u8 {
        (x + y) % 2
    }

    #[inline]
    fn mul(x: u8, y: u8) -> u8 {
        (x * y) % 2
    }
}

/// Real coefficients ℝ for homology.
///
/// Used when torsion information is not needed.
/// Simplifies computations and allows real-valued chains.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct RealCoefficients;

impl private::Sealed for RealCoefficients {}
impl CoefficientRing for RealCoefficients {
    type Scalar = f64;

    #[inline]
    fn zero() -> f64 {
        0.0
    }

    #[inline]
    fn one() -> f64 {
        1.0
    }

    #[inline]
    fn negate(x: f64) -> f64 {
        -x
    }

    #[inline]
    fn add(x: f64, y: f64) -> f64 {
        x + y
    }

    #[inline]
    fn mul(x: f64, y: f64) -> f64 {
        x * y
    }
}

// ============================================================================
// Complex Boundary Properties
// ============================================================================

/// Marker trait for complex boundary properties.
pub trait BoundaryProperty: private::Sealed + Clone + Copy + Default + 'static {}

/// Closed complex: has no boundary (∂K = ∅).
///
/// Examples: spheres, tori, closed manifolds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Closed;

impl private::Sealed for Closed {}
impl BoundaryProperty for Closed {}

/// Complex with boundary: may have non-empty boundary.
///
/// Examples: disks, cylinders, manifolds with boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct WithBoundary;

impl private::Sealed for WithBoundary {}
impl BoundaryProperty for WithBoundary {}

// ============================================================================
// Connectivity Properties
// ============================================================================

/// Marker trait for connectivity properties.
pub trait ConnectivityProperty: private::Sealed + Clone + Copy + Default + 'static {}

/// Connected complex: path-connected (β₀ = 1).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Connected;

impl private::Sealed for Connected {}
impl ConnectivityProperty for Connected {}

/// Possibly disconnected complex.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Disconnected;

impl private::Sealed for Disconnected {}
impl ConnectivityProperty for Disconnected {}

// ============================================================================
// Filtration Validation Properties
// ============================================================================

/// Marker trait for filtration validation state.
pub trait ValidationProperty: private::Sealed + Clone + Copy + Default + 'static {}

/// Filtration has been validated: faces precede their cofaces.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Validated;

impl private::Sealed for Validated {}
impl ValidationProperty for Validated {}

/// Filtration validity not yet verified.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Unvalidated;

impl private::Sealed for Unvalidated {}
impl ValidationProperty for Unvalidated {}

// ============================================================================
// Type-Safe Wrappers
// ============================================================================

/// A typed simplex with compile-time orientation tracking.
#[derive(Clone, Debug)]
pub struct TypedSimplex<O: OrientationProperty = Oriented> {
    pub(crate) vertices: Vec<usize>,
    pub(crate) orientation: i8,
    pub(crate) _marker: PhantomData<O>,
}

impl<O: OrientationProperty> TypedSimplex<O> {
    /// Get the vertices.
    #[inline]
    pub fn vertices(&self) -> &[usize] {
        &self.vertices
    }

    /// Get the dimension.
    #[inline]
    pub fn dimension(&self) -> usize {
        if self.vertices.is_empty() {
            0
        } else {
            self.vertices.len() - 1
        }
    }
}

impl TypedSimplex<Oriented> {
    /// Get the orientation (only available for oriented simplices).
    #[inline]
    pub fn orientation(&self) -> i8 {
        self.orientation
    }

    /// Negate the orientation.
    pub fn negate(&self) -> Self {
        Self {
            vertices: self.vertices.clone(),
            orientation: -self.orientation,
            _marker: PhantomData,
        }
    }
}

impl TypedSimplex<Unoriented> {
    /// Convert to oriented simplex with default orientation.
    pub fn with_orientation(self) -> TypedSimplex<Oriented> {
        TypedSimplex {
            vertices: self.vertices,
            orientation: 1,
            _marker: PhantomData,
        }
    }
}

/// A typed chain with compile-time coefficient ring tracking.
#[derive(Clone, Debug)]
pub struct TypedChain<R: CoefficientRing = IntegerCoefficients> {
    pub(crate) coefficients: Vec<(usize, R::Scalar)>,
    pub(crate) _marker: PhantomData<R>,
}

impl<R: CoefficientRing> TypedChain<R> {
    /// Create the zero chain.
    pub fn zero() -> Self {
        Self {
            coefficients: Vec::new(),
            _marker: PhantomData,
        }
    }

    /// Check if this is the zero chain.
    pub fn is_zero(&self) -> bool {
        self.coefficients.is_empty()
    }

    /// Number of non-zero terms.
    pub fn support_size(&self) -> usize {
        self.coefficients.len()
    }
}

/// A typed filtration with validation state tracking.
#[derive(Clone, Debug)]
pub struct TypedFiltration<V: ValidationProperty = Unvalidated> {
    pub(crate) simplices: Vec<(f64, Vec<usize>)>,
    pub(crate) _marker: PhantomData<V>,
}

impl TypedFiltration<Unvalidated> {
    /// Create a new unvalidated filtration.
    pub fn new() -> Self {
        Self {
            simplices: Vec::new(),
            _marker: PhantomData,
        }
    }

    /// Add a simplex at the given filtration time.
    pub fn add(&mut self, time: f64, vertices: Vec<usize>) {
        self.simplices.push((time, vertices));
    }

    /// Validate the filtration, returning a validated version if valid.
    ///
    /// Returns `None` if validation fails (face appears after coface).
    pub fn validate(mut self) -> Option<TypedFiltration<Validated>> {
        // Sort by time
        self.simplices
            .sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(core::cmp::Ordering::Equal));

        // Check that all faces appear before their cofaces
        // (simplified validation - full implementation would check all faces)
        for (i, (t_i, verts_i)) in self.simplices.iter().enumerate() {
            for (t_j, verts_j) in self.simplices.iter().skip(i + 1) {
                // If j is a face of i and appears later, invalid
                if verts_j.len() < verts_i.len()
                    && verts_j.iter().all(|v| verts_i.contains(v))
                    && t_j > t_i
                {
                    return None;
                }
            }
        }

        Some(TypedFiltration {
            simplices: self.simplices,
            _marker: PhantomData,
        })
    }
}

impl Default for TypedFiltration<Unvalidated> {
    fn default() -> Self {
        Self::new()
    }
}

impl TypedFiltration<Validated> {
    /// Get the filtration times (only available for validated filtrations).
    pub fn times(&self) -> impl Iterator<Item = f64> + '_ {
        self.simplices.iter().map(|(t, _)| *t)
    }

    /// Get the number of simplices.
    pub fn len(&self) -> usize {
        self.simplices.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.simplices.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_coefficient_rings() {
        // Integer coefficients
        assert_eq!(IntegerCoefficients::zero(), 0);
        assert_eq!(IntegerCoefficients::one(), 1);
        assert_eq!(IntegerCoefficients::negate(5), -5);
        assert_eq!(IntegerCoefficients::add(3, 4), 7);

        // Mod 2 coefficients
        assert_eq!(Mod2Coefficients::zero(), 0);
        assert_eq!(Mod2Coefficients::one(), 1);
        assert_eq!(Mod2Coefficients::negate(1), 1); // -1 ≡ 1 (mod 2)
        assert_eq!(Mod2Coefficients::add(1, 1), 0); // 1 + 1 ≡ 0 (mod 2)

        // Real coefficients
        assert_eq!(RealCoefficients::zero(), 0.0);
        assert_eq!(RealCoefficients::one(), 1.0);
    }

    #[test]
    fn test_typed_filtration_validation() {
        let mut filt = TypedFiltration::new();
        filt.add(0.0, vec![0]);
        filt.add(0.0, vec![1]);
        filt.add(1.0, vec![0, 1]);

        // This should validate successfully (vertices before edges)
        let validated = filt.validate();
        assert!(validated.is_some());
    }

    #[test]
    fn test_invalid_filtration() {
        let mut filt = TypedFiltration::new();
        filt.add(1.0, vec![0, 1]); // Edge appears at t=1
        filt.add(2.0, vec![0]); // Vertex appears at t=2 (invalid!)

        // This should fail validation
        let validated = filt.validate();
        assert!(validated.is_none());
    }
}
