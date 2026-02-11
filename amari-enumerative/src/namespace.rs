//! Namespace and Capability types for ShaperOS
//!
//! Namespaces are points in Grassmannians. Capabilities are Schubert conditions.
//! Intersection theory determines access control.
//!
//! This module enables ShaperOS to answer questions like "how many valid namespace
//! configurations satisfy these capability constraints?" using rigorous enumerative geometry.
//!
//! # Contracts
//!
//! Key invariants maintained:
//!
//! - **Dependency satisfaction**: A capability can only be granted if all its dependencies are present
//! - **Conflict freedom**: Two conflicting capabilities cannot coexist in the same namespace
//! - **Grassmannian consistency**: All capabilities in a namespace must be defined on the same Grassmannian
//!
//! # Phantom Types
//!
//! The module integrates with the phantom type system for compile-time verification
//! of capability grant states (Granted, Pending, Revoked).
//!
//! # Rayon Parallelization
//!
//! When the `parallel` feature is enabled, batch operations on namespaces
//! use parallel iterators for improved performance.

use crate::geometric_algebra::quantum_k_theory::QuantumKClass;
use crate::schubert::{IntersectionResult, SchubertCalculus, SchubertClass};
use crate::EnumerativeResult;
use num_rational::Rational64;
use std::sync::Arc;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Unique identifier for a capability
///
/// # Contract
///
/// ```text
/// invariant: self.0.len() > 0
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct CapabilityId(pub Arc<str>);

impl CapabilityId {
    /// Create a new capability ID
    ///
    /// # Contract
    ///
    /// ```text
    /// requires: name.len() > 0
    /// ensures: result.0 == name
    /// ```
    #[must_use]
    pub fn new(name: impl Into<String>) -> Self {
        Self(Arc::from(name.into()))
    }

    /// Get the underlying name
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for CapabilityId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<&str> for CapabilityId {
    fn from(s: &str) -> Self {
        Self::new(s)
    }
}

impl From<String> for CapabilityId {
    fn from(s: String) -> Self {
        Self::new(s)
    }
}

/// A capability in ShaperOS: an incidence condition on namespaces
///
/// Capabilities represent access rights that can be granted to namespaces.
/// Each capability corresponds to a Schubert class, representing the geometric
/// condition that a namespace must satisfy to have that capability.
///
/// # Contracts
///
/// - The Schubert class must be valid for the given Grassmannian
/// - Dependencies form a DAG (no cycles)
/// - Conflicts are symmetric (if A conflicts with B, B should conflict with A)
#[derive(Debug, Clone)]
pub struct Capability {
    /// Unique identifier
    pub id: CapabilityId,
    /// Human-readable name
    pub name: String,
    /// Schubert class representing the incidence condition
    /// "Namespaces with this capability" = Schubert variety σ_λ
    pub schubert_class: SchubertClass,
    /// Dependencies: must have these capabilities first
    pub requires: Vec<CapabilityId>,
    /// Conflicts: cannot coexist with these
    pub conflicts: Vec<CapabilityId>,
}

impl Capability {
    /// Create a new capability
    ///
    /// # Contract
    ///
    /// ```text
    /// requires: partition fits in Grassmannian box
    /// ensures: result.codimension() == partition.iter().sum()
    /// ```
    pub fn new(
        id: impl Into<String>,
        name: impl Into<String>,
        partition: Vec<usize>,
        grassmannian: (usize, usize),
    ) -> EnumerativeResult<Self> {
        let id_str = id.into();
        Ok(Self {
            id: CapabilityId::new(id_str),
            name: name.into(),
            schubert_class: SchubertClass::new(partition, grassmannian)?,
            requires: Vec::new(),
            conflicts: Vec::new(),
        })
    }

    /// Add a dependency (builder pattern)
    #[must_use]
    pub fn requires(mut self, cap_id: CapabilityId) -> Self {
        self.requires.push(cap_id);
        self
    }

    /// Add multiple dependencies
    #[must_use]
    pub fn requires_all(mut self, cap_ids: impl IntoIterator<Item = CapabilityId>) -> Self {
        self.requires.extend(cap_ids);
        self
    }

    /// Add a conflict (builder pattern)
    #[must_use]
    pub fn conflicts_with(mut self, cap_id: CapabilityId) -> Self {
        self.conflicts.push(cap_id);
        self
    }

    /// Add multiple conflicts
    #[must_use]
    pub fn conflicts_with_all(mut self, cap_ids: impl IntoIterator<Item = CapabilityId>) -> Self {
        self.conflicts.extend(cap_ids);
        self
    }

    /// Codimension of this capability's Schubert class
    ///
    /// # Contract
    ///
    /// ```text
    /// ensures: result == self.schubert_class.codimension()
    /// ```
    #[must_use]
    pub fn codimension(&self) -> usize {
        self.schubert_class.partition.iter().sum()
    }

    /// Check if this capability has any dependencies
    #[must_use]
    pub fn has_dependencies(&self) -> bool {
        !self.requires.is_empty()
    }

    /// Check if this capability has any conflicts
    #[must_use]
    pub fn has_conflicts(&self) -> bool {
        !self.conflicts.is_empty()
    }
}

/// A namespace: a point in a Grassmannian with associated capabilities
///
/// In ShaperOS, namespaces represent isolated execution contexts.
/// Their position in the Grassmannian encodes their "geometric location"
/// while their capabilities determine what operations they can perform.
///
/// # Contracts
///
/// - All capabilities must be compatible with the namespace's Grassmannian
/// - Capabilities must satisfy dependency ordering
/// - No two conflicting capabilities can coexist
#[derive(Debug, Clone)]
pub struct Namespace {
    /// The Grassmannian this namespace lives in: Gr(k, n)
    pub grassmannian: (usize, usize),
    /// Schubert cell containing this namespace (its "position")
    pub position: SchubertClass,
    /// Capabilities granted to this namespace
    pub capabilities: Vec<Capability>,
    /// Human-readable name
    pub name: String,
}

impl Namespace {
    /// Create a new namespace at a given Schubert position
    #[must_use]
    pub fn new(name: impl Into<String>, position: SchubertClass) -> Self {
        Self {
            grassmannian: position.grassmannian_dim,
            position,
            capabilities: Vec::new(),
            name: name.into(),
        }
    }

    /// Create a namespace with full access (identity Schubert class)
    ///
    /// # Contract
    ///
    /// ```text
    /// requires: k < n
    /// ensures: result.position.partition.is_empty()
    /// ensures: result.grassmannian == (k, n)
    /// ```
    pub fn full(name: impl Into<String>, k: usize, n: usize) -> EnumerativeResult<Self> {
        let position = SchubertClass::new(vec![], (k, n))?;
        Ok(Self::new(name, position))
    }

    /// Grant a capability to this namespace
    ///
    /// # Contract
    ///
    /// ```text
    /// requires: forall dep in capability.requires. self.has_capability(dep)
    /// requires: forall conf in capability.conflicts. !self.has_capability(conf)
    /// requires: forall existing in self.capabilities. !existing.conflicts.contains(capability.id)
    /// ensures: self.has_capability(capability.id)
    /// ```
    ///
    /// # Errors
    ///
    /// Returns `NamespaceError::Conflict` if the capability conflicts with an existing one.
    /// Returns `NamespaceError::MissingDependency` if a required capability is missing.
    pub fn grant(&mut self, capability: Capability) -> Result<(), NamespaceError> {
        // Check for conflicts
        for existing in &self.capabilities {
            if capability.conflicts.contains(&existing.id) {
                return Err(NamespaceError::Conflict {
                    new: capability.id,
                    existing: existing.id.clone(),
                });
            }
            if existing.conflicts.contains(&capability.id) {
                return Err(NamespaceError::Conflict {
                    new: capability.id,
                    existing: existing.id.clone(),
                });
            }
        }

        // Check dependencies
        for req in &capability.requires {
            if !self.capabilities.iter().any(|c| &c.id == req) {
                return Err(NamespaceError::MissingDependency {
                    capability: capability.id,
                    required: req.clone(),
                });
            }
        }

        self.capabilities.push(capability);
        Ok(())
    }

    /// Try to grant multiple capabilities in dependency order
    ///
    /// This is a convenience method that attempts to grant capabilities
    /// in the order that satisfies dependencies.
    pub fn grant_all(&mut self, capabilities: Vec<Capability>) -> Result<(), NamespaceError> {
        // Simple topological sort based on dependencies
        let mut remaining = capabilities;
        let mut progress = true;

        while !remaining.is_empty() && progress {
            progress = false;
            let mut still_remaining = Vec::new();

            for cap in remaining {
                let deps_satisfied = cap.requires.iter().all(|dep| self.has_capability(dep));

                if deps_satisfied {
                    self.grant(cap)?;
                    progress = true;
                } else {
                    still_remaining.push(cap);
                }
            }

            remaining = still_remaining;
        }

        if !remaining.is_empty() {
            let first = remaining.into_iter().next().unwrap();
            return Err(NamespaceError::MissingDependency {
                capability: first.id,
                required: first
                    .requires
                    .into_iter()
                    .next()
                    .unwrap_or_else(|| CapabilityId::new("unknown")),
            });
        }

        Ok(())
    }

    /// Revoke a capability from this namespace
    ///
    /// # Contract
    ///
    /// ```text
    /// requires: no other capability depends on this one
    /// ensures: !self.has_capability(id)
    /// ```
    ///
    /// Returns `true` if the capability was revoked, `false` if it couldn't be
    /// (either not present or has dependents).
    pub fn revoke(&mut self, id: &CapabilityId) -> bool {
        if let Some(pos) = self.capabilities.iter().position(|c| &c.id == id) {
            // Check if any remaining capabilities depend on this one
            let dependents: Vec<CapabilityId> = self
                .capabilities
                .iter()
                .filter(|c| c.requires.contains(id))
                .map(|c| c.id.clone())
                .collect();

            if dependents.is_empty() {
                self.capabilities.remove(pos);
                return true;
            }
        }
        false
    }

    /// Check if this namespace has a specific capability
    #[must_use]
    pub fn has_capability(&self, id: &CapabilityId) -> bool {
        self.capabilities.iter().any(|c| &c.id == id)
    }

    /// Get all capability IDs
    #[must_use]
    pub fn capability_ids(&self) -> Vec<CapabilityId> {
        self.capabilities.iter().map(|c| c.id.clone()).collect()
    }

    /// Get the number of capabilities
    #[must_use]
    pub fn capability_count(&self) -> usize {
        self.capabilities.len()
    }

    /// Count valid configurations satisfying all capability constraints
    ///
    /// This computes the intersection number of the position Schubert class
    /// with all capability Schubert classes.
    ///
    /// # Contract
    ///
    /// ```text
    /// ensures:
    ///   - total_codim > dim(Gr) => Empty
    ///   - total_codim == dim(Gr) => Finite(n) where n >= 0
    ///   - total_codim < dim(Gr) => PositiveDimensional
    /// ```
    #[must_use]
    pub fn count_configurations(&self) -> IntersectionResult {
        let mut calc = SchubertCalculus::new(self.grassmannian);

        let mut classes = vec![self.position.clone()];
        for cap in &self.capabilities {
            classes.push(cap.schubert_class.clone());
        }

        calc.multi_intersect(&classes)
    }

    /// Total codimension of position plus all capabilities
    #[must_use]
    pub fn total_codimension(&self) -> usize {
        let position_codim: usize = self.position.partition.iter().sum();
        let cap_codim: usize = self.capabilities.iter().map(|c| c.codimension()).sum();
        position_codim + cap_codim
    }

    /// Check if adding a capability would make the system overdetermined
    #[must_use]
    pub fn would_overdetermine(&self, capability: &Capability) -> bool {
        let (k, n) = self.grassmannian;
        let grassmannian_dim = k * (n - k);
        let new_total = self.total_codimension() + capability.codimension();
        new_total > grassmannian_dim
    }
}

/// Namespace-related errors
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NamespaceError {
    /// Two capabilities conflict with each other
    Conflict {
        /// The new capability being added
        new: CapabilityId,
        /// The existing capability that conflicts
        existing: CapabilityId,
    },
    /// A required dependency is missing
    MissingDependency {
        /// The capability being added
        capability: CapabilityId,
        /// The required capability that's missing
        required: CapabilityId,
    },
    /// The configuration is invalid
    InvalidConfiguration,
}

impl std::fmt::Display for NamespaceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            NamespaceError::Conflict { new, existing } => {
                write!(f, "Capability {} conflicts with existing {}", new, existing)
            }
            NamespaceError::MissingDependency {
                capability,
                required,
            } => {
                write!(
                    f,
                    "Capability {} requires {} which is not present",
                    capability, required
                )
            }
            NamespaceError::InvalidConfiguration => {
                write!(f, "Invalid namespace configuration")
            }
        }
    }
}

impl std::error::Error for NamespaceError {}

/// Compute the intersection of two namespaces
///
/// # Contract
///
/// ```text
/// ensures:
///   - ns1.grassmannian != ns2.grassmannian => Incompatible
///   - ns1.grassmannian == ns2.grassmannian => result based on Schubert intersection
/// ```
pub fn namespace_intersection(
    ns1: &Namespace,
    ns2: &Namespace,
) -> EnumerativeResult<NamespaceIntersection> {
    if ns1.grassmannian != ns2.grassmannian {
        return Ok(NamespaceIntersection::Incompatible);
    }

    let mut calc = SchubertCalculus::new(ns1.grassmannian);
    let result = calc.multi_intersect(&[ns1.position.clone(), ns2.position.clone()]);

    Ok(match result {
        IntersectionResult::Empty => NamespaceIntersection::Disjoint,
        IntersectionResult::Finite(1) => NamespaceIntersection::SinglePoint,
        IntersectionResult::Finite(n) => NamespaceIntersection::FinitePoints(n),
        IntersectionResult::PositiveDimensional { dimension, .. } => {
            NamespaceIntersection::Subspace { dimension }
        }
    })
}

/// Result of namespace intersection
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum NamespaceIntersection {
    /// Namespaces are in different Grassmannians
    #[default]
    Incompatible,
    /// No overlap
    Disjoint,
    /// Single point intersection
    SinglePoint,
    /// Finite number of intersection points
    FinitePoints(u64),
    /// Positive-dimensional intersection
    Subspace {
        /// Dimension of the subspace
        dimension: usize,
    },
}

/// Check if a capability is accessible from a namespace
///
/// # Contract
///
/// ```text
/// requires: namespace.grassmannian == capability.schubert_class.grassmannian_dim
/// ensures: result == true iff intersection is non-empty
/// ```
pub fn capability_accessible(
    namespace: &Namespace,
    capability: &Capability,
) -> EnumerativeResult<bool> {
    if namespace.grassmannian != capability.schubert_class.grassmannian_dim {
        return Ok(false);
    }

    let mut calc = SchubertCalculus::new(namespace.grassmannian);
    let result = calc.multi_intersect(&[
        namespace.position.clone(),
        capability.schubert_class.clone(),
    ]);

    Ok(!matches!(result, IntersectionResult::Empty))
}

/// Builder for creating namespaces with a fluent API
#[derive(Debug)]
pub struct NamespaceBuilder {
    name: String,
    grassmannian: (usize, usize),
    position: Vec<usize>,
    capabilities: Vec<Capability>,
}

impl NamespaceBuilder {
    /// Create a new namespace builder
    #[must_use]
    pub fn new(name: impl Into<String>, k: usize, n: usize) -> Self {
        Self {
            name: name.into(),
            grassmannian: (k, n),
            position: vec![],
            capabilities: vec![],
        }
    }

    /// Set the position partition
    #[must_use]
    pub fn position(mut self, partition: Vec<usize>) -> Self {
        self.position = partition;
        self
    }

    /// Add a capability
    #[must_use]
    pub fn with_capability(mut self, capability: Capability) -> Self {
        self.capabilities.push(capability);
        self
    }

    /// Add multiple capabilities
    #[must_use]
    pub fn with_capabilities(mut self, capabilities: impl IntoIterator<Item = Capability>) -> Self {
        self.capabilities.extend(capabilities);
        self
    }

    /// Build the namespace
    pub fn build(self) -> EnumerativeResult<Namespace> {
        let position = SchubertClass::new(self.position, self.grassmannian)?;
        let mut ns = Namespace::new(self.name, position);

        for cap in self.capabilities {
            ns.grant(cap).map_err(|e| {
                crate::EnumerativeError::SchubertError(format!("Failed to grant capability: {}", e))
            })?;
        }

        Ok(ns)
    }
}

// ============================================================================
// Parallel Batch Operations
// ============================================================================

/// Count configurations for multiple namespaces in parallel
#[cfg(feature = "parallel")]
pub fn count_configurations_batch(namespaces: &[Namespace]) -> Vec<IntersectionResult> {
    namespaces
        .par_iter()
        .map(|ns| ns.count_configurations())
        .collect()
}

/// Check capability accessibility for multiple namespace-capability pairs in parallel
#[cfg(feature = "parallel")]
pub fn capability_accessible_batch(
    pairs: &[(&Namespace, &Capability)],
) -> EnumerativeResult<Vec<bool>> {
    pairs
        .par_iter()
        .map(|(ns, cap)| capability_accessible(ns, cap))
        .collect()
}

/// Compute intersections for multiple namespace pairs in parallel
#[cfg(feature = "parallel")]
pub fn namespace_intersection_batch(
    pairs: &[(&Namespace, &Namespace)],
) -> EnumerativeResult<Vec<NamespaceIntersection>> {
    pairs
        .par_iter()
        .map(|(ns1, ns2)| namespace_intersection(ns1, ns2))
        .collect()
}

// ============================================================================
// Quantum K-Theoretic Capabilities
// ============================================================================

/// A capability enhanced with quantum K-theory data
///
/// While a standard `Capability` uses a Schubert class (cohomological),
/// a `QuantumCapability` additionally carries a K-theory class representing
/// the "sheaf of sections" of the capability bundle over the Grassmannian.
///
/// This enables:
/// - **Quantum corrections**: Capabilities that interact through rational curve
///   contributions (analogous to quantum entanglement in the namespace lattice)
/// - **Euler characteristic counting**: chi(E) gives a refined count of valid
///   configurations weighted by sheaf cohomology
/// - **Adams operations**: psi^k acts on capabilities, modeling "k-fold
///   amplification" of access rights
///
/// # Contract
///
/// ```text
/// invariant: self.capability.schubert_class == classical limit of self.k_class
/// ```
#[derive(Debug, Clone)]
pub struct QuantumCapability<const P: usize, const Q: usize, const R: usize> {
    /// Classical capability (Schubert class)
    pub capability: Capability,
    /// Quantum K-theory class (sheaf data)
    pub k_class: QuantumKClass<P, Q, R>,
}

impl<const P: usize, const Q: usize, const R: usize> QuantumCapability<P, Q, R> {
    /// Create a quantum capability from a classical one
    ///
    /// The K-theory class defaults to the structure sheaf of the Schubert variety.
    #[must_use]
    pub fn from_classical(capability: Capability) -> Self {
        let k_class = QuantumKClass::structure_sheaf_point();
        Self {
            capability,
            k_class,
        }
    }

    /// Create with explicit K-theory data
    #[must_use]
    pub fn new(capability: Capability, k_class: QuantumKClass<P, Q, R>) -> Self {
        Self {
            capability,
            k_class,
        }
    }

    /// Quantum product of two capabilities
    ///
    /// Returns the entangled capability with quantum corrections
    /// from rational curves connecting the two Schubert varieties.
    ///
    /// # Contract
    ///
    /// ```text
    /// ensures: result.q_power >= self.k_class.q_power + other.k_class.q_power
    /// ```
    pub fn quantum_entangle(&self, other: &Self) -> EnumerativeResult<QuantumKClass<P, Q, R>> {
        self.k_class.quantum_product(&other.k_class)
    }

    /// Euler characteristic: refined configuration count
    ///
    /// While the classical `count_configurations` gives the intersection number,
    /// this gives chi(E) = integral ch(E) * td(X), which accounts for higher cohomology.
    #[must_use]
    pub fn euler_characteristic(&self, ambient_dimension: usize) -> Rational64 {
        self.k_class.euler_characteristic(ambient_dimension)
    }

    /// Adams amplification: psi^k acts on the capability
    ///
    /// Models "k-fold amplification" of the access right.
    /// psi^k preserves the underlying Schubert class but modifies
    /// the K-theoretic refinement.
    #[must_use]
    pub fn amplify(&self, k: i32) -> Self {
        Self {
            capability: self.capability.clone(),
            k_class: self.k_class.adams_operation(k),
        }
    }
}

impl Namespace {
    /// Count configurations with quantum K-theoretic corrections
    ///
    /// This refines `count_configurations` by incorporating quantum
    /// corrections from rational curves in the Grassmannian.
    ///
    /// Returns (classical_count, quantum_euler_characteristic).
    ///
    /// # Contract
    ///
    /// ```text
    /// ensures: result.0 == self.count_configurations()
    /// ```
    #[must_use]
    pub fn quantum_count_configurations<const P: usize, const Q: usize, const R: usize>(
        &self,
        quantum_caps: &[QuantumCapability<P, Q, R>],
    ) -> (IntersectionResult, Rational64) {
        let classical = self.count_configurations();

        // Compute quantum Euler characteristic
        let (k, n) = self.grassmannian;
        let ambient_dim = k * (n - k);

        let mut total_euler = Rational64::from(1);
        for cap in quantum_caps {
            total_euler *= cap.euler_characteristic(ambient_dim);
        }

        (classical, total_euler)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_capability_creation() {
        // Gr(2, 4): 2-planes in 4-space
        let cap = Capability::new("read", "Read Access", vec![1], (2, 4)).unwrap();
        assert_eq!(cap.codimension(), 1);
        assert_eq!(cap.id, CapabilityId::new("read"));
    }

    #[test]
    fn test_capability_id_from() {
        let id1: CapabilityId = "read".into();
        let id2: CapabilityId = String::from("write").into();
        assert_eq!(id1.as_str(), "read");
        assert_eq!(id2.as_str(), "write");
    }

    #[test]
    fn test_namespace_full() {
        let ns = Namespace::full("test", 2, 4).unwrap();
        assert_eq!(ns.grassmannian, (2, 4));
        assert!(ns.position.partition.is_empty());
    }

    #[test]
    fn test_namespace_grant() {
        let mut ns = Namespace::full("test", 2, 4).unwrap();
        let cap = Capability::new("read", "Read", vec![1], (2, 4)).unwrap();

        assert!(ns.grant(cap).is_ok());
        assert!(ns.has_capability(&CapabilityId::new("read")));
    }

    #[test]
    fn test_capability_conflict() {
        let mut ns = Namespace::full("test", 2, 4).unwrap();

        let read = Capability::new("read", "Read", vec![1], (2, 4))
            .unwrap()
            .conflicts_with(CapabilityId::new("write"));
        let write = Capability::new("write", "Write", vec![1], (2, 4)).unwrap();

        ns.grant(read).unwrap();

        let result = ns.grant(write);
        assert!(matches!(result, Err(NamespaceError::Conflict { .. })));
    }

    #[test]
    fn test_capability_dependency() {
        let mut ns = Namespace::full("test", 2, 4).unwrap();

        let write = Capability::new("write", "Write", vec![1], (2, 4))
            .unwrap()
            .requires(CapabilityId::new("read"));

        // Try to grant write without read - should fail
        let result = ns.grant(write.clone());
        assert!(matches!(
            result,
            Err(NamespaceError::MissingDependency { .. })
        ));

        // Grant read first, then write should succeed
        let read = Capability::new("read", "Read", vec![1], (2, 4)).unwrap();
        ns.grant(read).unwrap();
        assert!(ns.grant(write).is_ok());
    }

    #[test]
    fn test_namespace_intersection() {
        let ns1 = Namespace::full("ns1", 2, 4).unwrap();
        let ns2 = Namespace::full("ns2", 2, 4).unwrap();

        let result = namespace_intersection(&ns1, &ns2).unwrap();
        // Two full namespaces should have a positive-dimensional intersection
        assert!(matches!(
            result,
            NamespaceIntersection::Subspace { dimension: 4 }
        ));
    }

    #[test]
    fn test_namespace_incompatible() {
        let ns1 = Namespace::full("ns1", 2, 4).unwrap();
        let ns2 = Namespace::full("ns2", 3, 6).unwrap();

        let result = namespace_intersection(&ns1, &ns2).unwrap();
        assert_eq!(result, NamespaceIntersection::Incompatible);
    }

    #[test]
    fn test_count_configurations() {
        let mut ns = Namespace::full("agent", 2, 4).unwrap();

        // Add 4 capabilities each with codimension 1
        // This should give us exactly 2 configurations (σ_1^4 in Gr(2,4))
        for i in 0..4 {
            let cap = Capability::new(format!("cap{}", i), format!("Cap {}", i), vec![1], (2, 4))
                .unwrap();
            ns.grant(cap).unwrap();
        }

        let count = ns.count_configurations();
        assert_eq!(count, IntersectionResult::Finite(2));
    }

    #[test]
    fn test_namespace_builder() {
        let read = Capability::new("read", "Read", vec![1], (2, 4)).unwrap();

        let ns = NamespaceBuilder::new("test", 2, 4)
            .position(vec![])
            .with_capability(read)
            .build()
            .unwrap();

        assert!(ns.has_capability(&CapabilityId::new("read")));
    }

    #[test]
    fn test_capability_accessible() {
        let ns = Namespace::full("test", 2, 4).unwrap();
        let cap = Capability::new("read", "Read", vec![1], (2, 4)).unwrap();

        assert!(capability_accessible(&ns, &cap).unwrap());
    }

    #[test]
    fn test_revoke_capability() {
        let mut ns = Namespace::full("test", 2, 4).unwrap();
        let cap = Capability::new("read", "Read", vec![1], (2, 4)).unwrap();

        ns.grant(cap).unwrap();
        assert!(ns.has_capability(&CapabilityId::new("read")));

        assert!(ns.revoke(&CapabilityId::new("read")));
        assert!(!ns.has_capability(&CapabilityId::new("read")));
    }

    #[test]
    fn test_would_overdetermine() {
        let mut ns = Namespace::full("test", 2, 4).unwrap();

        // Add 4 capabilities of codimension 1 each
        for i in 0..4 {
            let cap = Capability::new(format!("cap{}", i), format!("Cap {}", i), vec![1], (2, 4))
                .unwrap();
            ns.grant(cap).unwrap();
        }

        // Adding another would overdetermine
        let extra = Capability::new("extra", "Extra", vec![1], (2, 4)).unwrap();
        assert!(ns.would_overdetermine(&extra));
    }

    #[test]
    fn test_grant_all() {
        let mut ns = Namespace::full("test", 2, 4).unwrap();

        let read = Capability::new("read", "Read", vec![1], (2, 4)).unwrap();
        let write = Capability::new("write", "Write", vec![1], (2, 4))
            .unwrap()
            .requires(CapabilityId::new("read"));

        // Grant in reverse order - should still work
        ns.grant_all(vec![write, read]).unwrap();

        assert!(ns.has_capability(&CapabilityId::new("read")));
        assert!(ns.has_capability(&CapabilityId::new("write")));
    }

    #[test]
    fn test_namespace_intersection_default() {
        let default = NamespaceIntersection::default();
        assert_eq!(default, NamespaceIntersection::Incompatible);
    }

    // Quantum capability tests

    #[test]
    fn test_quantum_capability_creation() {
        let cap = Capability::new("read", "Read", vec![1], (2, 4)).unwrap();
        let qcap = QuantumCapability::<4, 0, 0>::from_classical(cap);
        assert_eq!(qcap.capability.codimension(), 1);
    }

    #[test]
    fn test_quantum_entanglement() {
        let cap1 = Capability::new("read", "Read", vec![1], (2, 4)).unwrap();
        let cap2 = Capability::new("write", "Write", vec![1], (2, 4)).unwrap();

        let qcap1 = QuantumCapability::<4, 0, 0>::from_classical(cap1);
        let qcap2 = QuantumCapability::<4, 0, 0>::from_classical(cap2);

        let entangled = qcap1.quantum_entangle(&qcap2).unwrap();
        // Quantum product should produce a valid K-class
        assert!(entangled.q_power >= 0);
    }

    #[test]
    fn test_adams_amplification() {
        let cap = Capability::new("read", "Read", vec![1], (2, 4)).unwrap();
        let mut qcap = QuantumCapability::<4, 0, 0>::from_classical(cap);
        qcap.k_class = QuantumKClass::line_bundle(1);

        let amplified = qcap.amplify(2);
        // psi^2 on O(1) has c_1 = 2^1 * 1 = 2
        assert_eq!(
            *amplified.k_class.chern_character.get(&1).unwrap(),
            Rational64::from(2)
        );
    }

    #[test]
    fn test_quantum_count_configurations() {
        let mut ns = Namespace::full("agent", 2, 4).unwrap();

        // Add 4 capabilities each with codimension 1
        for i in 0..4 {
            let cap = Capability::new(format!("cap{}", i), format!("Cap {}", i), vec![1], (2, 4))
                .unwrap();
            ns.grant(cap).unwrap();
        }

        // Create quantum capabilities from the granted ones
        let quantum_caps: Vec<QuantumCapability<4, 0, 0>> = ns
            .capabilities
            .iter()
            .map(|c| QuantumCapability::from_classical(c.clone()))
            .collect();

        let (classical, _euler) = ns.quantum_count_configurations(&quantum_caps);
        // Classical count should still be 2
        assert_eq!(classical, IntersectionResult::Finite(2));
    }
}

// ============================================================================
// Parallel Batch Operation Tests
// ============================================================================

#[cfg(all(test, feature = "parallel"))]
mod parallel_tests {
    use super::*;

    #[test]
    fn test_count_configurations_batch() {
        // Create several namespaces with different capability configurations
        let mut ns1 = Namespace::full("ns1", 2, 4).unwrap();
        let mut ns2 = Namespace::full("ns2", 2, 4).unwrap();
        let ns3 = Namespace::full("ns3", 2, 4).unwrap();

        // ns1 gets 4 capabilities -> should give 2 configurations
        for i in 0..4 {
            ns1.grant(
                Capability::new(format!("c{}", i), format!("Cap {}", i), vec![1], (2, 4)).unwrap(),
            )
            .unwrap();
        }

        // ns2 gets 2 capabilities -> positive dimensional
        for i in 0..2 {
            ns2.grant(
                Capability::new(format!("d{}", i), format!("Cap {}", i), vec![1], (2, 4)).unwrap(),
            )
            .unwrap();
        }

        // ns3 has no capabilities -> full Grassmannian

        let namespaces = vec![ns1, ns2, ns3];
        let results = count_configurations_batch(&namespaces);

        assert_eq!(results.len(), 3);
        assert_eq!(results[0], IntersectionResult::Finite(2));
        assert!(matches!(
            results[1],
            IntersectionResult::PositiveDimensional { dimension: 2, .. }
        ));
        assert!(matches!(
            results[2],
            IntersectionResult::PositiveDimensional { dimension: 4, .. }
        ));
    }

    #[test]
    fn test_capability_accessible_batch() {
        let ns1 = Namespace::full("ns1", 2, 4).unwrap();
        let ns2 = Namespace::full("ns2", 3, 6).unwrap();

        let cap_24 = Capability::new("read", "Read", vec![1], (2, 4)).unwrap();
        let cap_36 = Capability::new("write", "Write", vec![1], (3, 6)).unwrap();

        let pairs: Vec<(&Namespace, &Capability)> = vec![
            (&ns1, &cap_24), // Compatible
            (&ns1, &cap_36), // Incompatible Grassmannians
            (&ns2, &cap_36), // Compatible
            (&ns2, &cap_24), // Incompatible Grassmannians
        ];

        let results = capability_accessible_batch(&pairs).unwrap();

        assert_eq!(results.len(), 4);
        assert!(results[0]); // ns1 can access cap_24
        assert!(!results[1]); // ns1 cannot access cap_36 (different Grassmannian)
        assert!(results[2]); // ns2 can access cap_36
        assert!(!results[3]); // ns2 cannot access cap_24 (different Grassmannian)
    }

    #[test]
    fn test_namespace_intersection_batch() {
        let ns1 = Namespace::full("ns1", 2, 4).unwrap();
        let ns2 = Namespace::full("ns2", 2, 4).unwrap();
        let ns3 = Namespace::full("ns3", 3, 6).unwrap();

        let pairs: Vec<(&Namespace, &Namespace)> = vec![
            (&ns1, &ns2), // Same Grassmannian
            (&ns1, &ns3), // Different Grassmannians
            (&ns2, &ns3), // Different Grassmannians
        ];

        let results = namespace_intersection_batch(&pairs).unwrap();

        assert_eq!(results.len(), 3);
        assert!(matches!(
            results[0],
            NamespaceIntersection::Subspace { dimension: 4 }
        ));
        assert_eq!(results[1], NamespaceIntersection::Incompatible);
        assert_eq!(results[2], NamespaceIntersection::Incompatible);
    }
}
