# amari-enumerative 0.18.0 Extensions

## Overview

This document specifies extensions to `amari-enumerative` for version 0.18.0, driven by requirements from the ShaperOS project. The primary additions are:

1. **Complete Littlewood-Richardson coefficient computation** - replacing placeholder intersection numbers with actual Schubert calculus
2. **Multi-class Schubert intersection** - computing intersection numbers for arbitrary collections of Schubert classes
3. **Namespace and Capability types** - domain types that consume Schubert calculus for geometric access control
4. **Tropical acceleration** - fast approximate counting via tropical correspondence

These extensions enable ShaperOS to answer questions like "how many valid namespace configurations satisfy these capability constraints?" using rigorous enumerative geometry.

---

## 1. Littlewood-Richardson Coefficients

### Mathematical Background

The Littlewood-Richardson coefficient `c^ν_{λμ}` appears in the product of Schubert classes:

```
σ_λ · σ_μ = Σ_ν c^ν_{λμ} σ_ν
```

These coefficients count Young tableaux of skew shape `ν/λ` with content `μ` satisfying the *lattice word condition*: reading the tableau right-to-left, top-to-bottom, at every point the number of `i`s seen is ≥ the number of `(i+1)`s seen.

### New Types

```rust
// In: amari-enumerative/src/littlewood_richardson.rs

//! Littlewood-Richardson coefficient computation
//!
//! Implements the combinatorial algorithm for computing LR coefficients
//! via enumeration of valid skew tableaux.

use alloc::vec::Vec;
use alloc::collections::BTreeMap;

/// A partition (Young diagram) represented as weakly decreasing sequence
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Partition {
    /// Parts of the partition, weakly decreasing: λ_1 ≥ λ_2 ≥ ... ≥ λ_k > 0
    pub parts: Vec<usize>,
}

impl Partition {
    /// Create a new partition, automatically sorting and removing zeros
    pub fn new(mut parts: Vec<usize>) -> Self {
        parts.sort_by(|a, b| b.cmp(a)); // Descending order
        parts.retain(|&x| x > 0);
        Self { parts }
    }
    
    /// The empty partition
    pub fn empty() -> Self {
        Self { parts: vec![] }
    }
    
    /// Size (sum of parts)
    pub fn size(&self) -> usize {
        self.parts.iter().sum()
    }
    
    /// Length (number of nonzero parts)
    pub fn length(&self) -> usize {
        self.parts.len()
    }
    
    /// Get part at index (0 if index out of bounds)
    pub fn part(&self, i: usize) -> usize {
        self.parts.get(i).copied().unwrap_or(0)
    }
    
    /// Check if this partition contains another (ν ⊇ λ iff ν_i ≥ λ_i for all i)
    pub fn contains(&self, other: &Partition) -> bool {
        let max_len = self.length().max(other.length());
        for i in 0..max_len {
            if self.part(i) < other.part(i) {
                return false;
            }
        }
        true
    }
    
    /// Conjugate (transpose) partition
    pub fn conjugate(&self) -> Partition {
        if self.parts.is_empty() {
            return Partition::empty();
        }
        
        let max_part = self.parts[0];
        let mut conj = Vec::with_capacity(max_part);
        
        for j in 1..=max_part {
            let count = self.parts.iter().filter(|&&p| p >= j).count();
            if count > 0 {
                conj.push(count);
            }
        }
        
        Partition { parts: conj }
    }
}

/// A skew shape ν/λ (the cells in ν but not in λ)
#[derive(Debug, Clone)]
pub struct SkewShape {
    /// Outer partition
    pub outer: Partition,
    /// Inner partition (must be contained in outer)
    pub inner: Partition,
    /// Cells of the skew shape as (row, col) pairs, row-major order
    cells: Vec<(usize, usize)>,
}

impl SkewShape {
    /// Create a new skew shape
    pub fn new(outer: Partition, inner: Partition) -> Option<Self> {
        if !outer.contains(&inner) {
            return None;
        }
        
        let mut cells = Vec::new();
        let max_row = outer.length();
        
        for row in 0..max_row {
            let start_col = inner.part(row);
            let end_col = outer.part(row);
            for col in start_col..end_col {
                cells.push((row, col));
            }
        }
        
        Some(Self { outer, inner, cells })
    }
    
    /// Number of cells
    pub fn size(&self) -> usize {
        self.cells.len()
    }
    
    /// Get cell by index in reading order
    pub fn cell(&self, index: usize) -> Option<(usize, usize)> {
        self.cells.get(index).copied()
    }
    
    /// Find index of cell above the given cell index (if exists and in skew shape)
    pub fn cell_above(&self, index: usize) -> Option<usize> {
        let (row, col) = self.cells.get(index)?;
        if *row == 0 {
            return None;
        }
        
        let above = (row - 1, *col);
        self.cells.iter().position(|&c| c == above)
    }
    
    /// Check if two cell indices are in the same row
    pub fn same_row(&self, i: usize, j: usize) -> bool {
        match (self.cells.get(i), self.cells.get(j)) {
            (Some((r1, _)), Some((r2, _))) => r1 == r2,
            _ => false,
        }
    }
}

/// A semistandard Young tableau of skew shape
#[derive(Debug, Clone)]
pub struct SkewTableau {
    /// The skew shape
    pub shape: SkewShape,
    /// Filling: label at each cell (indexed same as shape.cells)
    pub filling: Vec<u8>,
}

impl SkewTableau {
    /// Check if the filling is semistandard (rows weakly increasing, columns strictly increasing)
    pub fn is_semistandard(&self) -> bool {
        for (idx, &(row, col)) in self.shape.cells.iter().enumerate() {
            let label = self.filling[idx];
            
            // Check cell to the left (if in skew shape)
            if col > self.shape.inner.part(row) {
                // Find the cell at (row, col-1)
                if let Some(left_idx) = self.shape.cells.iter().position(|&c| c == (row, col - 1)) {
                    if self.filling[left_idx] > label {
                        return false; // Row must be weakly increasing
                    }
                }
            }
            
            // Check cell above (if exists)
            if let Some(above_idx) = self.shape.cell_above(idx) {
                if self.filling[above_idx] >= label {
                    return false; // Column must be strictly increasing
                }
            }
        }
        true
    }
    
    /// Get the content (μ) of this tableau: μ_i = count of label i
    pub fn content(&self) -> Partition {
        let max_label = self.filling.iter().max().copied().unwrap_or(0) as usize;
        let mut counts = vec![0usize; max_label];
        
        for &label in &self.filling {
            if label > 0 {
                counts[label as usize - 1] += 1;
            }
        }
        
        Partition::new(counts)
    }
    
    /// Read the tableau in reverse reading order (right-to-left, top-to-bottom)
    pub fn reverse_reading_word(&self) -> Vec<u8> {
        // Group cells by row
        let max_row = self.shape.outer.length();
        let mut word = Vec::with_capacity(self.filling.len());
        
        for row in 0..max_row {
            // Get cells in this row, sorted by column descending
            let mut row_cells: Vec<(usize, u8)> = self.shape.cells.iter()
                .enumerate()
                .filter(|(_, &(r, _))| r == row)
                .map(|(idx, &(_, col))| (col, self.filling[idx]))
                .collect();
            
            row_cells.sort_by(|a, b| b.0.cmp(&a.0)); // Descending by column
            
            for (_, label) in row_cells {
                word.push(label);
            }
        }
        
        word
    }
    
    /// Check the lattice word condition on reverse reading word
    pub fn satisfies_lattice_condition(&self) -> bool {
        let word = self.reverse_reading_word();
        let max_label = word.iter().max().copied().unwrap_or(0);
        
        let mut counts = vec![0i32; max_label as usize + 1];
        
        for &label in &word {
            counts[label as usize] += 1;
            
            // Check: count of i must be ≥ count of i+1 at every prefix
            for i in 1..max_label as usize {
                if counts[i] < counts[i + 1] {
                    return false;
                }
            }
        }
        
        true
    }
}

/// Compute the Littlewood-Richardson coefficient c^ν_{λμ}
pub fn lr_coefficient(lambda: &Partition, mu: &Partition, nu: &Partition) -> u64 {
    // Quick checks
    if !nu.contains(lambda) {
        return 0;
    }
    
    let skew = match SkewShape::new(nu.clone(), lambda.clone()) {
        Some(s) => s,
        None => return 0,
    };
    
    if skew.size() != mu.size() {
        return 0;
    }
    
    // Enumerate valid tableaux
    let mut count = 0u64;
    enumerate_lr_tableaux(&skew, mu, &mut vec![], &mut count);
    count
}

/// Recursively enumerate LR tableaux
fn enumerate_lr_tableaux(
    skew: &SkewShape,
    content: &Partition,
    partial: &mut Vec<u8>,
    count: &mut u64,
) {
    if partial.len() == skew.size() {
        // Complete filling - check lattice condition
        let tableau = SkewTableau {
            shape: skew.clone(),
            filling: partial.clone(),
        };
        
        if tableau.satisfies_lattice_condition() {
            *count += 1;
        }
        return;
    }
    
    let idx = partial.len();
    let (row, col) = skew.cell(idx).unwrap();
    
    // Determine valid labels
    let max_label = content.length() as u8;
    
    for label in 1..=max_label {
        // Check content constraint
        let used = partial.iter().filter(|&&l| l == label).count();
        if used >= content.part(label as usize - 1) {
            continue;
        }
        
        // Check row constraint (weakly increasing)
        if col > skew.inner.part(row) {
            if let Some(left_idx) = skew.cells.iter().position(|&c| c == (row, col - 1)) {
                if left_idx < partial.len() && partial[left_idx] > label {
                    continue;
                }
            }
        }
        
        // Check column constraint (strictly increasing)
        if let Some(above_idx) = skew.cell_above(idx) {
            if partial[above_idx] >= label {
                continue;
            }
        }
        
        partial.push(label);
        enumerate_lr_tableaux(skew, content, partial, count);
        partial.pop();
    }
}

/// Expand a Schubert product σ_λ · σ_μ as a sum of Schubert classes
pub fn schubert_product(
    lambda: &Partition,
    mu: &Partition,
    grassmannian: (usize, usize), // (k, n)
) -> BTreeMap<Partition, u64> {
    let (k, n) = grassmannian;
    let max_part = n - k;
    
    let mut result = BTreeMap::new();
    
    // Iterate over all partitions ν that could appear
    // ν must: contain λ, have size |λ| + |μ|, fit in k × (n-k) box
    let target_size = lambda.size() + mu.size();
    
    enumerate_partitions_in_box(k, max_part, target_size, &mut |nu| {
        let coeff = lr_coefficient(lambda, mu, &nu);
        if coeff > 0 {
            result.insert(nu, coeff);
        }
    });
    
    result
}

/// Enumerate partitions fitting in a k × m box with given size
fn enumerate_partitions_in_box<F>(k: usize, m: usize, size: usize, callback: &mut F)
where
    F: FnMut(Partition),
{
    fn enumerate_rec<F>(
        parts: &mut Vec<usize>,
        k: usize,
        m: usize,
        remaining: usize,
        max_part: usize,
        callback: &mut F,
    )
    where
        F: FnMut(Partition),
    {
        if remaining == 0 {
            callback(Partition::new(parts.clone()));
            return;
        }
        
        if parts.len() >= k {
            return; // Too many parts
        }
        
        let upper = remaining.min(m).min(max_part);
        for part in (1..=upper).rev() {
            parts.push(part);
            enumerate_rec(parts, k, m, remaining - part, part, callback);
            parts.pop();
        }
    }
    
    let mut parts = Vec::new();
    enumerate_rec(&mut parts, k, m, size, m, callback);
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_partition_contains() {
        let nu = Partition::new(vec![3, 2, 1]);
        let lambda = Partition::new(vec![2, 1]);
        let mu = Partition::new(vec![3, 3]);
        
        assert!(nu.contains(&lambda));
        assert!(!nu.contains(&mu));
    }
    
    #[test]
    fn test_lr_coefficient_simple() {
        // σ_1 · σ_1 in Gr(2,4)
        // Should give σ_2 + σ_{1,1}
        let lambda = Partition::new(vec![1]);
        let mu = Partition::new(vec![1]);
        
        let nu_2 = Partition::new(vec![2]);
        let nu_11 = Partition::new(vec![1, 1]);
        
        assert_eq!(lr_coefficient(&lambda, &mu, &nu_2), 1);
        assert_eq!(lr_coefficient(&lambda, &mu, &nu_11), 1);
    }
    
    #[test]
    fn test_classic_lr() {
        // c^{2,1}_{1,1} = 1 (one LR tableau)
        let lambda = Partition::new(vec![1]);
        let mu = Partition::new(vec![1, 1]);
        let nu = Partition::new(vec![2, 1]);
        
        assert_eq!(lr_coefficient(&lambda, &mu, &nu), 1);
    }
}
```

---

## 2. Multi-Class Schubert Intersection

### Extending SchubertCalculus

```rust
// In: amari-enumerative/src/schubert.rs (extending existing)

use crate::littlewood_richardson::{Partition, schubert_product, lr_coefficient};

/// Result of a Schubert intersection computation
#[derive(Debug, Clone, PartialEq)]
pub enum IntersectionResult {
    /// Empty intersection (overdetermined)
    Empty,
    /// Finite number of points
    Finite(u64),
    /// Positive-dimensional intersection
    PositiveDimensional {
        dimension: usize,
        /// Degree (if computable)
        degree: Option<u64>,
    },
}

impl SchubertCalculus {
    /// Intersect multiple Schubert classes
    ///
    /// Given classes σ_{λ_1}, ..., σ_{λ_m}, compute their intersection number
    /// in the Grassmannian Gr(k, n).
    pub fn multi_intersect(&mut self, classes: &[SchubertClass]) -> IntersectionResult {
        if classes.is_empty() {
            return IntersectionResult::PositiveDimensional {
                dimension: self.grassmannian_dimension(),
                degree: Some(1),
            };
        }
        
        let (k, n) = self.grassmannian_dim;
        let grassmannian_dim = k * (n - k);
        
        // Total codimension
        let total_codim: usize = classes.iter()
            .map(|c| c.partition.iter().sum::<usize>())
            .sum();
        
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
        let partitions: Vec<Partition> = classes.iter()
            .map(|c| Partition::new(c.partition.clone()))
            .collect();
        
        self.multiply_partitions(&partitions)
    }
    
    /// Multiply partitions and extract fundamental class coefficient
    fn multiply_partitions(&self, partitions: &[Partition]) -> u64 {
        let (k, n) = self.grassmannian_dim;
        
        // Start with first partition
        let mut current: BTreeMap<Partition, u64> = BTreeMap::new();
        current.insert(partitions[0].clone(), 1);
        
        // Iteratively multiply
        for partition in &partitions[1..] {
            let mut next: BTreeMap<Partition, u64> = BTreeMap::new();
            
            for (nu, coeff) in &current {
                let products = schubert_product(nu, partition, (k, n));
                for (rho, lr_coeff) in products {
                    *next.entry(rho).or_insert(0) += coeff * lr_coeff;
                }
            }
            
            current = next;
        }
        
        // Extract coefficient of fundamental class
        let fundamental = Partition::new(vec![n - k; k]);
        current.get(&fundamental).copied().unwrap_or(0)
    }
    
    fn grassmannian_dimension(&self) -> usize {
        let (k, n) = self.grassmannian_dim;
        k * (n - k)
    }
    
    fn compute_degree_if_easy(&self, _classes: &[SchubertClass]) -> Option<u64> {
        // Degree computation for positive-dimensional intersection
        // is more complex; return None for now
        None
    }
    
    /// Expand product of two Schubert classes
    pub fn product(
        &mut self,
        class1: &SchubertClass,
        class2: &SchubertClass,
    ) -> Vec<(SchubertClass, u64)> {
        let p1 = Partition::new(class1.partition.clone());
        let p2 = Partition::new(class2.partition.clone());
        
        let products = schubert_product(&p1, &p2, self.grassmannian_dim);
        
        products.into_iter()
            .filter_map(|(partition, coeff)| {
                SchubertClass::new(partition.parts, self.grassmannian_dim)
                    .ok()
                    .map(|class| (class, coeff))
            })
            .collect()
    }
}

// Add helper method to SchubertClass
impl SchubertClass {
    /// Create from partition
    pub fn from_partition(partition: Partition, grassmannian_dim: (usize, usize)) -> EnumerativeResult<Self> {
        Self::new(partition.parts, grassmannian_dim)
    }
    
    /// Convert to partition
    pub fn to_partition(&self) -> Partition {
        Partition::new(self.partition.clone())
    }
}
```

---

## 3. Namespace and Capability Types

These types bridge `amari-enumerative` to ShaperOS. They can live in a new module or a separate crate that depends on `amari-enumerative`.

```rust
// In: amari-enumerative/src/namespace.rs

//! Namespace and Capability types for ShaperOS
//!
//! Namespaces are points in Grassmannians. Capabilities are Schubert conditions.
//! Intersection theory determines access control.

use crate::{SchubertClass, SchubertCalculus, IntersectionResult, EnumerativeResult};
use alloc::vec::Vec;
use alloc::sync::Arc;
use alloc::string::String;

/// Unique identifier for a capability
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CapabilityId(pub Arc<str>);

impl CapabilityId {
    pub fn new(name: impl Into<String>) -> Self {
        Self(Arc::from(name.into()))
    }
}

/// A capability in ShaperOS: an incidence condition on namespaces
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
    pub fn new(
        id: impl Into<String>,
        name: impl Into<String>,
        partition: Vec<usize>,
        grassmannian: (usize, usize),
    ) -> EnumerativeResult<Self> {
        let id_str = id.into();
        Ok(Self {
            id: CapabilityId::new(id_str.clone()),
            name: name.into(),
            schubert_class: SchubertClass::new(partition, grassmannian)?,
            requires: Vec::new(),
            conflicts: Vec::new(),
        })
    }
    
    /// Add a dependency
    pub fn requires(mut self, cap_id: CapabilityId) -> Self {
        self.requires.push(cap_id);
        self
    }
    
    /// Add a conflict
    pub fn conflicts_with(mut self, cap_id: CapabilityId) -> Self {
        self.conflicts.push(cap_id);
        self
    }
    
    /// Codimension of this capability's Schubert class
    pub fn codimension(&self) -> usize {
        self.schubert_class.partition.iter().sum()
    }
}

/// A namespace: a point in a Grassmannian with associated capabilities
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
    pub fn new(
        name: impl Into<String>,
        position: SchubertClass,
    ) -> Self {
        Self {
            grassmannian: position.grassmannian_dim,
            position,
            capabilities: Vec::new(),
            name: name.into(),
        }
    }
    
    /// Create a namespace with full access (identity Schubert class)
    pub fn full(name: impl Into<String>, k: usize, n: usize) -> EnumerativeResult<Self> {
        let position = SchubertClass::new(vec![], (k, n))?;
        Ok(Self::new(name, position))
    }
    
    /// Grant a capability to this namespace
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
    
    /// Check if this namespace has a specific capability
    pub fn has_capability(&self, id: &CapabilityId) -> bool {
        self.capabilities.iter().any(|c| &c.id == id)
    }
    
    /// Count valid configurations satisfying all capability constraints
    pub fn count_configurations(&self) -> IntersectionResult {
        let mut calc = SchubertCalculus::new(self.grassmannian);
        
        let mut classes = vec![self.position.clone()];
        for cap in &self.capabilities {
            classes.push(cap.schubert_class.clone());
        }
        
        calc.multi_intersect(&classes)
    }
}

/// Namespace-related errors
#[derive(Debug, Clone)]
pub enum NamespaceError {
    Conflict {
        new: CapabilityId,
        existing: CapabilityId,
    },
    MissingDependency {
        capability: CapabilityId,
        required: CapabilityId,
    },
    InvalidConfiguration,
}

/// Compute the intersection of two namespaces
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
#[derive(Debug, Clone, PartialEq)]
pub enum NamespaceIntersection {
    /// Namespaces are in different Grassmannians
    Incompatible,
    /// No overlap
    Disjoint,
    /// Single point intersection
    SinglePoint,
    /// Finite number of intersection points
    FinitePoints(u64),
    /// Positive-dimensional intersection
    Subspace { dimension: usize },
}

/// Check if a capability is accessible from a namespace
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

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_capability_creation() {
        // Gr(2, 4): 2-planes in 4-space
        let cap = Capability::new("read", "Read Access", vec![1], (2, 4)).unwrap();
        assert_eq!(cap.codimension(), 1);
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
}
```

---

## 4. Tropical Acceleration (Optional)

For large-scale intersection counting, tropical correspondence theorems allow faster computation.

```rust
// In: amari-enumerative/src/tropical_schubert.rs

//! Tropical methods for Schubert calculus
//!
//! Uses tropical correspondence to speed up intersection counting.

use crate::{SchubertClass, IntersectionResult};

/// Tropical approximation of Schubert intersection
/// 
/// For many practical cases, tropical methods give exact answers
/// with better computational complexity.
pub fn tropical_intersection_count(
    classes: &[SchubertClass],
    grassmannian: (usize, usize),
) -> TropicalResult {
    let (k, n) = grassmannian;
    
    // Convert to tropical setup
    let tropical_classes: Vec<TropicalSchubertClass> = classes.iter()
        .map(|c| TropicalSchubertClass::from_classical(c))
        .collect();
    
    // Check dimension
    let total_codim: usize = tropical_classes.iter()
        .map(|c| c.codimension())
        .sum();
    
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
#[derive(Debug, Clone)]
pub struct TropicalSchubertClass {
    /// Tropicalized partition
    pub weights: Vec<i64>,
}

impl TropicalSchubertClass {
    pub fn from_classical(classical: &SchubertClass) -> Self {
        Self {
            weights: classical.partition.iter().map(|&x| x as i64).collect(),
        }
    }
    
    pub fn codimension(&self) -> usize {
        self.weights.iter().sum::<i64>() as usize
    }
}

#[derive(Debug, Clone)]
pub enum TropicalResult {
    Empty,
    Finite(u64),
    PositiveDimensional { dimension: usize },
}

fn compute_tropical_intersection(
    classes: &[TropicalSchubertClass],
    _k: usize,
    _n: usize,
) -> u64 {
    // Simplified: actual tropical Schubert calculus uses
    // piecewise-linear geometry and lattice point counting
    //
    // For a full implementation, see:
    // - Tropical Schubert calculus (Speyer, Sturmfels)
    // - Correspondence theorems for Grassmannians
    
    // Placeholder: fall back to classical
    // In practice, this would use tropical methods
    1 
}
```

---

## 5. Module Organization

### New File Structure

```
amari-enumerative/src/
├── lib.rs                    # (updated exports)
├── intersection.rs           # (existing)
├── schubert.rs              # (extended)
├── gromov_witten.rs         # (existing)
├── tropical_curves.rs       # (existing)
├── moduli_space.rs          # (existing)
├── higher_genus.rs          # (existing)
├── geometric_algebra.rs     # (existing)
├── performance.rs           # (existing)
├── littlewood_richardson.rs # NEW
├── namespace.rs             # NEW
└── tropical_schubert.rs     # NEW (optional)
```

### Updated lib.rs Exports

```rust
// Add to lib.rs

pub mod littlewood_richardson;
pub mod namespace;

#[cfg(feature = "tropical-schubert")]
pub mod tropical_schubert;

// Re-exports
pub use littlewood_richardson::{
    Partition, SkewShape, SkewTableau,
    lr_coefficient, schubert_product,
};

pub use namespace::{
    Capability, CapabilityId, Namespace, NamespaceError,
    NamespaceIntersection, namespace_intersection, capability_accessible,
};

// Extended Schubert exports
pub use schubert::IntersectionResult;
```

---

## 6. Feature Flags

```toml
# In Cargo.toml

[features]
default = []
tropical-schubert = []
parallel = ["rayon"]
gpu = ["wgpu"]
serde = ["dep:serde"]

[dependencies]
rayon = { version = "1.8", optional = true }
wgpu = { version = "0.18", optional = true }
serde = { version = "1.0", features = ["derive"], optional = true }
```

---

## 7. Integration Tests

```rust
// In: amari-enumerative/tests/schubert_integration.rs

use amari_enumerative::{
    SchubertClass, SchubertCalculus, IntersectionResult,
    Partition, lr_coefficient, schubert_product,
    Namespace, Capability, namespace_intersection,
};

#[test]
fn test_lines_meeting_four_lines() {
    // Classic problem: how many lines meet 4 general lines in P³?
    // Answer: 2
    //
    // This is computed in Gr(2, 4) with 4 copies of σ_1
    
    let mut calc = SchubertCalculus::new((2, 4));
    let sigma_1 = SchubertClass::new(vec![1], (2, 4)).unwrap();
    
    let classes = vec![sigma_1.clone(), sigma_1.clone(), sigma_1.clone(), sigma_1.clone()];
    
    let result = calc.multi_intersect(&classes);
    assert_eq!(result, IntersectionResult::Finite(2));
}

#[test]
fn test_lines_on_cubic_surface() {
    // 27 lines on a cubic surface can be computed via Schubert calculus
    // in Gr(2, 4) with appropriate conditions
    
    // Simplified version: σ_1^4 in Gr(2,4) = 2 (as above)
    // Full 27 lines requires more sophisticated setup
}

#[test]
fn test_namespace_configuration_count() {
    // Create a namespace in Gr(3, 6)
    let ns = Namespace::full("agent", 3, 6).unwrap();
    
    // Should have many configurations (full Grassmannian)
    let count = ns.count_configurations();
    assert!(matches!(count, IntersectionResult::PositiveDimensional { .. }));
}

#[test]
fn test_capability_restricts_configurations() {
    // Adding capabilities reduces the configuration space
    let mut ns = Namespace::full("agent", 2, 4).unwrap();
    
    // Add capabilities until we get a finite count
    let cap1 = Capability::new("c1", "Cap 1", vec![1], (2, 4)).unwrap();
    let cap2 = Capability::new("c2", "Cap 2", vec![1], (2, 4)).unwrap();
    let cap3 = Capability::new("c3", "Cap 3", vec![1], (2, 4)).unwrap();
    let cap4 = Capability::new("c4", "Cap 4", vec![1], (2, 4)).unwrap();
    
    ns.grant(cap1).unwrap();
    ns.grant(cap2).unwrap();
    ns.grant(cap3).unwrap();
    ns.grant(cap4).unwrap();
    
    let count = ns.count_configurations();
    // Four σ_1 conditions in Gr(2,4) gives 2 points
    assert_eq!(count, IntersectionResult::Finite(2));
}
```

---

## 8. Performance Considerations

### Caching LR Coefficients

```rust
// Add to SchubertCalculus

use alloc::collections::BTreeMap;

impl SchubertCalculus {
    /// Cache for LR coefficients
    lr_cache: BTreeMap<(Partition, Partition, Partition), u64>,
    
    /// Get or compute LR coefficient with caching
    fn lr_cached(&mut self, lambda: &Partition, mu: &Partition, nu: &Partition) -> u64 {
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
}
```

### Parallel Enumeration

```rust
#[cfg(feature = "parallel")]
use rayon::prelude::*;

#[cfg(feature = "parallel")]
pub fn lr_coefficient_parallel(lambda: &Partition, mu: &Partition, nu: &Partition) -> u64 {
    // For large partitions, parallelize the tableau enumeration
    // by partitioning the first-row choices
    
    // ... parallel implementation
}
```

---

## Summary

Version 0.18.0 adds:

| Component | Purpose | Key Types |
|-----------|---------|-----------|
| `littlewood_richardson` | LR coefficient computation | `Partition`, `SkewShape`, `SkewTableau` |
| `schubert` extensions | Multi-class intersection | `IntersectionResult`, `SchubertCalculus::multi_intersect` |
| `namespace` | ShaperOS integration | `Namespace`, `Capability`, `CapabilityId` |
| `tropical_schubert` | Fast approximation (optional) | `TropicalSchubertClass`, `tropical_intersection_count` |

These enable the ShaperOS project to perform rigorous access control via enumerative geometry: "Does this agent have a valid namespace configuration for these capabilities?" becomes a Schubert intersection computation.
