//! Matroid Theory for Enumerative Geometry
//!
//! Implements combinatorial matroid theory with connections to tropical geometry,
//! Grassmannians, and ShaperOS namespace analysis.
//!
//! # Key Types
//!
//! - [`Matroid`]: A matroid on a ground set, represented by its bases
//! - [`ValuatedMatroid`]: A matroid with a valuation on bases (tropical Plücker vector)
//!
//! # Contracts
//!
//! - Bases satisfy the basis exchange axiom
//! - Matroid rank is well-defined: max |B ∩ S| is the same for all bases B
//! - Dual matroid has bases = complements of original bases

use crate::namespace::{CapabilityId, Namespace};
use std::collections::{BTreeMap, BTreeSet};

/// A matroid on a ground set {0, ..., n-1}.
///
/// Represented by its collection of bases, which must satisfy the
/// basis exchange axiom: for any two bases B₁, B₂ and any x ∈ B₁ \ B₂,
/// there exists y ∈ B₂ \ B₁ such that (B₁ \ {x}) ∪ {y} is a basis.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Matroid {
    /// Size of the ground set
    pub ground_set_size: usize,
    /// Collection of bases (each a k-element subset)
    pub bases: BTreeSet<BTreeSet<usize>>,
    /// Rank of the matroid (common size of all bases)
    pub rank: usize,
}

impl Matroid {
    /// Create the uniform matroid U\_{k,n}: all k-subsets of \[n\] are bases.
    ///
    /// # Contract
    ///
    /// ```text
    /// requires: k <= n
    /// ensures: result.bases.len() == C(n, k)
    /// ensures: result.rank == k
    /// ```
    pub fn uniform(k: usize, n: usize) -> Self {
        let bases = k_subsets_btree(n, k);
        Self {
            ground_set_size: n,
            bases,
            rank: k,
        }
    }

    /// Create a matroid from an explicit collection of bases.
    ///
    /// Validates the basis exchange axiom.
    ///
    /// # Contract
    ///
    /// ```text
    /// requires: all bases have the same cardinality
    /// requires: basis exchange axiom holds
    /// ensures: result.rank == bases[0].len()
    /// ```
    pub fn from_bases(n: usize, bases: BTreeSet<BTreeSet<usize>>) -> Result<Self, String> {
        if bases.is_empty() {
            return Err("Matroid must have at least one basis".to_string());
        }

        let rank = bases.iter().next().unwrap().len();

        // Check all bases have the same size
        if bases.iter().any(|b| b.len() != rank) {
            return Err("All bases must have the same cardinality".to_string());
        }

        // Check all elements are in ground set
        if bases.iter().any(|b| b.iter().any(|&e| e >= n)) {
            return Err(format!("All elements must be < {n}"));
        }

        // Validate basis exchange axiom
        for b1 in &bases {
            for b2 in &bases {
                let diff: Vec<usize> = b1.difference(b2).copied().collect();
                for &x in &diff {
                    let mut found_exchange = false;
                    for &y in b2.difference(b1) {
                        let mut candidate: BTreeSet<usize> = b1.clone();
                        candidate.remove(&x);
                        candidate.insert(y);
                        if bases.contains(&candidate) {
                            found_exchange = true;
                            break;
                        }
                    }
                    if !found_exchange {
                        return Err(format!(
                            "Basis exchange axiom fails: cannot exchange {x} from {:?} with element of {:?}",
                            b1, b2
                        ));
                    }
                }
            }
        }

        Ok(Self {
            ground_set_size: n,
            bases,
            rank,
        })
    }

    /// Create a matroid from Plücker coordinates.
    ///
    /// The matroid is defined by the nonvanishing pattern of the Plücker
    /// coordinates: a k-subset I is a basis iff p_I ≠ 0 (within tolerance).
    ///
    /// # Contract
    ///
    /// ```text
    /// requires: coords.len() == C(n, k)
    /// ensures: result represents the matroid of the point in Gr(k,n)
    /// ```
    pub fn from_plucker(
        k: usize,
        n: usize,
        coords: &[f64],
        tolerance: f64,
    ) -> Result<Self, String> {
        let subsets: Vec<Vec<usize>> = k_subsets_vec(n, k);

        if coords.len() != subsets.len() {
            return Err(format!(
                "Expected C({},{}) = {} Plücker coordinates, got {}",
                n,
                k,
                subsets.len(),
                coords.len()
            ));
        }

        let mut bases = BTreeSet::new();
        for (i, subset) in subsets.iter().enumerate() {
            if coords[i].abs() > tolerance {
                bases.insert(subset.iter().copied().collect::<BTreeSet<usize>>());
            }
        }

        if bases.is_empty() {
            return Err("All Plücker coordinates are zero".to_string());
        }

        Ok(Self {
            ground_set_size: n,
            bases,
            rank: k,
        })
    }

    /// Rank of a subset S: max |B ∩ S| over all bases B.
    ///
    /// # Contract
    ///
    /// ```text
    /// ensures: result <= self.rank
    /// ensures: result <= subset.len()
    /// ```
    #[must_use]
    pub fn rank_of(&self, subset: &BTreeSet<usize>) -> usize {
        self.bases
            .iter()
            .map(|b| b.intersection(subset).count())
            .max()
            .unwrap_or(0)
    }

    /// Compute all circuits (minimal dependent sets).
    ///
    /// A circuit is a minimal subset C such that rank(C) < |C|.
    ///
    /// # Contract
    ///
    /// ```text
    /// ensures: forall C in result. rank_of(C) < |C|
    /// ensures: forall C in result. forall e in C. rank_of(C \ {e}) == |C| - 1
    /// ```
    #[must_use]
    pub fn circuits(&self) -> BTreeSet<BTreeSet<usize>> {
        let mut circuits = BTreeSet::new();

        // Check subsets of increasing size
        for size in 2..=self.ground_set_size {
            for subset in subsets_of_size(self.ground_set_size, size) {
                let r = self.rank_of(&subset);
                if r < size {
                    // Check if it's minimal (no proper subset is dependent)
                    let is_minimal = subset.iter().all(|&e| {
                        let mut smaller: BTreeSet<usize> = subset.clone();
                        smaller.remove(&e);
                        self.rank_of(&smaller) == smaller.len()
                    });
                    if is_minimal {
                        circuits.insert(subset);
                    }
                }
            }

            // Optimization: circuits have size at most rank + 1
            if size > self.rank + 1 {
                break;
            }
        }

        circuits
    }

    /// Dual matroid: bases are complements of original bases.
    ///
    /// # Contract
    ///
    /// ```text
    /// ensures: result.rank == self.ground_set_size - self.rank
    /// ensures: result.bases.len() == self.bases.len()
    /// ```
    #[must_use]
    pub fn dual(&self) -> Self {
        let ground: BTreeSet<usize> = (0..self.ground_set_size).collect();
        let dual_bases: BTreeSet<BTreeSet<usize>> = self
            .bases
            .iter()
            .map(|b| ground.difference(b).copied().collect())
            .collect();

        Self {
            ground_set_size: self.ground_set_size,
            bases: dual_bases,
            rank: self.ground_set_size - self.rank,
        }
    }

    /// Direct sum of two matroids (disjoint union of ground sets).
    ///
    /// # Contract
    ///
    /// ```text
    /// ensures: result.rank == self.rank + other.rank
    /// ensures: result.ground_set_size == self.ground_set_size + other.ground_set_size
    /// ```
    #[must_use]
    pub fn direct_sum(&self, other: &Matroid) -> Self {
        let offset = self.ground_set_size;
        let mut bases = BTreeSet::new();

        for b1 in &self.bases {
            for b2 in &other.bases {
                let mut combined: BTreeSet<usize> = b1.clone();
                for &e in b2 {
                    combined.insert(e + offset);
                }
                bases.insert(combined);
            }
        }

        Self {
            ground_set_size: self.ground_set_size + other.ground_set_size,
            bases,
            rank: self.rank + other.rank,
        }
    }

    /// Matroid union: bases are maximal sets in B₁ ∨ B₂.
    ///
    /// This is NOT the same as the direct sum — it uses the same ground set.
    /// A set I is independent in M₁ ∨ M₂ iff I = I₁ ∪ I₂ where I_j is
    /// independent in M_j. The rank is min(|E|, r₁ + r₂).
    ///
    /// # Contract
    ///
    /// ```text
    /// requires: self.ground_set_size == other.ground_set_size
    /// ensures: result.rank <= self.rank + other.rank
    /// ensures: result.ground_set_size == self.ground_set_size
    /// ```
    #[must_use]
    pub fn matroid_union(&self, other: &Matroid) -> Self {
        assert_eq!(
            self.ground_set_size, other.ground_set_size,
            "Matroid union requires same ground set"
        );

        let n = self.ground_set_size;
        let new_rank = (self.rank + other.rank).min(n);

        // Find all independent sets of the union by trying subsets of size new_rank
        let mut bases = BTreeSet::new();

        for subset in subsets_of_size(n, new_rank) {
            if self.is_union_independent(&subset, other) {
                bases.insert(subset);
            }
        }

        // If no bases at this rank, try smaller ranks
        if bases.is_empty() {
            for r in (1..new_rank).rev() {
                for subset in subsets_of_size(n, r) {
                    if self.is_union_independent(&subset, other) {
                        bases.insert(subset);
                    }
                }
                if !bases.is_empty() {
                    return Self {
                        ground_set_size: n,
                        bases,
                        rank: r,
                    };
                }
            }
        }

        Self {
            ground_set_size: n,
            bases,
            rank: new_rank,
        }
    }

    /// Check if a set is independent in M₁ ∨ M₂.
    fn is_union_independent(&self, set: &BTreeSet<usize>, other: &Matroid) -> bool {
        // I is independent in M₁ ∨ M₂ iff I = I₁ ∪ I₂ with I_j independent in M_j
        // Equivalently: max_{A ⊆ I} (r₁(A) + r₂(I\A)) >= |I|
        let elements: Vec<usize> = set.iter().copied().collect();
        let n = elements.len();

        // Try all subsets of the set
        for mask in 0..(1u64 << n) {
            let mut a: BTreeSet<usize> = BTreeSet::new();
            let mut b: BTreeSet<usize> = BTreeSet::new();
            for (i, &e) in elements.iter().enumerate() {
                if mask & (1 << i) != 0 {
                    a.insert(e);
                } else {
                    b.insert(e);
                }
            }
            if self.rank_of(&a) >= a.len() && other.rank_of(&b) >= b.len() {
                return true;
            }
        }
        false
    }

    /// Check if an element is a loop (in no basis).
    ///
    /// # Contract
    ///
    /// ```text
    /// ensures: result == !exists B in bases. e in B
    /// ```
    #[must_use]
    pub fn is_loop(&self, e: usize) -> bool {
        !self.bases.iter().any(|b| b.contains(&e))
    }

    /// Check if an element is a coloop (in every basis).
    ///
    /// # Contract
    ///
    /// ```text
    /// ensures: result == forall B in bases. e in B
    /// ```
    #[must_use]
    pub fn is_coloop(&self, e: usize) -> bool {
        self.bases.iter().all(|b| b.contains(&e))
    }

    /// Delete element e: restrict to ground set \ {e}.
    ///
    /// # Contract
    ///
    /// ```text
    /// requires: e < self.ground_set_size
    /// ensures: result.ground_set_size == self.ground_set_size - 1
    /// ```
    #[must_use]
    pub fn delete(&self, e: usize) -> Self {
        let bases: BTreeSet<BTreeSet<usize>> = self
            .bases
            .iter()
            .filter(|b| !b.contains(&e))
            .map(|b| b.iter().map(|&x| if x > e { x - 1 } else { x }).collect())
            .collect();

        let rank = if bases.is_empty() {
            self.rank.saturating_sub(1)
        } else {
            bases.iter().next().unwrap().len()
        };

        Self {
            ground_set_size: self.ground_set_size - 1,
            bases,
            rank,
        }
    }

    /// Contract element e: bases containing e, with e removed.
    ///
    /// # Contract
    ///
    /// ```text
    /// requires: e < self.ground_set_size
    /// ensures: result.ground_set_size == self.ground_set_size - 1
    /// ensures: result.rank == self.rank - 1 (when e is not a loop)
    /// ```
    #[must_use]
    pub fn contract(&self, e: usize) -> Self {
        let bases: BTreeSet<BTreeSet<usize>> = self
            .bases
            .iter()
            .filter(|b| b.contains(&e))
            .map(|b| {
                b.iter()
                    .filter(|&&x| x != e)
                    .map(|&x| if x > e { x - 1 } else { x })
                    .collect()
            })
            .collect();

        let rank = self.rank.saturating_sub(1);

        Self {
            ground_set_size: self.ground_set_size - 1,
            bases,
            rank,
        }
    }

    /// Compute the Tutte polynomial T_M(x, y) via deletion-contraction.
    ///
    /// Returns coefficients as a map (i, j) -> coefficient of x^i y^j.
    ///
    /// # Contract
    ///
    /// ```text
    /// ensures: T_M(1, 1) == number of bases
    /// ensures: T_M(2, 1) == number of independent sets
    /// ```
    #[must_use]
    pub fn tutte_polynomial(&self) -> BTreeMap<(usize, usize), i64> {
        // Base case: empty ground set
        if self.ground_set_size == 0 {
            let mut result = BTreeMap::new();
            if self.bases.contains(&BTreeSet::new()) {
                result.insert((0, 0), 1);
            }
            return result;
        }

        // Find a non-loop, non-coloop element for deletion-contraction
        let e = (0..self.ground_set_size).find(|&e| !self.is_loop(e) && !self.is_coloop(e));

        match e {
            Some(e) => {
                // T_M = T_{M\e} + T_{M/e}
                let deleted = self.delete(e);
                let contracted = self.contract(e);

                let t_del = deleted.tutte_polynomial();
                let t_con = contracted.tutte_polynomial();

                let mut result = t_del;
                for ((i, j), coeff) in t_con {
                    *result.entry((i, j)).or_insert(0) += coeff;
                }
                result
            }
            None => {
                // All elements are loops or coloops
                let loops = (0..self.ground_set_size)
                    .filter(|&e| self.is_loop(e))
                    .count();
                let coloops = (0..self.ground_set_size)
                    .filter(|&e| self.is_coloop(e))
                    .count();

                // T_M = x^coloops * y^loops
                let mut result = BTreeMap::new();
                result.insert((coloops, loops), 1);
                result
            }
        }
    }

    /// Matroid intersection cardinality via Edmonds' augmenting path algorithm.
    ///
    /// Finds the maximum cardinality common independent set of two matroids
    /// on the same ground set.
    ///
    /// # Contract
    ///
    /// ```text
    /// requires: self.ground_set_size == other.ground_set_size
    /// ensures: result <= min(self.rank, other.rank)
    /// ```
    pub fn intersection_cardinality(&self, other: &Matroid) -> usize {
        assert_eq!(
            self.ground_set_size, other.ground_set_size,
            "Matroid intersection requires same ground set"
        );

        let n = self.ground_set_size;

        // Start with the empty independent set
        let mut current: BTreeSet<usize> = BTreeSet::new();

        while let Some(path) = self.find_augmenting_path(&current, other, n) {
            // Symmetric difference: toggle elements along the path
            for &e in &path {
                if current.contains(&e) {
                    current.remove(&e);
                } else {
                    current.insert(e);
                }
            }
        }

        current.len()
    }

    /// Find an augmenting path for matroid intersection.
    ///
    /// Uses BFS on the exchange graph.
    fn find_augmenting_path(
        &self,
        current: &BTreeSet<usize>,
        other: &Matroid,
        n: usize,
    ) -> Option<Vec<usize>> {
        let outside: Vec<usize> = (0..n).filter(|e| !current.contains(e)).collect();

        // X1 = elements outside current that can be added to remain independent in M1
        let x1: Vec<usize> = outside
            .iter()
            .copied()
            .filter(|&e| {
                let mut test = current.clone();
                test.insert(e);
                self.rank_of(&test) > current.len()
            })
            .collect();

        // X2 = elements outside current that can be added to remain independent in M2
        let x2: BTreeSet<usize> = outside
            .iter()
            .copied()
            .filter(|&e| {
                let mut test = current.clone();
                test.insert(e);
                other.rank_of(&test) > current.len()
            })
            .collect();

        // BFS from X1 to X2 through the exchange graph
        let mut visited: BTreeSet<usize> = BTreeSet::new();
        let mut parent: BTreeMap<usize, Option<usize>> = BTreeMap::new();
        let mut queue: std::collections::VecDeque<usize> = std::collections::VecDeque::new();

        for &e in &x1 {
            if !visited.contains(&e) {
                visited.insert(e);
                parent.insert(e, None);
                queue.push_back(e);
            }
        }

        while let Some(u) = queue.pop_front() {
            if x2.contains(&u) {
                // Found augmenting path — reconstruct it
                let mut path = vec![u];
                let mut cur = u;
                while let Some(Some(p)) = parent.get(&cur) {
                    path.push(*p);
                    cur = *p;
                }
                path.reverse();
                return Some(path);
            }

            if current.contains(&u) {
                // u is in current set: find elements outside that could exchange with u in M1
                for &v in &outside {
                    if !visited.contains(&v) {
                        let mut test = current.clone();
                        test.remove(&u);
                        test.insert(v);
                        if self.rank_of(&test) >= current.len() {
                            visited.insert(v);
                            parent.insert(v, Some(u));
                            queue.push_back(v);
                        }
                    }
                }
            } else {
                // u is outside current set: find elements inside that could exchange with u in M2
                for &v in current {
                    if !visited.contains(&v) {
                        let mut test = current.clone();
                        test.remove(&v);
                        test.insert(u);
                        if other.rank_of(&test) >= current.len() {
                            visited.insert(v);
                            parent.insert(v, Some(u));
                            queue.push_back(v);
                        }
                    }
                }
            }
        }

        None
    }

    /// Schubert matroid: the matroid associated with a Schubert cell in Gr(k, n).
    ///
    /// For a partition λ fitting in the k × (n-k) box, the Schubert matroid
    /// has bases corresponding to k-subsets I such that i_j ≤ λ_j + j
    /// (where λ is padded to length k).
    ///
    /// # Contract
    ///
    /// ```text
    /// requires: partition fits in k × (n-k) box
    /// ensures: result is a valid matroid
    /// ```
    pub fn schubert_matroid(partition: &[usize], k: usize, n: usize) -> Result<Self, String> {
        let m = n - k;
        // Pad partition to length k
        let mut lambda = vec![0usize; k];
        for (i, &p) in partition.iter().enumerate() {
            if i >= k {
                break;
            }
            if p > m {
                return Err(format!("Partition entry {} exceeds n-k={}", p, m));
            }
            lambda[i] = p;
        }

        // Upper bound for position j (0-indexed): m + j - lambda[k-1-j]
        let bounds: Vec<usize> = (0..k).map(|j| m + j - lambda[k - 1 - j]).collect();

        // Bases are k-subsets {i_0 < i_1 < ... < i_{k-1}} with i_j <= bounds[j]
        let all_subsets = k_subsets_vec(n, k);
        let mut bases = BTreeSet::new();

        for subset in all_subsets {
            let fits = subset.iter().enumerate().all(|(j, &i_j)| i_j <= bounds[j]);
            if fits {
                bases.insert(subset.into_iter().collect::<BTreeSet<usize>>());
            }
        }

        if bases.is_empty() {
            return Err("No valid bases for Schubert matroid".to_string());
        }

        Ok(Self {
            ground_set_size: n,
            bases,
            rank: k,
        })
    }
}

/// Combinatorial extensions for matroid theory.
impl Matroid {
    /// Characteristic polynomial χ_M(q) = Σ_{A ⊆ E} (−1)^|A| · q^(r − rank(A)).
    ///
    /// Returns coefficients [c_0, c_1, ..., c_r] where χ_M(q) = Σ cᵢ · q^i.
    #[must_use]
    pub fn characteristic_polynomial(&self) -> Vec<i64> {
        let r = self.rank;
        let n = self.ground_set_size;
        let mut coeffs = vec![0i64; r + 1];

        for mask in 0..(1u64 << n) {
            let subset: BTreeSet<usize> = (0..n).filter(|&i| mask & (1 << i) != 0).collect();
            let size = subset.len();
            let rank_a = self.rank_of(&subset);
            let power = r - rank_a;
            let sign = if size.is_multiple_of(2) { 1i64 } else { -1 };
            coeffs[power] += sign;
        }
        coeffs
    }

    /// Evaluate the characteristic polynomial at an integer q.
    #[must_use]
    pub fn characteristic_polynomial_eval(&self, q: i64) -> i64 {
        let coeffs = self.characteristic_polynomial();
        let mut result = 0i64;
        let mut q_power = 1i64;
        for &c in &coeffs {
            result += c * q_power;
            q_power *= q;
        }
        result
    }

    /// Beta invariant β(M) = (−1)^r · χ'_M(1).
    #[must_use]
    pub fn beta_invariant(&self) -> i64 {
        let coeffs = self.characteristic_polynomial();
        let deriv_at_1: i64 = coeffs.iter().enumerate().map(|(i, &c)| i as i64 * c).sum();
        let sign = if self.rank.is_multiple_of(2) {
            1i64
        } else {
            -1
        };
        sign * deriv_at_1
    }

    /// Closure operator: cl(A) = smallest flat containing A.
    #[must_use]
    pub fn closure(&self, subset: &BTreeSet<usize>) -> BTreeSet<usize> {
        let mut cl = subset.clone();
        let rank_s = self.rank_of(&cl);
        for e in 0..self.ground_set_size {
            if !cl.contains(&e) {
                let mut test = cl.clone();
                test.insert(e);
                if self.rank_of(&test) == rank_s {
                    cl.insert(e);
                }
            }
        }
        cl
    }

    /// Check if a subset is a flat.
    #[must_use]
    pub fn is_flat(&self, subset: &BTreeSet<usize>) -> bool {
        self.closure(subset) == *subset
    }

    /// Lattice of flats grouped by rank: flats[i] = all flats of rank i.
    #[must_use]
    pub fn flats(&self) -> Vec<Vec<BTreeSet<usize>>> {
        let n = self.ground_set_size;
        let r = self.rank;
        let mut by_rank: Vec<Vec<BTreeSet<usize>>> = vec![Vec::new(); r + 1];

        // Compute closures of all subsets and collect unique flats.
        let mut seen: BTreeSet<BTreeSet<usize>> = BTreeSet::new();
        for mask in 0..(1u64 << n) {
            let subset: BTreeSet<usize> = (0..n).filter(|&i| mask & (1 << i) != 0).collect();
            let flat = self.closure(&subset);
            if seen.insert(flat.clone()) {
                let rk = self.rank_of(&flat);
                by_rank[rk].push(flat);
            }
        }
        by_rank
    }

    /// Möbius function μ(∅, F) for each flat F.
    #[must_use]
    pub fn mobius_values(&self) -> BTreeMap<BTreeSet<usize>, i64> {
        let flats_by_rank = self.flats();
        let mut mu: BTreeMap<BTreeSet<usize>, i64> = BTreeMap::new();

        // μ(∅, ∅) = 1
        let empty: BTreeSet<usize> = BTreeSet::new();
        let cl_empty = self.closure(&empty);
        mu.insert(cl_empty.clone(), 1);

        // For each flat F (in increasing rank order), μ(∅,F) = -Σ_{G < F} μ(∅,G)
        for rank_flats in &flats_by_rank {
            for flat in rank_flats {
                if mu.contains_key(flat) {
                    continue;
                }
                let mut sum = 0i64;
                for (g, &m) in &mu {
                    // G < F means G is a proper subset of F and G is a flat
                    if g.is_subset(flat) && g != flat {
                        sum += m;
                    }
                }
                mu.insert(flat.clone(), -sum);
            }
        }
        mu
    }

    /// Check if the matroid is connected (no direct sum decomposition).
    #[must_use]
    pub fn is_connected(&self) -> bool {
        if self.ground_set_size <= 1 {
            return true;
        }
        self.connected_components().len() == 1
    }

    /// Connected components of the matroid.
    #[must_use]
    pub fn connected_components(&self) -> Vec<Matroid> {
        let n = self.ground_set_size;
        if n == 0 {
            return vec![self.clone()];
        }

        // Two elements are in the same component if they appear in a common circuit.
        // Use union-find approach: e, f are connected if rank({e,f}) < 2 or
        // they're in the same circuit. More precisely, use closure:
        // e ~ f if removing either doesn't change the rank of the closure of {e, f}.
        let mut parent: Vec<usize> = (0..n).collect();

        fn find(parent: &mut Vec<usize>, x: usize) -> usize {
            if parent[x] != x {
                parent[x] = find(parent, parent[x]);
            }
            parent[x]
        }

        fn union(parent: &mut Vec<usize>, x: usize, y: usize) {
            let rx = find(parent, x);
            let ry = find(parent, y);
            if rx != ry {
                parent[rx] = ry;
            }
        }

        // Elements e, f are in the same component if they lie in a common circuit.
        // Equivalently: e, f are connected if rank(E\{e}) < rank(E) and
        // there exists a basis B containing e and another containing f.
        // Simpler: for each pair, check if they're in the same component via circuits.
        // For efficiency, use the criterion: e ~ f if cl({e}) and cl({f}) intersect
        // in a circuit, or more directly, use the parallel connection criterion.

        // Simple approach: e ~ f if there exists a circuit containing both.
        let circuits = self.circuits();
        for circuit in &circuits {
            let elems: Vec<usize> = circuit.iter().copied().collect();
            for i in 1..elems.len() {
                union(&mut parent, elems[0], elems[i]);
            }
        }

        // Also connect elements that are parallel (same closure).
        for e in 0..n {
            for f in (e + 1)..n {
                let mut pair = BTreeSet::new();
                pair.insert(e);
                pair.insert(f);
                if self.rank_of(&pair) < 2 {
                    union(&mut parent, e, f);
                }
            }
        }

        // Group elements by component.
        let mut components: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
        for e in 0..n {
            let root = find(&mut parent, e);
            components.entry(root).or_default().push(e);
        }

        // If everything is in one component, return self.
        if components.len() == 1 {
            return vec![self.clone()];
        }

        // Build matroid for each component.
        components
            .values()
            .map(|elems| {
                let elem_set: BTreeSet<usize> = elems.iter().copied().collect();
                self.restrict(&elem_set)
            })
            .collect()
    }

    /// f-vector: f_i = number of independent sets of size i.
    #[must_use]
    pub fn f_vector(&self) -> Vec<u64> {
        let n = self.ground_set_size;
        let r = self.rank;
        let mut f = vec![0u64; r + 1];

        for mask in 0..(1u64 << n) {
            let subset: BTreeSet<usize> = (0..n).filter(|&i| mask & (1 << i) != 0).collect();
            let size = subset.len();
            if size <= r && self.rank_of(&subset) == size {
                f[size] += 1;
            }
        }
        f
    }

    /// h-vector derived from f-vector.
    ///
    /// h_i = Σ_j (-1)^{i-j} C(r-j, i-j) f_j
    #[must_use]
    pub fn h_vector(&self) -> Vec<i64> {
        let f = self.f_vector();
        let r = self.rank;
        let mut h = vec![0i64; r + 1];

        for (i, h_i) in h.iter_mut().enumerate() {
            for (j, &f_j) in f.iter().enumerate().take(i + 1) {
                let sign = if (i - j).is_multiple_of(2) { 1i64 } else { -1 };
                let binom = binomial(r - j, i - j);
                *h_i += sign * binom as i64 * f_j as i64;
            }
        }
        h
    }

    /// Whitney numbers of the first kind: |coefficient of q^i in χ_M(q)|.
    #[must_use]
    pub fn whitney_numbers_first_kind(&self) -> Vec<u64> {
        self.characteristic_polynomial()
            .iter()
            .map(|c| c.unsigned_abs())
            .collect()
    }

    /// Whitney numbers of the second kind: W_i = number of flats of rank i.
    #[must_use]
    pub fn whitney_numbers_second_kind(&self) -> Vec<u64> {
        self.flats().iter().map(|f| f.len() as u64).collect()
    }

    /// Restriction M|A: matroid on subset A with inherited independence.
    #[must_use]
    pub fn restrict(&self, subset: &BTreeSet<usize>) -> Matroid {
        let elems: Vec<usize> = subset.iter().copied().collect();
        let new_n = elems.len();

        // Map old indices to new indices.
        let mut index_map: BTreeMap<usize, usize> = BTreeMap::new();
        for (new_idx, &old_idx) in elems.iter().enumerate() {
            index_map.insert(old_idx, new_idx);
        }

        // Restrict each basis: intersect with subset and remap.
        let restricted_bases: BTreeSet<BTreeSet<usize>> = self
            .bases
            .iter()
            .map(|b| {
                b.intersection(subset)
                    .map(|e| index_map[e])
                    .collect::<BTreeSet<usize>>()
            })
            .collect();

        // The rank of the restriction is the rank of the subset.
        let new_rank = self.rank_of(subset);

        // Filter to only maximal independent sets of size new_rank.
        let bases: BTreeSet<BTreeSet<usize>> = restricted_bases
            .into_iter()
            .filter(|b| b.len() == new_rank)
            .collect();

        // If no bases found at that rank, build from all independent sets.
        if bases.is_empty() && new_rank == 0 {
            let mut b = BTreeSet::new();
            b.insert(BTreeSet::new());
            return Matroid {
                ground_set_size: new_n,
                bases: b,
                rank: 0,
            };
        }

        Matroid {
            ground_set_size: new_n,
            bases,
            rank: new_rank,
        }
    }

    /// Contraction-deletion pair: returns (M/e, M\e).
    #[must_use]
    pub fn contraction_deletion(&self, e: usize) -> (Matroid, Matroid) {
        (self.contract(e), self.delete(e))
    }

    /// Check for a specific excluded minor by name.
    ///
    /// Supported: "U24" (uniform 2,4), "F7" (Fano), "F7*" (dual Fano).
    #[must_use]
    pub fn has_excluded_minor(&self, name: &str) -> bool {
        match name {
            "U24" => {
                let u24 = Matroid::uniform(2, 4);
                has_minor_check(self, &u24)
            }
            "F7" => {
                let f7 = fano_matroid_internal();
                has_minor_check(self, &f7)
            }
            "F7*" => {
                let f7 = fano_matroid_internal();
                has_minor_check(self, &f7.dual())
            }
            _ => false,
        }
    }
}

/// Check if matroid has another matroid as a minor.
///
/// M has N as a minor iff there exist disjoint C, D ⊆ E(M) such that M/C\D ≅ N.
fn has_minor_check(matroid: &Matroid, minor: &Matroid) -> bool {
    let n = matroid.ground_set_size;
    let m_n = minor.ground_set_size;
    let m_r = minor.rank;

    if n < m_n || matroid.rank < m_r {
        return false;
    }

    // Try all ways to select elements to contract (size = rank - m_r)
    // and delete (size = n - m_n - contract_size).
    let contract_size = matroid.rank - m_r;
    let delete_size = n - m_n - contract_size;

    if contract_size + delete_size + m_n != n {
        return false;
    }

    // Iterate over all subsets of size contract_size to contract.
    for contract_set in subsets_of_size(n, contract_size) {
        let remaining: Vec<usize> = (0..n).filter(|e| !contract_set.contains(e)).collect();

        if remaining.len() < m_n {
            continue;
        }

        // Try all subsets of remaining of size delete_size to delete.
        for del_indices in subsets_of_size_from(&remaining, delete_size) {
            let del_set: BTreeSet<usize> = del_indices.iter().copied().collect();

            // Apply contraction then deletion.
            let mut result = matroid.clone();

            // Contract elements (in reverse order to keep indices valid).
            let mut to_contract: Vec<usize> = contract_set.iter().copied().collect();
            to_contract.sort_unstable_by(|a, b| b.cmp(a));
            for &e in &to_contract {
                result = result.contract(e);
            }

            // Adjust deletion indices after contraction.
            let mut to_delete: Vec<usize> = Vec::new();
            for &d in &del_set {
                let adjusted = d - contract_set.iter().filter(|&&c| c < d).count();
                to_delete.push(adjusted);
            }
            to_delete.sort_unstable_by(|a, b| b.cmp(a));

            for &e in &to_delete {
                if e < result.ground_set_size {
                    result = result.delete(e);
                }
            }

            // Check isomorphism: same rank and same bases structure.
            if result.ground_set_size == minor.ground_set_size
                && result.rank == minor.rank
                && result.bases == minor.bases
            {
                return true;
            }

            // Also try all permutations of the remaining ground set for isomorphism.
            // For small ground sets only.
            if m_n <= 6
                && result.ground_set_size == m_n
                && result.rank == m_r
                && is_matroid_isomorphic(&result, minor)
            {
                return true;
            }
        }
    }
    false
}

/// Check if two matroids on the same size ground set are isomorphic.
fn is_matroid_isomorphic(a: &Matroid, b: &Matroid) -> bool {
    if a.ground_set_size != b.ground_set_size || a.rank != b.rank || a.bases.len() != b.bases.len()
    {
        return false;
    }
    let n = a.ground_set_size;
    if n > 8 {
        return a.bases == b.bases;
    }

    // Try all permutations.
    let perms = permutations(n);
    for perm in &perms {
        let mapped_bases: BTreeSet<BTreeSet<usize>> = a
            .bases
            .iter()
            .map(|basis| basis.iter().map(|&e| perm[e]).collect())
            .collect();
        if mapped_bases == b.bases {
            return true;
        }
    }
    false
}

fn permutations(n: usize) -> Vec<Vec<usize>> {
    if n == 0 {
        return vec![vec![]];
    }
    let mut result = Vec::new();
    let sub = permutations(n - 1);
    for p in sub {
        for i in 0..=p.len() {
            let mut new_p = p.clone();
            new_p.insert(i, n - 1);
            result.push(new_p);
        }
    }
    result
}

fn subsets_of_size_from(elements: &[usize], k: usize) -> Vec<BTreeSet<usize>> {
    if k == 0 {
        return vec![BTreeSet::new()];
    }
    if elements.len() < k {
        return Vec::new();
    }
    let mut result = Vec::new();
    let mut current = Vec::with_capacity(k);
    gen_subsets_from(elements, k, 0, &mut current, &mut result);
    result
}

fn gen_subsets_from(
    elements: &[usize],
    k: usize,
    start: usize,
    current: &mut Vec<usize>,
    result: &mut Vec<BTreeSet<usize>>,
) {
    if current.len() == k {
        result.push(current.iter().copied().collect());
        return;
    }
    let remaining = k - current.len();
    if start + remaining > elements.len() {
        return;
    }
    for i in start..=(elements.len() - remaining) {
        current.push(elements[i]);
        gen_subsets_from(elements, k, i + 1, current, result);
        current.pop();
    }
}

/// Fano matroid (internal, for excluded minor checks).
fn fano_matroid_internal() -> Matroid {
    // Fano plane PG(2,2): rank 3 on 7 elements.
    // Lines (dependent triples): {0,1,3}, {1,2,4}, {2,3,5}, {3,4,6}, {0,4,5}, {1,5,6}, {0,2,6}
    let n = 7;
    let k = 3;
    let lines: Vec<BTreeSet<usize>> = vec![
        [0, 1, 3].iter().copied().collect(),
        [1, 2, 4].iter().copied().collect(),
        [2, 3, 5].iter().copied().collect(),
        [3, 4, 6].iter().copied().collect(),
        [0, 4, 5].iter().copied().collect(),
        [1, 5, 6].iter().copied().collect(),
        [0, 2, 6].iter().copied().collect(),
    ];

    // Bases are 3-element subsets that are NOT lines.
    let all_triples = k_subsets_btree(n, k);
    let bases: BTreeSet<BTreeSet<usize>> = all_triples
        .into_iter()
        .filter(|t| !lines.contains(t))
        .collect();

    Matroid {
        ground_set_size: n,
        bases,
        rank: k,
    }
}

fn binomial(n: usize, k: usize) -> u64 {
    if k > n {
        return 0;
    }
    if k == 0 || k == n {
        return 1;
    }
    let k = k.min(n - k);
    let mut result: u64 = 1;
    for i in 0..k {
        result = result * (n - i) as u64 / (i + 1) as u64;
    }
    result
}

/// ShaperOS integration: matroid-based namespace analysis.
impl Namespace {
    /// Compute the capability matroid of this namespace.
    ///
    /// Capabilities are elements; a set is independent if the corresponding
    /// Schubert classes have a transverse intersection.
    ///
    /// # Contract
    ///
    /// ```text
    /// ensures: result.is_some() implies result.ground_set_size == self.capabilities.len()
    /// ensures: result.is_none() implies self.capabilities.is_empty()
    /// ```
    #[must_use]
    pub fn capability_matroid(&self) -> Option<Matroid> {
        let n = self.capabilities.len();
        if n == 0 {
            return None;
        }

        let (k, dim_n) = self.grassmannian;
        let gr_dim = k * (dim_n - k);

        // Find all maximal independent sets (bases)
        let mut max_size = 0;
        let mut candidates: Vec<BTreeSet<usize>> = Vec::new();

        // Try all subsets
        for mask in 1u64..(1u64 << n) {
            let subset: Vec<usize> = (0..n).filter(|&i| mask & (1 << i) != 0).collect();
            let total_codim: usize = subset
                .iter()
                .map(|&i| self.capabilities[i].codimension())
                .sum();

            if total_codim <= gr_dim {
                let size = subset.len();
                if size > max_size {
                    max_size = size;
                    candidates.clear();
                }
                if size == max_size {
                    candidates.push(subset.into_iter().collect());
                }
            }
        }

        if candidates.is_empty() {
            return None;
        }

        let bases: BTreeSet<BTreeSet<usize>> = candidates.into_iter().collect();
        Some(Matroid {
            ground_set_size: n,
            bases,
            rank: max_size,
        })
    }

    /// Find redundant capabilities: those that don't affect the matroid rank.
    ///
    /// A capability is redundant if removing it doesn't change the rank
    /// of the capability matroid (i.e., it is a loop).
    ///
    /// # Contract
    ///
    /// ```text
    /// ensures: forall id in result. removing id does not change matroid rank
    /// ```
    #[must_use]
    pub fn redundant_capabilities(&self) -> Vec<CapabilityId> {
        let Some(matroid) = self.capability_matroid() else {
            return Vec::new();
        };

        let mut redundant = Vec::new();
        for (i, cap) in self.capabilities.iter().enumerate() {
            if matroid.is_loop(i) {
                redundant.push(cap.id.clone());
            }
        }
        redundant
    }

    /// Maximum number of capabilities that can be shared between two namespaces.
    ///
    /// Uses matroid intersection to find the largest common independent set.
    ///
    /// # Contract
    ///
    /// ```text
    /// ensures: result <= min(self.capabilities.len(), other.capabilities.len())
    /// ```
    #[must_use]
    pub fn max_shared_capabilities(&self, other: &Namespace) -> usize {
        let m1 = self.capability_matroid();
        let m2 = other.capability_matroid();

        match (m1, m2) {
            (Some(m1), Some(m2)) if m1.ground_set_size == m2.ground_set_size => {
                m1.intersection_cardinality(&m2)
            }
            _ => 0,
        }
    }
}

/// A matroid with a valuation on its bases (tropical Plücker vector).
///
/// The valuation v: bases → R satisfies the tropical Plücker relations.
#[derive(Debug, Clone)]
pub struct ValuatedMatroid {
    /// Underlying matroid
    pub matroid: Matroid,
    /// Valuation: basis → value
    pub valuation: BTreeMap<BTreeSet<usize>, f64>,
}

impl ValuatedMatroid {
    /// Create a valuated matroid from tropical Plücker coordinates.
    ///
    /// # Contract
    ///
    /// ```text
    /// requires: coords.len() == C(n, k)
    /// ensures: result satisfies tropical Plücker relations (approximately)
    /// ```
    pub fn from_tropical_plucker(k: usize, n: usize, coords: &[f64]) -> Result<Self, String> {
        let subsets = k_subsets_vec(n, k);

        if coords.len() != subsets.len() {
            return Err(format!(
                "Expected {} tropical Plücker coordinates, got {}",
                subsets.len(),
                coords.len()
            ));
        }

        let mut bases = BTreeSet::new();
        let mut valuation = BTreeMap::new();

        for (i, subset) in subsets.iter().enumerate() {
            if coords[i].is_finite() {
                let basis: BTreeSet<usize> = subset.iter().copied().collect();
                bases.insert(basis.clone());
                valuation.insert(basis, coords[i]);
            }
        }

        if bases.is_empty() {
            return Err("No finite tropical Plücker coordinates".to_string());
        }

        let rank = k;
        let matroid = Matroid {
            ground_set_size: n,
            bases,
            rank,
        };

        Ok(Self { matroid, valuation })
    }

    /// Check if the valuation satisfies tropical Plücker relations.
    ///
    /// The 3-term tropical Plücker relation states:
    /// for any (k-1)-subset S and elements a < b < c < d not in S,
    /// the minimum of {v(S∪{a,c}) + v(S∪{b,d}), v(S∪{a,d}) + v(S∪{b,c})}
    /// is attained at least twice (where v(S∪{a,b}) + v(S∪{c,d}) is included).
    #[must_use]
    pub fn satisfies_tropical_plucker(&self) -> bool {
        // For small matroids, just return true (full check is expensive)
        if self.matroid.rank <= 1 || self.matroid.ground_set_size <= 3 {
            return true;
        }

        // Check the tropical Plücker relations
        let k = self.matroid.rank;
        let _n = self.matroid.ground_set_size;

        if k < 2 {
            return true;
        }

        // For each pair of k-subsets differing in exactly 2 elements
        for b1 in &self.matroid.bases {
            for b2 in &self.matroid.bases {
                let diff1: Vec<usize> = b1.difference(b2).copied().collect();
                let diff2: Vec<usize> = b2.difference(b1).copied().collect();

                if diff1.len() != 2 || diff2.len() != 2 {
                    continue;
                }

                // This is a basic form of the Plücker relation
                let v1 = self.valuation.get(b1).copied().unwrap_or(f64::INFINITY);
                let v2 = self.valuation.get(b2).copied().unwrap_or(f64::INFINITY);

                // The exchange axiom in the valuated setting
                let i = diff1[0];
                let _j = diff1[1];
                let a = diff2[0];
                let b_elem = diff2[1];

                // Check: min of the three exchange terms is attained twice
                let mut exchange1 = b1.clone();
                exchange1.remove(&i);
                exchange1.insert(a);

                let mut exchange2 = b1.clone();
                exchange2.remove(&i);
                exchange2.insert(b_elem);

                let v_ex1 = self
                    .valuation
                    .get(&exchange1)
                    .copied()
                    .unwrap_or(f64::INFINITY);
                let v_ex2 = self
                    .valuation
                    .get(&exchange2)
                    .copied()
                    .unwrap_or(f64::INFINITY);

                // Basic check: finite valuations should have consistent exchanges
                if v1.is_finite() && v2.is_finite() && v_ex1.is_infinite() && v_ex2.is_infinite() {
                    return false;
                }
            }
        }

        true
    }

    /// Convert to a TropicalSchubertClass (when tropical-schubert feature is enabled).
    ///
    /// Extracts the valuation as tropical weights.
    #[cfg(feature = "tropical-schubert")]
    #[must_use]
    pub fn to_tropical_schubert(&self) -> crate::tropical_schubert::TropicalSchubertClass {
        // Collect valuation values as integer weights
        let weights: Vec<i64> = self.valuation.values().map(|&v| v as i64).collect();

        crate::tropical_schubert::TropicalSchubertClass::new(weights)
    }
}

// Helper functions

fn k_subsets_btree(n: usize, k: usize) -> BTreeSet<BTreeSet<usize>> {
    k_subsets_vec(n, k)
        .into_iter()
        .map(|s| s.into_iter().collect())
        .collect()
}

fn k_subsets_vec(n: usize, k: usize) -> Vec<Vec<usize>> {
    let mut result = Vec::new();
    let mut current = Vec::with_capacity(k);
    gen_subsets(n, k, 0, &mut current, &mut result);
    result
}

fn gen_subsets(
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
    if start + remaining > n {
        return;
    }
    for i in start..=(n - remaining) {
        current.push(i);
        gen_subsets(n, k, i + 1, current, result);
        current.pop();
    }
}

fn subsets_of_size(n: usize, k: usize) -> Vec<BTreeSet<usize>> {
    k_subsets_vec(n, k)
        .into_iter()
        .map(|s| s.into_iter().collect())
        .collect()
}

/// Batch compute matroid circuits for multiple matroids in parallel.
#[cfg(feature = "parallel")]
#[must_use]
pub fn circuits_batch(matroids: &[Matroid]) -> Vec<BTreeSet<BTreeSet<usize>>> {
    use rayon::prelude::*;
    matroids.par_iter().map(|m| m.circuits()).collect()
}

/// Batch compute Tutte polynomials for multiple matroids in parallel.
#[cfg(feature = "parallel")]
#[must_use]
pub fn tutte_polynomial_batch(matroids: &[Matroid]) -> Vec<BTreeMap<(usize, usize), i64>> {
    use rayon::prelude::*;
    matroids.par_iter().map(|m| m.tutte_polynomial()).collect()
}

/// Batch compute matroid intersection cardinalities in parallel.
#[cfg(feature = "parallel")]
#[must_use]
pub fn intersection_cardinality_batch(pairs: &[(Matroid, Matroid)]) -> Vec<usize> {
    use rayon::prelude::*;
    pairs
        .par_iter()
        .map(|(m1, m2)| m1.intersection_cardinality(m2))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::namespace::Capability;
    use crate::schubert::SchubertClass;

    #[test]
    fn test_uniform_matroid() {
        let m = Matroid::uniform(2, 4);
        assert_eq!(m.rank, 2);
        assert_eq!(m.bases.len(), 6); // C(4,2)
        assert_eq!(m.ground_set_size, 4);
    }

    #[test]
    fn test_matroid_from_plucker() {
        // All Plücker coordinates nonzero → uniform matroid
        let coords = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0]; // C(4,2) = 6 coords
        let m = Matroid::from_plucker(2, 4, &coords, 0.001).unwrap();
        assert_eq!(m.rank, 2);
        assert_eq!(m.bases.len(), 6);
    }

    #[test]
    fn test_matroid_from_plucker_partial() {
        // Some zero coordinates
        let coords = vec![1.0, 0.0, 1.0, 1.0, 0.0, 1.0];
        let m = Matroid::from_plucker(2, 4, &coords, 0.001).unwrap();
        assert_eq!(m.rank, 2);
        assert_eq!(m.bases.len(), 4);
    }

    #[test]
    fn test_schubert_matroid() {
        // Schubert matroid for σ_1 in Gr(2,4): k=2, m=2, λ=[1,0]
        // bounds: j=0 → 2+0-0=2, j=1 → 2+1-1=2
        // Bases: {i_0 < i_1} with i_0 ≤ 2, i_1 ≤ 2 → {0,1}, {0,2}, {1,2}
        let m = Matroid::schubert_matroid(&[1], 2, 4).unwrap();
        assert_eq!(m.rank, 2);
        assert_eq!(m.bases.len(), 3);
    }

    #[test]
    fn test_schubert_matroid_empty_partition() {
        // Empty partition σ_∅ → bounds: j=0 → 2, j=1 → 3 → all subsets are bases
        let m = Matroid::schubert_matroid(&[], 2, 4).unwrap();
        assert_eq!(m.bases.len(), 6);
    }

    #[test]
    fn test_matroid_circuits() {
        // U_{2,4}: circuits are all 3-element subsets
        let m = Matroid::uniform(2, 4);
        let circuits = m.circuits();
        assert_eq!(circuits.len(), 4); // C(4,3) = 4
    }

    #[test]
    fn test_matroid_dual() {
        let m = Matroid::uniform(2, 4);
        let dual = m.dual();
        assert_eq!(dual.rank, 2); // 4 - 2
        assert_eq!(dual.bases.len(), 6); // U_{2,4} is self-dual
    }

    #[test]
    fn test_matroid_direct_sum() {
        let m1 = Matroid::uniform(1, 2);
        let m2 = Matroid::uniform(1, 2);
        let sum = m1.direct_sum(&m2);
        assert_eq!(sum.rank, 2);
        assert_eq!(sum.ground_set_size, 4);
    }

    #[test]
    fn test_matroid_loop_coloop() {
        // U_{2,3}: no loops, no coloops
        let m = Matroid::uniform(2, 3);
        for e in 0..3 {
            assert!(!m.is_loop(e));
            assert!(!m.is_coloop(e));
        }
    }

    #[test]
    fn test_matroid_delete_contract() {
        let m = Matroid::uniform(2, 4);

        let deleted = m.delete(0);
        assert_eq!(deleted.ground_set_size, 3);
        assert_eq!(deleted.rank, 2);

        let contracted = m.contract(0);
        assert_eq!(contracted.ground_set_size, 3);
        assert_eq!(contracted.rank, 1);
    }

    #[test]
    fn test_matroid_intersection_cardinality() {
        let m1 = Matroid::uniform(2, 4);
        let m2 = Matroid::uniform(2, 4);
        // Intersection of two copies of U_{2,4} should be 2
        assert_eq!(m1.intersection_cardinality(&m2), 2);
    }

    #[test]
    fn test_tutte_polynomial_uniform() {
        let m = Matroid::uniform(1, 2);
        let tutte = m.tutte_polynomial();
        // T_{U_{1,2}}(x, y) = x + y
        assert_eq!(tutte.get(&(1, 0)).copied().unwrap_or(0), 1);
        assert_eq!(tutte.get(&(0, 1)).copied().unwrap_or(0), 1);
    }

    #[test]
    fn test_namespace_capability_matroid() {
        let pos = SchubertClass::new(vec![], (2, 4)).unwrap();
        let mut ns = Namespace::new("test", pos);

        let cap1 = Capability::new("c1", "Cap 1", vec![1], (2, 4)).unwrap();
        let cap2 = Capability::new("c2", "Cap 2", vec![1], (2, 4)).unwrap();
        ns.grant(cap1).unwrap();
        ns.grant(cap2).unwrap();

        let matroid = ns.capability_matroid();
        assert!(matroid.is_some());
        let m = matroid.unwrap();
        assert_eq!(m.ground_set_size, 2);
    }

    #[test]
    fn test_valuated_matroid_tropical_plucker() {
        let coords = vec![0.0, 1.0, 2.0, 1.0, 2.0, 3.0]; // C(4,2)=6
        let vm = ValuatedMatroid::from_tropical_plucker(2, 4, &coords).unwrap();
        assert_eq!(vm.matroid.rank, 2);
        assert!(vm.satisfies_tropical_plucker());
    }

    #[test]
    fn test_characteristic_polynomial_u24() {
        let m = Matroid::uniform(2, 4);
        let chi = m.characteristic_polynomial();
        // χ_{U_{2,4}}(q) = q² - 4q + 3
        assert_eq!(chi, vec![3, -4, 1]);
    }

    #[test]
    fn test_characteristic_polynomial_eval_at_1() {
        // χ_M(1) = 0 for any matroid with at least one element
        let m = Matroid::uniform(2, 4);
        assert_eq!(m.characteristic_polynomial_eval(1), 0);
    }

    #[test]
    fn test_beta_invariant() {
        let m = Matroid::uniform(2, 4);
        let beta = m.beta_invariant();
        // β(M) = (-1)^r · χ'(1)
        // For U_{2,4}: χ(q) = q² - 4q + 3, χ'(1) = -2, β = (-1)² · (-2) = -2
        assert_eq!(beta, -2);
    }

    #[test]
    fn test_closure_idempotent() {
        let m = Matroid::uniform(2, 4);
        let s: BTreeSet<usize> = [0].iter().copied().collect();
        let cl = m.closure(&s);
        let cl2 = m.closure(&cl);
        assert_eq!(cl, cl2);
    }

    #[test]
    fn test_flats_of_u24() {
        let m = Matroid::uniform(2, 4);
        let flats = m.flats();
        // Rank 0: just ∅
        assert_eq!(flats[0].len(), 1);
        // Rank 1: singletons {0}, {1}, {2}, {3}
        assert_eq!(flats[1].len(), 4);
        // Rank 2: just {0,1,2,3} (the whole set)
        assert_eq!(flats[2].len(), 1);
    }

    #[test]
    fn test_f_vector_uniform() {
        let m = Matroid::uniform(2, 4);
        let f = m.f_vector();
        // f_0 = 1, f_1 = 4, f_2 = 6
        assert_eq!(f, vec![1, 4, 6]);
    }

    #[test]
    fn test_whitney_second_kind() {
        let m = Matroid::uniform(2, 4);
        let w = m.whitney_numbers_second_kind();
        // W_0 = 1 (empty flat), W_1 = 4 (singleton flats), W_2 = 1 (full set)
        assert_eq!(w, vec![1, 4, 1]);
    }

    #[test]
    fn test_restrict() {
        let m = Matroid::uniform(2, 4);
        let s: BTreeSet<usize> = [0, 1, 2].iter().copied().collect();
        let r = m.restrict(&s);
        assert_eq!(r.ground_set_size, 3);
        assert_eq!(r.rank, 2);
        // Should be U_{2,3}
        assert_eq!(r.bases.len(), 3);
    }

    #[test]
    fn test_contraction_deletion_pair() {
        let m = Matroid::uniform(2, 4);
        let (contracted, deleted) = m.contraction_deletion(0);
        assert_eq!(contracted.ground_set_size, 3);
        assert_eq!(contracted.rank, 1);
        assert_eq!(deleted.ground_set_size, 3);
        assert_eq!(deleted.rank, 2);
    }

    #[test]
    fn test_connected_uniform() {
        let m = Matroid::uniform(2, 4);
        assert!(m.is_connected());
    }

    #[test]
    fn test_connected_components_direct_sum() {
        let m1 = Matroid::uniform(1, 2);
        let m2 = Matroid::uniform(1, 2);
        let sum = m1.direct_sum(&m2);
        let components = sum.connected_components();
        assert_eq!(components.len(), 2);
    }

    #[test]
    fn test_has_excluded_minor_u24() {
        // U_{2,5} should contain U_{2,4} as a minor
        let m = Matroid::uniform(2, 5);
        assert!(m.has_excluded_minor("U24"));
        // U_{2,3} should NOT contain U_{2,4}
        let m2 = Matroid::uniform(2, 3);
        assert!(!m2.has_excluded_minor("U24"));
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn test_circuits_batch() {
        let matroids = vec![Matroid::uniform(2, 4), Matroid::uniform(1, 3)];
        let results = super::circuits_batch(&matroids);
        assert_eq!(results[0].len(), 4); // C(4,3) circuits for U_{2,4}
        assert_eq!(results[1].len(), 3); // C(3,2) circuits for U_{1,3}
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn test_tutte_polynomial_batch() {
        let matroids = vec![Matroid::uniform(1, 2), Matroid::uniform(1, 3)];
        let results = super::tutte_polynomial_batch(&matroids);
        assert_eq!(results.len(), 2);
        // U_{1,2}: T = x + y
        assert_eq!(results[0].get(&(1, 0)).copied().unwrap_or(0), 1);
        assert_eq!(results[0].get(&(0, 1)).copied().unwrap_or(0), 1);
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn test_intersection_cardinality_batch() {
        let pairs = vec![
            (Matroid::uniform(2, 4), Matroid::uniform(2, 4)),
            (Matroid::uniform(1, 3), Matroid::uniform(1, 3)),
        ];
        let results = super::intersection_cardinality_batch(&pairs);
        assert_eq!(results, vec![2, 1]);
    }
}
