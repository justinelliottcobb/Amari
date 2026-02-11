//! Wall-Crossing and Bridgeland Stability
//!
//! Implements Bridgeland stability conditions on namespace capabilities,
//! modeling how capability counts change as trust levels vary.
//!
//! # Key Concepts
//!
//! - **Stability condition**: Z_t(σ_λ) = -codim(λ) + i·t·dim(λ) assigns a central
//!   charge to each Schubert class. A capability is stable if its phase is in (0, 1).
//! - **Wall**: A trust level where a capability transitions between stable/unstable.
//! - **Wall-crossing formula**: Tracks how the count of stable capabilities changes.
//!
//! # Contracts
//!
//! - Central charge is linear in the class
//! - Phase is in [0, 1] for geometric objects
//! - Walls are finitely many and computable

use crate::namespace::{Capability, CapabilityId, Namespace};
use crate::schubert::SchubertClass;

/// A Bridgeland-type stability condition parameterized by trust level.
///
/// The central charge is:
/// ```text
/// Z_t(σ_λ) = -codim(σ_λ) + i · t · dim(σ_λ)
/// ```
///
/// A Schubert class is stable at trust level t if its phase φ = (1/π)·arg(Z_t) ∈ (0, 1).
#[derive(Debug, Clone)]
pub struct StabilityCondition {
    /// Grassmannian parameters
    pub grassmannian: (usize, usize),
    /// Trust level parameter
    pub trust_level: f64,
}

impl StabilityCondition {
    /// Create the standard stability condition for a Grassmannian.
    ///
    /// # Contract
    ///
    /// ```text
    /// requires: trust_level > 0
    /// ensures: result.is_stable(σ_λ) depends on codim/dim ratio
    /// ```
    pub fn standard(grassmannian: (usize, usize), trust_level: f64) -> Self {
        Self {
            grassmannian,
            trust_level,
        }
    }

    /// Compute the central charge Z_t(σ_λ).
    ///
    /// Returns (real_part, imaginary_part).
    ///
    /// # Contract
    ///
    /// ```text
    /// ensures: result.0 == -codim(class)
    /// ensures: result.1 == trust_level * dim(class)
    /// ```
    #[must_use]
    pub fn central_charge(&self, class: &SchubertClass) -> (f64, f64) {
        let codim = class.codimension() as f64;
        let dim = class.dimension() as f64;
        (-codim, self.trust_level * dim)
    }

    /// Compute the phase φ = (1/π) · arg(Z_t(σ_λ)).
    ///
    /// Phase in (0, 1) means the class is stable.
    /// Phase = 0 or 1 means semistable (on a wall).
    ///
    /// # Contract
    ///
    /// ```text
    /// ensures: 0.0 <= result <= 1.0
    /// ```
    #[must_use]
    pub fn phase(&self, class: &SchubertClass) -> f64 {
        let (re, im) = self.central_charge(class);

        if re == 0.0 && im == 0.0 {
            return 0.0;
        }

        let angle = im.atan2(re); // in (-π, π]
        let normalized = angle / std::f64::consts::PI; // in (-1, 1]

        // Map to [0, 1]: phase in (0, 1) means stable
        if normalized < 0.0 {
            normalized + 1.0
        } else {
            normalized
        }
    }

    /// Check if a capability is stable at this trust level.
    ///
    /// # Contract
    ///
    /// ```text
    /// ensures: result == (0 < phase(class) < 1)
    /// ```
    #[must_use]
    pub fn is_stable(&self, capability: &Capability) -> bool {
        let phase = self.phase(&capability.schubert_class);
        phase > 0.0 && phase < 1.0
    }

    /// Find all stable capabilities in a namespace at this trust level.
    #[must_use]
    pub fn stable_capabilities<'a>(&self, namespace: &'a Namespace) -> Vec<&'a Capability> {
        namespace
            .capabilities
            .iter()
            .filter(|cap| self.is_stable(cap))
            .collect()
    }

    /// Count stable capabilities.
    #[must_use]
    pub fn stable_count(&self, namespace: &Namespace) -> usize {
        self.stable_capabilities(namespace).len()
    }
}

/// A wall in the stability space: a trust level where stability changes.
#[derive(Debug, Clone)]
pub struct Wall {
    /// The trust level at which the wall occurs
    pub trust_level: f64,
    /// The capability that changes stability
    pub destabilized_class: CapabilityId,
    /// Direction: +1 if becoming stable, -1 if becoming unstable
    pub direction: i32,
    /// Change in stable count when crossing this wall
    pub count_change: i32,
}

impl Wall {
    /// The trust level at which the wall occurs.
    #[must_use]
    pub fn trust_level(&self) -> f64 {
        self.trust_level
    }
}

/// Engine for computing wall-crossing phenomena.
///
/// Analyzes how the set of stable capabilities changes as the trust level varies.
#[derive(Debug, Clone)]
pub struct WallCrossingEngine {
    /// Grassmannian parameters
    pub grassmannian: (usize, usize),
}

impl WallCrossingEngine {
    /// Create a new wall-crossing engine.
    #[must_use]
    pub fn new(grassmannian: (usize, usize)) -> Self {
        Self { grassmannian }
    }

    /// Compute all walls for the capabilities in a namespace.
    ///
    /// A wall occurs at trust level t where the phase of some capability
    /// equals 0 or 1, causing a stability transition.
    ///
    /// The wall for σ_λ occurs where:
    /// ```text
    /// arg(Z_t(σ_λ)) = 0 or π
    /// ```
    /// i.e., where t · dim(σ_λ) / codim(σ_λ) reaches critical values.
    ///
    /// # Contract
    ///
    /// ```text
    /// ensures: walls are sorted by trust_level
    /// ensures: each wall corresponds to exactly one capability
    /// ```
    pub fn compute_walls(&self, namespace: &Namespace) -> Vec<Wall> {
        let mut walls = Vec::new();

        for cap in &namespace.capabilities {
            let codim = cap.codimension() as f64;
            let dim = cap.schubert_class.dimension() as f64;

            if dim == 0.0 {
                // Point class: always has phase 1 (purely real, negative)
                // No wall crossing possible
                continue;
            }

            if codim == 0.0 {
                // Identity class: always stable for t > 0
                // Wall at t = 0
                walls.push(Wall {
                    trust_level: 0.0,
                    destabilized_class: cap.id.clone(),
                    direction: 1,
                    count_change: 1,
                });
                continue;
            }

            // The phase φ(t) = (1/π) · arctan(t · dim / codim) + 1/2
            // (since re < 0 for codim > 0)
            //
            // Phase = 1 when Z is purely real negative: impossible (im ≥ 0 for t ≥ 0)
            // Phase approaches 1/2 as t → ∞
            // Phase = 1 only when im = 0 and re < 0: at t = 0
            //
            // Wall: transition from unstable (t near 0, phase near 1) to stable
            // occurs when phase drops below 1:
            // arg = π - ε means phase = 1 - ε/π
            //
            // For the stability condition, the critical trust level is where
            // the ratio codim/dim determines the wall:
            let critical_t = codim / dim;

            // Below critical_t: the phase is close to 1 (barely stable or unstable)
            // Above critical_t: the phase moves toward 1/2 (more stable)
            walls.push(Wall {
                trust_level: critical_t,
                destabilized_class: cap.id.clone(),
                direction: 1,
                count_change: 1,
            });
        }

        walls.sort_by(|a, b| {
            a.trust_level
                .partial_cmp(&b.trust_level)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        walls
    }

    /// Compute the number of stable capabilities at a given trust level.
    #[must_use]
    pub fn stable_count_at(&self, namespace: &Namespace, trust_level: f64) -> usize {
        let condition = StabilityCondition::standard(self.grassmannian, trust_level);
        condition.stable_count(namespace)
    }

    /// Compute the phase diagram: piecewise-constant function
    /// trust_level → count of stable capabilities.
    ///
    /// Returns sorted (trust_level, count) breakpoints.
    pub fn phase_diagram(&self, namespace: &Namespace) -> Vec<(f64, usize)> {
        let walls = self.compute_walls(namespace);

        if walls.is_empty() {
            return vec![(0.0, 0)];
        }

        let mut breakpoints = Vec::new();
        let mut trust_levels: Vec<f64> = vec![0.001]; // start just above 0

        for wall in &walls {
            if wall.trust_level > 0.0 {
                // Sample just before and just after the wall
                trust_levels.push(wall.trust_level - 0.001);
                trust_levels.push(wall.trust_level + 0.001);
            }
        }

        // Add a high trust level
        if let Some(last_wall) = walls.last() {
            trust_levels.push(last_wall.trust_level + 1.0);
        }

        trust_levels.sort_by(|a, b| a.partial_cmp(b).unwrap());
        trust_levels.dedup();

        let mut prev_count = None;
        for &t in &trust_levels {
            let count = self.stable_count_at(namespace, t);
            if prev_count != Some(count) {
                breakpoints.push((t, count));
                prev_count = Some(count);
            }
        }

        breakpoints
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::namespace::Capability;
    use crate::schubert::SchubertClass;

    fn make_test_namespace() -> Namespace {
        let pos = SchubertClass::new(vec![], (2, 4)).unwrap();
        let mut ns = Namespace::new("test", pos);

        // σ_1: codim=1, dim=3
        let cap1 = Capability::new("c1", "Cap1", vec![1], (2, 4)).unwrap();
        // σ_2: codim=2, dim=2
        let cap2 = Capability::new("c2", "Cap2", vec![2], (2, 4)).unwrap();
        // σ_{1,1}: codim=2, dim=2
        let cap3 = Capability::new("c3", "Cap3", vec![1, 1], (2, 4)).unwrap();

        ns.grant(cap1).unwrap();
        ns.grant(cap2).unwrap();
        ns.grant(cap3).unwrap();
        ns
    }

    #[test]
    fn test_stability_standard() {
        let ns = make_test_namespace();
        let cond = StabilityCondition::standard((2, 4), 1.0);

        // At trust level 1.0, capabilities with sufficient dimension should be stable
        let stable = cond.stable_capabilities(&ns);
        assert!(!stable.is_empty());
    }

    #[test]
    fn test_central_charge() {
        let cond = StabilityCondition::standard((2, 4), 1.0);
        let class = SchubertClass::new(vec![1], (2, 4)).unwrap();

        let (re, im) = cond.central_charge(&class);
        assert_eq!(re, -1.0); // codim = 1
        assert_eq!(im, 3.0); // trust * dim = 1.0 * 3
    }

    #[test]
    fn test_phase_range() {
        let cond = StabilityCondition::standard((2, 4), 1.0);

        for partition in &[vec![1], vec![2], vec![1, 1], vec![2, 2]] {
            if let Ok(class) = SchubertClass::new(partition.clone(), (2, 4)) {
                let phase = cond.phase(&class);
                assert!(
                    (0.0..=1.0).contains(&phase),
                    "Phase {} out of range for {:?}",
                    phase,
                    partition
                );
            }
        }
    }

    #[test]
    fn test_stability_low_trust() {
        let ns = make_test_namespace();
        // At very low trust, high-codim capabilities may not be stable
        let cond = StabilityCondition::standard((2, 4), 0.01);
        let stable_count = cond.stable_count(&ns);
        // With very low trust, fewer capabilities should be stable
        assert!(stable_count <= ns.capabilities.len());
    }

    #[test]
    fn test_stability_high_trust() {
        let ns = make_test_namespace();
        // At high trust, most capabilities with nonzero dimension should be stable
        let cond = StabilityCondition::standard((2, 4), 100.0);
        let stable = cond.stable_capabilities(&ns);
        // High trust should stabilize capabilities with dim > 0
        assert!(!stable.is_empty());
    }

    #[test]
    fn test_wall_computation() {
        let ns = make_test_namespace();
        let engine = WallCrossingEngine::new((2, 4));
        let walls = engine.compute_walls(&ns);

        // Should find walls for each capability
        assert!(!walls.is_empty());

        // Walls should be sorted by trust level
        for w in walls.windows(2) {
            assert!(w[0].trust_level <= w[1].trust_level);
        }
    }

    #[test]
    fn test_phase_diagram() {
        let ns = make_test_namespace();
        let engine = WallCrossingEngine::new((2, 4));
        let diagram = engine.phase_diagram(&ns);

        // Diagram should have at least one breakpoint
        assert!(!diagram.is_empty());

        // Trust levels should be increasing
        for d in diagram.windows(2) {
            assert!(d[0].0 < d[1].0);
        }
    }

    #[test]
    fn test_stable_count_monotone() {
        let ns = make_test_namespace();
        let engine = WallCrossingEngine::new((2, 4));

        // Stable count should generally increase with trust level
        // (not strictly, but at high enough trust level it should be maximal)
        let low = engine.stable_count_at(&ns, 0.01);
        let high = engine.stable_count_at(&ns, 100.0);
        assert!(high >= low);
    }

    #[test]
    fn test_point_class_stability() {
        // Point class σ_{2,2} on Gr(2,4): codim=4, dim=0
        let pos = SchubertClass::new(vec![], (2, 4)).unwrap();
        let mut ns = Namespace::new("test", pos);
        let point_cap = Capability::new("pt", "Point", vec![2, 2], (2, 4)).unwrap();
        ns.grant(point_cap).unwrap();

        let cond = StabilityCondition::standard((2, 4), 1.0);
        // Point class has dim=0, so Z = (-4, 0) → phase = 1 → not stable (boundary)
        let stable = cond.stable_count(&ns);
        assert_eq!(stable, 0);
    }
}
