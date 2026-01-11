//! Bifurcation types and classification
//!
//! This module provides the classification of bifurcations in dynamical systems.
//! Bifurcations are qualitative changes in system behavior as parameters vary.
//!
//! # Types of Bifurcations
//!
//! ## Local Bifurcations (equilibrium changes)
//!
//! - **Saddle-Node**: Two equilibria collide and annihilate
//! - **Transcritical**: Two equilibria exchange stability
//! - **Pitchfork**: One equilibrium becomes three (or vice versa)
//! - **Hopf**: Equilibrium loses stability and limit cycle appears
//!
//! ## Global Bifurcations
//!
//! - **Homoclinic**: Limit cycle collides with saddle point
//! - **Heteroclinic**: Cycle connects different saddle points
//!
//! # Example
//!
//! ```ignore
//! use amari_dynamics::bifurcation::{BifurcationType, BifurcationPoint};
//!
//! let bifurcation = BifurcationPoint {
//!     parameter_value: 1.5,
//!     bifurcation_type: BifurcationType::HopfSupercritical,
//!     critical_state: state,
//!     eigenvalues: vec![(0.0, 1.0), (0.0, -1.0)],
//! };
//!
//! println!("Found {} at μ = {}", bifurcation.bifurcation_type, bifurcation.parameter_value);
//! ```

use std::fmt;

use amari_core::Multivector;

/// Classification of bifurcation types
///
/// Each type represents a qualitatively different way that system behavior
/// can change as parameters vary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BifurcationType {
    /// Saddle-Node (fold): Two equilibria collide and annihilate
    ///
    /// Normal form: dx/dt = μ - x²
    /// - For μ < 0: no equilibria
    /// - For μ = 0: one semi-stable equilibrium (bifurcation point)
    /// - For μ > 0: two equilibria (stable and unstable)
    SaddleNode,

    /// Transcritical: Two equilibria exchange stability
    ///
    /// Normal form: dx/dt = μx - x²
    /// - Equilibrium at x=0 exists for all μ
    /// - Second equilibrium at x=μ
    /// - Stability switches at μ=0
    Transcritical,

    /// Supercritical Pitchfork: Stable equilibrium becomes unstable,
    /// two stable equilibria emerge
    ///
    /// Normal form: dx/dt = μx - x³
    /// - For μ < 0: single stable equilibrium at x=0
    /// - For μ > 0: x=0 unstable, two stable at x=±√μ
    PitchforkSupercritical,

    /// Subcritical Pitchfork: Unstable equilibrium becomes stable,
    /// two unstable equilibria emerge
    ///
    /// Normal form: dx/dt = μx + x³
    /// Dangerous bifurcation - system can jump to distant attractor
    PitchforkSubcritical,

    /// Supercritical Hopf: Stable equilibrium spawns stable limit cycle
    ///
    /// Eigenvalues cross imaginary axis with nonzero imaginary part.
    /// A stable limit cycle emerges smoothly as the equilibrium
    /// becomes unstable.
    HopfSupercritical,

    /// Subcritical Hopf: Unstable limit cycle shrinks to unstable equilibrium
    ///
    /// Dangerous bifurcation - system can jump to distant attractor
    /// when equilibrium loses stability.
    HopfSubcritical,

    /// Period Doubling: Limit cycle doubles its period
    ///
    /// One of the Floquet multipliers crosses -1.
    /// Often seen in route to chaos (Feigenbaum cascade).
    PeriodDoubling,

    /// Neimark-Sacker (Hopf for maps): Fixed point spawns invariant circle
    ///
    /// Floquet multiplier crosses unit circle at angle ≠ 0, π.
    /// Creates quasiperiodic motion on torus.
    NeimarkSacker,

    /// Homoclinic: Limit cycle collides with saddle point
    ///
    /// Global bifurcation - cycle period goes to infinity.
    Homoclinic,

    /// Heteroclinic: Cycle connects multiple saddle points
    ///
    /// Global bifurcation creating or destroying heteroclinic orbits.
    Heteroclinic,

    /// Blue-Sky Catastrophe: Limit cycle disappears with bounded period
    ///
    /// The cycle collides with a saddle-node of cycles.
    BlueSky,

    /// Unknown or indeterminate bifurcation type
    Unknown,
}

impl fmt::Display for BifurcationType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SaddleNode => write!(f, "Saddle-Node"),
            Self::Transcritical => write!(f, "Transcritical"),
            Self::PitchforkSupercritical => write!(f, "Supercritical Pitchfork"),
            Self::PitchforkSubcritical => write!(f, "Subcritical Pitchfork"),
            Self::HopfSupercritical => write!(f, "Supercritical Hopf"),
            Self::HopfSubcritical => write!(f, "Subcritical Hopf"),
            Self::PeriodDoubling => write!(f, "Period Doubling"),
            Self::NeimarkSacker => write!(f, "Neimark-Sacker"),
            Self::Homoclinic => write!(f, "Homoclinic"),
            Self::Heteroclinic => write!(f, "Heteroclinic"),
            Self::BlueSky => write!(f, "Blue-Sky Catastrophe"),
            Self::Unknown => write!(f, "Unknown"),
        }
    }
}

impl BifurcationType {
    /// Check if this is a local bifurcation (involving equilibrium changes)
    pub fn is_local(&self) -> bool {
        matches!(
            self,
            Self::SaddleNode
                | Self::Transcritical
                | Self::PitchforkSupercritical
                | Self::PitchforkSubcritical
                | Self::HopfSupercritical
                | Self::HopfSubcritical
        )
    }

    /// Check if this is a global bifurcation
    pub fn is_global(&self) -> bool {
        matches!(self, Self::Homoclinic | Self::Heteroclinic | Self::BlueSky)
    }

    /// Check if this is a codimension-1 bifurcation
    ///
    /// Codimension-1 bifurcations are generically encountered when
    /// varying a single parameter.
    pub fn is_codimension_one(&self) -> bool {
        matches!(
            self,
            Self::SaddleNode
                | Self::Transcritical
                | Self::PitchforkSupercritical
                | Self::PitchforkSubcritical
                | Self::HopfSupercritical
                | Self::HopfSubcritical
                | Self::PeriodDoubling
        )
    }

    /// Check if this is a dangerous bifurcation
    ///
    /// Dangerous bifurcations can cause large sudden jumps in system state.
    pub fn is_dangerous(&self) -> bool {
        matches!(
            self,
            Self::PitchforkSubcritical | Self::HopfSubcritical | Self::BlueSky
        )
    }

    /// Check if this bifurcation involves eigenvalues crossing the imaginary axis
    pub fn involves_hopf(&self) -> bool {
        matches!(self, Self::HopfSupercritical | Self::HopfSubcritical)
    }
}

/// A detected bifurcation point
#[derive(Debug, Clone)]
pub struct BifurcationPoint<const P: usize, const Q: usize, const R: usize> {
    /// The parameter value at the bifurcation
    pub parameter_value: f64,

    /// The type of bifurcation
    pub bifurcation_type: BifurcationType,

    /// The critical state (equilibrium or cycle point) at bifurcation
    pub critical_state: Multivector<P, Q, R>,

    /// Eigenvalues at the bifurcation point as (real, imaginary) pairs
    pub eigenvalues: Vec<(f64, f64)>,

    /// Confidence in the classification (0 to 1)
    pub confidence: f64,
}

impl<const P: usize, const Q: usize, const R: usize> BifurcationPoint<P, Q, R> {
    /// Create a new bifurcation point
    pub fn new(
        parameter_value: f64,
        bifurcation_type: BifurcationType,
        critical_state: Multivector<P, Q, R>,
        eigenvalues: Vec<(f64, f64)>,
    ) -> Self {
        Self {
            parameter_value,
            bifurcation_type,
            critical_state,
            eigenvalues,
            confidence: 1.0,
        }
    }

    /// Create with explicit confidence
    pub fn with_confidence(mut self, confidence: f64) -> Self {
        self.confidence = confidence.clamp(0.0, 1.0);
        self
    }

    /// Get the critical eigenvalue (closest to imaginary axis)
    pub fn critical_eigenvalue(&self) -> Option<(f64, f64)> {
        self.eigenvalues
            .iter()
            .min_by(|a, b| a.0.abs().partial_cmp(&b.0.abs()).unwrap())
            .copied()
    }

    /// Get the frequency at Hopf bifurcation
    ///
    /// Returns the imaginary part of the critical eigenvalue.
    pub fn hopf_frequency(&self) -> Option<f64> {
        if self.bifurcation_type.involves_hopf() {
            self.critical_eigenvalue().map(|(_, im)| im.abs())
        } else {
            None
        }
    }
}

impl<const P: usize, const Q: usize, const R: usize> fmt::Display for BifurcationPoint<P, Q, R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} bifurcation at μ = {:.6}",
            self.bifurcation_type, self.parameter_value
        )
    }
}

/// Criticality conditions for bifurcation detection
#[derive(Debug, Clone, Copy)]
pub struct CriticalityCondition {
    /// Tolerance for zero eigenvalue detection (saddle-node, etc.)
    pub zero_tolerance: f64,

    /// Tolerance for pure imaginary eigenvalue detection (Hopf)
    pub imaginary_tolerance: f64,

    /// Tolerance for unit eigenvalue detection (discrete maps)
    pub unit_tolerance: f64,
}

impl Default for CriticalityCondition {
    fn default() -> Self {
        Self {
            zero_tolerance: 1e-6,
            imaginary_tolerance: 1e-6,
            unit_tolerance: 1e-6,
        }
    }
}

impl CriticalityCondition {
    /// Check if eigenvalue is effectively zero
    pub fn is_zero(&self, eigenvalue: (f64, f64)) -> bool {
        eigenvalue.0.abs() < self.zero_tolerance && eigenvalue.1.abs() < self.zero_tolerance
    }

    /// Check if eigenvalue is purely imaginary (nonzero)
    pub fn is_purely_imaginary(&self, eigenvalue: (f64, f64)) -> bool {
        eigenvalue.0.abs() < self.imaginary_tolerance && eigenvalue.1.abs() > self.zero_tolerance
    }

    /// Check if eigenvalue has unit modulus (for discrete maps)
    pub fn is_unit_modulus(&self, eigenvalue: (f64, f64)) -> bool {
        let modulus = (eigenvalue.0 * eigenvalue.0 + eigenvalue.1 * eigenvalue.1).sqrt();
        (modulus - 1.0).abs() < self.unit_tolerance
    }

    /// Check for saddle-node condition: one zero eigenvalue
    pub fn has_saddle_node_condition(&self, eigenvalues: &[(f64, f64)]) -> bool {
        let zero_count = eigenvalues.iter().filter(|e| self.is_zero(**e)).count();
        zero_count == 1
    }

    /// Check for Hopf condition: pair of purely imaginary eigenvalues
    pub fn has_hopf_condition(&self, eigenvalues: &[(f64, f64)]) -> bool {
        let purely_imaginary: Vec<_> = eigenvalues
            .iter()
            .filter(|e| self.is_purely_imaginary(**e))
            .collect();

        // Need a conjugate pair (same magnitude, opposite signs)
        purely_imaginary.len() >= 2
            && purely_imaginary
                .windows(2)
                .any(|w| (w[0].1 + w[1].1).abs() < self.zero_tolerance)
    }
}

/// Configuration for bifurcation detection algorithms
#[derive(Debug, Clone)]
pub struct BifurcationConfig {
    /// Conditions for detecting critical eigenvalues
    pub criticality: CriticalityCondition,

    /// Step size for parameter continuation
    pub parameter_step: f64,

    /// Minimum step size for adaptive continuation
    pub min_step: f64,

    /// Maximum step size for adaptive continuation
    pub max_step: f64,

    /// Maximum number of continuation steps
    pub max_steps: usize,

    /// Tolerance for Newton iteration in fixed point finding
    pub newton_tolerance: f64,

    /// Maximum Newton iterations
    pub max_newton_iterations: usize,
}

impl Default for BifurcationConfig {
    fn default() -> Self {
        Self {
            criticality: CriticalityCondition::default(),
            parameter_step: 0.01,
            min_step: 1e-6,
            max_step: 0.1,
            max_steps: 10000,
            newton_tolerance: 1e-10,
            max_newton_iterations: 50,
        }
    }
}

impl BifurcationConfig {
    /// Create configuration for fine-grained analysis
    pub fn fine() -> Self {
        Self {
            criticality: CriticalityCondition {
                zero_tolerance: 1e-8,
                imaginary_tolerance: 1e-8,
                unit_tolerance: 1e-8,
            },
            parameter_step: 0.001,
            min_step: 1e-8,
            max_step: 0.01,
            max_steps: 100000,
            newton_tolerance: 1e-12,
            max_newton_iterations: 100,
        }
    }

    /// Create configuration for coarse exploration
    pub fn coarse() -> Self {
        Self {
            criticality: CriticalityCondition {
                zero_tolerance: 1e-4,
                imaginary_tolerance: 1e-4,
                unit_tolerance: 1e-4,
            },
            parameter_step: 0.1,
            min_step: 1e-4,
            max_step: 0.5,
            max_steps: 1000,
            newton_tolerance: 1e-8,
            max_newton_iterations: 30,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bifurcation_type_display() {
        assert_eq!(format!("{}", BifurcationType::SaddleNode), "Saddle-Node");
        assert_eq!(
            format!("{}", BifurcationType::HopfSupercritical),
            "Supercritical Hopf"
        );
        assert_eq!(
            format!("{}", BifurcationType::PitchforkSubcritical),
            "Subcritical Pitchfork"
        );
    }

    #[test]
    fn test_bifurcation_type_classification() {
        assert!(BifurcationType::SaddleNode.is_local());
        assert!(BifurcationType::HopfSupercritical.is_local());
        assert!(!BifurcationType::Homoclinic.is_local());

        assert!(BifurcationType::Homoclinic.is_global());
        assert!(!BifurcationType::SaddleNode.is_global());

        assert!(BifurcationType::HopfSubcritical.is_dangerous());
        assert!(!BifurcationType::HopfSupercritical.is_dangerous());
    }

    #[test]
    fn test_criticality_condition_zero() {
        let cond = CriticalityCondition::default();

        assert!(cond.is_zero((1e-8, 0.0)));
        assert!(cond.is_zero((0.0, 1e-8)));
        assert!(!cond.is_zero((0.1, 0.0)));
    }

    #[test]
    fn test_criticality_condition_purely_imaginary() {
        let cond = CriticalityCondition::default();

        assert!(cond.is_purely_imaginary((0.0, 1.0)));
        assert!(cond.is_purely_imaginary((1e-8, -2.0)));
        assert!(!cond.is_purely_imaginary((0.1, 1.0))); // Not purely imaginary
        assert!(!cond.is_purely_imaginary((0.0, 0.0))); // Zero, not purely imaginary
    }

    #[test]
    fn test_saddle_node_condition() {
        let cond = CriticalityCondition::default();

        // One zero eigenvalue: saddle-node
        let eigenvalues = vec![(0.0, 0.0), (-1.0, 0.0)];
        assert!(cond.has_saddle_node_condition(&eigenvalues));

        // No zero eigenvalue
        let eigenvalues2 = vec![(0.1, 0.0), (-1.0, 0.0)];
        assert!(!cond.has_saddle_node_condition(&eigenvalues2));

        // Two zero eigenvalues: not saddle-node
        let eigenvalues3 = vec![(0.0, 0.0), (0.0, 0.0)];
        assert!(!cond.has_saddle_node_condition(&eigenvalues3));
    }

    #[test]
    fn test_hopf_condition() {
        let cond = CriticalityCondition::default();

        // Conjugate pair on imaginary axis: Hopf
        let eigenvalues = vec![(0.0, 1.0), (0.0, -1.0)];
        assert!(cond.has_hopf_condition(&eigenvalues));

        // Not purely imaginary
        let eigenvalues2 = vec![(0.1, 1.0), (0.1, -1.0)];
        assert!(!cond.has_hopf_condition(&eigenvalues2));
    }

    #[test]
    fn test_bifurcation_point_creation() {
        let state = Multivector::<2, 0, 0>::zero();
        let eigenvalues = vec![(0.0, 1.0), (0.0, -1.0)];

        let bp = BifurcationPoint::new(1.5, BifurcationType::HopfSupercritical, state, eigenvalues);

        assert_eq!(bp.parameter_value, 1.5);
        assert_eq!(bp.bifurcation_type, BifurcationType::HopfSupercritical);
        assert!((bp.confidence - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_hopf_frequency() {
        let state = Multivector::<2, 0, 0>::zero();
        let eigenvalues = vec![(0.0, 2.5), (0.0, -2.5)];

        let bp = BifurcationPoint::new(1.5, BifurcationType::HopfSupercritical, state, eigenvalues);

        let freq = bp.hopf_frequency();
        assert!(freq.is_some());
        assert!((freq.unwrap() - 2.5).abs() < 1e-10);
    }

    #[test]
    fn test_bifurcation_config() {
        let default = BifurcationConfig::default();
        let fine = BifurcationConfig::fine();
        let coarse = BifurcationConfig::coarse();

        assert!(fine.parameter_step < default.parameter_step);
        assert!(coarse.parameter_step > default.parameter_step);
    }
}
