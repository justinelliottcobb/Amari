//! Bifurcation theory for dynamical systems
//!
//! This module provides tools for analyzing bifurcations - qualitative changes
//! in system behavior as parameters vary.
//!
//! # Overview
//!
//! Bifurcation theory studies how the topological structure of a dynamical system's
//! phase portrait changes as parameters are varied. Key concepts include:
//!
//! - **Fixed point bifurcations**: Changes in number or stability of equilibria
//! - **Periodic orbit bifurcations**: Changes in limit cycles
//! - **Global bifurcations**: Changes involving connections between invariant sets
//!
//! # Types of Bifurcations
//!
//! ## Local (Codimension-1)
//!
//! | Type | Condition | Normal Form |
//! |------|-----------|-------------|
//! | Saddle-Node | Zero eigenvalue | dx/dt = μ - x² |
//! | Transcritical | Zero eigenvalue + symmetry | dx/dt = μx - x² |
//! | Pitchfork | Zero eigenvalue + Z₂ symmetry | dx/dt = μx - x³ |
//! | Hopf | Purely imaginary pair | Complex normal form |
//!
//! ## Global
//!
//! - Homoclinic: Orbit connecting saddle to itself
//! - Heteroclinic: Orbit connecting different saddle points
//!
//! # Geometric Algebra Context
//!
//! In Clifford algebra spaces Cl(P,Q,R), bifurcation analysis extends naturally:
//!
//! - Multivector states allow tracking rotational degrees of freedom
//! - Spinor-valued vector fields can exhibit bifurcations in their rotor components
//! - The geometric product enables analysis of coupled orientation-position dynamics
//!
//! # Example
//!
//! ```ignore
//! use amari_dynamics::{
//!     flow::ParametricSystem,
//!     bifurcation::{
//!         BifurcationType, BifurcationConfig, BifurcationDiagram,
//!         NaturalContinuation, ParameterContinuation,
//!     },
//! };
//!
//! // Define a parametric system
//! struct SaddleNodeSystem { mu: f64 }
//!
//! impl ParametricSystem<1, 0, 0> for SaddleNodeSystem {
//!     type Parameter = f64;
//!     fn parameter(&self) -> &f64 { &self.mu }
//!     fn set_parameter(&mut self, param: f64) { self.mu = param; }
//!     // ... vector_field implementation
//! }
//!
//! // Trace solution branch
//! let continuation = NaturalContinuation::new();
//! let config = BifurcationConfig::default();
//!
//! let (branch, bifurcations) = continuation.continue_branch(
//!     &mut system,
//!     initial_state,
//!     0.0,           // start parameter
//!     (-1.0, 1.0),   // parameter range
//!     &config,
//! )?;
//!
//! for bif in &bifurcations {
//!     println!("Found {} at μ = {:.4}", bif.bifurcation_type, bif.parameter_value);
//! }
//! ```

mod continuation;
mod diagram;
mod traits;
mod types;

// Re-export main types
pub use continuation::{
    NaturalContinuation, PseudoArclengthContinuation, SimpleBifurcationDetector,
};
pub use diagram::{BifurcationDiagram, DiagramBuilder, DiagramConfig, DiagramPoint};
pub use traits::{BifurcationDetector, BranchPoint, ParameterContinuation, SolutionBranch};
pub use types::{BifurcationConfig, BifurcationPoint, BifurcationType, CriticalityCondition};

/// Convenience function to compute a bifurcation diagram
///
/// Creates a bifurcation diagram using default settings.
pub fn compute_diagram<S, const P: usize, const Q: usize, const R: usize>(
    system: &mut S,
    param_range: (f64, f64),
) -> crate::error::Result<BifurcationDiagram<P, Q, R>>
where
    S: crate::flow::ParametricSystem<P, Q, R, Parameter = f64>,
{
    let config = DiagramConfig {
        param_range,
        ..Default::default()
    };
    BifurcationDiagram::compute_continuous(system, &config)
}

/// Find all bifurcations in a parameter range
///
/// Scans the parameter range and returns all detected bifurcation points.
pub fn find_bifurcations<S, const P: usize, const Q: usize, const R: usize>(
    system: &mut S,
    initial_states: &[amari_core::Multivector<P, Q, R>],
    initial_param: f64,
    param_range: (f64, f64),
) -> crate::error::Result<Vec<BifurcationPoint<P, Q, R>>>
where
    S: crate::flow::ParametricSystem<P, Q, R, Parameter = f64>,
{
    let detector = SimpleBifurcationDetector::new();
    let config = BifurcationConfig::default();
    detector.detect_from_initials(system, initial_states, initial_param, param_range, &config)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::Result;
    use crate::flow::DynamicalSystem;
    use amari_core::Multivector;

    // Test system: transcritical normal form dx/dt = μx - x²
    struct TranscriticalSystem {
        mu: f64,
    }

    impl DynamicalSystem<1, 0, 0> for TranscriticalSystem {
        fn vector_field(&self, state: &Multivector<1, 0, 0>) -> Result<Multivector<1, 0, 0>> {
            let x = state.get(0);
            let mut result = Multivector::zero();
            result.set(0, self.mu * x - x * x);
            Ok(result)
        }
    }

    impl crate::flow::ParametricSystem<1, 0, 0> for TranscriticalSystem {
        type Parameter = f64;

        fn parameter(&self) -> &f64 {
            &self.mu
        }

        fn set_parameter(&mut self, param: f64) {
            self.mu = param;
        }

        fn vector_field_with_param(
            &self,
            state: &Multivector<1, 0, 0>,
            param: &f64,
        ) -> Result<Multivector<1, 0, 0>> {
            let x = state.get(0);
            let mut result = Multivector::zero();
            result.set(0, param * x - x * x);
            Ok(result)
        }
    }

    #[test]
    fn test_bifurcation_type_properties() {
        assert!(BifurcationType::SaddleNode.is_local());
        assert!(BifurcationType::SaddleNode.is_codimension_one());
        assert!(!BifurcationType::SaddleNode.is_dangerous());

        assert!(BifurcationType::HopfSubcritical.is_dangerous());
        assert!(BifurcationType::HopfSupercritical.involves_hopf());

        assert!(BifurcationType::Homoclinic.is_global());
        assert!(!BifurcationType::Homoclinic.is_local());
    }

    #[test]
    fn test_criticality_conditions() {
        let cond = CriticalityCondition::default();

        // Test zero eigenvalue detection
        assert!(cond.is_zero((0.0, 0.0)));
        assert!(cond.is_zero((1e-8, 1e-8)));
        assert!(!cond.is_zero((0.01, 0.0)));

        // Test purely imaginary
        assert!(cond.is_purely_imaginary((0.0, 1.5)));
        assert!(!cond.is_purely_imaginary((0.0, 0.0))); // Zero is not "purely imaginary"
    }

    #[test]
    fn test_compute_diagram_convenience() {
        let mut system = TranscriticalSystem { mu: 0.0 };

        let result = compute_diagram(&mut system, (-0.5, 0.5));
        assert!(result.is_ok());
    }

    #[test]
    fn test_natural_continuation_basic() {
        let mut system = TranscriticalSystem { mu: 0.5 };

        let continuation = NaturalContinuation::new();
        let config = BifurcationConfig::coarse();

        // Start at x = 0 (always a fixed point for transcritical)
        let initial = Multivector::<1, 0, 0>::zero();

        let result = continuation.continue_branch(&mut system, initial, 0.5, (-0.5, 1.0), &config);

        assert!(result.is_ok());
        let (branch, _bifurcations) = result.unwrap();
        assert!(!branch.is_empty());
    }
}
