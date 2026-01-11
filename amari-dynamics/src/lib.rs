//! # amari-dynamics
//!
//! Dynamical systems analysis on geometric algebra spaces.
//!
//! This crate provides tools for analyzing dynamical systems whose states live
//! in Clifford algebra spaces Cl(P,Q,R). It includes:
//!
//! - **ODE Solvers**: Runge-Kutta methods for numerical integration
//! - **Stability Analysis**: Fixed point finding, linearization, eigenvalue classification
//! - **Bifurcation Theory**: Parameter continuation, bifurcation detection
//! - **Attractor Analysis**: Basin computation, limit cycle detection
//! - **Lyapunov Exponents**: Chaos characterization via QR methods
//! - **Phase Space Tools**: Trajectories, phase portraits, nullclines
//!
//! # Quick Start
//!
//! ```ignore
//! use amari_dynamics::{
//!     flow::{DynamicalSystem, HarmonicOscillator},
//!     solver::{ODESolver, RungeKutta4, Trajectory},
//! };
//! use amari_core::Multivector;
//!
//! // Create a harmonic oscillator
//! let system = HarmonicOscillator::new(1.0);
//!
//! // Set initial conditions: x=1, v=0
//! let mut initial = Multivector::<2, 0, 0>::zero();
//! initial.set(1, 1.0);
//!
//! // Integrate for one period
//! let solver = RungeKutta4::new();
//! let trajectory = solver.solve(&system, initial, 0.0, 6.28, 1000)?;
//!
//! // Analyze the trajectory
//! println!("Final state: {:?}", trajectory.final_state());
//! # Ok::<(), amari_dynamics::DynamicsError>(())
//! ```
//!
//! # Geometric Algebra Integration
//!
//! States are represented as multivectors in Cl(P,Q,R), enabling:
//!
//! - **Rotation Dynamics**: Rotor evolution for attitude dynamics
//! - **Geometric Constraints**: Natural encoding of manifold constraints
//! - **Grade-Aware Evolution**: Separate evolution of scalar, vector, bivector components
//!
//! # Phantom Types
//!
//! The crate uses phantom types for compile-time verification of system properties:
//!
//! ```ignore
//! use amari_dynamics::phantom::*;
//!
//! // A system verified to be stable and non-chaotic
//! let verified: VerifiedStableSystem<MySystem> = analyze_and_verify(system)?;
//!
//! // Functions can require specific properties
//! fn process_stable<S>(sys: &VerifiedStableSystem<S>) { ... }
//! ```
//!
//! # Feature Flags
//!
//! - `std` (default): Standard library support
//! - `parallel`: Rayon-based parallel algorithms
//! - `stochastic`: Stochastic dynamics via amari-probabilistic
//! - `contracts`: Creusot formal verification contracts
//!
//! # Module Overview
//!
//! | Module | Description |
//! |--------|-------------|
//! | [`flow`] | Dynamical system traits and definitions |
//! | [`solver`] | ODE integration methods |
//! | [`stability`] | Fixed points, linearization, eigenvalue analysis |
//! | [`bifurcation`] | Bifurcation detection and continuation |
//! | [`attractor`] | Attractor analysis, Poincaré sections, basins |
//! | [`lyapunov`] | Lyapunov exponents and chaos characterization |
//! | [`ergodic`] | Invariant measures, Birkhoff averages, ergodicity tests |
//! | [`phase`] | Phase portraits, nullclines, trajectory analysis |
//! | [`systems`] | Built-in systems (Lorenz, Van der Pol, Hénon, etc.) |
//! | [`stochastic`] | Stochastic dynamics (requires `stochastic` feature) |
//! | [`gpu`] | GPU acceleration (requires `gpu` feature) |
//! | [`wasm`] | WASM bindings (requires `wasm` feature) |
//! | [`phantom`] | Compile-time type markers |
//! | [`error`] | Error types |

#![cfg_attr(not(feature = "std"), no_std)]
#![warn(missing_docs)]
#![warn(clippy::all)]

#[cfg(not(feature = "std"))]
extern crate alloc;

#[cfg(not(feature = "std"))]
use alloc::{string::String, vec, vec::Vec};

// Re-export core types
pub use amari_core::Multivector;

// Modules
pub mod attractor;
pub mod bifurcation;
pub mod ergodic;
pub mod error;
pub mod flow;
pub mod lyapunov;
pub mod phantom;
pub mod phase;
pub mod solver;
pub mod stability;
#[cfg(feature = "stochastic")]
pub mod stochastic;
pub mod systems;

// GPU acceleration (feature-gated)
#[cfg(feature = "gpu")]
pub mod gpu;

// WASM bindings (feature-gated)
#[cfg(feature = "wasm")]
pub mod wasm;

// Re-export common types at crate root
pub use attractor::{
    Attractor, AttractorConfig, AttractorType, BasinConfig, BasinResult, FixedPointAttractor,
    LimitCycleAttractor, MultiBasinResult, PoincareCrossing, PoincareResult, PoincareSection,
    StrangeAttractor,
};
pub use bifurcation::{
    BifurcationConfig, BifurcationDiagram, BifurcationPoint, BifurcationType, NaturalContinuation,
    ParameterContinuation,
};
pub use ergodic::{
    birkhoff_average, compute_histogram_measure, compute_invariant_measure, test_ergodicity,
    BirkhoffConfig, BirkhoffResult, EmpiricalMeasure, ErgodicityTest, HistogramMeasure,
    InvariantMeasure, MeasureConfig,
};
pub use error::{DynamicsError, Result};
pub use flow::{DiscreteMap, DynamicalSystem, HarmonicOscillator, NonAutonomousSystem};
pub use lyapunov::{
    compute_largest_lyapunov, compute_lyapunov_spectrum, is_chaotic, kaplan_yorke_dimension,
    ks_entropy, LyapunovClassification, LyapunovConfig, LyapunovSpectrum,
};
pub use phantom::{
    Autonomous, Chaotic, ContinuousTime, DiscreteTime, NonAutonomous, Regular, Stable, TypedSystem,
    UnknownChaos, UnknownStability, Unstable,
};
pub use phase::{
    compute_nullclines, interpolate_trajectory, resample_trajectory, AnalyzedTrajectory,
    BundleStatistics, ClassifiedFixedPoint, NullclineConfig, NullclineResult, PhasePortrait,
    PortraitConfig, TrajectoryBundle, TrajectoryMetadata, TrajectoryType, VectorFieldPoint,
};
pub use solver::{ODESolver, RungeKutta4, Trajectory};
pub use stability::{
    analyze_stability, find_and_analyze, find_fixed_point, stability_report, DifferentiationConfig,
    EigenvalueAnalysis, FixedPointConfig, FixedPointResult, StabilityReport, StabilityType,
};
pub use systems::{
    DoublePendulum, DrivenPendulum, DuffingOscillator, ForcedDuffing, ForcedVanDerPol,
    GeneralizedHenon, HenonMap, LorenzSystem, LoziMap, RosslerSystem, SimplePendulum,
    VanDerPolOscillator,
};

// Stochastic dynamics (feature-gated)
#[cfg(feature = "stochastic")]
pub use stochastic::{
    kramers_escape_time, kramers_rate, noise_for_escape_time, residence_times, BoundaryCondition,
    FirstPassageResult, FokkerPlanck1D, FokkerPlanck2D, FokkerPlanckConfig, LangevinConfig,
    LangevinSystem, LangevinTrajectory, Region, TransitionAnalyzer, TransitionConfig,
    TransitionCounts, UnderdampedLangevin,
};

// GPU acceleration (feature-gated)
#[cfg(feature = "gpu")]
pub use gpu::{
    AdaptiveDynamics, BatchTrajectoryConfig, BatchTrajectoryResult, FlowFieldConfig,
    FlowFieldResult, GpuDynamics, GpuSystemType,
};

// WASM bindings (feature-gated)
#[cfg(feature = "wasm")]
pub use wasm::{
    compute_flow_field, WasmDuffing, WasmLorenz, WasmPendulum, WasmRossler, WasmTrajectory,
    WasmVanDerPol,
};

/// Crate version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version() {
        assert!(!VERSION.is_empty());
    }

    #[test]
    fn test_harmonic_oscillator_full_integration() {
        let system = HarmonicOscillator::new(1.0);
        let solver = RungeKutta4::new();

        // Initial condition: x = 1, v = 0
        let mut initial = Multivector::<2, 0, 0>::zero();
        initial.set(1, 1.0);

        // Integrate for half a period
        let trajectory = solver
            .solve(&system, initial, 0.0, core::f64::consts::PI, 1000)
            .unwrap();

        // After half period, x should be approximately -1
        let final_state = trajectory.final_state().unwrap();
        let x = final_state.get(1);
        let v = final_state.get(2);

        assert!((x - (-1.0)).abs() < 1e-4, "Expected x ≈ -1, got {}", x);
        assert!(v.abs() < 1e-4, "Expected v ≈ 0, got {}", v);
    }
}
