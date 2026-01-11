//! Phase space analysis
//!
//! This module provides tools for analyzing and visualizing dynamical systems
//! in phase space.
//!
//! # Overview
//!
//! Phase space is the space of all possible states of a dynamical system.
//! This module provides:
//!
//! - **Phase portraits**: Visualization of trajectories in state space
//! - **Nullclines**: Curves where vector field components are zero
//! - **Trajectory analysis**: Classification and statistical analysis
//!
//! # Example
//!
//! ```ignore
//! use amari_dynamics::phase::{PhasePortrait, PortraitConfig, compute_nullclines};
//!
//! // Generate phase portrait
//! let config = PortraitConfig::simple_2d((-2.0, 2.0), (-2.0, 2.0));
//! let portrait = PhasePortrait::generate(&system, &config)?;
//!
//! // Find fixed points
//! for fp in &portrait.fixed_points {
//!     println!("Fixed point at {:?}: {:?}", fp.point, fp.stability);
//! }
//!
//! // Compute nullclines
//! let nullclines = compute_nullclines(&system, &NullclineConfig::default())?;
//! println!("Found {} x-nullcline segments", nullclines.x_nullcline.len());
//! ```

pub mod nullcline;
pub mod portrait;
pub mod trajectory;

// Re-export main types and functions
pub use nullcline::{
    compute_nullclines, nullcline_flow_direction, Direction, FlowDirection, NullclineConfig,
    NullclineFlowInfo, NullclinePoint, NullclineResult, NullclineSegment,
};

pub use portrait::{ClassifiedFixedPoint, PhasePortrait, PortraitConfig, VectorFieldPoint};

pub use trajectory::{
    interpolate_trajectory, resample_trajectory, AnalyzedTrajectory, BundleStatistics,
    TrajectoryBundle, TrajectoryMetadata, TrajectoryType,
};

#[cfg(feature = "parallel")]
pub use portrait::generate_portrait_parallel;
