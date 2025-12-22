//! Random variable traits for geometric algebra
//!
//! This module provides traits for random variables over multivector spaces,
//! including moment computation and characteristic functions.
//!
//! # Core Trait
//!
//! The `GeometricRandomVariable` trait extends distributions with:
//!
//! - Mean and covariance computation
//! - Higher-order moments
//! - Characteristic functions
//!
//! # Examples
//!
//! ```ignore
//! use amari_probabilistic::random::GeometricRandomVariable;
//! use amari_probabilistic::distribution::GaussianMultivector;
//!
//! let dist = GaussianMultivector::<3, 0, 0>::standard();
//! let mean = dist.expectation();
//! let cov = dist.covariance();
//! ```

mod moments;

pub use moments::{CovarianceMatrix, GeometricRandomVariable, MomentComputer};
