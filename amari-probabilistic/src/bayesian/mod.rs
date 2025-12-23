//! Bayesian inference on geometric algebra spaces
//!
//! This module provides tools for Bayesian inference with multivector-valued
//! parameters and observations.
//!
//! # Core Concepts
//!
//! - **Prior**: Distribution over parameters before observing data
//! - **Likelihood**: Probability of data given parameters
//! - **Posterior**: Updated distribution after observing data
//!
//! # Example
//!
//! ```ignore
//! use amari_probabilistic::bayesian::{BayesianGA, JeffreysPrior};
//! use amari_probabilistic::distribution::GaussianMultivector;
//!
//! // Create Bayesian model with Jeffreys prior
//! let prior = JeffreysPrior::new();
//! let likelihood = GaussianMultivector::<3, 0, 0>::standard();
//! let model = BayesianGA::new(prior, likelihood);
//!
//! // Observe data and compute posterior
//! let observations = vec![sample1, sample2, sample3];
//! let posterior = model.posterior(&observations)?;
//! ```

mod posterior;

pub use posterior::{BayesianGA, JeffreysPrior, PosteriorSampler};
