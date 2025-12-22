//! Sampling algorithms for multivector distributions
//!
//! This module provides MCMC and other sampling methods for distributions
//! over geometric algebra spaces.
//!
//! # Algorithms
//!
//! - **Metropolis-Hastings**: General-purpose MCMC with configurable proposals
//! - **Hamiltonian Monte Carlo**: Uses geometric product for momentum updates
//! - **Rejection Sampling**: For distributions with tractable bounds
//! - **Importance Sampling**: For expectation estimation
//!
//! # Example
//!
//! ```ignore
//! use amari_probabilistic::sampling::{MetropolisHastings, Sampler};
//! use amari_probabilistic::distribution::GaussianMultivector;
//!
//! let target = GaussianMultivector::<3, 0, 0>::standard();
//! let proposal_std = 0.1;
//! let sampler = MetropolisHastings::new(&target, proposal_std);
//!
//! let mut rng = rand::thread_rng();
//! let samples = sampler.run(&mut rng, 10000, 1000)?; // 10k samples, 1k burnin
//! ```

mod mcmc;

pub use mcmc::{MCMCDiagnostics, MetropolisHastings, Sampler, SamplerState};
