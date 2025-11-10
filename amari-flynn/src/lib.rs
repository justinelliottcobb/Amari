//! # Amari Flynn - Probabilistic Contracts and Verification
//!
//! Named after Kevin Flynn from Tron: Legacy, who discovered that spontaneous,
//! imperfect emergence (the ISOs) represented a form of perfection beyond rigid
//! determinism. This library embodies that philosophy: formal verification should
//! prove what's impossible while allowing rare, emergent possibilities.
//!
//! ## Philosophy: Three Types of Events
//!
//! Flynn teaches us to distinguish between three categories of events:
//!
//! 1. **Impossible** (P=0): Formally proven to never occur
//!    - Violates mathematical axioms
//!    - Proved unreachable via formal verification
//!    - System invariants guarantee exclusion
//!
//! 2. **Rare** (0 < P << 1): Bounded probability, tracked and verified
//!    - Low but non-zero probability
//!    - Statistical bounds verified via Monte Carlo
//!    - Tracked as legitimate possibilities
//!
//! 3. **Emergent** (P > 0): Possible but not prescribed, enabling discovery
//!    - Not predicted or designed
//!    - Arise spontaneously from system rules
//!    - The "ISOs" of your system
//!
//! ## Core Concept
//!
//! Traditional verification asks: "Can this happen?"
//! Flynn asks: "Is this impossible, rare, or merely unpredicted?"
//!
//! This library provides tools to:
//! - Prove what cannot happen (P=0)
//! - Bound what rarely happens (P<<1)
//! - Enable what might emerge (P>0)
//!
//! ## Quick Start
//!
//! ```rust
//! use amari_flynn::prelude::*;
//!
//! // Create probabilistic values
//! let coin_flip = Prob::with_probability(0.5, true);
//!
//! // Sample from distributions
//! let die_roll = Uniform::new(1, 6).sample();
//!
//! // Track rare events
//! let miracle_shot = RareEvent::<()>::new(0.001, "critical_hit");
//! ```
//!
//! ## Integration with Amari
//!
//! With the `geometric` feature, Flynn integrates with Amari's geometric algebra
//! types, enabling probabilistic contracts over geometric computations.
//!
//! ## Roadmap
//!
//! - **Why3 Integration**: Formal verification of probability bounds
//! - **Creusot Support**: Rust-native formal verification
//! - **SMT Backend**: Automated theorem proving for event impossibility
//!
//! ## The ISO Philosophy
//!
//! Like the ISOs in Tron: Legacy, the most valuable behaviors in a system
//! are often those that emerge spontaneously, unpredicted by design. Flynn
//! enables you to prove safety boundaries while preserving space for emergence.
//!
//! > "The ISOs, they were a miracle. They weren't meant to be - they just were."
//! >
//! > - Kevin Flynn
//!
//! ## Example: Game Mechanics Verification
//!
//! ```rust
//! use amari_flynn::prelude::*;
//!
//! // Verify critical hit rate is bounded
//! let crit_rate = 0.15;
//! let crit_prob = Prob::with_probability(crit_rate, ());
//!
//! // Statistical verification that P(crit) ≤ 0.20
//! // (leaving room for buffs/modifiers while maintaining balance)
//! ```

#![warn(missing_docs)]
#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(feature = "std")]
extern crate std;

pub mod backend;
pub mod contracts;
pub mod distributions;
pub mod prob;
pub mod statistical;

/// Prelude module for common imports
pub mod prelude {
    pub use crate::backend::monte_carlo::MonteCarloVerifier;
    pub use crate::contracts::{
        EventVerification, ProbabilisticContract, RareEvent, StatisticalProperty,
        VerificationResult,
    };
    pub use crate::distributions::{Bernoulli, Exponential, Normal, Uniform};
    pub use crate::prob::Prob;
}

// Re-export procedural macros
pub use amari_flynn_macros::*;
