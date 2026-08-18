// SPDX-License-Identifier: MIT OR Apache-2.0

//! Typed residuals for exact backward replay (0.25 Cohort 2).
//!
//! A normal rewrite may erase information. Exact reversal uses a
//! [`RewriteResidual`] produced during forward execution: rule
//! identity, position, erased bindings (LHS variables absent from the
//! RHS), and source/target authority hashes. Replay validates every
//! field, reconstructs privately, compares the canonical source hash,
//! and returns only on exact match.

mod residual;
mod step;

pub use residual::RewriteResidual;
pub use step::{ForwardTransition, ReversibleStep};
