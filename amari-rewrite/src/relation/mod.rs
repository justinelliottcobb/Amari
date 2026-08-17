// SPDX-License-Identifier: MIT OR Apache-2.0

//! Constrained rewrite-relation foundations (0.25 Cohort 2).
//!
//! This module holds the authority types every backward-relation
//! feature builds on: deterministic query-scoped [`LogicVarNamespace`]
//! allocation of [`LogicVar`]s, caller-tightenable [`RelationLimits`]
//! against fixed ceilings, runtime [`RelationResources`] accounting,
//! and framed [`Sha256Digest`] canonical identity. Unification,
//! constraint normalization, and backward clauses build on these in
//! later tasks.

mod clause;
mod constraints;
pub(crate) mod digest;
mod ground;
mod limits;
mod variable;

pub use clause::{BackwardClause, RuleId};
pub use constraints::{ConstraintOutcome, ConstraintSet, TermConstraint};
pub use digest::Sha256Digest;
pub use ground::{
    ground_predecessor, GroundedPredecessor, GroundingDomain, GroundingOutcome, RankedSymbol,
};
pub use limits::{RelationLimits, RelationResources};
pub use variable::{LogicVar, LogicVarNamespace};
