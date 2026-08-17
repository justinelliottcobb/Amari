// SPDX-License-Identifier: MIT OR Apache-2.0

//! Analysis infrastructure for rewrite systems.
//!
//! Currently holds the deterministic first-order unifier shared by
//! critical pairs, inverse frontier meeting, constraint solving, and
//! synthesis. Critical pairs, LPO, and confluence land in later
//! cohorts.

mod unify;

// The module and its primary entry point share the name `unify`;
// they live in different namespaces, so `analysis::unify(...)` calls
// the function while `analysis::unify::unify` remains the module path.
pub use self::unify::{unify, unify_with};
