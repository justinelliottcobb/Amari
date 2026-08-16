// SPDX-License-Identifier: MIT OR Apache-2.0

//! Logic variables and deterministic freshening namespaces.
//!
//! [`LogicVar`] is distinct from display-level [`crate::trs::Variable`]
//! names: it carries a query scope and an index so backward-relation
//! machinery never collides with user-named variables. Every backward
//! application allocates from a deterministic query-scoped namespace;
//! canonical serialization (see [`crate::relation::Sha256Digest`])
//! renumbers by first structural occurrence, so caller names and
//! traversal accidents cannot change hashes.

use core::fmt;

/// A freshened logic variable: `(scope, index)` within a query.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct LogicVar {
    scope: u32,
    index: u32,
}

impl LogicVar {
    /// Construct a logic variable directly ( deserialization and
    /// fixed-point testing; freshening goes through
    /// [`LogicVarNamespace`]).
    pub fn new(scope: u32, index: u32) -> Self {
        Self { scope, index }
    }

    /// The query scope this variable was freshened in.
    pub fn scope(&self) -> u32 {
        self.scope
    }

    /// The allocation index within the scope.
    pub fn index(&self) -> u32 {
        self.index
    }
}

impl fmt::Display for LogicVar {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "?{}.{}", self.scope, self.index)
    }
}

/// Deterministic query-scoped logic-variable allocator.
///
/// Two namespaces constructed with the same scope yield identical
/// sequences; different scopes yield disjoint identities.
#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct LogicVarNamespace {
    scope: u32,
    next: u32,
}

impl LogicVarNamespace {
    /// Open a namespace for one query scope.
    pub fn new(scope: u32) -> Self {
        Self { scope, next: 0 }
    }

    /// Allocate the next logic variable in this scope.
    pub fn fresh(&mut self) -> LogicVar {
        let var = LogicVar::new(self.scope, self.next);
        self.next = self.next.saturating_add(1);
        var
    }

    /// Number of variables allocated so far.
    pub fn allocated(&self) -> u32 {
        self.next
    }
}
