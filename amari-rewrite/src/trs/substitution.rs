//! Substitutions for first-order terms.

use alloc::{collections::BTreeMap, string::String};

use super::{Term, Variable};

/// A variable-to-term substitution.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct Substitution {
    map: BTreeMap<Variable, Term>,
}

impl Substitution {
    /// Create an empty substitution.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a binding, returning the updated substitution.
    pub fn with(mut self, variable: impl Into<Variable>, term: Term) -> Self {
        self.insert(variable, term);
        self
    }

    /// Insert or replace a binding.
    pub fn insert(&mut self, variable: impl Into<Variable>, term: Term) -> Option<Term> {
        self.map.insert(variable.into(), term)
    }

    /// Borrow a binding by variable.
    pub fn get(&self, variable: &str) -> Option<&Term> {
        self.map.get(&Variable::new(String::from(variable)))
    }

    /// Borrow a binding by typed variable.
    pub fn get_var(&self, variable: &Variable) -> Option<&Term> {
        self.map.get(variable)
    }

    /// Iterate bindings.
    pub fn iter(&self) -> impl Iterator<Item = (&Variable, &Term)> {
        self.map.iter()
    }

    /// Apply this substitution recursively to `term`.
    ///
    /// Bound ranges are themselves resolved (deep application), so
    /// unifier-produced substitutions whose ranges mention bound
    /// variables still normalize fully. Termination follows from the
    /// occurs check performed at binding time.
    pub fn apply(&self, term: &Term) -> Term {
        match term {
            Term::Var(var) => match self.map.get(var) {
                Some(bound) => self.apply(&bound.clone()),
                None => term.clone(),
            },
            Term::Sym(symbol, args) => Term::Sym(
                symbol.clone(),
                args.iter().map(|arg| self.apply(arg)).collect(),
            ),
        }
    }

    /// Number of bindings.
    pub fn len(&self) -> usize {
        self.map.len()
    }

    /// True when no bindings exist.
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    /// Checked composition `self ∘ inner` (apply `inner`, then `self`).
    ///
    /// Rejects incompatible bindings for the same variable and results
    /// that would not be idempotent (a domain variable occurring in a
    /// range).
    pub fn compose(&self, inner: &Self) -> crate::error::RewriteResult<Self> {
        use crate::error::RewriteError;
        let invalid = |message: &str| RewriteError::InvalidSubstitution {
            message: String::from(message),
        };
        let mut composed = Self::new();
        for (variable, term) in &inner.map {
            let through = self.apply(term);
            if let Some(outer) = self.map.get(variable) {
                if *outer != through {
                    return Err(invalid("incompatible bindings for one variable"));
                }
            }
            composed.insert(variable.clone(), through);
        }
        for (variable, term) in &self.map {
            if !inner.map.contains_key(variable) {
                composed.insert(variable.clone(), term.clone());
            }
        }
        let non_idempotent = composed
            .map
            .values()
            .flat_map(|term| term.variables())
            .any(|range| composed.map.contains_key(&range));
        if non_idempotent {
            return Err(invalid("composition would not be idempotent"));
        }
        Ok(composed)
    }
}
