// SPDX-License-Identifier: MIT OR Apache-2.0

//! Deterministic first-order unification with occurs check.
//!
//! The worklist algorithm resolves both sides through the
//! accumulating substitution before every step, so bindings stay
//! idempotent by construction. Canonical binding order comes from the
//! substitution's `BTreeMap`; operation and term limits are enforced
//! through [`RelationResources`].

use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;

use crate::error::{RewriteError, RewriteResult};
use crate::relation::RelationResources;
use crate::trs::{Substitution, Term, Variable};

/// Most-general unifier of two terms under a fresh substitution.
pub fn unify(
    left: &Term,
    right: &Term,
    resources: &mut RelationResources,
) -> RewriteResult<Substitution> {
    let mut substitution = Substitution::new();
    unify_with(&mut substitution, left, right, resources)?;
    Ok(substitution)
}

/// Extend an existing substitution so `left` and `right` become equal.
///
/// On error the substitution may be partially extended; callers treat
/// failure as terminal for the current goal.
pub fn unify_with(
    substitution: &mut Substitution,
    left: &Term,
    right: &Term,
    resources: &mut RelationResources,
) -> RewriteResult<()> {
    let mut worklist: Vec<(Term, Term)> = vec![(left.clone(), right.clone())];
    while let Some((lhs, rhs)) = worklist.pop() {
        resources.record_operations(1)?;
        let lhs = resolve(substitution, &lhs);
        let rhs = resolve(substitution, &rhs);
        match (&lhs, &rhs) {
            (Term::Var(x), Term::Var(y)) if x == y => {}
            (Term::Var(x), term) => {
                bind(substitution, x, term.clone(), resources)?;
            }
            (term, Term::Var(y)) => {
                bind(substitution, y, term.clone(), resources)?;
            }
            (Term::Sym(f, f_args), Term::Sym(g, g_args)) => {
                if f != g {
                    return Err(failure("head symbol clash"));
                }
                if f_args.len() != g_args.len() {
                    return Err(failure("arity mismatch"));
                }
                for (f_arg, g_arg) in f_args.iter().zip(g_args.iter()) {
                    worklist.push((f_arg.clone(), g_arg.clone()));
                }
            }
        }
    }
    Ok(())
}

fn failure(reason: &str) -> RewriteError {
    RewriteError::UnificationFailure {
        reason: String::from(reason),
    }
}

/// Resolve a term through the substitution until the root stabilizes.
fn resolve(substitution: &Substitution, term: &Term) -> Term {
    match term {
        Term::Var(variable) => match substitution.get_var(variable) {
            Some(bound) => resolve(substitution, &bound.clone()),
            None => term.clone(),
        },
        Term::Sym(..) => term.clone(),
    }
}

fn bind(
    substitution: &mut Substitution,
    variable: &Variable,
    term: Term,
    resources: &mut RelationResources,
) -> RewriteResult<()> {
    resources.record_term(term_nodes(&term), term_depth(&term))?;
    if occurs(substitution, variable, &term) {
        return Err(failure("occurs check violation"));
    }
    // `resolve` guarantees the variable is unbound here, and the term
    // is resolved, so insertion preserves idempotence.
    substitution.insert(variable.clone(), term);
    Ok(())
}

fn occurs(substitution: &Substitution, variable: &Variable, term: &Term) -> bool {
    let resolved = resolve(substitution, term);
    match &resolved {
        Term::Var(other) => other == variable,
        Term::Sym(_, args) => args.iter().any(|arg| occurs(substitution, variable, arg)),
    }
}

/// Count term nodes (variables and symbols).
pub(crate) fn term_nodes(term: &Term) -> usize {
    match term {
        Term::Var(_) => 1,
        Term::Sym(_, args) => 1 + args.iter().map(term_nodes).sum::<usize>(),
    }
}

/// Measure term depth (leaf = 1).
pub(crate) fn term_depth(term: &Term) -> usize {
    match term {
        Term::Var(_) => 1,
        Term::Sym(_, args) => 1 + args.iter().map(term_depth).max().unwrap_or(0),
    }
}
