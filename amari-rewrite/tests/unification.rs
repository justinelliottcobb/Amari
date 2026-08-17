//! Unification and checked substitution contract tests
//! (0.25 Cohort 2, Task 7).
//!
//! Deterministic first-order unification with occurs check, canonical
//! binding order, checked idempotent composition, and strict
//! term/operation limits. Shared by critical pairs, inverse frontier
//! meeting, constraints, and synthesis.

use amari_rewrite::analysis::unify;
use amari_rewrite::relation::{RelationLimits, RelationResources};
use amari_rewrite::trs::{Substitution, Term};
use amari_rewrite::RewriteError;

fn resources() -> RelationResources {
    RelationResources::new(&RelationLimits::default())
}

#[test]
fn mgu_binds_variables_deterministically() {
    let mut budget = resources();
    let subst = unify(&Term::var("X"), &Term::constant("a"), &mut budget).unwrap();
    assert_eq!(subst.get("X"), Some(&Term::constant("a")));
    assert_eq!(subst.len(), 1);

    // Variable-variable: first side binds to the second.
    let mut budget = resources();
    let subst = unify(&Term::var("X"), &Term::var("Y"), &mut budget).unwrap();
    assert_eq!(subst.get("X"), Some(&Term::var("Y")));

    // Structured terms bind both variables.
    let mut budget = resources();
    let subst = unify(
        &Term::sym("f", vec![Term::var("X"), Term::constant("b")]),
        &Term::sym("f", vec![Term::constant("a"), Term::var("Y")]),
        &mut budget,
    )
    .unwrap();
    assert_eq!(subst.get("X"), Some(&Term::constant("a")));
    assert_eq!(subst.get("Y"), Some(&Term::constant("b")));
}

#[test]
fn mgu_makes_sides_equal() {
    let left = Term::sym(
        "f",
        vec![Term::var("X"), Term::sym("g", vec![Term::var("X")])],
    );
    let right = Term::sym(
        "f",
        vec![Term::sym("h", vec![Term::var("Y")]), Term::var("Z")],
    );
    let mut budget = resources();
    let subst = unify(&left, &right, &mut budget).unwrap();
    assert_eq!(subst.apply(&left), subst.apply(&right));
}

#[test]
fn occurs_check_rejects_cyclic_bindings() {
    let mut budget = resources();
    let result = unify(
        &Term::var("X"),
        &Term::sym("f", vec![Term::var("X")]),
        &mut budget,
    );
    assert!(matches!(
        result,
        Err(RewriteError::UnificationFailure { .. })
    ));

    // Nested: X ~ f(Y), Y ~ g(X) is cyclic through the substitution.
    let mut budget = resources();
    let result = unify(
        &Term::sym("f", vec![Term::var("X"), Term::var("Y")]),
        &Term::sym(
            "f",
            vec![
                Term::sym("g", vec![Term::var("Y")]),
                Term::sym("h", vec![Term::var("X")]),
            ],
        ),
        &mut budget,
    );
    assert!(matches!(
        result,
        Err(RewriteError::UnificationFailure { .. })
    ));
}

#[test]
fn repeated_variables_and_shape_mismatches_fail() {
    // f(X, X) cannot equal f(a, b).
    let mut budget = resources();
    let result = unify(
        &Term::sym("f", vec![Term::var("X"), Term::var("X")]),
        &Term::sym("f", vec![Term::constant("a"), Term::constant("b")]),
        &mut budget,
    );
    assert!(matches!(
        result,
        Err(RewriteError::UnificationFailure { .. })
    ));

    // Arity mismatch.
    let mut budget = resources();
    let result = unify(
        &Term::sym("f", vec![Term::var("X")]),
        &Term::sym("f", vec![Term::var("X"), Term::var("Y")]),
        &mut budget,
    );
    assert!(matches!(
        result,
        Err(RewriteError::UnificationFailure { .. })
    ));

    // Head-symbol clash.
    let mut budget = resources();
    let result = unify(
        &Term::sym("f", vec![Term::var("X")]),
        &Term::sym("g", vec![Term::var("X")]),
        &mut budget,
    );
    assert!(matches!(
        result,
        Err(RewriteError::UnificationFailure { .. })
    ));
}

#[test]
fn alpha_renamed_pairs_unify_isomorphically() {
    let left = Term::sym("f", vec![Term::var("X"), Term::var("X")]);
    let right = Term::sym("f", vec![Term::var("Y"), Term::var("Y")]);
    let mut budget = resources();
    let subst = unify(&left, &right, &mut budget).unwrap();
    assert_eq!(subst.apply(&left), subst.apply(&right));
    // Exactly one binding: X ↦ Y.
    assert_eq!(subst.len(), 1);
}

#[test]
fn checked_composition_combines_and_validates() {
    // {Y ↦ X} then {X ↦ a} composes to {Y ↦ a, X ↦ a}.
    let inner = Substitution::new().with("Y", Term::var("X"));
    let outer = Substitution::new().with("X", Term::constant("a"));
    let composed = outer.compose(&inner).unwrap();
    assert_eq!(composed.get("Y"), Some(&Term::constant("a")));
    assert_eq!(composed.get("X"), Some(&Term::constant("a")));

    // Incompatible bindings for the same variable are rejected.
    let a = Substitution::new().with("X", Term::constant("a"));
    let b = Substitution::new().with("X", Term::constant("b"));
    assert!(matches!(
        a.compose(&b),
        Err(RewriteError::InvalidSubstitution { .. })
    ));

    // Non-idempotent results are rejected: {X ↦ f(Y)} ∘ {Y ↦ X}.
    let inner = Substitution::new().with("Y", Term::var("X"));
    let outer = Substitution::new().with("X", Term::sym("f", vec![Term::var("Y")]));
    assert!(matches!(
        outer.compose(&inner),
        Err(RewriteError::InvalidSubstitution { .. })
    ));
}

#[test]
fn unification_respects_operation_limits() {
    let limits = RelationLimits::new(4_096, 64, 4_096, 2).unwrap();
    let mut budget = RelationResources::new(&limits);
    // Three argument pairs exceed a two-operation budget.
    let result = unify(
        &Term::sym(
            "f",
            vec![
                Term::constant("a"),
                Term::constant("b"),
                Term::constant("c"),
            ],
        ),
        &Term::sym(
            "f",
            vec![
                Term::constant("a"),
                Term::constant("b"),
                Term::constant("c"),
            ],
        ),
        &mut budget,
    );
    assert!(matches!(
        result,
        Err(RewriteError::RelationLimitExceeded { .. })
    ));
}
