//! Checked `term!`/`rule!` macro behavior (0.25 Cohort 1, Task 5).
//!
//! Grammar contract: lowercase-leading identifiers and string literals
//! are symbols (`term!(f)` == `term!("f")`); uppercase-leading
//! identifiers are variables; applications parenthesize arguments.
//! `rule!(lhs => rhs)` expands to fully qualified
//! `amari_rewrite::trs::Rule::new` — checked construction, never
//! unchecked. Compile-time grammar rejections live in the trybuild
//! fixtures under `amari-rewrite-macros/tests/ui/`.

#![cfg(feature = "macros")]

use amari_rewrite::trs::{Rule, Term};
use amari_rewrite::{rule, term};

#[test]
fn constants_and_symbols() {
    assert_eq!(term!(zero), Term::constant("zero"));
    assert_eq!(term!("zero"), Term::constant("zero"));
    // Identifier and string spellings are equivalent.
    assert_eq!(term!(f), term!("f"));
    // A bare parenthesized symbol is the same constant.
    assert_eq!(term!(f()), term!(f));
}

#[test]
fn variables_are_uppercase_identifiers() {
    assert_eq!(term!(X), Term::var("X"));
    assert_eq!(term!(Big), Term::var("Big"));
    assert_ne!(term!(x), Term::var("x")); // lowercase x is a symbol
}

#[test]
fn nested_applications() {
    let expected = Term::sym("add", vec![Term::constant("zero"), Term::var("X")]);
    assert_eq!(term!(add(zero, X)), expected);

    let nested = term!(f(g(X), "h", zero()));
    let expected = Term::sym(
        "f",
        vec![
            Term::sym("g", vec![Term::var("X")]),
            Term::constant("h"),
            Term::constant("zero"),
        ],
    );
    assert_eq!(nested, expected);
}

#[test]
fn rule_constructs_checked_trs_rule() {
    let built = rule!(add(zero, X) => X).unwrap();
    assert_eq!(built.lhs(), &term!(add(zero, X)));
    assert_eq!(built.rhs(), &term!(X));
}

#[test]
fn rule_rhs_variables_are_checked() {
    // Y never occurs on the left: checked construction must refuse.
    assert!(rule!(f(X) => g(Y)).is_err());
    assert!(rule!(f(X) => g(X)).is_ok());
}

#[test]
fn rule_macro_resolves_trs_rule_not_ars_rule() {
    // Both Rule types in scope; the macro output is the TRS rule.
    use amari_rewrite::ars::Rule as _;
    fn assert_trs_rule(_: &Rule) {}
    let built = rule!(f(X) => X).unwrap();
    assert_trs_rule(&built);
}
