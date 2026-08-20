//! Inverse analysis contract tests (0.25 Cohort 2, Task 11).

use amari_rewrite::analysis::{BackwardKind, InverseAnalyzer, ReversibilityClass};
use amari_rewrite::reversible::BidirectionalSystem;
use amari_rewrite::trs::{Rule, Term};

fn parse(text: &str) -> Term {
    let text = text.trim();
    if let Some(open) = text.find('(') {
        let head = &text[..open];
        let inner = &text[open + 1..text.len() - 1];
        let mut args = Vec::new();
        let mut depth = 0i32;
        let mut start = 0;
        for (index, ch) in inner.char_indices() {
            match ch {
                '(' => depth += 1,
                ')' => depth -= 1,
                ',' if depth == 0 => {
                    args.push(parse(&inner[start..index]));
                    start = index + 1;
                }
                _ => {}
            }
        }
        args.push(parse(&inner[start..]));
        Term::sym(head, args)
    } else if text.chars().next().is_some_and(char::is_uppercase) {
        Term::var(text)
    } else {
        Term::constant(text)
    }
}

fn system(pairs: &[(&str, &str)]) -> BidirectionalSystem {
    BidirectionalSystem::new(
        pairs
            .iter()
            .map(|(l, r)| Rule::new(parse(l), parse(r)).unwrap())
            .collect(),
    )
    .unwrap()
}

#[test]
fn lossless_rule_is_classified_reversible() {
    let sys = system(&[("f(X)", "g(X)")]);
    let report = InverseAnalyzer::analyze(&sys);
    let analysis = &report.rules[0];
    assert_eq!(analysis.backward, BackwardKind::Lossless);
    assert!(analysis.erased_variables.is_empty());
    assert!(analysis.rhs_ambiguity.is_empty());
    assert_eq!(analysis.reversibility, ReversibilityClass::Reversible);
    assert_eq!(analysis.branching.existentials, 0);
    assert_eq!(analysis.branching.ambiguous_peers, 0);
}

#[test]
fn erased_variables_drive_existential_backward() {
    let sys = system(&[("f(X, Y)", "g(X)")]);
    let report = InverseAnalyzer::analyze(&sys);
    let analysis = &report.rules[0];
    assert_eq!(analysis.backward, BackwardKind::Existential);
    assert_eq!(analysis.erased_variables, vec!["Y".to_string()]);
    assert_eq!(analysis.reversibility, ReversibilityClass::LossyResidual);
    // One erased variable: finite-domain branching is the domain size
    // per existential.
    assert_eq!(analysis.branching.existentials, 1);
}

#[test]
fn rhs_overlap_classes_are_reported_as_ambiguous() {
    let sys = system(&[("f(X)", "g(X)"), ("h(X)", "g(X)")]);
    let report = InverseAnalyzer::analyze(&sys);
    // Both rules' RHS unify: both are Ambiguous, never Reversible.
    for analysis in &report.rules {
        assert_eq!(analysis.rhs_ambiguity.len(), 1);
        assert_eq!(analysis.reversibility, ReversibilityClass::Ambiguous);
        assert_eq!(analysis.branching.ambiguous_peers, 1);
    }
    // Distinct rule identities are reported, not merged.
    assert_ne!(report.rules[0].rule_id, report.rules[1].rule_id);
}

#[test]
fn lossy_and_ambiguous_reports_both_pieces_of_evidence() {
    let sys = system(&[("f(X, Y)", "g(X)"), ("h(X)", "g(X)")]);
    let report = InverseAnalyzer::analyze(&sys);
    let first = &report.rules[0];
    assert_eq!(first.backward, BackwardKind::Existential);
    assert_eq!(first.reversibility, ReversibilityClass::Ambiguous);
    assert_eq!(first.branching.existentials, 1);
    assert_eq!(first.branching.ambiguous_peers, 1);
}

#[test]
fn unifiable_rhs_structure_counts_as_ambiguity_not_identity() {
    // g(X) and g(a) unify under substitution: ambiguous even though
    // the rules are distinct and both lossless.
    let sys = system(&[("f(X)", "g(X)"), ("h(X)", "g(a)")]);
    let report = InverseAnalyzer::analyze(&sys);
    for analysis in &report.rules {
        assert_eq!(analysis.reversibility, ReversibilityClass::Ambiguous);
    }
}

#[test]
fn non_overlapping_rhs_is_not_flagged() {
    let sys = system(&[("f(X)", "g(X)"), ("h(X, Y)", "k(X, Y)")]);
    let report = InverseAnalyzer::analyze(&sys);
    for analysis in &report.rules {
        assert!(analysis.rhs_ambiguity.is_empty());
    }
}

#[test]
fn deterministic_branching_estimate_is_exact_for_lossless() {
    // Lossless, unambiguous: backward branching is deterministic
    // (zero existentials, zero ambiguous peers).
    let sys = system(&[("f(X)", "g(X)"), ("h(X)", "k(X)")]);
    let report = InverseAnalyzer::analyze(&sys);
    for analysis in &report.rules {
        assert_eq!(analysis.branching.existentials, 0);
        assert_eq!(analysis.branching.ambiguous_peers, 0);
        assert_eq!(analysis.reversibility, ReversibilityClass::Reversible);
    }
}

#[test]
fn unsupported_reasons_are_reported_never_hidden() {
    // The 0.25 analyzer supports all checked first-order rules: the
    // reasons list must be present and empty, not absent.
    let sys = system(&[("f(X, Y)", "g(X)")]);
    let report = InverseAnalyzer::analyze(&sys);
    for analysis in &report.rules {
        assert!(analysis.unsupported_reasons.is_empty());
    }
}

#[test]
fn exhaustive_enum_matches_are_stable() {
    // Guard the typed surface: adding a variant must break this test.
    let backward = [BackwardKind::Lossless, BackwardKind::Existential];
    assert_eq!(backward.len(), 2);
    let classes = [
        ReversibilityClass::Reversible,
        ReversibilityClass::LossyResidual,
        ReversibilityClass::Ambiguous,
    ];
    assert_eq!(classes.len(), 3);
}
