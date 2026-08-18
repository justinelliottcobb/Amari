//! Bidirectional system contract tests (0.25 Cohort 2, Task 11).

use amari_rewrite::relation::RelationLimits;
use amari_rewrite::reversible::BidirectionalSystem;
use amari_rewrite::trs::{Rule, Term};
use amari_rewrite::RewriteError;

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

fn limits() -> RelationLimits {
    RelationLimits::default()
}

fn root() -> amari_rewrite::rewritable::Path {
    amari_rewrite::rewritable::Path::root()
}

fn rules(pairs: &[(&str, &str)]) -> Vec<Rule> {
    pairs
        .iter()
        .map(|(l, r)| Rule::new(parse(l), parse(r)).unwrap())
        .collect()
}

#[test]
fn construction_rechecks_every_rule() {
    let sys = BidirectionalSystem::new(rules(&[("f(X, Y)", "g(X)"), ("h(X)", "k(X)")])).unwrap();
    assert_eq!(sys.rules().len(), 2);
    // Clauses are compiled up front: erased variables are known.
    assert_eq!(sys.rules()[0].clause().erased_variables(), &["Y"]);
    assert!(sys.rules()[1].clause().erased_variables().is_empty());
}

#[test]
fn forward_step_with_optional_residual() {
    let sys = BidirectionalSystem::new(rules(&[("f(X, Y)", "g(X)")])).unwrap();
    let rule_id = sys.rules()[0].rule_id();
    let source = parse("f(a, b)");

    let with = sys
        .forward_step(&source, &rule_id, &root(), true, &limits())
        .unwrap();
    assert_eq!(with.target, parse("g(a)"));
    let residual = with.residual.expect("residual requested");
    assert_eq!(residual.erased_bindings[0].1, parse("b"));

    let without = sys
        .forward_step(&source, &rule_id, &root(), false, &limits())
        .unwrap();
    assert_eq!(without.target, with.target);
    assert!(without.residual.is_none());
}

#[test]
fn forward_step_rejects_unknown_identity_and_nonmatching_position() {
    let sys = BidirectionalSystem::new(rules(&[("f(X)", "g(X)")])).unwrap();
    let bogus = amari_rewrite::relation::RuleId::from_rule(
        &Rule::new(parse("p(X)"), parse("q(X)")).unwrap(),
    );
    assert!(matches!(
        sys.forward_step(&parse("f(a)"), &bogus, &root(), true, &limits()),
        Err(RewriteError::ResidualMismatch { .. })
            | Err(RewriteError::NoRule { .. })
            | Err(RewriteError::InvalidRule { .. })
    ));
    let rule_id = sys.rules()[0].rule_id();
    assert!(matches!(
        sys.forward_step(&parse("c"), &rule_id, &root(), true, &limits()),
        Err(RewriteError::NoRule { .. })
    ));
}

#[test]
fn symbolic_predecessor_relation_via_system() {
    let sys = BidirectionalSystem::new(rules(&[("f(X, Y)", "g(X)")])).unwrap();
    let preds = sys.predecessors(&parse("g(a)"), &limits(), 61).unwrap();
    assert_eq!(preds.len(), 1);
    assert_eq!(preds[0].existentials.len(), 1);
    assert_eq!(preds[0].term.positions().len(), 3);
}

#[test]
fn exact_backward_replay_via_system() {
    let sys = BidirectionalSystem::new(rules(&[("f(X, Y)", "g(X)")])).unwrap();
    let rule_id = sys.rules()[0].rule_id();
    let source = parse("f(a, b)");
    let step = sys
        .forward_step(&source, &rule_id, &root(), true, &limits())
        .unwrap();
    let residual = step.residual.unwrap();
    let back = sys.replay(&residual, &step.target, &limits()).unwrap();
    assert_eq!(back, source);
    // Replay rejects tampering through the same path.
    let mut forged = residual.clone();
    forged.erased_bindings[0].1 = parse("c");
    assert!(matches!(
        sys.replay(&forged, &step.target, &limits()),
        Err(RewriteError::ResidualMismatch { .. })
    ));
    // A residual from a different system does not replay here.
    let other = BidirectionalSystem::new(rules(&[("h(X)", "g(X)")])).unwrap();
    assert!(matches!(
        other.replay(&residual, &step.target, &limits()),
        Err(RewriteError::ResidualMismatch { .. })
    ));
}

#[test]
fn steps_compose_through_typed_derivations() {
    let sys = BidirectionalSystem::new(rules(&[("f(X)", "g(X)"), ("g(X)", "h(X)")])).unwrap();
    let ids: Vec<_> = sys.rules().iter().map(|r| r.rule_id()).collect();
    let s0 = parse("f(a)");
    let step1 = sys
        .forward_step(&s0, &ids[0], &root(), true, &limits())
        .unwrap();
    let step2 = sys
        .forward_step(&step1.target, &ids[1], &root(), true, &limits())
        .unwrap();
    assert_eq!(step2.target, parse("h(a)"));
    // Replay in reverse order reconstructs the original source.
    let s1 = sys
        .replay(&step2.residual.unwrap(), &step2.target, &limits())
        .unwrap();
    assert_eq!(s1, step1.target);
    let back = sys
        .replay(&step1.residual.unwrap(), &s1, &limits())
        .unwrap();
    assert_eq!(back, s0);
}

#[test]
fn reversibility_report_is_structural_and_honest() {
    let sys = BidirectionalSystem::new(rules(&[
        ("f(X)", "g(X)"),
        ("f(X, Y)", "g(X)"),
        ("h(X)", "k(X)"),
    ]))
    .unwrap();
    let report = sys.reversibility();
    assert_eq!(report.rules.len(), 3);
    // h(X) -> k(X): lossless and unambiguous.
    let lossless = &report.rules[2];
    assert!(lossless.erased_variables.is_empty());
    // f(X,Y) -> g(X) is lossy; f(X) -> g(X) shares a unifiable RHS.
    let erased = &report.rules[1];
    assert_eq!(erased.erased_variables, vec!["Y".to_string()]);
    assert!(!report.rules[0].rhs_ambiguity.is_empty());
    assert!(!report.rules[1].rhs_ambiguity.is_empty());
}
