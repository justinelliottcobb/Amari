// SPDX-License-Identifier: MIT OR Apache-2.0

//! Relation/reversibility probe parity: symbolic predecessors,
//! inverse analysis, and residual round trip (0.25 Cohort 2, Task 12).

#![cfg(feature = "standard-probes")]

use amari_discovery::{
    ProbeEngine, RewriteInverseAnalysisOutput, RewriteInverseAnalysisRequest,
    RewriteResidualReplayOutput, RewriteResidualReplayRequest, RewriteRule,
    RewriteSymbolicPredecessorsOutput, RewriteSymbolicPredecessorsRequest, RewriteTerm,
};
use amari_rewrite::{
    inverse::symbolic_predecessors,
    relation::RelationLimits,
    reversible::BidirectionalSystem,
    trs::{Rule, Term, TermSystem},
};

const SYMBOLIC: &str = "amari-probe:rewrite:symbolic-predecessors:v1";
const ANALYSIS: &str = "amari-probe:rewrite:inverse-analysis:v1";
const REPLAY: &str = "amari-probe:rewrite:residual-replay:v1";

fn var(name: &str) -> RewriteTerm {
    RewriteTerm::Variable {
        name: name.to_owned(),
    }
}

fn sym(name: &str, arguments: Vec<RewriteTerm>) -> RewriteTerm {
    RewriteTerm::Symbol {
        name: name.to_owned(),
        arguments,
    }
}

fn rule(lhs: RewriteTerm, rhs: RewriteTerm) -> RewriteRule {
    RewriteRule { lhs, rhs }
}

fn lossy_rules() -> Vec<RewriteRule> {
    vec![rule(
        sym("f", vec![var("X"), var("Y")]),
        sym("g", vec![var("X")]),
    )]
}

fn native_rules(dtos: &[RewriteRule]) -> Vec<Rule> {
    dtos.iter()
        .map(|rule| Rule::new(dto_term(&rule.lhs), dto_term(&rule.rhs)).unwrap())
        .collect()
}

fn dto_term(dto: &RewriteTerm) -> Term {
    match dto {
        RewriteTerm::Variable { name } => Term::var(name.clone()),
        RewriteTerm::Symbol { name, arguments } => {
            let args: Vec<Term> = arguments.iter().map(dto_term).collect();
            Term::sym(name.clone(), args)
        }
    }
}

#[test]
fn symbolic_predecessors_match_the_library_relation() {
    let request = RewriteSymbolicPredecessorsRequest {
        target: sym("g", vec![sym("a", vec![])]),
        rules: lossy_rules(),
        max_results: 16,
    };
    let engine = ProbeEngine::new().unwrap();
    let first = engine
        .execute(
            &SYMBOLIC.parse().unwrap(),
            &serde_json::to_value(&request).unwrap(),
        )
        .unwrap();
    let second = engine
        .execute(
            &SYMBOLIC.parse().unwrap(),
            &serde_json::to_value(&request).unwrap(),
        )
        .unwrap();
    assert_eq!(first, second);

    let output: RewriteSymbolicPredecessorsOutput = serde_json::from_value(first.output).unwrap();
    let system = TermSystem::new(native_rules(&request.rules));
    let expected = symbolic_predecessors(
        &system,
        &dto_term(&request.target),
        &RelationLimits::default(),
        0,
    )
    .unwrap();
    assert_eq!(output.predecessors.len(), expected.len());
    assert_eq!(output.predecessors.len(), 1);
    // f(X, ?0.0): one existential, target-hash provenance present.
    assert_eq!(output.predecessors[0].existentials.len(), 1);
    assert_eq!(output.predecessors[0].provenance.target_hash.len(), 64);
    assert!(!output.truncated);
}

#[test]
fn symbolic_predecessor_input_is_strict_and_bounded() {
    let engine = ProbeEngine::new().unwrap();
    // Unknown fields are rejected.
    let mut value = serde_json::to_value(RewriteSymbolicPredecessorsRequest {
        target: sym("a", vec![]),
        rules: lossy_rules(),
        max_results: 4,
    })
    .unwrap();
    value["extra"] = serde_json::json!(true);
    assert!(engine.execute(&SYMBOLIC.parse().unwrap(), &value).is_err());
    // max_results cap is enforced.
    let mut value = serde_json::to_value(RewriteSymbolicPredecessorsRequest {
        target: sym("a", vec![]),
        rules: lossy_rules(),
        max_results: 1,
    })
    .unwrap();
    value["max_results"] = serde_json::json!(u64::MAX);
    assert!(engine.execute(&SYMBOLIC.parse().unwrap(), &value).is_err());
}

#[test]
fn inverse_analysis_matches_the_library_report() {
    let request = RewriteInverseAnalysisRequest {
        rules: vec![
            rule(sym("f", vec![var("X")]), sym("g", vec![var("X")])),
            rule(sym("f", vec![var("X"), var("Y")]), sym("g", vec![var("X")])),
            rule(sym("h", vec![var("X")]), sym("k", vec![var("X")])),
        ],
    };
    let engine = ProbeEngine::new().unwrap();
    let execution = engine
        .execute(
            &ANALYSIS.parse().unwrap(),
            &serde_json::to_value(&request).unwrap(),
        )
        .unwrap();
    let output: RewriteInverseAnalysisOutput = serde_json::from_value(execution.output).unwrap();

    let native = BidirectionalSystem::new(native_rules(&request.rules)).unwrap();
    let report = amari_rewrite::analysis::InverseAnalyzer::analyze(&native);
    assert_eq!(output.rules.len(), report.rules.len());
    assert_eq!(output.rules[0].backward, "lossless");
    assert_eq!(output.rules[1].backward, "existential");
    assert_eq!(output.rules[1].erased_variables, vec!["Y".to_owned()]);
    assert_eq!(output.rules[0].reversibility, "ambiguous");
    assert_eq!(output.rules[1].reversibility, "ambiguous");
    assert_eq!(output.rules[2].reversibility, "reversible");
    // Rule identities are the canonical 64-hex rule digests.
    assert!(output.rules.iter().all(|rule| rule.rule_id.len() == 64));
}

#[test]
fn residual_replay_round_trips_with_authority() {
    let request = RewriteResidualReplayRequest {
        source: sym("f", vec![sym("a", vec![]), sym("b", vec![])]),
        rules: lossy_rules(),
        rule_index: 0,
        path: vec![],
    };
    let engine = ProbeEngine::new().unwrap();
    let first = engine
        .execute(
            &REPLAY.parse().unwrap(),
            &serde_json::to_value(&request).unwrap(),
        )
        .unwrap();
    let second = engine
        .execute(
            &REPLAY.parse().unwrap(),
            &serde_json::to_value(&request).unwrap(),
        )
        .unwrap();
    assert_eq!(first, second);

    let output: RewriteResidualReplayOutput = serde_json::from_value(first.output).unwrap();
    assert_eq!(output.target, sym("g", vec![sym("a", vec![])]));
    assert_eq!(output.reconstructed, request.source);
    assert!(output.matches_source);
    assert_eq!(output.residual.erased_bindings.len(), 1);
    assert_eq!(output.residual.erased_bindings[0].term, sym("b", vec![]));
    assert_eq!(output.residual.rule_id.len(), 64);
    assert_eq!(output.residual.source_hash.len(), 64);
}

#[test]
fn residual_replay_rejects_bad_index_path_and_match() {
    let engine = ProbeEngine::new().unwrap();
    let base = RewriteResidualReplayRequest {
        source: sym("f", vec![sym("a", vec![]), sym("b", vec![])]),
        rules: lossy_rules(),
        rule_index: 0,
        path: vec![],
    };
    // Out-of-range rule index.
    let mut bad = base.clone();
    bad.rule_index = 9;
    assert!(engine
        .execute(
            &REPLAY.parse().unwrap(),
            &serde_json::to_value(&bad).unwrap()
        )
        .is_err());
    // Invalid path.
    let mut bad = base.clone();
    bad.path = vec![5];
    assert!(engine
        .execute(
            &REPLAY.parse().unwrap(),
            &serde_json::to_value(&bad).unwrap()
        )
        .is_err());
    // Path where the rule does not match.
    let mut bad = base;
    bad.path = vec![0];
    assert!(engine
        .execute(
            &REPLAY.parse().unwrap(),
            &serde_json::to_value(&bad).unwrap()
        )
        .is_err());
}
