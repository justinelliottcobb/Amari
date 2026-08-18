// SPDX-License-Identifier: MIT OR Apache-2.0

//! Structural inverse analysis over bidirectional systems.
//!
//! [`InverseAnalyzer`] reports per-rule evidence: erased variables and
//! required residual schema, RHS overlap/ambiguity classes, lossless
//! versus existential backward behavior, deterministic branching
//! estimates, and reversibility classifiers. Classification is
//! conservative: a rule is `Reversible` only when it is structurally
//! lossless AND its RHS unifies with no other rule's RHS — the report
//! never claims a functional inverse without that evidence.

use alloc::string::String;
use alloc::vec::Vec;

use crate::analysis::unify::unify;
use crate::relation::{RelationLimits, RelationResources, RuleId};
use crate::reversible::BidirectionalSystem;

/// Structural backward behavior of one rule.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub enum BackwardKind {
    /// No variable is erased: every forward match is recoverable from
    /// the target alone.
    Lossless,
    /// The forward step erases variables: backward reconstruction is
    /// existential over the residual schema or a grounding domain.
    Existential,
}

/// Conservative reversibility classifier.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub enum ReversibilityClass {
    /// Lossless AND unambiguous: residual-free backward reconstruction
    /// is unique.
    Reversible,
    /// Lossless in bindings but ambiguous: the RHS overlaps another
    /// rule's RHS, so backward rule choice is not unique.
    Ambiguous,
    /// Information-losing: exact reversal requires residual authority
    /// or grounding over the erased variables.
    LossyResidual,
}

/// Structural branching estimate for backward exploration.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct BranchingEstimate {
    /// Existential choices introduced by erased variables; over a
    /// finite grounding domain of size `d`, each existential
    /// multiplies the branching by `d`.
    pub existentials: usize,
    /// Other rules whose RHS unifies with this rule's RHS.
    pub ambiguous_peers: usize,
}

/// Per-rule inverse analysis with stable evidence.
#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct RuleAnalysis {
    /// Identity of the analyzed rule.
    pub rule_id: RuleId,
    /// LHS variables absent from the RHS (the residual schema).
    pub erased_variables: Vec<String>,
    /// Lossless versus existential backward behavior.
    pub backward: BackwardKind,
    /// Other rules whose RHS unifies with this rule's RHS.
    pub rhs_ambiguity: Vec<RuleId>,
    /// Conservative reversibility classifier.
    pub reversibility: ReversibilityClass,
    /// Structural branching estimate.
    pub branching: BranchingEstimate,
    /// Unsupported or unknown reasons. Empty for checked first-order
    /// rules in 0.25; when non-empty, the classifier above was
    /// reached with the listed caveats.
    pub unsupported_reasons: Vec<String>,
}

/// System-wide inverse report.
#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct InverseReport {
    /// Per-rule analysis, in system rule order.
    pub rules: Vec<RuleAnalysis>,
}

/// Structural analyzer over [`BidirectionalSystem`]s.
pub struct InverseAnalyzer;

impl InverseAnalyzer {
    /// Analyze every wrapped rule. Infallible: construction already
    /// checked the rules and compiled their clauses; RHS non-overlap
    /// is evidence, not an error.
    pub fn analyze(system: &BidirectionalSystem) -> InverseReport {
        let rules = system.rules();
        // Pairwise RHS unifiability: ambiguity classes.
        let mut ambiguity: Vec<Vec<RuleId>> = rules.iter().map(|_| Vec::new()).collect();
        // RHS overlap probing is structural evidence, not budgeted
        // search: default relation resources are sufficient because
        // checked rules are already size-bounded.
        let limits = RelationLimits::default();
        let mut resources = RelationResources::new(&limits);
        for i in 0..rules.len() {
            for j in (i + 1)..rules.len() {
                let overlap = unify(
                    rules[i].forward().rhs(),
                    rules[j].forward().rhs(),
                    &mut resources,
                )
                .is_ok();
                if overlap {
                    ambiguity[i].push(rules[j].rule_id());
                    ambiguity[j].push(rules[i].rule_id());
                }
            }
        }
        let analyses = rules
            .iter()
            .enumerate()
            .map(|(index, rule)| {
                let erased: Vec<String> = rule.clause().erased_variables().to_vec();
                let peers = ambiguity[index].clone();
                let backward = if erased.is_empty() {
                    BackwardKind::Lossless
                } else {
                    BackwardKind::Existential
                };
                let reversibility = if !peers.is_empty() {
                    ReversibilityClass::Ambiguous
                } else if !erased.is_empty() {
                    ReversibilityClass::LossyResidual
                } else {
                    ReversibilityClass::Reversible
                };
                RuleAnalysis {
                    rule_id: rule.rule_id(),
                    erased_variables: erased.clone(),
                    backward,
                    rhs_ambiguity: peers.clone(),
                    reversibility,
                    branching: BranchingEstimate {
                        existentials: erased.len(),
                        ambiguous_peers: peers.len(),
                    },
                    unsupported_reasons: Vec::new(),
                }
            })
            .collect();
        InverseReport { rules: analyses }
    }
}
