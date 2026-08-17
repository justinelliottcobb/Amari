// SPDX-License-Identifier: MIT OR Apache-2.0

//! Finite grounding domains and safe existential grounding.
//!
//! A [`GroundingDomain`] is a caller-supplied finite ranked symbol
//! domain with depth/count ceilings (fixed ceilings: 256 symbols,
//! rank 16, depth 16, 65,536 emitted terms). Enumeration is canonical:
//! ascending node count, then head-symbol (name, arity) order, then
//! lexicographic children. [`ground_predecessor`] assigns a
//! predecessor's existentials from the domain in canonical order and
//! validates constraints and forward replay before emitting — every
//! emitted grounding is replay-validated by construction.

use alloc::collections::BTreeSet;
use alloc::string::{String, ToString};
use alloc::vec;
use alloc::vec::Vec;

use crate::error::{RewriteError, RewriteResult};
use crate::inverse::SymbolicPredecessor;
use crate::relation::{RelationLimits, RelationResources, TermConstraint};
use crate::trs::{Substitution, Term, TermSystem, Variable};

/// A ranked domain symbol: a name with a fixed arity.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct RankedSymbol {
    name: String,
    arity: usize,
}

impl RankedSymbol {
    /// A ranked symbol (rank validated by [`GroundingDomain::new`]).
    pub fn new(name: impl Into<String>, arity: usize) -> Self {
        Self {
            name: name.into(),
            arity,
        }
    }

    /// Symbol name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Fixed arity.
    pub fn arity(&self) -> usize {
        self.arity
    }
}

/// A validated finite ranked symbol domain.
#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct GroundingDomain {
    /// Sorted by (name, arity) at construction: declaration order
    /// never affects enumeration.
    symbols: Vec<RankedSymbol>,
    max_depth: usize,
    max_terms: usize,
}

impl GroundingDomain {
    /// Hard ceiling on ranked symbols per domain.
    pub const MAX_SYMBOLS: usize = 256;
    /// Hard ceiling on symbol rank.
    pub const MAX_RANK: usize = 16;
    /// Hard ceiling on enumerated term depth.
    pub const MAX_DEPTH: usize = 16;
    /// Hard ceiling on emitted terms.
    pub const MAX_TERMS: usize = 65_536;

    /// Validate a domain. Empty domains, domains without constants (no
    /// finite term is constructible), duplicate ranked symbols, and
    /// zero/above-ceiling depth or term counts all fail before any
    /// allocation happens.
    pub fn new(
        symbols: Vec<RankedSymbol>,
        max_depth: usize,
        max_terms: usize,
    ) -> RewriteResult<Self> {
        let invalid =
            |resource: &'static str, value: usize, ceiling: usize| RewriteError::InvalidLimit {
                resource,
                value,
                ceiling,
            };
        if symbols.is_empty() || symbols.len() > Self::MAX_SYMBOLS {
            return Err(invalid(
                "grounding symbols",
                symbols.len(),
                Self::MAX_SYMBOLS,
            ));
        }
        if let Some(symbol) = symbols.iter().find(|s| s.arity > Self::MAX_RANK) {
            return Err(invalid("grounding rank", symbol.arity, Self::MAX_RANK));
        }
        if !symbols.iter().any(|s| s.arity == 0) {
            return Err(RewriteError::InvalidRule {
                message: String::from("grounding domain needs at least one constant symbol"),
            });
        }
        let mut sorted = symbols;
        sorted.sort();
        if sorted.windows(2).any(|pair| pair[0] == pair[1]) {
            return Err(RewriteError::InvalidRule {
                message: String::from("grounding domain contains duplicate ranked symbols"),
            });
        }
        if max_depth == 0 || max_depth > Self::MAX_DEPTH {
            return Err(invalid("grounding depth", max_depth, Self::MAX_DEPTH));
        }
        if max_terms == 0 || max_terms > Self::MAX_TERMS {
            return Err(invalid("grounding terms", max_terms, Self::MAX_TERMS));
        }
        Ok(Self {
            symbols: sorted,
            max_depth,
            max_terms,
        })
    }

    /// The validated ranked symbols, in canonical (name, arity) order.
    pub fn symbols(&self) -> &[RankedSymbol] {
        &self.symbols
    }

    /// Enumerate domain terms in canonical order: ascending node
    /// count, then head-symbol (name, arity) order, then
    /// lexicographic children. At most `max_terms` terms are emitted;
    /// every generated candidate is accounted against `resources`.
    ///
    /// The sweep covers term sizes `1..=max_terms`. If the operation
    /// budget is exhausted mid-sweep, enumeration stops and returns
    /// the canonical prefix; callers detect truncation through
    /// `resources.operations()` (grounding reports it as
    /// [`GroundingOutcome::Partial`]).
    pub fn enumerate(&self, resources: &mut RelationResources) -> RewriteResult<Vec<Term>> {
        let mut out: Vec<Term> = Vec::new();
        for size in 1..=self.max_terms {
            if out.len() >= self.max_terms {
                break;
            }
            for symbol in &self.symbols {
                if out.len() >= self.max_terms {
                    break;
                }
                match self.enumerate_headed(symbol, size, self.max_depth, resources, &mut out) {
                    Ok(()) => {}
                    Err(RewriteError::RelationLimitExceeded { .. }) => {
                        return Ok(out);
                    }
                    Err(error) => return Err(error),
                }
            }
        }
        Ok(out)
    }

    /// Generate all terms of exactly `size` nodes with head `symbol`,
    /// depth at most `depth_left`, in canonical child order.
    fn enumerate_headed(
        &self,
        symbol: &RankedSymbol,
        size: usize,
        depth_left: usize,
        resources: &mut RelationResources,
        out: &mut Vec<Term>,
    ) -> RewriteResult<()> {
        resources.record_operations(1)?;
        if symbol.arity == 0 {
            if size == 1 {
                resources.record_term(1, 1)?;
                if out.len() < self.max_terms {
                    out.push(Term::constant(symbol.name.clone()));
                }
            }
            return Ok(());
        }
        if depth_left < 2 || size < symbol.arity + 1 {
            return Ok(());
        }
        let mut child_sizes = vec![1usize; symbol.arity];
        self.enumerate_compositions(
            symbol,
            size,
            depth_left,
            resources,
            out,
            &mut child_sizes,
            0,
        )
    }

    /// Lexicographic compositions of `size - 1` nodes into
    /// `symbol.arity` child sizes, each at least 1.
    #[allow(clippy::too_many_arguments)]
    fn enumerate_compositions(
        &self,
        symbol: &RankedSymbol,
        size: usize,
        depth_left: usize,
        resources: &mut RelationResources,
        out: &mut Vec<Term>,
        child_sizes: &mut Vec<usize>,
        position: usize,
    ) -> RewriteResult<()> {
        if out.len() >= self.max_terms {
            return Ok(());
        }
        if position + 1 == symbol.arity {
            let used: usize = child_sizes[..position].iter().sum();
            let remaining = size - 1;
            if used >= remaining {
                return Ok(());
            }
            child_sizes[position] = remaining - used;
            return self.enumerate_product(symbol, child_sizes, depth_left, resources, out);
        }
        let remaining_positions = symbol.arity - position - 1;
        let used: usize = child_sizes[..position].iter().sum();
        let max_here = size - 1 - used - remaining_positions;
        for child_size in 1..=max_here {
            if out.len() >= self.max_terms {
                return Ok(());
            }
            child_sizes[position] = child_size;
            self.enumerate_compositions(
                symbol,
                size,
                depth_left,
                resources,
                out,
                child_sizes,
                position + 1,
            )?;
        }
        Ok(())
    }

    /// Lexicographic product of per-child enumerations for a fixed
    /// child-size composition.
    fn enumerate_product(
        &self,
        symbol: &RankedSymbol,
        child_sizes: &[usize],
        depth_left: usize,
        resources: &mut RelationResources,
        out: &mut Vec<Term>,
    ) -> RewriteResult<()> {
        let mut per_child: Vec<Vec<Term>> = Vec::with_capacity(child_sizes.len());
        for child_size in child_sizes {
            let mut child_terms = Vec::new();
            for child_symbol in &self.symbols {
                self.enumerate_headed(
                    child_symbol,
                    *child_size,
                    depth_left - 1,
                    resources,
                    &mut child_terms,
                )?;
            }
            if child_terms.is_empty() {
                return Ok(());
            }
            per_child.push(child_terms);
        }
        let mut indices = vec![0usize; per_child.len()];
        loop {
            if out.len() >= self.max_terms {
                return Ok(());
            }
            let args: Vec<Term> = per_child
                .iter()
                .zip(indices.iter())
                .map(|(terms, index)| terms[*index].clone())
                .collect();
            let nodes = 1 + child_sizes.iter().sum::<usize>();
            resources.record_term(nodes, self.max_depth)?;
            out.push(Term::sym(symbol.name.clone(), args));
            // Odometer increment, rightmost fastest: lexicographic.
            let mut position = indices.len();
            loop {
                if position == 0 {
                    return Ok(());
                }
                position -= 1;
                indices[position] += 1;
                if indices[position] < per_child[position].len() {
                    break;
                }
                indices[position] = 0;
            }
        }
    }
}

/// One replay-validated grounded predecessor.
#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct GroundedPredecessor {
    /// The existential assignment (freshened variable → domain term).
    pub substitution: Substitution,
    /// The fully grounded predecessor term.
    pub term: Term,
}

/// Typed grounding result: `Complete` only when the canonical
/// enumeration finished without truncation.
#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub enum GroundingOutcome {
    /// Every canonical assignment within the domain was validated.
    Complete {
        /// Replay-validated grounded predecessors, in canonical order.
        grounded: Vec<GroundedPredecessor>,
    },
    /// A budget stopped enumeration; `grounded` is a strict canonical
    /// prefix and never claims exhaustiveness.
    Partial {
        /// Replay-validated grounded predecessors emitted so far.
        grounded: Vec<GroundedPredecessor>,
    },
}

/// Ground a symbolic predecessor's existentials over a finite domain.
///
/// Assignments enumerate in canonical order; every candidate is
/// validated against the predecessor's constraints and by forward
/// replay (the grounded predecessor must rewrite to the grounded
/// target through the original system) before emission. Duplicate
/// grounded terms are suppressed. Budget exhaustion yields
/// [`GroundingOutcome::Partial`], never an exhaustiveness claim.
pub fn ground_predecessor(
    system: &TermSystem,
    target: &Term,
    predecessor: &SymbolicPredecessor,
    domain: &GroundingDomain,
    limits: &RelationLimits,
) -> RewriteResult<GroundingOutcome> {
    let mut resources = RelationResources::new(limits);
    let terms = domain.enumerate(&mut resources)?;
    let mut partial = resources.operations() >= limits.max_operations();

    let mut names: Vec<String> = predecessor
        .existentials
        .iter()
        .map(|var| var.to_string())
        .collect();
    names.sort();
    names.dedup();

    let mut grounded: Vec<GroundedPredecessor> = Vec::new();
    let mut seen: BTreeSet<Term> = BTreeSet::new();

    // Canonical assignment product over the existential names. No
    // existentials means exactly one (identity) candidate.
    if names.is_empty() {
        try_assignment(
            system,
            target,
            predecessor,
            &Substitution::new(),
            &mut grounded,
            &mut seen,
        )?;
    } else if terms.is_empty() {
        partial = true;
    } else {
        let mut indices = vec![0usize; names.len()];
        'assignments: loop {
            if resources.operations() >= limits.max_operations() {
                partial = true;
                break;
            }
            resources.record_operations(1)?;
            let mut assignment = Substitution::new();
            for (name, index) in names.iter().zip(indices.iter()) {
                assignment.insert(Variable::new(name.clone()), terms[*index].clone());
            }
            match try_assignment(
                system,
                target,
                predecessor,
                &assignment,
                &mut grounded,
                &mut seen,
            ) {
                Ok(()) => {}
                Err(RewriteError::RelationLimitExceeded { .. }) => {
                    partial = true;
                    break;
                }
                Err(error) => return Err(error),
            }
            // Odometer increment, rightmost fastest: canonical product.
            let mut position = names.len();
            loop {
                if position == 0 {
                    break 'assignments;
                }
                position -= 1;
                indices[position] += 1;
                if indices[position] < terms.len() {
                    break;
                }
                indices[position] = 0;
            }
        }
    }

    if partial {
        Ok(GroundingOutcome::Partial { grounded })
    } else {
        Ok(GroundingOutcome::Complete { grounded })
    }
}

/// Validate one assignment by constraints and forward replay; emit on
/// success, suppress duplicates.
fn try_assignment(
    system: &TermSystem,
    target: &Term,
    predecessor: &SymbolicPredecessor,
    assignment: &Substitution,
    grounded: &mut Vec<GroundedPredecessor>,
    seen: &mut BTreeSet<Term>,
) -> RewriteResult<()> {
    let constraints_ok = predecessor
        .constraints
        .constraints()
        .iter()
        .all(|c| match c {
            TermConstraint::Equal(left, right) => assignment.apply(left) == assignment.apply(right),
            TermConstraint::NotEqual(left, right) => {
                assignment.apply(left) != assignment.apply(right)
            }
        });
    if !constraints_ok {
        return Ok(());
    }
    let grounded_term = assignment.apply(&predecessor.term);
    let grounded_target = assignment.apply(&predecessor.substitution.apply(target));
    let replays = system
        .successors(&grounded_term)?
        .contains(&grounded_target);
    if replays && seen.insert(grounded_term.clone()) {
        grounded.push(GroundedPredecessor {
            substitution: assignment.clone(),
            term: grounded_term,
        });
    }
    Ok(())
}
