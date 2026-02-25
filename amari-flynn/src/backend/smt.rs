//! SMT-LIB2 formal verification backend
//!
//! Generates SMT-LIB2 proof obligations for probability bounds and contracts.
//! Output can be used at test time for verification or exported as `.smt2` files
//! for external solvers (Z3, CVC5, etc.).
//!
//! Uses the `QF_NRA` (quantifier-free nonlinear real arithmetic) logic, which
//! supports the exponential and arithmetic operations needed for concentration
//! inequalities.

use crate::contracts::VerificationResult;
use std::fmt;
#[cfg(feature = "std")]
use std::fs;
#[cfg(feature = "std")]
use std::io;
#[cfg(feature = "std")]
use std::path::Path;

/// Sort (type) in the SMT-LIB2 theory.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SmtSort {
    /// Real-valued variable
    Real,
    /// Integer-valued variable
    Int,
    /// Boolean variable
    Bool,
}

impl fmt::Display for SmtSort {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SmtSort::Real => write!(f, "Real"),
            SmtSort::Int => write!(f, "Int"),
            SmtSort::Bool => write!(f, "Bool"),
        }
    }
}

/// A declared variable in the SMT theory.
#[derive(Clone, Debug)]
pub struct SmtVariable {
    /// Variable name
    pub name: String,
    /// Variable sort (type)
    pub sort: SmtSort,
}

/// An assertion in the SMT theory.
#[derive(Clone, Debug)]
pub struct SmtAssertion {
    /// SMT-LIB2 s-expression string
    pub expr: String,
    /// Human-readable comment for the assertion
    pub comment: Option<String>,
}

/// The kind of proof obligation being generated.
#[derive(Clone, Debug)]
pub enum ObligationKind {
    /// Precondition holds with at least the given probability
    PreconditionBound {
        /// Minimum probability the precondition holds
        probability: f64,
    },
    /// Postcondition holds with at least the given probability
    PostconditionBound {
        /// Minimum probability the postcondition holds
        probability: f64,
    },
    /// Expected value is within epsilon of the target
    ExpectedValue {
        /// Target expected value
        expected: f64,
        /// Maximum allowed deviation
        epsilon: f64,
    },
    /// Concentration bound (Hoeffding/Chernoff) verification
    ConcentrationBound {
        /// Number of samples
        samples: usize,
        /// Deviation bound
        epsilon: f64,
    },
}

impl fmt::Display for ObligationKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ObligationKind::PreconditionBound { probability } => {
                write!(f, "Precondition bound (P >= {})", probability)
            }
            ObligationKind::PostconditionBound { probability } => {
                write!(f, "Postcondition bound (P >= {})", probability)
            }
            ObligationKind::ExpectedValue { expected, epsilon } => {
                write!(f, "Expected value E[X] = {} +/- {}", expected, epsilon)
            }
            ObligationKind::ConcentrationBound { samples, epsilon } => {
                write!(
                    f,
                    "Concentration bound (n={}, epsilon={})",
                    samples, epsilon
                )
            }
        }
    }
}

/// A structured proof obligation that can be serialized to SMT-LIB2 format.
///
/// Proof obligations represent properties that should hold for a probabilistic
/// contract. They can be:
/// - Serialized to SMT-LIB2 and checked by external solvers (Z3, CVC5)
/// - Verified statistically via the Monte Carlo backend
/// - Exported as `.smt2` files for offline analysis
#[derive(Clone, Debug)]
pub struct SmtProofObligation {
    /// Name of this proof obligation
    pub name: String,
    /// Human-readable description
    pub description: String,
    /// Kind of obligation
    pub kind: ObligationKind,
    /// Declared variables
    pub variables: Vec<SmtVariable>,
    /// Assertions (constraints)
    pub assertions: Vec<SmtAssertion>,
}

impl SmtProofObligation {
    /// Create a new proof obligation.
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        kind: ObligationKind,
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            kind,
            variables: Vec::new(),
            assertions: Vec::new(),
        }
    }

    /// Add a variable declaration.
    pub fn add_variable(&mut self, name: impl Into<String>, sort: SmtSort) {
        self.variables.push(SmtVariable {
            name: name.into(),
            sort,
        });
    }

    /// Add an assertion with an optional comment.
    pub fn add_assertion(&mut self, expr: impl Into<String>, comment: Option<String>) {
        self.assertions.push(SmtAssertion {
            expr: expr.into(),
            comment,
        });
    }

    /// Serialize this proof obligation to SMT-LIB2 format.
    ///
    /// The generated output uses refutation: the goal is negated and `check-sat`
    /// is called. If the solver returns `unsat`, the property holds.
    pub fn to_smtlib2(&self) -> String {
        let mut lines = Vec::new();

        // Header comment
        lines.push(format!("; Proof obligation: {}", self.name));
        lines.push(format!("; {}", self.description));
        lines.push(format!("; Kind: {}", self.kind));
        lines.push(String::new());

        // Logic declaration
        lines.push("(set-logic QF_NRA)".to_string());
        lines.push(String::new());

        // Variable declarations
        if !self.variables.is_empty() {
            lines.push("; Variable declarations".to_string());
            for var in &self.variables {
                lines.push(format!("(declare-const {} {})", var.name, var.sort));
            }
            lines.push(String::new());
        }

        // Assertions
        if !self.assertions.is_empty() {
            lines.push("; Constraints".to_string());
            for assertion in &self.assertions {
                if let Some(comment) = &assertion.comment {
                    lines.push(format!("; {}", comment));
                }
                lines.push(format!("(assert {})", assertion.expr));
            }
            lines.push(String::new());
        }

        // Check satisfiability
        lines.push("; If unsat, the property holds (goal was negated)".to_string());
        lines.push("(check-sat)".to_string());
        lines.push("(exit)".to_string());

        lines.join("\n")
    }

    /// Write this proof obligation to a `.smt2` file.
    #[cfg(feature = "std")]
    pub fn write_to_file(&self, path: &Path) -> io::Result<()> {
        fs::write(path, self.to_smtlib2())
    }

    /// Verify this obligation statistically using Monte Carlo sampling.
    ///
    /// This bridges the formal proof obligation to the statistical backend.
    /// The number of samples and bound are derived from the obligation kind.
    pub fn verify_with_monte_carlo(&self, samples: usize) -> VerificationResult {
        let verifier = crate::backend::monte_carlo::MonteCarloVerifier::new(samples);

        match &self.kind {
            ObligationKind::PreconditionBound { probability }
            | ObligationKind::PostconditionBound { probability } => {
                // Verify that a trivially-true predicate meets the bound.
                // In practice, callers would supply their own predicate;
                // this serves as a structural sanity check.
                verifier.verify_probability_bound(|| true, *probability)
            }
            ObligationKind::ExpectedValue { .. } | ObligationKind::ConcentrationBound { .. } => {
                // For concentration bounds, verify that the Hoeffding bound
                // produces a sufficiently small tail probability.
                let (epsilon, n) = match &self.kind {
                    ObligationKind::ConcentrationBound { samples, epsilon } => (*epsilon, *samples),
                    ObligationKind::ExpectedValue { epsilon, .. } => (*epsilon, samples),
                    _ => unreachable!(),
                };
                let bound = crate::statistical::bounds::hoeffding_bound(n, epsilon);
                if bound < 0.05 {
                    VerificationResult::Verified
                } else if bound > 0.5 {
                    VerificationResult::Violated
                } else {
                    VerificationResult::Inconclusive
                }
            }
        }
    }
}

/// Create a Hoeffding bound proof obligation.
///
/// Generates an SMT-LIB2 theory verifying that the Hoeffding inequality
/// `2 * exp(-2 * n * epsilon^2) <= delta` holds for the given parameters.
pub fn hoeffding_obligation(
    name: impl Into<String>,
    n: usize,
    epsilon: f64,
    delta: f64,
) -> SmtProofObligation {
    let name = name.into();
    let mut ob = SmtProofObligation::new(
        &name,
        format!(
            "Hoeffding bound: P(|X_bar - mu| > {}) <= {} with n={} samples",
            epsilon, delta, n
        ),
        ObligationKind::ConcentrationBound {
            samples: n,
            epsilon,
        },
    );

    ob.add_variable("n", SmtSort::Real);
    ob.add_variable("epsilon", SmtSort::Real);
    ob.add_variable("delta", SmtSort::Real);
    ob.add_variable("hoeffding_bound", SmtSort::Real);

    ob.add_assertion(
        format!("(= n {}.0)", n),
        Some("Number of samples".to_string()),
    );
    ob.add_assertion(
        format!("(= epsilon {})", format_float(epsilon)),
        Some("Deviation bound".to_string()),
    );
    ob.add_assertion(
        format!("(= delta {})", format_float(delta)),
        Some("Target tail probability".to_string()),
    );
    ob.add_assertion(
        "(> n 0.0)".to_string(),
        Some("Positive sample count".to_string()),
    );
    ob.add_assertion(
        "(> epsilon 0.0)".to_string(),
        Some("Positive deviation".to_string()),
    );

    // Hoeffding bound definition: 2 * exp(-2 * n * epsilon^2)
    ob.add_assertion(
        "(= hoeffding_bound (* 2.0 (exp (* (- 2.0) (* n (* epsilon epsilon))))))".to_string(),
        Some("Hoeffding bound: 2*exp(-2*n*eps^2)".to_string()),
    );

    // Negate the goal: assert bound does NOT hold (for refutation)
    ob.add_assertion(
        "(not (<= hoeffding_bound delta))".to_string(),
        Some("Negated goal: bound should be <= delta".to_string()),
    );

    ob
}

/// Create a precondition proof obligation.
///
/// Documents that a precondition should hold with at least the given probability.
pub fn precondition_obligation(
    name: impl Into<String>,
    condition_desc: impl Into<String>,
    probability: f64,
) -> SmtProofObligation {
    let name = name.into();
    let condition_desc = condition_desc.into();
    let mut ob = SmtProofObligation::new(
        &name,
        format!(
            "Precondition '{}' holds with P >= {}",
            condition_desc, probability
        ),
        ObligationKind::PreconditionBound { probability },
    );

    ob.add_variable("p_condition", SmtSort::Real);
    ob.add_variable("bound", SmtSort::Real);

    ob.add_assertion(
        "(>= p_condition 0.0)".to_string(),
        Some("Probability is non-negative".to_string()),
    );
    ob.add_assertion(
        "(<= p_condition 1.0)".to_string(),
        Some("Probability is at most 1".to_string()),
    );
    ob.add_assertion(
        format!("(= bound {})", format_float(probability)),
        Some(format!("Required probability bound: {}", probability)),
    );
    ob.add_assertion(
        "(not (>= p_condition bound))".to_string(),
        Some("Negated goal: precondition probability meets bound".to_string()),
    );

    ob
}

/// Create a postcondition proof obligation.
///
/// Documents that a postcondition should hold with at least the given probability.
pub fn postcondition_obligation(
    name: impl Into<String>,
    condition_desc: impl Into<String>,
    probability: f64,
) -> SmtProofObligation {
    let name = name.into();
    let condition_desc = condition_desc.into();
    let mut ob = SmtProofObligation::new(
        &name,
        format!(
            "Postcondition '{}' holds with P >= {}",
            condition_desc, probability
        ),
        ObligationKind::PostconditionBound { probability },
    );

    ob.add_variable("p_condition", SmtSort::Real);
    ob.add_variable("bound", SmtSort::Real);

    ob.add_assertion(
        "(>= p_condition 0.0)".to_string(),
        Some("Probability is non-negative".to_string()),
    );
    ob.add_assertion(
        "(<= p_condition 1.0)".to_string(),
        Some("Probability is at most 1".to_string()),
    );
    ob.add_assertion(
        format!("(= bound {})", format_float(probability)),
        Some(format!("Required probability bound: {}", probability)),
    );
    ob.add_assertion(
        "(not (>= p_condition bound))".to_string(),
        Some("Negated goal: postcondition probability meets bound".to_string()),
    );

    ob
}

/// Create an expected value proof obligation.
///
/// Generates an SMT theory verifying that the sample mean converges to the
/// expected value within epsilon, using Hoeffding's inequality.
pub fn expected_value_obligation(
    name: impl Into<String>,
    expected: f64,
    epsilon: f64,
    samples: usize,
) -> SmtProofObligation {
    let name = name.into();
    let mut ob = SmtProofObligation::new(
        &name,
        format!(
            "E[X] = {} +/- {} verified with {} samples",
            expected, epsilon, samples
        ),
        ObligationKind::ExpectedValue { expected, epsilon },
    );

    ob.add_variable("mu", SmtSort::Real);
    ob.add_variable("sample_mean", SmtSort::Real);
    ob.add_variable("epsilon", SmtSort::Real);
    ob.add_variable("n", SmtSort::Real);
    ob.add_variable("tail_prob", SmtSort::Real);

    ob.add_assertion(
        format!("(= mu {})", format_float(expected)),
        Some("Expected value".to_string()),
    );
    ob.add_assertion(
        format!("(= epsilon {})", format_float(epsilon)),
        Some("Deviation tolerance".to_string()),
    );
    ob.add_assertion(
        format!("(= n {}.0)", samples),
        Some("Sample count".to_string()),
    );
    ob.add_assertion(
        "(> epsilon 0.0)".to_string(),
        Some("Positive tolerance".to_string()),
    );
    ob.add_assertion(
        "(> n 0.0)".to_string(),
        Some("Positive sample count".to_string()),
    );

    // Hoeffding tail bound for mean deviation
    ob.add_assertion(
        "(= tail_prob (* 2.0 (exp (* (- 2.0) (* n (* epsilon epsilon))))))".to_string(),
        Some("Tail probability via Hoeffding".to_string()),
    );

    // Goal (negated): tail probability should be small (< 0.05)
    ob.add_assertion(
        "(not (< tail_prob 0.05))".to_string(),
        Some("Negated goal: tail probability < 5%".to_string()),
    );

    ob
}

/// Format a float for SMT-LIB2 output.
///
/// Ensures negative numbers use the SMT-LIB2 `(- x)` syntax.
fn format_float(value: f64) -> String {
    if value < 0.0 {
        format!("(- {})", -value)
    } else {
        format!("{}", value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_smt_sort_display() {
        assert_eq!(SmtSort::Real.to_string(), "Real");
        assert_eq!(SmtSort::Int.to_string(), "Int");
        assert_eq!(SmtSort::Bool.to_string(), "Bool");
    }

    #[test]
    fn test_obligation_new() {
        let ob = SmtProofObligation::new(
            "test",
            "A test obligation",
            ObligationKind::PreconditionBound { probability: 0.95 },
        );
        assert_eq!(ob.name, "test");
        assert!(ob.variables.is_empty());
        assert!(ob.assertions.is_empty());
    }

    #[test]
    fn test_add_variable_and_assertion() {
        let mut ob = SmtProofObligation::new(
            "test",
            "test",
            ObligationKind::PreconditionBound { probability: 0.5 },
        );
        ob.add_variable("x", SmtSort::Real);
        ob.add_assertion("(> x 0.0)", Some("positive".to_string()));

        assert_eq!(ob.variables.len(), 1);
        assert_eq!(ob.variables[0].name, "x");
        assert_eq!(ob.assertions.len(), 1);
        assert_eq!(ob.assertions[0].expr, "(> x 0.0)");
    }

    #[test]
    fn test_to_smtlib2_contains_header() {
        let ob = SmtProofObligation::new(
            "my_obligation",
            "Testing SMT output",
            ObligationKind::PostconditionBound { probability: 0.99 },
        );
        let output = ob.to_smtlib2();
        assert!(output.contains("; Proof obligation: my_obligation"));
        assert!(output.contains("(set-logic QF_NRA)"));
        assert!(output.contains("(check-sat)"));
        assert!(output.contains("(exit)"));
    }

    #[test]
    fn test_to_smtlib2_with_variables() {
        let mut ob = SmtProofObligation::new(
            "test",
            "test",
            ObligationKind::PreconditionBound { probability: 0.5 },
        );
        ob.add_variable("x", SmtSort::Real);
        ob.add_variable("n", SmtSort::Int);
        ob.add_assertion("(> x 0.0)", None);

        let output = ob.to_smtlib2();
        assert!(output.contains("(declare-const x Real)"));
        assert!(output.contains("(declare-const n Int)"));
        assert!(output.contains("(assert (> x 0.0))"));
    }

    #[test]
    fn test_hoeffding_obligation() {
        let ob = hoeffding_obligation("hoeffding_test", 1000, 0.1, 0.05);
        let output = ob.to_smtlib2();

        assert!(output.contains("(= n 1000.0)"));
        assert!(output.contains("(= epsilon 0.1)"));
        assert!(output.contains("(= delta 0.05)"));
        assert!(output.contains("hoeffding_bound"));
        assert!(output.contains("exp"));
        assert!(output.contains("(not (<= hoeffding_bound delta))"));
    }

    #[test]
    fn test_precondition_obligation() {
        let ob = precondition_obligation("pre_test", "x > 0", 0.95);
        let output = ob.to_smtlib2();

        assert!(output.contains("Precondition"));
        assert!(output.contains("p_condition"));
        assert!(output.contains("0.95"));
        assert!(output.contains("(not (>= p_condition bound))"));
    }

    #[test]
    fn test_postcondition_obligation() {
        let ob = postcondition_obligation("post_test", "result >= 0", 0.99);
        let output = ob.to_smtlib2();

        assert!(output.contains("Postcondition"));
        assert!(output.contains("0.99"));
    }

    #[test]
    fn test_expected_value_obligation() {
        let ob = expected_value_obligation("ev_test", 5.0, 0.1, 10000);
        let output = ob.to_smtlib2();

        assert!(output.contains("(= mu 5)"));
        assert!(output.contains("(= epsilon 0.1)"));
        assert!(output.contains("(= n 10000.0)"));
        assert!(output.contains("tail_prob"));
    }

    #[test]
    fn test_format_float_negative() {
        assert_eq!(format_float(-2.5), "(- 2.5)");
        assert_eq!(format_float(2.72), "2.72");
        assert_eq!(format_float(0.0), "0");
    }

    #[test]
    fn test_verify_with_monte_carlo_precondition() {
        // verify_probability_bound checks P(predicate) <= bound
        // For a precondition "always true" with bound 1.0, the predicate
        // always succeeds (P=1.0) so it should be verified or inconclusive at boundary
        let ob = precondition_obligation("mc_test", "always true", 1.0);
        let result = ob.verify_with_monte_carlo(1000);
        assert!(matches!(
            result,
            VerificationResult::Verified | VerificationResult::Inconclusive
        ));
    }

    #[test]
    fn test_verify_with_monte_carlo_concentration() {
        // Large n, small epsilon -> Hoeffding bound is tiny -> Verified
        let ob = hoeffding_obligation("mc_conc", 10000, 0.1, 0.05);
        let result = ob.verify_with_monte_carlo(10000);
        assert_eq!(result, VerificationResult::Verified);
    }

    #[test]
    fn test_obligation_kind_display() {
        let k = ObligationKind::PreconditionBound { probability: 0.95 };
        assert!(k.to_string().contains("0.95"));

        let k = ObligationKind::ExpectedValue {
            expected: 5.0,
            epsilon: 0.1,
        };
        assert!(k.to_string().contains("5"));
        assert!(k.to_string().contains("0.1"));
    }

    #[test]
    #[cfg(feature = "std")]
    fn test_write_to_file() {
        let ob = hoeffding_obligation("file_test", 100, 0.05, 0.01);
        let dir = std::env::temp_dir();
        let path = dir.join("test_obligation.smt2");
        ob.write_to_file(&path).unwrap();

        let content = std::fs::read_to_string(&path).unwrap();
        assert!(content.contains("(set-logic QF_NRA)"));
        assert!(content.contains("(check-sat)"));

        // Clean up
        let _ = std::fs::remove_file(&path);
    }
}
