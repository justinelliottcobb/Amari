//! Why3 formal verification backend (stub)
//!
//! This module is deprecated. Use [`super::smt`] for SMT-LIB2 proof obligation generation.

#![allow(deprecated)]

/// Why3 verification generator (placeholder)
///
/// This type is a placeholder from the initial release. Use
/// [`super::smt::SmtProofObligation`] for actual formal verification output.
#[deprecated(since = "0.19.0", note = "Use amari_flynn::backend::smt instead")]
#[derive(Debug)]
pub struct Why3Generator;

impl Why3Generator {
    /// Create new Why3 generator
    pub fn new() -> Self {
        Self
    }

    /// Generate Why3 theory for probability verification
    ///
    /// This is a placeholder for future Why3 integration
    pub fn generate_probability_theory(&self, _description: &str) -> String {
        // Placeholder: Future implementation will generate Why3 theories
        "// Why3 integration coming soon\n\
         // This will generate formal proofs for probability bounds"
            .to_string()
    }
}

impl Default for Why3Generator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_placeholder() {
        let gen = Why3Generator::new();
        let theory = gen.generate_probability_theory("test");
        assert!(theory.contains("Why3"));
    }
}
