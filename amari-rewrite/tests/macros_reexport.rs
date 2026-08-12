//! Macro crate wiring proofs (0.25 Cohort 1, Task 3).
//!
//! Under the `macros` feature, `amari-rewrite` re-exports the procedural
//! macro surface from `amari-rewrite-macros`. Without the feature, the macro
//! crate must not be compiled at all (proved by dependency-tree checks in CI
//! review evidence; this file compiles to nothing without `macros`).

#![cfg(feature = "macros")]

/// The derive macro is re-exported at the crate root so user code can write
/// `#[derive(amari_rewrite::Rewritable)]` without depending on the macro
/// crate directly.
#[test]
fn rewritable_derive_is_reexported() {
    // Existence proof: naming the derive macro as an item path compiles only
    // if the re-export exists. Invoking it on user types arrives in Task 4.
    #[allow(unused)]
    use amari_rewrite::Rewritable;
}

/// The checked constructor macros are re-exported at the crate root.
#[test]
fn constructor_macros_are_reexported() {
    // Existence proof via macro namespace paths. Checked expansion behavior
    // arrives in Task 5.
    #[allow(unused)]
    use amari_rewrite::{relation, rule, term};
}
