// SPDX-License-Identifier: MIT OR Apache-2.0

//! Checked procedural macros for `amari-rewrite`.
//!
//! This crate is the only procedural-macro surface for the rewrite engine.
//! Algorithms stay in `amari-rewrite`; this crate contains parsing and code
//! generation only, so the macro crate never depends on the library crate
//! and always publishes before it.
//!
//! # Surface (0.25 scaffold)
//!
//! - `#[derive(Rewritable)]` — generates `Rewritable` implementations for
//!   enum expression trees with explicit `#[rewritable(child)]` fields.
//! - `term!(f(a, X))` — checked `trs::Term` construction.
//! - `rule!(lhs => rhs)` — checked `trs::Rule` construction.
//! - `relation!(lhs <=> rhs)` — checked relational construction.
//!
//! Entry points currently expand to stable `compile_error!` diagnostics
//! naming the implementing task; checked expansion lands in Tasks 4–5 of the
//! 0.25 inverse-rewrite expansion plan.

use proc_macro::TokenStream;

fn not_yet(implemented_in: &str) -> TokenStream {
    let message = format!(
        "this macro is scaffolded but not yet implemented; checked expansion \
         lands in {implemented_in} of the amari-rewrite 0.25 expansion plan"
    );
    let diagnostic = quote::quote! {
        ::core::compile_error!(#message);
    };
    diagnostic.into()
}

/// Derive `Rewritable` for an enum expression tree.
///
/// Only named-field enums with explicit `#[rewritable(child)]` annotations
/// will be supported; unsupported shapes will fail at compile time so term
/// structure cannot silently drift from the checked contract.
#[proc_macro_derive(Rewritable, attributes(rewritable))]
pub fn derive_rewritable(_input: TokenStream) -> TokenStream {
    not_yet("Task 4")
}

/// Checked `trs::Term` construction: `term!(add(zero, X))`.
///
/// Identifier and string symbol spellings are equivalent. Expansion is to
/// checked constructors; malformed terms are compile errors.
#[proc_macro]
pub fn term(_input: TokenStream) -> TokenStream {
    not_yet("Task 5")
}

/// Checked `trs::Rule` construction: `rule!(add(zero, X) => X)`.
///
/// Resolves the fully qualified TRS rule, not the ARS `Rule` type.
#[proc_macro]
pub fn rule(_input: TokenStream) -> TokenStream {
    not_yet("Task 5")
}

/// Checked relational construction: `relation!(add(zero, X) <=> X)`.
///
/// Expands only to checked declarative constructors for the constrained
/// rewrite-relation model (Cohort 2).
#[proc_macro]
pub fn relation(_input: TokenStream) -> TokenStream {
    not_yet("Cohort 2")
}
