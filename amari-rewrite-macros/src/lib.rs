// SPDX-License-Identifier: MIT OR Apache-2.0

//! Checked procedural macros for `amari-rewrite`.
//!
//! This crate is the only procedural-macro surface for the rewrite engine.
//! Algorithms stay in `amari-rewrite`; this crate contains parsing and code
//! generation only, so the macro crate never depends on the library crate
//! and always publishes before it.
//!
//! - `#[derive(Rewritable)]` — generates `Rewritable` implementations
//!   for structs and enums with explicit `#[rewritable(child)]` fields.
//! - `term!(f(a, X))` — checked `trs::Term` construction (Task 5).
//! - `rule!(lhs => rhs)` — checked `trs::Rule` construction (Task 5).
//! - `relation!(lhs <=> rhs)` — checked relational construction
//!   (Cohort 2).
//!
//! Entry points marked with a task expand to stable `compile_error!`
//! diagnostics naming the implementing task until they land.

use proc_macro::TokenStream;

mod rewritable_derive;

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

/// Derive `Rewritable` for an expression tree.
///
/// Children are exactly the fields marked `#[rewritable(child)]`, in
/// declaration order; recursion is never inferred. A child field's type
/// must deref to `Self` and rebuild from `Self` (`Box<Self>`,
/// `Rc<Self>`, `Arc<Self>`); unmarked fields are cloned through
/// replacement and must be `Clone`. Unions, generics, collections as
/// children, and malformed attributes are compile errors at precise
/// spans.
#[proc_macro_derive(Rewritable, attributes(rewritable))]
pub fn derive_rewritable(input: TokenStream) -> TokenStream {
    let input = syn::parse_macro_input!(input as syn::DeriveInput);
    rewritable_derive::expand(input).into()
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
