// SPDX-License-Identifier: MIT OR Apache-2.0

//! Struct shapes: a leaf struct and a tuple struct carrying children
//! (compile-only; boxed self-recursion cannot terminate at runtime).

use amari_rewrite_macros::Rewritable;

#[derive(Clone, Debug, PartialEq, Rewritable)]
struct Meta {
    name: String,
    arity: usize,
}

#[derive(Clone, Debug, PartialEq, Rewritable)]
struct Pair(
    #[rewritable(child)] Box<Pair>,
    #[rewritable(child)] Box<Pair>,
);

fn main() {
    let meta = Meta {
        name: "const".to_string(),
        arity: 0,
    };
    assert_eq!(amari_rewrite::Rewritable::child_count(&meta), 0);
    let _ = std::mem::size_of::<Pair>();
}
