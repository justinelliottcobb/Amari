// SPDX-License-Identifier: MIT OR Apache-2.0

//! The derive must resolve `amari-rewrite` through the user's chosen
//! dependency name (`rewrite_lib` here) via proc-macro-crate.

use amari_rewrite_macros::Rewritable;

#[derive(Clone, Debug, PartialEq, Rewritable)]
enum Expr {
    Zero,
    Succ(#[rewritable(child)] Box<Expr>),
}

fn main() {
    let one = Expr::Succ(Box::new(Expr::Zero));
    assert_eq!(rewrite_lib::Rewritable::child_count(&one), 1);
    assert_eq!(
        rewrite_lib::Rewritable::child(&one, 0),
        Some(&Expr::Zero)
    );

    // term!/rule! must resolve through the renamed crate as well.
    let lhs = amari_rewrite_macros::term!(add(zero, X));
    let built = amari_rewrite_macros::rule!(add(zero, X) => X).unwrap();
    assert_eq!(built.lhs(), &lhs);
    assert_eq!(built.rhs(), &amari_rewrite_macros::term!(X));
}
