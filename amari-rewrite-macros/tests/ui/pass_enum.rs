// SPDX-License-Identifier: MIT OR Apache-2.0

//! Enum expression tree: unit, tuple, and named-field variants with
//! unmarked payload fields cloned through replacement.

use amari_rewrite_macros::Rewritable;

#[derive(Clone, Debug, PartialEq, Rewritable)]
enum Expr {
    Num(i64),
    Neg(#[rewritable(child)] Box<Expr>),
    Add {
        #[rewritable(child)]
        lhs: Box<Expr>,
        #[rewritable(child)]
        rhs: Box<Expr>,
        note: String,
    },
}

fn main() {
    let tree = Expr::Add {
        lhs: Box::new(Expr::Num(1)),
        rhs: Box::new(Expr::Neg(Box::new(Expr::Num(2)))),
        note: "kept".to_string(),
    };
    assert_eq!(amari_rewrite::Rewritable::child_count(&tree), 2);
    let one = amari_rewrite::Rewritable::child(&tree, 1).unwrap();
    assert_eq!(amari_rewrite::Rewritable::child_count(one), 1);
}
