// SPDX-License-Identifier: MIT OR Apache-2.0

use amari_rewrite_macros::Rewritable;

#[derive(Clone, Debug, PartialEq, Rewritable)]
enum Expr {
    Zero,
    Succ(
        #[rewritable(child)]
        #[rewritable(child)]
        Box<Expr>,
    ),
}

fn main() {}
