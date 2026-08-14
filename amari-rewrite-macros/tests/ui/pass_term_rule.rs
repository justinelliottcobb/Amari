// SPDX-License-Identifier: MIT OR Apache-2.0

//! term!/rule! in the external-crate resolution path.

use amari_rewrite_macros::{rule, term};

fn main() {
    let lhs = term!(add(zero, X));
    let built = rule!(add(zero, X) => X).unwrap();
    assert_eq!(built.lhs(), &lhs);
    assert_eq!(built.rhs(), &term!(X));
    assert!(rule!(f(X) => g(Y)).is_err());
}
