// SPDX-License-Identifier: MIT OR Apache-2.0

use amari_rewrite_macros::rule;

fn main() {
    let _ = rule!(f(X) -> g(X));
}
