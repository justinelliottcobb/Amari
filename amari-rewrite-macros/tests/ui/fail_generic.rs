// SPDX-License-Identifier: MIT OR Apache-2.0

use amari_rewrite_macros::Rewritable;

#[derive(Clone, Debug, PartialEq, Rewritable)]
struct Wrapper<T> {
    value: T,
}

fn main() {}
