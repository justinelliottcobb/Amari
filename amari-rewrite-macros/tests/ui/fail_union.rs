// SPDX-License-Identifier: MIT OR Apache-2.0

use amari_rewrite_macros::Rewritable;

#[derive(Rewritable)]
union Number {
    integer: u32,
    float: f32,
}

fn main() {}
