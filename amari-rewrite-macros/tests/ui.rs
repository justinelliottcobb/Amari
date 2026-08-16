// SPDX-License-Identifier: MIT OR Apache-2.0

//! UI contracts for `derive(Rewritable)`: diagnostics, hygiene, and
//! renamed-crate support. `term!`/`rule!` fixtures arrive in Task 5.

#[test]
fn rewritable_ui() {
    let cases = trybuild::TestCases::new();
    cases.pass("tests/ui/pass_enum.rs");
    cases.pass("tests/ui/pass_struct_children.rs");
    cases.pass("tests/ui/pass_term_rule.rs");
    cases.compile_fail("tests/ui/fail_union.rs");
    cases.compile_fail("tests/ui/fail_generic.rs");
    cases.compile_fail("tests/ui/fail_duplicate_child.rs");
    cases.compile_fail("tests/ui/fail_unknown_attribute.rs");
    cases.compile_fail("tests/ui/fail_collection_child.rs");
    cases.compile_fail("tests/ui/fail_term_variable_arguments.rs");
    cases.compile_fail("tests/ui/fail_term_literal.rs");
    cases.compile_fail("tests/ui/fail_rule_arrow.rs");
    cases.compile_fail("tests/ui/fail_rule_trailing.rs");
}

/// Renamed-crate support: the fixture under `tests/renamed` depends on
/// `amari-rewrite` as `rewrite_lib`; proc-macro-crate must resolve the
/// alias. Compiled with the real cargo against path dependencies.
#[test]
fn renamed_crate_fixture_compiles() {
    use std::path::PathBuf;

    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let fixture = manifest_dir.join("tests/renamed");
    let target_dir = std::env::var_os("CARGO_TARGET_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| manifest_dir.join("../target"));

    let status = std::process::Command::new(env!("CARGO"))
        .args(["check", "--offline"])
        .current_dir(&fixture)
        .env("CARGO_TARGET_DIR", target_dir)
        .status()
        .expect("cargo check for renamed fixture should launch");
    assert!(status.success(), "renamed-crate fixture must compile");
}
