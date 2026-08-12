//! derive(Rewritable) runtime behavior tests (0.25 Cohort 1, Task 4).
//!
//! Exercises the real macro expansion through the public trait surface:
//! preorder positions, child access, replacement, and checked
//! invalid-index errors. Compile-time rejection contracts (unions,
//! generics, duplicate attributes, collections) live in the trybuild
//! fixtures under `amari-rewrite-macros/tests/ui/`.

#![cfg(feature = "macros")]

use amari_rewrite::{Path, Rewritable, RewriteError};

/// Expression AST mixing a unit leaf, a tuple variant child, and a
/// struct variant with named children plus an unmarked payload field.
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

fn num(n: i64) -> Expr {
    Expr::Num(n)
}

fn add(lhs: Expr, rhs: Expr) -> Expr {
    Expr::Add {
        lhs: Box::new(lhs),
        rhs: Box::new(rhs),
        note: String::new(),
    }
}

#[test]
fn replace_child_preserves_unmarked_payloads() {
    let tree = Expr::Add {
        lhs: Box::new(num(1)),
        rhs: Box::new(num(2)),
        note: "kept".to_string(),
    };
    let replaced = tree.replace_child(1, num(3)).unwrap();
    match replaced {
        Expr::Add { lhs, rhs, note } => {
            assert_eq!(*lhs, num(1));
            assert_eq!(*rhs, num(3));
            assert_eq!(note, "kept");
        }
        other => panic!("expected Add, got {other:?}"),
    }
    let err = tree.replace_child(2, num(3)).unwrap_err();
    assert_eq!(err, RewriteError::InvalidChildIndex { index: 2 });
}

/// Leaf struct: no marked children, unmarked payloads only.
/// (Struct variants carrying children are exercised through `Expr::Add`;
/// a pure struct with `Box<Self>` children cannot terminate.)
#[derive(Clone, Debug, PartialEq, Rewritable)]
struct Meta {
    name: String,
    arity: usize,
}

#[test]
fn struct_without_children_is_a_leaf() {
    let meta = Meta {
        name: "const".to_string(),
        arity: 0,
    };
    assert_eq!(meta.child_count(), 0);
    assert_eq!(meta.child(0), None);
    let err = meta
        .replace_child(
            0,
            Meta {
                name: "other".to_string(),
                arity: 1,
            },
        )
        .unwrap_err();
    assert_eq!(err, RewriteError::InvalidChildIndex { index: 0 });
}

#[test]
fn leaf_variants_have_no_children() {
    let leaf = num(7);
    assert_eq!(leaf.child_count(), 0);
    assert_eq!(leaf.child(0), None);
    let err = leaf.replace_child(0, num(8)).unwrap_err();
    assert_eq!(err, RewriteError::InvalidChildIndex { index: 0 });
}

#[test]
fn tuple_variant_children_are_positional() {
    let neg = Expr::Neg(Box::new(num(1)));
    assert_eq!(neg.child_count(), 1);
    assert_eq!(neg.child(0), Some(&num(1)));
    assert_eq!(neg.child(1), None);
    let replaced = neg.replace_child(0, num(2)).unwrap();
    assert_eq!(replaced, Expr::Neg(Box::new(num(2))));
}

#[test]
fn struct_variant_children_follow_declaration_order() {
    let tree = add(num(1), num(2));
    assert_eq!(tree.child_count(), 2);
    assert_eq!(tree.child(0), Some(&num(1)));
    assert_eq!(tree.child(1), Some(&num(2)));
    assert_eq!(tree.child(2), None);
}

#[test]
fn preorder_positions_cover_the_whole_tree() {
    let tree = add(add(num(1), num(2)), num(3));
    let expected: Vec<Path> = vec![
        Path::root(),
        Path::from([0]),
        Path::from([0, 0]),
        Path::from([0, 1]),
        Path::from([1]),
    ];
    assert_eq!(tree.positions(), expected);
}

#[test]
fn replace_at_rebuilds_only_the_target_path() {
    let tree = add(add(num(1), num(2)), num(3));
    let replaced = tree.replace_at(&Path::from([0, 1]), num(9)).unwrap();
    assert_eq!(replaced, add(add(num(1), num(9)), num(3)));
    let err = tree.replace_at(&Path::from([1, 0]), num(9)).unwrap_err();
    assert_eq!(err, RewriteError::InvalidChildIndex { index: 0 });
}
