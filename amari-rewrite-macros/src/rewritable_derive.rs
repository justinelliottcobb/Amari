// SPDX-License-Identifier: MIT OR Apache-2.0

//! `derive(Rewritable)` implementation.
//!
//! Supported containers: named/tuple/unit structs and enums whose
//! variants use named, tuple, or unit shapes. Children are exactly the
//! fields marked `#[rewritable(child)]`, in declaration order; the
//! macro never infers recursion. A child field's type must deref to
//! `Self` and rebuild from `Self` (`Box<Self>`, `Rc<Self>`, `Arc<Self>`).
//! Unmarked fields are cloned through replacement and must be `Clone`.
//!
//! Rejected with precise spans: unions, generic parameters, duplicate
//! or unknown `#[rewritable(...)]` arguments, and collection-typed
//! children (`Vec` and friends — use one child field per position).

use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use syn::spanned::Spanned;
use syn::{Data, DeriveInput, Error, Fields, Ident, Index, Member, Type};

/// Expand the derive; parse/shape errors become `compile_error!`.
pub fn expand(input: DeriveInput) -> TokenStream {
    match plan(&input).and_then(|ctors| generate(&input, &ctors)) {
        Ok(tokens) => tokens,
        Err(error) => error.into_compile_error(),
    }
}

/// One constructor's shape: the struct itself or one enum variant.
struct Constructor {
    /// Pattern/construction path: `Self` or `Self::Variant`.
    path: TokenStream,
    fields: Vec<FieldPlan>,
}

struct FieldPlan {
    member: Member,
    binding: Ident,
    is_child: bool,
}

fn plan(input: &DeriveInput) -> Result<Vec<Constructor>, Error> {
    if !input.generics.params.is_empty() {
        // Single-token span: trybuild .stderr files must render
        // identically across rustc patch versions (joined spans render
        // first-token-only on some toolchains).
        let span = input
            .generics
            .params
            .first()
            .map(Spanned::span)
            .unwrap_or_else(|| input.generics.span());
        return Err(Error::new(
            span,
            "derive(Rewritable) does not support generic parameters",
        ));
    }
    match &input.data {
        Data::Union(data) => Err(Error::new(
            data.fields.span(),
            "derive(Rewritable) does not support unions",
        )),
        Data::Struct(data) => Ok(vec![Constructor {
            path: quote!(Self),
            fields: plan_fields(&data.fields)?,
        }]),
        Data::Enum(data) => data
            .variants
            .iter()
            .map(|variant| {
                let ident = &variant.ident;
                Ok(Constructor {
                    path: quote!(Self::#ident),
                    fields: plan_fields(&variant.fields)?,
                })
            })
            .collect(),
    }
}

fn plan_fields(fields: &Fields) -> Result<Vec<FieldPlan>, Error> {
    let mut planned = Vec::new();
    for (position, field) in fields.iter().enumerate() {
        let (member, binding) = match &field.ident {
            Some(ident) => (Member::Named(ident.clone()), ident.clone()),
            None => (
                Member::Unnamed(Index::from(position)),
                format_ident!("__rewritable_field_{position}"),
            ),
        };
        let is_child = child_attribute(field)?;
        if is_child {
            reject_collection(&field.ty)?;
        }
        planned.push(FieldPlan {
            member,
            binding,
            is_child,
        });
    }
    Ok(planned)
}

/// Parse `#[rewritable(child)]`; reject duplicates and unknown keys.
fn child_attribute(field: &syn::Field) -> Result<bool, Error> {
    let mut child = false;
    for attr in &field.attrs {
        if !attr.path().is_ident("rewritable") {
            continue;
        }
        if matches!(attr.meta, syn::Meta::Path(_)) {
            return Err(Error::new(attr.span(), "expected `#[rewritable(child)]`"));
        }
        let mut seen_here = false;
        attr.parse_nested_meta(|meta| {
            if meta.path.is_ident("child") {
                if seen_here {
                    return Err(meta.error("duplicate `child` in rewritable attribute"));
                }
                seen_here = true;
                Ok(())
            } else {
                Err(meta.error("unsupported rewritable argument; expected `child`"))
            }
        })?;
        if seen_here {
            if child {
                return Err(Error::new(
                    attr.path().span(),
                    "duplicate `#[rewritable(child)]` on one field",
                ));
            }
            child = true;
        }
    }
    Ok(child)
}

/// Collection-typed children get a targeted diagnostic; the trait
/// models children as individual positions, not containers.
fn reject_collection(ty: &Type) -> Result<(), Error> {
    let Type::Path(path) = ty else {
        return Ok(());
    };
    let Some(segment) = path.path.segments.last() else {
        return Ok(());
    };
    const COLLECTIONS: &[&str] = &[
        "Vec",
        "VecDeque",
        "LinkedList",
        "HashSet",
        "BTreeSet",
        "HashMap",
        "BTreeMap",
        "BinaryHeap",
    ];
    if COLLECTIONS.iter().any(|c| segment.ident == c) {
        return Err(Error::new(
            segment.ident.span(),
            "collection-typed children are not supported by \
             derive(Rewritable); use one child field per position",
        ));
    }
    Ok(())
}

/// Resolve `amari-rewrite` for hygienic generated paths. `Itself`
/// (expansion inside amari-rewrite's own targets) still uses
/// `::amari_rewrite`, valid through `extern crate self as amari_rewrite`
/// plus the implicit extern in integration/doc targets.
fn rewrite_path() -> TokenStream {
    let found = proc_macro_crate::crate_name("amari-rewrite");
    match found {
        Ok(proc_macro_crate::FoundCrate::Name(name)) => {
            let ident = Ident::new(&name, proc_macro2::Span::call_site());
            quote!(::#ident)
        }
        _ => quote!(::amari_rewrite),
    }
}

fn generate(input: &DeriveInput, constructors: &[Constructor]) -> Result<TokenStream, Error> {
    let name = &input.ident;
    let rewrite = rewrite_path();

    if constructors.is_empty() {
        // Uninhabited enum: empty matches diverge and typecheck.
        return Ok(quote! {
            #[automatically_derived]
            impl #rewrite::Rewritable for #name {
                fn child_count(&self) -> usize {
                    match *self {}
                }
                fn child(
                    &self,
                    _index: usize,
                ) -> ::core::option::Option<&Self> {
                    match *self {}
                }
                fn replace_child(
                    &self,
                    _index: usize,
                    _replacement: Self,
                ) -> #rewrite::RewriteResult<Self> {
                    match *self {}
                }
            }
        });
    }

    let count_arms = constructors.iter().map(count_arm);
    let child_arms = constructors.iter().map(child_arm);
    let replace_arms = constructors.iter().map(|c| replace_arm(c, &rewrite));

    Ok(quote! {
        #[automatically_derived]
        impl #rewrite::Rewritable for #name {
            fn child_count(&self) -> usize {
                match self {
                    #(#count_arms)*
                }
            }
            fn child(
                &self,
                index: usize,
            ) -> ::core::option::Option<&Self> {
                match self {
                    #(#child_arms)*
                }
            }
            fn replace_child(
                &self,
                index: usize,
                replacement: Self,
            ) -> #rewrite::RewriteResult<Self> {
                match self {
                    #(#replace_arms)*
                }
            }
        }
    })
}

/// `Constructor { .. } => <child count>` (unit: `Constructor => 0`).
fn count_arm(constructor: &Constructor) -> TokenStream {
    let path = &constructor.path;
    let count = constructor.fields.iter().filter(|f| f.is_child).count();
    let pattern = ignored_pattern(constructor);
    quote!(#path #pattern => #count,)
}

fn ignored_pattern(constructor: &Constructor) -> TokenStream {
    match constructor.fields.first() {
        None => quote!(),
        Some(field) => match &field.member {
            Member::Named(_) => quote!({ .. }),
            Member::Unnamed(_) => quote!((..)),
        },
    }
}

/// Bind child fields, dispatch on `index`.
fn child_arm(constructor: &Constructor) -> TokenStream {
    let path = &constructor.path;
    let children: Vec<&FieldPlan> = constructor.fields.iter().filter(|f| f.is_child).collect();
    if children.is_empty() {
        let pattern = ignored_pattern(constructor);
        return quote!(#path #pattern => ::core::option::Option::None,);
    }
    let bindings: Vec<&Ident> = children.iter().map(|f| &f.binding).collect();
    let pattern = child_pattern(constructor, &bindings);
    let arms = children.iter().enumerate().map(|(index, field)| {
        let binding = &field.binding;
        quote! {
            #index => ::core::option::Option::Some(
                ::core::ops::Deref::deref(#binding)
            ),
        }
    });
    quote! {
        #path #pattern => match index {
            #(#arms)*
            _ => ::core::option::Option::None,
        },
    }
}

fn child_pattern(constructor: &Constructor, bindings: &[&Ident]) -> TokenStream {
    match constructor.fields.first() {
        None => quote!(),
        Some(field) => match &field.member {
            Member::Named(_) => quote!({ #(#bindings,)* .. }),
            Member::Unnamed(_) => {
                let slots = constructor.fields.iter().map(|f| {
                    if f.is_child {
                        let binding = &f.binding;
                        quote!(#binding)
                    } else {
                        quote!(_)
                    }
                });
                quote!(( #(#slots),* ))
            }
        },
    }
}

/// Bind all fields, rebuild with one child swapped via `From`.
fn replace_arm(constructor: &Constructor, rewrite: &TokenStream) -> TokenStream {
    let path = &constructor.path;
    let has_children = constructor.fields.iter().any(|f| f.is_child);
    if constructor.fields.is_empty() || !has_children {
        let pattern = ignored_pattern(constructor);
        return quote! {
            #path #pattern => ::core::result::Result::Err(
                #rewrite::RewriteError::InvalidChildIndex { index }
            ),
        };
    }
    let bindings: Vec<&Ident> = constructor.fields.iter().map(|f| &f.binding).collect();
    let pattern = full_pattern(constructor, &bindings);

    let mut child_index = 0usize;
    let mut arms = Vec::new();
    for field in &constructor.fields {
        if !field.is_child {
            continue;
        }
        let constructions = constructor.fields.iter().map(|f| {
            let member = &f.member;
            let binding = &f.binding;
            if std::ptr::eq(f, field) {
                construct_field(
                    member,
                    quote! {
                        ::core::convert::From::from(replacement)
                    },
                )
            } else {
                construct_field(
                    member,
                    quote! {
                        ::core::clone::Clone::clone(#binding)
                    },
                )
            }
        });
        let body = construct_body(constructor, constructions);
        arms.push(quote!(#child_index => ::core::result::Result::Ok(#body),));
        child_index += 1;
    }

    quote! {
        #path #pattern => match index {
            #(#arms)*
            _ => ::core::result::Result::Err(
                #rewrite::RewriteError::InvalidChildIndex { index }
            ),
        },
    }
}

fn full_pattern(constructor: &Constructor, bindings: &[&Ident]) -> TokenStream {
    match constructor.fields.first().map(|f| &f.member) {
        None => quote!(),
        Some(Member::Named(_)) => quote!({ #(#bindings),* }),
        Some(Member::Unnamed(_)) => quote!(( #(#bindings),* )),
    }
}

fn construct_field(member: &Member, value: TokenStream) -> TokenStream {
    match member {
        Member::Named(ident) => quote!(#ident: #value),
        Member::Unnamed(_) => quote!(#value),
    }
}

fn construct_body(
    constructor: &Constructor,
    fields: impl Iterator<Item = TokenStream>,
) -> TokenStream {
    let path = &constructor.path;
    let fields: Vec<TokenStream> = fields.collect();
    match constructor.fields.first().map(|f| &f.member) {
        None => quote!(#path),
        Some(Member::Named(_)) => quote!(#path { #(#fields),* }),
        Some(Member::Unnamed(_)) => quote!(#path ( #(#fields),* )),
    }
}
