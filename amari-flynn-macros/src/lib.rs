//! Procedural macros for amari-flynn probabilistic contracts
//!
//! These macros provide syntactic sugar for probabilistic contract verification,
//! integrating with the Flynn verification framework.

use proc_macro::TokenStream;
use quote::quote;
use syn::parse::{Parse, ParseStream};
use syn::{parse_macro_input, Expr, ItemFn, LitFloat, Token};

/// Parsed attribute for probabilistic requires/ensures
struct ProbabilisticAttr {
    condition: Expr,
    _comma: Token![,],
    bound: LitFloat,
}

impl Parse for ProbabilisticAttr {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        Ok(ProbabilisticAttr {
            condition: input.parse()?,
            _comma: input.parse()?,
            bound: input.parse()?,
        })
    }
}

/// Parsed attribute for expected value constraints
struct ExpectedValueAttr {
    expression: Expr,
    _comma: Token![,],
    expected: LitFloat,
    _comma2: Token![,],
    epsilon: LitFloat,
}

impl Parse for ExpectedValueAttr {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        Ok(ExpectedValueAttr {
            expression: input.parse()?,
            _comma: input.parse()?,
            expected: input.parse()?,
            _comma2: input.parse()?,
            epsilon: input.parse()?,
        })
    }
}

/// Probabilistic precondition
///
/// Generates documentation and verification infrastructure for preconditions.
///
/// # Syntax
///
/// ```ignore
/// #[prob_requires(condition, probability_bound)]
/// ```
///
/// - `condition`: Boolean expression that should hold for inputs
/// - `probability_bound`: Maximum probability the condition is violated (0.0 to 1.0)
///
/// # Example
///
/// ```ignore
/// use amari_flynn_macros::prob_requires;
///
/// #[prob_requires(x > 0.0, 0.95)]
/// fn compute(x: f64) -> f64 {
///     x.sqrt()
/// }
/// ```
///
/// This documents that the function expects x > 0.0 to hold with probability ≥ 0.95.
#[proc_macro_attribute]
pub fn prob_requires(attr: TokenStream, item: TokenStream) -> TokenStream {
    let input = parse_macro_input!(item as ItemFn);

    // Parse attribute or use default documentation
    let doc_string = if attr.is_empty() {
        "Probabilistic precondition: Contract to be verified statistically".to_string()
    } else {
        match syn::parse::<ProbabilisticAttr>(attr.clone()) {
            Ok(parsed) => {
                let condition = &parsed.condition;
                let bound = &parsed.bound;
                format!(
                    "Probabilistic precondition: Requires `{}` with P ≥ {}",
                    quote!(#condition),
                    bound.base10_parse::<f64>().unwrap_or(0.0)
                )
            }
            Err(_) => "Probabilistic precondition: Invalid contract specification".to_string(),
        }
    };

    let fn_name = &input.sig.ident;

    // Generate verification helper function name
    let verify_fn_name =
        syn::Ident::new(&format!("verify_{}_precondition", fn_name), fn_name.span());

    // Extract parameter type and name for binding in verification helper
    let (param_type, param_bindings) = if input.sig.inputs.len() == 1 {
        if let Some(syn::FnArg::Typed(pat_type)) = input.sig.inputs.first() {
            let param_type = &pat_type.ty;
            if let syn::Pat::Ident(pat_ident) = &*pat_type.pat {
                let param_name = &pat_ident.ident;
                (quote! { #param_type }, quote! { let #param_name = inputs; })
            } else {
                (quote! { _ }, quote! {})
            }
        } else {
            (quote! { _ }, quote! {})
        }
    } else {
        // For multiple parameters, we'd need tuple destructuring - skip for now
        (quote! { _ }, quote! {})
    };

    // If we successfully parsed the attribute, generate a verification helper
    let verification_helper = if !attr.is_empty() {
        match syn::parse::<ProbabilisticAttr>(attr) {
            Ok(parsed) => {
                let condition = &parsed.condition;
                let bound = &parsed.bound;

                quote! {
                    #[cfg(test)]
                    #[doc = concat!("Verification helper for ", stringify!(#fn_name), " precondition")]
                    #[allow(dead_code)]
                    fn #verify_fn_name<F>(
                        input_generator: F,
                        samples: usize,
                    ) -> amari_flynn::contracts::VerificationResult
                    where
                        F: Fn() -> #param_type,
                    {
                        let verifier = amari_flynn::backend::monte_carlo::MonteCarloVerifier::new(samples);
                        verifier.verify_probability_bound(
                            || {
                                let inputs = input_generator();
                                #param_bindings
                                #condition
                            },
                            #bound,
                        )
                    }
                }
            }
            Err(_) => quote! {},
        }
    } else {
        quote! {}
    };

    let output = quote! {
        #[doc = #doc_string]
        #input

        #verification_helper
    };

    output.into()
}

/// Probabilistic postcondition
///
/// Generates documentation and verification infrastructure for postconditions.
///
/// # Syntax
///
/// ```ignore
/// #[prob_ensures(condition, probability_bound)]
/// ```
///
/// - `condition`: Boolean expression that should hold for outputs
/// - `probability_bound`: Maximum probability the condition is violated (0.0 to 1.0)
///
/// # Example
///
/// ```ignore
/// use amari_flynn_macros::prob_ensures;
///
/// #[prob_ensures(result >= 0.0, 0.99)]
/// fn compute(x: f64) -> f64 {
///     x.abs()
/// }
/// ```
///
/// This documents that the result should be non-negative with probability ≥ 0.99.
#[proc_macro_attribute]
pub fn prob_ensures(attr: TokenStream, item: TokenStream) -> TokenStream {
    let input = parse_macro_input!(item as ItemFn);

    let doc_string = if attr.is_empty() {
        "Probabilistic postcondition: Contract to be verified statistically".to_string()
    } else {
        match syn::parse::<ProbabilisticAttr>(attr.clone()) {
            Ok(parsed) => {
                let condition = &parsed.condition;
                let bound = &parsed.bound;
                format!(
                    "Probabilistic postcondition: Ensures `{}` with P ≥ {}",
                    quote!(#condition),
                    bound.base10_parse::<f64>().unwrap_or(0.0)
                )
            }
            Err(_) => "Probabilistic postcondition: Invalid contract specification".to_string(),
        }
    };

    let fn_name = &input.sig.ident;
    let verify_fn_name =
        syn::Ident::new(&format!("verify_{}_postcondition", fn_name), fn_name.span());

    // Extract parameter names for binding
    let param_bindings = if input.sig.inputs.len() == 1 {
        if let Some(syn::FnArg::Typed(pat_type)) = input.sig.inputs.first() {
            if let syn::Pat::Ident(pat_ident) = &*pat_type.pat {
                let param_name = &pat_ident.ident;
                quote! { let #param_name = inputs; }
            } else {
                quote! {}
            }
        } else {
            quote! {}
        }
    } else {
        quote! {}
    };

    let verification_helper = if !attr.is_empty() {
        match syn::parse::<ProbabilisticAttr>(attr) {
            Ok(parsed) => {
                let condition = &parsed.condition;
                let bound = &parsed.bound;

                quote! {
                    #[cfg(test)]
                    #[doc = concat!("Verification helper for ", stringify!(#fn_name), " postcondition")]
                    #[allow(dead_code)]
                    fn #verify_fn_name<F>(
                        input_generator: F,
                        samples: usize,
                    ) -> amari_flynn::contracts::VerificationResult
                    where
                        F: Fn() -> _,
                    {
                        let verifier = amari_flynn::backend::monte_carlo::MonteCarloVerifier::new(samples);
                        verifier.verify_probability_bound(
                            || {
                                let inputs = input_generator();
                                #param_bindings
                                let result = #fn_name(inputs);
                                #condition
                            },
                            #bound,
                        )
                    }
                }
            }
            Err(_) => quote! {},
        }
    } else {
        quote! {}
    };

    let output = quote! {
        #[doc = #doc_string]
        #input

        #verification_helper
    };

    output.into()
}

/// Expected value constraint
///
/// Generates documentation and verification for expected value bounds.
///
/// # Syntax
///
/// ```ignore
/// #[ensures_expected(expression, expected_value, epsilon)]
/// ```
///
/// - `expression`: Expression to evaluate
/// - `expected_value`: Expected mean value
/// - `epsilon`: Maximum deviation from expected value
///
/// # Example
///
/// ```ignore
/// use amari_flynn_macros::ensures_expected;
///
/// #[ensures_expected(result, 5.0, 0.1)]
/// fn random_around_five() -> f64 {
///     5.0 + (rand::random::<f64>() - 0.5) * 0.2
/// }
/// ```
///
/// This documents that the result should have expected value 5.0 ± 0.1.
#[proc_macro_attribute]
pub fn ensures_expected(attr: TokenStream, item: TokenStream) -> TokenStream {
    let input = parse_macro_input!(item as ItemFn);

    let doc_string = if attr.is_empty() {
        "Expected value constraint: Statistical verification of mean".to_string()
    } else {
        match syn::parse::<ExpectedValueAttr>(attr.clone()) {
            Ok(parsed) => {
                let expression = &parsed.expression;
                let expected = &parsed.expected;
                let epsilon = &parsed.epsilon;
                format!(
                    "Expected value constraint: E[{}] = {} ± {}",
                    quote!(#expression),
                    expected.base10_parse::<f64>().unwrap_or(0.0),
                    epsilon.base10_parse::<f64>().unwrap_or(0.0)
                )
            }
            Err(_) => "Expected value constraint: Invalid specification".to_string(),
        }
    };

    let fn_name = &input.sig.ident;
    let verify_fn_name = syn::Ident::new(
        &format!("verify_{}_expected_value", fn_name),
        fn_name.span(),
    );

    // Extract parameter names for binding
    let param_bindings = if input.sig.inputs.len() == 1 {
        if let Some(syn::FnArg::Typed(pat_type)) = input.sig.inputs.first() {
            if let syn::Pat::Ident(pat_ident) = &*pat_type.pat {
                let param_name = &pat_ident.ident;
                quote! { let #param_name = inputs; }
            } else {
                quote! {}
            }
        } else {
            quote! {}
        }
    } else {
        quote! {}
    };

    let verification_helper = if !attr.is_empty() {
        match syn::parse::<ExpectedValueAttr>(attr) {
            Ok(parsed) => {
                let expression = &parsed.expression;
                let expected = &parsed.expected;
                let epsilon = &parsed.epsilon;

                quote! {
                    #[cfg(test)]
                    #[doc = concat!("Verification helper for ", stringify!(#fn_name), " expected value")]
                    #[allow(dead_code)]
                    fn #verify_fn_name<F>(
                        input_generator: F,
                        samples: usize,
                    ) -> bool
                    where
                        F: Fn() -> _,
                    {
                        let mut sum = 0.0;
                        for _ in 0..samples {
                            let inputs = input_generator();
                            #param_bindings
                            let value = #fn_name(inputs);
                            let #expression = value;
                            sum += #expression;
                        }
                        let mean = sum / samples as f64;
                        let expected = #expected;
                        let epsilon = #epsilon;
                        (mean - expected).abs() <= epsilon
                    }
                }
            }
            Err(_) => quote! {},
        }
    } else {
        quote! {}
    };

    let output = quote! {
        #[doc = #doc_string]
        #input

        #verification_helper
    };

    output.into()
}
