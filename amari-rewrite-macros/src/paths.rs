// SPDX-License-Identifier: MIT OR Apache-2.0

//! Shared resolution of the `amari-rewrite` crate path for generated
//! code. `Itself` (expansion inside amari-rewrite's own targets) still
//! uses `::amari_rewrite`, valid through
//! `extern crate self as amari_rewrite` plus the implicit extern in
//! integration/doc targets.

use proc_macro2::{Ident, Span, TokenStream};
use quote::quote;

pub fn rewrite_path() -> TokenStream {
    let found = proc_macro_crate::crate_name("amari-rewrite");
    match found {
        Ok(proc_macro_crate::FoundCrate::Name(name)) => {
            let ident = Ident::new(&name, Span::call_site());
            quote!(::#ident)
        }
        _ => quote!(::amari_rewrite),
    }
}
