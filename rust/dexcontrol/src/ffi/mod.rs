// Copyright (C) 2026 Dexmate Inc.
// SPDX-License-Identifier: Apache-2.0

//! Private C ABI declarations for the safe public API.
#![allow(dead_code)]
#![allow(non_camel_case_types, non_snake_case, non_upper_case_globals)]
#![allow(rustdoc::broken_intra_doc_links, rustdoc::bare_urls)]
include!("bindings.rs");
include!("constants.rs");
pub const EXPECTED_ABI_VERSION: u32 = DEX_ABI_VERSION;
