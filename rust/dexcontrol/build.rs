// Copyright (C) 2026 Dexmate Inc.
// SPDX-License-Identifier: Apache-2.0

use std::{env, path::PathBuf};
fn main() {
    println!("cargo:rerun-if-env-changed=DEXCONTROL_SDK_DIR");
    let target = env::var("TARGET").unwrap();
    if ![
        "x86_64-unknown-linux-gnu",
        "aarch64-unknown-linux-gnu",
        "x86_64-apple-darwin",
        "aarch64-apple-darwin",
    ]
    .contains(&target.as_str())
    {
        panic!("No DexControl native SDK is available for target {target}");
    }
    if let Some(root) = env::var_os("DEXCONTROL_SDK_DIR") {
        let lib = PathBuf::from(root).join("lib");
        let pc = lib.join("pkgconfig/dexcontrol.pc");
        println!("cargo:rerun-if-changed={}", pc.display());
        let metadata = std::fs::read_to_string(&pc)
            .expect("SDK metadata lib/pkgconfig/dexcontrol.pc is missing");
        let version = metadata
            .lines()
            .find_map(|line| line.strip_prefix("Version:"))
            .unwrap_or("")
            .trim();
        if version != env!("CARGO_PKG_VERSION") {
            panic!(
                "SDK version {version} does not match Rust bindings {}; install the matching SDK",
                env!("CARGO_PKG_VERSION")
            );
        }
        if !lib.join("libdexcontrol_capi.so").is_file()
            && !lib.join("libdexcontrol_capi.dylib").is_file()
        {
            panic!("DEXCONTROL_SDK_DIR must point to an installed SDK containing lib/libdexcontrol_capi.so or .dylib");
        }
        println!("cargo:rustc-link-search=native={}", lib.display());
        println!("cargo:rustc-link-lib=dylib=dexcontrol_capi");
    } else {
        pkg_config::Config::new()
            .exactly_version(env!("CARGO_PKG_VERSION"))
            .statik(false)
            .probe("dexcontrol")
            .expect(
                "Install the DexControl native SDK and set DEXCONTROL_SDK_DIR to its directory",
            );
    }
}
