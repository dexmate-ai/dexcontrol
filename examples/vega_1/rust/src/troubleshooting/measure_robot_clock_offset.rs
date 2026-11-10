// Copyright (C) 2026 Dexmate Inc.
// SPDX-License-Identifier: Apache-2.0

//! Estimate robot clock offset and network round-trip time.
//!
//! Use a read-only DiagnosticClient to collect 30 time-query samples and report server-minus-client offset, round-trip time, and replies. Does not synchronize or change either clock; no simulation connection is supported.
//!
//! Pretty-printed diagnostic JSON.

use dexcontrol_examples::{Args, Result};

#[tokio::main]
async fn main() -> Result {
    let args = Args::parse(
        "Estimate robot clock offset and network round-trip time. Use a read-only DiagnosticClient to collect 30 time-query samples and report server-minus-client offset, round-trip time, and replies. Does not synchronize or change either clock; no simulation connection is supported. Pretty-printed diagnostic JSON.",
        &["samples"],
    );
    let client = args.diagnostics().await?;
    let stats = serde_json::from_str::<serde_json::Value>(
        &client.query_ntp_json(args.integer("samples", 30)?)?,
    )?;
    println!("{}", serde_json::to_string_pretty(&stats)?);
    client.close()?;
    Ok(())
}
