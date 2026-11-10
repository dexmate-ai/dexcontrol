// Copyright (C) 2026 Dexmate Inc.
// SPDX-License-Identifier: Apache-2.0

//! Send raw bytes to end effectors and poll for replies.
//!
//! By default, send hex 09 10 03 E8 00 03 06 01 00 00 00 00 00 72 E1 to both configured arm pass-through interfaces and poll for up to 1 s each. The attached device determines what these bytes do; this is not a generic motion command.
//!
//! Uses a Robot control connection: compatible arm modes may be enabled during startup, and shutdown requests a stop. Real hardware is selected unless --simulated is supplied.

use dexcontrol_examples::{component_name, invalid, pause, Args, Result};
use std::time::{Duration, Instant};

fn parse_hex(value: &str) -> Result<Vec<u8>> {
    let compact = value.split_whitespace().collect::<String>();
    if !compact.is_ascii() || compact.len() % 2 != 0 {
        return Err(invalid("--message-hex needs an even number of digits"));
    }
    (0..compact.len())
        .step_by(2)
        .map(|index| u8::from_str_radix(&compact[index..index + 2], 16).map_err(Into::into))
        .collect()
}

#[tokio::main]
async fn main() -> Result {
    let args = Args::parse(
        "Send raw bytes to end effectors and poll for replies. By default, send hex 09 10 03 E8 00 03 06 01 00 00 00 00 00 72 E1 to both configured arm pass-through interfaces and poll for up to 1 s each. The attached device determines what these bytes do; this is not a generic motion command. Uses a Robot control connection: compatible arm modes may be enabled during startup, and shutdown requests a stop. Real hardware is selected unless --simulated is supplied.",
        &["message-hex", "side", "timeout"],
    );
    let payload = parse_hex(&args.text(
        "message-hex",
        "09 10 03 E8 00 03 06 01 00 00 00 00 00 72 E1",
    ))?;
    let timeout = args.duration("timeout", 1.0)?;
    let robot = args.connect(&[]).await?;
    dexcontrol_examples::run_robot(&robot, async {
        for side in args.sides()? {
            let arm = robot.joints(&component_name(side, "arm"))?;
            if !arm.ee_pass_through_enabled()? {
                println!("{side}_arm: pass-through endpoints are not configured");
                continue;
            }
            arm.ee_pass_through_send(&payload)?;
            let started = Instant::now();
            loop {
                match arm.ee_pass_through_latest() {
                    Ok(Some(response)) => {
                        println!(
                            "{side}_arm: {}",
                            response
                                .iter()
                                .map(|byte| format!("{byte:02x}"))
                                .collect::<Vec<_>>()
                                .join(" ")
                        );
                        break;
                    }
                    Ok(None) if started.elapsed() < timeout => {
                        pause(Duration::from_millis(10)).await
                    }
                    Ok(None) => {
                        println!(
                            "{side}_arm: no response within {:.2}s",
                            timeout.as_secs_f64()
                        );
                        break;
                    }
                    Err(error) => return Err(error.into()),
                }
            }
        }
        Ok(())
    })
    .await
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn hexadecimal_input_rejects_unicode_without_panicking() {
        assert_eq!(parse_hex("09 10 ff").unwrap(), vec![9, 16, 255]);
        for text in ["e", "zz", "éé", "💥"] {
            assert!(parse_hex(text).is_err());
        }
    }
}
