// Copyright (C) 2026 Dexmate Inc.
//
// This software is dual-licensed:
//
// 1. GNU Affero General Public License v3.0 (AGPL-3.0)
//    See LICENSE-AGPL for details
//
// 2. Commercial License
//    For commercial licensing terms, contact: contact@dexmate.ai

// Shared scaffolding for the Rust examples.

use dexcontrol::{ConnectOptions, DiagnosticClient, Robot};
use std::collections::BTreeMap;
use std::error::Error;
use std::io;
use std::time::Duration;

pub type Result<T = ()> = std::result::Result<T, Box<dyn Error + Send + Sync>>;

/// Options that take no value (`--flag`, or `--flag=true|false|1|0`).
const FLAGS: &[&str] = &[
    "help",
    "simulated",
    "unfold",
    "visualize",
    "no-confirm",
    "no-display",
    "no-zero-force",
    "passthrough",
    "relative",
    "velocity-compensation",
    "all",
    "yes",
];

#[derive(Debug, Default)]
pub struct Args {
    description: &'static str,
    options: BTreeMap<String, Vec<String>>,
    positionals: Vec<String>,
}

pub struct ImageViewer {
    #[cfg(feature = "native-viewer")]
    inner: dexcontrol_viewer::Viewer,
}

impl ImageViewer {
    fn is_open(&self) -> Result<bool> {
        #[cfg(feature = "native-viewer")]
        {
            Ok(self.inner.is_open()?)
        }
        #[cfg(not(feature = "native-viewer"))]
        {
            Ok(false)
        }
    }

    fn log_frame(&self, path: &str, frame: &dexcontrol::Frame) -> Result {
        #[cfg(feature = "native-viewer")]
        {
            let info = frame.info()?;
            let format = match frame.encoding()? {
                dexcontrol::FrameEncoding::Rgb8 => dexcontrol_viewer::PixelFormat::Rgb8,
                dexcontrol::FrameEncoding::Bgr8 => dexcontrol_viewer::PixelFormat::Bgr8,
                dexcontrol::FrameEncoding::Gray8 => dexcontrol_viewer::PixelFormat::Gray8,
                dexcontrol::FrameEncoding::Depth32F => dexcontrol_viewer::PixelFormat::Depth32F,
                dexcontrol::FrameEncoding::Unknown(_) => {
                    return Err(invalid("unsupported image encoding"))
                }
            };
            let mut image =
                dexcontrol_viewer::Image::new(info.width, info.height, format, frame.data()?)?;
            image.timestamp_ns = (info.timestamp_ns != 0)
                .then(|| i64::try_from(info.timestamp_ns).ok())
                .flatten();
            self.inner.log_image(path, image)?;
        }
        #[cfg(not(feature = "native-viewer"))]
        let _ = (path, frame);
        Ok(())
    }
}

impl Args {
    pub fn image_viewer(&self) -> Result<Option<ImageViewer>> {
        if self.flag("no-display") {
            return Ok(None);
        }
        #[cfg(feature = "native-viewer")]
        {
            let mut options = dexcontrol_viewer::Options::new("dexcontrol-cameras");
            let url = self.text("viewer-url", "");
            if !url.is_empty() {
                options.viewer_url = Some(url);
            }
            Ok(Some(ImageViewer {
                inner: dexcontrol_viewer::Viewer::new(options)?,
            }))
        }
        #[cfg(not(feature = "native-viewer"))]
        {
            if self.has("viewer-url") {
                return Err(invalid("this build does not include image visualization"));
            }
            eprintln!(
                "Image visualization is not included in this build; printing frame metadata."
            );
            Ok(None)
        }
    }

    pub fn parse(description: &'static str, task_options: &[&str]) -> Self {
        let mut parsed = Self {
            description,
            ..Self::default()
        };
        let mut tokens = std::env::args().skip(1).peekable();
        while let Some(token) = tokens.next() {
            let Some(option) = token.strip_prefix("--") else {
                parsed.positionals.push(token);
                continue;
            };
            if let Some((key, value)) = option.split_once('=') {
                parsed
                    .options
                    .entry(key.to_owned())
                    .or_default()
                    .push(value.to_owned());
                continue;
            }
            let is_flag = FLAGS.contains(&option);
            let value = if is_flag {
                String::new()
            } else {
                tokens
                    .next_if(|next| !next.starts_with("--"))
                    .unwrap_or_else(|| {
                        eprintln!("--{option} requires a value");
                        std::process::exit(2);
                    })
            };
            parsed
                .options
                .entry(option.to_owned())
                .or_default()
                .push(value);
        }
        const COMMON: &[&str] = &["help", "profile", "config", "enable-sensor", "simulated"];
        for key in parsed.options.keys() {
            if !COMMON.contains(&key.as_str()) && !task_options.contains(&key.as_str()) {
                eprintln!("unknown option --{key}");
                std::process::exit(2);
            }
        }
        for name in FLAGS.iter().copied() {
            if let Some(values) = parsed.options.get(name) {
                if values.len() != 1
                    || values
                        .iter()
                        .any(|v| !["", "true", "false", "1", "0"].contains(&v.as_str()))
                {
                    eprintln!("--{name} expects true, false, 1, or 0");
                    std::process::exit(2);
                }
            }
        }
        if parsed.flag("help") {
            println!("{}", parsed.description);
            println!(
                "Task options: {}",
                task_options
                    .iter()
                    .map(|name| format!("--{name}"))
                    .collect::<Vec<_>>()
                    .join(", ")
            );
            println!(
                "\nCommon options: --profile NAME, --config FILE, \
                 --enable-sensor NAME (repeatable), --simulated"
            );
            std::process::exit(0);
        }
        parsed
    }

    pub fn flag(&self, name: &str) -> bool {
        self.options.get(name).is_some_and(|values| {
            values
                .first()
                .is_none_or(|value| value.is_empty() || value == "true" || value == "1")
        })
    }

    pub fn text(&self, name: &str, fallback: &str) -> String {
        self.options
            .get(name)
            .and_then(|values| values.first())
            .filter(|value| !value.is_empty())
            .cloned()
            .unwrap_or_else(|| fallback.to_owned())
    }

    pub fn list(&self, name: &str) -> Vec<String> {
        self.options
            .get(name)
            .into_iter()
            .flatten()
            .filter(|value| !value.is_empty())
            .cloned()
            .collect()
    }

    pub fn numbers(&self, name: &str, fallback: Vec<f64>) -> Result<Vec<f64>> {
        let value = self.text(name, "");
        if value.is_empty() {
            return Ok(fallback);
        }
        value
            .split(',')
            .map(|part| {
                finite_number(part.trim()).map_err(|_| {
                    invalid(format!(
                        "--{name} expects comma-separated numbers, got {value:?}"
                    ))
                })
            })
            .collect()
    }

    /// True when the option was given on the command line at all.
    pub fn has(&self, name: &str) -> bool {
        self.options.contains_key(name)
    }

    /// `--side`, for operations too hazardous to default: the caller must
    /// name left, right, or both.
    pub fn explicit_sides(&self) -> Result<Vec<&'static str>> {
        if !self.has("side") {
            return Err(invalid(
                "--side {left,right,both} is required for this action; there is no default",
            ));
        }
        self.sides()
    }

    pub fn sides(&self) -> Result<Vec<&'static str>> {
        match self.text("side", "both").as_str() {
            "left" => Ok(vec!["left"]),
            "right" => Ok(vec!["right"]),
            "both" => Ok(vec!["left", "right"]),
            value => Err(invalid(format!(
                "--side must be left, right, or both, got {value:?}"
            ))),
        }
    }

    pub fn number(&self, name: &str, fallback: f64) -> Result<f64> {
        let value = self.text(name, "");
        if value.is_empty() {
            return Ok(fallback);
        }
        finite_number(&value)
            .map_err(|_| invalid(format!("--{name} expects a finite number, got {value:?}")))
    }

    pub fn integer(&self, name: &str, fallback: usize) -> Result<usize> {
        let value = self.text(name, "");
        if value.is_empty() {
            return Ok(fallback);
        }
        value.parse().map_err(|_| {
            invalid(format!(
                "--{name} expects a non-negative integer, got {value:?}"
            ))
        })
    }

    pub fn positional(&self, index: usize, fallback: &str) -> String {
        self.positionals
            .get(index)
            .cloned()
            .unwrap_or_else(|| fallback.to_owned())
    }

    pub fn side(&self) -> Result<&'static str> {
        match self.text("side", "right").as_str() {
            "left" => Ok("left"),
            "right" => Ok("right"),
            value => Err(invalid(format!(
                "--side must be left or right, got {value:?}"
            ))),
        }
    }

    pub fn duration(&self, name: &str, fallback: f64) -> Result<Duration> {
        let seconds = self.number(name, fallback)?;
        if !seconds.is_finite() || seconds < 0.0 {
            return Err(invalid(format!("--{name} must be finite and non-negative")));
        }
        Duration::try_from_secs_f64(seconds)
            .map_err(|_| invalid(format!("--{name} exceeds the duration range")))
    }

    fn options<'a>(&'a self, sensors: &'a [String]) -> Result<ConnectOptions<'a>> {
        let profile = self
            .options
            .get("profile")
            .and_then(|v| v.last())
            .map(String::as_str);
        let config = self
            .options
            .get("config")
            .and_then(|v| v.last())
            .map(String::as_str);
        if profile.is_some() && config.is_some() {
            return Err(invalid("--profile and --config are mutually exclusive"));
        }
        let mut enable_sensors = self
            .options
            .get("enable-sensor")
            .into_iter()
            .flatten()
            .map(String::as_str)
            .collect::<Vec<_>>();
        enable_sensors.extend(sensors.iter().map(String::as_str));
        Ok(ConnectOptions {
            profile,
            config_file: config,
            enable_sensors,
            simulated: self.flag("simulated"),
            ..Default::default()
        })
    }
    pub fn resolved_config(&self) -> Result<serde_json::Value> {
        Ok(serde_json::from_str(
            &self.options(&[])?.resolved_config_json()?,
        )?)
    }
    pub async fn connect(&self, sensors: &[String]) -> Result<Robot> {
        Ok(Robot::connect(self.options(sensors)?)?)
    }

    /// Ask before multi-joint benchmarks/replay, before opening a connection.
    pub fn confirm_motion(&self) -> Result {
        if self.flag("no-confirm") || self.flag("simulated") {
            return Ok(());
        }
        use std::io::Write;
        print!("This example moves robot joints. Continue? [y/N] ");
        io::stdout().flush()?;
        let mut reply = String::new();
        io::stdin().read_line(&mut reply)?;
        if !matches!(reply.trim().to_ascii_lowercase().as_str(), "y" | "yes") {
            return Err(invalid("motion declined"));
        }
        Ok(())
    }

    /// Interactive confirmation for an action that can let hardware move or
    /// drop without being commanded to (brake release, board reboot). Prints
    /// `warning`, then requires the operator to type `yes`. Skipped with
    /// `--yes` and for `--simulated` runs; anything else, end of input
    /// included, declines. Call it before connecting.
    pub fn confirm_hazard(&self, warning: &str) -> Result {
        if self.flag("yes") || self.flag("simulated") {
            return Ok(());
        }
        use std::io::Write;
        println!("WARNING: {warning}");
        print!("Type 'yes' to continue: ");
        io::stdout().flush()?;
        let mut reply = String::new();
        io::stdin().read_line(&mut reply)?;
        if reply.trim() != "yes" {
            return Err(invalid("not confirmed; nothing was sent"));
        }
        Ok(())
    }

    pub async fn diagnostics(&self) -> Result<DiagnosticClient> {
        if self.flag("simulated") {
            return Err(invalid("DiagnosticClient does not support simulation"));
        }
        Ok(DiagnosticClient::connect(self.options(&[])?)?)
    }
}

pub fn invalid(message: impl Into<String>) -> Box<dyn Error + Send + Sync> {
    Box::new(io::Error::new(io::ErrorKind::InvalidInput, message.into()))
}

pub async fn pause(duration: Duration) {
    if !duration.is_zero() {
        tokio::time::sleep(duration).await;
    }
}

/// Wait for the typed IMU sample, independently of a camera's image activity.
pub async fn wait_for_imu(robot: &Robot, sensor: &str, timeout: Duration) -> Result {
    let deadline = tokio::time::Instant::now() + timeout;
    loop {
        match robot.imu(sensor, None) {
            Ok(_) => return Ok(()),
            // DEX_STATE_UNAVAILABLE from the public C ABI: retry only missing data.
            Err(error) if error.code == 6 => {}
            Err(error) => return Err(error.into()),
        }
        if tokio::time::Instant::now() >= deadline {
            return Err(invalid(format!("{sensor} IMU did not become active")));
        }
        pause(Duration::from_millis(20)).await;
    }
}

pub fn component_name(side: &str, component: &str) -> String {
    format!("{side}_{component}")
}

pub fn format(values: &[f64]) -> String {
    let values = values
        .iter()
        .map(|value| format!("{value:.3}"))
        .collect::<Vec<_>>()
        .join(", ");
    format!("[{values}]")
}

pub async fn print_camera_frames(
    robot: &Robot,
    sensors: &[String],
    requested_streams: &[String],
    samples: Option<usize>,
    period: Duration,
    viewer: Option<&ImageViewer>,
) -> Result {
    let mut subscriptions = Vec::new();
    // Keep subscription setup in the guarded scope so partial setup is cleaned up too.
    let outcome: Result = async {
        for sensor in sensors {
            let camera = robot.camera(sensor)?;
            let streams = if requested_streams.is_empty() {
                camera.streams()?
            } else {
                requested_streams.to_vec()
            };
            for stream in streams {
                camera.subscribe(&stream, 0)?;
                subscriptions.push((sensor.clone(), stream));
            }
        }
        let mut index = 0;
        let mut previous = BTreeMap::new();
        while samples.is_none_or(|limit| index < limit) {
            if let Some(viewer) = viewer {
                if !viewer.is_open()? {
                    break;
                }
            }
            for (sensor, stream) in &subscriptions {
                if let Some(frame) = robot.camera(sensor)?.latest_frame(stream)? {
                    let info = frame.info()?;
                    if let Some(viewer) = viewer {
                        let path = format!("{sensor}/{stream}");
                        if info.timestamp_ns != 0 && previous.get(&path) == Some(&info.timestamp_ns)
                        {
                            continue;
                        }
                        viewer.log_frame(&path, &frame)?;
                        previous.insert(path, info.timestamp_ns);
                        continue;
                    }
                    println!(
                        "{sensor}/{stream}: shape=({}, {}) {:?} {} bytes",
                        info.height,
                        info.width,
                        frame.encoding()?,
                        frame.data()?.len()
                    );
                }
            }
            index += 1;
            if samples.is_none_or(|limit| index < limit) {
                pause(period).await;
            }
        }
        Ok(())
    }
    .await;
    let mut cleanup_error = None;
    for (sensor, stream) in subscriptions {
        if let Err(e) = robot.camera(&sensor).and_then(|c| c.unsubscribe(&stream)) {
            cleanup_error.get_or_insert(e);
        }
    }
    outcome?;
    cleanup_error.map_or(Ok(()), |e| Err(e.into()))
}

/// A separate task closes the connection on Ctrl-C even while a synchronous
/// native call blocks the example's task. Cleanup also runs if this future is
/// dropped or its body panics. Use a multi-thread Tokio runtime.
pub async fn run_robot(robot: &Robot, body: impl std::future::Future<Output = Result>) -> Result {
    run_robot_until(robot, body, tokio::signal::ctrl_c()).await
}

async fn run_robot_until(
    robot: &Robot,
    body: impl std::future::Future<Output = Result>,
    cancelled: impl std::future::Future<Output = io::Result<()>> + Send + 'static,
) -> Result {
    let interrupt_robot = robot.try_clone()?;
    let (ready, registered) = tokio::sync::oneshot::channel();
    let monitor = tokio::spawn(async move {
        let mut cancelled = Box::pin(cancelled);
        let mut ready = Some(ready);
        // Poll once to install the signal handler before the body can block.
        std::future::poll_fn(|cx| {
            let result = cancelled.as_mut().poll(cx);
            if let Some(ready) = ready.take() {
                let _ = ready.send(());
            }
            result
        })
        .await?;
        tokio::task::spawn_blocking(move || interrupt_robot.close()).await??;
        Err::<(), Box<dyn Error + Send + Sync>>(invalid("interrupted"))
    });
    let mut session = ExampleSession {
        robot,
        monitor,
        closed: false,
    };
    registered.await?;
    let outcome = tokio::select! {
        biased;
        result = &mut session.monitor => result.unwrap_or_else(|e| Err(e.into())),
        result = body => result,
    };
    let cleanup = session.close();
    match (outcome, cleanup) {
        (Err(e), cleanup) => {
            if let Err(c) = cleanup {
                eprintln!("shutdown also failed: {c}");
            }
            Err(e)
        }
        (Ok(()), cleanup) => Ok(cleanup?),
    }
}

// A JoinHandle detaches on drop. Keep it with the connection so abandoning the
// example cannot leave a signal waiter retaining a live robot indefinitely.
struct ExampleSession<'a> {
    robot: &'a Robot,
    monitor: tokio::task::JoinHandle<Result>,
    closed: bool,
}
impl ExampleSession<'_> {
    fn close(&mut self) -> dexcontrol::Result<()> {
        self.monitor.abort();
        self.closed = true;
        self.robot.close()
    }
}
impl Drop for ExampleSession<'_> {
    fn drop(&mut self) {
        if !self.closed {
            if let Err(error) = self.close() {
                eprintln!("shutdown failed: {error}");
            }
        }
    }
}

#[cfg(test)]
mod cleanup_tests {
    use super::*;
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn errors_close_robot_and_preserve_original_failure() {
        let robot = Robot::simulated("vega_1").unwrap();
        let head = robot.joints("head").unwrap();
        let error = run_robot_until(
            &robot,
            async { Err(invalid("original failure")) },
            std::future::pending(),
        )
        .await
        .unwrap_err();
        assert!(error.to_string().contains("original failure"));
        assert!(head.get_joint_pos().is_err());
    }
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn cancellation_closes_connection_even_while_body_blocks() {
        let robot = Robot::simulated("vega_1").unwrap();
        let head = robot.joints("head").unwrap();
        let (tx, rx) = tokio::sync::oneshot::channel();
        let outcome = run_robot_until(
            &robot,
            async {
                tx.send(()).unwrap();
                let deadline = std::time::Instant::now() + Duration::from_secs(2);
                while head.get_joint_pos().is_ok() && std::time::Instant::now() < deadline {
                    std::thread::sleep(Duration::from_millis(5));
                }
                assert!(
                    head.get_joint_pos().is_err(),
                    "blocked body prevented shutdown"
                );
                Err(invalid("body observed shutdown"))
            },
            async {
                rx.await.unwrap();
                Ok(())
            },
        )
        .await;
        assert!(outcome.is_err());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn aborting_example_closes_connection() {
        let robot = Robot::simulated("vega_1").unwrap();
        let head = robot.joints("head").unwrap();
        let (ready, started) = tokio::sync::oneshot::channel();
        let task = tokio::spawn(async move {
            run_robot_until(
                &robot,
                async {
                    ready.send(()).unwrap();
                    std::future::pending().await
                },
                std::future::pending(),
            )
            .await
        });
        started.await.unwrap();
        task.abort();
        assert!(task.await.unwrap_err().is_cancelled());
        assert!(head.get_joint_pos().is_err());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn panicking_example_closes_connection() {
        let robot = Robot::simulated("vega_1").unwrap();
        let head = robot.joints("head").unwrap();
        let task = tokio::spawn(async move {
            run_robot_until(
                &robot,
                async {
                    panic!("example failed");
                },
                std::future::pending(),
            )
            .await
        });
        assert!(task.await.unwrap_err().is_panic());
        assert!(head.get_joint_pos().is_err());
    }

    #[tokio::test]
    async fn camera_imu_readiness_does_not_require_image_frames() {
        let robot = Robot::connect(ConnectOptions {
            profile: Some("vega_1"),
            simulated: true,
            enable_sensors: vec!["head_camera"],
            ..Default::default()
        })
        .unwrap();
        wait_for_imu(&robot, "head_camera", Duration::ZERO)
            .await
            .unwrap();
        assert!(!robot
            .camera("head_camera")
            .unwrap()
            .is_active(None, Duration::from_secs(1))
            .unwrap());
        assert!(wait_for_imu(&robot, "missing", Duration::ZERO)
            .await
            .is_err());
        robot.close().unwrap();
    }

    #[test]
    fn simulated_motion_needs_no_interactive_input() {
        let args = Args {
            options: BTreeMap::from([("simulated".into(), vec!["true".into()])]),
            ..Default::default()
        };
        args.confirm_motion().unwrap();
    }
    #[test]
    fn numeric_options_reject_nonfinite_values() {
        for v in ["NaN", "inf", "-inf"] {
            assert!(finite_number(v).is_err());
        }
    }
}

fn finite_number(value: &str) -> Result<f64> {
    let number: f64 = value.parse()?;
    if !number.is_finite() {
        return Err(invalid("number must be finite"));
    }
    Ok(number)
}

/// Radians kept clear of a joint limit by benchmark excursions.
pub const LIMIT_MARGIN: f64 = 0.02;
/// Fraction of the velocity limit kept clear by benchmark feed-forward.
pub const VELOCITY_MARGIN: f64 = 0.02;

/// Amplitude of a benchmark excursion from `zero` that stays within the
/// joint's limits.
///
/// The client refuses a target outside the model's joint limits, and the
/// arm-tracking reference pose sits close to a limit on some joints (R_arm_j2
/// has 0.45 rad of room upward for a pi/6 step). For a step (`symmetric =
/// false`) the signed step is returned: the requested direction if it fits,
/// the other direction if only that fits, otherwise the largest step that
/// fits. For a sine (`symmetric = true`) the magnitude is returned, limited
/// by the smaller of the two sides and, given the angular frequency `omega`
/// (rad/s), by the joint's velocity limit: the feed-forward peaks at
/// `amplitude * omega` and the client refuses one above the limit (0.4 rad at
/// 1 Hz is 2.51 rad/s against R_arm_j1's 2.4 rad/s). Every change is printed.
pub fn fit_amplitude(
    arm: &dexcontrol::JointComponent,
    joint: usize,
    zero: f64,
    amplitude: f64,
    symmetric: bool,
    omega: f64,
) -> Result<f64> {
    let Some(limits) = arm.joint_limits()? else {
        return Ok(amplitude);
    };
    let (Some(&lower), Some(&upper), Some(&velocity)) = (
        limits.lower.get(joint),
        limits.upper.get(joint),
        limits.velocity.get(joint),
    ) else {
        return Ok(amplitude);
    };
    let name = format!("joint_{joint}");
    let mut size = amplitude.abs();
    if velocity.is_finite() && symmetric && omega > 0.0 {
        let room = velocity * (1.0 - VELOCITY_MARGIN) / omega;
        if size > room {
            println!("WARNING: {name}: sine amplitude reduced from {size:.3} to {room:.3} rad so the velocity feed-forward stays within {velocity:.2} rad/s");
            size = room;
        }
    }

    let room_up = (upper - zero - LIMIT_MARGIN).max(0.0);
    let room_down = (zero - lower - LIMIT_MARGIN).max(0.0);
    if symmetric {
        let room = room_up.min(room_down);
        if size <= room {
            return Ok(size);
        }
        println!("WARNING: {name}: sine amplitude reduced from {size:.3} to {room:.3} rad to stay within [{lower:.3}, {upper:.3}]");
        return Ok(room);
    }
    let wanted_up = amplitude >= 0.0;
    if size <= if wanted_up { room_up } else { room_down } {
        return Ok(amplitude);
    }
    if size <= if wanted_up { room_down } else { room_up } {
        let direction = if wanted_up { "negative" } else { "positive" };
        println!("WARNING: {name}: stepping {direction} instead; {amplitude:+.3} rad from the reference would leave [{lower:.3}, {upper:.3}]");
        return Ok(if wanted_up { -size } else { size });
    }
    let best = if room_up >= room_down {
        room_up
    } else {
        -room_down
    };
    println!(
        "WARNING: {name}: step reduced to {best:+.3} rad to stay within [{lower:.3}, {upper:.3}]"
    );
    Ok(best)
}

/// Prevent float-to-integer saturation and unbounded benchmark allocation.
pub fn sample_count(seconds: f64, hz: f64) -> Result<usize> {
    let count = seconds * hz;
    if !seconds.is_finite()
        || seconds < 0.0
        || !hz.is_finite()
        || hz <= 0.0
        || !count.is_finite()
        || count > 1_000_000.0
    {
        return Err(invalid("benchmark requires finite non-negative duration, positive rate, and at most 1000000 samples"));
    }
    Ok(count as usize)
}

/// CSV artifact layout shared with C++ (time_s, cmd_0..6, actual_0..6).
pub fn benchmark_csv(
    args: &Args,
    kind: &str,
    joint: usize,
) -> Result<std::io::BufWriter<std::fs::File>> {
    use std::io::Write;
    let directory = std::path::PathBuf::from(args.text("output-dir", "results")).join(format!(
        "{kind}-{}-{}",
        args.side()?,
        std::process::id()
    ));
    std::fs::create_dir_all(&directory)?;
    let mut file = std::io::BufWriter::new(std::fs::File::create(
        directory.join(format!("joint_{joint}.csv")),
    )?);
    writeln!(file, "time_s,cmd_0,cmd_1,cmd_2,cmd_3,cmd_4,cmd_5,cmd_6,actual_0,actual_1,actual_2,actual_3,actual_4,actual_5,actual_6")?;
    Ok(file)
}

pub fn record_sample(
    file: &mut impl std::io::Write,
    seconds: f64,
    target: &[f64],
    actual: &[f64],
) -> Result {
    write!(file, "{seconds:.9}")?;
    for value in target.iter().chain(actual) {
        write!(file, ",{value:.9}")?;
    }
    writeln!(file)?;
    Ok(())
}

#[cfg(test)]
mod sample_count_tests {
    #[test]
    fn huge_or_invalid_counts_fail_before_allocation() {
        for (duration, hz) in [(1e300, 200.0), (-1.0, 200.0), (1.0, 0.0), (1.0, f64::NAN)] {
            assert!(super::sample_count(duration, hz).is_err());
        }
        assert_eq!(super::sample_count(0.3, 200.0).unwrap(), 60);
    }
}

pub mod diagnostic_display;
