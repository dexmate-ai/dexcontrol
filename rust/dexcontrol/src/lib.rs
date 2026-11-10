// Copyright (C) 2026 Dexmate Inc.
// SPDX-License-Identifier: Apache-2.0
//! Safe handles for the precompiled DexControl runtime.
//!
//! Control algorithms and safety guards execute in the native runtime. Dropping
//! a handle releases its reference; components and motions keep the robot alive.
//! Call `Robot::close` for explicit shutdown with error reporting.
//!
//! The supported interface is the safe API. Raw ABI declarations are private:
//!
//! ```compile_fail
//! use dexcontrol::ffi::dex_robot_create;
//! ```
mod ffi;
mod options;
use options::duration_ms;
pub use options::*;
mod component;
pub use component::*;
mod robot;
pub use robot::*;
mod sensors;
pub use sensors::*;
mod motion;
pub use motion::*;
mod diagnostics;
pub use diagnostics::*;
use std::{
    ffi::{CStr, CString},
    ptr::{self, NonNull},
    time::Duration,
};

/// Stable categories for branching on errors without parsing messages or magic numbers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ErrorKind {
    InvalidArgument,
    Runtime,
    Timeout,
    Panic,
    StaleState,
    StateUnavailable,
    ServiceRejected,
    EStopActive,
    Stopped,
    Unknown(i32),
}
impl Error {
    pub fn kind(&self) -> ErrorKind {
        match self.code {
            1 => ErrorKind::InvalidArgument,
            2 => ErrorKind::Runtime,
            3 => ErrorKind::Timeout,
            4 => ErrorKind::Panic,
            5 => ErrorKind::StaleState,
            6 => ErrorKind::StateUnavailable,
            7 => ErrorKind::ServiceRejected,
            8 => ErrorKind::EStopActive,
            9 => ErrorKind::Stopped,
            other => ErrorKind::Unknown(other),
        }
    }
}

/// A native error, preserving its status code and actionable diagnostic.
#[derive(Debug, Clone)]
pub struct Error {
    pub code: i32,
    pub message: String,
}
impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}
impl std::error::Error for Error {}
pub type Result<T> = std::result::Result<T, Error>;
fn invalid(message: &str) -> Error {
    Error {
        code: 1,
        message: message.into(),
    }
}
fn string(value: &str) -> Result<CString> {
    CString::new(value).map_err(|_| invalid("Strings cannot contain NUL bytes"))
}
fn call(f: impl FnOnce(*mut ffi::dex_error_t) -> i32) -> Result<()> {
    let mut error = ffi::dex_error_t::default();
    unsafe { ffi::dex_error_init(&mut error) };
    let status = f(&mut error);
    if status == 0 {
        return Ok(());
    }
    // Native code guarantees NUL termination within the initialized buffer.
    let message = unsafe { CStr::from_ptr(error.message.as_ptr()) }
        .to_string_lossy()
        .into_owned();
    Err(Error {
        code: status,
        message: if message.is_empty() {
            format!("DexControl status {status}")
        } else {
            message
        },
    })
}
fn millis(duration: Duration) -> Result<u64> {
    let value =
        u64::try_from(duration.as_millis()).map_err(|_| invalid("Duration is too large"))?;
    if value == 0 || value == u64::MAX {
        return Err(invalid(
            "Duration must be at least 1 ms and below u64::MAX ms",
        ));
    }
    Ok(value)
}
fn text(
    f: impl FnOnce(*mut *mut std::ffi::c_char, *mut ffi::dex_error_t) -> i32,
) -> Result<String> {
    let mut out = ptr::null_mut();
    let result = call(|error| f(&mut out, error));
    let value = if out.is_null() {
        String::new()
    } else {
        let value = unsafe { CStr::from_ptr(out) }
            .to_string_lossy()
            .into_owned();
        unsafe { ffi::dex_string_free(out) };
        value
    };
    result?;
    Ok(value)
}

/// How long to wait for a direct command. Motion handles have their own wait.
#[derive(Clone, Copy, Debug, Default)]
pub enum Wait {
    #[default]
    NoWait,
    UntilComplete,
    Timeout(Duration),
}
impl Wait {
    fn native(self) -> Result<ffi::dex_wait_policy_t> {
        let mut wait = ffi::dex_wait_policy_t::default();
        unsafe { ffi::dex_wait_policy_init(&mut wait) };
        match self {
            Self::NoWait => wait.mode = 0,
            Self::UntilComplete => wait.mode = 1,
            Self::Timeout(d) => {
                wait.mode = 2;
                wait.timeout_ms = millis(d)?;
            }
        }
        Ok(wait)
    }
}
/// Motion planning options. Joint limits are enforced by default.
#[derive(Clone, Copy, Debug, Default)]
pub struct MotionOptions {
    pub relative: bool,
    pub velocity_scale: Option<f64>,
    pub disable_limit_enforcement: bool,
    pub max_state_age: DurationLimit,
    pub wait_ceiling: DurationLimit,
}
impl MotionOptions {
    fn native(self) -> Result<ffi::dex_motion_options_t> {
        let mut options = ffi::dex_motion_options_t::default();
        unsafe { ffi::dex_motion_options_init(&mut options) };
        options.relative = u8::from(self.relative);
        options.disable_limit_enforcement = self.disable_limit_enforcement.into();
        options.max_state_age_ms = self.max_state_age.native()?;
        options.wait_ceiling_ms = self.wait_ceiling.native()?;
        if let Some(scale) = self.velocity_scale {
            if !scale.is_finite() || scale <= 0.0 || scale > 1.0 {
                return Err(invalid("velocity_scale must be finite and in (0, 1]"));
            }
            options.velocity_scale = scale;
        }
        Ok(options)
    }
}
/// Configuration strings are borrowed only during connection.
#[derive(Default)]
pub struct ConnectOptions<'a> {
    pub profile: Option<&'a str>,
    pub config_file: Option<&'a str>,
    pub simulated: bool,
    pub enable_sensors: Vec<&'a str>,
    pub require_version_check: bool,
    pub disable_watchdog: bool,
    pub exit_on_termination: bool,
    pub watchdog_command: Option<&'a str>,
    /// Connect without putting the arms into position mode. The default
    /// enables every arm once it is ready, and a required arm whose motor
    /// cannot enable then fails the connection; set this to clear that
    /// arm's error, or when the application sets modes itself.
    pub skip_arm_initialization: bool,
}
/// An owning robot connection. Share with `Arc` when using several threads.
pub struct Robot(NonNull<ffi::dex_robot_t>);
impl Robot {
    pub fn connect(options: ConnectOptions<'_>) -> Result<Self> {
        options.with_native(|native| {
            let mut out = ptr::null_mut();
            let result = call(|error| unsafe { ffi::dex_robot_create(native, &mut out, error) });
            if let Err(error) = result {
                if !out.is_null() {
                    unsafe { ffi::dex_robot_release(out) };
                }
                return Err(error);
            }
            Ok(Self(
                NonNull::new(out).ok_or_else(|| invalid("Runtime returned no robot"))?,
            ))
        })
    }
    pub fn simulated(profile: &str) -> Result<Self> {
        Self::connect(ConnectOptions {
            profile: Some(profile),
            simulated: true,
            ..Default::default()
        })
    }
    pub fn joints(&self, name: &str) -> Result<JointComponent> {
        let name = string(name)?;
        let mut out = ptr::null_mut();
        call(|error| unsafe {
            ffi::dex_robot_joint_component(self.0.as_ptr(), name.as_ptr(), &mut out, error)
        })?;
        Ok(JointComponent(
            NonNull::new(out).ok_or_else(|| invalid("Runtime returned no component"))?,
        ))
    }
    /// Stops motion and closes the connection; `close` is the same call.
    pub fn shutdown(&self) -> Result<()> {
        call(|error| unsafe { ffi::dex_robot_close(self.0.as_ptr(), error) })
    }
    pub fn close(&self) -> Result<()> {
        self.shutdown()
    }
    pub fn stop_all(&self) -> Result<()> {
        call(|error| unsafe { ffi::dex_robot_stop_all(self.0.as_ptr(), error) })
    }
    pub fn model_name(&self) -> Result<String> {
        text(|out, error| unsafe { ffi::dex_robot_model_name(self.0.as_ptr(), out, error) })
    }
    pub fn health_json(&self) -> Result<String> {
        text(|out, error| unsafe { ffi::dex_robot_health_json(self.0.as_ptr(), out, error) })
    }
    pub fn joint_state_json(&self, name: &str) -> Result<String> {
        let name = string(name)?;
        text(|out, error| unsafe {
            ffi::dex_robot_joint_state_json(self.0.as_ptr(), name.as_ptr(), out, error)
        })
    }
    /// Engages the software E-stop.
    pub fn estop_activate(&self) -> Result<()> {
        call(|e| unsafe { ffi::dex_robot_estop_set(self.0.as_ptr(), true, e) })
    }
    /// Releases the software E-stop.
    pub fn estop_deactivate(&self) -> Result<()> {
        call(|e| unsafe { ffi::dex_robot_estop_set(self.0.as_ptr(), false, e) })
    }
    /// Deprecated: use [`Robot::estop_activate`] / [`Robot::estop_deactivate`].
    #[deprecated(since = "0.8.1", note = "use estop_activate / estop_deactivate")]
    pub fn set_software_estop(&self, enabled: bool) -> Result<()> {
        call(|e| unsafe { ffi::dex_robot_estop_set(self.0.as_ptr(), enabled, e) })
    }
    pub fn chassis_set_velocity(&self, name: &str, vx: f64, vy: f64, wz: f64) -> Result<()> {
        let name = string(name)?;
        call(|e| unsafe {
            ffi::dex_robot_chassis_set_velocity_named(self.0.as_ptr(), name.as_ptr(), vx, vy, wz, e)
        })
    }
    pub fn chassis_stop(&self, name: &str) -> Result<()> {
        let name = string(name)?;
        call(|e| unsafe { ffi::dex_robot_chassis_stop_named(self.0.as_ptr(), name.as_ptr(), e) })
    }
}
impl Drop for Robot {
    fn drop(&mut self) {
        unsafe { ffi::dex_robot_release(self.0.as_ptr()) }
    }
}

/// Owns a native component reference, independently of its originating Robot.
pub struct JointComponent(NonNull<ffi::dex_component_t>);
impl JointComponent {
    pub fn name(&self) -> Result<String> {
        text(|out, error| unsafe { ffi::dex_component_name(self.0.as_ptr(), out, error) })
    }
    fn read_array(
        &self,
        f: impl FnOnce(*mut f64, usize, *mut usize, *mut ffi::dex_error_t) -> i32,
    ) -> Result<Vec<f64>> {
        let mut count = 0;
        call(|_| unsafe { ffi::dex_component_joint_count(self.0.as_ptr(), &mut count) })?;
        let mut values = vec![0.0; count];
        let mut written = 0;
        call(|error| f(values.as_mut_ptr(), count, &mut written, error))?;
        if written > count {
            return Err(invalid("Runtime returned an invalid joint count"));
        }
        values.truncate(written);
        Ok(values)
    }
    pub fn get_joint_pos(&self) -> Result<Vec<f64>> {
        self.read_array(|out, cap, count, error| unsafe {
            ffi::dex_component_get_joint_pos(self.0.as_ptr(), out, cap, count, error)
        })
    }
    pub fn get_pose(&self, name: &str, passthrough: bool) -> Result<Vec<f64>> {
        let name = string(name)?;
        self.read_array(|out, cap, count, error| unsafe {
            ffi::dex_component_get_pose(
                self.0.as_ptr(),
                name.as_ptr(),
                passthrough,
                out,
                cap,
                count,
                error,
            )
        })
    }
    pub fn set_joint_pos(&self, target: &[f64], wait: Wait) -> Result<()> {
        let wait = wait.native()?;
        call(|error| unsafe {
            ffi::dex_component_set_joint_pos(
                self.0.as_ptr(),
                target.as_ptr(),
                target.len(),
                &wait,
                error,
            )
        })
    }
    /// Starts a tracked move without waiting. On failure the error carries the
    /// motion handle when the command may have reached the robot; `?` into
    /// [`Error`] drops it.
    pub fn move_to_joint_pos(
        &self,
        target: &[f64],
        options: MotionOptions,
    ) -> std::result::Result<MotionHandle, MotionStartError<MotionHandle>> {
        let options = options.native()?;
        let wait = Wait::NoWait.native()?;
        MotionHandle::start_recoverable(|out, error| unsafe {
            ffi::dex_component_move_to_joint_pos_opt(
                self.0.as_ptr(),
                target.as_ptr(),
                target.len(),
                &options,
                &wait,
                out,
                error,
            )
        })
    }
    /// Deprecated: [`JointComponent::move_to_joint_pos`] now returns the same
    /// `MotionStartError`, with the handle when the command may have reached the robot.
    #[deprecated(since = "0.8.1", note = "use move_to_joint_pos")]
    pub fn move_to_joint_pos_recoverable(
        &self,
        target: &[f64],
        options: MotionOptions,
    ) -> std::result::Result<MotionHandle, MotionStartError<MotionHandle>> {
        self.move_to_joint_pos(target, options)
    }
    /// Starts a tracked move to a named pose without waiting; the error carries
    /// the handle like [`JointComponent::move_to_joint_pos`].
    pub fn go_to_pose(
        &self,
        pose: &str,
        passthrough: bool,
        options: MotionOptions,
    ) -> std::result::Result<MotionHandle, MotionStartError<MotionHandle>> {
        let pose = string(pose)?;
        let options = options.native()?;
        let wait = Wait::NoWait.native()?;
        MotionHandle::start_recoverable(|out, error| unsafe {
            ffi::dex_component_go_to_pose_opt(
                self.0.as_ptr(),
                pose.as_ptr(),
                &options,
                &wait,
                passthrough,
                out,
                error,
            )
        })
    }
    /// Deprecated: [`JointComponent::go_to_pose`] now returns the same
    /// `MotionStartError`, with the handle when the command may have reached the robot.
    #[deprecated(since = "0.8.1", note = "use go_to_pose")]
    pub fn go_to_pose_recoverable(
        &self,
        pose: &str,
        passthrough: bool,
        options: MotionOptions,
    ) -> std::result::Result<MotionHandle, MotionStartError<MotionHandle>> {
        self.go_to_pose(pose, passthrough, options)
    }
}
impl Drop for JointComponent {
    fn drop(&mut self) {
        unsafe { ffi::dex_component_release(self.0.as_ptr()) }
    }
}
#[derive(Debug, PartialEq, Eq)]
pub enum MotionState {
    Pending,
    Running,
    Succeeded,
    Cancelled,
    Failed,
    Superseded,
    Unknown(u32),
}
/// Owns one motion reference. Dropping it does not implicitly cancel a motion.
#[derive(Debug)]
pub struct MotionHandle(NonNull<ffi::dex_motion_t>);
impl MotionHandle {
    /// Takes ownership of the handle an out-pointer call produced.
    fn start(
        f: impl FnOnce(*mut *mut ffi::dex_motion_t, *mut ffi::dex_error_t) -> i32,
    ) -> Result<Self> {
        Self::start_recoverable(f).map_err(|failure| failure.error)
    }
    fn start_recoverable(
        f: impl FnOnce(*mut *mut ffi::dex_motion_t, *mut ffi::dex_error_t) -> i32,
    ) -> std::result::Result<Self, MotionStartError<Self>> {
        let mut out = ptr::null_mut();
        let result = call(|error| f(&mut out, error));
        let handle = NonNull::new(out).map(Self);
        match result {
            Err(error) => Err(MotionStartError { error, handle }),
            Ok(()) => handle.ok_or_else(|| invalid("Runtime returned no motion").into()),
        }
    }
    /// Explicit success wait; same contract as the existing wait method.
    pub fn wait_success(&self, timeout: Duration) -> Result<()> {
        self.wait(timeout)
    }
    pub fn wait(&self, timeout: Duration) -> Result<()> {
        let wait = Wait::Timeout(timeout).native()?;
        call(|error| unsafe { ffi::dex_motion_wait_success(self.0.as_ptr(), &wait, error) })
    }
    /// Wait for a terminal outcome; cancellation is returned as a state.
    pub fn wait_terminal(&self, timeout: Duration) -> Result<MotionState> {
        let wait = Wait::Timeout(timeout).native()?;
        call(|error| unsafe { ffi::dex_motion_wait(self.0.as_ptr(), &wait, error) })?;
        self.state()
    }
    pub fn cancel_and_wait(&self, timeout: Duration) -> Result<MotionState> {
        self.cancel()?;
        self.wait_terminal(timeout)
    }
    pub fn cancel(&self) -> Result<()> {
        call(|error| unsafe { ffi::dex_motion_cancel(self.0.as_ptr(), error) })
    }
    pub fn state(&self) -> Result<MotionState> {
        let mut state = 0;
        call(|_| unsafe { ffi::dex_motion_state(self.0.as_ptr(), &mut state) })?;
        Ok(MotionState::from_native(state))
    }
}
impl Drop for MotionHandle {
    fn drop(&mut self) {
        unsafe { ffi::dex_motion_release(self.0.as_ptr()) }
    }
}

/// A failed submission with an owning recovery handle when publication was ambiguous.
#[derive(Debug)]
pub struct MotionStartError<T> {
    pub error: Error,
    pub handle: Option<T>,
}
impl<T> From<Error> for MotionStartError<T> {
    fn from(error: Error) -> Self {
        Self {
            error,
            handle: None,
        }
    }
}
impl<T> std::fmt::Display for MotionStartError<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.error.fmt(f)
    }
}
/// `?` in a function returning [`Result`] drops the handle and keeps the error.
impl<T> From<MotionStartError<T>> for Error {
    fn from(failure: MotionStartError<T>) -> Self {
        failure.error
    }
}

impl<T: std::fmt::Debug> std::error::Error for MotionStartError<T> {}

/// Battery readings in volts, amperes, degrees Celsius and percent.
pub use ffi::dex_battery_t as Battery;
/// All reported emergency stop sources, including hardware and software.
pub use ffi::dex_estop_status_t as EStopStatus;
impl Robot {
    pub fn battery(&self) -> Result<Battery> {
        let mut value = Battery {
            struct_size: std::mem::size_of::<Battery>() as u32,
            ..Default::default()
        };
        call(|e| unsafe { ffi::dex_robot_battery(self.0.as_ptr(), &mut value, e) })?;
        Ok(value)
    }
    pub fn estop_status(&self) -> Result<EStopStatus> {
        let mut value = EStopStatus {
            struct_size: std::mem::size_of::<EStopStatus>() as u32,
            ..Default::default()
        };
        call(|e| unsafe { ffi::dex_robot_estop_status(self.0.as_ptr(), &mut value, e) })?;
        Ok(value)
    }
    pub fn component_names(&self) -> Result<Vec<String>> {
        let mut count = 0;
        call(|_| unsafe { ffi::dex_robot_component_count(self.0.as_ptr(), &mut count) })?;
        (0..count)
            .map(|i| {
                text(|out, e| unsafe { ffi::dex_robot_component_name(self.0.as_ptr(), i, out, e) })
            })
            .collect()
    }
}
impl JointComponent {
    pub fn joint_names(&self) -> Result<Vec<String>> {
        let mut count = 0;
        call(|_| unsafe { ffi::dex_component_joint_count(self.0.as_ptr(), &mut count) })?;
        (0..count)
            .map(|i| {
                text(|out, e| unsafe { ffi::dex_component_joint_name(self.0.as_ptr(), i, out, e) })
            })
            .collect()
    }
    pub fn pose_names(&self) -> Result<Vec<String>> {
        let mut count = 0;
        call(|_| unsafe { ffi::dex_component_pose_count(self.0.as_ptr(), &mut count) })?;
        (0..count)
            .map(|i| {
                text(|out, e| unsafe { ffi::dex_component_pose_name(self.0.as_ptr(), i, out, e) })
            })
            .collect()
    }
}

// SAFETY: The public C ABI guarantees concurrent calls on these handles are
// internally synchronized. Rust borrows/Arc ownership ensure Drop cannot race
// with a call on the same handle. Each component/motion owns a native reference.
unsafe impl Send for Robot {}
unsafe impl Sync for Robot {}
unsafe impl Send for JointComponent {}
unsafe impl Sync for JointComponent {}
unsafe impl Send for MotionHandle {}
unsafe impl Sync for MotionHandle {}

#[cfg(test)]
mod recovery_tests {
    use super::*;
    #[test]
    fn failed_ffi_start_retains_non_null_owning_handle() {
        let robot = Robot::simulated("vega_1").unwrap();
        let motion = robot
            .joints("head")
            .unwrap()
            .move_to_joint_pos(&[0.0; 3], MotionOptions::default())
            .unwrap();
        let failure = MotionHandle::start_recoverable(|out, _| {
            unsafe {
                *out = motion.0.as_ptr();
            }
            std::mem::forget(motion);
            ffi::DEX_TIMEOUT
        })
        .unwrap_err();
        assert_eq!(failure.error.kind(), ErrorKind::Timeout);
        assert_eq!(
            failure.handle.unwrap().refresh().unwrap(),
            MotionState::Succeeded
        );
        robot.close().unwrap();
    }
}
