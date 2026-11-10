// Copyright (C) 2026 Dexmate Inc.
// SPDX-License-Identifier: Apache-2.0
//! Command options retain native safety defaults unless explicitly overridden.
use crate::*;

/// A configured limit, an explicit positive duration, or an expert opt-out.
#[derive(Clone, Copy, Debug, Default)]
pub enum DurationLimit {
    #[default]
    Configured,
    Maximum(Duration),
    Unlimited,
}
impl DurationLimit {
    pub(crate) fn native(self) -> Result<u64> {
        match self {
            Self::Configured => Ok(0),
            Self::Maximum(d) => millis(d),
            Self::Unlimited => Ok(u64::MAX),
        }
    }
}
#[derive(Clone, Copy, Debug, Default)]
pub enum StepLimit {
    #[default]
    Configured,
    Maximum(f64),
    Unlimited,
}
impl StepLimit {
    pub(crate) fn native(self) -> Result<f64> {
        match self {
            Self::Configured => Ok(0.0),
            Self::Unlimited => Ok(-1.0),
            Self::Maximum(v) if v.is_finite() && v > 0.0 => Ok(v),
            _ => Err(invalid("Step limit must be finite and positive")),
        }
    }
}
#[derive(Clone, Copy, Debug, Default)]
pub struct WaitOptions {
    pub policy: Wait,
    pub convergence_tolerance: Option<f64>,
    pub settle_time: Duration,
    pub max_state_age: DurationLimit,
    pub wait_ceiling: DurationLimit,
}
impl From<Wait> for WaitOptions {
    fn from(policy: Wait) -> Self {
        Self {
            policy,
            ..Default::default()
        }
    }
}
impl WaitOptions {
    pub(crate) fn native(self) -> Result<ffi::dex_wait_policy_t> {
        let mut n = self.policy.native()?;
        if let Some(v) = self.convergence_tolerance {
            if !v.is_finite() || v <= 0.0 {
                return Err(invalid("Convergence tolerance must be finite and positive"));
            }
            n.convergence_tolerance = v;
        }
        n.settle_time_ms = duration_ms(self.settle_time)?;
        n.max_state_age_ms = self.max_state_age.native()?;
        n.wait_ceiling_ms = self.wait_ceiling.native()?;
        Ok(n)
    }
}
#[derive(Clone, Copy, Debug, Default)]
pub struct CommandOptions {
    pub relative: bool,
    pub disable_limit_enforcement: bool,
    pub max_step: StepLimit,
}
impl CommandOptions {
    pub(crate) fn native(self) -> Result<ffi::dex_command_options_t> {
        let mut n = ffi::dex_command_options_t::default();
        unsafe { ffi::dex_command_options_init(&mut n) };
        n.relative = self.relative.into();
        n.disable_limit_enforcement = self.disable_limit_enforcement.into();
        n.max_step_rad = self.max_step.native()?;
        Ok(n)
    }
}
#[derive(Clone, Copy, Debug, Default)]
pub struct TrajectoryOptions {
    /// None retains the native default. Some(0.0) explicitly disables tracking-error checking.
    pub max_tracking_error: Option<f64>,
    pub max_state_age: DurationLimit,
    pub max_step: StepLimit,
}
impl TrajectoryOptions {
    pub(crate) fn native(self) -> Result<ffi::dex_trajectory_options_t> {
        let mut n = ffi::dex_trajectory_options_t::default();
        unsafe { ffi::dex_trajectory_options_init(&mut n) };
        if let Some(v) = self.max_tracking_error {
            if !v.is_finite() || v < 0.0 {
                return Err(invalid("Tracking error must be finite and non-negative"));
            }
            n.max_tracking_error = v;
        }
        n.max_state_age_ms = self.max_state_age.native()?;
        n.max_step_rad = self.max_step.native()?;
        Ok(n)
    }
}
#[derive(Clone, Copy, Debug, Default)]
pub struct ChassisOptions {
    pub disable_sequential_steering: bool,
    pub steering_tolerance: Option<f64>,
    pub steering_wait: Option<Duration>,
    pub max_state_age: DurationLimit,
}
impl ChassisOptions {
    pub(crate) fn native(self) -> Result<ffi::dex_chassis_options_t> {
        let mut n = ffi::dex_chassis_options_t::default();
        unsafe { ffi::dex_chassis_options_init(&mut n) };
        n.disable_sequential_steering = self.disable_sequential_steering.into();
        if let Some(v) = self.steering_tolerance {
            if !v.is_finite() || v <= 0.0 {
                return Err(invalid("Steering tolerance must be finite and positive"));
            }
            n.steering_tolerance = v;
        }
        n.steering_wait_ms = self.steering_wait.map(millis).transpose()?.unwrap_or(0);
        n.max_state_age_ms = self.max_state_age.native()?;
        Ok(n)
    }
}

pub(crate) fn duration_ms(d: Duration) -> Result<u64> {
    let n = u64::try_from(d.as_millis()).map_err(|_| invalid("Duration is too large"))?;
    if n == u64::MAX || (!d.is_zero() && n == 0) {
        return Err(invalid(
            "Duration must be zero or at least 1 ms and below u64::MAX ms",
        ));
    }
    Ok(n)
}

/// Where the odometry heading comes from. See the native
/// `dexcontrol::YawSource`.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum YawSource {
    /// An enabled base-mounted IMU (`chassis_imu`) when there is one, else
    /// the wheels.
    #[default]
    Auto,
    /// The named enabled IMU.
    Imu(String),
    /// The wheels only.
    Wheels,
}

impl YawSource {
    pub(crate) fn native(&self) -> Result<CString> {
        string(match self {
            Self::Auto => "auto",
            Self::Wheels => "wheels",
            Self::Imu(name) => name.as_str(),
        })
    }
}

/// Odometry configuration for [`crate::Robot::start_odometry`]. `None`
/// keeps each default (100 Hz, 1 m per encoder unit, 0.5 m gap guard,
/// 0.3 rad/s slip threshold).
#[derive(Debug, Clone, PartialEq, Default)]
pub struct OdometryOptions {
    pub yaw_source: YawSource,
    pub sample_hz: Option<f64>,
    /// Metres per encoder unit, left then right.
    pub encoder_scale: Option<[f64; 2]>,
    pub max_sample_travel: Option<f64>,
    pub slip_rate_threshold: Option<f64>,
}

impl OdometryOptions {
    pub(crate) fn native(&self) -> Result<ffi::dex_odometry_options_t> {
        let mut n = ffi::dex_odometry_options_t::default();
        unsafe { ffi::dex_odometry_options_init(&mut n) };
        for (name, value) in [
            ("sample_hz", self.sample_hz),
            ("max_sample_travel", self.max_sample_travel),
            ("slip_rate_threshold", self.slip_rate_threshold),
        ] {
            if value.is_some_and(|v| !v.is_finite() || v <= 0.0) {
                return Err(invalid(&format!("{name} must be finite and positive")));
            }
        }
        n.sample_hz = self.sample_hz.unwrap_or(0.0);
        n.encoder_scale = self.encoder_scale.unwrap_or([0.0; 2]);
        n.max_sample_travel = self.max_sample_travel.unwrap_or(0.0);
        n.slip_rate_threshold = self.slip_rate_threshold.unwrap_or(0.0);
        Ok(n)
    }
}

/// Options of [`crate::Chassis::turn_by`] and [`crate::Chassis::move_by`].
/// `None` keeps each default: cruise 0.5 rad/s or 0.2 m/s, 2° or 2 cm,
/// 3 passes, 60 s, 300 ms.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct ClosedLoopOptions {
    /// Cruise rate: rad/s for a turn, m/s for a move.
    pub speed: Option<f64>,
    /// rad for a turn, m for a move.
    pub tolerance: Option<f64>,
    /// Passes: the motion plus correction passes.
    pub max_steps: Option<u32>,
    pub timeout: Option<Duration>,
    pub settle: Option<Duration>,
}

impl ClosedLoopOptions {
    pub(crate) fn native(&self) -> Result<ffi::dex_closed_loop_options_t> {
        let mut n = ffi::dex_closed_loop_options_t::default();
        unsafe { ffi::dex_closed_loop_options_init(&mut n) };
        for (name, value) in [("speed", self.speed), ("tolerance", self.tolerance)] {
            if value.is_some_and(|v| !v.is_finite() || v <= 0.0) {
                return Err(invalid(&format!("{name} must be finite and positive")));
            }
        }
        n.speed = self.speed.unwrap_or(0.0);
        n.tolerance = self.tolerance.unwrap_or(0.0);
        n.max_steps = self.max_steps.unwrap_or(0);
        n.timeout_ms = self.timeout.map(millis).transpose()?.unwrap_or(0);
        n.settle_ms = self.settle.map(millis).transpose()?.unwrap_or(0);
        Ok(n)
    }
}
