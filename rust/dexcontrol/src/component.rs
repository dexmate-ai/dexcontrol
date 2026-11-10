// Copyright (C) 2026 Dexmate Inc.
// SPDX-License-Identifier: Apache-2.0
//! Joint feedback and firmware operations through the public native ABI.
use crate::*;
#[derive(Debug, Clone)]
pub struct JointLimits {
    pub lower: Vec<f64>,
    pub upper: Vec<f64>,
    pub velocity: Vec<f64>,
}
#[derive(Debug, Clone)]
pub struct JointState {
    pub position: Vec<f64>,
    pub velocity: Vec<f64>,
    pub torque: Vec<f64>,
    pub current: Vec<f64>,
    pub timestamp_ns: u64,
    pub age: Duration,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(i32)]
pub enum JointMode {
    Disable = 1,
    Enable = 2,
    Calibration = 3,
    Position = 4,
    Velocity = 5,
    Torque = 6,
    Current = 7,
}
impl JointComponent {
    fn count(&self) -> Result<usize> {
        let mut n = 0;
        call(|_| unsafe { ffi::dex_component_joint_count(self.0.as_ptr(), &mut n) })?;
        Ok(n)
    }
    pub fn joint_limits(&self) -> Result<Option<JointLimits>> {
        let n = self.count()?;
        let mut lower = vec![0.; n];
        let mut upper = lower.clone();
        let mut velocity = lower.clone();
        let mut count = 0;
        let mut present = false;
        call(|e| unsafe {
            ffi::dex_component_joint_limits(
                self.0.as_ptr(),
                lower.as_mut_ptr(),
                upper.as_mut_ptr(),
                velocity.as_mut_ptr(),
                n,
                &mut count,
                &mut present,
                e,
            )
        })?;
        if count > n {
            return Err(invalid("Invalid joint limit count"));
        }
        lower.truncate(count);
        upper.truncate(count);
        velocity.truncate(count);
        Ok(present.then_some(JointLimits {
            lower,
            upper,
            velocity,
        }))
    }
    /// Read independent channel lengths and local arrival age, without fabricating missing values.
    pub fn get_joint_state(&self, max_age: DurationLimit) -> Result<JointState> {
        let n = self.count()?;
        let mut position = vec![0.; n];
        let mut velocity = position.clone();
        let mut torque = position.clone();
        let mut current = position.clone();
        let mut counts = ffi::dex_joint_state_counts_t {
            struct_size: std::mem::size_of::<ffi::dex_joint_state_counts_t>() as u32,
            ..Default::default()
        };
        let mut timestamp = 0;
        let mut age = 0;
        let max_age = max_age.native()?;
        call(|e| unsafe {
            ffi::dex_component_get_joint_state_ex(
                self.0.as_ptr(),
                max_age,
                position.as_mut_ptr(),
                velocity.as_mut_ptr(),
                torque.as_mut_ptr(),
                current.as_mut_ptr(),
                n,
                &mut counts,
                &mut timestamp,
                &mut age,
                e,
            )
        })?;
        for (v, count) in [
            (&mut position, counts.position),
            (&mut velocity, counts.velocity),
            (&mut torque, counts.torque),
            (&mut current, counts.current),
        ] {
            if count > n {
                return Err(invalid("Invalid joint state count"));
            }
            v.truncate(count);
        }
        Ok(JointState {
            position,
            velocity,
            torque,
            current,
            timestamp_ns: timestamp,
            age: Duration::from_millis(age),
        })
    }
    pub fn get_joint_err(&self) -> Result<Vec<u32>> {
        array(|v, n, c, e| unsafe { ffi::dex_component_get_joint_err(self.0.as_ptr(), v, n, c, e) })
    }
    pub fn is_joint_pos_reached(&self, target: &[f64], tolerance: f64) -> Result<bool> {
        let mut yes = false;
        call(|e| unsafe {
            ffi::dex_component_is_joint_pos_reached(
                self.0.as_ptr(),
                target.as_ptr(),
                target.len(),
                tolerance,
                &mut yes,
                e,
            )
        })?;
        Ok(yes)
    }
    pub fn set_joint_pos_with(
        &self,
        target: &[f64],
        velocities: Option<&[f64]>,
        options: CommandOptions,
        wait: WaitOptions,
    ) -> Result<()> {
        if velocities.is_some_and(|v| v.len() != target.len()) {
            return Err(invalid("Position and velocity lengths differ"));
        }
        let options = options.native()?;
        let wait = wait.native()?;
        call(|e| unsafe {
            ffi::dex_component_set_joint_pos_opt(
                self.0.as_ptr(),
                target.as_ptr(),
                velocities.map_or(ptr::null(), |v| v.as_ptr()),
                target.len(),
                &options,
                &wait,
                e,
            )
        })
    }
    /// Starts a tracked move and applies `wait` before returning. On failure the
    /// error carries the motion handle when the command may have reached the
    /// robot, so the caller can cancel or observe it; `?` into [`Error`] drops it.
    pub fn move_to_joint_pos_with(
        &self,
        target: &[f64],
        options: MotionOptions,
        wait: WaitOptions,
    ) -> std::result::Result<MotionHandle, MotionStartError<MotionHandle>> {
        let options = options.native()?;
        let wait = wait.native()?;
        MotionHandle::start_recoverable(|out, e| unsafe {
            ffi::dex_component_move_to_joint_pos_opt(
                self.0.as_ptr(),
                target.as_ptr(),
                target.len(),
                &options,
                &wait,
                out,
                e,
            )
        })
    }
    /// Deprecated: [`JointComponent::move_to_joint_pos_with`] now returns the same
    /// `MotionStartError`, with the handle when the command may have reached the robot.
    #[deprecated(since = "0.8.1", note = "use move_to_joint_pos_with")]
    pub fn move_to_joint_pos_with_recoverable(
        &self,
        target: &[f64],
        options: MotionOptions,
        wait: WaitOptions,
    ) -> std::result::Result<MotionHandle, MotionStartError<MotionHandle>> {
        self.move_to_joint_pos_with(target, options, wait)
    }
    /// Streams an untracked goal to the motion plugin: the counterpart of
    /// `move_to_joint_pos_with` for a control loop that updates the goal
    /// every tick. The plugin smooths towards the newest goal within its own
    /// velocity, acceleration and jerk bounds, so no step guard applies and
    /// nothing is returned to wait on. Refused while a tracked motion of this
    /// component is still running. `wait_ceiling` in the options is ignored.
    pub fn stream_joint_pos(&self, target: &[f64], options: MotionOptions) -> Result<()> {
        let options = options.native()?;
        call(|e| unsafe {
            ffi::dex_component_stream_joint_pos(
                self.0.as_ptr(),
                target.as_ptr(),
                target.len(),
                &options,
                e,
            )
        })
    }
    /// Starts a tracked move to a named pose and applies `wait` before
    /// returning; the error carries the handle like [`JointComponent::move_to_joint_pos_with`].
    pub fn go_to_pose_with(
        &self,
        pose: &str,
        passthrough: bool,
        options: MotionOptions,
        wait: WaitOptions,
    ) -> std::result::Result<MotionHandle, MotionStartError<MotionHandle>> {
        let pose = string(pose)?;
        let options = options.native()?;
        let wait = wait.native()?;
        MotionHandle::start_recoverable(|out, e| unsafe {
            ffi::dex_component_go_to_pose_opt(
                self.0.as_ptr(),
                pose.as_ptr(),
                &options,
                &wait,
                passthrough,
                out,
                e,
            )
        })
    }
    /// Deprecated: [`JointComponent::go_to_pose_with`] now returns the same
    /// `MotionStartError`, with the handle when the command may have reached the robot.
    #[deprecated(since = "0.8.1", note = "use go_to_pose_with")]
    pub fn go_to_pose_with_recoverable(
        &self,
        pose: &str,
        passthrough: bool,
        options: MotionOptions,
        wait: WaitOptions,
    ) -> std::result::Result<MotionHandle, MotionStartError<MotionHandle>> {
        self.go_to_pose_with(pose, passthrough, options, wait)
    }
    /// Starts a tracked joint-space trajectory and applies `wait` before
    /// returning; the error carries the handle like [`JointComponent::move_to_joint_pos_with`].
    pub fn move_joint_trajectory(
        &self,
        positions: &[Vec<f64>],
        times: &[f64],
        options: MotionOptions,
        wait: WaitOptions,
    ) -> std::result::Result<MotionHandle, MotionStartError<MotionHandle>> {
        let joints = rectangular(positions)?;
        if times.len() != positions.len() {
            return Err(invalid("One timestamp is required per waypoint").into());
        }
        let flat: Vec<f64> = positions.iter().flatten().copied().collect();
        let options = options.native()?;
        let wait = wait.native()?;
        MotionHandle::start_recoverable(|out, e| unsafe {
            ffi::dex_component_move_joint_trajectory_opt(
                self.0.as_ptr(),
                flat.as_ptr(),
                times.as_ptr(),
                times.len(),
                joints,
                &options,
                &wait,
                out,
                e,
            )
        })
    }
    /// Deprecated: [`JointComponent::move_joint_trajectory`] now returns the same
    /// `MotionStartError`, with the handle when the command may have reached the robot.
    #[deprecated(since = "0.8.1", note = "use move_joint_trajectory")]
    pub fn move_joint_trajectory_recoverable(
        &self,
        positions: &[Vec<f64>],
        times: &[f64],
        options: MotionOptions,
        wait: WaitOptions,
    ) -> std::result::Result<MotionHandle, MotionStartError<MotionHandle>> {
        self.move_joint_trajectory(positions, times, options, wait)
    }
    /// Send a position command with optional per-joint step limits and a feedback-age bound.
    pub fn command_position(
        &self,
        target: &[f64],
        limits: Option<&[f64]>,
        max_state_age: DurationLimit,
    ) -> Result<()> {
        if limits.is_some_and(|v| v.len() != target.len()) {
            return Err(invalid("Position and limit lengths differ"));
        }
        let age = max_state_age.native()?;
        call(|e| unsafe {
            ffi::dex_component_command_position(
                self.0.as_ptr(),
                target.as_ptr(),
                limits.map_or(ptr::null(), |v| v.as_ptr()),
                target.len(),
                age,
                e,
            )
        })
    }
    /// Deprecated alias of [`JointComponent::get_joint_state`].
    #[deprecated(since = "0.8.1", note = "use get_joint_state")]
    pub fn joint_state(&self, max_age: DurationLimit) -> Result<JointState> {
        self.get_joint_state(max_age)
    }
    /// Deprecated alias of [`JointComponent::get_joint_err`].
    #[deprecated(since = "0.8.1", note = "use get_joint_err")]
    pub fn joint_errors(&self) -> Result<Vec<u32>> {
        self.get_joint_err()
    }
    /// Deprecated alias of [`JointComponent::get_modes`].
    #[deprecated(since = "0.8.1", note = "use get_modes")]
    pub fn modes(&self) -> Result<Vec<Option<JointMode>>> {
        self.get_modes()
    }
    /// Sets one firmware mode on every joint; see [`JointComponent::set_modes`].
    pub fn set_mode(&self, mode: JointMode) -> Result<String> {
        let count = self.count()?;
        self.set_modes(&vec![mode; count])
    }
    /// Deprecated alias of [`JointComponent::get_idle_mode`].
    #[deprecated(since = "0.8.1", note = "use get_idle_mode")]
    pub fn idle_mode(&self) -> Result<Option<bool>> {
        self.get_idle_mode()
    }
    /// Deprecated alias of [`JointComponent::get_button_state`].
    #[deprecated(since = "0.8.1", note = "use get_button_state")]
    pub fn button_state(&self) -> Result<Option<(bool, bool)>> {
        self.get_button_state()
    }
    /// Deprecated alias of [`JointComponent::get_touch_data`].
    #[deprecated(since = "0.8.1", note = "use get_touch_data")]
    pub fn touch_forces(&self) -> Result<Option<Vec<f64>>> {
        self.get_touch_data()
    }
    /// Deprecated alias of [`JointComponent::get_brake_status`].
    #[deprecated(since = "0.8.1", note = "use get_brake_status")]
    pub fn brake_status(&self) -> Result<String> {
        self.get_brake_status()
    }
    /// Deprecated alias of [`JointComponent::get_force_torque_sensor_mode`].
    #[deprecated(since = "0.8.1", note = "use get_force_torque_sensor_mode")]
    pub fn force_torque_sensor_mode(&self) -> Result<String> {
        self.get_force_torque_sensor_mode()
    }
    /// Deprecated alias of [`JointComponent::get_ee_baud_rate`].
    #[deprecated(since = "0.8.1", note = "use get_ee_baud_rate")]
    pub fn ee_baud_rate(&self) -> Result<String> {
        self.get_ee_baud_rate()
    }
    /// Whether the state stream is usable now: a sample within the configured
    /// freshness limit.
    pub fn is_active(&self) -> Result<bool> {
        let mut active = false;
        call(|_| unsafe { ffi::dex_component_is_active(self.0.as_ptr(), &mut active) })?;
        Ok(active)
    }
    /// Waits for the state stream to become usable; `Ok(false)` on timeout.
    pub fn wait_for_active(&self, timeout: Duration) -> Result<bool> {
        let timeout = duration_ms(timeout)?;
        let mut active = false;
        call(|e| unsafe {
            ffi::dex_component_wait_for_active(self.0.as_ptr(), timeout, &mut active, e)
        })?;
        Ok(active)
    }
    pub fn stop(&self) -> Result<()> {
        call(|e| unsafe { ffi::dex_component_stop(self.0.as_ptr(), e) })
    }
    /// Unknown native modes are returned as None, not silently mapped to an enabled mode.
    pub fn get_modes(&self) -> Result<Vec<Option<JointMode>>> {
        let codes: Vec<i32> = array(|v, n, c, e| unsafe {
            ffi::dex_component_get_modes(self.0.as_ptr(), v, n, c, e)
        })?;
        Ok(codes
            .into_iter()
            .map(|c| match c {
                1 => Some(JointMode::Disable),
                2 => Some(JointMode::Enable),
                3 => Some(JointMode::Calibration),
                4 => Some(JointMode::Position),
                5 => Some(JointMode::Velocity),
                6 => Some(JointMode::Torque),
                7 => Some(JointMode::Current),
                _ => None,
            })
            .collect())
    }
    pub fn set_modes(&self, modes: &[JointMode]) -> Result<String> {
        let modes: Vec<i32> = modes.iter().map(|v| *v as i32).collect();
        text(|out, e| unsafe {
            ffi::dex_component_set_modes(self.0.as_ptr(), modes.as_ptr(), modes.len(), out, e)
        })
    }
    pub fn request_json(&self, role: &str, request: &str) -> Result<String> {
        let role = string(role)?;
        let request = string(request)?;
        text(|out, e| unsafe {
            ffi::dex_component_request_json(
                self.0.as_ptr(),
                role.as_ptr(),
                request.as_ptr(),
                out,
                e,
            )
        })
    }
    pub fn set_pid(&self, p: &[f64]) -> Result<String> {
        text(|out, e| unsafe {
            ffi::dex_component_set_pid(self.0.as_ptr(), p.as_ptr(), p.len(), out, e)
        })
    }
    /// An empty selection is rejected by the native API; use release_all_brakes explicitly.
    pub fn release_brake(&self, release: bool, joints: &[usize]) -> Result<String> {
        text(|out, e| unsafe {
            ffi::dex_component_release_brake(
                self.0.as_ptr(),
                release,
                joints.as_ptr(),
                joints.len(),
                out,
                e,
            )
        })
    }
    pub fn release_all_brakes(&self, release: bool) -> Result<String> {
        text(|out, e| unsafe {
            ffi::dex_component_release_all_brakes(self.0.as_ptr(), release, out, e)
        })
    }
    pub fn set_ee_baud_rate(&self, baud: u32) -> Result<String> {
        text(|out, e| unsafe { ffi::dex_component_set_ee_baud_rate(self.0.as_ptr(), baud, out, e) })
    }
    pub fn set_force_torque_sensor(&self, enabled: bool) -> Result<String> {
        text(|out, e| unsafe {
            ffi::dex_component_set_force_torque_sensor(self.0.as_ptr(), enabled, out, e)
        })
    }
    pub fn set_idle_mode(&self, enabled: bool) -> Result<()> {
        call(|e| unsafe { ffi::dex_component_set_idle_mode(self.0.as_ptr(), enabled, e) })
    }
    pub fn get_idle_mode(&self) -> Result<Option<bool>> {
        let mut v = -1;
        call(|e| unsafe { ffi::dex_component_idle_mode(self.0.as_ptr(), &mut v, e) })?;
        Ok(match v {
            0 => Some(false),
            1 => Some(true),
            _ => None,
        })
    }
    pub fn ee_pass_through_enabled(&self) -> Result<bool> {
        let mut v = false;
        call(|_| unsafe { ffi::dex_component_ee_pass_through_enabled(self.0.as_ptr(), &mut v) })?;
        Ok(v)
    }
    pub fn ee_pass_through_send(&self, data: &[u8]) -> Result<()> {
        call(|e| unsafe {
            ffi::dex_component_ee_pass_through_send(self.0.as_ptr(), data.as_ptr(), data.len(), e)
        })
    }
    pub fn ee_pass_through_latest(&self) -> Result<Option<Vec<u8>>> {
        let mut present = false;
        let data = buffer(|out, e| unsafe {
            ffi::dex_component_ee_pass_through_latest(self.0.as_ptr(), out, &mut present, e)
        })?;
        Ok(present.then_some(data))
    }
    pub fn get_button_state(&self) -> Result<Option<(bool, bool)>> {
        let (mut blue, mut green, mut present) = (false, false, false);
        call(|e| unsafe {
            ffi::dex_component_button_state(self.0.as_ptr(), &mut blue, &mut green, &mut present, e)
        })?;
        Ok(present.then_some((blue, green)))
    }
    pub fn get_touch_data(&self) -> Result<Option<Vec<f64>>> {
        let mut present = false;
        let data = array(|v, n, c, e| unsafe {
            ffi::dex_component_touch_forces(self.0.as_ptr(), v, n, c, &mut present, e)
        })?;
        Ok(present.then_some(data))
    }
    pub fn grasp_torque(&self) -> Result<Option<f64>> {
        let mut v = 0.;
        let mut has = 0;
        call(|e| unsafe { ffi::dex_component_grasp_torque(self.0.as_ptr(), &mut v, &mut has, e) })?;
        Ok((has != 0).then_some(v))
    }
    pub fn set_grasp_torque(&self, value: f64, force: bool) -> Result<f64> {
        let mut v = 0.;
        call(|e| unsafe {
            ffi::dex_component_set_grasp_torque(self.0.as_ptr(), value, force.into(), &mut v, e)
        })?;
        Ok(v)
    }
    pub fn get_pid(&self) -> Result<String> {
        text(|out, e| unsafe { ffi::dex_component_get_pid(self.0.as_ptr(), out, e) })
    }
    pub fn get_brake_status(&self) -> Result<String> {
        text(|out, e| unsafe { ffi::dex_component_brake_status(self.0.as_ptr(), out, e) })
    }
    pub fn get_force_torque_sensor_mode(&self) -> Result<String> {
        text(|out, e| unsafe {
            ffi::dex_component_force_torque_sensor_mode(self.0.as_ptr(), out, e)
        })
    }
    pub fn get_ee_baud_rate(&self) -> Result<String> {
        text(|out, e| unsafe { ffi::dex_component_ee_baud_rate(self.0.as_ptr(), out, e) })
    }
}

pub(crate) fn array<T: Default + Clone>(
    mut f: impl FnMut(*mut T, usize, *mut usize, *mut ffi::dex_error_t) -> i32,
) -> Result<Vec<T>> {
    let mut values = Vec::new();
    for _ in 0..8 {
        let mut count = 0;
        let result = call(|e| {
            f(
                if values.is_empty() {
                    ptr::null_mut()
                } else {
                    values.as_mut_ptr()
                },
                values.len(),
                &mut count,
                e,
            )
        });
        match result {
            Ok(()) if count <= values.len() => {
                values.truncate(count);
                return Ok(values);
            }
            Err(e) if e.code == ffi::DEX_INVALID_ARGUMENT && count > values.len() => {
                if count > 16 * 1024 * 1024 {
                    return Err(invalid("Native array exceeds allocation limit"));
                }
                values
                    .try_reserve(count - values.len())
                    .map_err(|_| invalid("Cannot allocate native array"))?;
                values.resize(count, T::default());
            }
            Err(e) => return Err(e),
            _ => return Err(invalid("Invalid native array size")),
        }
    }
    Err(invalid("Native array size did not stabilize"))
}
pub(crate) fn buffer(
    f: impl FnOnce(*mut ffi::dex_buffer_t, *mut ffi::dex_error_t) -> i32,
) -> Result<Vec<u8>> {
    struct Owned(ffi::dex_buffer_t);
    impl Drop for Owned {
        fn drop(&mut self) {
            unsafe { ffi::dex_buffer_free(&mut self.0) }
        }
    }
    let mut value = Owned(ffi::dex_buffer_t {
        data: ptr::null_mut(),
        len: 0,
    });
    call(|e| f(&mut value.0, e))?;
    if value.0.len == 0 {
        return Ok(Vec::new());
    }
    if value.0.data.is_null() || value.0.len > isize::MAX as usize {
        return Err(invalid("Invalid native buffer"));
    }
    // SAFETY: Successful C call returns an owned buffer valid until its guard drops.
    Ok(unsafe { std::slice::from_raw_parts(value.0.data, value.0.len) }.to_vec())
}
pub(crate) fn rectangular(rows: &[Vec<f64>]) -> Result<usize> {
    let joints = rows.first().map_or(0, Vec::len);
    if joints == 0 || rows.iter().any(|r| r.len() != joints) {
        return Err(invalid(
            "Trajectory must contain nonempty, equally sized joint rows",
        ));
    }
    rows.len()
        .checked_mul(joints)
        .ok_or_else(|| invalid("Trajectory dimensions overflow"))?;
    Ok(joints)
}
