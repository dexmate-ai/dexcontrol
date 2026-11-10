// Copyright (C) 2026 Dexmate Inc.
// SPDX-License-Identifier: Apache-2.0
//! Multi-component commands and owned motion groups.
use crate::*;
use std::collections::BTreeMap;
use std::ffi::c_char;
#[derive(Debug, Clone)]
pub struct TrajectoryTrack {
    pub positions: Vec<Vec<f64>>,
    pub velocities: Option<Vec<Vec<f64>>>,
}
/// The parallel C arrays of a `{component: values}` fan-out. The `CString`s
/// own the bytes `names` points into and `positions` borrows `targets`, so a
/// value lives exactly as long as the call it is built for.
struct FanOut<'a> {
    _strings: Vec<CString>,
    names: Vec<*const c_char>,
    positions: Vec<*const f64>,
    counts: Vec<usize>,
    _targets: std::marker::PhantomData<&'a BTreeMap<String, Vec<f64>>>,
}
impl<'a> FanOut<'a> {
    fn new(targets: &'a BTreeMap<String, Vec<f64>>) -> Result<Self> {
        let strings: Vec<CString> = targets.keys().map(|s| string(s)).collect::<Result<_>>()?;
        Ok(Self {
            names: strings.iter().map(|s| s.as_ptr()).collect(),
            positions: targets.values().map(|v| v.as_ptr()).collect(),
            counts: targets.values().map(Vec::len).collect(),
            _strings: strings,
            _targets: std::marker::PhantomData,
        })
    }
    fn len(&self) -> usize {
        self.names.len()
    }
}
impl Robot {
    pub fn set_joint_pos(
        &self,
        targets: &BTreeMap<String, Vec<f64>>,
        options: CommandOptions,
    ) -> Result<()> {
        let fan_out = FanOut::new(targets)?;
        let options = options.native()?;
        call(|e| unsafe {
            ffi::dex_robot_set_joint_pos_multi(
                self.0.as_ptr(),
                fan_out.names.as_ptr(),
                fan_out.positions.as_ptr(),
                fan_out.counts.as_ptr(),
                fan_out.len(),
                &options,
                e,
            )
        })
    }
    /// Starts one tracked motion per component. A partial delivery failure
    /// is an error whose `MotionStartError` carries the group of motions that
    /// did start, so they can be cancelled or waited on; `?` into [`Error`]
    /// drops the group.
    pub fn move_to_joint_pos(
        &self,
        targets: &BTreeMap<String, Vec<f64>>,
        options: MotionOptions,
    ) -> std::result::Result<MotionGroup, MotionStartError<MotionGroup>> {
        let fan_out = FanOut::new(targets)?;
        let options = options.native()?;
        MotionGroup::start_recoverable(|out, e| unsafe {
            ffi::dex_robot_move_to_joint_pos_multi(
                self.0.as_ptr(),
                fan_out.names.as_ptr(),
                fan_out.positions.as_ptr(),
                fan_out.counts.as_ptr(),
                fan_out.len(),
                &options,
                out,
                e,
            )
        })
    }
    /// Streamed-goal fan-out: one untracked goal per component, validated
    /// all-or-none before anything is published. See
    /// `JointComponent::stream_joint_pos`.
    pub fn stream_joint_pos(
        &self,
        targets: &BTreeMap<String, Vec<f64>>,
        options: MotionOptions,
    ) -> Result<()> {
        let fan_out = FanOut::new(targets)?;
        let options = options.native()?;
        call(|e| unsafe {
            ffi::dex_robot_stream_joint_pos_multi(
                self.0.as_ptr(),
                fan_out.names.as_ptr(),
                fan_out.positions.as_ptr(),
                fan_out.counts.as_ptr(),
                fan_out.len(),
                &options,
                e,
            )
        })
    }
    /// Deprecated alias of [`Robot::set_joint_pos`].
    #[deprecated(since = "0.8.1", note = "use set_joint_pos")]
    pub fn set_joint_positions(
        &self,
        targets: &BTreeMap<String, Vec<f64>>,
        options: CommandOptions,
    ) -> Result<()> {
        self.set_joint_pos(targets, options)
    }
    /// Deprecated alias of [`Robot::move_to_joint_pos`].
    #[deprecated(since = "0.8.1", note = "use move_to_joint_pos")]
    pub fn move_to_joint_positions(
        &self,
        targets: &BTreeMap<String, Vec<f64>>,
        options: MotionOptions,
    ) -> Result<MotionGroup> {
        self.move_to_joint_pos(targets, options)
            .map_err(Error::from)
    }
    /// Deprecated: [`Robot::move_to_joint_pos`] now returns the same
    /// `MotionStartError`, with the group when the command may have reached the robot.
    #[deprecated(since = "0.8.1", note = "use move_to_joint_pos")]
    pub fn move_to_joint_pos_recoverable(
        &self,
        targets: &BTreeMap<String, Vec<f64>>,
        options: MotionOptions,
    ) -> std::result::Result<MotionGroup, MotionStartError<MotionGroup>> {
        self.move_to_joint_pos(targets, options)
    }
    /// Deprecated alias of [`Robot::move_to_joint_pos_recoverable`].
    #[deprecated(since = "0.8.1", note = "use move_to_joint_pos_recoverable")]
    pub fn move_to_joint_positions_recoverable(
        &self,
        targets: &BTreeMap<String, Vec<f64>>,
        options: MotionOptions,
    ) -> std::result::Result<MotionGroup, MotionStartError<MotionGroup>> {
        self.move_to_joint_pos(targets, options)
    }
    /// Deprecated alias of [`Robot::stream_joint_pos`].
    #[deprecated(since = "0.8.1", note = "use stream_joint_pos")]
    pub fn stream_joint_positions(
        &self,
        targets: &BTreeMap<String, Vec<f64>>,
        options: MotionOptions,
    ) -> Result<()> {
        self.stream_joint_pos(targets, options)
    }
    /// Best-effort submission with a per-component JSON report. A successful return
    /// means the report is available, not that every command was delivered.
    /// Inspect `all_delivered` in the report before proceeding to later stages.
    pub fn start_joint_motions(
        &self,
        targets: &BTreeMap<String, Vec<f64>>,
        options: MotionOptions,
    ) -> Result<MotionSubmission> {
        let fan_out = FanOut::new(targets)?;
        let options = options.native()?;
        let mut out = ptr::null_mut();
        let result = text(|json, e| unsafe {
            ffi::dex_robot_start_joint_motions_opt(
                self.0.as_ptr(),
                fan_out.names.as_ptr(),
                fan_out.positions.as_ptr(),
                fan_out.counts.as_ptr(),
                fan_out.len(),
                &options,
                &mut out,
                json,
                e,
            )
        });
        let group = NonNull::new(out).map(MotionGroup);
        let report_json = result?;
        let group = group.ok_or_else(|| invalid("Runtime returned no motion group"))?;
        Ok(MotionSubmission { group, report_json })
    }
    /// Streams the entire recording before returning; validation happens before publication.
    pub fn execute_trajectory(
        &self,
        tracks: &BTreeMap<String, TrajectoryTrack>,
        hz: f64,
        options: TrajectoryOptions,
    ) -> Result<()> {
        if tracks.is_empty() {
            return Err(invalid("At least one trajectory track is required"));
        }
        let ticks = tracks.values().next().unwrap().positions.len();
        let mut counts = Vec::new();
        let mut positions = Vec::new();
        let mut velocities = Vec::new();
        for track in tracks.values() {
            let joints = rectangular(&track.positions)?;
            if track.positions.len() != ticks {
                return Err(invalid(
                    "All trajectory tracks must have the same tick count",
                ));
            }
            if let Some(v) = &track.velocities {
                if rectangular(v)? != joints || v.len() != ticks {
                    return Err(invalid("Trajectory velocity shape differs from positions"));
                }
            }
            counts.push(joints);
            positions.push(
                track
                    .positions
                    .iter()
                    .flatten()
                    .copied()
                    .collect::<Vec<f64>>(),
            );
            velocities.push(
                track
                    .velocities
                    .as_ref()
                    .map(|v| v.iter().flatten().copied().collect::<Vec<f64>>()),
            );
        }
        let strings: Vec<CString> = tracks.keys().map(|s| string(s)).collect::<Result<_>>()?;
        let names: Vec<_> = strings.iter().map(|s| s.as_ptr()).collect();
        let positions: Vec<_> = positions.iter().map(|v| v.as_ptr()).collect();
        let velocities: Vec<_> = velocities
            .iter()
            .map(|v| v.as_ref().map_or(ptr::null(), |v| v.as_ptr()))
            .collect();
        let options = options.native()?;
        call(|e| unsafe {
            ffi::dex_robot_execute_trajectory_opt(
                self.0.as_ptr(),
                names.as_ptr(),
                positions.as_ptr(),
                velocities.as_ptr(),
                ticks,
                counts.as_ptr(),
                names.len(),
                hz,
                &options,
                e,
            )
        })
    }
}
impl MotionState {
    pub(crate) fn from_native(n: u32) -> Self {
        match n {
            0 => Self::Pending,
            1 => Self::Running,
            2 => Self::Succeeded,
            3 => Self::Cancelled,
            4 => Self::Failed,
            5 => Self::Superseded,
            n => Self::Unknown(n),
        }
    }
}
impl MotionHandle {
    pub fn id(&self) -> Result<u64> {
        let mut id = 0;
        call(|_| unsafe { ffi::dex_motion_id(self.0.as_ptr(), &mut id) })?;
        Ok(id)
    }
    pub fn message(&self) -> Result<String> {
        text(|out, e| unsafe { ffi::dex_motion_message(self.0.as_ptr(), out, e) })
    }
    pub fn refresh(&self) -> Result<MotionState> {
        let mut state = 0;
        call(|e| unsafe { ffi::dex_motion_refresh(self.0.as_ptr(), &mut state, e) })?;
        Ok(MotionState::from_native(state))
    }
    pub fn wait_with(&self, options: WaitOptions) -> Result<()> {
        let options = options.native()?;
        call(|e| unsafe { ffi::dex_motion_wait_success(self.0.as_ptr(), &options, e) })
    }
}
#[derive(Debug)]
pub struct MotionGroup(NonNull<ffi::dex_motion_group_t>);
impl MotionGroup {
    fn start_recoverable(
        f: impl FnOnce(*mut *mut ffi::dex_motion_group_t, *mut ffi::dex_error_t) -> i32,
    ) -> std::result::Result<Self, MotionStartError<Self>> {
        let mut out = ptr::null_mut();
        let result = call(|e| f(&mut out, e));
        let handle = NonNull::new(out).map(Self);
        match result {
            Err(error) => Err(MotionStartError { error, handle }),
            Ok(()) => handle.ok_or_else(|| invalid("Runtime returned no motion group").into()),
        }
    }
    /// Explicit success wait; same contract as wait.
    pub fn wait_success(&self, timeout: Duration) -> Result<()> {
        self.wait(timeout)
    }
    pub fn len(&self) -> Result<usize> {
        let mut n = 0;
        call(|_| unsafe { ffi::dex_motion_group_len(self.0.as_ptr(), &mut n) })?;
        Ok(n)
    }
    pub fn is_empty(&self) -> Result<bool> {
        Ok(self.len()? == 0)
    }
    pub fn member(&self, index: usize) -> Result<MotionHandle> {
        MotionHandle::start(|out, e| unsafe {
            ffi::dex_motion_group_member(self.0.as_ptr(), index, out, e)
        })
    }
    pub fn state(&self) -> Result<MotionState> {
        let mut n = 0;
        call(|e| unsafe { ffi::dex_motion_group_state(self.0.as_ptr(), &mut n, e) })?;
        Ok(MotionState::from_native(n))
    }
    /// Wait for a terminal aggregate outcome without treating cancellation as success.
    pub fn wait_terminal(&self, timeout: Duration) -> Result<MotionState> {
        let wait = Wait::Timeout(timeout).native()?;
        let mut state = 0;
        call(|e| unsafe { ffi::dex_motion_group_wait(self.0.as_ptr(), &wait, &mut state, e) })?;
        Ok(MotionState::from_native(state))
    }
    pub fn cancel_and_wait(&self, timeout: Duration) -> Result<MotionState> {
        self.cancel()?;
        self.wait_terminal(timeout)
    }
    pub fn cancel(&self) -> Result<()> {
        call(|e| unsafe { ffi::dex_motion_group_cancel(self.0.as_ptr(), e) })
    }
    /// Requires successful completion of every member, not merely terminal states.
    pub fn wait(&self, timeout: Duration) -> Result<()> {
        self.wait_with(Wait::Timeout(timeout).into())
    }
    /// Wait with explicit tolerance, settling interval and feedback freshness.
    pub fn wait_with(&self, options: WaitOptions) -> Result<()> {
        let wait = options.native()?;
        let mut n = 0;
        call(|e| unsafe { ffi::dex_motion_group_wait(self.0.as_ptr(), &wait, &mut n, e) })?;
        Self::require_success(MotionState::from_native(n))
    }
    fn require_success(state: MotionState) -> Result<()> {
        if state != MotionState::Succeeded {
            return Err(Error {
                code: ffi::DEX_RUNTIME_ERROR,
                message: format!("Motion group did not succeed: {state:?}"),
            });
        }
        Ok(())
    }
}
impl Drop for MotionGroup {
    fn drop(&mut self) {
        unsafe { ffi::dex_motion_group_release(self.0.as_ptr()) }
    }
}
// SAFETY: C ABI group operations synchronize internally; ownership prevents release during calls.
unsafe impl Send for MotionGroup {}
unsafe impl Sync for MotionGroup {}

/// The group contains only successfully submitted motions, in report order.
#[must_use = "Inspect delivery outcomes before waiting or starting another stage"]
pub struct MotionSubmission {
    pub group: MotionGroup,
    pub report_json: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn group_wait_requires_success_not_just_a_terminal_state() {
        assert!(MotionGroup::require_success(MotionState::Succeeded).is_ok());
        for state in [
            MotionState::Pending,
            MotionState::Running,
            MotionState::Failed,
            MotionState::Cancelled,
            MotionState::Superseded,
            MotionState::Unknown(99),
        ] {
            assert!(MotionGroup::require_success(state).is_err());
        }
    }
}
