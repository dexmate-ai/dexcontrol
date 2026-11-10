// Copyright (C) 2026 Dexmate Inc.
// SPDX-License-Identifier: Apache-2.0
//! Sensor snapshots own their native storage; borrowed channels cannot outlive them.
use crate::*;
pub use ffi::{
    dex_frame_info_t as FrameInfo, dex_imu_t as Imu, dex_lidar_2d_info_t as Lidar2dInfo,
    dex_lidar_3d_info_t as Lidar3dInfo, dex_stream_stats_t as StreamStats,
    dex_ultrasonic_t as Ultrasonic,
};
#[derive(Debug)]
pub struct Sample<T> {
    pub value: T,
    pub age: Duration,
}
#[derive(Debug)]
pub struct Wrench {
    pub values: [f64; 6],
    pub timestamp_ns: u64,
}
/// Where a sensor read comes from: a control connection or the read-only
/// diagnostic client. Both expose the same [`Sensors`] view.
#[derive(Clone, Copy)]
pub(crate) enum SensorSource<'a> {
    Robot(&'a Robot),
    Diagnostic(&'a DiagnosticClient),
}
/// The native handle a sensor read dispatches on.
pub(crate) enum Handle {
    Robot(*mut ffi::dex_robot_t),
    Diagnostic(*mut ffi::dex_diagnostic_t),
}
impl SensorSource<'_> {
    pub(crate) fn call(&self, read: impl FnOnce(Handle) -> i32) -> i32 {
        match self {
            Self::Robot(robot) => read(Handle::Robot(robot.0.as_ptr())),
            Self::Diagnostic(client) => read(Handle::Diagnostic(client.0.as_ptr())),
        }
    }
}
/// The typed sensors of a [`Robot`] or a [`DiagnosticClient`]:
/// `robot.sensors().imu("head_camera", None)`. Sensors are keyed by sensor
/// name (cameras, IMUs, lidars, ultrasonics) or by the component they are
/// mounted on (wrench, temperatures). A view borrowing the connection.
#[derive(Clone, Copy)]
pub struct Sensors<'a> {
    source: SensorSource<'a>,
}
impl Robot {
    /// The typed sensor view; the same object Python exposes as `robot.sensors`.
    pub fn sensors(&self) -> Sensors<'_> {
        Sensors {
            source: SensorSource::Robot(self),
        }
    }
    /// Deprecated: use [`Robot::sensors`] and [`Sensors::battery`].
    #[deprecated(since = "0.8.1", note = "use sensors().battery(max_age)")]
    pub fn battery_with_age(&self, max_age: Option<Duration>) -> Result<Sample<Battery>> {
        self.sensors().battery(max_age)
    }
    /// Deprecated: use [`Robot::sensors`] and [`Sensors::wrench`].
    #[deprecated(since = "0.8.1", note = "use sensors().wrench(name, max_age)")]
    pub fn wrench(&self, name: &str, max_age: Option<Duration>) -> Result<Sample<Wrench>> {
        self.sensors().wrench(name, max_age)
    }
    /// Deprecated: use [`Robot::sensors`] and [`Sensors::temperature_json`].
    #[deprecated(since = "0.8.1", note = "use sensors().temperature_json(name)")]
    pub fn temperature_json(&self, name: &str) -> Result<String> {
        self.sensors().temperature_json(name)
    }
    /// Deprecated: use [`Robot::sensors`] and [`Sensors::sensor_json`].
    #[deprecated(since = "0.8.1", note = "use sensors().sensor_json(name, role)")]
    pub fn sensor_json(&self, name: &str, role: Option<&str>) -> Result<String> {
        self.sensors().sensor_json(name, role)
    }
    /// Deprecated: use [`Robot::sensors`] and [`Sensors::camera`].
    #[deprecated(since = "0.8.1", note = "use sensors().camera(name)")]
    pub fn camera(&self, name: &str) -> Result<Camera<'_>> {
        self.sensors().camera(name)
    }
    /// Deprecated: use [`Robot::sensors`] and [`Sensors::imu`].
    #[deprecated(since = "0.8.1", note = "use sensors().imu(name, max_age)")]
    pub fn imu(&self, name: &str, max_age: Option<Duration>) -> Result<Sample<Imu>> {
        self.sensors().imu(name, max_age)
    }
    /// Deprecated: use [`Robot::sensors`] and [`Sensors::ultrasonic`].
    #[deprecated(since = "0.8.1", note = "use sensors().ultrasonic(name, max_age)")]
    pub fn ultrasonic(&self, name: &str, max_age: Option<Duration>) -> Result<Sample<Ultrasonic>> {
        self.sensors().ultrasonic(name, max_age)
    }
    /// Deprecated: use [`Robot::sensors`] and [`Sensors::lidar_2d`].
    #[deprecated(since = "0.8.1", note = "use sensors().lidar_2d(name, max_age)")]
    pub fn lidar_2d(&self, name: &str, max_age: Option<Duration>) -> Result<Lidar2d> {
        self.sensors().lidar_2d(name, max_age)
    }
    /// Deprecated: use [`Robot::sensors`] and [`Sensors::lidar_3d`].
    #[deprecated(since = "0.8.1", note = "use sensors().lidar_3d(name, max_age)")]
    pub fn lidar_3d(&self, name: &str, max_age: Option<Duration>) -> Result<Lidar3d> {
        self.sensors().lidar_3d(name, max_age)
    }
}
impl DiagnosticClient {
    /// The same typed sensor view a [`Robot`] offers, read-only.
    pub fn sensors(&self) -> Sensors<'_> {
        Sensors {
            source: SensorSource::Diagnostic(self),
        }
    }
    /// The battery state through the read-only client.
    pub fn battery(&self, max_age: Option<Duration>) -> Result<Sample<Battery>> {
        self.sensors().battery(max_age)
    }
}
impl<'a> Sensors<'a> {
    /// The battery state; `None` accepts any sample age, `Some` is a bound.
    pub fn battery(&self, max_age: Option<Duration>) -> Result<Sample<Battery>> {
        let mut value = Battery {
            struct_size: std::mem::size_of::<Battery>() as u32,
            ..Default::default()
        };
        let mut age = 0;
        let bound = sensor_age(max_age)?;
        call(|e| unsafe {
            self.source.call(|h| match h {
                Handle::Robot(r) => ffi::dex_robot_battery_ex(r, bound, &mut value, &mut age, e),
                Handle::Diagnostic(c) => {
                    ffi::dex_diagnostic_battery_ex(c, bound, &mut value, &mut age, e)
                }
            })
        })?;
        Ok(Sample {
            value,
            age: Duration::from_millis(age),
        })
    }
    pub fn wrench(&self, name: &str, max_age: Option<Duration>) -> Result<Sample<Wrench>> {
        let name = string(name)?;
        let bound = sensor_age(max_age)?;
        let mut values = [0.; 6];
        let mut time = 0;
        let mut age = 0;
        call(|e| unsafe {
            self.source.call(|h| match h {
                Handle::Robot(r) => ffi::dex_robot_wrench_ex(
                    r,
                    name.as_ptr(),
                    bound,
                    values.as_mut_ptr(),
                    &mut time,
                    &mut age,
                    e,
                ),
                Handle::Diagnostic(c) => ffi::dex_diagnostic_wrench_ex(
                    c,
                    name.as_ptr(),
                    bound,
                    values.as_mut_ptr(),
                    &mut time,
                    &mut age,
                    e,
                ),
            })
        })?;
        Ok(Sample {
            value: Wrench {
                values,
                timestamp_ns: time,
            },
            age: Duration::from_millis(age),
        })
    }
    pub fn temperature_json(&self, name: &str) -> Result<String> {
        let name = string(name)?;
        text(|out, e| unsafe {
            self.source.call(|h| match h {
                Handle::Robot(r) => ffi::dex_robot_temperature_json(r, name.as_ptr(), out, e),
                Handle::Diagnostic(c) => {
                    ffi::dex_diagnostic_temperature_json(c, name.as_ptr(), out, e)
                }
            })
        })
    }
    pub fn sensor_json(&self, name: &str, role: Option<&str>) -> Result<String> {
        let name = string(name)?;
        let role = role.map(string).transpose()?;
        text(|out, e| unsafe {
            self.source.call(|h| match h {
                Handle::Robot(r) => ffi::dex_robot_sensor_json(
                    r,
                    name.as_ptr(),
                    role.as_ref().map_or(ptr::null(), |s| s.as_ptr()),
                    out,
                    e,
                ),
                Handle::Diagnostic(c) => ffi::dex_diagnostic_sensor_json(
                    c,
                    name.as_ptr(),
                    role.as_ref().map_or(ptr::null(), |s| s.as_ptr()),
                    out,
                    e,
                ),
            })
        })
    }
    pub fn camera(&self, name: &str) -> Result<Camera<'a>> {
        Ok(Camera {
            source: self.source,
            name: string(name)?,
        })
    }
    pub fn imu(&self, name: &str, max_age: Option<Duration>) -> Result<Sample<Imu>> {
        let name = string(name)?;
        let bound = sensor_age(max_age)?;
        let mut value = Imu {
            struct_size: std::mem::size_of::<Imu>() as u32,
            ..Default::default()
        };
        let mut age = 0;
        call(|e| unsafe {
            self.source.call(|h| match h {
                Handle::Robot(r) => {
                    ffi::dex_robot_imu_ex(r, name.as_ptr(), bound, &mut value, &mut age, e)
                }
                Handle::Diagnostic(c) => {
                    ffi::dex_diagnostic_imu_ex(c, name.as_ptr(), bound, &mut value, &mut age, e)
                }
            })
        })?;
        Ok(Sample {
            value,
            age: Duration::from_millis(age),
        })
    }
    pub fn ultrasonic(&self, name: &str, max_age: Option<Duration>) -> Result<Sample<Ultrasonic>> {
        let name = string(name)?;
        let bound = sensor_age(max_age)?;
        let mut value = Ultrasonic {
            struct_size: std::mem::size_of::<Ultrasonic>() as u32,
            ..Default::default()
        };
        let mut age = 0;
        call(|e| unsafe {
            self.source.call(|h| match h {
                Handle::Robot(r) => {
                    ffi::dex_robot_ultrasonic_ex(r, name.as_ptr(), bound, &mut value, &mut age, e)
                }
                Handle::Diagnostic(c) => ffi::dex_diagnostic_ultrasonic_ex(
                    c,
                    name.as_ptr(),
                    bound,
                    &mut value,
                    &mut age,
                    e,
                ),
            })
        })?;
        Ok(Sample {
            value,
            age: Duration::from_millis(age),
        })
    }
    pub fn lidar_2d(&self, name: &str, max_age: Option<Duration>) -> Result<Lidar2d> {
        let name = string(name)?;
        let bound = sensor_age(max_age)?;
        let mut out = ptr::null_mut();
        let result = call(|e| unsafe {
            self.source.call(|h| match h {
                Handle::Robot(r) => {
                    ffi::dex_robot_lidar_2d_take(r, name.as_ptr(), bound, &mut out, e)
                }
                Handle::Diagnostic(c) => {
                    ffi::dex_diagnostic_lidar_2d_take(c, name.as_ptr(), bound, &mut out, e)
                }
            })
        });
        let handle = NonNull::new(out).map(Lidar2d);
        result?;
        handle.ok_or_else(|| invalid("Runtime returned no sensor snapshot"))
    }
    pub fn lidar_3d(&self, name: &str, max_age: Option<Duration>) -> Result<Lidar3d> {
        let name = string(name)?;
        let bound = sensor_age(max_age)?;
        let mut out = ptr::null_mut();
        let result = call(|e| unsafe {
            self.source.call(|h| match h {
                Handle::Robot(r) => {
                    ffi::dex_robot_lidar_3d_take(r, name.as_ptr(), bound, &mut out, e)
                }
                Handle::Diagnostic(c) => {
                    ffi::dex_diagnostic_lidar_3d_take(c, name.as_ptr(), bound, &mut out, e)
                }
            })
        });
        let handle = NonNull::new(out).map(Lidar3d);
        result?;
        handle.ok_or_else(|| invalid("Runtime returned no sensor snapshot"))
    }
}
/// An owning immutable scan. Borrowed channels stay valid until the scan is dropped.
/// ```compile_fail
/// fn ranges(scan: dexcontrol::Lidar2d) -> &'static [f64] {
///     scan.ranges().unwrap()
/// }
/// ```
pub struct Lidar2d(NonNull<ffi::dex_lidar_2d_t>);
impl Lidar2d {
    pub fn info(&self) -> Result<Lidar2dInfo> {
        let mut value = Lidar2dInfo {
            struct_size: std::mem::size_of::<Lidar2dInfo>() as u32,
            ..Default::default()
        };
        call(|e| unsafe { ffi::dex_lidar_2d_info(self.0.as_ptr(), &mut value, e) })?;
        Ok(value)
    }
    pub fn age(&self) -> Result<Duration> {
        let mut value = 0;
        call(|_| unsafe { ffi::dex_lidar_2d_age_ms(self.0.as_ptr(), &mut value) })?;
        Ok(Duration::from_millis(value))
    }
    pub fn ranges(&self) -> Result<&[f64]> {
        let mut data = ptr::null();
        let mut count = 0;
        call(|_| unsafe {
            ffi::dex_lidar_2d_channel_f64(self.0.as_ptr(), 0, &mut data, &mut count)
        })?;
        // SAFETY: the immutable snapshot owns this channel; the result borrows self.
        unsafe { borrowed(data, count) }
    }
    pub fn angles(&self) -> Result<&[f64]> {
        let mut data = ptr::null();
        let mut count = 0;
        call(|_| unsafe {
            ffi::dex_lidar_2d_channel_f64(self.0.as_ptr(), 1, &mut data, &mut count)
        })?;
        // SAFETY: the immutable snapshot owns this channel; the result borrows self.
        unsafe { borrowed(data, count) }
    }
    pub fn intensities(&self) -> Result<&[u32]> {
        let mut data = ptr::null();
        let mut count = 0;
        call(|_| unsafe {
            ffi::dex_lidar_2d_channel_u32(self.0.as_ptr(), 0, &mut data, &mut count)
        })?;
        // SAFETY: the immutable snapshot owns this channel; the result borrows self.
        unsafe { borrowed(data, count) }
    }
}
impl Drop for Lidar2d {
    fn drop(&mut self) {
        unsafe { ffi::dex_lidar_2d_release(self.0.as_ptr()) }
    }
}
// SAFETY: snapshot data is immutable; its owning handle is kept alive by Rust borrows.
unsafe impl Send for Lidar2d {}
unsafe impl Sync for Lidar2d {}
pub struct Lidar3d(NonNull<ffi::dex_lidar_3d_t>);
impl Lidar3d {
    pub fn info(&self) -> Result<Lidar3dInfo> {
        let mut value = Lidar3dInfo {
            struct_size: std::mem::size_of::<Lidar3dInfo>() as u32,
            ..Default::default()
        };
        call(|e| unsafe { ffi::dex_lidar_3d_info(self.0.as_ptr(), &mut value, e) })?;
        Ok(value)
    }
    pub fn age(&self) -> Result<Duration> {
        let mut value = 0;
        call(|_| unsafe { ffi::dex_lidar_3d_age_ms(self.0.as_ptr(), &mut value) })?;
        Ok(Duration::from_millis(value))
    }
    pub fn x(&self) -> Result<&[f64]> {
        let mut data = ptr::null();
        let mut count = 0;
        call(|_| unsafe {
            ffi::dex_lidar_3d_channel_f64(self.0.as_ptr(), 0, &mut data, &mut count)
        })?;
        // SAFETY: the immutable snapshot owns this channel; the result borrows self.
        unsafe { borrowed(data, count) }
    }
    pub fn y(&self) -> Result<&[f64]> {
        let mut data = ptr::null();
        let mut count = 0;
        call(|_| unsafe {
            ffi::dex_lidar_3d_channel_f64(self.0.as_ptr(), 1, &mut data, &mut count)
        })?;
        // SAFETY: the immutable snapshot owns this channel; the result borrows self.
        unsafe { borrowed(data, count) }
    }
    pub fn z(&self) -> Result<&[f64]> {
        let mut data = ptr::null();
        let mut count = 0;
        call(|_| unsafe {
            ffi::dex_lidar_3d_channel_f64(self.0.as_ptr(), 2, &mut data, &mut count)
        })?;
        // SAFETY: the immutable snapshot owns this channel; the result borrows self.
        unsafe { borrowed(data, count) }
    }
    pub fn intensity(&self) -> Result<&[u32]> {
        let mut data = ptr::null();
        let mut count = 0;
        call(|_| unsafe {
            ffi::dex_lidar_3d_channel_u32(self.0.as_ptr(), 0, &mut data, &mut count)
        })?;
        // SAFETY: the immutable snapshot owns this channel; the result borrows self.
        unsafe { borrowed(data, count) }
    }
    pub fn ring(&self) -> Result<&[u32]> {
        let mut data = ptr::null();
        let mut count = 0;
        call(|_| unsafe {
            ffi::dex_lidar_3d_channel_u32(self.0.as_ptr(), 1, &mut data, &mut count)
        })?;
        // SAFETY: the immutable snapshot owns this channel; the result borrows self.
        unsafe { borrowed(data, count) }
    }
    pub fn point_timestamps_ns(&self) -> Result<&[u32]> {
        let mut data = ptr::null();
        let mut count = 0;
        call(|_| unsafe {
            ffi::dex_lidar_3d_channel_u32(self.0.as_ptr(), 2, &mut data, &mut count)
        })?;
        // SAFETY: the immutable snapshot owns this channel; the result borrows self.
        unsafe { borrowed(data, count) }
    }
}
impl Drop for Lidar3d {
    fn drop(&mut self) {
        unsafe { ffi::dex_lidar_3d_release(self.0.as_ptr()) }
    }
}
// SAFETY: snapshot data is immutable; its owning handle is kept alive by Rust borrows.
unsafe impl Send for Lidar3d {}
unsafe impl Sync for Lidar3d {}

fn sensor_age(age: Option<Duration>) -> Result<u64> {
    age.map(millis).transpose().map(|v| v.unwrap_or(0))
}
/// Caller must tie the returned slice's lifetime to the native buffer's owner.
unsafe fn borrowed<'a, T>(data: *const T, count: usize) -> Result<&'a [T]> {
    if count == 0 {
        return Ok(&[]);
    }
    if data.is_null() || count > isize::MAX as usize / std::mem::size_of::<T>() {
        return Err(invalid("Invalid native channel"));
    }
    // SAFETY: caller guarantees initialized storage valid for the owner's borrow.
    Ok(unsafe { std::slice::from_raw_parts(data, count) })
}
pub struct Camera<'a> {
    source: SensorSource<'a>,
    name: CString,
}
impl Camera<'_> {
    /// Query the server JSON object without subscribing to images.
    /// Fields are device-dependent; each call queries fresh information.
    pub fn info_json(&self, timeout: Duration) -> Result<String> {
        let timeout = millis(timeout)?;
        text(|out, e| unsafe {
            self.source.call(|h| match h {
                Handle::Robot(r) => {
                    ffi::dex_robot_camera_info_json(r, self.name.as_ptr(), timeout, out, e)
                }
                Handle::Diagnostic(c) => {
                    ffi::dex_diagnostic_camera_info_json(c, self.name.as_ptr(), timeout, out, e)
                }
            })
        })
    }
    pub fn streams(&self) -> Result<Vec<String>> {
        let mut n = 0;
        call(|e| unsafe {
            self.source.call(|h| match h {
                Handle::Robot(r) => {
                    ffi::dex_robot_camera_stream_count(r, self.name.as_ptr(), &mut n, e)
                }
                Handle::Diagnostic(c) => {
                    ffi::dex_diagnostic_camera_stream_count(c, self.name.as_ptr(), &mut n, e)
                }
            })
        })?;
        (0..n)
            .map(|i| {
                text(|out, e| unsafe {
                    self.source.call(|h| match h {
                        Handle::Robot(r) => {
                            ffi::dex_robot_camera_stream_name(r, self.name.as_ptr(), i, out, e)
                        }
                        Handle::Diagnostic(c) => {
                            ffi::dex_diagnostic_camera_stream_name(c, self.name.as_ptr(), i, out, e)
                        }
                    })
                })
            })
            .collect()
    }
    /// history=0 keeps only the latest frame; positive values retain bounded history.
    pub fn subscribe(&self, stream: &str, history: usize) -> Result<()> {
        let stream = string(stream)?;
        call(|e| unsafe {
            self.source.call(|h| match h {
                Handle::Robot(r) => ffi::dex_robot_camera_subscribe(
                    r,
                    self.name.as_ptr(),
                    stream.as_ptr(),
                    history,
                    e,
                ),
                Handle::Diagnostic(c) => ffi::dex_diagnostic_camera_subscribe(
                    c,
                    self.name.as_ptr(),
                    stream.as_ptr(),
                    history,
                    e,
                ),
            })
        })
    }
    pub fn unsubscribe(&self, stream: &str) -> Result<()> {
        let stream = string(stream)?;
        call(|e| unsafe {
            self.source.call(|h| match h {
                Handle::Robot(r) => {
                    ffi::dex_robot_camera_unsubscribe(r, self.name.as_ptr(), stream.as_ptr(), e)
                }
                Handle::Diagnostic(c) => ffi::dex_diagnostic_camera_unsubscribe(
                    c,
                    self.name.as_ptr(),
                    stream.as_ptr(),
                    e,
                ),
            })
        })
    }
    pub fn latest_frame(&self, stream: &str) -> Result<Option<Frame>> {
        let stream = string(stream)?;
        let mut out = ptr::null_mut();
        let result = call(|e| unsafe {
            self.source.call(|h| match h {
                Handle::Robot(r) => ffi::dex_robot_camera_latest_frame(
                    r,
                    self.name.as_ptr(),
                    stream.as_ptr(),
                    &mut out,
                    e,
                ),
                Handle::Diagnostic(c) => ffi::dex_diagnostic_camera_latest_frame(
                    c,
                    self.name.as_ptr(),
                    stream.as_ptr(),
                    &mut out,
                    e,
                ),
            })
        });
        let frame = NonNull::new(out).map(Frame);
        result?;
        Ok(frame)
    }
    pub fn stats(&self, stream: &str) -> Result<StreamStats> {
        let stream = string(stream)?;
        let mut out = StreamStats {
            struct_size: std::mem::size_of::<StreamStats>() as u32,
            ..Default::default()
        };
        call(|e| unsafe {
            self.source.call(|h| match h {
                Handle::Robot(r) => {
                    ffi::dex_robot_camera_stats(r, self.name.as_ptr(), stream.as_ptr(), &mut out, e)
                }
                Handle::Diagnostic(c) => ffi::dex_diagnostic_camera_stats(
                    c,
                    self.name.as_ptr(),
                    stream.as_ptr(),
                    &mut out,
                    e,
                ),
            })
        })?;
        Ok(out)
    }
    pub fn is_active(&self, stream: Option<&str>, window: Duration) -> Result<bool> {
        let stream = stream.map(string).transpose()?;
        let window = duration_ms(window)?;
        let mut active = false;
        call(|e| unsafe {
            self.source.call(|h| match h {
                Handle::Robot(r) => ffi::dex_robot_camera_is_active(
                    r,
                    self.name.as_ptr(),
                    stream.as_ref().map_or(ptr::null(), |s| s.as_ptr()),
                    window,
                    &mut active,
                    e,
                ),
                Handle::Diagnostic(c) => ffi::dex_diagnostic_camera_is_active(
                    c,
                    self.name.as_ptr(),
                    stream.as_ref().map_or(ptr::null(), |s| s.as_ptr()),
                    window,
                    &mut active,
                    e,
                ),
            })
        })?;
        Ok(active)
    }
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrameEncoding {
    Rgb8,
    Bgr8,
    Gray8,
    Depth32F,
    Unknown(u32),
}
/// An owning immutable image. Pixel slices cannot outlive this frame.
/// ```compile_fail
/// fn pixels(frame: dexcontrol::Frame) -> &'static [u8] {
///     frame.data().unwrap()
/// }
/// ```
pub struct Frame(NonNull<ffi::dex_frame_t>);
impl Frame {
    pub fn info(&self) -> Result<FrameInfo> {
        let mut out = FrameInfo {
            struct_size: std::mem::size_of::<FrameInfo>() as u32,
            ..Default::default()
        };
        call(|e| unsafe { ffi::dex_frame_info(self.0.as_ptr(), &mut out, e) })?;
        Ok(out)
    }
    pub fn encoding(&self) -> Result<FrameEncoding> {
        Ok(match self.info()?.encoding {
            0 => FrameEncoding::Rgb8,
            1 => FrameEncoding::Bgr8,
            2 => FrameEncoding::Gray8,
            3 => FrameEncoding::Depth32F,
            n => FrameEncoding::Unknown(n),
        })
    }
    pub fn data(&self) -> Result<&[u8]> {
        let mut out = ptr::null();
        let mut n = 0;
        call(|_| unsafe { ffi::dex_frame_data(self.0.as_ptr(), &mut out, &mut n) })?;
        // SAFETY: the frame owns immutable pixels; the returned slice borrows self.
        unsafe { borrowed(out, n) }
    }
}
impl Drop for Frame {
    fn drop(&mut self) {
        unsafe { ffi::dex_frame_release(self.0.as_ptr()) }
    }
}
// SAFETY: frames are immutable native snapshots with ownership tied to Rust borrows.
unsafe impl Send for Frame {}
unsafe impl Sync for Frame {}
