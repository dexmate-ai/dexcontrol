// Copyright (C) 2026 Dexmate Inc.
// SPDX-License-Identifier: Apache-2.0
//! Read-only diagnostics, runtime contexts and timing utilities.
use crate::*;
pub struct DiagnosticClient(pub(crate) NonNull<ffi::dex_diagnostic_t>);
impl DiagnosticClient {
    pub fn connect(options: ConnectOptions<'_>) -> Result<Self> {
        options.with_native(|n| {
            let mut out = ptr::null_mut();
            let result = call(|e| unsafe { ffi::dex_diagnostic_create(n, &mut out, e) });
            let handle = NonNull::new(out).map(Self);
            result?;
            handle.ok_or_else(|| invalid("Runtime returned no diagnostic client"))
        })
    }
    pub fn snapshot_json(&self, timeout: Duration) -> Result<String> {
        let timeout = duration_ms(timeout)?;
        text(|out, e| unsafe {
            ffi::dex_diagnostic_snapshot_json(self.0.as_ptr(), timeout, out, e)
        })
    }
    pub fn query_ntp_json(&self, samples: usize) -> Result<String> {
        text(|out, e| unsafe {
            ffi::dex_diagnostic_query_ntp_json(self.0.as_ptr(), samples, out, e)
        })
    }
    pub fn close(&self) -> Result<()> {
        call(|e| unsafe { ffi::dex_diagnostic_shutdown(self.0.as_ptr(), e) })
    }
}
impl Drop for DiagnosticClient {
    fn drop(&mut self) {
        unsafe { ffi::dex_diagnostic_release(self.0.as_ptr()) }
    }
}
/// An explicitly sized native runtime shared by connections created through it.
pub struct Context(NonNull<ffi::dex_context_t>);
impl Context {
    pub fn new(workers: usize) -> Result<Self> {
        let mut out = ptr::null_mut();
        let result = call(|e| unsafe { ffi::dex_context_create(workers, &mut out, e) });
        let handle = NonNull::new(out).map(Self);
        result?;
        handle.ok_or_else(|| invalid("Runtime returned no context"))
    }
    pub fn connect(&self, options: ConnectOptions<'_>) -> Result<Robot> {
        options.with_native(|n| {
            let mut out = ptr::null_mut();
            let result =
                call(|e| unsafe { ffi::dex_context_connect(self.0.as_ptr(), n, &mut out, e) });
            let robot = NonNull::new(out).map(Robot);
            result?;
            robot.ok_or_else(|| invalid("Runtime returned no robot"))
        })
    }
}
impl Drop for Context {
    fn drop(&mut self) {
        unsafe { ffi::dex_context_release(self.0.as_ptr()) }
    }
}
pub fn configure_runtime(workers: u32, thread_name: Option<&str>) -> Result<()> {
    let name = thread_name.map(string).transpose()?;
    let mut n = ffi::dex_runtime_options_t::default();
    unsafe { ffi::dex_runtime_options_init(&mut n) };
    n.worker_threads = workers;
    n.thread_name_utf8 = name.as_ref().map_or(ptr::null(), |s| s.as_ptr());
    call(|e| unsafe { ffi::dex_runtime_configure(&n, e) })
}
pub fn configure_watchdog(command: Option<&str>, disabled: bool) -> Result<()> {
    let command = command.map(string).transpose()?;
    call(|e| unsafe {
        ffi::dex_watchdog_configure(
            command.as_ref().map_or(ptr::null(), |s| s.as_ptr()),
            disabled,
            e,
        )
    })
}
pub fn configure_safety(exit_on_termination: bool) -> Result<()> {
    call(|e| unsafe { ffi::dex_safety_configure(exit_on_termination, e) })
}
pub use ffi::dex_rate_limiter_stats_t as RateLimiterStats;
pub struct RateLimiter(NonNull<ffi::dex_rate_limiter_t>);
impl RateLimiter {
    pub fn new(hz: f64, window: usize, adaptive: bool) -> Result<Self> {
        let mut out = ptr::null_mut();
        let result =
            call(|e| unsafe { ffi::dex_rate_limiter_create(hz, window, adaptive, &mut out, e) });
        let handle = NonNull::new(out).map(Self);
        result?;
        handle.ok_or_else(|| invalid("Runtime returned no rate limiter"))
    }
    pub fn sleep(&self) -> Result<()> {
        call(|_| unsafe { ffi::dex_rate_limiter_sleep(self.0.as_ptr()) })
    }
    pub fn reset(&self) -> Result<()> {
        call(|_| unsafe { ffi::dex_rate_limiter_reset(self.0.as_ptr()) })
    }
    pub fn done(&self) -> Result<()> {
        call(|_| unsafe { ffi::dex_rate_limiter_done(self.0.as_ptr()) })
    }
    pub fn stats(&self) -> Result<RateLimiterStats> {
        let mut out = RateLimiterStats {
            struct_size: std::mem::size_of::<RateLimiterStats>() as u32,
            ..Default::default()
        };
        call(|_| unsafe { ffi::dex_rate_limiter_stats(self.0.as_ptr(), &mut out) })?;
        Ok(out)
    }
    pub fn actual_rate(&self) -> Result<f64> {
        let mut v = 0.;
        call(|_| unsafe { ffi::dex_rate_limiter_actual_rate(self.0.as_ptr(), &mut v) })?;
        Ok(v)
    }
    pub fn average_rate(&self) -> Result<f64> {
        let mut v = 0.;
        call(|_| unsafe { ffi::dex_rate_limiter_average_rate(self.0.as_ptr(), &mut v) })?;
        Ok(v)
    }
}
impl Drop for RateLimiter {
    fn drop(&mut self) {
        unsafe { ffi::dex_rate_limiter_release(self.0.as_ptr()) }
    }
}
// SAFETY: the C ABI synchronizes these owning handles; Drop cannot race a borrow.
unsafe impl Send for DiagnosticClient {}
unsafe impl Sync for DiagnosticClient {}
unsafe impl Send for Context {}
unsafe impl Sync for Context {}
unsafe impl Send for RateLimiter {}
unsafe impl Sync for RateLimiter {}
