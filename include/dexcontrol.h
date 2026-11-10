// Copyright (C) 2026 Dexmate Inc.
// SPDX-License-Identifier: Apache-2.0

#ifndef DEXCONTROL_H
#define DEXCONTROL_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct dex_robot_t dex_robot_t;
typedef struct dex_diagnostic_t dex_diagnostic_t;
typedef struct dex_component_t dex_component_t;
typedef struct dex_motion_t dex_motion_t;
typedef struct dex_motion_group_t dex_motion_group_t;

/* ======================================================================
 * Threading and lifetime
 *
 *   - Every handle (robot, diagnostic, component, motion, motion group,
 *     context, frame, lidar snapshot, rate limiter) may be used from several
 *     threads at the same time. Calls are internally synchronized; a blocking
 *     call (a waited move, dex_robot_wait_safety_event_json) blocks only its
 *     own thread.
 *   - Releasing a handle must not race with any other call on that same
 *     handle: a *_release is the one operation that needs exclusive access.
 *     Give a second thread its own reference with dex_robot_retain().
 *   - Components, motions and motion groups keep their robot alive. The
 *     robot disconnects when the LAST of them (and the robot handle itself)
 *     is released; if nobody called dex_robot_close() by then, that last
 *     release first runs a bounded (2 s) best-effort stop and shutdown, so a
 *     handle lost to an early return or a C++ exception never leaves a
 *     server-tracked motion running with no client. Call dex_robot_close()
 *     to stop at a moment you choose and to see the error.
 *   - dex_runtime_configure(), dex_watchdog_configure() and
 *     dex_safety_configure() are PROCESS-GLOBAL: they affect every robot
 *     created afterwards, from any thread, and racing them against a
 *     dex_robot_create*() on another thread is a bug in the caller. Prefer
 *     the per-robot fields of dex_robot_options_t.
 *   - No function may be called from a signal handler.
 * ====================================================================== */

/* Compile-time ABI version of these declarations. Compare against the
 * runtime dex_abi_version(): within one major ABI version, changes are
 * additive only. Extensible structs are always passed by pointer and carry
 * a caller-set struct_size announcing which prefix is present. */
#define DEX_ABI_VERSION ((uint32_t)5)

typedef int32_t dex_status_t;
/* Every dex_status_t is a result to look at: a command that was refused
 * (E-stop, stale state, a limit) reports it only here. C++17 and C23 callers
 * get a diagnostic for a discarded status; older C has no portable spelling
 * that a deliberate (void) cast can still silence, so it gets none. */
#if defined(__cplusplus) && __cplusplus >= 201703L
#define DEX_NODISCARD [[nodiscard]]
#elif defined(__STDC_VERSION__) && __STDC_VERSION__ >= 202311L
#define DEX_NODISCARD [[nodiscard]]
#else
#define DEX_NODISCARD
#endif
#define DEX_OK ((dex_status_t)0)
#define DEX_INVALID_ARGUMENT ((dex_status_t)1) /* a bad argument or an invalid command: the call, not the robot */
#define DEX_RUNTIME_ERROR ((dex_status_t)2)
#define DEX_TIMEOUT ((dex_status_t)3)
#define DEX_PANIC ((dex_status_t)4)
/* The measured state a command needed was older than the configured
 * freshness limit (max_state_age_ms), so nothing was published. Recoverable:
 * retry once the state stream recovers, raise the limit, or disable it with
 * DEX_LIMIT_DISABLED. */
#define DEX_STALE_STATE ((dex_status_t)5)
/* No sample has been received yet on the endpoint the call needed. */
#define DEX_STATE_UNAVAILABLE ((dex_status_t)6)
#define DEX_SERVICE_REJECTED ((dex_status_t)7) /* delivered, explicitly refused by server */
/* A well-formed write was refused because an emergency stop is engaged, or
 * because the E-stop state cannot be read so its release cannot be confirmed.
 * Nothing was published. A robot condition, never a caller mistake: release
 * the E-stop (or restore its state stream) and retry. */
#define DEX_ESTOP_ACTIVE ((dex_status_t)8)
/* A blocking call (a waited move, a host trajectory, a reconnect) ended
 * because motion was stopped first: dex_robot_stop_all(), a safety action, a
 * shutdown, or the motion being cancelled or superseded. A waited move still
 * writes its live motion to *out_motion, exactly as DEX_TIMEOUT does: release
 * it whatever the status. */
#define DEX_STOPPED ((dex_status_t)9)
/* Status codes are append-only. Treat a value you do not know as a failure
 * of the DEX_RUNTIME_ERROR class. */

#define DEX_NO_WAIT ((uint32_t)0)
#define DEX_WAIT_UNTIL_COMPLETE ((uint32_t)1)
#define DEX_WAIT_WITH_TIMEOUT ((uint32_t)2)

/* Motion states reported by dex_motion_state(). Values are frozen within
 * this major ABI version; new states may only be appended. */
#define DEX_MOTION_PENDING ((uint32_t)0)
#define DEX_MOTION_RUNNING ((uint32_t)1)
#define DEX_MOTION_SUCCEEDED ((uint32_t)2)
#define DEX_MOTION_CANCELLED ((uint32_t)3)
#define DEX_MOTION_FAILED ((uint32_t)4)
#define DEX_MOTION_SUPERSEDED ((uint32_t)5)

/* Sentinel for max_state_age_ms / wait_ceiling_ms: disables the check. */
#define DEX_LIMIT_DISABLED ((uint64_t)UINT64_MAX)
/* Sentinel for max_state_age_ms: use the component's configured limit (its
 * state_max_age_ms model metadata, else DEX_DEFAULT_MAX_STATE_AGE_MS). This
 * is what dex_wait_policy_init() writes. */
#define DEX_STATE_AGE_CONFIGURED ((uint64_t)0)
/* Library-wide defaults. A 0 in wait_ceiling_ms selects the ceiling default;
 * a 0 in max_state_age_ms selects the configured limit above. 500 ms matches
 * the native default: measured state gaps on healthy robots reach 201 ms,
 * and the earlier 100 ms rejected commands on robots behaving correctly. */
#define DEX_DEFAULT_MAX_STATE_AGE_MS ((uint64_t)500)
#define DEX_DEFAULT_WAIT_CEILING_MS ((uint64_t)300000)

/* Additive struct: call dex_wait_policy_init(), which sets struct_size and
 * the library defaults, or zero it and set struct_size = sizeof: every
 * field's zero selects its default. A caller compiled against an older
 * header announces a smaller struct_size and transparently gets the defaults
 * for the trailing fields. */
typedef struct dex_wait_policy_t {
    uint32_t struct_size;
    uint32_t mode;
    uint64_t timeout_ms;
    /* Radians; 0 selects the default (0.01). */
    double convergence_tolerance;
    uint64_t settle_time_ms;
    /* Maximum age of the measured state a command may be computed from
     * (relative targets, the per-joint step guard, automatic
     * position-velocity feed-forward, and convergence waiting).
     * DEX_STATE_AGE_CONFIGURED (0) selects the component's configured limit;
     * DEX_LIMIT_DISABLED turns the check off. State older than this is
     * rejected with DEX_STALE_STATE before anything is published. */
    uint64_t max_state_age_ms;
    /* Hard ceiling on a DEX_WAIT_UNTIL_COMPLETE convergence wait, so no call
     * can block forever. 0 selects DEX_DEFAULT_WAIT_CEILING_MS;
     * DEX_LIMIT_DISABLED removes the ceiling. Exceeding it reports
     * DEX_TIMEOUT. */
    uint64_t wait_ceiling_ms;
} dex_wait_policy_t;

/* Error details for one call. The returned dex_status_t is always
 * authoritative; this struct adds the message.
 *
 * struct_size IS MANDATORY. Initialize with dex_error_init(), or set
 * struct_size = sizeof(dex_error_t) yourself. In particular
 *
 *     dex_error_t error = {0};        // WRONG: announces a size of 0
 *
 * tells the library that not one byte may be written, so a failing call
 * returns its status and leaves `message` empty. A struct_size that reaches
 * into `message` always receives a NUL-terminated message inside the
 * announced region, truncated at a UTF-8 character boundary. Every function
 * accepts error == NULL. */
typedef struct dex_error_t {
    uint32_t struct_size;
    int32_t code;
    char message[256];
} dex_error_t;

/* Runtime construction options for robots created after
 * dex_runtime_configure(). worker_threads == 0 selects the library default;
 * thread_name_utf8 == NULL keeps the default thread name (the string is
 * copied during the call). */
typedef struct dex_runtime_options_t {
    uint32_t struct_size;
    uint32_t worker_threads;
    const char *thread_name_utf8;
} dex_runtime_options_t;

uint32_t dex_abi_version(void);
void dex_wait_policy_init(dex_wait_policy_t *policy);
void dex_error_init(dex_error_t *error);
void dex_runtime_options_init(dex_runtime_options_t *options);

/* Applies to robots created after this call; NULL restores defaults.
 * PROCESS-GLOBAL; see "Threading and lifetime". A dex_context_t is the
 * per-application alternative. */
DEX_NODISCARD dex_status_t dex_runtime_configure(const dex_runtime_options_t *options,
                                   dex_error_t *error);

/* LEGACY, PROCESS-GLOBAL: prefer dex_robot_options_t.disable_watchdog and
 * .watchdog_command_utf8, which are per robot and cannot race.
 *
 * Configures the dedicated-process safety watchdog for robots created after
 * this call (mirrors dex_runtime_configure). command_utf8 is a
 * whitespace-split spawn command for the dexcontrol-watchdog executable;
 * NULL keeps the automatic resolution (DEXCONTROL_WATCHDOG_CMD env, a
 * binary next to the current executable, then PATH). enabled=false opts out
 * of process-level supervision; enabled=true keeps the default behavior:
 * on for production robots whose configuration monitors a heartbeat or
 * E-stop, never for simulated robots. An unresolvable command degrades to
 * in-process safety monitors with a reported warning; robot creation never
 * fails because of the watchdog. */
DEX_NODISCARD dex_status_t dex_watchdog_configure(const char *command_utf8,
                                    bool enabled,
                                    dex_error_t *error);

/* LEGACY, PROCESS-GLOBAL: prefer dex_robot_options_t.exit_on_termination.
 *
 * Configures the safety termination policy for robots created after this
 * call (mirrors dex_runtime_configure). With exit_on_termination_request
 * true, a safety monitor firing the configured request_process_termination
 * action first makes the robot safe (stop motion plus coordinated robot
 * shutdown), emits the safety event, and then terminates this process with
 * exit code 1 (no unwinding, no atexit-style cleanup). Default false: the
 * library reports the request and the host decides whether to exit. Enable
 * only in a process that exists solely to run the robot. */
DEX_NODISCARD dex_status_t dex_safety_configure(bool exit_on_termination_request,
                                  dex_error_t *error);

/* Production-only, read-only diagnostics. The opaque handle exposes no
 * command API and does not start safety actors or a watchdog. Snapshot JSON
 * is allocated by the library and must be released with dex_string_free(). */
DEX_NODISCARD dex_status_t dex_diagnostic_create_from_profile(
    const char *profile_name_utf8, dex_diagnostic_t **out_client, dex_error_t *error);
DEX_NODISCARD dex_status_t dex_diagnostic_create_from_file(
    const char *path_utf8, dex_diagnostic_t **out_client, dex_error_t *error);
DEX_NODISCARD dex_status_t dex_diagnostic_snapshot_json(
    dex_diagnostic_t *client, uint64_t settle_timeout_ms,
    char **out_json, dex_error_t *error);
DEX_NODISCARD dex_status_t dex_diagnostic_shutdown(dex_diagnostic_t *client, dex_error_t *error);
void dex_diagnostic_release(dex_diagnostic_t *client);
void dex_string_free(char *value);

DEX_NODISCARD dex_status_t dex_robot_create_from_profile(const char *profile_name_utf8,
                                            dex_robot_t **out_robot,
                                            dex_error_t *error);
DEX_NODISCARD dex_status_t dex_robot_create_from_file(const char *path_utf8,
                                         dex_robot_t **out_robot,
                                         dex_error_t *error);
DEX_NODISCARD dex_status_t dex_robot_create_from_resolved_json(const uint8_t *json,
                                                  size_t json_len,
                                                  dex_robot_t **out_robot,
                                                  dex_error_t *error);

/* Deterministic in-process simulation. Never contacts robot hardware. */
DEX_NODISCARD dex_status_t dex_robot_create_simulated_from_profile(
    const char *profile_name_utf8, dex_robot_t **out_robot, dex_error_t *error);
DEX_NODISCARD dex_status_t dex_robot_create_simulated_from_resolved_json(
    const uint8_t *json, size_t json_len, dex_robot_t **out_robot, dex_error_t *error);
/* Test support for simulated robots: while enabled, newly accepted tracked
 * motions stay Running instead of succeeding immediately, so timeout and
 * cancellation paths can be exercised deterministically. Disabling releases
 * held motions as Succeeded. Fails on robots not created simulated. */
DEX_NODISCARD dex_status_t dex_robot_simulation_hold_motions(dex_robot_t *robot,
                                               bool hold,
                                               dex_error_t *error);
/* Adds a reference, e.g. for a second thread; pair with dex_robot_release. */
DEX_NODISCARD dex_status_t dex_robot_retain(dex_robot_t *robot);
/* Drops one reference; NULL is accepted. The robot stays connected while any
 * component, motion or motion group obtained from it is alive. Releasing the
 * LAST reference to a robot that was never closed first runs a bounded (2 s)
 * best-effort stop and shutdown; see "Threading and lifetime". */
void dex_robot_release(dex_robot_t *robot);
/* Stops motion and the mobile base, shuts supervision down and closes the
 * transport, now. Idempotent. Handles stay valid (and must still be
 * released); commands on them fail. Wakes a pending
 * dex_robot_wait_safety_event_json(). */
DEX_NODISCARD dex_status_t dex_robot_close(dex_robot_t *robot, dex_error_t *error);
/* The original name of dex_robot_close(); identical behavior. */
DEX_NODISCARD dex_status_t dex_robot_shutdown(dex_robot_t *robot, dex_error_t *error);
DEX_NODISCARD dex_status_t dex_robot_reconnect(dex_robot_t *robot, dex_error_t *error);
DEX_NODISCARD dex_status_t dex_robot_chassis_set_velocity(dex_robot_t *robot,
                                              double vx,
                                              double vy,
                                              double wz,
                                              dex_error_t *error);
DEX_NODISCARD dex_status_t dex_robot_chassis_stop(dex_robot_t *robot, dex_error_t *error);

/* Pauses heartbeat staleness supervision indefinitely (in-process monitor
 * and the dedicated watchdog process alike; E-stop monitoring is
 * unaffected). SAFETY CAVEAT: while paused, a dead robot server goes
 * unnoticed by heartbeat supervision -- pair every pause with a guaranteed
 * dex_robot_resume_heartbeat. On resume, supervision re-arms from the next
 * observed sample, so staleness accrued during the pause never fires
 * retroactively. Both calls are idempotent. */
DEX_NODISCARD dex_status_t dex_robot_pause_heartbeat(dex_robot_t *robot, dex_error_t *error);
DEX_NODISCARD dex_status_t dex_robot_resume_heartbeat(dex_robot_t *robot, dex_error_t *error);

DEX_NODISCARD dex_status_t dex_robot_joint_component(dex_robot_t *robot,
                                       const char *name_utf8,
                                       dex_component_t **out_component,
                                       dex_error_t *error);
void dex_component_release(dex_component_t *component);

DEX_NODISCARD dex_status_t dex_component_joint_count(const dex_component_t *component,
                                         size_t *out_count);
DEX_NODISCARD dex_status_t dex_component_get_joint_pos(const dex_component_t *component,
                                          double *values_out,
                                          size_t capacity,
                                          size_t *out_count,
                                          dex_error_t *error);

/* Normalized grasping torque sent with every command. Writes 1 to
 * has_torque_out for a torque-commanded component (a gripper), 0 otherwise. */
DEX_NODISCARD dex_status_t dex_component_grasp_torque(dex_component_t *component,
                                        double *torque_out,
                                        int *has_torque_out,
                                        dex_error_t *error);

/* Sets the grasping torque for every subsequent command, writing the value
 * actually applied (clipped to [0, 1]) to applied_out. Fails with
 * DEX_INVALID_ARGUMENT when the component is not torque-commanded; a non-zero
 * force suppresses the high-torque warning. */
DEX_NODISCARD dex_status_t dex_component_set_grasp_torque(dex_component_t *component,
                                            double torque,
                                            int force,
                                            double *applied_out,
                                            dex_error_t *error);

/* Direct, unplanned position command: one setpoint, no interpolation. The
 * component's own guards apply exactly as they do for Rust and Python
 * callers: URDF joint limits, and on plugin-managed components (arms, head,
 * torso) a 0.3 rad per-joint step ceiling, because a larger unshaped step is
 * not reliably executed. Cover longer distances with
 * dex_component_move_to_joint_pos(). */
DEX_NODISCARD dex_status_t dex_component_set_joint_pos(dex_component_t *component,
                                         const double *values,
                                         size_t count,
                                         const dex_wait_policy_t *wait,
                                         dex_error_t *error);
DEX_NODISCARD dex_status_t dex_component_set_joint_pos_vel(dex_component_t *component,
                                             const double *positions,
                                             const double *velocities,
                                             size_t count,
                                             const dex_wait_policy_t *wait,
                                             dex_error_t *error);
DEX_NODISCARD dex_status_t dex_component_move_to_joint_pos(dex_component_t *component,
                                             const double *values,
                                             size_t count,
                                             const dex_wait_policy_t *wait,
                                             dex_motion_t **out_motion,
                                             dex_error_t *error);

/* Explicit per-application runtime ownership: one runtime of `workers`
 * threads (1..256) shared by the robots connected through it, with no
 * process-global configuration involved. Robots keep the runtime alive after
 * the context is released. dex_context_connect() is declared with
 * dex_robot_options_t below. */
typedef struct dex_context_t dex_context_t;
DEX_NODISCARD dex_status_t dex_context_create(size_t workers, dex_context_t **out_context, dex_error_t *error);
void dex_context_release(dex_context_t *context);
/* LEGACY forward to dex_context_connect(): three adjacent positional bools
 * are easy to transpose and cannot grow. watchdog=true keeps the library
 * default. profile must not be NULL. */
DEX_NODISCARD dex_status_t dex_context_connect_profile(const dex_context_t *context, const char *profile,
    bool simulation, bool watchdog, bool exit_on_termination, dex_robot_t **out_robot, dex_error_t *error);

/* DEX_OK means report available; inspect all_delivered. Release both outputs.
 * Library-default motion options; dex_robot_start_joint_motions_opt() below
 * takes dex_motion_options_t. */
DEX_NODISCARD dex_status_t dex_robot_start_joint_motions(dex_robot_t *robot, const char *const *names,
    const double *const *values, const size_t *counts, size_t n,
    dex_motion_group_t **out_group, char **out_report, dex_error_t *error);

DEX_NODISCARD dex_status_t dex_component_command_position(const dex_component_t *component,
    const double *values, const double *limits, size_t count, uint64_t max_state_age_ms, dex_error_t *error);

DEX_NODISCARD dex_status_t dex_robot_initialize_arms_json(dex_robot_t *robot, char **out, dex_error_t *error);
DEX_NODISCARD dex_status_t dex_robot_initialize_components_json(dex_robot_t *robot, char **out, dex_error_t *error);
DEX_NODISCARD dex_status_t dex_robot_joint_state_json(dex_robot_t *robot, const char *name, char **out, dex_error_t *error);

/* Native source/build identity; release the result with dex_string_free(). */
DEX_NODISCARD dex_status_t dex_build_info_json(char **out_json, dex_error_t *error);

DEX_NODISCARD dex_status_t dex_motion_wait_success(dex_motion_t *motion,
    const dex_wait_policy_t *wait, dex_error_t *error);
DEX_NODISCARD dex_status_t dex_motion_wait(dex_motion_t *motion,
                             const dex_wait_policy_t *wait,
                             dex_error_t *error);
DEX_NODISCARD dex_status_t dex_motion_cancel(dex_motion_t *motion, dex_error_t *error);
DEX_NODISCARD dex_status_t dex_motion_id(const dex_motion_t *motion, uint64_t *out_id);
DEX_NODISCARD dex_status_t dex_motion_state(const dex_motion_t *motion, uint32_t *out_state);
void dex_motion_release(dex_motion_t *motion);

/* ======================================================================
 * The full facade.
 *
 * Conventions for everything below:
 *   - Text comes back as an owned, NUL-terminated string in `*out` and is
 *     released with dex_string_free(). Raw bytes come back as a
 *     dex_buffer_t released with dex_buffer_free().
 *   - Numeric arrays fill caller storage of `capacity` elements. The
 *     required element count is always written to `*out_count`, so a call
 *     with a null array or zero capacity is a size probe that fails with
 *     DEX_INVALID_ARGUMENT after reporting the count. Optional arrays
 *     (documented as "may be null") are simply skipped when null.
 *   - Extensible structs carry struct_size; call their _init first, or
 *     zero them and set struct_size = sizeof. For every INPUT struct the two
 *     are equivalent by construction: each field's zero is its safe default
 *     (limits enforced, guards on, configured freshness), and every opt-out
 *     is a non-zero value you have to write on purpose. Flags in input
 *     structs are uint8_t, any non-zero byte meaning "set". Output structs
 *     must have struct_size set before the call; only that prefix is
 *     written.
 *   - Reads that need a sample that has not arrived report
 *     DEX_STATE_UNAVAILABLE. dex_robot_wait_for_state() waits for one.
 * ====================================================================== */

typedef struct dex_frame_t dex_frame_t;
typedef struct dex_rate_limiter_t dex_rate_limiter_t;

typedef struct dex_buffer_t {
    uint8_t *data;
    size_t len;
} dex_buffer_t;
void dex_buffer_free(dex_buffer_t *buffer);

/* ---- Robot: introspection ------------------------------------------- */
/* How to build a robot or diagnostic client. Zero with
 * dex_robot_options_init(), then set what you need. A null profile resolves
 * the profile from the environment (ROBOT_CONFIG, then ROBOT_NAME, then
 * vega_1 when neither is set); a ROBOT_NAME that names no known robot is an
 * error, never a guess. config_file_utf8 wins over the profile when both are
 * set. */
typedef struct dex_robot_options_t {
    uint32_t struct_size;
    const char *profile_utf8;
    const char *config_file_utf8;
    const char *const *enable_sensors;
    size_t enable_sensor_count;
    /* Non-zero: the deterministic in-process simulation; never contacts
     * hardware. */
    uint8_t simulated;
    /* Non-zero: fail the connection when the server version cannot be
     * verified, instead of continuing unverified. */
    uint8_t require_version_check;
    /* Per-robot safety policy. Zero/NULL defers to the legacy process-global
     * setters in dex_robot_create() and to the library default in
     * dex_context_connect(). disable_watchdog opts out of the
     * dedicated-process watchdog (default: on for production robots, never
     * for simulated ones). exit_on_termination terminates this process after
     * a request_process_termination safety action made the robot safe; see
     * dex_safety_configure(). watchdog_command_utf8 is a whitespace-split
     * spawn command; NULL keeps the automatic resolution. */
    uint8_t disable_watchdog;
    uint8_t exit_on_termination;
    const char *watchdog_command_utf8;
    /* Non-zero: connect without putting the arms into position mode. The
     * default enables every arm once it is ready, and a required arm whose
     * motor cannot enable then fails the connection; a troubleshooting tool
     * (clearing that arm's error) or an application that sets modes itself
     * connects with this set. Everything else starts as usual. Added within
     * ABI 5: callers announcing the shorter struct_size get the default. */
    uint8_t skip_arm_initialization;
} dex_robot_options_t;

DEX_NODISCARD dex_status_t dex_robot_options_init(dex_robot_options_t *options);
/* Null options build the environment-selected production robot. */
DEX_NODISCARD dex_status_t dex_robot_create(const dex_robot_options_t *options,
                              dex_robot_t **out_robot, dex_error_t *error);
/* Connects on the context's runtime; NULL options build the
 * environment-selected production robot. The process-global dex_*_configure
 * setters are ignored here. */
DEX_NODISCARD dex_status_t dex_context_connect(const dex_context_t *context,
                                 const dex_robot_options_t *options,
                                 dex_robot_t **out_robot, dex_error_t *error);
/* Read-only diagnostic client; `simulated` selects the in-process mock. */
DEX_NODISCARD dex_status_t dex_diagnostic_create(const dex_robot_options_t *options,
                                   dex_diagnostic_t **out_client,
                                   dex_error_t *error);
/* The statically resolved configuration for `options` as normalized JSON,
 * without connecting: component and sensor names with their `enabled`
 * flags. Release with dex_string_free(). */
DEX_NODISCARD dex_status_t dex_resolved_config_json(const dex_robot_options_t *options,
                                      char **out_json, dex_error_t *error);
/* The profile the environment selects; DEX_INVALID_ARGUMENT for a ROBOT_NAME
 * that names no known robot. Release with dex_string_free(). */
DEX_NODISCARD dex_status_t dex_profile_from_environment(char **out_name, dex_error_t *error);

DEX_NODISCARD dex_status_t dex_robot_config_json(dex_robot_t *robot, char **out_json, dex_error_t *error);
DEX_NODISCARD dex_status_t dex_robot_health_json(dex_robot_t *robot, char **out_json, dex_error_t *error);
DEX_NODISCARD dex_status_t dex_robot_version_info_json(dex_robot_t *robot, char **out_json, dex_error_t *error);
DEX_NODISCARD dex_status_t dex_robot_connection_report_json(dex_robot_t *robot, char **out_json, dex_error_t *error);
DEX_NODISCARD dex_status_t dex_robot_profile_name(dex_robot_t *robot, char **out_name, dex_error_t *error);
DEX_NODISCARD dex_status_t dex_robot_model_name(dex_robot_t *robot, char **out_name, dex_error_t *error);
DEX_NODISCARD dex_status_t dex_robot_is_active(const dex_robot_t *robot, bool *out_active);
DEX_NODISCARD dex_status_t dex_robot_wait_for_active(dex_robot_t *robot, uint64_t timeout_ms,
                                       bool *out_active, dex_error_t *error);
DEX_NODISCARD dex_status_t dex_robot_has_component(dex_robot_t *robot, const char *name_utf8,
                                     bool *out_present, dex_error_t *error);
DEX_NODISCARD dex_status_t dex_robot_has_sensor(dex_robot_t *robot, const char *name_utf8,
                                  bool *out_present, dex_error_t *error);
/* Enabled component and sensor names, sorted; index 0..count-1. */
DEX_NODISCARD dex_status_t dex_robot_component_count(const dex_robot_t *robot, size_t *out_count);
DEX_NODISCARD dex_status_t dex_robot_component_name(dex_robot_t *robot, size_t index,
                                      char **out_name, dex_error_t *error);
DEX_NODISCARD dex_status_t dex_robot_sensor_count(const dex_robot_t *robot, size_t *out_count);
DEX_NODISCARD dex_status_t dex_robot_sensor_name(dex_robot_t *robot, size_t index,
                                   char **out_name, dex_error_t *error);
/* Waits until every state stream of a component or sensor delivered a
 * sample; *out_ready is false on timeout. */
/* Component liveness: the state stream has a sample within the configured
 * freshness limit. The wait reports a timeout through *out_active, not as
 * an error. Added within ABI 5. */
DEX_NODISCARD dex_status_t dex_component_is_active(const dex_component_t *component, bool *out_active);
DEX_NODISCARD dex_status_t dex_component_wait_for_active(const dex_component_t *component, uint64_t timeout_ms,
                                                       bool *out_active, dex_error_t *error);
DEX_NODISCARD dex_status_t dex_robot_wait_for_state(dex_robot_t *robot, const char *name_utf8,
                                      uint64_t timeout_ms, bool *out_ready,
                                      dex_error_t *error);

/* ---- Robot: commands -------------------------------------------------- */
typedef struct dex_motion_options_t {
    uint32_t struct_size;
    uint8_t relative; /* non-zero: target is relative to the measured position */
    /* Non-zero turns the joint-limit check OFF. Negative sense on purpose:
     * a zeroed struct leaves limits enforced. */
    uint8_t disable_limit_enforcement;
    /* Per-joint fraction (0, 1] of the hardware velocity ceiling; 0 leaves
     * the choice to the motion plugin. */
    double velocity_scale;
    uint64_t max_state_age_ms; /* as dex_wait_policy_t */
    uint64_t wait_ceiling_ms;  /* 0 = the component's configured ceiling */
} dex_motion_options_t;
void dex_motion_options_init(dex_motion_options_t *options);

/* names/values/counts are parallel arrays of n targets. Validation is
 * all-or-none before anything is published. */
/* Direct-command options, shared by the robot-level fan-out here and the
 * component commands below. */
#define DEX_STEP_CONFIGURED (0.0)
#define DEX_STEP_DISABLED (-1.0)
typedef struct dex_command_options_t {
    uint32_t struct_size;
    uint8_t relative; /* non-zero: target is relative to the measured position */
    /* Non-zero turns the joint-limit check OFF. Negative sense on purpose:
     * a zeroed struct leaves limits enforced. */
    uint8_t disable_limit_enforcement;
    /* Per-joint step ceiling in radians: DEX_STEP_CONFIGURED lets the
     * component decide (0.3 rad on arms/head/torso, none on hands and the
     * chassis); exactly DEX_STEP_DISABLED turns the guard off. NaN, infinity
     * or any other negative value is DEX_INVALID_ARGUMENT on every entry
     * point: a sign error never removes the guard. */
    double max_step_rad;
} dex_command_options_t;
/* Direct fan-out to several components, validated all-or-none. Null options
 * are the defaults: absolute targets, joint limits enforced, each
 * component's own step rule and configured freshness limit. */
DEX_NODISCARD dex_status_t dex_robot_set_joint_pos_multi(dex_robot_t *robot,
                                           const char *const *names,
                                           const double *const *values,
                                           const size_t *counts, size_t n,
                                           const dex_command_options_t *options,
                                           dex_error_t *error);
/* options may be null. The group must be released with
 * dex_motion_group_release(). */
/* Streamed-goal fan-out; see dex_component_stream_joint_pos. */
DEX_NODISCARD dex_status_t dex_robot_stream_joint_pos_multi(dex_robot_t *robot,
    const char *const *names, const double *const *values, const size_t *counts, size_t n,
    const dex_motion_options_t *options, dex_error_t *error);
/* On partial delivery failure, out_group may be non-null and owns the
 * recovery group. Inspect/release it even when the returned status is not OK.
 * For the full per-component report use dex_robot_start_joint_motions_opt. */
DEX_NODISCARD dex_status_t dex_robot_move_to_joint_pos_multi(dex_robot_t *robot,
                                               const char *const *names,
                                               const double *const *values,
                                               const size_t *counts, size_t n,
                                               const dex_motion_options_t *options,
                                               dex_motion_group_t **out_group,
                                               dex_error_t *error);
/* As dex_robot_start_joint_motions() with options (NULL for the defaults).
 * max_state_age_ms == 0 lets every component apply its own configured
 * freshness limit. */
DEX_NODISCARD dex_status_t dex_robot_start_joint_motions_opt(dex_robot_t *robot, const char *const *names,
    const double *const *values, const size_t *counts, size_t n,
    const dex_motion_options_t *options,
    dex_motion_group_t **out_group, char **out_report, dex_error_t *error);

/* Streams direct setpoints at control_hz, one per track per tick.
 * positions[i] holds ticks * joint_counts[i] values, tick-major.
 * velocities may be null, or hold a null entry for a track without
 * feed-forward. max_tracking_error <= 0 disables the per-tick guard.
 *
 * Every tick is one unshaped setpoint, so it gets the protection
 * dex_component_set_joint_pos() has: on a step-guarded component (arms,
 * head, torso) the first waypoint must lie within the step ceiling of the
 * MEASURED position and every later waypoint within the ceiling of the one
 * before it, all checked before the first tick is published. Precede a
 * trajectory that starts elsewhere with a planned move to its first
 * waypoint. */
DEX_NODISCARD dex_status_t dex_robot_execute_trajectory(dex_robot_t *robot,
                                          const char *const *names,
                                          const double *const *positions,
                                          const double *const *velocities,
                                          size_t ticks, const size_t *joint_counts,
                                          size_t n, double control_hz,
                                          double max_tracking_error,
                                          dex_error_t *error);
typedef struct dex_trajectory_options_t {
    uint32_t struct_size;
    double max_tracking_error; /* rad; <= 0 disables the per-tick guard */
    uint64_t max_state_age_ms; /* as dex_wait_policy_t; 0 = DEX_DEFAULT_MAX_STATE_AGE_MS */
    /* The start-distance and waypoint-spacing guard, read exactly like
     * dex_command_options_t.max_step_rad: DEX_STEP_CONFIGURED (0) applies
     * each component's own rule, DEX_STEP_DISABLED (-1) is the expert
     * opt-out, a positive value is one ceiling for every track. */
    double max_step_rad;
} dex_trajectory_options_t;
void dex_trajectory_options_init(dex_trajectory_options_t *options);
/* options may be null for the defaults. */
DEX_NODISCARD dex_status_t dex_robot_execute_trajectory_opt(dex_robot_t *robot,
                                              const char *const *names,
                                              const double *const *positions,
                                              const double *const *velocities,
                                              size_t ticks, const size_t *joint_counts,
                                              size_t n, double control_hz,
                                              const dex_trajectory_options_t *options,
                                              dex_error_t *error);
DEX_NODISCARD dex_status_t dex_robot_stop_all(dex_robot_t *robot, dex_error_t *error);
/* part is "left_arm", "right_arm" or "head". */
DEX_NODISCARD dex_status_t dex_robot_compensate_torso_pitch(dex_robot_t *robot, const double *values,
                                              size_t count, const char *part_utf8,
                                              double *out_values, size_t capacity,
                                              size_t *out_count, dex_error_t *error);
DEX_NODISCARD dex_status_t dex_robot_torso_pitch(dex_robot_t *robot, double *out_pitch, dex_error_t *error);

/* ---- Robot: maintenance and diagnostics ------------------------------ */
DEX_NODISCARD dex_status_t dex_robot_clear_error(dex_robot_t *robot, const char *name_utf8, dex_error_t *error);
DEX_NODISCARD dex_status_t dex_robot_clear_all_errors_json(dex_robot_t *robot, char **out_json, dex_error_t *error);
/* board is "arm", "torso" or "chassis"; one board drives both arms. */
DEX_NODISCARD dex_status_t dex_robot_reboot(dex_robot_t *robot, const char *board_utf8, dex_error_t *error);
DEX_NODISCARD dex_status_t dex_robot_query(dex_robot_t *robot, const char *name_utf8,
                             const uint8_t *payload, size_t payload_len,
                             dex_buffer_t *out_reply, dex_error_t *error);
DEX_NODISCARD dex_status_t dex_robot_query_ntp_json(dex_robot_t *robot, size_t sample_count,
                                      char **out_json, dex_error_t *error);
DEX_NODISCARD dex_status_t dex_robot_component_status_json(dex_robot_t *robot, char **out_json, dex_error_t *error);
/* Every safety event delivered so far, as a JSON array, oldest first. Never
 * blocks: while another thread is parked in
 * dex_robot_wait_safety_event_json() this returns "[]" and that wait
 * delivers the event. */
DEX_NODISCARD dex_status_t dex_robot_take_safety_events_json(dex_robot_t *robot, char **out_json, dex_error_t *error);
/* Blocks until the next safety event or timeout_ms and writes it as one
 * JSON object, or the JSON `null` on timeout. Buffered events come first.
 * dex_robot_close(), dex_robot_shutdown() and the last dex_robot_release()
 * wake a pending wait, which then also writes `null`: dex_robot_is_active()
 * tells that from a timeout. On a robot that is already closed the call
 * returns at once with whatever is still buffered. */
DEX_NODISCARD dex_status_t dex_robot_wait_safety_event_json(dex_robot_t *robot,
                                              uint64_t timeout_ms,
                                              char **out_json,
                                              dex_error_t *error);
/* Built-in profile names as a JSON array. Release with dex_string_free(). */
DEX_NODISCARD dex_status_t dex_available_profiles_json(char **out_json, dex_error_t *error);
/* The built-in profile a robot identity (ROBOT_NAME style, e.g. dm/vg1p-...)
 * or profile name selects. An identity that names no known robot fails with
 * DEX_INVALID_ARGUMENT and the model's message; it is never guessed. Release
 * with dex_string_free(). */
DEX_NODISCARD dex_status_t dex_profile_for_robot_name(const char *robot_name_utf8,
                                        char **out_name, dex_error_t *error);
DEX_NODISCARD dex_status_t dex_diagnostic_query_ntp_json(dex_diagnostic_t *client, size_t sample_count,
                                           char **out_json, dex_error_t *error);


/* ---- Robot: E-stop, battery, mobile base ----------------------------- */
typedef struct dex_estop_status_t {
    uint32_t struct_size;
    bool engaged;
    bool software_estop_enabled;
    /* False when nothing was ever published: on event-driven firmware that
     * is the healthy released state, and observed_age_ms is UINT64_MAX. */
    bool state_observed;
    uint64_t observed_age_ms;
    bool left_base_estop_enabled;
    bool right_base_estop_enabled;
    bool torso_estop_enabled;
    bool remote_estop_enabled;

} dex_estop_status_t;
DEX_NODISCARD dex_status_t dex_robot_estop_status(dex_robot_t *robot, dex_estop_status_t *out, dex_error_t *error);
DEX_NODISCARD dex_status_t dex_robot_estop_set(dex_robot_t *robot, bool enabled, dex_error_t *error);

typedef struct dex_battery_t {
    uint32_t struct_size;
    double voltage;      /* V */
    double current;      /* A */
    double temperature;  /* C */
    uint32_t percentage;
    bool is_charging;
    uint64_t timestamp_ns;
} dex_battery_t;
DEX_NODISCARD dex_status_t dex_robot_battery(dex_robot_t *robot, dex_battery_t *out, dex_error_t *error);
/* With a freshness bound and the local arrival age; see dex_robot_imu_ex(). */
DEX_NODISCARD dex_status_t dex_robot_battery_ex(dex_robot_t *robot, uint64_t max_age_ms, dex_battery_t *out,
                                  uint64_t *out_age_ms, dex_error_t *error);

typedef struct dex_chassis_state_t {
    uint32_t struct_size;
    double steering[2];       /* rad, left then right */
    double wheel_velocity[2]; /* rad/s */
    double wheel_position[2]; /* rad */
} dex_chassis_state_t;
typedef struct dex_chassis_options_t {
    uint32_t struct_size;
    /* Non-zero turns sequential steering OFF. By default the base steers
     * first, then drives, when the wheels are far from the new heading. */
    uint8_t disable_sequential_steering;
    double steering_tolerance; /* rad; 0 selects the default (0.05) */
    uint64_t steering_wait_ms; /* 0 selects the default (1000) */
    uint64_t max_state_age_ms; /* as dex_wait_policy_t */
} dex_chassis_options_t;
void dex_chassis_options_init(dex_chassis_options_t *options);
DEX_NODISCARD dex_status_t dex_robot_chassis_state(dex_robot_t *robot, dex_chassis_state_t *out, dex_error_t *error);
DEX_NODISCARD dex_status_t dex_robot_chassis_set_motion_state(dex_robot_t *robot, const double steering[2],
                                                const double wheel_velocity[2],
                                                dex_error_t *error);
/* Model-sized feedback: kind 0=steering, 1=wheel velocity, 2=encoder position. */
DEX_NODISCARD dex_status_t dex_robot_chassis_values(dex_robot_t *robot, uint32_t kind, double *out,
                                      size_t capacity, size_t *count, dex_error_t *error);
DEX_NODISCARD dex_status_t dex_robot_chassis_set_motion_state_n(dex_robot_t *robot,
    const double *steering, size_t steering_count, const double *wheels,
    size_t wheel_count, dex_error_t *error);
/* Stream body velocity at control_hz for duration_ms, then stop. */
DEX_NODISCARD dex_status_t dex_robot_chassis_drive_for(dex_robot_t *robot, double vx, double vy,
                                         double wz, uint64_t duration_ms,
                                         double control_hz, dex_error_t *error);
DEX_NODISCARD dex_status_t dex_robot_chassis_set_velocity_opt(dex_robot_t *robot, double vx, double vy,
                                                double wz, const dex_chassis_options_t *options,
                                                dex_error_t *error);

/* Named base APIs: null name selects the unique enabled base; ambiguous lookup errors. */
DEX_NODISCARD dex_status_t dex_robot_chassis_set_velocity_named(dex_robot_t *robot, const char *name_utf8,
                                              double vx,
                                              double vy,
                                              double wz,
                                              dex_error_t *error);
DEX_NODISCARD dex_status_t dex_robot_chassis_stop_named(dex_robot_t *robot, const char *name_utf8, dex_error_t *error);
DEX_NODISCARD dex_status_t dex_robot_chassis_values_named(dex_robot_t *robot, const char *name_utf8, uint32_t kind, double *out,
                                      size_t capacity, size_t *count, dex_error_t *error);
DEX_NODISCARD dex_status_t dex_robot_chassis_set_motion_state_n_named(dex_robot_t *robot, const char *name_utf8,
    const double *steering, size_t steering_count, const double *wheels,
    size_t wheel_count, dex_error_t *error);
DEX_NODISCARD dex_status_t dex_robot_chassis_drive_for_named(dex_robot_t *robot, const char *name_utf8, double vx, double vy,
                                         double wz, uint64_t duration_ms,
                                         double control_hz, dex_error_t *error);
DEX_NODISCARD dex_status_t dex_robot_chassis_set_velocity_opt_named(dex_robot_t *robot, const char *name_utf8, double vx, double vy,
                                                double wz, const dex_chassis_options_t *options,
                                                dex_error_t *error);

/* ---- Odometry and closed-loop base motions (ABI 5, additive) ----------
 * Optional per robot model: dex_robot_odometry_supported reports whether the
 * chassis driver contributes a wheel model. Translation comes from the wheel
 * encoders through the chassis' forward kinematics, heading from an enabled
 * base-mounted IMU ("chassis_imu") when there is one, else from the wheels.
 * The pose is dead reckoning relative to the last reset: for turn-by and
 * move-by, not for localisation. */
typedef struct dex_odometry_options_t {
    uint32_t struct_size;
    double sample_hz;          /* integration rate; 0 selects the default (100) */
    double encoder_scale[2];   /* metres per encoder unit, left then right; 0 selects 1 */
    double max_sample_travel;  /* m; an encoder jump above it is a state gap (default 0.5) */
    double slip_rate_threshold; /* rad/s wheel-vs-gyro disagreement counted as slip (default 0.3) */
} dex_odometry_options_t;
void dex_odometry_options_init(dex_odometry_options_t *options);
typedef struct dex_pose2d_t {
    uint32_t struct_size;
    double x;   /* m */
    double y;   /* m */
    double yaw; /* rad, counter-clockwise, unwrapped */
} dex_pose2d_t;
typedef struct dex_odometry_status_t {
    uint32_t struct_size;
    double x;
    double y;
    double yaw;
    double wheel_yaw;     /* heading from the wheels alone, for comparison */
    double gyro_bias;     /* rad/s, 0 without an IMU */
    uint64_t samples;     /* integrated wheel samples */
    uint64_t imu_samples; /* distinct IMU samples integrated */
    uint64_t gaps;        /* samples dropped by the encoder gap guard */
    uint64_t slip_samples;
    uint8_t stationary;
    uint8_t running;      /* at least one sample integrated */
    char yaw_source[64];  /* "wheels" or the IMU name, NUL-terminated */
} dex_odometry_status_t;
/* Options of the closed-loop motions; 0 selects each default: cruise speed
 * 0.5 rad/s or 0.2 m/s, tolerance 2 deg or 2 cm, 3 passes, 60 s timeout,
 * 300 ms settle. */
typedef struct dex_closed_loop_options_t {
    uint32_t struct_size;
    double speed;     /* cruise rate: rad/s for a turn, m/s for a move */
    double tolerance; /* rad for a turn, m for a move */
    uint32_t max_steps; /* passes: the motion plus correction passes */
    uint64_t timeout_ms;
    uint64_t settle_ms;
} dex_closed_loop_options_t;
void dex_closed_loop_options_init(dex_closed_loop_options_t *options);
DEX_NODISCARD dex_status_t dex_robot_odometry_supported(dex_robot_t *robot, uint8_t *out, dex_error_t *error);
/* yaw_source_utf8: null or "" for automatic, "wheels", or an enabled IMU's
 * name. options may be null. Not required: the first pose read starts the
 * odometry with defaults. */
DEX_NODISCARD dex_status_t dex_robot_odometry_start(dex_robot_t *robot, const char *yaw_source_utf8,
                                      const dex_odometry_options_t *options, dex_error_t *error);
DEX_NODISCARD dex_status_t dex_robot_odometry_stop(dex_robot_t *robot, dex_error_t *error);
DEX_NODISCARD dex_status_t dex_robot_odometry_pose(dex_robot_t *robot, dex_pose2d_t *out, dex_error_t *error);
DEX_NODISCARD dex_status_t dex_robot_odometry_reset(dex_robot_t *robot, dex_error_t *error);
DEX_NODISCARD dex_status_t dex_robot_odometry_status(dex_robot_t *robot, dex_odometry_status_t *out, dex_error_t *error);
/* Blocking: one continuous motion servoed on the odometry (the speed follows
 * the remaining amount down to a floor), then a correction pass only if the
 * settled error is outside the tolerance. Writes the amount measured. The
 * base is stopped on every exit. */
DEX_NODISCARD dex_status_t dex_robot_chassis_turn_by(dex_robot_t *robot, double angle_rad,
                                       const dex_closed_loop_options_t *options, double *out_achieved,
                                       dex_error_t *error);
DEX_NODISCARD dex_status_t dex_robot_chassis_move_by(dex_robot_t *robot, double distance_m,
                                       const dex_closed_loop_options_t *options, double *out_achieved,
                                       dex_error_t *error);
DEX_NODISCARD dex_status_t dex_robot_chassis_turn_by_named(dex_robot_t *robot, const char *name_utf8, double angle_rad,
                                             const dex_closed_loop_options_t *options, double *out_achieved,
                                             dex_error_t *error);
DEX_NODISCARD dex_status_t dex_robot_chassis_move_by_named(dex_robot_t *robot, const char *name_utf8, double distance_m,
                                             const dex_closed_loop_options_t *options, double *out_achieved,
                                             dex_error_t *error);

/* ---- Component: introspection and state ------------------------------ */
DEX_NODISCARD dex_status_t dex_component_name(const dex_component_t *component, char **out_name, dex_error_t *error);
DEX_NODISCARD dex_status_t dex_component_joint_name(const dex_component_t *component, size_t index,
                                      char **out_name, dex_error_t *error);
/* lower/upper/velocity may be null; unbounded joints read as +/-INFINITY.
 * *out_has_limits is false (and nothing is written) without joint metadata. */
DEX_NODISCARD dex_status_t dex_component_joint_limits(const dex_component_t *component, double *lower,
                                        double *upper, double *velocity, size_t capacity,
                                        size_t *out_count, bool *out_has_limits,
                                        dex_error_t *error);
DEX_NODISCARD dex_status_t dex_component_pose_count(const dex_component_t *component, size_t *out_count);
DEX_NODISCARD dex_status_t dex_component_pose_name(const dex_component_t *component, size_t index,
                                     char **out_name, dex_error_t *error);
DEX_NODISCARD dex_status_t dex_component_predefined_pose(const dex_component_t *component, const char *name_utf8,
                                           double *out_values, size_t capacity,
                                           size_t *out_count, dex_error_t *error);
/* Resolves frame metadata and validates the final target; sends no commands. */
DEX_NODISCARD dex_status_t dex_component_resolve_pose(const dex_component_t *component, const char *name_utf8,
                                           double *out_values, size_t capacity,
                                           size_t *out_count, dex_error_t *error);
/* Passthrough skips transformation only; the final target is always validated. */
DEX_NODISCARD dex_status_t dex_component_get_pose(const dex_component_t *component, const char *name_utf8,
                                    bool passthrough, double *out_values, size_t capacity,
                                    size_t *out_count, dex_error_t *error);
/* position/velocity/torque/current may each be null. ONE count for four
 * channels: a channel the component does not publish (many publish no torque
 * or current) is zero-filled to *out_count, which reads like real zeros, and
 * there is no freshness bound. Prefer dex_component_get_joint_state_ex(). */
DEX_NODISCARD dex_status_t dex_component_get_joint_state(const dex_component_t *component, double *position,
                                           double *velocity, double *torque, double *current,
                                           size_t capacity, size_t *out_count,
                                           uint64_t *out_timestamp_ns, dex_error_t *error);
/* Elements written per channel: its real length, 0 when not published. */
typedef struct dex_joint_state_counts_t {
    uint32_t struct_size;
    size_t position, velocity, torque, current;
} dex_joint_state_counts_t;
/* The latest joint state with a freshness bound, per-channel lengths and the
 * sample's LOCAL ARRIVAL AGE (out_timestamp_ns is the producer's clock and
 * says nothing about delivery). max_state_age_ms: DEX_STATE_AGE_CONFIGURED
 * (0) applies the component's configured limit, DEX_LIMIT_DISABLED none; an
 * older sample fails with DEX_STALE_STATE. A bounded read also rejects a
 * non-finite position or velocity. *out_age_ms is written whenever a sample
 * exists, stale or not. Every array may be null; a non-null one must hold
 * `capacity` elements and receives exactly counts-><channel> of them, so an
 * absent channel is left untouched. counts, out_timestamp_ns and out_age_ms
 * may be null; counts must have struct_size set. */
DEX_NODISCARD dex_status_t dex_component_get_joint_state_ex(const dex_component_t *component,
                                              uint64_t max_state_age_ms, double *position,
                                              double *velocity, double *torque, double *current,
                                              size_t capacity, dex_joint_state_counts_t *counts,
                                              uint64_t *out_timestamp_ns, uint64_t *out_age_ms,
                                              dex_error_t *error);
DEX_NODISCARD dex_status_t dex_component_get_joint_err(const dex_component_t *component, uint32_t *out_codes,
                                         size_t capacity, size_t *out_count, dex_error_t *error);
DEX_NODISCARD dex_status_t dex_component_is_joint_pos_reached(const dex_component_t *component,
                                                const double *target, size_t count,
                                                double tolerance, bool *out_reached,
                                                dex_error_t *error);

/* ---- Component: commands --------------------------------------------- */
void dex_command_options_init(dex_command_options_t *options);
/* velocities, options and wait may each be null. */
DEX_NODISCARD dex_status_t dex_component_set_joint_pos_opt(const dex_component_t *component,
                                             const double *values, const double *velocities,
                                             size_t count, const dex_command_options_t *options,
                                             const dex_wait_policy_t *wait, dex_error_t *error);
DEX_NODISCARD dex_status_t dex_component_move_to_joint_pos_opt(const dex_component_t *component,
                                                 const double *values, size_t count,
                                                 const dex_motion_options_t *options,
                                                 const dex_wait_policy_t *wait,
                                                 dex_motion_t **out_motion, dex_error_t *error);
/* Streams an untracked goal to the motion plugin (motion_id 0): call it from
 * a control loop, once per tick. The plugin smooths towards the newest goal
 * within its own velocity/acceleration/jerk bounds, so no step guard
 * applies and nothing is returned to wait on. DEX_INVALID_ARGUMENT while a
 * tracked motion of the component is still running. options may be null;
 * wait_ceiling_ms is ignored. */
DEX_NODISCARD dex_status_t dex_component_stream_joint_pos(const dex_component_t *component,
                                                          const double *values, size_t count,
                                                          const dex_motion_options_t *options,
                                                          dex_error_t *error);
DEX_NODISCARD dex_status_t dex_component_go_to_pose(const dex_component_t *component, const char *pose_utf8,
                                      const dex_motion_options_t *options,
                                      const dex_wait_policy_t *wait, dex_motion_t **out_motion,
                                      dex_error_t *error);
DEX_NODISCARD dex_status_t dex_component_go_to_pose_opt(const dex_component_t *component, const char *pose_utf8,
                                          const dex_motion_options_t *options,
                                          const dex_wait_policy_t *wait, bool passthrough,
                                          dex_motion_t **out_motion, dex_error_t *error);

/* positions holds point_count * joint_count values, point-major; times_s
 * one time-from-start per point. velocity_scale 0 leaves the default. */
DEX_NODISCARD dex_status_t dex_component_move_joint_trajectory(const dex_component_t *component,
                                                 const double *positions, const double *times_s,
                                                 size_t point_count, size_t joint_count,
                                                 double velocity_scale,
                                                 const dex_wait_policy_t *wait,
                                                 dex_motion_t **out_motion, dex_error_t *error);
/* As above with dex_motion_options_t (NULL for the defaults):
 * velocity_scale, disable_limit_enforcement and wait_ceiling_ms apply; a
 * trajectory is absolute, so a non-zero `relative` is rejected. */
DEX_NODISCARD dex_status_t dex_component_move_joint_trajectory_opt(const dex_component_t *component,
                                                     const double *positions,
                                                     const double *times_s, size_t point_count,
                                                     size_t joint_count,
                                                     const dex_motion_options_t *options,
                                                     const dex_wait_policy_t *wait,
                                                     dex_motion_t **out_motion,
                                                     dex_error_t *error);
DEX_NODISCARD dex_status_t dex_component_stop(const dex_component_t *component, dex_error_t *error);

/* ---- Component: firmware services ----------------------------------- */
#define DEX_JOINT_MODE_DISABLE ((int32_t)1)
#define DEX_JOINT_MODE_ENABLE ((int32_t)2)
#define DEX_JOINT_MODE_CALIBRATION ((int32_t)3)
#define DEX_JOINT_MODE_POSITION ((int32_t)4)
#define DEX_JOINT_MODE_VELOCITY ((int32_t)5)
#define DEX_JOINT_MODE_TORQUE ((int32_t)6)
#define DEX_JOINT_MODE_CURRENT ((int32_t)7)
/* Stable DEX_JOINT_MODE_* API values, translated by the selected driver.
 * get_modes returns 0 for a device mode without a corresponding API value. */
DEX_NODISCARD dex_status_t dex_component_set_modes(const dex_component_t *component, const int32_t *codes,
                                     size_t count, char **out_json, dex_error_t *error);
DEX_NODISCARD dex_status_t dex_component_get_modes(const dex_component_t *component, int32_t *out_codes,
                                     size_t capacity, size_t *out_count, dex_error_t *error);
/* Generic escape hatch: a configured service role with a JSON request. */
DEX_NODISCARD dex_status_t dex_component_request_json(const dex_component_t *component, const char *role_utf8,
                                        const char *request_json_utf8, char **out_json,
                                        dex_error_t *error);
/* Every p must be in [0.1, 4.0]. */
DEX_NODISCARD dex_status_t dex_component_set_pid(const dex_component_t *component, const double *p, size_t count,
                                   char **out_json, dex_error_t *error);
DEX_NODISCARD dex_status_t dex_component_get_pid(const dex_component_t *component, char **out_json,
                                   dex_error_t *error);
/* Releases (true) or engages the brakes of exactly the listed joints. An
 * empty list (NULL or joint_count == 0) is DEX_INVALID_ARGUMENT, never
 * "every joint": a caller that filters a joint list down to nothing must not
 * drop the whole arm. */
DEX_NODISCARD dex_status_t dex_component_release_brake(const dex_component_t *component, bool enable,
                                         const size_t *joints, size_t joint_count,
                                         char **out_json, dex_error_t *error);
/* Releases (true) or engages the brake of EVERY joint of the component.
 * Releasing lets an unsupported arm fall. */
DEX_NODISCARD dex_status_t dex_component_release_all_brakes(const dex_component_t *component, bool enable,
                                              char **out_json, dex_error_t *error);
DEX_NODISCARD dex_status_t dex_component_brake_status(const dex_component_t *component, char **out_json,
                                        dex_error_t *error);
DEX_NODISCARD dex_status_t dex_component_set_force_torque_sensor(const dex_component_t *component, bool enable,
                                                   char **out_json, dex_error_t *error);
DEX_NODISCARD dex_status_t dex_component_force_torque_sensor_mode(const dex_component_t *component,
                                                    char **out_json, dex_error_t *error);
DEX_NODISCARD dex_status_t dex_component_set_ee_baud_rate(const dex_component_t *component, uint32_t baud_rate,
                                            char **out_json, dex_error_t *error);
DEX_NODISCARD dex_status_t dex_component_ee_baud_rate(const dex_component_t *component, char **out_json,
                                        dex_error_t *error);
DEX_NODISCARD dex_status_t dex_component_set_idle_mode(const dex_component_t *component, bool enabled,
                                         dex_error_t *error);
/* *out_enabled: 1 enabled, 0 disabled, -1 not reported. */
DEX_NODISCARD dex_status_t dex_component_idle_mode(const dex_component_t *component, int32_t *out_enabled,
                                     dex_error_t *error);

/* ---- Component: end effector, buttons, touch ------------------------- */
DEX_NODISCARD dex_status_t dex_component_ee_pass_through_enabled(const dex_component_t *component, bool *out_enabled);
DEX_NODISCARD dex_status_t dex_component_ee_pass_through_send(const dex_component_t *component, const uint8_t *data,
                                                size_t len, dex_error_t *error);
DEX_NODISCARD dex_status_t dex_component_ee_pass_through_latest(const dex_component_t *component,
                                                  dex_buffer_t *out_response, bool *out_has_response,
                                                  dex_error_t *error);
DEX_NODISCARD dex_status_t dex_component_button_state(const dex_component_t *component, bool *out_blue,
                                        bool *out_green, bool *out_has_state, dex_error_t *error);
DEX_NODISCARD dex_status_t dex_component_touch_forces(const dex_component_t *component, double *out_forces,
                                        size_t capacity, size_t *out_count, bool *out_has_data,
                                        dex_error_t *error);

/* ---- Sensors ----------------------------------------------------------
 * The plain reads carry no freshness guarantee: a stream that stopped
 * publishing keeps returning its last sample, and timestamp_ns is the
 * producer's clock. The _ex reads take max_age_ms (0 or DEX_LIMIT_DISABLED:
 * no bound; otherwise an older sample fails with DEX_STALE_STATE) and write
 * the sample's local arrival age to *out_age_ms (may be null) whenever a
 * sample exists, stale or not. */
typedef struct dex_imu_t {
    uint32_t struct_size;
    double acceleration[3];     /* m/s^2 */
    double angular_velocity[3]; /* rad/s */
    double orientation[4];      /* quaternion as published */
    double magnetic_field[3];
    bool has_magnetic_field;
    uint64_t timestamp_ns;
} dex_imu_t;
DEX_NODISCARD dex_status_t dex_robot_imu(dex_robot_t *robot, const char *name_utf8, dex_imu_t *out,
                           dex_error_t *error);
DEX_NODISCARD dex_status_t dex_robot_imu_ex(dex_robot_t *robot, const char *name_utf8, uint64_t max_age_ms,
                              dex_imu_t *out, uint64_t *out_age_ms, dex_error_t *error);

typedef struct dex_ultrasonic_t {
    uint32_t struct_size;
    double front_left, front_right, back_left, back_right; /* m */
    uint64_t timestamp_ns;
} dex_ultrasonic_t;
DEX_NODISCARD dex_status_t dex_robot_ultrasonic(dex_robot_t *robot, const char *name_utf8,
                                  dex_ultrasonic_t *out, dex_error_t *error);
DEX_NODISCARD dex_status_t dex_robot_ultrasonic_ex(dex_robot_t *robot, const char *name_utf8,
                                     uint64_t max_age_ms, dex_ultrasonic_t *out,
                                     uint64_t *out_age_ms, dex_error_t *error);

typedef struct dex_lidar_2d_info_t {
    uint32_t struct_size;
    double angle_min, angle_max, angle_increment, time_increment, scan_time;
    double range_min, range_max;
    uint64_t timestamp_ns;
} dex_lidar_2d_info_t;
/* Copying read; ranges/angles/intensities may be null; info may be null.
 * Prefer dex_robot_lidar_2d_take(). This call reads a fresh sample each
 * time, so a probe followed by a fill can see a larger scan and fail with
 * DEX_INVALID_ARGUMENT on a healthy sensor: retry with the reported count.
 * It also reports ONE count: a channel the sensor does not publish is
 * zero-filled to *out_count. */
DEX_NODISCARD dex_status_t dex_robot_lidar_2d(dex_robot_t *robot, const char *name_utf8, double *ranges,
                                double *angles, uint32_t *intensities, size_t capacity,
                                size_t *out_count, dex_lidar_2d_info_t *info, dex_error_t *error);

typedef struct dex_lidar_3d_info_t {
    uint32_t struct_size;
    uint32_t height, width;
    bool is_dense;
    uint32_t point_count;
    uint64_t timestamp_ns;
} dex_lidar_3d_info_t;
/* Copying read; every channel may be null; info may be null. Prefer
 * dex_robot_lidar_3d_take(), for the reasons given on dex_robot_lidar_2d(). */
DEX_NODISCARD dex_status_t dex_robot_lidar_3d(dex_robot_t *robot, const char *name_utf8, double *x, double *y,
                                double *z, uint32_t *intensity, uint32_t *ring,
                                uint32_t *point_timestamps_ns, size_t capacity, size_t *out_count,
                                dex_lidar_3d_info_t *info, dex_error_t *error);

/* Lidar snapshots: one owned sample. Channels are borrowed from it without
 * copying, each with its OWN length, and stay valid until the snapshot is
 * released; a channel the sensor does not publish is (NULL, 0). max_age_ms
 * as for the _ex sensor reads. */
typedef struct dex_lidar_2d_t dex_lidar_2d_t;
typedef struct dex_lidar_3d_t dex_lidar_3d_t;
#define DEX_LIDAR_2D_RANGES ((uint32_t)0)      /* f64 */
#define DEX_LIDAR_2D_ANGLES ((uint32_t)1)      /* f64 */
#define DEX_LIDAR_2D_INTENSITIES ((uint32_t)0) /* u32 */
#define DEX_LIDAR_3D_X ((uint32_t)0)                /* f64 */
#define DEX_LIDAR_3D_Y ((uint32_t)1)                /* f64 */
#define DEX_LIDAR_3D_Z ((uint32_t)2)                /* f64 */
#define DEX_LIDAR_3D_INTENSITY ((uint32_t)0)        /* u32 */
#define DEX_LIDAR_3D_RING ((uint32_t)1)             /* u32 */
#define DEX_LIDAR_3D_POINT_TIMESTAMPS ((uint32_t)2) /* u32 */
DEX_NODISCARD dex_status_t dex_robot_lidar_2d_take(dex_robot_t *robot, const char *name_utf8,
                                     uint64_t max_age_ms, dex_lidar_2d_t **out_scan,
                                     dex_error_t *error);
DEX_NODISCARD dex_status_t dex_lidar_2d_info(const dex_lidar_2d_t *scan, dex_lidar_2d_info_t *out,
                               dex_error_t *error);
DEX_NODISCARD dex_status_t dex_lidar_2d_channel_f64(const dex_lidar_2d_t *scan, uint32_t channel,
                                      const double **out_data, size_t *out_count);
DEX_NODISCARD dex_status_t dex_lidar_2d_channel_u32(const dex_lidar_2d_t *scan, uint32_t channel,
                                      const uint32_t **out_data, size_t *out_count);
/* Local arrival age at the moment the snapshot was taken. */
DEX_NODISCARD dex_status_t dex_lidar_2d_age_ms(const dex_lidar_2d_t *scan, uint64_t *out_age_ms);
void dex_lidar_2d_release(dex_lidar_2d_t *scan);
DEX_NODISCARD dex_status_t dex_robot_lidar_3d_take(dex_robot_t *robot, const char *name_utf8,
                                     uint64_t max_age_ms, dex_lidar_3d_t **out_cloud,
                                     dex_error_t *error);
DEX_NODISCARD dex_status_t dex_lidar_3d_info(const dex_lidar_3d_t *cloud, dex_lidar_3d_info_t *out,
                               dex_error_t *error);
DEX_NODISCARD dex_status_t dex_lidar_3d_channel_f64(const dex_lidar_3d_t *cloud, uint32_t channel,
                                      const double **out_data, size_t *out_count);
DEX_NODISCARD dex_status_t dex_lidar_3d_channel_u32(const dex_lidar_3d_t *cloud, uint32_t channel,
                                      const uint32_t **out_data, size_t *out_count);
DEX_NODISCARD dex_status_t dex_lidar_3d_age_ms(const dex_lidar_3d_t *cloud, uint64_t *out_age_ms);
void dex_lidar_3d_release(dex_lidar_3d_t *cloud);

/* Force (N) then torque (N*m): six values. */
DEX_NODISCARD dex_status_t dex_robot_wrench(dex_robot_t *robot, const char *component_utf8, double out_wrench[6],
                              uint64_t *out_timestamp_ns, dex_error_t *error);
DEX_NODISCARD dex_status_t dex_robot_wrench_ex(dex_robot_t *robot, const char *component_utf8,
                                 uint64_t max_age_ms, double out_wrench[6],
                                 uint64_t *out_timestamp_ns, uint64_t *out_age_ms,
                                 dex_error_t *error);
DEX_NODISCARD dex_status_t dex_robot_temperature_json(dex_robot_t *robot, const char *component_utf8,
                                        char **out_json, dex_error_t *error);
/* Canonical decoded JSON of any sensor's endpoint; role may be null. */
DEX_NODISCARD dex_status_t dex_robot_sensor_json(dex_robot_t *robot, const char *name_utf8, const char *role_utf8,
                                   char **out_json, dex_error_t *error);

/* ---- Cameras ---------------------------------------------------------- */
#define DEX_FRAME_RGB8 ((uint32_t)0)
#define DEX_FRAME_BGR8 ((uint32_t)1)
#define DEX_FRAME_GRAY8 ((uint32_t)2)
#define DEX_FRAME_DEPTH32F ((uint32_t)3) /* little-endian float meters */
typedef struct dex_frame_info_t {
    uint32_t struct_size;
    uint32_t width, height, encoding;
    uint64_t timestamp_ns;
    size_t data_len;
    float depth_scale, depth_min_range, depth_max_range;
} dex_frame_info_t;
typedef struct dex_stream_stats_t {
    uint32_t struct_size;
    uint64_t frames_received, frames_replaced_unread, decode_errors;
} dex_stream_stats_t;
/* Camera information JSON; no image subscription required. timeout_ms > 0.
 * Fields depend on the device/server; release *out_json with dex_string_free. */
DEX_NODISCARD dex_status_t dex_robot_camera_info_json(dex_robot_t *robot, const char *sensor_utf8,
                                                     uint64_t timeout_ms, char **out_json,
                                                     dex_error_t *error);
DEX_NODISCARD dex_status_t dex_robot_camera_stream_count(dex_robot_t *robot, const char *sensor_utf8,
                                           size_t *out_count, dex_error_t *error);
DEX_NODISCARD dex_status_t dex_robot_camera_stream_name(dex_robot_t *robot, const char *sensor_utf8, size_t index,
                                          char **out_name, dex_error_t *error);
/* history 0 keeps the newest frame only; n keeps the n newest. */
DEX_NODISCARD dex_status_t dex_robot_camera_subscribe(dex_robot_t *robot, const char *sensor_utf8,
                                        const char *stream_utf8, size_t history, dex_error_t *error);
DEX_NODISCARD dex_status_t dex_robot_camera_unsubscribe(dex_robot_t *robot, const char *sensor_utf8,
                                          const char *stream_utf8, dex_error_t *error);
/* *out_frame is null before the first frame. Release with dex_frame_release. */
DEX_NODISCARD dex_status_t dex_robot_camera_latest_frame(dex_robot_t *robot, const char *sensor_utf8,
                                           const char *stream_utf8, dex_frame_t **out_frame,
                                           dex_error_t *error);
DEX_NODISCARD dex_status_t dex_robot_camera_stats(dex_robot_t *robot, const char *sensor_utf8,
                                    const char *stream_utf8, dex_stream_stats_t *out,
                                    dex_error_t *error);
/* stream may be null to ask about any configured stream. */
DEX_NODISCARD dex_status_t dex_robot_camera_is_active(dex_robot_t *robot, const char *sensor_utf8,
                                        const char *stream_utf8, uint64_t window_ms,
                                        bool *out_active, dex_error_t *error);
DEX_NODISCARD dex_status_t dex_frame_info(const dex_frame_t *frame, dex_frame_info_t *out, dex_error_t *error);
/* Borrowed pixels, valid until dex_frame_release(); never copied. */
DEX_NODISCARD dex_status_t dex_frame_data(const dex_frame_t *frame, const uint8_t **out_data, size_t *out_len);
void dex_frame_release(dex_frame_t *frame);

/* ---- Diagnostic client: typed sensor reads ----------------------------
 * The dex_robot_* sensor reads below, on the read-only client. Same
 * arguments, same outputs, same freshness semantics; added within ABI 5. */
DEX_NODISCARD dex_status_t dex_diagnostic_battery_ex(dex_diagnostic_t *client, uint64_t max_age_ms,
                                       dex_battery_t *out, uint64_t *out_age_ms, dex_error_t *error);
DEX_NODISCARD dex_status_t dex_diagnostic_imu_ex(dex_diagnostic_t *client, const char *name_utf8,
                                   uint64_t max_age_ms, dex_imu_t *out, uint64_t *out_age_ms,
                                   dex_error_t *error);
DEX_NODISCARD dex_status_t dex_diagnostic_ultrasonic_ex(dex_diagnostic_t *client, const char *name_utf8,
                                          uint64_t max_age_ms, dex_ultrasonic_t *out,
                                          uint64_t *out_age_ms, dex_error_t *error);
DEX_NODISCARD dex_status_t dex_diagnostic_lidar_2d_take(dex_diagnostic_t *client, const char *name_utf8,
                                          uint64_t max_age_ms, dex_lidar_2d_t **out_scan,
                                          dex_error_t *error);
DEX_NODISCARD dex_status_t dex_diagnostic_lidar_3d_take(dex_diagnostic_t *client, const char *name_utf8,
                                          uint64_t max_age_ms, dex_lidar_3d_t **out_cloud,
                                          dex_error_t *error);
DEX_NODISCARD dex_status_t dex_diagnostic_wrench_ex(dex_diagnostic_t *client, const char *component_utf8,
                                      uint64_t max_age_ms, double out_wrench[6],
                                      uint64_t *out_timestamp_ns, uint64_t *out_age_ms,
                                      dex_error_t *error);
DEX_NODISCARD dex_status_t dex_diagnostic_temperature_json(dex_diagnostic_t *client, const char *component_utf8,
                                             char **out_json, dex_error_t *error);
DEX_NODISCARD dex_status_t dex_diagnostic_sensor_json(dex_diagnostic_t *client, const char *name_utf8,
                                        const char *role_utf8, char **out_json, dex_error_t *error);
DEX_NODISCARD dex_status_t dex_diagnostic_camera_info_json(dex_diagnostic_t *client, const char *sensor_utf8,
                                             uint64_t timeout_ms, char **out_json, dex_error_t *error);
DEX_NODISCARD dex_status_t dex_diagnostic_camera_stream_count(dex_diagnostic_t *client, const char *sensor_utf8,
                                                size_t *out_count, dex_error_t *error);
DEX_NODISCARD dex_status_t dex_diagnostic_camera_stream_name(dex_diagnostic_t *client, const char *sensor_utf8,
                                               size_t index, char **out_name, dex_error_t *error);
DEX_NODISCARD dex_status_t dex_diagnostic_camera_subscribe(dex_diagnostic_t *client, const char *sensor_utf8,
                                             const char *stream_utf8, size_t history, dex_error_t *error);
DEX_NODISCARD dex_status_t dex_diagnostic_camera_unsubscribe(dex_diagnostic_t *client, const char *sensor_utf8,
                                               const char *stream_utf8, dex_error_t *error);
DEX_NODISCARD dex_status_t dex_diagnostic_camera_latest_frame(dex_diagnostic_t *client, const char *sensor_utf8,
                                                const char *stream_utf8, dex_frame_t **out_frame,
                                                dex_error_t *error);
DEX_NODISCARD dex_status_t dex_diagnostic_camera_stats(dex_diagnostic_t *client, const char *sensor_utf8,
                                         const char *stream_utf8, dex_stream_stats_t *out,
                                         dex_error_t *error);
DEX_NODISCARD dex_status_t dex_diagnostic_camera_is_active(dex_diagnostic_t *client, const char *sensor_utf8,
                                             const char *stream_utf8, uint64_t window_ms,
                                             bool *out_active, dex_error_t *error);

/* ---- Motion groups and motion extras --------------------------------- */
DEX_NODISCARD dex_status_t dex_motion_group_len(const dex_motion_group_t *group, size_t *out_len);
DEX_NODISCARD dex_status_t dex_motion_group_member(const dex_motion_group_t *group, size_t index,
                                     dex_motion_t **out_motion, dex_error_t *error);
/* Waits for every member; *out_state is the worst member outcome. A null
 * wait means until complete. */
DEX_NODISCARD dex_status_t dex_motion_group_wait(const dex_motion_group_t *group, const dex_wait_policy_t *wait,
                                   uint32_t *out_state, dex_error_t *error);
DEX_NODISCARD dex_status_t dex_motion_group_cancel(const dex_motion_group_t *group, dex_error_t *error);
DEX_NODISCARD dex_status_t dex_motion_group_state(const dex_motion_group_t *group, uint32_t *out_state,
                                    dex_error_t *error);
void dex_motion_group_release(dex_motion_group_t *group);
/* Refreshes from the status stream; a broken stream is an error rather
 * than a stale state (dex_motion_state reads the cache). */
DEX_NODISCARD dex_status_t dex_motion_refresh(const dex_motion_t *motion, uint32_t *out_state, dex_error_t *error);
DEX_NODISCARD dex_status_t dex_motion_message(const dex_motion_t *motion, char **out_message, dex_error_t *error);

/* ---- Rate limiter ----------------------------------------------------- */
#define DEX_RATE_LIMITER_MAX_WINDOW ((size_t)100000)
typedef struct dex_rate_limiter_stats_t {
    uint32_t struct_size;
    double target_rate_hz;
    double actual_rate_hz;  /* over the moving-average window */
    double average_rate_hz; /* since creation */
    uint64_t iterations;       /* sleeps since the schedule was last anchored */
    uint64_t missed_deadlines; /* sleeps whose deadline had already passed */
} dex_rate_limiter_stats_t;
/* window_size sizes the moving average: 0 selects 100; more than
 * DEX_RATE_LIMITER_MAX_WINDOW is DEX_INVALID_ARGUMENT. The schedule is
 * anchored by the FIRST dex_rate_limiter_sleep(), not by this call, so setup
 * done between creating the pacer and entering the loop is never counted as
 * a missed deadline. */
DEX_NODISCARD dex_status_t dex_rate_limiter_create(double rate_hz, size_t window_size, bool adaptive,
                                     dex_rate_limiter_t **out_limiter, dex_error_t *error);
DEX_NODISCARD dex_status_t dex_rate_limiter_sleep(const dex_rate_limiter_t *limiter);
/* Re-anchors the schedule at "now" and clears the counters and the window:
 * call after a pause in the loop (a blocking move, a reconnect). */
DEX_NODISCARD dex_status_t dex_rate_limiter_reset(const dex_rate_limiter_t *limiter);
/* out must have struct_size set. */
DEX_NODISCARD dex_status_t dex_rate_limiter_stats(const dex_rate_limiter_t *limiter,
                                    dex_rate_limiter_stats_t *out);
DEX_NODISCARD dex_status_t dex_rate_limiter_done(const dex_rate_limiter_t *limiter);
DEX_NODISCARD dex_status_t dex_rate_limiter_actual_rate(const dex_rate_limiter_t *limiter, double *out_rate);
DEX_NODISCARD dex_status_t dex_rate_limiter_average_rate(const dex_rate_limiter_t *limiter, double *out_rate);
void dex_rate_limiter_release(dex_rate_limiter_t *limiter);

#ifdef __cplusplus
}
#endif
#endif
