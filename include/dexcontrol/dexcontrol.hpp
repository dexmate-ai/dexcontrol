// Copyright (C) 2026 Dexmate Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

/* Header-only C++17 wrapper over the DexControl C ABI.
 *
 * This file adds RAII, exceptions and standard containers to dexcontrol.h
 * and nothing else: every method is one C call plus conversion. All robot
 * logic -- validation, guards, safety, decoding -- lives in the native
 * library, so a robot driven from here behaves exactly like one driven from
 * Rust or Python.
 *
 * Structured results that are not on a control loop's hot path (health,
 * firmware replies, temperatures, diagnostics) are returned as JSON text;
 * parse them with whatever JSON library your application already uses.
 */

#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <map>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "dexcontrol.h"

namespace dexcontrol {

/* Every failure of the native library. status() is the DEX_* code, so a
 * control loop branches on it rather than on message text:
 * DEX_STALE_STATE / DEX_STATE_UNAVAILABLE (retry), DEX_ESTOP_ACTIVE (release
 * the E-stop), DEX_STOPPED (motion was stopped under a blocking call),
 * DEX_TIMEOUT, DEX_INVALID_ARGUMENT (the call was wrong). */
class Error : public std::runtime_error {
public:
    Error(dex_status_t status, const char *message)
        : std::runtime_error(message), status_(status) {}
    [[nodiscard]] dex_status_t status() const noexcept { return status_; }
private:
    dex_status_t status_;
};

inline void check(dex_status_t status, const dex_error_t &error) {
    if (status != DEX_OK) {
        throw Error(status, error.message);
    }
}

struct NoWait {};
struct WaitUntilComplete {};
struct WaitFor {
    std::chrono::milliseconds timeout;
};

/* Typed view of the DEX_MOTION_* wire constants. */
enum class MotionState : uint32_t {
    Pending = DEX_MOTION_PENDING,
    Running = DEX_MOTION_RUNNING,
    Succeeded = DEX_MOTION_SUCCEEDED,
    Cancelled = DEX_MOTION_CANCELLED,
    Failed = DEX_MOTION_FAILED,
    Superseded = DEX_MOTION_SUPERSEDED,
};

inline const char *to_string(MotionState state) {
    switch (state) {
    case MotionState::Pending: return "pending";
    case MotionState::Running: return "running";
    case MotionState::Succeeded: return "succeeded";
    case MotionState::Cancelled: return "cancelled";
    case MotionState::Failed: return "failed";
    case MotionState::Superseded: return "superseded";
    }
    return "unknown";
}

/* Typed view of the DEX_JOINT_MODE_* firmware codes. */
enum class JointMode : int32_t {
    Disable = DEX_JOINT_MODE_DISABLE,
    Enable = DEX_JOINT_MODE_ENABLE,
    Calibration = DEX_JOINT_MODE_CALIBRATION,
    Position = DEX_JOINT_MODE_POSITION,
    Velocity = DEX_JOINT_MODE_VELOCITY,
    Torque = DEX_JOINT_MODE_TORQUE,
    Current = DEX_JOINT_MODE_CURRENT,
};

/* Typed view of the DEX_FRAME_* pixel encodings. */
enum class FrameEncoding : uint32_t {
    Rgb8 = DEX_FRAME_RGB8,
    Bgr8 = DEX_FRAME_BGR8,
    Gray8 = DEX_FRAME_GRAY8,
    Depth32F = DEX_FRAME_DEPTH32F,
};

namespace detail {

/* Milliseconds for the C ABI; a negative duration is a caller bug, not a
 * huge unsigned timeout. */
inline uint64_t millis(std::chrono::milliseconds value) {
    if (value.count() < 0) throw std::invalid_argument("duration must not be negative");
    return static_cast<uint64_t>(value.count());
}

inline dex_error_t error() {
    dex_error_t value{};
    dex_error_init(&value);
    return value;
}

/* The header this file was compiled against and the library it is linked to
 * must speak the same ABI: struct layouts differ between major versions, so
 * a mismatch is refused once, loudly, before the first handle exists. */
inline void require_abi() {
    static const uint32_t linked = dex_abi_version();
    if (linked != DEX_ABI_VERSION) {
        throw Error(DEX_RUNTIME_ERROR,
                    ("dexcontrol ABI mismatch: headers are ABI " +
                     std::to_string(DEX_ABI_VERSION) + ", the linked library is ABI " +
                     std::to_string(linked) + "; rebuild against the installed SDK")
                        .c_str());
    }
}

/* Arrival age as the C ABI reports it. */
inline std::chrono::milliseconds age(uint64_t millis) {
    constexpr auto limit = static_cast<uint64_t>(std::chrono::milliseconds::max().count());
    return std::chrono::milliseconds{
        static_cast<std::chrono::milliseconds::rep>(millis < limit ? millis : limit)};
}

template <typename T> T sized() {
    T value{};
    value.struct_size = sizeof(T);
    return value;
}

/* Runs a call that hands back an owned C string and returns it as
 * std::string, releasing the C allocation. */
template <typename Call> std::string owned_string(Call &&call) {
    auto error = detail::error();
    char *text = nullptr;
    struct Owner {
        char *&text;
        ~Owner() { dex_string_free(text); }
    } owner{text};
    check(call(&text, &error), error);
    return text == nullptr ? std::string{} : std::string{text};
}

/* Runs a call that hands back an owned byte buffer. */
template <typename Call> std::vector<uint8_t> owned_bytes(Call &&call) {
    auto error = detail::error();
    dex_buffer_t buffer{nullptr, 0};
    struct Owner {
        dex_buffer_t &buffer;
        ~Owner() { dex_buffer_free(&buffer); }
    } owner{buffer};
    check(call(&buffer, &error), error);
    if (buffer.len == 0) return {};
    if (buffer.data == nullptr) throw std::runtime_error("null native buffer with nonzero length");
    return {buffer.data, buffer.data + buffer.len};
}

/* Runs a caller-sized array read in two phases: probe the required count,
 * then fill. `call(out, capacity, out_count, error)` must follow the header
 * convention of always writing the required count. */
template <typename T, typename Call> std::vector<T> read_array(Call &&call) {
    auto error = detail::error();
    size_t count = 0;
    const auto probe = call(static_cast<T *>(nullptr), 0, &count, &error);
    if (probe == DEX_OK) {
        return {};
    }
    if (probe != DEX_INVALID_ARGUMENT || count == 0) {
        check(probe, error);
    }
    std::vector<T> values(count);
    check(call(values.data(), values.size(), &count, &error), error);
    values.resize(count);
    return values;
}

/* Enumerates names through a count/name pair of C calls. */
template <typename Count, typename Name>
std::vector<std::string> names(Count &&count_call, Name &&name_call) {
    size_t count = 0;
    const auto status = count_call(&count);
    if (status != DEX_OK) {
        throw Error(status, "count query failed (invalid handle or name)");
    }
    std::vector<std::string> names;
    names.reserve(count);
    for (size_t index = 0; index < count; ++index) {
        names.push_back(owned_string([&](char **out, dex_error_t *err) {
            return name_call(index, out, err);
        }));
    }
    return names;
}

}  // namespace detail

/* Configures the runtime used by robots created afterwards.
 * worker_threads == 0 selects the library default. */
inline void configure_runtime(uint32_t worker_threads,
                              const std::string &thread_name = {}) {
    auto options = detail::sized<dex_runtime_options_t>();
    options.worker_threads = worker_threads;
    if (!thread_name.empty()) {
        options.thread_name_utf8 = thread_name.c_str();
    }
    auto error = detail::error();
    check(dex_runtime_configure(&options, &error), error);
}

/* Restores the library-default runtime options. */
inline void reset_runtime_configuration() {
    auto error = detail::error();
    check(dex_runtime_configure(nullptr, &error), error);
}

/* Configures the dedicated-process safety watchdog for robots created
 * afterwards. An empty command keeps the automatic spawn resolution
 * (DEXCONTROL_WATCHDOG_CMD, a dexcontrol-watchdog binary next to the
 * current executable, then PATH); enabled=false opts out of process-level
 * supervision. Simulated robots never spawn a watchdog. */
inline void configure_watchdog(const std::string &command = {},
                               bool enabled = true) {
    auto error = detail::error();
    const char *command_utf8 = command.empty() ? nullptr : command.c_str();
    check(dex_watchdog_configure(command_utf8, enabled, &error), error);
}

/* Configures the safety termination policy for robots created afterwards.
 * With exit_on_termination_request true, a request_process_termination
 * safety action first makes the robot safe, emits the event, then
 * terminates this process with exit code 1 (no unwinding or cleanup).
 * Default false: the library reports and the host decides. Enable only in
 * a process that exists solely to run the robot. */
inline void configure_safety(bool exit_on_termination_request) {
    auto error = detail::error();
    check(dex_safety_configure(exit_on_termination_request, &error), error);
}

[[nodiscard]] inline dex_wait_policy_t wait_policy(NoWait) {
    dex_wait_policy_t value{};
    dex_wait_policy_init(&value);
    return value;
}
[[nodiscard]] inline dex_wait_policy_t wait_policy(WaitUntilComplete) {
    auto value = wait_policy(NoWait{});
    value.mode = DEX_WAIT_UNTIL_COMPLETE;
    return value;
}
[[nodiscard]] inline dex_wait_policy_t wait_policy(WaitFor wait) {
    if (wait.timeout.count() <= 0) {
        throw std::invalid_argument("WaitFor timeout must be positive");
    }
    auto value = wait_policy(NoWait{});
    value.mode = DEX_WAIT_WITH_TIMEOUT;
    value.timeout_ms = static_cast<uint64_t>(wait.timeout.count());
    return value;
}

/* Identity overload so an already-built (and tuned) policy can be passed
 * anywhere a wait tag is accepted. */
[[nodiscard]] inline dex_wait_policy_t wait_policy(dex_wait_policy_t value) { return value; }

/* Freshness of the measured state a direct command may be computed from.
 * dex_wait_policy_init() selects the component's configured limit (500 ms
 * unless the robot model overrides it); these helpers exist for callers on
 * unusually slow state streams, and for the deliberate opt-out.
 *
 * These take the policy by value and return the adjusted copy, matching what
 * `with_` reads as everywhere else in C++ (and what `std::filesystem`,
 * chrono, and the standard "with" idiom do). Chain them:
 *
 *     auto tuned = with_wait_ceiling(
 *         with_max_state_age(wait_policy(NoWait{}), 500ms), 5s);
 */
[[nodiscard]] inline dex_wait_policy_t with_max_state_age(dex_wait_policy_t policy,
                                            std::chrono::milliseconds age) {
    if (age.count() <= 0) {
        throw std::invalid_argument("max_state_age must be positive");
    }
    policy.max_state_age_ms = static_cast<uint64_t>(age.count());
    return policy;
}
[[nodiscard]] inline dex_wait_policy_t without_state_freshness_check(dex_wait_policy_t policy) {
    policy.max_state_age_ms = DEX_LIMIT_DISABLED;
    return policy;
}
/* Hard ceiling on an unbounded (DEX_WAIT_UNTIL_COMPLETE) convergence wait. */
[[nodiscard]] inline dex_wait_policy_t with_wait_ceiling(dex_wait_policy_t policy,
                                           std::chrono::milliseconds ceiling) {
    if (ceiling.count() <= 0) {
        throw std::invalid_argument("wait_ceiling must be positive");
    }
    policy.wait_ceiling_ms = static_cast<uint64_t>(ceiling.count());
    return policy;
}
/* Convergence tolerance (rad) and settle time for wait=true direct commands. */
[[nodiscard]] inline dex_wait_policy_t with_convergence(dex_wait_policy_t policy, double tolerance,
                                          std::chrono::milliseconds settle = {}) {
    if (!std::isfinite(tolerance) || tolerance < 0) {
        throw std::invalid_argument("convergence tolerance must be finite and non-negative");
    }
    if (settle.count() < 0) {
        throw std::invalid_argument("settle time must not be negative");
    }
    policy.convergence_tolerance = tolerance;
    policy.settle_time_ms = static_cast<uint64_t>(settle.count());
    return policy;
}

/* Direct-command options: dex_command_options_t with the library defaults. */
[[nodiscard]] inline dex_command_options_t command_options() {
    dex_command_options_t value{};
    dex_command_options_init(&value);
    return value;
}
[[nodiscard]] inline dex_command_options_t relative(dex_command_options_t options) {
    options.relative = 1;
    return options;
}
/* Expert opt-out of the model's joint-limit check. */
[[nodiscard]] inline dex_command_options_t without_limit_enforcement(dex_command_options_t options) {
    options.disable_limit_enforcement = 1;
    return options;
}
[[nodiscard]] inline dex_command_options_t with_max_step(dex_command_options_t options, double radians) {
    options.max_step_rad = radians;
    return options;
}
[[nodiscard]] inline dex_command_options_t without_step_guard(dex_command_options_t options) {
    options.max_step_rad = DEX_STEP_DISABLED;
    return options;
}

/* Planned-motion options: dex_motion_options_t with the library defaults. */
[[nodiscard]] inline dex_motion_options_t motion_options() {
    dex_motion_options_t value{};
    dex_motion_options_init(&value);
    return value;
}
[[nodiscard]] inline dex_motion_options_t relative(dex_motion_options_t options) {
    options.relative = 1;
    return options;
}
[[nodiscard]] inline dex_motion_options_t without_limit_enforcement(dex_motion_options_t options) {
    options.disable_limit_enforcement = 1;
    return options;
}

/* Host-trajectory options: dex_trajectory_options_t with the defaults. */
[[nodiscard]] inline dex_trajectory_options_t trajectory_options() {
    dex_trajectory_options_t value{};
    dex_trajectory_options_init(&value);
    return value;
}
[[nodiscard]] inline dex_motion_options_t with_velocity_scale(dex_motion_options_t options, double scale) {
    options.velocity_scale = scale;
    return options;
}

/* How to build a Robot or DiagnosticClient. Everything is optional: an
 * empty profile and config file resolve the profile from the environment
 * (ROBOT_CONFIG, then ROBOT_NAME, then vega_1). */
struct ConnectOptions {
    std::string profile;
    std::string config_file;
    std::vector<std::string> enable_sensors;
    bool simulated = false;
    /* Fail the connection when the server version cannot be verified. */
    bool require_version_check = false;
    /* Per-robot safety policy; the defaults defer to configure_watchdog() /
     * configure_safety() in Robot::connect(options) and to the library
     * default in Robot::connect(context, options). */
    bool disable_watchdog = false;
    bool exit_on_termination = false;
    std::string watchdog_command;
    /* Connect without putting the arms into position mode. The default
     * enables every arm once it is ready, and a required arm whose motor
     * cannot enable then fails the connection; set this to clear that arm's
     * error, or when the application sets modes itself. */
    bool skip_arm_initialization = false;
};

/* The built-in profile names, as JSON array text. */
[[nodiscard]] inline std::string available_profiles_json() {
    return detail::owned_string([](char **out, dex_error_t *error) {
        return dex_available_profiles_json(out, error);
    });
}

/* The built-in profile a robot identity (ROBOT_NAME style) or profile name
 * selects. */
[[nodiscard]] inline std::string profile_for_robot_name(const std::string &robot_name) {
    return detail::owned_string([&](char **out, dex_error_t *error) {
        return dex_profile_for_robot_name(robot_name.c_str(), out, error);
    });
}

/* The profile name the environment selects. */
[[nodiscard]] inline std::string profile_from_environment() {
    return detail::owned_string([](char **out, dex_error_t *error) {
        return dex_profile_from_environment(out, error);
    });
}

namespace detail {
/* Runs `call` with a dex_robot_options_t view of `options`; the C strings
 * stay alive for the duration of the call. */
template <typename Call> void with_options(const ConnectOptions &options, Call &&call) {
    std::vector<const char *> sensors;
    sensors.reserve(options.enable_sensors.size());
    for (const auto &name : options.enable_sensors) sensors.push_back(name.c_str());
    dex_robot_options_t native{};
    if (const auto status = dex_robot_options_init(&native); status != DEX_OK) {
        throw Error(status, "robot options could not be initialized");
    }
    native.profile_utf8 = options.profile.empty() ? nullptr : options.profile.c_str();
    native.config_file_utf8 = options.config_file.empty() ? nullptr : options.config_file.c_str();
    native.enable_sensors = sensors.empty() ? nullptr : sensors.data();
    native.enable_sensor_count = sensors.size();
    native.simulated = options.simulated ? 1 : 0;
    native.require_version_check = options.require_version_check ? 1 : 0;
    native.disable_watchdog = options.disable_watchdog ? 1 : 0;
    native.exit_on_termination = options.exit_on_termination ? 1 : 0;
    native.watchdog_command_utf8 =
        options.watchdog_command.empty() ? nullptr : options.watchdog_command.c_str();
    native.skip_arm_initialization = options.skip_arm_initialization ? 1 : 0;
    call(native);
}
}  // namespace detail

/* The statically resolved configuration for `options` as normalized JSON,
 * without connecting; lists every declared component and sensor with its
 * enabled flag. */
[[nodiscard]] inline std::string resolved_config_json(const ConnectOptions &options = {}) {
    std::string json;
    detail::with_options(options, [&](const dex_robot_options_t &native) {
        json = detail::owned_string([&](char **out, dex_error_t *error) {
            return dex_resolved_config_json(&native, out, error);
        });
    });
    return json;
}

/* Chassis options: dex_chassis_options_t with the component defaults. */
[[nodiscard]] inline dex_chassis_options_t chassis_options() {
    dex_chassis_options_t value{};
    dex_chassis_options_init(&value);
    return value;
}

class MotionHandle {
public:
    MotionHandle() noexcept = default;
    explicit MotionHandle(dex_motion_t *value) noexcept : value_(value) {}
    ~MotionHandle() { dex_motion_release(value_); }
    MotionHandle(const MotionHandle &) = delete;
    MotionHandle &operator=(const MotionHandle &) = delete;
    MotionHandle(MotionHandle &&other) noexcept
        : value_(std::exchange(other.value_, nullptr)) {}
    MotionHandle &operator=(MotionHandle &&other) noexcept {
        if (this != &other) {
            dex_motion_release(value_);
            value_ = std::exchange(other.value_, nullptr);
        }
        return *this;
    }
    [[nodiscard]] uint64_t id() const {
        uint64_t id = 0;
        const auto status = dex_motion_id(value_, &id);
        if (status != DEX_OK) throw Error(status, "invalid motion handle");
        return id;
    }
    /* Cached state, as last refreshed. */
    [[nodiscard]] uint32_t state() const {
        uint32_t state = 0;
        const auto status = dex_motion_state(value_, &state);
        if (status != DEX_OK) throw Error(status, "invalid motion handle");
        return state;
    }
    /* Typed accessor over the raw state() value. */
    [[nodiscard]] MotionState motion_state() const {
        return static_cast<MotionState>(state());
    }
    /* Refreshes from the status stream; a broken stream throws instead of
     * returning a stale state, so a polling loop cannot spin forever. */
    MotionState refresh() {
        auto error = detail::error();
        uint32_t state = 0;
        check(dex_motion_refresh(value_, &state, &error), error);
        return static_cast<MotionState>(state);
    }
    bool is_done() {
        const auto state = refresh();
        return state != MotionState::Pending && state != MotionState::Running;
    }
    [[nodiscard]] std::string message() const {
        return detail::owned_string([&](char **out, dex_error_t *error) {
            return dex_motion_message(value_, out, error);
        });
    }
    /* Waits for a terminal state and returns it. Throws Error with
     * DEX_TIMEOUT on a caller deadline or the library's hard ceiling, and
     * DEX_RUNTIME_ERROR when the server reports the motion failed. */
    template <typename Wait = WaitUntilComplete>
    MotionState wait(Wait policy = {}) {
        auto error = detail::error();
        const auto native = wait_policy(policy);
        check(dex_motion_wait(value_, &native, &error), error);
        return motion_state();
    }
    template <typename Wait = WaitUntilComplete>
    MotionState wait_success(Wait policy = {}) {
        auto error = detail::error();
        const auto native = wait_policy(policy);
        check(dex_motion_wait_success(value_, &native, &error), error);
        return motion_state();
    }
    template <typename Wait = WaitUntilComplete>
    MotionState cancel_and_wait(Wait policy = {}) {
        cancel();
        return wait(policy);
    }
    void cancel() {
        auto error = detail::error();
        check(dex_motion_cancel(value_, &error), error);
    }
private:
    dex_motion_t *value_ = nullptr;
};

/* A waited move whose wait failed while the motion itself was started: the
 * motion stays reachable through take_motion(). Copyable, as an exception
 * type must be, by sharing the one handle. */
class MotionWaitError : public Error {
public:
    MotionWaitError(dex_status_t status, const char *message, MotionHandle motion)
        : Error(status, message),
          motion_(std::make_shared<MotionHandle>(std::move(motion))) {}
    MotionHandle take_motion() { return std::move(*motion_); }
private:
    std::shared_ptr<MotionHandle> motion_;
};
/* The timed wait ran out; the motion is still live. */
class MotionTimeout : public MotionWaitError {
public:
    MotionTimeout(const char *message, MotionHandle motion)
        : MotionWaitError(DEX_TIMEOUT, message, std::move(motion)) {}
};
/* The motion was stopped, cancelled or superseded under the wait
 * (DEX_STOPPED); the handle reports which. */
class MotionStopped : public MotionWaitError {
public:
    MotionStopped(const char *message, MotionHandle motion)
        : MotionWaitError(DEX_STOPPED, message, std::move(motion)) {}
};

/* Several tracked motions from a robot-level fan-out, waited and cancelled
 * as one unit. */
class MotionGroup {
public:
    MotionGroup() noexcept = default;
    explicit MotionGroup(dex_motion_group_t *value) noexcept : value_(value) {}
    ~MotionGroup() { dex_motion_group_release(value_); }
    MotionGroup(const MotionGroup &) = delete;
    MotionGroup &operator=(const MotionGroup &) = delete;
    MotionGroup(MotionGroup &&other) noexcept
        : value_(std::exchange(other.value_, nullptr)) {}
    MotionGroup &operator=(MotionGroup &&other) noexcept {
        if (this != &other) {
            dex_motion_group_release(value_);
            value_ = std::exchange(other.value_, nullptr);
        }
        return *this;
    }
    [[nodiscard]] size_t size() const {
        size_t count = 0;
        const auto status = dex_motion_group_len(value_, &count);
        if (status != DEX_OK) throw Error(status, "invalid motion group");
        return count;
    }
    [[nodiscard]] MotionHandle member(size_t index) const {
        auto error = detail::error();
        dex_motion_t *motion = nullptr;
        check(dex_motion_group_member(value_, index, &motion, &error), error);
        return MotionHandle(motion);
    }
    /* Waits for every member and returns the worst outcome. Throws after
     * every member was waited on if one timed out or failed. */
    template <typename Wait = WaitUntilComplete>
    MotionState wait(Wait policy = {}) {
        auto error = detail::error();
        const auto native = wait_policy(policy);
        uint32_t state = 0;
        check(dex_motion_group_wait(value_, &native, &state, &error), error);
        return static_cast<MotionState>(state);
    }
    template <typename Wait = WaitUntilComplete>
    MotionState wait_success(Wait policy = {}) {
        const auto result = wait(policy);
        if (result != MotionState::Succeeded) {
            throw Error(DEX_STOPPED, "motion group reached a non-success terminal state");
        }
        return result;
    }
    template <typename Wait = WaitUntilComplete>
    MotionState cancel_and_wait(Wait policy = {}) {
        cancel();
        return wait(policy);
    }
    [[nodiscard]] MotionState state() {
        auto error = detail::error();
        uint32_t state = 0;
        check(dex_motion_group_state(value_, &state, &error), error);
        return static_cast<MotionState>(state);
    }
    void cancel() {
        auto error = detail::error();
        check(dex_motion_group_cancel(value_, &error), error);
    }
private:
    dex_motion_group_t *value_ = nullptr;
};

/* A partial fan-out retains the group for inspection and recovery. */
class MotionGroupStartError : public Error {
public:
    MotionGroupStartError(dex_status_t status, const char *message, MotionGroup group)
        : Error(status, message), group_(std::make_shared<MotionGroup>(std::move(group))) {}
    MotionGroup &group() const { return *group_; }
private:
    std::shared_ptr<MotionGroup> group_;
};

/* One decoded camera frame. Pixels are borrowed from the native buffer
 * and stay valid while the frame is alive; nothing is copied unless you
 * ask for it. */
class Frame {
public:
    explicit Frame(dex_frame_t *value) : value_(value) {
        info_ = detail::sized<dex_frame_info_t>();
        auto error = detail::error();
        const auto info_status = dex_frame_info(value_, &info_, &error);
        const auto data_status =
            info_status == DEX_OK ? dex_frame_data(value_, &data_, &length_) : DEX_OK;
        if (info_status != DEX_OK || data_status != DEX_OK) {
            dex_frame_release(value_);
            value_ = nullptr;
            if (info_status != DEX_OK) check(info_status, error);
            throw Error(data_status, "invalid frame handle");
        }
    }
    ~Frame() { dex_frame_release(value_); }
    Frame(const Frame &) = delete;
    Frame &operator=(const Frame &) = delete;
    Frame(Frame &&other) noexcept
        : value_(std::exchange(other.value_, nullptr)), info_(std::exchange(other.info_, {})),
          data_(std::exchange(other.data_, nullptr)), length_(std::exchange(other.length_, 0)) {}
    Frame &operator=(Frame &&other) noexcept {
        if (this != &other) {
            dex_frame_release(value_);
            value_ = std::exchange(other.value_, nullptr);
            info_ = std::exchange(other.info_, {});
            data_ = std::exchange(other.data_, nullptr);
            length_ = std::exchange(other.length_, 0);
        }
        return *this;
    }
    [[nodiscard]] const dex_frame_info_t &info() const noexcept { return info_; }
    [[nodiscard]] uint32_t width() const noexcept { return info_.width; }
    [[nodiscard]] uint32_t height() const noexcept { return info_.height; }
    [[nodiscard]] FrameEncoding encoding() const noexcept { return static_cast<FrameEncoding>(info_.encoding); }
    [[nodiscard]] uint64_t timestamp_ns() const noexcept { return info_.timestamp_ns; }
    [[nodiscard]] const uint8_t *data() const noexcept { return data_; }
    [[nodiscard]] size_t size() const noexcept { return length_; }
    /* Depth frames only: the payload as meters. Copies. */
    [[nodiscard]] std::vector<float> depth_meters() const {
        if (encoding() != FrameEncoding::Depth32F || length_ == 0) return {};
        std::vector<float> meters(length_ / sizeof(float));
        std::memcpy(meters.data(), data_, meters.size() * sizeof(float));
        return meters;
    }
private:
    dex_frame_t *value_ = nullptr;
    dex_frame_info_t info_{};
    const uint8_t *data_ = nullptr;
    size_t length_ = 0;
};

/* The native loop pacer. sleep() blocks until the next period boundary. */
class RateLimiter {
public:
    explicit RateLimiter(double rate_hz, size_t window_size = 0, bool adaptive = false) {
        auto error = detail::error();
        check(dex_rate_limiter_create(rate_hz, window_size, adaptive, &value_, &error), error);
    }
    ~RateLimiter() { dex_rate_limiter_release(value_); }
    RateLimiter(const RateLimiter &) = delete;
    RateLimiter &operator=(const RateLimiter &) = delete;
    RateLimiter(RateLimiter &&other) noexcept : value_(std::exchange(other.value_, nullptr)) {}
    RateLimiter &operator=(RateLimiter &&other) noexcept {
        if (this != &other) {
            dex_rate_limiter_release(value_);
            value_ = std::exchange(other.value_, nullptr);
        }
        return *this;
    }
    /* The first sleep() anchors the schedule, so setup time after
     * construction is never a missed deadline. */
    void sleep() { check_status(dex_rate_limiter_sleep(value_), "rate limiter sleep failed"); }
    /* Re-anchors the schedule and clears the counters after a pause. */
    void reset() { check_status(dex_rate_limiter_reset(value_), "rate limiter reset failed"); }
    [[nodiscard]] dex_rate_limiter_stats_t stats() const {
        auto stats = detail::sized<dex_rate_limiter_stats_t>();
        check_status(dex_rate_limiter_stats(value_, &stats), "rate limiter stats unavailable");
        return stats;
    }
    void done() { check_status(dex_rate_limiter_done(value_), "rate limiter completion failed"); }
    [[nodiscard]] double actual_rate() const {
        double rate = 0.0;
        check_status(dex_rate_limiter_actual_rate(value_, &rate), "rate limiter actual rate unavailable");
        return rate;
    }
    [[nodiscard]] double average_rate() const {
        double rate = 0.0;
        check_status(dex_rate_limiter_average_rate(value_, &rate), "rate limiter average rate unavailable");
        return rate;
    }
private:
    static void check_status(dex_status_t status, const char *message) {
        if (status != DEX_OK) throw Error(status, message);
    }
    dex_rate_limiter_t *value_ = nullptr;
};

/* The latest decoded joint state of one component. A channel the component
 * does not publish (many publish no torque or current) is EMPTY, as in Rust
 * and Python, never a vector of zeros. */
struct JointState {
    std::vector<double> position;
    std::vector<double> velocity;
    std::vector<double> torque;
    std::vector<double> current;
    uint64_t timestamp_ns = 0;  /* the producer's clock */
    /* How long ago the sample arrived locally: the only value that shows a
     * stream has stalled. */
    std::chrono::milliseconds age{0};
};

/* URDF limits in command order; unbounded joints read as +/-infinity. */
struct JointLimits {
    std::vector<double> lower;
    std::vector<double> upper;
    std::vector<double> velocity;
};

class JointComponent {
public:
    JointComponent() noexcept = default;
    explicit JointComponent(dex_component_t *value) noexcept : value_(value) {}
    ~JointComponent() { dex_component_release(value_); }
    JointComponent(const JointComponent &) = delete;
    JointComponent &operator=(const JointComponent &) = delete;
    JointComponent(JointComponent &&other) noexcept
        : value_(std::exchange(other.value_, nullptr)) {}
    JointComponent &operator=(JointComponent &&other) noexcept {
        if (this != &other) {
            dex_component_release(value_);
            value_ = std::exchange(other.value_, nullptr);
        }
        return *this;
    }

    /* ---- introspection ---- */
    [[nodiscard]] std::string name() const {
        return detail::owned_string([&](char **out, dex_error_t *error) {
            return dex_component_name(value_, out, error);
        });
    }
    [[nodiscard]] size_t joint_count() const {
        size_t count = 0;
        const auto status = dex_component_joint_count(value_, &count);
        if (status != DEX_OK) throw Error(status, "invalid joint component");
        return count;
    }
    [[nodiscard]] std::vector<std::string> joint_names() const {
        return detail::names(
            [&](size_t *count) { return dex_component_joint_count(value_, count); },
            [&](size_t index, char **out, dex_error_t *error) {
                return dex_component_joint_name(value_, index, out, error);
            });
    }
    /* Empty when the model carries no joint metadata. */
    [[nodiscard]] std::optional<JointLimits> joint_limits() const {
        const size_t count = joint_count();
        JointLimits limits{std::vector<double>(count), std::vector<double>(count),
                           std::vector<double>(count)};
        size_t written = 0;
        bool has_limits = false;
        auto error = detail::error();
        check(dex_component_joint_limits(value_, limits.lower.data(), limits.upper.data(),
                                         limits.velocity.data(), count, &written, &has_limits,
                                         &error),
              error);
        if (!has_limits) return std::nullopt;
        return limits;
    }
    [[nodiscard]] std::vector<std::string> pose_names() const {
        return detail::names(
            [&](size_t *count) { return dex_component_pose_count(value_, count); },
            [&](size_t index, char **out, dex_error_t *error) {
                return dex_component_pose_name(value_, index, out, error);
            });
    }
    [[nodiscard]] std::vector<double> predefined_pose(const std::string &name) const {
        return detail::read_array<double>([&](double *out, size_t capacity, size_t *count,
                                              dex_error_t *error) {
            return dex_component_predefined_pose(value_, name.c_str(), out, capacity, count, error);
        });
    }
    [[nodiscard]] std::vector<double> resolve_pose(const std::string &name) const {
        return detail::read_array<double>([&](double *out, size_t capacity, size_t *count,
                                              dex_error_t *error) {
            return dex_component_resolve_pose(value_, name.c_str(), out, capacity, count, error);
        });
    }
    [[nodiscard]] std::vector<double> get_pose(const std::string &name, bool passthrough = false) const {
        return detail::read_array<double>([&](double *out, size_t capacity, size_t *count,
                                              dex_error_t *error) {
            return dex_component_get_pose(value_, name.c_str(), passthrough, out, capacity, count, error);
        });
    }

    /* ---- state ---- */
    /* Observational read: no freshness bound; check JointState::age. */
    [[nodiscard]] JointState get_joint_state() const { return read_state(DEX_LIMIT_DISABLED); }
    /* Throws Error(DEX_STALE_STATE) for a sample older than max_age. */
    [[nodiscard]] JointState get_joint_state(std::chrono::milliseconds max_age) const {
        if (max_age.count() <= 0) throw std::invalid_argument("max_age must be positive");
        return read_state(detail::millis(max_age));
    }
    /* The control-path read: the component's configured freshness limit. */
    [[nodiscard]] JointState get_fresh_joint_state() const {
        return read_state(DEX_STATE_AGE_CONFIGURED);
    }
    [[nodiscard]] std::vector<double> get_joint_pos() const {
        return detail::read_array<double>([&](double *out, size_t capacity, size_t *count,
                                              dex_error_t *error) {
            return dex_component_get_joint_pos(value_, out, capacity, count, error);
        });
    }
    [[nodiscard]] std::vector<double> get_joint_vel() const { return get_joint_state().velocity; }
    [[nodiscard]] std::vector<double> get_joint_torque() const { return get_joint_state().torque; }
    [[nodiscard]] std::vector<double> get_joint_current() const { return get_joint_state().current; }
    [[nodiscard]] std::vector<uint32_t> get_joint_err() const {
        return detail::read_array<uint32_t>([&](uint32_t *out, size_t capacity, size_t *count,
                                                dex_error_t *error) {
            return dex_component_get_joint_err(value_, out, capacity, count, error);
        });
    }
    [[nodiscard]] bool is_joint_pos_reached(const std::vector<double> &target, double tolerance = 0.05) const {
        bool reached = false;
        auto error = detail::error();
        check(dex_component_is_joint_pos_reached(value_, target.data(), target.size(), tolerance,
                                                 &reached, &error),
              error);
        return reached;
    }
    [[nodiscard]] bool is_pose_reached(const std::string &pose, double tolerance = 0.05,
                         bool passthrough = false) const {
        return is_joint_pos_reached(get_pose(pose, passthrough), tolerance);
    }

    /* ---- direct commands: one unshaped setpoint, no interpolation ---- */
    template <typename Wait = NoWait>
    void set_joint_pos(const std::vector<double> &values, Wait wait = {}) {
        auto error = detail::error();
        const auto native = wait_policy(wait);
        check(dex_component_set_joint_pos(value_, values.data(), values.size(),
                                          &native, &error), error);
    }
    template <typename Wait = NoWait>
    void set_joint_pos(const std::vector<double> &values, const dex_command_options_t &options,
                       Wait wait = {}) {
        auto error = detail::error();
        const auto native = wait_policy(wait);
        check(dex_component_set_joint_pos_opt(value_, values.data(), nullptr, values.size(),
                                              &options, &native, &error),
              error);
    }
    /* Direct command with a velocity feed-forward per joint (Python:
     * set_joint_pos(q, joint_vel=qd)). */
    template <typename Wait = NoWait>
    void set_joint_pos(const std::vector<double> &positions,
                       const std::vector<double> &velocities,
                       const dex_command_options_t &options = command_options(),
                       Wait wait = {}) {
        if (positions.size() != velocities.size()) {
            throw std::invalid_argument("positions and velocities must have equal length");
        }
        auto error = detail::error();
        const auto native = wait_policy(wait);
        check(dex_component_set_joint_pos_opt(value_, positions.data(), velocities.data(),
                                              positions.size(), &options, &native, &error),
              error);
    }
    template <typename Wait = NoWait>
    [[deprecated("use set_joint_pos(positions, velocities, options, wait)")]]
    void set_joint_pos_vel(const std::vector<double> &positions,
                           const std::vector<double> &velocities,
                           Wait wait = {}) {
        set_joint_pos(positions, velocities, command_options(), wait);
    }
    template <typename Wait = NoWait>
    [[deprecated("use set_joint_pos(positions, velocities, options, wait)")]]
    void set_joint_pos_vel(const std::vector<double> &positions,
                           const std::vector<double> &velocities,
                           const dex_command_options_t &options, Wait wait = {}) {
        set_joint_pos(positions, velocities, options, wait);
    }
    /* Hands: the model's "open" / "close" poses as one direct command. */
    template <typename Wait = NoWait> void open_hand(Wait wait = {}) {
        set_joint_pos(predefined_pose("open"), wait);
    }
    template <typename Wait = NoWait> void close_hand(Wait wait = {}) {
        set_joint_pos(predefined_pose("close"), wait);
    }
    /* Whether the state stream is usable now: a sample within the
     * configured freshness limit (Python: is_active()). */
    [[nodiscard]] bool is_active() const {
        bool active = false;
        if (dex_component_is_active(value_, &active) != DEX_OK) return false;
        return active;
    }
    /* Waits for the state stream to become usable; false on timeout. */
    [[nodiscard]] bool wait_for_active(std::chrono::milliseconds timeout = std::chrono::seconds{5}) const {
        auto error = detail::error();
        bool active = false;
        check(dex_component_wait_for_active(value_, detail::millis(timeout), &active, &error), error);
        return active;
    }
    /* Cancels the active planned motion and holds the measured position. */
    void stop() {
        auto error = detail::error();
        check(dex_component_stop(value_, &error), error);
    }

    /* ---- streamed goal (untracked, plugin-smoothed) ----
     * Call it from a loop, once per tick; nothing is returned to wait on.
     * Refused while a tracked motion of this component is still running. */
    void stream_joint_pos(const std::vector<double> &values,
                          const dex_motion_options_t &options = motion_options()) {
        auto error = detail::error();
        check(dex_component_stream_joint_pos(value_, values.data(), values.size(), &options,
                                             &error),
              error);
    }

    /* ---- planned, server-tracked motion ---- */
    template <typename Wait = NoWait>
    MotionHandle move_to_joint_pos(const std::vector<double> &values, Wait wait = {}) {
        return move_to_joint_pos(values, motion_options(), wait);
    }
    template <typename Wait = NoWait>
    MotionHandle move_to_joint_pos(const std::vector<double> &values,
                                   const dex_motion_options_t &options, Wait wait = {}) {
        auto error = detail::error();
        dex_motion_t *motion = nullptr;
        const auto native = wait_policy(wait);
        const auto status = dex_component_move_to_joint_pos_opt(
            value_, values.data(), values.size(), &options, &native, &motion, &error);
        return finish_move(status, error, motion);
    }
    MotionHandle go_to_pose(const std::string &pose, const dex_motion_options_t &options) {
        return go_to_pose(pose, WaitUntilComplete{}, options);
    }
    template <typename Wait = WaitUntilComplete>
    MotionHandle go_to_pose(const std::string &pose, Wait wait = {},
                            const dex_motion_options_t &options = motion_options(),
                            bool passthrough = false) {
        auto error = detail::error();
        dex_motion_t *motion = nullptr;
        const auto native = wait_policy(wait);
        const auto status = dex_component_go_to_pose_opt(value_, pose.c_str(), &options, &native, passthrough,
                                                     &motion, &error);
        return finish_move(status, error, motion);
    }
    // Convenience overload for default wait/options with explicit passthrough.
    MotionHandle go_to_pose(const std::string &pose, bool passthrough) {
        return go_to_pose(pose, WaitUntilComplete{}, motion_options(), passthrough);
    }
    /* points[i] is one waypoint in command order; times_s[i] its time from
     * start. velocity_scale 0 leaves the plugin default. */
    template <typename Wait = NoWait>
    MotionHandle move_joint_trajectory(const std::vector<std::vector<double>> &points,
                                       const std::vector<double> &times_s,
                                       double velocity_scale = 0.0, Wait wait = {}) {
        return move_joint_trajectory(points, times_s,
                                     with_velocity_scale(motion_options(), velocity_scale), wait);
    }
    /* velocity_scale, disable_limit_enforcement and wait_ceiling_ms apply. */
    template <typename Wait = NoWait>
    MotionHandle move_joint_trajectory(const std::vector<std::vector<double>> &points,
                                       const std::vector<double> &times_s,
                                       const dex_motion_options_t &options, Wait wait = {}) {
        if (points.empty() || points.size() != times_s.size()) {
            throw std::invalid_argument("one time per waypoint is required");
        }
        const size_t joints = points.front().size();
        std::vector<double> flat;
        flat.reserve(points.size() * joints);
        for (const auto &point : points) {
            if (point.size() != joints) {
                throw std::invalid_argument("every waypoint must have the same joint count");
            }
            flat.insert(flat.end(), point.begin(), point.end());
        }
        auto error = detail::error();
        dex_motion_t *motion = nullptr;
        const auto native = wait_policy(wait);
        const auto status = dex_component_move_joint_trajectory_opt(
            value_, flat.data(), times_s.data(), points.size(), joints, &options, &native,
            &motion, &error);
        return finish_move(status, error, motion);
    }

    [[deprecated("use move_to_joint_pos(target, NoWait{})")]]
    MotionHandle start_position(const std::vector<double> &target) {
        return move_to_joint_pos(target, NoWait{});
    }
    void command_position(const std::vector<double> &target,
                          const std::vector<double> &max_steps = {},
                          uint64_t max_state_age_ms = DEX_STATE_AGE_CONFIGURED) {
        if (!max_steps.empty() && max_steps.size() != target.size()) {
            throw std::invalid_argument("one step limit per joint is required");
        }
        auto error = detail::error();
        check(dex_component_command_position(value_, target.data(), max_steps.empty() ? nullptr : max_steps.data(),
                                              target.size(), max_state_age_ms, &error), error);
    }

    /* ---- grasping torque (grippers) ---- */
    [[nodiscard]] std::optional<double> grasp_torque() const {
        auto error = detail::error();
        double value = 0.0;
        int has_torque = 0;
        check(dex_component_grasp_torque(value_, &value, &has_torque, &error), error);
        if (has_torque == 0) return std::nullopt;
        return value;
    }
    double set_grasp_torque(double torque, bool force = false) {
        auto error = detail::error();
        double applied = 0.0;
        check(dex_component_set_grasp_torque(value_, torque, force ? 1 : 0, &applied, &error),
              error);
        return applied;
    }

    /* ---- firmware services (JSON replies) ---- */
    std::string set_modes(const std::vector<JointMode> &modes) {
        std::vector<int32_t> codes;
        codes.reserve(modes.size());
        for (const auto mode : modes) codes.push_back(static_cast<int32_t>(mode));
        return detail::owned_string([&](char **out, dex_error_t *error) {
            return dex_component_set_modes(value_, codes.data(), codes.size(), out, error);
        });
    }
    std::string set_mode(JointMode mode) {
        return set_modes(std::vector<JointMode>(joint_count(), mode));
    }
    /* One firmware round trip. The native driver translates device mode codes
     * into stable API modes; an unrepresentable mode is 0 (unspecified). */
    std::vector<JointMode> get_modes() const {
        std::vector<int32_t> codes(joint_count());
        size_t count = 0;
        auto error = detail::error();
        check(dex_component_get_modes(value_, codes.data(), codes.size(), &count, &error), error);
        codes.resize(count);
        std::vector<JointMode> modes;
        modes.reserve(codes.size());
        for (const auto code : codes) modes.push_back(static_cast<JointMode>(code));
        return modes;
    }
    std::string request_json(const std::string &role, const std::string &request_json) {
        return detail::owned_string([&](char **out, dex_error_t *error) {
            return dex_component_request_json(value_, role.c_str(), request_json.c_str(), out,
                                              error);
        });
    }
    std::string set_pid(const std::vector<double> &p) {
        return detail::owned_string([&](char **out, dex_error_t *error) {
            return dex_component_set_pid(value_, p.data(), p.size(), out, error);
        });
    }
    std::string get_pid() {
        return detail::owned_string([&](char **out, dex_error_t *error) {
            return dex_component_get_pid(value_, out, error);
        });
    }
    /* Releases (true) or engages the brakes of exactly the listed joints.
     * An empty list throws; it never means "every joint". */
    std::string release_brake(bool enable, const std::vector<size_t> &joints) {
        return detail::owned_string([&](char **out, dex_error_t *error) {
            return dex_component_release_brake(value_, enable, joints.data(), joints.size(), out,
                                               error);
        });
    }
    /* Every joint of the component. Releasing lets an unsupported arm fall. */
    std::string release_all_brakes(bool enable) {
        return detail::owned_string([&](char **out, dex_error_t *error) {
            return dex_component_release_all_brakes(value_, enable, out, error);
        });
    }
    std::string get_brake_status() {
        return detail::owned_string([&](char **out, dex_error_t *error) {
            return dex_component_brake_status(value_, out, error);
        });
    }
    std::string set_force_torque_sensor(bool enable) {
        return detail::owned_string([&](char **out, dex_error_t *error) {
            return dex_component_set_force_torque_sensor(value_, enable, out, error);
        });
    }
    std::string get_force_torque_sensor_mode() {
        return detail::owned_string([&](char **out, dex_error_t *error) {
            return dex_component_force_torque_sensor_mode(value_, out, error);
        });
    }
    std::string set_ee_baud_rate(uint32_t baud_rate) {
        return detail::owned_string([&](char **out, dex_error_t *error) {
            return dex_component_set_ee_baud_rate(value_, baud_rate, out, error);
        });
    }
    std::string get_ee_baud_rate() {
        return detail::owned_string([&](char **out, dex_error_t *error) {
            return dex_component_ee_baud_rate(value_, out, error);
        });
    }
    void set_idle_mode(bool enabled) {
        auto error = detail::error();
        check(dex_component_set_idle_mode(value_, enabled, &error), error);
    }
    std::optional<bool> get_idle_mode() {
        auto error = detail::error();
        int32_t enabled = -1;
        check(dex_component_idle_mode(value_, &enabled, &error), error);
        if (enabled < 0) return std::nullopt;
        return enabled != 0;
    }

    /* Deprecated spellings of the reads above (removed at 1.0). */
    [[deprecated("use get_brake_status")]]
    std::string brake_status() { return get_brake_status(); }
    [[deprecated("use get_force_torque_sensor_mode")]]
    std::string force_torque_sensor_mode() { return get_force_torque_sensor_mode(); }
    [[deprecated("use get_ee_baud_rate")]]
    std::string ee_baud_rate() { return get_ee_baud_rate(); }
    [[deprecated("use get_idle_mode")]]
    std::optional<bool> idle_mode() { return get_idle_mode(); }
    [[deprecated("use get_button_state")]]
    std::optional<std::pair<bool, bool>> button_state() const { return get_button_state(); }
    [[deprecated("use get_touch_data")]]
    std::optional<std::vector<double>> touch_forces() const { return get_touch_data(); }

    /* ---- end effector, buttons, touch ---- */
    [[nodiscard]] bool ee_pass_through_enabled() const {
        bool enabled = false;
        const auto status = dex_component_ee_pass_through_enabled(value_, &enabled);
        if (status != DEX_OK) throw Error(status, "end-effector pass-through status unavailable");
        return enabled;
    }
    void ee_pass_through_send(const std::vector<uint8_t> &data) {
        auto error = detail::error();
        check(dex_component_ee_pass_through_send(value_, data.data(), data.size(), &error), error);
    }
    std::optional<std::vector<uint8_t>> ee_pass_through_latest() {
        bool has_response = false;
        auto bytes = detail::owned_bytes([&](dex_buffer_t *out, dex_error_t *error) {
            return dex_component_ee_pass_through_latest(value_, out, &has_response, error);
        });
        if (!has_response) return std::nullopt;
        return bytes;
    }
    /* (blue, green); empty before the first sample. */
    [[nodiscard]] std::optional<std::pair<bool, bool>> get_button_state() const {
        auto error = detail::error();
        bool blue = false, green = false, has_state = false;
        check(dex_component_button_state(value_, &blue, &green, &has_state, &error), error);
        if (!has_state) return std::nullopt;
        return std::make_pair(blue, green);
    }
    [[nodiscard]] std::optional<std::vector<double>> get_touch_data() const {
        bool has_data = false;
        auto forces = detail::read_array<double>([&](double *out, size_t capacity, size_t *count,
                                                     dex_error_t *error) {
            return dex_component_touch_forces(value_, out, capacity, count, &has_data, error);
        });
        if (!has_data) return std::nullopt;
        return forces;
    }

private:
    /* A failed wait can come back with the live motion (timeout, stop,
     * cancellation). Ownership is taken first, so every path releases it. */
    static MotionHandle finish_move(dex_status_t status, const dex_error_t &error,
                                    dex_motion_t *motion) {
        MotionHandle handle(motion);
        if (motion != nullptr && status == DEX_TIMEOUT) {
            throw MotionTimeout(error.message, std::move(handle));
        }
        if (motion != nullptr && status == DEX_STOPPED) {
            throw MotionStopped(error.message, std::move(handle));
        }
        check(status, error);
        return handle;
    }
    JointState read_state(uint64_t max_state_age_ms) const {
        const size_t capacity = joint_count();
        JointState state{std::vector<double>(capacity), std::vector<double>(capacity),
                         std::vector<double>(capacity), std::vector<double>(capacity)};
        auto counts = detail::sized<dex_joint_state_counts_t>();
        uint64_t age_ms = 0;
        auto error = detail::error();
        check(dex_component_get_joint_state_ex(
                  value_, max_state_age_ms, state.position.data(), state.velocity.data(),
                  state.torque.data(), state.current.data(), capacity, &counts,
                  &state.timestamp_ns, &age_ms, &error),
              error);
        state.position.resize(counts.position);
        state.velocity.resize(counts.velocity);
        state.torque.resize(counts.torque);
        state.current.resize(counts.current);
        state.age = detail::age(age_ms);
        return state;
    }
    dex_component_t *value_ = nullptr;
};

/* One 2D lidar scan. Every channel comes from the same sample; one the
 * sensor does not publish is empty. */
struct Lidar2dScan {
    std::vector<double> ranges;
    std::vector<double> angles;
    std::vector<uint32_t> intensities;
    dex_lidar_2d_info_t info{};
    std::chrono::milliseconds age{0}; /* local arrival age */
};

/* One 3D lidar cloud, as separate channels of one sample. */
struct Lidar3dCloud {
    std::vector<double> x, y, z;
    std::vector<uint32_t> intensity, ring, point_timestamps_ns;
    dex_lidar_3d_info_t info{};
    std::chrono::milliseconds age{0}; /* local arrival age */
};

/* One force-torque reading: force (N) then torque (N*m). */
struct Wrench {
    std::array<double, 6> values{};
    uint64_t timestamp_ns = 0;
};


namespace detail {
/* The handle behind a sensor read: a control connection or the read-only
 * diagnostic client. Sensor views dispatch on it, so both expose one API. */
struct SensorSource {
    dex_robot_t *robot = nullptr;
    dex_diagnostic_t *client = nullptr;
    template <typename OnRobot, typename OnClient>
    dex_status_t call(OnRobot &&on_robot, OnClient &&on_client) const {
        if (robot == nullptr && client == nullptr) {
            throw Error(DEX_INVALID_ARGUMENT, "sensor view has no connection");
        }
        return robot != nullptr ? on_robot(robot) : on_client(client);
    }
};
/* Runs an _ex sensor read; the arrival age is handed over even when the
 * read then throws as stale. */
template <typename Call> void aged(std::chrono::milliseconds *age, Call &&call) {
    auto error = detail::error();
    uint64_t age_ms = 0;
    const auto status = call(&age_ms, &error);
    if (age != nullptr) *age = detail::age(age_ms);
    check(status, error);
}
inline void snapshot_status(dex_status_t status) {
    if (status != DEX_OK) throw Error(status, "invalid lidar snapshot");
}
/* Copies one borrowed snapshot channel; an absent channel is empty. */
template <typename T, typename Snapshot, typename Call>
std::vector<T> channel(const Snapshot *snapshot, uint32_t kind, Call call) {
    const T *data = nullptr;
    size_t count = 0;
    snapshot_status(call(snapshot, kind, &data, &count));
    return count == 0 ? std::vector<T>{} : std::vector<T>(data, data + count);
}
}  // namespace detail

/* One camera of a Robot or DiagnosticClient: `robot.sensors().camera("head_camera")`.
 * A view, not an owner: it must not outlive the connection it came from.
 * Images flow only between subscribe() and unsubscribe(). */
class Camera {
public:
    Camera(detail::SensorSource source, std::string name) : source_(source), name_(std::move(name)) {}
    [[nodiscard]] const std::string &name() const noexcept { return name_; }
    /* Query the server JSON object; no image subscription required. Calibration
     * fields are device-dependent. Each call queries fresh information. */
    [[nodiscard]] std::string info_json(
        std::chrono::milliseconds timeout = std::chrono::milliseconds{2000}) const {
        return detail::owned_string([&](char **out, dex_error_t *error) {
            return source_.call(
                [&](dex_robot_t *r) { return dex_robot_camera_info_json(r, name_.c_str(), detail::millis(timeout), out, error); },
                [&](dex_diagnostic_t *c) { return dex_diagnostic_camera_info_json(c, name_.c_str(), detail::millis(timeout), out, error); });
        });
    }
    [[nodiscard]] std::vector<std::string> streams() const {
        return detail::names(
            [&](size_t *count) {
                auto error = detail::error();
                const auto status = source_.call(
                    [&](dex_robot_t *r) { return dex_robot_camera_stream_count(r, name_.c_str(), count, &error); },
                    [&](dex_diagnostic_t *c) { return dex_diagnostic_camera_stream_count(c, name_.c_str(), count, &error); });
                check(status, error);
                return status;
            },
            [&](size_t index, char **out, dex_error_t *error) {
                return source_.call(
                    [&](dex_robot_t *r) { return dex_robot_camera_stream_name(r, name_.c_str(), index, out, error); },
                    [&](dex_diagnostic_t *c) { return dex_diagnostic_camera_stream_name(c, name_.c_str(), index, out, error); });
            });
    }
    /* history 0 keeps only the newest frame; n keeps the n newest. */
    void subscribe(const std::string &stream, size_t history = 0) {
        auto error = detail::error();
        check(source_.call(
                  [&](dex_robot_t *r) { return dex_robot_camera_subscribe(r, name_.c_str(), stream.c_str(), history, &error); },
                  [&](dex_diagnostic_t *c) { return dex_diagnostic_camera_subscribe(c, name_.c_str(), stream.c_str(), history, &error); }),
              error);
    }
    void unsubscribe(const std::string &stream) {
        auto error = detail::error();
        check(source_.call(
                  [&](dex_robot_t *r) { return dex_robot_camera_unsubscribe(r, name_.c_str(), stream.c_str(), &error); },
                  [&](dex_diagnostic_t *c) { return dex_diagnostic_camera_unsubscribe(c, name_.c_str(), stream.c_str(), &error); }),
              error);
    }
    /* Empty before the first frame arrives. */
    [[nodiscard]] std::optional<Frame> latest_frame(const std::string &stream) const {
        auto error = detail::error();
        dex_frame_t *frame = nullptr;
        check(source_.call(
                  [&](dex_robot_t *r) { return dex_robot_camera_latest_frame(r, name_.c_str(), stream.c_str(), &frame, &error); },
                  [&](dex_diagnostic_t *c) { return dex_diagnostic_camera_latest_frame(c, name_.c_str(), stream.c_str(), &frame, &error); }),
              error);
        if (frame == nullptr) return std::nullopt;
        return Frame(frame);
    }
    [[nodiscard]] dex_stream_stats_t stats(const std::string &stream) const {
        auto stats = detail::sized<dex_stream_stats_t>();
        auto error = detail::error();
        check(source_.call(
                  [&](dex_robot_t *r) { return dex_robot_camera_stats(r, name_.c_str(), stream.c_str(), &stats, &error); },
                  [&](dex_diagnostic_t *c) { return dex_diagnostic_camera_stats(c, name_.c_str(), stream.c_str(), &stats, &error); }),
              error);
        return stats;
    }
    /* An empty stream asks about every configured stream. */
    [[nodiscard]] bool is_active(const std::string &stream = {},
                                 std::chrono::milliseconds window = std::chrono::seconds{1}) const {
        auto error = detail::error();
        bool active = false;
        const char *which = stream.empty() ? nullptr : stream.c_str();
        check(source_.call(
                  [&](dex_robot_t *r) { return dex_robot_camera_is_active(r, name_.c_str(), which, detail::millis(window), &active, &error); },
                  [&](dex_diagnostic_t *c) { return dex_diagnostic_camera_is_active(c, name_.c_str(), which, detail::millis(window), &active, &error); }),
              error);
        return active;
    }
private:
    detail::SensorSource source_;
    std::string name_;
};

/* The typed sensors of a Robot or DiagnosticClient: `robot.sensors()`.
 * Sensors are keyed by sensor name (cameras, IMUs, lidars, ultrasonics) or
 * by the component they are mounted on (wrench, temperatures). A view, not
 * an owner: it must not outlive the connection. */
class Sensors {
public:
    explicit Sensors(detail::SensorSource source) : source_(source) {}
    [[nodiscard]] Camera camera(const std::string &name) const { return Camera(source_, name); }
    [[nodiscard]] dex_imu_t imu(const std::string &name) const {
        auto state = detail::sized<dex_imu_t>();
        auto error = detail::error();
        check(source_.call(
                  [&](dex_robot_t *r) { return dex_robot_imu_ex(r, name.c_str(), 0, &state, nullptr, &error); },
                  [&](dex_diagnostic_t *c) { return dex_diagnostic_imu_ex(c, name.c_str(), 0, &state, nullptr, &error); }),
              error);
        return state;
    }
    /* The max_age overloads throw Error(DEX_STALE_STATE) for a sample that
     * arrived more than max_age ago; *age (optional) receives the arrival
     * age either way. timestamp_ns is the producer's clock and cannot show a
     * stalled stream. */
    [[nodiscard]] dex_imu_t imu(const std::string &name, std::chrono::milliseconds max_age,
                                std::chrono::milliseconds *age = nullptr) const {
        auto state = detail::sized<dex_imu_t>();
        detail::aged(age, [&](uint64_t *age_ms, dex_error_t *error) {
            return source_.call(
                [&](dex_robot_t *r) { return dex_robot_imu_ex(r, name.c_str(), detail::millis(max_age), &state, age_ms, error); },
                [&](dex_diagnostic_t *c) { return dex_diagnostic_imu_ex(c, name.c_str(), detail::millis(max_age), &state, age_ms, error); });
        });
        return state;
    }
    [[nodiscard]] dex_ultrasonic_t ultrasonic(const std::string &name) const {
        auto state = detail::sized<dex_ultrasonic_t>();
        auto error = detail::error();
        check(source_.call(
                  [&](dex_robot_t *r) { return dex_robot_ultrasonic_ex(r, name.c_str(), 0, &state, nullptr, &error); },
                  [&](dex_diagnostic_t *c) { return dex_diagnostic_ultrasonic_ex(c, name.c_str(), 0, &state, nullptr, &error); }),
              error);
        return state;
    }
    [[nodiscard]] dex_ultrasonic_t ultrasonic(const std::string &name, std::chrono::milliseconds max_age,
                                              std::chrono::milliseconds *age = nullptr) const {
        auto state = detail::sized<dex_ultrasonic_t>();
        detail::aged(age, [&](uint64_t *age_ms, dex_error_t *error) {
            return source_.call(
                [&](dex_robot_t *r) { return dex_robot_ultrasonic_ex(r, name.c_str(), detail::millis(max_age), &state, age_ms, error); },
                [&](dex_diagnostic_t *c) { return dex_diagnostic_ultrasonic_ex(c, name.c_str(), detail::millis(max_age), &state, age_ms, error); });
        });
        return state;
    }
    /* One owned snapshot per read: every channel has its own length and
     * comes from the same sample. max_age of zero applies no bound. */
    [[nodiscard]] Lidar2dScan lidar_2d(const std::string &name, std::chrono::milliseconds max_age = {}) const {
        auto error = detail::error();
        dex_lidar_2d_t *raw = nullptr;
        check(source_.call(
                  [&](dex_robot_t *r) { return dex_robot_lidar_2d_take(r, name.c_str(), detail::millis(max_age), &raw, &error); },
                  [&](dex_diagnostic_t *c) { return dex_diagnostic_lidar_2d_take(c, name.c_str(), detail::millis(max_age), &raw, &error); }),
              error);
        std::unique_ptr<dex_lidar_2d_t, decltype(&dex_lidar_2d_release)> snapshot(raw, dex_lidar_2d_release);
        Lidar2dScan scan;
        scan.info = detail::sized<dex_lidar_2d_info_t>();
        check(dex_lidar_2d_info(raw, &scan.info, &error), error);
        scan.ranges = detail::channel<double>(raw, DEX_LIDAR_2D_RANGES, dex_lidar_2d_channel_f64);
        scan.angles = detail::channel<double>(raw, DEX_LIDAR_2D_ANGLES, dex_lidar_2d_channel_f64);
        scan.intensities = detail::channel<uint32_t>(raw, DEX_LIDAR_2D_INTENSITIES, dex_lidar_2d_channel_u32);
        uint64_t age_ms = 0;
        detail::snapshot_status(dex_lidar_2d_age_ms(raw, &age_ms));
        scan.age = detail::age(age_ms);
        return scan;
    }
    [[nodiscard]] Lidar3dCloud lidar_3d(const std::string &name, std::chrono::milliseconds max_age = {}) const {
        auto error = detail::error();
        dex_lidar_3d_t *raw = nullptr;
        check(source_.call(
                  [&](dex_robot_t *r) { return dex_robot_lidar_3d_take(r, name.c_str(), detail::millis(max_age), &raw, &error); },
                  [&](dex_diagnostic_t *c) { return dex_diagnostic_lidar_3d_take(c, name.c_str(), detail::millis(max_age), &raw, &error); }),
              error);
        std::unique_ptr<dex_lidar_3d_t, decltype(&dex_lidar_3d_release)> snapshot(raw, dex_lidar_3d_release);
        Lidar3dCloud cloud;
        cloud.info = detail::sized<dex_lidar_3d_info_t>();
        check(dex_lidar_3d_info(raw, &cloud.info, &error), error);
        cloud.x = detail::channel<double>(raw, DEX_LIDAR_3D_X, dex_lidar_3d_channel_f64);
        cloud.y = detail::channel<double>(raw, DEX_LIDAR_3D_Y, dex_lidar_3d_channel_f64);
        cloud.z = detail::channel<double>(raw, DEX_LIDAR_3D_Z, dex_lidar_3d_channel_f64);
        cloud.intensity = detail::channel<uint32_t>(raw, DEX_LIDAR_3D_INTENSITY, dex_lidar_3d_channel_u32);
        cloud.ring = detail::channel<uint32_t>(raw, DEX_LIDAR_3D_RING, dex_lidar_3d_channel_u32);
        cloud.point_timestamps_ns = detail::channel<uint32_t>(raw, DEX_LIDAR_3D_POINT_TIMESTAMPS, dex_lidar_3d_channel_u32);
        uint64_t age_ms = 0;
        detail::snapshot_status(dex_lidar_3d_age_ms(raw, &age_ms));
        cloud.age = detail::age(age_ms);
        return cloud;
    }
    [[nodiscard]] Wrench wrench(const std::string &component) const {
        Wrench reading;
        auto error = detail::error();
        check(source_.call(
                  [&](dex_robot_t *r) { return dex_robot_wrench_ex(r, component.c_str(), 0, reading.values.data(), &reading.timestamp_ns, nullptr, &error); },
                  [&](dex_diagnostic_t *c) { return dex_diagnostic_wrench_ex(c, component.c_str(), 0, reading.values.data(), &reading.timestamp_ns, nullptr, &error); }),
              error);
        return reading;
    }
    [[nodiscard]] Wrench wrench(const std::string &component, std::chrono::milliseconds max_age,
                                std::chrono::milliseconds *age = nullptr) const {
        Wrench reading;
        detail::aged(age, [&](uint64_t *age_ms, dex_error_t *error) {
            return source_.call(
                [&](dex_robot_t *r) { return dex_robot_wrench_ex(r, component.c_str(), detail::millis(max_age), reading.values.data(), &reading.timestamp_ns, age_ms, error); },
                [&](dex_diagnostic_t *c) { return dex_diagnostic_wrench_ex(c, component.c_str(), detail::millis(max_age), reading.values.data(), &reading.timestamp_ns, age_ms, error); });
        });
        return reading;
    }
    [[nodiscard]] std::string temperature_json(const std::string &component) const {
        return detail::owned_string([&](char **out, dex_error_t *error) {
            return source_.call(
                [&](dex_robot_t *r) { return dex_robot_temperature_json(r, component.c_str(), out, error); },
                [&](dex_diagnostic_t *c) { return dex_diagnostic_temperature_json(c, component.c_str(), out, error); });
        });
    }
    [[nodiscard]] std::string sensor_json(const std::string &name, const std::string &role = {}) const {
        const char *which = role.empty() ? nullptr : role.c_str();
        return detail::owned_string([&](char **out, dex_error_t *error) {
            return source_.call(
                [&](dex_robot_t *r) { return dex_robot_sensor_json(r, name.c_str(), which, out, error); },
                [&](dex_diagnostic_t *c) { return dex_diagnostic_sensor_json(c, name.c_str(), which, out, error); });
        });
    }
    [[nodiscard]] dex_battery_t battery() const {
        auto state = detail::sized<dex_battery_t>();
        auto error = detail::error();
        check(source_.call(
                  [&](dex_robot_t *r) { return dex_robot_battery_ex(r, 0, &state, nullptr, &error); },
                  [&](dex_diagnostic_t *c) { return dex_diagnostic_battery_ex(c, 0, &state, nullptr, &error); }),
              error);
        return state;
    }
    [[nodiscard]] dex_battery_t battery(std::chrono::milliseconds max_age,
                                        std::chrono::milliseconds *age = nullptr) const {
        auto state = detail::sized<dex_battery_t>();
        detail::aged(age, [&](uint64_t *age_ms, dex_error_t *error) {
            return source_.call(
                [&](dex_robot_t *r) { return dex_robot_battery_ex(r, detail::millis(max_age), &state, age_ms, error); },
                [&](dex_diagnostic_t *c) { return dex_diagnostic_battery_ex(c, detail::millis(max_age), &state, age_ms, error); });
        });
        return state;
    }
private:
    detail::SensorSource source_;
};

/* One mobile base of a Robot: `robot.chassis()` for the unique enabled base,
 * `robot.chassis("rear_base")` on multi-base robots. A view, not an owner. */
class Chassis {
public:
    explicit Chassis(dex_robot_t *robot, std::string name = {}) : robot_(robot), name_(std::move(name)) {}
    [[nodiscard]] const std::string &name() const noexcept { return name_; }
    void set_velocity(double vx, double vy, double wz) {
        auto error = detail::error();
        check(named() ? dex_robot_chassis_set_velocity_named(robot_, name_.c_str(), vx, vy, wz, &error)
                      : dex_robot_chassis_set_velocity(robot_, vx, vy, wz, &error),
              error);
    }
    void set_velocity(double vx, double vy, double wz, const dex_chassis_options_t &options) {
        auto error = detail::error();
        check(named() ? dex_robot_chassis_set_velocity_opt_named(robot_, name_.c_str(), vx, vy, wz, &options, &error)
                      : dex_robot_chassis_set_velocity_opt(robot_, vx, vy, wz, &options, &error),
              error);
    }
    void drive_for(double vx, double vy, double wz, std::chrono::milliseconds duration,
                   double control_hz = 50.0) {
        if (duration.count() < 0) throw std::invalid_argument("duration must be non-negative");
        auto error = detail::error();
        const auto ms = static_cast<uint64_t>(duration.count());
        check(named() ? dex_robot_chassis_drive_for_named(robot_, name_.c_str(), vx, vy, wz, ms, control_hz, &error)
                      : dex_robot_chassis_drive_for(robot_, vx, vy, wz, ms, control_hz, &error),
              error);
    }
    void stop() {
        auto error = detail::error();
        check(named() ? dex_robot_chassis_stop_named(robot_, name_.c_str(), &error)
                      : dex_robot_chassis_stop(robot_, &error),
              error);
    }
    template <typename Steering, typename Wheels>
    void set_motion_state(const Steering &steering, const Wheels &wheels) {
        auto error = detail::error();
        check(named() ? dex_robot_chassis_set_motion_state_n_named(robot_, name_.c_str(), steering.data(), steering.size(), wheels.data(), wheels.size(), &error)
                      : dex_robot_chassis_set_motion_state_n(robot_, steering.data(), steering.size(), wheels.data(), wheels.size(), &error),
              error);
    }
    /* The original two-steering/two-drive layout; brace lists work here. */
    void set_motion_state(const std::array<double, 2> &steering,
                          const std::array<double, 2> &wheel_velocity) {
        set_motion_state<std::array<double, 2>, std::array<double, 2>>(steering, wheel_velocity);
    }
    // Model-sized feedback; differential bases have no steering joints.
    [[nodiscard]] std::vector<double> values(uint32_t kind) const {
        return detail::read_array<double>([&](double *out, size_t capacity, size_t *count, dex_error_t *error) {
            return named() ? dex_robot_chassis_values_named(robot_, name_.c_str(), kind, out, capacity, count, error)
                           : dex_robot_chassis_values(robot_, kind, out, capacity, count, error);
        });
    }
    [[nodiscard]] std::vector<double> steering_angle() const { return values(0); }
    [[nodiscard]] std::vector<double> wheel_velocity() const { return values(1); }
    [[nodiscard]] std::vector<double> wheel_encoder_pos() const { return values(2); }
    // Compatibility snapshot for the original two-steering/two-drive layout.
    [[nodiscard]] dex_chassis_state_t state() const {
        auto state = detail::sized<dex_chassis_state_t>();
        auto error = detail::error();
        check(dex_robot_chassis_state(robot_, &state, &error), error);
        return state;
    }

    /* ---- closed-loop motions on the odometry (Robot::odometry()) ----
     * One continuous motion servoed on the odometry: the speed follows the
     * remaining amount down to a floor, the base is held when the target is
     * reached, and a correction pass runs only if the settled error is
     * outside the tolerance. Returns the amount measured; the base is
     * stopped on every exit. Zero fields in the options select the defaults
     * (cruise 0.5 rad/s or 0.2 m/s, 2 deg or 2 cm, 3 passes, 60 s, 300 ms).
     * Throws Error for a model whose chassis has no wheel model (see
     * Robot::supports_odometry()). */
    double turn_by(double angle_rad, const dex_closed_loop_options_t &options = closed_loop_options()) {
        double achieved = 0.0;
        auto error = detail::error();
        check(named() ? dex_robot_chassis_turn_by_named(robot_, name_.c_str(), angle_rad, &options, &achieved, &error)
                      : dex_robot_chassis_turn_by(robot_, angle_rad, &options, &achieved, &error),
              error);
        return achieved;
    }
    double move_by(double distance_m, const dex_closed_loop_options_t &options = closed_loop_options()) {
        double achieved = 0.0;
        auto error = detail::error();
        check(named() ? dex_robot_chassis_move_by_named(robot_, name_.c_str(), distance_m, &options, &achieved, &error)
                      : dex_robot_chassis_move_by(robot_, distance_m, &options, &achieved, &error),
              error);
        return achieved;
    }
    /* An initialized options struct to set fields on. */
    static dex_closed_loop_options_t closed_loop_options() {
        dex_closed_loop_options_t options{};
        dex_closed_loop_options_init(&options);
        return options;
    }
private:
    [[nodiscard]] bool named() const noexcept { return !name_.empty(); }
    dex_robot_t *robot_ = nullptr;
    std::string name_;
};

/* Dead-reckoned planar pose of the base: translation from the wheel
 * encoders through the chassis' forward kinematics, heading from an enabled
 * base IMU when there is one, else the wheels. Relative motion only; it
 * drifts without bound. The same object Python exposes as robot.odometry. */
class Odometry {
public:
    explicit Odometry(dex_robot_t *robot) : robot_(robot) {}
    [[nodiscard]] dex_pose2d_t pose() const {
        auto pose = detail::sized<dex_pose2d_t>();
        auto error = detail::error();
        check(dex_robot_odometry_pose(robot_, &pose, &error), error);
        return pose;
    }
    /* Moves the origin to the current pose; calibration is kept. */
    void reset() {
        auto error = detail::error();
        check(dex_robot_odometry_reset(robot_, &error), error);
    }
    [[nodiscard]] dex_odometry_status_t status() const {
        auto status = detail::sized<dex_odometry_status_t>();
        auto error = detail::error();
        check(dex_robot_odometry_status(robot_, &status, &error), error);
        return status;
    }
    /* "wheels" or the IMU name the heading comes from. */
    [[nodiscard]] std::string yaw_source() const { return std::string(status().yaw_source); }
    /* Ends the sampling task; the pose is kept. The next read starts a fresh one. */
    void stop() {
        auto error = detail::error();
        check(dex_robot_odometry_stop(robot_, &error), error);
    }
private:
    dex_robot_t *robot_ = nullptr;
};

class DiagnosticClient {
public:
    /* Connects with the given options; the default observes the
     * environment-selected robot. */
    static DiagnosticClient connect(const ConnectOptions &options = {}) {
        detail::require_abi();
        dex_diagnostic_t *client = nullptr;
        detail::with_options(options, [&](const dex_robot_options_t &native) {
            auto error = detail::error();
            check(dex_diagnostic_create(&native, &client, &error), error);
        });
        return DiagnosticClient(client);
    }
    static DiagnosticClient from_profile(const std::string &profile) {
        detail::require_abi();
        auto error = detail::error();
        dex_diagnostic_t *client = nullptr;
        check(dex_diagnostic_create_from_profile(
                  profile.c_str(), &client, &error),
              error);
        return DiagnosticClient(client);
    }
    static DiagnosticClient from_config_file(const std::string &path) {
        detail::require_abi();
        auto error = detail::error();
        dex_diagnostic_t *client = nullptr;
        check(dex_diagnostic_create_from_file(path.c_str(), &client, &error), error);
        return DiagnosticClient(client);
    }
    DiagnosticClient() noexcept = default;
    explicit DiagnosticClient(dex_diagnostic_t *value) noexcept : value_(value) {}
    ~DiagnosticClient() { dex_diagnostic_release(value_); }
    DiagnosticClient(const DiagnosticClient &) = delete;
    DiagnosticClient &operator=(const DiagnosticClient &) = delete;
    DiagnosticClient(DiagnosticClient &&other) noexcept
        : value_(std::exchange(other.value_, nullptr)) {}
    DiagnosticClient &operator=(DiagnosticClient &&other) noexcept {
        if (this != &other) {
            dex_diagnostic_release(value_);
            value_ = std::exchange(other.value_, nullptr);
        }
        return *this;
    }
    std::string snapshot_json(
        std::chrono::milliseconds settle_timeout = std::chrono::milliseconds{250}) const {
        if (settle_timeout.count() < 0) {
            throw std::invalid_argument("settle timeout must not be negative");
        }
        return detail::owned_string([&](char **out, dex_error_t *error) {
            return dex_diagnostic_snapshot_json(
                value_, static_cast<uint64_t>(settle_timeout.count()), out, error);
        });
    }
    std::string query_ntp_json(size_t sample_count = 30) const {
        return detail::owned_string([&](char **out, dex_error_t *error) {
            return dex_diagnostic_query_ntp_json(value_, sample_count, out, error);
        });
    }
    /* The same typed sensor reads a Robot offers, read-only. */
    [[nodiscard]] Sensors sensors() const { return Sensors(detail::SensorSource{nullptr, value_}); }
    [[nodiscard]] dex_battery_t battery() const { return sensors().battery(); }
    [[nodiscard]] dex_battery_t battery(std::chrono::milliseconds max_age,
                                        std::chrono::milliseconds *age = nullptr) const {
        return sensors().battery(max_age, age);
    }
    void shutdown() {
        auto error = detail::error();
        check(dex_diagnostic_shutdown(value_, &error), error);
    }
private:
    dex_diagnostic_t *value_ = nullptr;
};

class ControlContext {
public:
    explicit ControlContext(size_t workers = 2) {
        detail::require_abi();
        auto error = detail::error();
        check(dex_context_create(workers, &value_, &error), error);
    }
    ~ControlContext() { dex_context_release(value_); }
    ControlContext(const ControlContext &) = delete;
    ControlContext &operator=(const ControlContext &) = delete;
    ControlContext(ControlContext &&other) noexcept : value_(std::exchange(other.value_, nullptr)) {}
    [[nodiscard]] dex_context_t *native() const { return value_; }
private:
    dex_context_t *value_ = nullptr;
};

[[nodiscard]] inline std::string build_info() {
    return detail::owned_string([](char **out, dex_error_t *error) { return dex_build_info_json(out, error); });
}

class Robot {
public:
    /* Connects with the given options; the default connects to the
     * environment-selected robot, the way Robot() does in Python. */
    static Robot connect(const ConnectOptions &options = {}) {
        detail::require_abi();
        dex_robot_t *robot = nullptr;
        detail::with_options(options, [&](const dex_robot_options_t &native) {
            auto error = detail::error();
            check(dex_robot_create(&native, &robot, &error), error);
        });
        return Robot(robot);
    }
    /* Connects on an application-owned runtime. Every policy comes from
     * `options`; the process-global configure_*() setters are ignored. */
    static Robot connect(const ControlContext &context, const ConnectOptions &options = {}) {
        dex_robot_t *robot = nullptr;
        detail::with_options(options, [&](const dex_robot_options_t &native) {
            auto error = detail::error();
            check(dex_context_connect(context.native(), &native, &robot, &error), error);
        });
        return Robot(robot);
    }
    /* Positional form of connect(context, options), kept for source
     * compatibility; adjacent bools are easy to transpose. */
    static Robot from_context(const ControlContext &context, const std::string &profile,
                              bool simulation = false, bool watchdog = true, bool exit_on_termination = false) {
        ConnectOptions options;
        options.profile = profile;
        options.simulated = simulation;
        options.disable_watchdog = !watchdog;
        options.exit_on_termination = exit_on_termination;
        return connect(context, options);
    }
    static Robot from_profile(const std::string &profile) {
        detail::require_abi();
        auto error = detail::error();
        dex_robot_t *robot = nullptr;
        check(dex_robot_create_from_profile(profile.c_str(), &robot, &error), error);
        return Robot(robot);
    }
    static Robot from_config_file(const std::string &path) {
        detail::require_abi();
        auto error = detail::error();
        dex_robot_t *robot = nullptr;
        check(dex_robot_create_from_file(path.c_str(), &robot, &error), error);
        return Robot(robot);
    }
    static Robot simulated(const std::string &profile) {
        detail::require_abi();
        auto error = detail::error();
        dex_robot_t *robot = nullptr;
        check(dex_robot_create_simulated_from_profile(
                  profile.c_str(), &robot, &error),
              error);
        return Robot(robot);
    }
    /* Static configuration produced by resolved_config_json (not the operational config_json snapshot).
     * Explicit simulation avoids contacting hardware in tests and tooling. */
    static Robot from_resolved_json(const std::string &json) {
        detail::require_abi();
        auto error = detail::error();
        dex_robot_t *robot = nullptr;
        check(dex_robot_create_from_resolved_json(
                  reinterpret_cast<const uint8_t *>(json.data()), json.size(), &robot, &error), error);
        return Robot(robot);
    }
    static Robot simulated_from_resolved_json(const std::string &json) {
        detail::require_abi();
        auto error = detail::error();
        dex_robot_t *robot = nullptr;
        check(dex_robot_create_simulated_from_resolved_json(
                  reinterpret_cast<const uint8_t *>(json.data()), json.size(), &robot, &error), error);
        return Robot(robot);
    }
    /* Releases this reference and nothing else: the native library stops
     * and disconnects the robot when the last reference (this one, or the
     * last JointComponent / MotionHandle / MotionGroup obtained from it) goes
     * away without close() having been called. An exception unwinding
     * through a Robot therefore never leaves a motion running with no
     * client; call close() to stop at a moment you choose. */
    ~Robot() { dex_robot_release(value_); }
    Robot(const Robot &) = delete;
    Robot &operator=(const Robot &) = delete;
    Robot(Robot &&other) noexcept : value_(std::exchange(other.value_, nullptr)) {}
    Robot &operator=(Robot &&other) noexcept {
        if (this != &other) {
            dex_robot_release(value_);
            value_ = std::exchange(other.value_, nullptr);
        }
        return *this;
    }

    /* ---- introspection ---- */
    [[nodiscard]] std::string config_json() const { return owned(dex_robot_config_json); }
    [[nodiscard]] std::string health_json() const { return owned(dex_robot_health_json); }
    std::string initialize_arms() { return owned(dex_robot_initialize_arms_json); }
    std::string initialize_components() { return owned(dex_robot_initialize_components_json); }
    [[nodiscard]] std::string joint_state_json(const std::string &name) const {
        return detail::owned_string([&](char **out, dex_error_t *error) { return dex_robot_joint_state_json(value_, name.c_str(), out, error); });
    }
    [[nodiscard]] std::string version_info_json() const { return owned(dex_robot_version_info_json); }
    [[nodiscard]] std::string connection_report_json() const { return owned(dex_robot_connection_report_json); }
    [[nodiscard]] std::string profile_name() const { return owned(dex_robot_profile_name); }
    [[nodiscard]] std::string model_name() const { return owned(dex_robot_model_name); }
    [[nodiscard]] bool is_active() const {
        bool active = false;
        const auto status = dex_robot_is_active(value_, &active);
        if (status != DEX_OK) throw Error(status, "robot activity status unavailable");
        return active;
    }
    bool wait_for_active(std::chrono::milliseconds timeout) {
        auto error = detail::error();
        bool active = false;
        check(dex_robot_wait_for_active(value_, detail::millis(timeout), &active, &error), error);
        return active;
    }
    [[nodiscard]] bool has_component(const std::string &name) const {
        auto error = detail::error();
        bool present = false;
        check(dex_robot_has_component(value_, name.c_str(), &present, &error), error);
        return present;
    }
    [[nodiscard]] bool has_sensor(const std::string &name) const {
        auto error = detail::error();
        bool present = false;
        check(dex_robot_has_sensor(value_, name.c_str(), &present, &error), error);
        return present;
    }
    [[nodiscard]] std::vector<std::string> component_names() const {
        return detail::names(
            [&](size_t *count) { return dex_robot_component_count(value_, count); },
            [&](size_t index, char **out, dex_error_t *error) {
                return dex_robot_component_name(value_, index, out, error);
            });
    }
    [[nodiscard]] std::vector<std::string> sensor_names() const {
        return detail::names(
            [&](size_t *count) { return dex_robot_sensor_count(value_, count); },
            [&](size_t index, char **out, dex_error_t *error) {
                return dex_robot_sensor_name(value_, index, out, error);
            });
    }
    /* True once every state stream of a component or sensor is publishing. */
    bool wait_for_state(const std::string &name, std::chrono::milliseconds timeout) {
        auto error = detail::error();
        bool ready = false;
        check(dex_robot_wait_for_state(value_, name.c_str(), detail::millis(timeout), &ready,
                                       &error),
              error);
        return ready;
    }
    JointComponent joints(const std::string &name) const {
        auto error = detail::error();
        dex_component_t *component = nullptr;
        check(dex_robot_joint_component(value_, name.c_str(), &component, &error), error);
        return JointComponent(component);
    }

    /* ---- robot-level commands ---- */
    using Targets = std::map<std::string, std::vector<double>>;
    /* Direct fan-out; validated all-or-none before anything is published.
     * Each component applies its own step rule and configured freshness
     * limit unless the options say otherwise. */
    void set_joint_pos(const Targets &targets,
                       const dex_command_options_t &options = command_options()) {
        const auto view = target_view(targets);
        auto error = detail::error();
        check(dex_robot_set_joint_pos_multi(value_, view.names.data(), view.values.data(),
                                            view.counts.data(), view.names.size(), &options,
                                            &error),
              error);
    }
    /* Streamed-goal fan-out; see JointComponent::stream_joint_pos. */
    void stream_joint_pos(const Targets &targets,
                          const dex_motion_options_t &options = motion_options()) {
        const auto view = target_view(targets);
        auto error = detail::error();
        check(dex_robot_stream_joint_pos_multi(value_, view.names.data(), view.values.data(),
                                               view.counts.data(), view.names.size(), &options,
                                               &error),
              error);
    }
    MotionGroup move_to_joint_pos(const Targets &targets,
                                  const dex_motion_options_t &options = motion_options()) {
        const auto view = target_view(targets);
        auto error = detail::error();
        dex_motion_group_t *group = nullptr;
        const auto status = dex_robot_move_to_joint_pos_multi(value_, view.names.data(), view.values.data(),
                                                view.counts.data(), view.names.size(), &options,
                                                &group, &error);
        MotionGroup owned(group);
        if (status != DEX_OK) {
            if (group) throw MotionGroupStartError(status, error.message, std::move(owned));
            check(status, error);
        }
        return owned;
    }
    std::pair<MotionGroup, std::string> start_joint_motions(
        const Targets &targets, const dex_motion_options_t &options = motion_options()) {
        const auto view = target_view(targets);
        std::unique_ptr<dex_motion_group_t, decltype(&dex_motion_group_release)> group(nullptr, dex_motion_group_release);
        auto report = detail::owned_string([&](char **out, dex_error_t *error) {
            dex_motion_group_t *raw = nullptr;
            const auto status = dex_robot_start_joint_motions_opt(value_, view.names.data(), view.values.data(),
                view.counts.data(), view.names.size(), &options, &raw, out, error);
            group.reset(raw);
            return status;
        });
        return {MotionGroup(group.release()), std::move(report)};
    }

    /* Per-component waypoint tracks (tick-major rows), all with the same
     * tick count. velocities is optional per track. max_tracking_error <= 0
     * disables the per-tick guard. */
    using Tracks = std::map<std::string, std::vector<std::vector<double>>>;
    void execute_trajectory(const Tracks &positions, double control_hz,
                            const Tracks &velocities = {}, double max_tracking_error = 0.0) {
        auto options = trajectory_options();
        options.max_tracking_error = max_tracking_error;
        execute_trajectory(positions, control_hz, velocities, options);
    }
    /* options.max_step_rad tunes or (DEX_STEP_DISABLED) opts out of the
     * start-distance and waypoint-spacing guard. */
    void execute_trajectory(const Tracks &positions, double control_hz, const Tracks &velocities,
                            const dex_trajectory_options_t &options) {
        if (positions.empty()) throw std::invalid_argument("at least one track is required");
        for (const auto &[name, rows] : velocities) {
            (void)rows;
            if (positions.find(name) == positions.end()) {
                throw std::invalid_argument("velocity track '" + name + "' has no position track");
            }
        }
        std::vector<const char *> names;
        std::vector<std::vector<double>> flat_positions, flat_velocities;
        std::vector<const double *> position_ptrs, velocity_ptrs;
        std::vector<size_t> joint_counts;
        const size_t ticks = positions.begin()->second.size();
        for (const auto &[name, rows] : positions) {
            if (rows.size() != ticks) {
                throw std::invalid_argument("every track must have the same tick count");
            }
            const size_t joints = rows.empty() ? 0 : rows.front().size();
            flat_positions.push_back(flatten(rows, joints));
            const auto velocity = velocities.find(name);
            if (velocity != velocities.end()) {
                if (velocity->second.size() != ticks) {
                    throw std::invalid_argument("velocity track must match the tick count");
                }
                flat_velocities.push_back(flatten(velocity->second, joints));
            } else {
                flat_velocities.emplace_back();
            }
            names.push_back(name.c_str());
            joint_counts.push_back(joints);
        }
        for (size_t index = 0; index < names.size(); ++index) {
            position_ptrs.push_back(flat_positions[index].data());
            velocity_ptrs.push_back(flat_velocities[index].empty() ? nullptr
                                                                  : flat_velocities[index].data());
        }
        auto error = detail::error();
        check(dex_robot_execute_trajectory_opt(value_, names.data(), position_ptrs.data(),
                                               velocities.empty() ? nullptr : velocity_ptrs.data(),
                                               ticks, joint_counts.data(), names.size(),
                                               control_hz, &options, &error),
              error);
    }
    void stop_all() {
        auto error = detail::error();
        check(dex_robot_stop_all(value_, &error), error);
    }
    [[nodiscard]] std::vector<double> compensate_torso_pitch(const std::vector<double> &values,
                                               const std::string &part) const {
        return detail::read_array<double>([&](double *out, size_t capacity, size_t *count,
                                              dex_error_t *error) {
            return dex_robot_compensate_torso_pitch(value_, values.data(), values.size(),
                                                    part.c_str(), out, capacity, count, error);
        });
    }
    [[nodiscard]] double torso_pitch() const {
        auto error = detail::error();
        double pitch = 0.0;
        check(dex_robot_torso_pitch(value_, &pitch, &error), error);
        return pitch;
    }

    /* ---- maintenance and diagnostics ---- */
    void clear_error(const std::string &component) {
        auto error = detail::error();
        check(dex_robot_clear_error(value_, component.c_str(), &error), error);
    }
    /* Clears every maintenance-capable controller; the per-component report
     * as JSON (Python: clear_errors()). */
    std::string clear_errors_json() { return owned(dex_robot_clear_all_errors_json); }
    [[deprecated("use clear_errors_json")]]
    std::string clear_all_errors_json() { return clear_errors_json(); }
    /* board is "arm", "torso" or "chassis". */
    void reboot(const std::string &board) {
        auto error = detail::error();
        check(dex_robot_reboot(value_, board.c_str(), &error), error);
    }
    std::vector<uint8_t> query(const std::string &name, const std::vector<uint8_t> &payload = {}) {
        return detail::owned_bytes([&](dex_buffer_t *out, dex_error_t *error) {
            return dex_robot_query(value_, name.c_str(), payload.data(), payload.size(), out,
                                   error);
        });
    }
    std::string query_ntp_json(size_t sample_count = 30) {
        return detail::owned_string([&](char **out, dex_error_t *error) {
            return dex_robot_query_ntp_json(value_, sample_count, out, error);
        });
    }
    std::string component_status_json() { return owned(dex_robot_component_status_json); }
    std::string take_safety_events_json() { return owned(dex_robot_take_safety_events_json); }
    /* Blocks for the next safety event; empty on timeout. Buffered events
     * come first, so a host thread can loop on this instead of polling. */
    std::optional<std::string> wait_safety_event_json(std::chrono::milliseconds timeout) {
        auto json = detail::owned_string([&](char **out, dex_error_t *error) {
            return dex_robot_wait_safety_event_json(value_, detail::millis(timeout), out, error);
        });
        if (json == "null") return std::nullopt;
        return json;
    }
    void reconnect() {
        auto error = detail::error();
        check(dex_robot_reconnect(value_, &error), error);
    }
    /* Stops motion and the base, then disconnects, now. Idempotent; wakes
     * a pending wait_safety_event_json(). Handles stay valid but refuse
     * commands. */
    void close() {
        auto error = detail::error();
        check(dex_robot_close(value_, &error), error);
    }
    /* The original name of close(). */
    void shutdown() { close(); }

    /* ---- E-stop and battery ---- */
    [[nodiscard]] dex_estop_status_t estop_status() const {
        auto status = detail::sized<dex_estop_status_t>();
        auto error = detail::error();
        check(dex_robot_estop_status(value_, &status, &error), error);
        return status;
    }
    void estop_activate() { estop_set(true); }
    void estop_deactivate() { estop_set(false); }
    void estop_set(bool enabled) {
        auto error = detail::error();
        check(dex_robot_estop_set(value_, enabled, &error), error);
    }
    [[nodiscard]] dex_battery_t battery() const { return sensors().battery(); }
    /* Throws Error(DEX_STALE_STATE) for a sample that arrived more than
     * max_age ago; *age (optional) receives the arrival age either way. */
    [[nodiscard]] dex_battery_t battery(std::chrono::milliseconds max_age,
                                        std::chrono::milliseconds *age = nullptr) const {
        return sensors().battery(max_age, age);
    }

    /* ---- heartbeat supervision ---- */
    /* Pauses heartbeat staleness supervision indefinitely (E-stop
     * monitoring is unaffected). SAFETY CAVEAT: while paused, a dead robot
     * server goes unnoticed by heartbeat supervision -- pair every pause
     * with a guaranteed resume_heartbeat() (RAII or try/catch). On resume,
     * supervision re-arms from the next observed sample. Idempotent. */
    void pause_heartbeat() {
        auto error = detail::error();
        check(dex_robot_pause_heartbeat(value_, &error), error);
    }
    void resume_heartbeat() {
        auto error = detail::error();
        check(dex_robot_resume_heartbeat(value_, &error), error);
    }

    /* ---- views: the mobile base and the typed sensors ----
     * The same objects Python exposes as robot.chassis and robot.sensors. */
    [[nodiscard]] Chassis chassis() const { return Chassis(value_); }
    [[nodiscard]] Chassis chassis(const std::string &name) const { return Chassis(value_, name); }
    [[nodiscard]] Sensors sensors() const { return Sensors(detail::SensorSource{value_, nullptr}); }

    /* ---- odometry (optional per robot model) ----
     * supports_odometry() is false when the chassis driver has no wheel
     * model; odometry() then throws and every other chassis call still works.
     * odometry() starts the estimator with defaults on first use (heading
     * from an enabled base IMU such as "chassis_imu", else the wheels);
     * start_odometry() restarts it with a chosen yaw source ("auto",
     * "wheels" or an enabled IMU's name) and options. */
    [[nodiscard]] bool supports_odometry() const {
        uint8_t supported = 0;
        auto error = detail::error();
        check(dex_robot_odometry_supported(value_, &supported, &error), error);
        return supported != 0;
    }
    [[nodiscard]] Odometry odometry() const {
        auto odometry = Odometry(value_);
        (void)odometry.status(); // starts the estimator, or throws where unsupported
        return odometry;
    }
    Odometry start_odometry(const std::string &yaw_source = "auto",
                            const dex_odometry_options_t *options = nullptr) {
        auto error = detail::error();
        check(dex_robot_odometry_start(value_, yaw_source.c_str(), options, &error), error);
        return Odometry(value_);
    }

    /* ---- deprecated flat spellings of the views above (removed at 1.0) ---- */
    [[deprecated("use chassis().values(kind)")]]
    [[nodiscard]] std::vector<double> chassis_values(uint32_t kind) const { return chassis().values(kind); }
    [[deprecated("use chassis().steering_angle()")]]
    [[nodiscard]] std::vector<double> chassis_steering_angle() const { return chassis().steering_angle(); }
    [[deprecated("use chassis().wheel_velocity()")]]
    [[nodiscard]] std::vector<double> chassis_wheel_velocity() const { return chassis().wheel_velocity(); }
    [[deprecated("use chassis().wheel_encoder_pos()")]]
    [[nodiscard]] std::vector<double> chassis_wheel_encoder_pos() const { return chassis().wheel_encoder_pos(); }
    [[deprecated("use chassis().state()")]]
    [[nodiscard]] dex_chassis_state_t chassis_state() const { return chassis().state(); }
    [[deprecated("use chassis().drive_for(...)")]]
    void drive_chassis_for(double vx, double vy, double wz, std::chrono::milliseconds duration,
                           double control_hz = 50.0) {
        chassis().drive_for(vx, vy, wz, duration, control_hz);
    }
    [[deprecated("use chassis().set_velocity(...)")]]
    void set_chassis_velocity(double vx, double vy, double wz) { chassis().set_velocity(vx, vy, wz); }
    [[deprecated("use chassis().set_velocity(...)")]]
    void set_chassis_velocity(double vx, double vy, double wz, const dex_chassis_options_t &options) {
        chassis().set_velocity(vx, vy, wz, options);
    }
    template <typename Steering, typename Wheels>
    [[deprecated("use chassis().set_motion_state(...)")]]
    void set_chassis_motion_state(const Steering &steering, const Wheels &wheels) {
        chassis().set_motion_state(steering, wheels);
    }
    [[deprecated("use chassis().set_motion_state(...)")]]
    void set_chassis_motion_state(const std::array<double, 2> &steering,
                                  const std::array<double, 2> &wheel_velocity) {
        chassis().set_motion_state(steering, wheel_velocity);
    }
    [[deprecated("use chassis().stop()")]]
    void stop_chassis() { chassis().stop(); }
    [[deprecated("use chassis(name).set_velocity(...)")]]
    void set_chassis_velocity(const std::string &name, double vx, double vy, double wz) {
        chassis(name).set_velocity(vx, vy, wz);
    }
    [[deprecated("use chassis(name).set_velocity(...)")]]
    void set_chassis_velocity(const std::string &name, double vx, double vy, double wz,
                              const dex_chassis_options_t &options) {
        chassis(name).set_velocity(vx, vy, wz, options);
    }
    [[deprecated("use chassis(name).drive_for(...)")]]
    void drive_chassis_for(const std::string &name, double vx, double vy, double wz,
                           std::chrono::milliseconds duration, double control_hz = 50.0) {
        chassis(name).drive_for(vx, vy, wz, duration, control_hz);
    }
    [[deprecated("use chassis(name).stop()")]]
    void stop_chassis(const std::string &name) { chassis(name).stop(); }
    [[deprecated("use chassis(name).values(kind)")]]
    [[nodiscard]] std::vector<double> chassis_values(const std::string &name, uint32_t kind) const {
        return chassis(name).values(kind);
    }
    [[deprecated("use chassis(name).steering_angle()")]]
    [[nodiscard]] std::vector<double> chassis_steering_angle(const std::string &name) const { return chassis(name).steering_angle(); }
    [[deprecated("use chassis(name).wheel_velocity()")]]
    [[nodiscard]] std::vector<double> chassis_wheel_velocity(const std::string &name) const { return chassis(name).wheel_velocity(); }
    [[deprecated("use chassis(name).wheel_encoder_pos()")]]
    [[nodiscard]] std::vector<double> chassis_wheel_encoder_pos(const std::string &name) const { return chassis(name).wheel_encoder_pos(); }
    template <typename Steering, typename Wheels>
    [[deprecated("use chassis(name).set_motion_state(...)")]]
    void set_chassis_motion_state(const std::string &name, const Steering &steering, const Wheels &wheels) {
        chassis(name).set_motion_state(steering, wheels);
    }
    [[deprecated("use sensors().imu(name)")]]
    [[nodiscard]] dex_imu_t imu(const std::string &name) const { return sensors().imu(name); }
    [[deprecated("use sensors().imu(name, max_age, age)")]]
    [[nodiscard]] dex_imu_t imu(const std::string &name, std::chrono::milliseconds max_age,
                                std::chrono::milliseconds *age = nullptr) const {
        return sensors().imu(name, max_age, age);
    }
    [[deprecated("use sensors().ultrasonic(name)")]]
    [[nodiscard]] dex_ultrasonic_t ultrasonic(const std::string &name) const { return sensors().ultrasonic(name); }
    [[deprecated("use sensors().ultrasonic(name, max_age, age)")]]
    [[nodiscard]] dex_ultrasonic_t ultrasonic(const std::string &name, std::chrono::milliseconds max_age,
                                              std::chrono::milliseconds *age = nullptr) const {
        return sensors().ultrasonic(name, max_age, age);
    }
    [[deprecated("use sensors().lidar_2d(name, max_age)")]]
    [[nodiscard]] Lidar2dScan lidar_2d(const std::string &name, std::chrono::milliseconds max_age = {}) const {
        return sensors().lidar_2d(name, max_age);
    }
    [[deprecated("use sensors().lidar_3d(name, max_age)")]]
    [[nodiscard]] Lidar3dCloud lidar_3d(const std::string &name, std::chrono::milliseconds max_age = {}) const {
        return sensors().lidar_3d(name, max_age);
    }
    [[deprecated("use sensors().wrench(component)")]]
    [[nodiscard]] Wrench wrench(const std::string &component) const { return sensors().wrench(component); }
    [[deprecated("use sensors().wrench(component, max_age, age)")]]
    [[nodiscard]] Wrench wrench(const std::string &component, std::chrono::milliseconds max_age,
                                std::chrono::milliseconds *age = nullptr) const {
        return sensors().wrench(component, max_age, age);
    }
    [[deprecated("use sensors().temperature_json(component)")]]
    [[nodiscard]] std::string temperature_json(const std::string &component) const { return sensors().temperature_json(component); }
    [[deprecated("use sensors().sensor_json(name, role)")]]
    [[nodiscard]] std::string sensor_json(const std::string &name, const std::string &role = {}) const {
        return sensors().sensor_json(name, role);
    }
    [[deprecated("use sensors().camera(sensor).info_json(timeout)")]]
    [[nodiscard]] std::string camera_info_json(const std::string &sensor,
                                               std::chrono::milliseconds timeout = std::chrono::milliseconds{2000}) const {
        return sensors().camera(sensor).info_json(timeout);
    }
    [[deprecated("use sensors().camera(sensor).streams()")]]
    [[nodiscard]] std::vector<std::string> camera_streams(const std::string &sensor) const { return sensors().camera(sensor).streams(); }
    [[deprecated("use sensors().camera(sensor).subscribe(stream, history)")]]
    void camera_subscribe(const std::string &sensor, const std::string &stream, size_t history = 0) {
        sensors().camera(sensor).subscribe(stream, history);
    }
    [[deprecated("use sensors().camera(sensor).unsubscribe(stream)")]]
    void camera_unsubscribe(const std::string &sensor, const std::string &stream) { sensors().camera(sensor).unsubscribe(stream); }
    [[deprecated("use sensors().camera(sensor).latest_frame(stream)")]]
    [[nodiscard]] std::optional<Frame> camera_latest_frame(const std::string &sensor, const std::string &stream) const {
        return sensors().camera(sensor).latest_frame(stream);
    }
    [[deprecated("use sensors().camera(sensor).stats(stream)")]]
    [[nodiscard]] dex_stream_stats_t camera_stats(const std::string &sensor, const std::string &stream) const {
        return sensors().camera(sensor).stats(stream);
    }
    [[deprecated("use sensors().camera(sensor).is_active(stream, window)")]]
    [[nodiscard]] bool camera_is_active(const std::string &sensor, const std::string &stream = {},
                                        std::chrono::milliseconds window = std::chrono::seconds{1}) const {
        return sensors().camera(sensor).is_active(stream, window);
    }

    /* Test support on simulated robots: while enabled, tracked motions stay
     * Running so timeout paths can be exercised; disabling releases held
     * motions as Succeeded. Throws on robots that are not simulated. */
    void simulation_hold_motions(bool hold) {
        auto error = detail::error();
        check(dex_robot_simulation_hold_motions(value_, hold, &error), error);
    }

private:
    explicit Robot(dex_robot_t *value) noexcept : value_(value) {}


    template <typename Call> std::string owned(Call call) const {
        return detail::owned_string([&](char **out, dex_error_t *error) {
            return call(value_, out, error);
        });
    }
    struct TargetView {
        std::vector<const char *> names;
        std::vector<const double *> values;
        std::vector<size_t> counts;
    };
    static TargetView target_view(const Targets &targets) {
        TargetView view;
        for (const auto &[name, values] : targets) {
            view.names.push_back(name.c_str());
            view.values.push_back(values.data());
            view.counts.push_back(values.size());
        }
        return view;
    }
    static std::vector<double> flatten(const std::vector<std::vector<double>> &rows,
                                       size_t joints) {
        std::vector<double> flat;
        flat.reserve(rows.size() * joints);
        for (const auto &row : rows) {
            if (row.size() != joints) {
                throw std::invalid_argument("every row of a track must have the same joint count");
            }
            flat.insert(flat.end(), row.begin(), row.end());
        }
        return flat;
    }
    dex_robot_t *value_ = nullptr;
};

}  // namespace dexcontrol
