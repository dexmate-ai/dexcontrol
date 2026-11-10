// Copyright (C) 2026 Dexmate Inc.
// SPDX-License-Identifier: Apache-2.0

// Display robot topology, observed state, and server diagnostics.
//
// Use a read-only DiagnosticClient and a 0.25 s state observation window, then print the snapshot. Optional --enable-sensor includes declared optional sensors. Does not enable motors or issue stop/motion commands; no simulation connection is supported.
//
// Formatted native diagnostic tables.

#include "example.hpp"
#include "json.hpp"

#include <algorithm>
#include <cctype>
#include <set>

namespace {

using examples::Json;

const std::string DASH = "--";
const std::string NA = "N/A";

using Row = std::vector<std::string>;

void plain_table(const std::string &title, const Row &headers, const std::vector<Row> &rows) {
    std::vector<size_t> widths;
    for (const auto &header : headers) widths.push_back(header.size());
    for (const auto &row : rows) {
        for (size_t index = 0; index < row.size() && index < widths.size(); ++index) {
            widths[index] = std::max(widths[index], row[index].size());
        }
    }
    auto line = [&](const Row &cells) {
        std::string out;
        for (size_t index = 0; index < cells.size(); ++index) {
            if (index > 0) out += "  ";
            out += cells[index];
            if (index + 1 < cells.size()) {
                out += std::string(widths[index] > cells[index].size() ? widths[index] - cells[index].size() : 0, ' ');
            }
        }
        std::cout << out << "\n";
    };
    std::cout << "\n" << title << "\n";
    line(headers);
    Row rule;
    for (const auto width : widths) {
        std::string dashes;
        for (size_t i = 0; i < width; ++i) dashes += "─";
        rule.push_back(dashes);
    }
    // Box-drawing characters are three bytes each, so the rule is joined by hand.
    std::string rule_line;
    for (size_t index = 0; index < rule.size(); ++index) rule_line += (index > 0 ? "  " : "") + rule[index];
    std::cout << rule_line << "\n";
    for (const auto &row : rows) line(row);
}

std::string number(const Json &values, size_t index, const std::string &unit = "") {
    const auto &value = values[index];
    if (value.is_null()) return DASH;
    if (!value.is_number()) return value.text();
    std::ostringstream out;
    out << std::showpos << std::fixed << std::setprecision(4) << value.as_number() << unit;
    return out.str();
}

std::string fixed(double value, int precision) {
    std::ostringstream out;
    out << std::fixed << std::setprecision(precision) << value;
    return out.str();
}

std::string humanize(std::string text) {
    bool start = true;
    for (auto &c : text) {
        if (c == '_') c = ' ';
        if (start && std::isalpha(static_cast<unsigned char>(c))) c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
        start = c == ' ';
    }
    return text;
}

std::string humanize(const Json &value) { return humanize(value.is_null() ? DASH : value.text()); }

std::string join(const Json &list, const std::string &fallback) {
    std::string out;
    for (const auto &item : list.items()) out += (out.empty() ? "" : ", ") + item.text();
    return out.empty() ? fallback : out;
}

// Vega's bring-up report: the hand each side configures and what the robot says is there.
std::string end_effectors(const Json &snapshot) {
    // The core renders the line (same words as the connect log) so a
    // mismatch names the profile and the variant to select. It quotes the
    // robot's hand-type reply, so control characters are scrubbed before
    // they reach the terminal.
    const auto &summary = snapshot["model_bringup"]["end_effectors"]["summary"];
    std::string line = summary.is_string() ? summary.text("") : "";
    if (line.empty()) return DASH;
    for (auto &ch : line) {
        if (std::iscntrl(static_cast<unsigned char>(ch))) ch = ' ';
    }
    return line;
}

void show_overview(const Json &snapshot) {
    plain_table("Robot overview", {"Field", "Value"},
                {{"Profile", snapshot["profile_name"].text(DASH)},
                 {"Transport", "DexComm"},
                 {"Components", join(snapshot["component_names"], DASH)},
                 {"Sensors", join(snapshot["sensor_names"], "None enabled")},
                 {"End effectors", end_effectors(snapshot)}});
}

void show_joint_state(const Json &snapshot) {
    const auto &states = snapshot["joint_states"];
    const auto &configured = snapshot["joint_names"];
    const auto &errors_by_key = snapshot["errors"];
    std::set<std::string> components;
    for (const auto &[name, _] : configured.members()) components.insert(name);
    for (const auto &[name, _] : states.members()) components.insert(name);

    std::vector<Row> rows;
    // Iterate the configured components, not just the ones that reported: a
    // component that published nothing renders as N/A rather than vanishing.
    for (const auto &component : components) {
        const auto &state = states[component];
        const auto &names = configured[component];
        if (state.is_null()) {
            const auto reason = errors_by_key["joint_state:" + component].text("no state received");
            if (names.items().empty()) {
                rows.push_back({component, DASH, NA, NA, NA, NA, "No data (" + reason + ")"});
            }
            for (const auto &joint : names.items()) {
                rows.push_back({component, joint.text(), NA, NA, NA, NA, "No data (" + reason + ")"});
            }
            continue;
        }
        const auto &positions = state["position"];
        std::set<size_t> error_indices;
        for (const auto &index : state["error_joint_indices"].items()) {
            error_indices.insert(static_cast<size_t>(index.as_number()));
        }
        const auto &errors = state["errors"];
        const size_t count = std::max(names.size(), positions.size());
        for (size_t index = 0; index < count; ++index) {
            const std::string joint =
                index < names.size() ? names[index].text() : "joint_" + std::to_string(index + 1);
            const auto &error = errors.contains(joint) ? errors[joint] : errors[std::to_string(index)];
            std::string status = "OK";
            if (!error.is_null() && !(error.is_string() && error.as_string().empty())) {
                status = error.text();
            } else if (error_indices.count(index) != 0) {
                status = "Error";
            }
            rows.push_back({component, joint, number(positions, index), number(state["velocity"], index),
                            number(state["current"], index), number(state["torque"], index), status});
        }
    }
    if (rows.empty()) rows.push_back({DASH, "No joint state received", DASH, DASH, DASH, DASH, DASH});
    plain_table("Joint state", {"Component", "Joint", "Position", "Velocity", "Current", "Torque", "Status"},
                rows);
}

/* The E-stop row, absence included: the robot server publishes E-stop state
 * only while the button is engaged, so a released E-stop publishes nothing.
 * That is the normal state of a healthy robot, and the snapshot says so with
 * source=no_event_published plus state_known. */
Row estop_row(const Json &estop) {
    if (estop["source"].as_string() == "no_event_published") {
        if (estop["state_known"].as_bool()) {
            return {"E-stop", "Ready", "not engaged (no event published; board connected)"};
        }
        const auto &board = estop["board_connected"];
        const std::string reason = board.is_bool() && !board.as_bool() ? "board disconnected"
                                                                        : "board status unavailable";
        return {"E-stop", "Unknown", "no event published; " + reason};
    }
    bool pressed = false;
    for (const char *name : {"left_base_estop_enabled", "right_base_estop_enabled", "torso_estop_enabled",
                             "remote_estop_enabled"}) {
        pressed = pressed || estop[name].as_bool();
    }
    const bool software = estop["software_estop_enabled"].as_bool();
    const std::string details = std::string("button=") + (pressed ? "pressed" : "released") +
                                ", software=" + (software ? "on" : "off");
    return {"E-stop", pressed || software ? "STOPPED" : "Clear", details};
}

void show_monitoring(const Json &snapshot) {
    const auto &monitoring = snapshot["monitoring"];
    std::vector<Row> rows;
    if (monitoring["estop"].is_object()) rows.push_back(estop_row(monitoring["estop"]));
    const auto &heartbeat = monitoring["heartbeat"];
    if (heartbeat.is_object()) {
        std::string status = heartbeat["monitoring_disabled"].as_bool() ? "Disabled"
                             : heartbeat["paused"].as_bool()            ? "Paused"
                             : heartbeat["is_active"].as_bool()         ? "Active"
                                                                        : "No sample";
        rows.push_back({"Heartbeat", status, "Native supervision"});
    }
    const auto &battery = monitoring["battery"];
    if (battery.is_object()) {
        const auto &percentage = battery["percentage"];
        const std::string status = percentage.is_number() ? fixed(percentage.as_number(), 0) + "%" : DASH;
        std::string details;
        if (battery["voltage"].is_number()) details += fixed(battery["voltage"].as_number(), 2) + " V";
        if (battery["current"].is_number()) {
            details += (details.empty() ? "" : ", ") + fixed(battery["current"].as_number(), 2) + " A";
        }
        for (const auto &[key, unit] : std::vector<std::pair<std::string, std::string>>{
                 {"power", " W"}, {"temperature", " °C"}}) {
            if (battery[key].is_number()) {
                details += (details.empty() ? "" : ", ") + fixed(battery[key].as_number(), 2) + unit;
            }
        }
        rows.push_back({"Battery", status, details.empty() ? DASH : details});
    }
    if (rows.empty()) return;
    plain_table("Monitoring", {"Monitor", "Status", "Details"}, rows);
}

void show_versions(const Json &snapshot) {
    const auto &info = snapshot["version_info"];
    if (!info.is_object()) {
        plain_table("Versions", {"Item", "Value"}, {{"Server versions", "Unavailable"}});
        return;
    }
    std::vector<Row> rows;
    for (const auto &[board, entry] : info["firmware_version"].members()) {
        rows.push_back({board, entry["hardware_version"].text(DASH), entry["software_version"].text(DASH),
                        entry["release_version"].text(DASH), entry["main_hash"].text(DASH),
                        entry["compile_time"].text(DASH)});
    }
    plain_table("Firmware versions", {"Board", "Hardware", "Software", "Release", "Commit", "Compiled"}, rows);
    std::cout << "Robot server " << info["version"].text(DASH) << "  •  minimum client "
              << info["min_client_version"].text(DASH) << "\n";
}

void show_component_status(const Json &snapshot) {
    const auto &report = snapshot["component_status"];
    std::set<std::string> configured;
    for (const auto &name : snapshot["component_names"].items()) configured.insert(name.text());
    for (const auto &name : snapshot["sensor_names"].items()) configured.insert(name.text());
    std::vector<Row> rows;
    for (const auto &[name, state] : report["states"].members()) {
        const auto &error = state["error"];
        const std::string message = error.is_object() ? error["error_message"].text() : "";
        rows.push_back({name, humanize(state["connection"]), humanize(state["operation"]),
                        message.empty() ? DASH : message,
                        configured.count(name) != 0 ? "Configured" : "Server only"});
    }
    if (rows.empty()) rows.push_back({DASH, "Unavailable", DASH, DASH, DASH});
    plain_table("Component status", {"Component", "Connection", "Operation", "Error", "Scope"}, rows);
}

void show_ntp(const Json &snapshot) {
    const auto &ntp = snapshot["ntp"];
    if (!ntp.is_object()) return;
    std::vector<Row> rows;
    const auto &success = ntp["success"];
    if (success.is_bool() && !success.as_bool()) {
        rows.push_back({"Status", "Unavailable"});
        rows.push_back({"Message", ntp["message"].text("Query failed")});
    } else {
        rows.push_back({"Status", success.is_bool() ? "Synchronized" : "Reported"});
        if (ntp.contains("offset")) {
            std::ostringstream out;
            out << std::showpos << std::fixed << std::setprecision(3) << ntp["offset"].as_number() * 1000.0
                << " ms";
            rows.push_back({"Clock offset", out.str()});
        }
        if (ntp.contains("rtt")) rows.push_back({"Round-trip time", fixed(ntp["rtt"].as_number() * 1000.0, 3) + " ms"});
        if (ntp.contains("replies") || ntp.contains("samples")) {
            rows.push_back({"Samples", ntp["replies"].text("0") + " / " + ntp["samples"].text("0") + " replies"});
        }
        for (const auto &[key, value] : ntp.members()) {
            static const std::set<std::string> shown{"success", "message", "offset", "rtt", "replies", "samples"};
            if (shown.count(key) == 0) rows.push_back({humanize(key), value.text()});
        }
        if (!ntp["message"].text().empty()) rows.push_back({"Message", ntp["message"].text()});
    }
    plain_table("Clock synchronization", {"Metric", "Value"}, rows);
}

void show_connection_timing(const Json &snapshot) {
    static const std::map<std::string, std::string> labels{
        {"configuration", "Configuration"},         {"discovery", "Transport discovery"},
        {"operational_config", "Operational model"}, {"subscription_activation", "Subscriptions"},
        {"version_check", "Version check"},         {"safety_startup", "Safety setup"},
        {"readiness", "Readiness"}};
    const auto &connection = snapshot["connection"];
    std::vector<Row> rows;
    for (const auto &phase : connection["phases"].items()) {
        const auto key = phase["phase"].text();
        const auto label = labels.count(key) != 0 ? labels.at(key) : humanize(phase["phase"]);
        rows.push_back({label, fixed(phase["duration_ms"].as_number(), 2) + " ms"});
    }
    rows.push_back({"Total", fixed(connection["total_ms"].as_number(), 2) + " ms"});
    plain_table("Connection timing", {"Phase", "Duration"}, rows);
}

void show_errors(const Json &snapshot) {
    const auto &errors = snapshot["errors"];
    if (errors.members().empty()) return;
    std::vector<Row> rows;
    for (const auto &[name, message] : errors.members()) rows.push_back({name, message.text()});
    plain_table("Unavailable diagnostics", {"Section", "Reason"}, rows);
}

}  // namespace

int main(int argc, char **argv) {
    return examples::run(argc, argv,
                         "Display robot topology, observed state, and server diagnostics. Use a read-only DiagnosticClient and a 0.25 s state observation window, then print the snapshot. Optional --enable-sensor includes declared optional sensors. Does not enable motors or issue stop/motion commands; no simulation connection is supported. Formatted native diagnostic tables.",
                         [](const examples::Args &args) {
        const double settle = std::max(args.number("settle-timeout", 0.25), 0.0);
        auto client = args.diagnostics();
        const auto snapshot = Json::parse(client.snapshot_json(examples::millis(settle)));
        show_overview(snapshot);
        show_joint_state(snapshot);
        show_monitoring(snapshot);
        show_versions(snapshot);
        show_component_status(snapshot);
        show_ntp(snapshot);
        show_connection_timing(snapshot);
        show_errors(snapshot);
    }, {"settle-timeout"});
}
