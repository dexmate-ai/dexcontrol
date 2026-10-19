// Copyright (C) 2026 Dexmate Inc.
//
// This software is dual-licensed:
//
// 1. GNU Affero General Public License v3.0 (AGPL-3.0)
//    See LICENSE-AGPL for details
//
// 2. Commercial License
//    For commercial licensing terms, contact: contact@dexmate.ai

#pragma once

// Shared by the camera examples: subscribe to every requested stream, print
// the shape of each frame that arrives until interrupted or the sample limit, and
// unsubscribe on every exit path.

#include "example.hpp"
#include <optional>
#include <map>
#include <memory>
#ifdef DEXCONTROL_WITH_VIEWER
#include <dexcontrol/viewer.hpp>
#endif

namespace examples {

#ifdef DEXCONTROL_WITH_VIEWER
using ImageViewer = dexcontrol::viewer::Viewer;
inline std::unique_ptr<ImageViewer> image_viewer(const Args &args) {
    if (args.flag("no-display")) return nullptr;
    dexcontrol::viewer::Options options;
    options.application_id = "dexcontrol-cameras";
    options.viewer_url = args.text("viewer-url");
    return std::make_unique<ImageViewer>(options);
}
#else
struct ImageViewer {};
inline std::unique_ptr<ImageViewer> image_viewer(const Args &args) {
    if (!args.flag("no-display")) throw std::runtime_error(
        "viewer support is not built; use ./run cpp or build with DEXCONTROL_WITH_VIEWER=ON; --no-display prints metadata");
    return nullptr;
}
#endif

inline const char *encoding_name(dexcontrol::FrameEncoding encoding) {
    switch (encoding) {
    case dexcontrol::FrameEncoding::Rgb8: return "rgb8";
    case dexcontrol::FrameEncoding::Bgr8: return "bgr8";
    case dexcontrol::FrameEncoding::Gray8: return "gray8";
    case dexcontrol::FrameEncoding::Depth32F: return "depth32f";
    }
    return "unknown";
}

inline void print_frames(dexcontrol::Robot &robot, const std::vector<std::string> &sensors,
                         const std::vector<std::string> &streams, std::optional<int> samples,
                         double period, ImageViewer *viewer = nullptr) {
#ifndef DEXCONTROL_WITH_VIEWER
    (void)viewer;
#endif
    if (samples && *samples < 0) throw std::invalid_argument("--samples must be non-negative");
    std::vector<std::pair<std::string, std::string>> subscribed;
    for (const auto &sensor : sensors) {
        for (const auto &stream : streams.empty() ? robot.camera_streams(sensor) : streams) {
            subscribed.emplace_back(sensor, stream);
        }
    }
    for (const auto &[sensor, stream] : subscribed) robot.camera_subscribe(sensor, stream);
    try {
        std::map<std::string, uint64_t> previous;
        while ((!samples || *samples > 0) && !interrupted()) {
#ifdef DEXCONTROL_WITH_VIEWER
            if (viewer && !viewer->is_open()) break;
#endif
            for (const auto &[sensor, stream] : subscribed) {
                const auto frame = robot.camera_latest_frame(sensor, stream);
                if (!frame) continue;
#ifdef DEXCONTROL_WITH_VIEWER
                if (viewer) {
                    const auto path = sensor + "/" + stream;
                    const auto stamp = frame->timestamp_ns();
                    if (stamp != 0 && previous.count(path) && previous[path] == stamp) continue;
                    const auto format = static_cast<dexcontrol::viewer::PixelFormat>(frame->encoding());
                    viewer->log_image(path, {frame->width(), frame->height(), format, frame->data(), frame->size(),
                        stamp != 0 && stamp <= static_cast<uint64_t>(INT64_MAX) ? static_cast<int64_t>(stamp) : INT64_MIN});
                    previous[path] = stamp;
                    continue;
                }
#endif
                std::cout << sensor << "/" << stream << ": shape=(" << frame->height() << ", "
                          << frame->width() << ") " << encoding_name(frame->encoding()) << " "
                          << frame->size() << " bytes\n";
            }
            if (samples) --*samples;
            if (!samples || *samples > 0) sleep_for(period);
        }
    } catch (...) {
        for (const auto &[sensor, stream] : subscribed) robot.camera_unsubscribe(sensor, stream);
        throw;
    }
    for (const auto &[sensor, stream] : subscribed) robot.camera_unsubscribe(sensor, stream);
}

}  // namespace examples
