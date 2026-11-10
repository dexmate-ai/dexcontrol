// Copyright (C) 2026 Dexmate Inc.
// SPDX-License-Identifier: Apache-2.0

// Measure full Robot connection time and query round-trip latency.
//
// Create one control connection, then issue 10 version_info queries and print timing statistics. This includes normal Robot startup/cleanup side effects; it is not a read-only DiagnosticClient benchmark.
//
// Also waits for active state (10 s default timeout) before finishing the connection timer.
//
// Uses a Robot control connection: compatible arm modes may be enabled during startup, and shutdown requests a stop. Real hardware is selected unless --simulated is supplied.

#include "example.hpp"

#include <algorithm>
#include <numeric>

namespace {

std::string milliseconds(double seconds) {
    std::ostringstream out;
    out << std::fixed << std::setprecision(1) << seconds * 1000.0 << " ms";
    return out.str();
}

}  // namespace

int main(int argc, char **argv) {
    return examples::run(argc, argv,
                         "Measure full Robot connection time and query round-trip latency. Create one control connection, then issue 10 version_info queries and print timing statistics. This includes normal Robot startup/cleanup side effects; it is not a read-only DiagnosticClient benchmark. Also waits for active state (10 s default timeout) before finishing the connection timer. Uses a Robot control connection: compatible arm modes may be enabled during startup, and shutdown requests a stop. Real hardware is selected unless --simulated is supplied.",
                         [](const examples::Args &args) {
        const int samples = args.integer("samples", 10);
        const double timeout = args.number("timeout", 10.0);
        if (samples < 1) examples::Args::fail("--samples must be at least 1");

        const auto started = std::chrono::steady_clock::now();
        auto robot = args.connect();
        if (!robot.wait_for_active(examples::millis(timeout))) {
            std::ostringstream message;
            message << "robot did not become active within " << std::fixed << std::setprecision(1)
                    << timeout << " seconds";
            throw std::runtime_error(message.str());
        }
        const double connection_time = examples::elapsed(started);

        std::vector<double> round_trips;
        for (int sample = 0; sample < samples; ++sample) {
            try {
                const auto query_started = std::chrono::steady_clock::now();
                // The raw query, deliberately: version_info_json() answers from
                // the connect-time response and would time a lookup rather
                // than a network round trip.
                robot.query("version_info");
                round_trips.push_back(examples::elapsed(query_started));
            } catch (const dexcontrol::Error &error) {
                std::cout << "query " << sample + 1 << " failed: " << error.what() << "\n";
            }
        }

        std::cout << "Connect Robot Latency Benchmark\n";
        std::cout << "  Init + discovery: " << milliseconds(connection_time) << "\n";
        if (round_trips.empty()) {
            std::cout << "  Query RTT: no successful samples\n";
            return;
        }
        const double mean = std::accumulate(round_trips.begin(), round_trips.end(), 0.0) /
                            static_cast<double>(round_trips.size());
        double variance = 0.0;
        for (const double value : round_trips) variance += (value - mean) * (value - mean);
        const double deviation =
            round_trips.size() > 1 ? std::sqrt(variance / static_cast<double>(round_trips.size() - 1))
                                   : 0.0;
        std::cout << "  Query RTT samples: " << round_trips.size() << "\n";
        std::cout << "  Mean: " << milliseconds(mean) << "\n";
        std::cout << "  Min:  " << milliseconds(*std::min_element(round_trips.begin(), round_trips.end()))
                  << "\n";
        std::cout << "  Max:  " << milliseconds(*std::max_element(round_trips.begin(), round_trips.end()))
                  << "\n";
        std::cout << "  Std:  " << milliseconds(deviation) << "\n";
    }, {"samples", "timeout"});
}
