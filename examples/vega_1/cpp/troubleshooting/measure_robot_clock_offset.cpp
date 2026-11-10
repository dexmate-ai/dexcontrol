// Copyright (C) 2026 Dexmate Inc.
// SPDX-License-Identifier: Apache-2.0

// Estimate robot clock offset and network round-trip time.
//
// Use a read-only DiagnosticClient to collect 30 time-query samples and report server-minus-client offset, round-trip time, and replies. Does not synchronize or change either clock; no simulation connection is supported.
//
// Prints diagnostic JSON.

#include "example.hpp"
#include "json.hpp"

int main(int argc, char **argv) {
    return examples::run(argc, argv,
                         "Estimate robot clock offset and network round-trip time. Use a read-only DiagnosticClient to collect 30 time-query samples and report server-minus-client offset, round-trip time, and replies. Does not synchronize or change either clock; no simulation connection is supported. Prints diagnostic JSON.",
                         [](const examples::Args &args) {
        const int samples = args.integer("samples", 30);
        if (samples < 1) examples::Args::fail("--samples must be at least 1");
        auto client = args.diagnostics();
        const auto result = examples::Json::parse(client.query_ntp_json(static_cast<size_t>(samples)));
        std::cout << result.dump() << "\n";
    }, {"samples"});
}
