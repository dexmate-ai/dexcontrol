# Copyright (C) 2026 Dexmate Inc.
# SPDX-License-Identifier: Apache-2.0
"""Publish the inspected API crate; verify content before skipping retries."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import urllib.error
import urllib.request
from check_crate import check
from release_version import parse_tag


def existing(crate, version):
    request = urllib.request.Request(
        f"https://crates.io/api/v1/crates/{crate}/{version}",
        headers={"User-Agent": "dexcontrol-release (contact@dexmate.ai)"})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.load(response)["version"]
    except urllib.error.HTTPError as error:
        if error.code != 404:
            raise
        return None


def publish(root, tag):
    version = parse_tag(tag).rust
    metadata = json.loads(subprocess.check_output(
        ["cargo", "metadata", "--no-deps", "--format-version=1"], cwd=root))
    crate = "dexcontrol"
    subprocess.run(["cargo", "package", "-p", crate, "--registry", "crates-io", "--locked"], cwd=root, check=True)
    archive = Path(metadata["target_directory"]) / "package" / f"{crate}-{version}.crate"
    check(archive, version, root / "rust")
    published = existing(crate, version)
    if published is not None:
        if published["yanked"] or published["checksum"] != hashlib.sha256(archive.read_bytes()).hexdigest():
            raise ValueError(f"{crate} {version} exists with different content or was yanked; use a new version")
        print(f"{crate} {version}: identical package already published")
    else:
        # Cargo waits until the published version is available in the registry.
        subprocess.run(["cargo", "publish", "-p", crate, "--registry", "crates-io", "--locked"], cwd=root, check=True)


if __name__ == "__main__":
    publish(Path(__file__).resolve().parents[1], sys.argv[1])
