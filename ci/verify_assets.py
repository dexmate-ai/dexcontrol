# Copyright (C) 2026 Dexmate Inc.
# SPDX-License-Identifier: Apache-2.0
"""Check a release directory before upload or installation (standard library only)."""
import hashlib
from pathlib import Path
import re
import sys
from release_version import parse_tag


def verify(directory, tag, *, subset=False, names_only=False):
    version = re.escape(parse_tag(tag).python)
    platform = r"(?:linux|macos)-(?:x86_64|aarch64|arm64)"
    patterns = [rf"dexcontrol-{version}-cp[0-9]+-abi3-(?:manylinux[^/]+|linux_[^/]+|macosx_[^/]+)\.whl",
                rf"dexcontrol-sdk-{re.escape(tag)}-{platform}\.tar\.gz",
                rf"dexcontrol-watchdog-{platform}"]
    def allowed(name):
        return any(re.fullmatch(p, name) for p in patterns)
    files = {}
    for path in Path(directory).iterdir():
        if path.is_symlink() or not path.is_file():
            raise ValueError("Release entries must be regular files")
        if path.name != "SHA256SUMS":
            if not allowed(path.name):
                raise ValueError(f"Unapproved release asset: {path.name}")
            files[path.name] = path
    if not files:
        raise ValueError("No release assets")
    if names_only:
        return
    checks = {}
    for line in (Path(directory) / "SHA256SUMS").read_text().splitlines():
        match = re.fullmatch(r"([0-9a-f]{64}) [ *](?:\./)?([^/]+)", line)
        if not match or not allowed(match[2]) or match[2] in checks:
            raise ValueError("Invalid or duplicate checksum entry")
        checks[match[2]] = match[1]
    if not set(files) <= checks.keys() or (not subset and set(files) != set(checks)):
        raise ValueError("Checksums do not match release assets")
    for name, path in files.items():
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != checks[name]:
            raise ValueError(f"Checksum mismatch: {name}")


if __name__ == "__main__":
    verify(Path(sys.argv[1]), sys.argv[2], subset="--subset" in sys.argv[3:],
           names_only="--names-only" in sys.argv[3:])
