#!/usr/bin/env python3
# Copyright (C) 2026 Dexmate Inc.
# SPDX-License-Identifier: Apache-2.0
"""Install a downloaded SDK safely. No administrator privileges required."""
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shlex
import shutil
import sys
import tarfile
import tempfile


def install(archive, prefix):
    prefix = Path(prefix).expanduser().absolute()
    if prefix.exists():
        raise ValueError(f"Destination already exists: {prefix}; choose a new version directory")
    prefix.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=prefix.parent) as temporary:
        staging = Path(temporary) / "sdk"
        staging.mkdir()
        with tarfile.open(archive, "r:gz") as tar:
            members = tar.getmembers()
            if len(members) > 10000 or sum(m.size for m in members) > 2 * 1024**3:
                raise ValueError("Archive exceeds SDK extraction limits")
            seen = set()
            links = []
            for member in members:
                path = PurePosixPath(member.name)
                if path.is_absolute() or ".." in path.parts or not path.parts or path.parts[0] != "customer-sdk":
                    raise ValueError(f"Unsafe archive path: {member.name}")
                relative = Path(*path.parts[1:])
                if relative in seen:
                    raise ValueError(f"Duplicate archive entry: {member.name}")
                seen.add(relative)
                target = staging / relative
                if member.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                elif member.isfile():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with tar.extractfile(member) as source, target.open("xb") as output:
                        shutil.copyfileobj(source, output)
                    target.chmod(0o755 if member.mode & 0o111 else 0o644)
                elif member.issym():
                    # Only the runtime's unversioned sibling link is needed.
                    if relative.parent != Path("lib") or relative.name not in {"libdexcontrol_capi.so", "libdexcontrol_capi.dylib"}:
                        raise ValueError("Unexpected SDK symlink")
                    if "/" in member.linkname or member.linkname in {".", "..", ""}:
                        raise ValueError("Unsafe SDK symlink")
                    links.append((target, member.linkname))
                else:
                    raise ValueError("SDK archives may not contain devices or hard links")
            for target, link in links:
                if not (target.parent / link).is_file():
                    raise ValueError("Missing SDK symlink target")
                target.symlink_to(link)
        manifest = json.loads((staging / "MANIFEST.json").read_text())
        actual = {str(p.relative_to(staging)) for p in staging.rglob("*") if p.is_file() or p.is_symlink()}
        if actual != set(manifest) | {"MANIFEST.json"}:
            raise ValueError("SDK contents do not match the manifest")
        for name, metadata in manifest.items():
            path = staging / name
            if hashlib.sha256(path.read_bytes()).hexdigest() != metadata["sha256"]:
                raise ValueError(f"SDK checksum mismatch: {name}")
            if path.is_symlink() != ("symlink" in metadata):
                raise ValueError(f"SDK link type mismatch: {name}")
            if path.is_symlink() and Path(os.readlink(path)).as_posix() != metadata["symlink"]:
                raise ValueError(f"SDK link target mismatch: {name}")
        root = shlex.quote(str(prefix))
        activation = f"export DEXCONTROL_SDK_DIR={root}\n"
        for name, suffix in [("PATH", "bin"), ("CMAKE_PREFIX_PATH", ""), ("PKG_CONFIG_PATH", "lib/pkgconfig"), ("LD_LIBRARY_PATH", "lib"), ("DYLD_LIBRARY_PATH", "lib")]:
            value = shlex.quote(str(prefix / suffix))
            activation += f'export {name}={value}${{{name}:+:${name}}}\n'
        (staging / "activate").write_text(activation)
        staging.rename(prefix)
    return prefix


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("Usage: python3 install_sdk.py DOWNLOADED_SDK.tar.gz INSTALL_DIRECTORY")
    try:
        prefix = install(sys.argv[1], sys.argv[2])
    except (ValueError, OSError, tarfile.TarError, KeyError) as error:
        sys.exit(f"SDK installation failed: {error}")
    print(f"Installed SDK: {prefix}\nActivate in your shell:\n  source {shlex.quote(str(prefix / 'activate'))}")
