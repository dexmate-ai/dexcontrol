#!/usr/bin/env python3
# Copyright (C) 2026 Dexmate Inc.
# SPDX-License-Identifier: Apache-2.0
"""Public pre-upload gate: allow only compiled wheels and reviewed Python API files."""
from pathlib import Path
import base64
import csv
import hashlib
import io
import json
import re
import stat
import sys
import zipfile
from publication_policy import findings
from release_version import parse_tag, PYTHON_VERSION_PATTERN

PYTHON_FILES = (
    "__init__.py", "robot.py", "diagnostics.py", "viewer.py", "types.py", "exceptions.py", "_native.pyi", "py.typed",
    "py.typed.license", "_logsink.py", "_prompt.py", "_watchdog_process.py",
    "sensors/__init__.py", "sensors/manager.py",
    "compat/__init__.py", "compat/v05.py",
)

def validate_public_sbom(data):
    value = json.loads(data)
    if set(value) != {"bomFormat", "specVersion", "version", "components"} or value["bomFormat"] != "CycloneDX" or value["specVersion"] != "1.5" or value["version"] != 1:
        raise ValueError("Unexpected public SBOM schema")
    names = {component.get("name") for component in value["components"]}
    required_native = {native for wrapper, native in (("ffmpeg-sys-next", "ffmpeg"), ("turbojpeg-sys", "libjpeg-turbo"), ("zstd-sys", "zstd")) if wrapper in names}
    native_names = {component.get("name") for component in value["components"] if component.get("purl", "").startswith("pkg:generic/")}
    if names & {"ffmpeg", "ffmpeg-next", "ffmpeg-sys-next", "webrtc"}:
        raise ValueError("Customer packages must not include FFmpeg or WebRTC dependencies")
    required_native |= {"libjpeg-turbo", "zstd"}
    if not required_native <= native_names:
        raise ValueError("Missing bundled native libraries in SBOM")
    for component in value["components"]:
        if set(component) - {"type", "name", "version", "purl", "licenses"} or component.get("type") != "library":
            raise ValueError("Unapproved SBOM metadata")
        name, version = component.get("name", ""), component.get("version", "")
        if name.startswith("dex") or not re.fullmatch(r"[A-Za-z0-9_-]+", name) or not re.fullmatch(r"[A-Za-z0-9.+_-]+", version):
            raise ValueError("Private or invalid SBOM component")
        ecosystem = "generic" if name in {"ffmpeg", "libjpeg-turbo", "zstd"} and component.get("purl", "").startswith("pkg:generic/") else "cargo"
        if component.get("purl") != f"pkg:{ecosystem}/{name}@{version}":
            raise ValueError("Private dependency location in SBOM")
        for entry in component.get("licenses", []):
            if set(entry) == {"expression"} and isinstance(entry["expression"], str):
                continue
            if set(entry) == {"license"} and set(entry["license"]) == {"id"} and isinstance(entry["license"]["id"], str):
                continue
            raise ValueError("Unapproved SBOM license metadata")


def check(path, version, source_root):
    if not re.fullmatch(r"dexcontrol-" + re.escape(version) + r"-cp[0-9]+-abi3-[A-Za-z0-9_.]+\.whl", path.name):
        raise ValueError("Only version-matched binary wheels may be published; source distributions are forbidden")
    if path.name.endswith(".whl"):
        with zipfile.ZipFile(path) as archive:
            entries = archive.infolist()
            names = [entry.filename for entry in entries]
            if len(names) != len(set(names)):
                raise ValueError("Duplicate wheel entries")
            if sum(entry.file_size for entry in entries) > 2 * 1024**3:
                raise ValueError("Wheel exceeds inspection size limit")
            metadata_dirs = {name.split("/")[0] for name in names if ".dist-info/" in name}
            if len(metadata_dirs) != 1:
                raise ValueError("Expected one wheel metadata directory")
            metadata = metadata_dirs.pop()
            if not re.fullmatch(rf"dexcontrol-{PYTHON_VERSION_PATTERN}\.dist-info", metadata):
                raise ValueError("Unexpected wheel metadata directory")
            if metadata != f"dexcontrol-{version}.dist-info":
                raise ValueError("Wheel metadata version does not match release")
            allowed = {"dexcontrol/" + name for name in PYTHON_FILES}
            allowed |= {"dexcontrol/_native.abi3.so"}
            required_metadata = {"sboms/third-party.cyclonedx.json", "licenses/THIRD-PARTY-NOTICES.txt", "METADATA", "WHEEL", "RECORD", "licenses/LICENSE", "licenses/NOTICE", "licenses/LICENSE-HEADER"}
            required = allowed | {f"{metadata}/{name}" for name in required_metadata}
            allowed |= {f"{metadata}/{name}" for name in required_metadata | {"sboms/third-party.cyclonedx.json", "licenses/THIRD-PARTY-NOTICES.txt"}}
            found = set()
            for entry in entries:
                name = entry.filename
                if entry.is_dir():
                    if not any(item.startswith(name) for item in allowed):
                        raise ValueError("Unapproved wheel directory")
                    continue
                if name not in allowed:
                    raise ValueError(f"Unapproved wheel file: {name}")
                if stat.S_ISLNK(entry.external_attr >> 16):
                    raise ValueError("Wheel symlinks are not allowed")
                found.add(name)
                data = archive.read(entry)
                if name.startswith("dexcontrol/") and name != "dexcontrol/_native.abi3.so":
                    source = source_root / "python" / name
                    if source.is_symlink() or not source.is_file() or source.read_bytes() != data:
                        raise ValueError(f"Wheel source differs from reviewed public API: {name}")
                if name == "dexcontrol/_native.abi3.so" and data[:4] not in (
                    b"\x7fELF", b"\xcf\xfa\xed\xfe", b"\xfe\xed\xfa\xcf", b"\xca\xfe\xba\xbe", b"\xbe\xba\xfe\xca",
                ):
                    raise ValueError("Expected compiled native runtime")
                if name == metadata + "/METADATA":
                    import email
                    message = email.message_from_bytes(data)
                    if message.get_all("Name") != ["dexcontrol"] or message.get_all("Version") != [version]:
                        raise ValueError("Incorrect wheel package identity")
                    for dependency in message.get_all("Requires-Dist", []):
                        normalized = dependency.lower().replace("_", "-")
                        if "@" in dependency or re.match(r"(?:dexcomm|dexcontrol-(?:core|protocol|transport|dexcomm))(?=[\s(;>=<!\[]|$)", normalized):
                            raise ValueError("Private or direct-URL wheel dependency")
                if name.endswith("/sboms/third-party.cyclonedx.json"):
                    validate_public_sbom(data)
                issues = findings(data, binary=name.endswith(".so") or name.startswith(metadata + "/"))
                if issues:
                    raise ValueError(f"Sensitive content in {name}: {', '.join(issues)}")
            if required - found:
                raise ValueError(f"Missing wheel files: {sorted(required - found)}")
            record_name = f"{metadata}/RECORD"
            records = list(csv.reader(io.StringIO(archive.read(record_name).decode())))
            if any(len(row) != 3 for row in records) or len({row[0] for row in records}) != len(records):
                raise ValueError("Malformed or duplicate wheel RECORD entries")
            records = {row[0]: row[1:] for row in records}
            if set(records) != found or records[record_name] != ["", ""]:
                raise ValueError("Wheel RECORD does not match contents")
            for name in found - {record_name}:
                data = archive.read(name)
                digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
                if records[name] != ["sha256=" + digest, str(len(data))]:
                    raise ValueError(f"Wheel RECORD mismatch: {name}")
        return


def check_directory(directory, tag, source_root):
    version = parse_tag(tag)
    paths = list(directory.iterdir())
    if not paths:
        raise ValueError("No wheels to publish")
    for path in paths:
        if path.is_symlink() or not path.is_file():
            raise ValueError("Only regular wheel files may be published")
        check(path, version.python, source_root)


if __name__ == "__main__":
    check_directory(Path(sys.argv[1]), sys.argv[2], Path(__file__).resolve().parents[1])
