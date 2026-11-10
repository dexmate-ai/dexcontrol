# Copyright (C) 2026 Dexmate Inc.
# SPDX-License-Identifier: Apache-2.0
"""Reject unreviewed source or dependencies in the actual public Cargo archive."""
from pathlib import Path, PurePosixPath
import json
import sys
import tarfile
import tomllib
from publication_policy import findings


def check(path, version, source_root=None):
    common = {"Cargo.toml", "Cargo.toml.orig", "Cargo.lock", ".cargo_vcs_info.json",
              "LICENSE", "NOTICE", "LICENSE-HEADER", "src/lib.rs"}
    with tarfile.open(path, "r:gz") as archive:
        members = archive.getmembers()
        if len(members) > 32 or sum(m.size for m in members) > 16 * 1024 * 1024:
            raise ValueError("API crate exceeds inspection limit")
        prefix = Path(path).name.removesuffix(".crate")
        name = prefix.removesuffix("-" + version)
        if name != "dexcontrol" or prefix != f"{name}-{version}":
            raise ValueError("Unexpected API crate name/version")
        allowed = common | {"build.rs", "header.sha256", "src/ffi/mod.rs",
                            "src/ffi/bindings.rs", "src/ffi/constants.rs",
                            "examples/read_joint_positions.rs",
                            "src/component.rs", "src/diagnostics.rs", "src/motion.rs", "src/options.rs", "src/robot.rs", "src/sensors.rs"}
        found = {}
        for member in members:
            parts = PurePosixPath(member.name).parts
            relative = "/".join(parts[1:])
            if member.name != "/".join(parts) or not member.isfile() or not parts or parts[0] != prefix or relative not in allowed or relative in found:
                raise ValueError(f"Unapproved crate entry: {member.name}")
            data = archive.extractfile(member).read()
            if findings(data):
                raise ValueError("Sensitive content in API crate: " + relative)
            found[relative] = data
        if allowed - {".cargo_vcs_info.json", "Cargo.lock"} - found.keys():
            raise ValueError("Missing API crate files")
        for manifest in ("Cargo.toml", "Cargo.toml.orig"):
            value = tomllib.loads(found[manifest].decode())
            package = value["package"]
            if package["name"] != name or package["version"] != version or package.get("publish") != ["crates-io"]:
                raise ValueError("Unexpected public crate metadata")
            if package.get("links") != "dexcontrol_capi" or package.get("build", "build.rs") != "build.rs":
                raise ValueError("Unreviewed native build entry point")
            library = value.get("lib", {})
            if (set(library) - {"name", "path"}
                    or library.get("name", "dexcontrol") != "dexcontrol"
                    or library.get("path", "src/lib.rs") != "src/lib.rs"):
                raise ValueError("Unreviewed library entry point")
            if "target" in value or "patch" in value or "replace" in value:
                raise ValueError("Unreviewed dependency override")
            for section in ("dependencies", "build-dependencies", "dev-dependencies"):
                expected = {"pkg-config"} if section == "build-dependencies" else set()
                dependencies = value.get(section, {})
                if set(dependencies) != expected:
                    raise ValueError("Unreviewed public crate dependency")
                for spec in dependencies.values():
                    if isinstance(spec, dict) and set(spec) != {"version"}:
                        raise ValueError("Nonpublic dependency location")
                    if (spec.get("version") if isinstance(spec, dict) else spec) != "0.3":
                        raise ValueError("Unreviewed build dependency version")
        if source_root is not None:
            source = Path(source_root) / name
            for filename, data in found.items():
                if filename.endswith(".rs") or filename in {"header.sha256", "Cargo.toml.orig"}:
                    original = source / ("Cargo.toml" if filename == "Cargo.toml.orig" else filename)
                    if (any(parent.is_symlink() for parent in (original, *original.parents))
                            or original.read_bytes() != data):
                        raise ValueError("Packaged source differs from reviewed API: " + filename)
        if "Cargo.lock" in found:
            for entry in tomllib.loads(found["Cargo.lock"].decode()).get("package", []):
                source = entry.get("source")
                if source is not None and source != "registry+https://github.com/rust-lang/crates.io-index":
                    raise ValueError("Nonpublic lockfile source")
                if entry["name"].startswith("dex") and entry["name"] != "dexcontrol":
                    raise ValueError("Internal dependency in API lockfile")
        if ".cargo_vcs_info.json" in found:
            info = json.loads(found[".cargo_vcs_info.json"])
            if info.get("path_in_vcs") != "rust/" + name:
                raise ValueError("Crate was not packaged from the public SDK tree")


if __name__ == "__main__":
    check(Path(sys.argv[1]), sys.argv[2], Path(__file__).resolve().parents[1] / "rust")
