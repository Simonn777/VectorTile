#!/usr/bin/env python3
"""Build VectorTile's optional C++17 shared library with no Make dependency."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shlex
import shutil
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--debug", action="store_true", help="Disable optimization and emit debug symbols")
    parser.add_argument("--sanitize", choices=("address", "undefined", "address,undefined"))
    args = parser.parse_args()
    if sys.platform not in ("darwin", "linux"):
        parser.error("This build script supports macOS and Linux.")
    compiler = shlex.split(os.environ["CXX"]) if os.environ.get("CXX") else []
    if not compiler:
        candidate = shutil.which("clang++") or shutil.which("c++")
        if not candidate:
            parser.error("No C++ compiler found. Install Clang or GCC, or set CXX.")
        compiler = [candidate]
    root = Path(__file__).resolve().parents[1]
    build = root / "native" / "build"
    build.mkdir(parents=True, exist_ok=True)
    output = Path("native/build") / ("libvectortile.dylib" if sys.platform == "darwin" else "libvectortile.so")
    flags = [
        "-std=c++17", "-O0" if args.debug else "-O3", "-fPIC", "-pthread",
        "-Wall", "-Wextra", "-Wpedantic", "-ffp-contract=off", "-fvisibility=hidden",
        "-dynamiclib" if sys.platform == "darwin" else "-shared",
        "-I", "native/include",
        "native/src/vectortile.cpp", "-o", str(output),
    ]
    if args.debug:
        flags.append("-g")
    if args.sanitize:
        flags.extend(["-g", "-fno-omit-frame-pointer", f"-fsanitize={args.sanitize}"])
    command = compiler + flags
    print(shlex.join(command), flush=True)
    subprocess.run(command, cwd=root, check=True)

    def public_text(value: str) -> str:
        # CXX may name a compiler or include toolchain flags under a home
        # directory. Keep useful flags without publishing local absolute paths.
        if value.startswith(("-I/", "-L/")):
            return value[:2] + public_text(value[2:])

        def relative_or_redacted(match):
            path = Path(match.group(0))
            try:
                return path.relative_to(root).as_posix()
            except ValueError:
                return "<external>/" + path.name
        return re.sub(r"(?<![A-Za-z0-9_.-])/[^\s\"'=,;:]+", relative_or_redacted, value)

    try:
        version_result = subprocess.run(
            compiler + ["--version"], cwd=root, check=False, capture_output=True,
            text=True, timeout=10,
        )
        version_lines = (version_result.stdout or version_result.stderr).splitlines()
        compiler_version = public_text(version_lines[0]) if version_lines else "unavailable"
    except (OSError, subprocess.TimeoutExpired):
        compiler_version = "unavailable"
    public_compiler = [Path(compiler[0]).name] + [public_text(arg) for arg in compiler[1:]]
    source_paths = [
        "native/include/vectortile.h", "native/src/vectortile.cpp", "scripts/build_native.py",
    ]
    provenance = {
        "schema_version": 1,
        "compiler": {"executable": public_compiler[0], "version": compiler_version},
        "platform": {"system": platform.system(), "release": platform.release(), "machine": platform.machine()},
        "flags": [public_text(arg) for arg in compiler[1:] + flags],
        "command": public_compiler + flags,
        "source_sha256": {
            path: hashlib.sha256((root / path).read_bytes()).hexdigest() for path in source_paths
        },
        "output": output.as_posix(),
        "library_sha256": hashlib.sha256((root / output).read_bytes()).hexdigest(),
    }
    (build / "build-info.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(f"Built {output}")


if __name__ == "__main__":
    main()
