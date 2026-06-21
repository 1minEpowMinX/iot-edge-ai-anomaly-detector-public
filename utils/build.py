#!/usr/bin/env python3
"""Cross-platform PyInstaller build — single source of truth.

PyInstaller does NOT cross-compile: run this on the OS you are building for.
The script auto-detects the platform and produces a onedir bundle in ``dist/``.

    python build.py                 # auto: win64 / linux64 / macos
    python build.py --name foo      # override the bundle name
    python build.py --dry-run       # print the command without running it

Prereqs (on the build host):
    python -m venv .venv-build
    # Windows: .venv-build\\Scripts\\activate   |  *nix: source .venv-build/bin/activate
    pip install -r requirements.txt pyinstaller
"""

from __future__ import annotations

import argparse
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# Shared across every platform. Kept in one place so the three targets cannot
# drift apart (the old build.ps1 / build.sh / .spec each held their own copy).
COMMON_ARGS = [
    "--noconfirm",
    "--clean",
    "--onedir",
    "--collect-all", "torch",
    "--collect-submodules", "sklearn",
    "--exclude-module", "tkinter",
    "--exclude-module", "tensorflow",
    "--exclude-module", "PyQt5",
    "--exclude-module", "PyQt6",
    "--exclude-module", "PySide2",
    "--exclude-module", "PySide6",
    "--exclude-module", "IPython",
    "--exclude-module", "pytest",
    "--exclude-module", "notebook",
]

# Per-platform: bundle name + extras. ``version_file`` is Windows-only (PE
# resources); it does not apply to ELF/Mach-O binaries.
TARGETS = {
    "Windows": {
        "name": "iot-edge-ai-anomaly-detector-win64",
        "extra": ["--version-file", str(ROOT / "resources" / "version_info.txt")],
    },
    "Linux": {
        "name": "iot-edge-ai-anomaly-detector-linux64",
        "extra": [],
    },
    "Darwin": {
        "name": "iot-edge-ai-anomaly-detector-macos",
        "extra": [],
    },
}


def build_command(name: str, extra: list[str]) -> list[str]:
    """Assemble the full PyInstaller argument vector.

    Args:
        name: Output bundle name.
        extra: Platform-specific extra flags.

    Returns:
        The command as a list of arguments (PyInstaller run as a module).
    """
    return [
        sys.executable, "-m", "PyInstaller",
        *COMMON_ARGS,
        "--name", name,
        *extra,
        str(ROOT / "main.py"),
    ]


def main(argv: list[str] | None = None) -> int:
    """Parse args, resolve the platform target and run (or print) the build.

    Args:
        argv: Argument list; defaults to ``sys.argv`` when ``None``.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(description="Cross-platform PyInstaller build.")
    parser.add_argument("--name", help="override the output bundle name")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the PyInstaller command without running it",
    )
    args = parser.parse_args(argv)

    system = platform.system()
    target = TARGETS.get(system)
    if target is None:
        parser.error(
            f"непідтримувана платформа: {system!r} "
            f"(очікувалось одне з {', '.join(TARGETS)})"
        )

    name = args.name or target["name"]
    cmd = build_command(name, target["extra"])

    print(f"[build] платформа={system}  bundle={name}")
    print("[build] " + " ".join(cmd))
    if args.dry_run:
        return 0

    subprocess.run(cmd, cwd=ROOT, check=True)

    dist = ROOT / "dist" / name
    exe = dist / (f"{name}.exe" if system == "Windows" else name)
    print(f"\n[build] готово -> {dist}{'/' if system != 'Windows' else chr(92)}")
    print("[build] перевірка:")
    subprocess.run([str(exe), "--version"], check=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
