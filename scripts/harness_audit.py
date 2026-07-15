#!/usr/bin/env python3
"""Harness Engineering audit — verify repo scaffolding for agent-first workflows."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

REQUIRED_FILES = [
    "AGENTS.md",
    "PLANS.md",
    "Makefile.harness",
    "docs/ARCHITECTURE.md",
    "docs/OBSERVABILITY.md",
    "docs/BRIDGE.md",
    "docs/DATA_CONTRACTS.md",
    "bridge/bridge.py",
    "scripts/check-bridge-sync.py",
    "scripts/sync-bridge-to-k8s.py",
]

AGENTS_LINKS = [
    "docs/ARCHITECTURE.md",
    "docs/OBSERVABILITY.md",
    "docs/BRIDGE.md",
    "PLANS.md",
    "README.md",
]

MAKEFILE_TARGETS = ("smoke", "check", "ci", "audit")


def ok(msg: str) -> None:
    print(f"  OK  {msg}")


def fail(msg: str) -> None:
    print(f"  FAIL {msg}", file=sys.stderr)


def check_files() -> list[str]:
    errors: list[str] = []
    for rel in REQUIRED_FILES:
        path = ROOT / rel
        if path.is_file():
            ok(rel)
        else:
            fail(f"missing {rel}")
            errors.append(rel)
    return errors


def check_agents_map() -> list[str]:
    errors: list[str] = []
    agents = (ROOT / "AGENTS.md").read_text()
    for link in AGENTS_LINKS:
        if link in agents:
            ok(f"AGENTS.md links to {link}")
        else:
            fail(f"AGENTS.md missing link to {link}")
            errors.append(f"AGENTS.md:{link}")
    if "Makefile.harness" in agents or "make -f Makefile.harness" in agents:
        ok("AGENTS.md documents harness commands")
    else:
        fail("AGENTS.md missing harness make commands")
        errors.append("AGENTS.md:harness")
    line_count = len(agents.splitlines())
    if line_count <= 120:
        ok(f"AGENTS.md length ({line_count} lines, map-style)")
    else:
        fail(f"AGENTS.md too long ({line_count} lines; target ≤120 for map-style)")
        errors.append("AGENTS.md:length")
    return errors


def check_makefile() -> list[str]:
    errors: list[str] = []
    mf = (ROOT / "Makefile.harness").read_text()
    for target in MAKEFILE_TARGETS:
        if re.search(rf"^{target}:", mf, re.MULTILINE):
            ok(f"Makefile.harness target {target}")
        else:
            fail(f"Makefile.harness missing target {target}")
            errors.append(f"Makefile.harness:{target}")
    return errors


def check_bridge_sync() -> list[str]:
    errors: list[str] = []
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "check-bridge-sync.py")],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        ok("k8s manifest uses published bridge GHCR image")
    else:
        fail("bridge sync check failed")
        errors.append("bridge-sync")
        if result.stderr:
            print(result.stderr.strip(), file=sys.stderr)
    return errors


def check_yaml() -> list[str]:
    errors: list[str] = []
    try:
        import yaml  # type: ignore
    except ImportError:
        ok("yaml parse skipped (PyYAML not installed)")
        return errors
    for rel in ("k8s/whisperx-stack.yaml", "docker-compose.yml"):
        path = ROOT / rel
        try:
            list(yaml.safe_load_all(path.read_text()))
            ok(f"{rel} valid YAML")
        except Exception as exc:
            fail(f"{rel} invalid YAML: {exc}")
            errors.append(rel)
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Harness Engineering audit")
    parser.add_argument(
        "--smoke-only",
        action="store_true",
        help="Skip bridge sync check (smoke target runs it separately)",
    )
    args = parser.parse_args()

    print("Harness audit — whisperx-cog\n")
    all_errors: list[str] = []

    print("[files]")
    all_errors.extend(check_files())

    print("\n[agents map]")
    all_errors.extend(check_agents_map())

    print("\n[makefile]")
    all_errors.extend(check_makefile())

    if not args.smoke_only:
        print("\n[bridge sync]")
        all_errors.extend(check_bridge_sync())

    print("\n[yaml]")
    all_errors.extend(check_yaml())

    print()
    if all_errors:
        print(f"AUDIT FAIL ({len(all_errors)} issue(s))", file=sys.stderr)
        return 1
    print("AUDIT OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
