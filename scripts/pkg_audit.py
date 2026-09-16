#!/usr/bin/env python3
"""Dependency vulnerability audit gate (pip-audit wrapper).

requirements.txt cannot be audited as-is by pip-audit:
- `whisperx @ git+...` is a git URL (pip-audit cannot resolve git deps);
- `torch==2.8.0` has no wheel on PyPI for the interpreter pip-audit
  provisions (cp314) — pip-based resolution fails;
- `qwen-asr` must be installed WITHOUT its transitive deps (gradio/flask/vllm
  are deliberately excluded), so a full pip resolution would audit packages
  the image intentionally never installs;
- a few lines are intentionally unpinned (ffmpeg-python, soundfile, librosa).

This wrapper therefore audits the auditable subset:
1. pinned PyPI deps extracted from requirements.txt (torch, git, qwen-asr
   excluded; qwen-asr is audited separately below since it is a real PyPI pin);
2. unpinned deps at their latest resolved version (informational: flags new
   CVEs but not version drift).

New known CVEs fail the gate (exit 1). Known-accepted IDs can be listed via
the PKG_AUDIT_IGNORE env var (space-separated), e.g. a CVE whose fixed
version is not yet compatible with the qwen_asr 0.0.6 wheel.

Known baseline (2026-09): transformers==4.57.6 carries 8 PYSEC entries whose
fix versions are transformers 5.x — incompatible with the qwen_asr 0.0.6
wheel's pinned API. Accept them via PKG_AUDIT_IGNORE until qwen_asr supports
transformers >= 5; the Makefile wires this default through PKG_AUDIT_IGNORE
so a NEW CVE id still fails the gate.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REQUIREMENTS = ROOT / "requirements.txt"
PKG_AUDIT_BIN = os.environ.get("PKG_AUDIT_BIN", "pip-audit")
IGNORE_ENV = "PKG_AUDIT_IGNORE"

# Lines skipped entirely: git URLs, editable installs, comments, options.
_SKIP_RE = re.compile(r"^\s*(#|-|$)")
_GIT_RE = re.compile(r"^\S+\s*@\s*git\+")
_NAME_RE = re.compile(r"^([A-Za-z0-9_.-]+)(?:\[[-,\w]+\])?(?:([<>=!~]+)(\S+))?")


def parse_requirements(text: str) -> tuple[list[tuple[str, str | None]], list[str]]:
    """Split requirements into (pinned name/version or None) pairs + git names."""
    pinned: list[tuple[str, str | None]] = []
    git_names: list[str] = []
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or _SKIP_RE.match(line):
            continue
        if _GIT_RE.match(line):
            git_names.append(line)
            continue
        m = _NAME_RE.match(line)
        if not m:
            continue
        name, op, version = m.group(1), m.group(2), m.group(3)
        if op and version:
            pinned.append((name, version))
        else:
            pinned.append((name, None))
    return pinned, git_names


def _pypi_latest_versions(names) -> dict[str, str]:
    """Resolve the latest published version of each package via PyPI JSON API."""

    latest: dict[str, str] = {}
    for name in names:
        try:
            with urllib.request.urlopen(f"https://pypi.org/pypi/{name}/json", timeout=30) as resp:  # noqa: S310
                data = json.load(resp)
            version = data.get("info", {}).get("version")
            if version:
                latest[name] = version
        except (OSError, ValueError) as exc:
            print(f"WARN pkg-audit: could not resolve latest {name}: {exc}", file=sys.stderr)
    return latest


def run_pip_audit(req_file: Path, ignore: set[str]) -> tuple[list[dict], list[str]]:
    """Run pip-audit against a requirements file; return (vulns, errors)."""
    cmd = [PKG_AUDIT_BIN, "--disable-pip", "--no-deps", "-r", str(req_file), "--desc", "off", "--format", "json"]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode not in (0, 1):
        return [], [f"pip-audit exited {proc.returncode}: {proc.stderr.strip()[-500:]}"]
    if not proc.stdout.strip():
        return [], [
            "pip-audit produced no JSON (unpinned requirement in audited set?): "
            + proc.stderr.strip()[-300:]
        ]
    try:
        data = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return [], [f"pip-audit returned non-JSON output: {proc.stdout[:300]}"]
    vulns = []
    for pkg in data.get("dependencies", []):
        if not isinstance(pkg, dict):
            continue
        for v in pkg.get("vulns") or []:
            if v.get("id") in ignore:
                continue
            fix_versions = v.get("fix_versions") or []
            vulns.append(
                {
                    "name": pkg.get("name"),
                    "version": pkg.get("version"),
                    "id": v.get("id"),
                    "fix": ", ".join(fix_versions),
                }
            )
    return vulns, []


def main() -> int:
    if shutil_which(PKG_AUDIT_BIN) is None:
        print(f"SKIP pkg-audit: {PKG_AUDIT_BIN} not on PATH (install: uv tool install pip-audit)")
        return 0
    ignore = set(os.environ.get(IGNORE_ENV, "").split())
    pinned, _git = parse_requirements(REQUIREMENTS.read_text())
    # torch==2.8.0 is pinned for the Cog CUDA wheel index (not on PyPI for
    # every interpreter); skip it here, it is audited in the Cog image build.
    # Unpinned names are audited at their latest PyPI version (informational).
    latest = _pypi_latest_versions(name for name, version in pinned if not version)
    audit_lines = [
        f"{name}=={version if version else latest.get(name, '')}" if version or name in latest else f"{name}"
        for name, version in pinned
        if name != "torch"
    ]
    unresolvable = [n for n, v in pinned if n != "torch" and not v and n not in latest]
    if unresolvable:
        print(f"WARN pkg-audit: could not pin latest for: {', '.join(unresolvable)}", file=sys.stderr)
    scratch = Path(__file__).resolve().parent.parent / ".pkg-audit-reqs.txt"
    scratch.write_text("\n".join(audit_lines) + "\n")
    vulns, errors = run_pip_audit(scratch, ignore)
    scratch.unlink(missing_ok=True)
    if errors:
        for e in errors:
            print(f"FAIL pkg-audit: {e}", file=sys.stderr)
        return 1
    if vulns:
        print(f"FAIL pkg-audit: {len(vulns)} known vulnerabilities (ignore via {IGNORE_ENV}=\"<ID>\" if triaged):", file=sys.stderr)
        for v in vulns:
            fix = f" fix={v['fix']}" if v["fix"] else ""
            print(f"  {v['name']} {v['version']} {v['id']}{fix}", file=sys.stderr)
        return 1
    print(f"OK pkg-audit: {len(audit_lines)} dependencies audited, 0 known vulnerabilities")
    return 0


def shutil_which(bin_name: str) -> str | None:
    import shutil

    return shutil.which(bin_name)


if __name__ == "__main__":
    sys.exit(main())
