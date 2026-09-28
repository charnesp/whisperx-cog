#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["pyyaml>=6.0", "requests>=2.31"]
# ///
"""models.lock v2 generator + drift audit (plan E5 T5).

Schema v2 (one source of truth, sha256 per file):
    version: 2
    models:
      - repo: <hf repo>
        revision: <40-hex commit sha>   # pin by commit sha
        required: true
        files:
          - path: <relative path>
            size: <bytes>
            sha256: <64-hex>            # uniform schema, LFS or not
            lfs: <bool>

Data sources:
  - LFS files: GET https://huggingface.co/api/models/<repo>/tree/<rev>?recursive=1
    field lfs.oid = sha256 of the content (verified on the 5 repos 2026-09-16).
  - Non-LFS files (config.json, tokenizer.json, vocabulary.txt of tiny — CT2
    format, DO NOT normalize) only expose a git blob sha1 there: this tool
    downloads them once and hashes them locally (uniform sha256 schema).

The VAD stays OUT of the lock: it is bundled in whisperx (pip pin).
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path
from typing import Any, Callable

import requests
import yaml

HF_API = "https://huggingface.co/api/models"
EXCLUDED = {"README.md", ".gitattributes"}
LOCK_PATH = Path(__file__).resolve().parent.parent / "models.lock"


class LockDrift(Exception):
    """Raised when the local tree does not match the lock."""


# ---------------------------------------------------------------------------
# Tree API → file specs


def fetch_tree(repo: str, revision: str, token: str | None = None) -> list[dict[str, Any]]:
    """Fetch the recursive tree listing for a repo revision."""
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    url = f"{HF_API}/{repo}/tree/{revision}"
    params = {"recursive": "true"}
    resp = requests.get(url, headers=headers, params=params, timeout=60)
    resp.raise_for_status()
    listing = resp.json()
    if isinstance(listing, dict):
        listing = listing.get("files", [])
    out = []
    for entry in listing:
        out.append(
            {
                "path": entry["path"],
                "size": entry.get("size", 0),
                "lfs": entry.get("lfs", False),
                "sha256": (entry.get("lfs") or {}).get("oid") if entry.get("lfs") else None,
            }
        )
    return out


def tree_entry_to_file_spec(entry: dict[str, Any], sha256: str | None = None) -> dict[str, Any]:
    """Convert a tree entry into a lock file spec.

    LFS entries carry sha256 (lfs.oid). Non-LFS entries need a download+hash
    pass (pass it via sha256 once computed).
    """
    return {
        "path": entry["path"],
        "size": entry["size"],
        "sha256": entry.get("sha256") or sha256,
        "lfs": bool(entry.get("lfs")),
    }


def hash_url(url: str, token: str | None = None) -> str:
    """Stream a file from the hub and return its sha256."""
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    with requests.get(url, headers=headers, timeout=300, stream=True) as resp:
        resp.raise_for_status()
        digest = hashlib.sha256()
        for chunk in resp.iter_content(chunk_size=8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def build_lock_from_tree(
    tree: dict[str, Any],
    required: bool = True,
    excluded: Callable[[str], bool] | None = None,
    token: str | None = None,
) -> dict[str, Any]:
    """Build a v2 lock entry structure from a fetched tree.

    Non-LFS files are downloaded once and hashed (uniform sha256 schema).
    """
    excluded = excluded or (lambda p: p in EXCLUDED)
    files = []
    for entry in tree["files"]:
        if excluded(entry["path"]):
            continue
        if entry.get("lfs") and entry.get("sha256"):
            files.append(tree_entry_to_file_spec(entry))
            continue
        # Non-LFS: download once, hash locally.
        url = f"https://huggingface.co/{tree['repo']}/resolve/{tree['rev']}/{entry['path']}"
        sha = hash_url(url, token)
        files.append(tree_entry_to_file_spec(entry, sha256=sha))
    return {
        "repo": tree["repo"],
        "revision": tree["rev"],
        "required": required,
        "files": sorted(files, key=lambda f: f["path"]),
    }


# ---------------------------------------------------------------------------
# Lock I/O


def load_lock_v2(lock_path: Path) -> dict[str, Any]:
    lock = yaml.safe_load(lock_path.read_text())
    if not isinstance(lock, dict) or lock.get("version") != 2:
        raise LockDrift(f"{lock_path}: schema v2 expected (version: 2)")
    for m in lock.get("models", []):
        rev = m.get("revision", "")
        if not (isinstance(rev, str) and len(rev) == 40 and all(c in "0123456789abcdef" for c in rev)):
            raise LockDrift(f"{m.get('repo')}: revision must be 40-hex, got {rev!r}")
    return lock


def check_against_lock(
    lock_path: Path,
    local_trees: dict[str, list[dict[str, Any]]] | None = None,
) -> None:
    """Fail with LockDrift when the local tree drifts from the lock.

    local_trees maps repo -> list of {path, size, sha256, lfs}. In CI the
    local tree comes from the HF API itself (round-trip check: lock content
    == regenerated content). Drift = sha256 or size mismatch, or file
    missing from / extra in the local tree.
    """
    lock = load_lock_v2(lock_path)
    for model in lock["models"]:
        repo = model["repo"]
        if local_trees and repo in local_trees:
            local = {f["path"]: f for f in local_trees[repo]}
        else:
            local = {f["path"]: f for f in fetch_tree(repo, model["revision"])}
        locked = {f["path"]: f for f in model["files"]}
        problems = []
        for path, spec in locked.items():
            if path not in local:
                problems.append(f"missing on remote/local tree: {path}")
                continue
            actual = local[path]
            if spec["sha256"] and actual.get("sha256") and spec["sha256"] != actual["sha256"]:
                problems.append(f"sha256 drift for {path}: lock={spec['sha256'][:12]}… actual={actual['sha256'][:12]}…")
            if spec["size"] != actual.get("size"):
                problems.append(f"size drift for {path}: lock={spec['size']} actual={actual.get('size')}")
        for path in local:
            if path not in locked:
                problems.append(f"file not in lock: {path}")
        if problems:
            raise LockDrift(f"{repo}@{model['revision']}: " + "; ".join(problems))
    print(f"lock_audit OK: {len(lock['models'])} models checked against {LOCK_PATH.name}")


# ---------------------------------------------------------------------------
# Regenerate


def regenerate(token: str | None = None, out: Path = LOCK_PATH) -> None:
    """Regenerate models.lock v2 from the HF API for the 5 pinned models."""
    repos = [
        ("tiny", "Systran/faster-whisper-tiny"),
        ("large-v3", "Systran/faster-whisper-large-v3"),
        ("large-v3-turbo", "mobiuslabsgmbh/faster-whisper-large-v3-turbo"),
        ("qwen3-asr-1.7b", "Qwen/Qwen3-ASR-1.7B"),
        ("qwen3-forced-aligner-0.6b", "Qwen/Qwen3-ForcedAligner-0.6B"),
    ]
    existing: dict[str, dict[str, Any]] = {}
    if out.exists():
        old = yaml.safe_load(out.read_text()) or {}
        for m in old.get("models", []):
            existing[m["repo"]] = m
    models = []
    for _key, repo in repos:
        prev = existing.get(repo, {})
        revision = prev.get("revision")
        if not revision:
            info = requests.get(f"{HF_API}/{repo}", timeout=60).json()
            revision = info["sha"]
        tree = {"repo": repo, "rev": revision, "files": fetch_tree(repo, revision, token)}
        models.append(build_lock_from_tree(tree, required=True, token=token))
    header = (
        "# models.lock v2 — one source of truth, sha256 per file (plan E5 T5).\n"
        "# LFS sha256 from the HF tree API (lfs.oid); non-LFS files downloaded\n"
        "# once and hashed by scripts/lock_audit.py (uniform sha256 schema).\n"
        "# VAD is OUT of the lock: bundled in whisperx (pip pin).\n"
        "# Regenerate: uv run scripts/lock_audit.py --regenerate\n"
    )
    out.write_text(header + yaml.dump({"version": 2, "models": models}, sort_keys=False, allow_unicode=True))
    print(f"written: {out} ({len(models)} models)")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="lock_audit")
    parser.add_argument("--check", action="store_true", help="fail on drift between lock and HF API")
    parser.add_argument("--regenerate", action="store_true", help="rewrite models.lock v2 from the HF API")
    parser.add_argument("--token", default=os.environ.get("HF_TOKEN"), help="HF token (read-only)")
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])
    try:
        if args.regenerate:
            regenerate(args.token)
            return 0
        if args.check:
            check_against_lock(LOCK_PATH)
            return 0
        parser.print_help()
        return 2
    except LockDrift as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
