#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["pyyaml>=6.0", "huggingface_hub>=0.24"]
# ///
"""Atomic model provisioner for whisperx-cog (plan E5, §2.2).

One source of truth: models.lock (v2, sha256 per file). This tool
provisions/imports model revisions onto the /models bind-mount host tree:

    <models>/<model>/<sha40>/... + .complete   (written LAST, after fsync)
    <models>/.locks/<model>.lock               (flock, OUTSIDE versioned dirs)
    <models>/.state/active.json + history.jsonl

Subcommands:
  provision --model <key> [--revision <sha>]  download from HF Hub
  import    --model <key> --from <dir> [--revision <sha>]  local source
  verify    --model <key> [--repair] [--from <dir>]  streamed sha256 check
  gc        --model <key> [--apply]  keep-2, dry-run by default

Dev usage: uv run scripts/provision.py <subcommand> ...
Prod usage: image ghcr.io/charnesp/whisperx-provisioner:<ver> (T4b, Dockerfile
separate from the runtime image — out of scope of the immediate code change).
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import yaml

DEFAULT_MODELS_ROOT = Path("/models")
DEFAULT_STAGING_ROOT = Path("/models_staging") if Path("/models_staging").exists() else None
DEFAULT_KEEP = 2
DEFAULT_RECENT_DAYS = 14
EXCLUDED_SUFFIXES = (".pt", ".h5")
EXCLUDED_PATTERNS = ("*.flax*", "README*")
COMPLETE_MARKER = ".complete"
STATE_DIRNAME = ".state"
LOCKS_DIRNAME = ".locks"
ACTIVE_FILE = "active.json"
HISTORY_FILE = "history.jsonl"


class CommandError(Exception):
    """Fatal, user-facing provisioner error (exit code 1)."""


# ---------------------------------------------------------------------------
# Lock (v2)


def load_lock(lock_path: Path) -> dict[str, Any]:
    """Parse models.lock v2 and validate its structure minimally."""
    if not lock_path.exists():
        raise CommandError(f"E_MODEL_NOT_PROVISIONED: lock file missing: {lock_path}")
    try:
        lock = yaml.safe_load(lock_path.read_text())
    except yaml.YAMLError as exc:
        raise CommandError(f"models.lock unparseable: {exc}") from exc
    if not isinstance(lock, dict) or lock.get("version") != 2:
        raise CommandError(
            f"models.lock schema v2 expected (version: 2) in {lock_path}; "
            "regenerate with scripts/lock_audit.py"
        )
    models = lock.get("models")
    if not isinstance(models, list) or not models:
        raise CommandError("models.lock contains no models")
    for entry in models:
        rev = entry.get("revision", "")
        if not (isinstance(rev, str) and len(rev) == 40 and all(c in "0123456789abcdef" for c in rev)):
            raise CommandError(
                f"models.lock revision must be a 40-hex commit sha, got {rev!r} for {entry.get('repo')}"
            )
        files = entry.get("files")
        if not isinstance(files, list) or not files:
            raise CommandError(f"models.lock entry {entry.get('repo')} has no files list")
    return lock


def lock_entry(lock: dict[str, Any], model: str) -> dict[str, Any]:
    """Find a model entry in the parsed lock by key (dirname) or repo name."""
    for entry in lock["models"]:
        repo = entry.get("repo", "")
        if repo == model:
            return entry
        if repo.split("/")[-1].lower() == model.lower():
            return entry
        if entry.get("model") == model:
            return entry
    known = ", ".join(e.get("repo", "?") for e in lock["models"])
    raise CommandError(f"model {model!r} not found in models.lock (known: {known})")


# ---------------------------------------------------------------------------
# Checksums / atomic staging


def compute_file_sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    """Stream a file through sha256 without loading it in memory."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def fsync_dir(path: Path) -> None:
    """fsync a directory so renames/creates inside it are durable."""
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def stage_from_lock(
    src_dir: Path,
    staging_dir: Path,
    entry: dict[str, Any],
    exclusion_ok,
) -> None:
    """Copy files matching the lock entry from src_dir into staging_dir.

    Files not in the lock (or excluded) are not copied. The lock's sha256
    check happens on the staging copy so the source may be a partial cache.
    """
    staging_dir.mkdir(parents=True, exist_ok=True)
    for spec in entry["files"]:
        rel = spec["path"]
        source = src_dir / rel
        if not exclusion_ok(rel):
            continue
        if not source.is_file():
            raise CommandError(f"missing file in source: {rel} (expected {spec['size']} bytes)")
        dest = staging_dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, dest)
        actual = compute_file_sha256(dest)
        expected = spec["sha256"]
        if actual != expected:
            raise CommandError(
                f"checksum mismatch for {rel}: expected {expected}, got {actual} "
                "(source corrupted or wrong revision)"
            )


def provision_one(
    staging_dir: Path,
    dest_dir: Path,
    files: list[dict[str, Any]],
    complete_last: bool = True,
) -> None:
    """Atomically move a verified staging dir to its final destination.

    Idempotent: a destination already holding a complete, matching snapshot
    is never rewritten. The .complete marker is created LAST, after fsync,
    so a directory is only visible-complete when all files are durable.
    """
    if dest_dir.exists():
        marker = dest_dir / COMPLETE_MARKER
        if marker.exists():
            return  # already provisioned and verified: no-op
        shutil.rmtree(dest_dir)  # incomplete leftover: rebuild from staging
    # Verify staged content one more time (source of truth = lock specs).
    for spec in files:
        f = staging_dir / spec["path"]
        if not f.is_file():
            raise CommandError(f"staging missing file {spec['path']}")
        actual = compute_file_sha256(f)
        if actual != spec["sha256"]:
            raise CommandError(
                f"checksum mismatch for {spec['path']}: expected {spec['sha256']}, got {actual}"
            )
    dest_dir.parent.mkdir(parents=True, exist_ok=True)
    os.rename(staging_dir, dest_dir)  # same-FS rename: atomic
    fsync_dir(dest_dir.parent)
    if complete_last:
        marker = dest_dir / COMPLETE_MARKER
        with marker.open("w") as fh:
            fh.write("ok\n")
            fh.flush()
            os.fsync(fh.fileno())
        fsync_dir(dest_dir)


def resolve_models_root(explicit: str | None) -> Path:
    """Resolve the /models root: explicit flag > MODELS_ROOT env > /models."""
    if explicit:
        return Path(explicit)
    env = os.environ.get("MODELS_ROOT")
    if env:
        return Path(env)
    return DEFAULT_MODELS_ROOT


def complete_marker_path(models_root: Path, model: str, revision: str) -> Path:
    return models_root / model / revision / COMPLETE_MARKER


def model_revisions_on_disk(models_root: Path, model: str) -> list[str]:
    """List revision dirs (40-hex) present for a model, complete or not."""
    base = models_root / model
    if not base.is_dir():
        return []
    return sorted(
        p.name
        for p in base.iterdir()
        if p.is_dir() and len(p.name) == 40 and all(c in "0123456789abcdef" for c in p.name)
    )


# ---------------------------------------------------------------------------
# flock


def lock_file_path(models_root: Path, model: str) -> Path:
    """flock path OUTSIDE the versioned model dirs (GC must not delete it)."""
    return models_root / LOCKS_DIRNAME / f"{model}.lock"


def acquire_lock(models_root: Path, model: str):
    """Context manager holding LOCK_EX|LOCK_NB on <models>/.locks/<model>.lock."""
    from contextlib import contextmanager

    @contextmanager
    def _ctx():
        path = lock_file_path(models_root, model)
        path.parent.mkdir(parents=True, exist_ok=True)
        fh = open(path, "a+")
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            fh.close()
            raise CommandError(
                f"model {model!r} is locked by another provisioner/gc process ({exc}); "
                "retry later or skip this model"
            ) from exc
        try:
            yield fh
        finally:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
            fh.close()

    return _ctx()


def is_locked_elsewhere(lock_path: Path) -> bool:
    """True when some other process holds LOCK_EX on lock_path."""
    try:
        fh = open(lock_path, "a+")
    except FileNotFoundError:
        return False
    try:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
        return False
    except OSError:
        return True
    finally:
        fh.close()


# ---------------------------------------------------------------------------
# State


def update_state(models_root: Path, model: str, revision: str) -> None:
    """Record the provisioned revision in active.json + history.jsonl."""
    state_dir = models_root / STATE_DIRNAME
    state_dir.mkdir(parents=True, exist_ok=True)
    active_path = state_dir / ACTIVE_FILE
    active: dict[str, str] = {}
    if active_path.exists():
        try:
            active = json.loads(active_path.read_text())
        except json.JSONDecodeError:
            active = {}
    active[model] = revision
    tmp = state_dir / f"{ACTIVE_FILE}.tmp.{os.getpid()}"
    tmp.write_text(json.dumps(active, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, active_path)
    with (state_dir / HISTORY_FILE).open("a") as fh:
        fh.write(json.dumps({"model": model, "sha": revision, "ts": int(time.time())}) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def state_previous_sha(models_root: Path, model: str, exclude: str | None) -> str | None:
    """Most recent previously-active sha for a model (GC keep-2 protection)."""
    history = models_root / STATE_DIRNAME / HISTORY_FILE
    if not history.exists():
        return None
    last_other: str | None = None
    for line in history.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if rec.get("model") == model and rec.get("sha") and rec.get("sha") != exclude:
            last_other = rec["sha"]
    return last_other


def state_history_shas(models_root: Path, model: str) -> set[str]:
    """Revisions previously active for this model (protected by GC keep-2)."""
    history = models_root / STATE_DIRNAME / HISTORY_FILE
    if not history.exists():
        return set()
    shas: set[str] = set()
    for line in history.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if rec.get("model") == model and rec.get("sha"):
            shas.add(rec["sha"])
    return shas


# ---------------------------------------------------------------------------
# Exclusions


def exclusion_ok(rel_path: str) -> bool:
    """Match the provisioner's exclusion list (plan §2.2).

    Excluded: *.pt, *.flax*, *.h5, README*.
    """
    from fnmatch import fnmatch

    name = Path(rel_path).name
    if name.endswith(EXCLUDED_SUFFIXES):
        return False
    if name.startswith("README"):
        return False
    for pattern in EXCLUDED_PATTERNS:
        if fnmatch(name, pattern):
            return False
    return True


def lock_files_to_fetch(entry: dict[str, Any]) -> list[dict[str, Any]]:
    """Lock specs minus excluded files (weights dupes, README...)."""
    return [s for s in entry["files"] if exclusion_ok(s["path"])]


# ---------------------------------------------------------------------------
# Subcommands


def _resolve_revision(entry: dict[str, Any], requested: str | None) -> str:
    revision = requested or entry.get("revision")
    if not revision:
        raise CommandError(
            f"no revision pinned in models.lock for {entry.get('repo')}; pass --revision"
        )
    if revision != entry.get("revision"):
        raise CommandError(
            f"revision {revision} does not match lock revision {entry.get('revision')} "
            f"for {entry.get('repo')}: refusing to provision an unpinned sha"
        )
    return revision


def import_command(argv, models_root: Path | None = None, staging_root: Path | None = None) -> int:
    """import --model <key> --from <dir> [--revision <sha>] [--models-root <p>]."""
    parser = argparse.ArgumentParser(prog="import")
    parser.add_argument("--model", required=True)
    parser.add_argument("--from", dest="from_dir", required=True)
    parser.add_argument("--revision")
    parser.add_argument("--models-root")
    parser.add_argument("--staging-root")
    args = parser.parse_args(argv)

    root = models_root or resolve_models_root(args.models_root)
    lock = load_lock(root / "models.lock")
    entry = lock_entry(lock, args.model)
    revision = args.revision or entry.get("revision")
    if not revision:
        raise CommandError(
            f"no revision pinned in models.lock for {entry.get('repo')}; pass --revision"
        )
    src_dir = Path(args.from_dir)
    if not src_dir.is_dir():
        raise CommandError(f"source dir not found: {src_dir}")

    files = lock_files_to_fetch(entry)
    with acquire_lock(root, args.model):
        dest_dir = root / args.model / revision
        if complete_marker_path(root, args.model, revision).exists():
            print(f"already provisioned, no-op: {dest_dir}")
            return 0
        stage_dir = (staging_root or root / ".." / "staging") / f"{args.model}-{revision[:12]}-{os.getpid()}"
        stage_dir.parent.mkdir(parents=True, exist_ok=True)
        try:
            stage_from_lock(src_dir, stage_dir, entry, exclusion_ok)
            provision_one(stage_dir, dest_dir, files, complete_last=True)
        except CommandError:
            shutil.rmtree(stage_dir, ignore_errors=True)
            raise
        except OSError as exc:
            shutil.rmtree(stage_dir, ignore_errors=True)
            raise CommandError(f"provisioning failed: {exc}") from exc
    update_state(root, args.model, revision)
    print(f"provisioned {args.model}@{revision} -> {dest_dir}")
    return 0


def provision_command(argv, models_root: Path | None = None, staging_root: Path | None = None) -> int:
    """provision --model <key> [--revision <sha>]: download from HF Hub."""
    parser = argparse.ArgumentParser(prog="provision")
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision")
    parser.add_argument("--models-root")
    parser.add_argument("--staging-root")
    args = parser.parse_args(argv)

    root = models_root or resolve_models_root(args.models_root)
    lock = load_lock(root / "models.lock")
    entry = lock_entry(lock, args.model)
    revision = _resolve_revision(entry, args.revision)

    with acquire_lock(root, args.model):
        dest_dir = root / args.model / revision
        if complete_marker_path(root, args.model, revision).exists():
            print(f"already provisioned, no-op: {dest_dir}")
            return 0
        stage_dir = (staging_root or root / ".." / "staging") / f"{args.model}-{revision[:12]}-{os.getpid()}"
        stage_dir.parent.mkdir(parents=True, exist_ok=True)
        try:
            from huggingface_hub import snapshot_download

            # Cache HF INSIDE the staging dir (no second full storage).
            snapshot_download(
                repo_id=entry["repo"],
                revision=revision,
                local_dir=stage_dir,
                allow_patterns=[s["path"] for s in lock_files_to_fetch(entry)],
                ignore_patterns=["*.pt", "*.flax*", "*.h5", "README*"],
            )
            # Checksum every file against the lock, then atomic rename.
            for spec in lock_files_to_fetch(entry):
                f = stage_dir / spec["path"]
                if not f.is_file():
                    raise CommandError(f"download missing file: {spec['path']}")
                actual = compute_file_sha256(f)
                if actual != spec["sha256"]:
                    raise CommandError(
                        f"checksum mismatch for {spec['path']}: expected {spec['sha256']}, got {actual}"
                    )
            provision_one(stage_dir, dest_dir, lock_files_to_fetch(entry), complete_last=True)
        except CommandError:
            shutil.rmtree(stage_dir, ignore_errors=True)
            raise
        except OSError as exc:
            shutil.rmtree(stage_dir, ignore_errors=True)
            raise CommandError(f"provisioning failed: {exc}") from exc
    update_state(root, args.model, revision)
    print(f"provisioned {args.model}@{revision} -> {dest_dir}")
    return 0


def verify_command(argv, models_root: Path | None = None, staging_root: Path | None = None) -> int:
    """verify --model <key> [--repair] [--from <dir>]: streamed sha256 report."""
    parser = argparse.ArgumentParser(prog="verify")
    parser.add_argument("--model", required=True)
    parser.add_argument("--repair", action="store_true")
    parser.add_argument("--from", dest="from_dir")
    parser.add_argument("--models-root")
    parser.add_argument("--staging-root")
    args = parser.parse_args(argv)

    root = models_root or resolve_models_root(args.models_root)
    lock = load_lock(root / "models.lock")
    entry = lock_entry(lock, args.model)
    revision = entry.get("revision")
    dest_dir = root / args.model / revision

    marker = complete_marker_path(root, args.model, revision)
    if not marker.exists():
        raise CommandError(
            f"E_MODEL_NOT_PROVISIONED model={args.model} rev={revision} path={dest_dir} "
            "(no .complete marker)\n"
            "remédiation: docker run --rm -v /files/data/whisperx-cog/models:/models "
            "ghcr.io/charnesp/whisperx-provisioner:<ver> provision --model " + args.model
        )

    files = lock_files_to_fetch(entry)
    failures: list[tuple[str, str, str]] = []
    for spec in files:
        f = dest_dir / spec["path"]
        if not f.is_file():
            failures.append((spec["path"], spec["sha256"], "MISSING"))
            continue
        actual = compute_file_sha256(f)
        if actual != spec["sha256"]:
            failures.append((spec["path"], spec["sha256"], actual))
        print(f"ok   {spec['path']}  {spec['size']} bytes  {actual[:12]}…")
    if not failures:
        print(f"verify OK: {args.model}@{revision} ({len(files)} files)")
        return 0
    for path, expected, got in failures:
        print(f"FAIL {path}: expected {expected}, got {got}", file=sys.stderr)
    if args.repair:
        if not args.from_dir:
            raise CommandError(
                "verify --repair needs a valid source: pass --from <staging-dir> "
                "(cache Coder rsync) or use provision (HF download)"
            )
        src_dir = Path(args.from_dir)
        stage_dir = (staging_root or root / ".." / "staging") / f"{args.model}-repair-{os.getpid()}"
        stage_dir.parent.mkdir(parents=True, exist_ok=True)
        with acquire_lock(root, args.model):
            try:
                # Remove the faulty file(s) then re-provision from source.
                for path, _expected, _got in failures:
                    faulty = dest_dir / path
                    if faulty.is_file():
                        faulty.unlink()
                    elif faulty.exists():
                        shutil.rmtree(faulty)
                stage_from_lock(src_dir, stage_dir, entry, exclusion_ok)
                # Repair in place: copy back only the faulty files.
                for path, _expected, _got in failures:
                    fixed = stage_dir / path
                    target = dest_dir / path
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(fixed, target)
                    actual = compute_file_sha256(target)
                    if actual != _expected:
                        raise CommandError(
                            f"repair failed for {path}: expected {_expected}, got {actual}"
                        )
                fsync_dir(dest_dir)
            except CommandError:
                shutil.rmtree(stage_dir, ignore_errors=True)
                raise
        print(f"repaired {len(failures)} file(s) in {dest_dir}")
        return 0
    raise CommandError(
        f"verify FAILED for {args.model}@{revision}: {len(failures)} file(s) drifted; "
        "run again with --repair --from <staging-dir>"
    )


def gc_command(argv, models_root: Path | None = None, staging_root: Path | None = None) -> int:
    """gc --model <key> [--apply]: keep-2, dry-run by default."""
    parser = argparse.ArgumentParser(prog="gc")
    parser.add_argument("--model", required=True)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--keep", type=int, default=DEFAULT_KEEP)
    parser.add_argument("--recent-days", type=int, default=DEFAULT_RECENT_DAYS)
    parser.add_argument("--models-root")
    args = parser.parse_args(argv)

    root = models_root or resolve_models_root(args.models_root)
    lock = load_lock(root / "models.lock")
    entry = lock_entry(lock, args.model)
    pinned = entry.get("revision")
    active = {}
    active_path = root / STATE_DIRNAME / ACTIVE_FILE
    if active_path.exists():
        try:
            active = json.loads(active_path.read_text())
        except json.JSONDecodeError:
            active = {}
    current = active.get(args.model, pinned)
    protected: set[str] = {current, pinned}
    prev = state_previous_sha(root, args.model, exclude=current)
    if prev:
        protected.add(prev)
    protected |= {  # revisions younger than the recent window
        r for r in model_revisions_on_disk(root, args.model)
        if (root / args.model / r).stat().st_mtime >= time.time() - args.recent_days * 86400
    }
    # keep-2: everything outside the protected set is deletable. The
    # protected set already holds keep-2 (current ∪ previous ∪ pinned).
    candidates = [r for r in model_revisions_on_disk(root, args.model) if r not in protected]
    to_delete = candidates
    if not args.apply:
        if to_delete:
            print(
                f"GC dry-run for {args.model}: would delete {len(to_delete)} revision(s): "
                + ", ".join(to_delete)
                + "  (pass --apply to delete)"
            )
        else:
            print(f"GC dry-run for {args.model}: nothing to delete")
        return 0
    lock_path = lock_file_path(root, args.model)
    if is_locked_elsewhere(lock_path):
        print(f"gc: model {args.model!r} locked by another process, skipping")
        return 0
    with acquire_lock(root, args.model):
        deleted = []
        for rev in to_delete:
            target = root / args.model / rev
            shutil.rmtree(target)
            deleted.append(rev)
    if deleted:
        print(f"GC deleted {len(deleted)} revision(s) for {args.model}: " + ", ".join(deleted))
    else:
        print(f"GC: nothing deleted for {args.model}")
    return 0


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        print(__doc__)
        return 2
    cmd = argv[0]
    handlers = {
        "provision": provision_command,
        "import": import_command,
        "verify": verify_command,
        "gc": gc_command,
    }
    if cmd not in handlers:
        print(f"unknown subcommand: {cmd} (expected one of {', '.join(handlers)})", file=sys.stderr)
        return 2
    try:
        return handlers[cmd]([a for a in argv[1:] if a != cmd])
    except CommandError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
