"""models.lock parsing + boot fail-fast validator (E5-CODE-1 T2).

Authority chain: models.lock (pinned revisions + expected files) -> the
validator runs FIRST in Predictor.setup(), before any model loading, so an
image started against an unprovisioned /models volume crash-loops with a
structured, actionable error instead of a deep HF_HUB_OFFLINE traceback.

Fast mode only at boot: os.stat (file presence) + exact size (when the
lock carries sizes) + the `.complete` marker. NEVER a sha256 at boot —
full hashing belongs to the provisioner's `verify` command.

Env knobs:
- MODELS_DIR (default /models): root holding <key>/<sha40>/ dirs
- MODELS_LOCK_PATH (default <repo>/models.lock): lock file override
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

BAKED_MODELS_ROOT = "/models"
DEFAULT_LOCK_PATH = Path(__file__).resolve().parent / "models.lock"

# sysexits.h: EX_CONFIG = 78 — distinct exit code for config/provisioning
# failures so orchestrators can tell them apart from runtime crashes.
EXIT_CONFIG = 78

PROVISIONER_IMAGE = "ghcr.io/charnesp/whisperx-provisioner:latest"
PROVISIONER_HOST_DIR = "/files/data/whisperx-cog/models"

REPO_ROOT = Path(__file__).resolve().parent
COMPLETE_MARKER = ".complete"

EXPECTED_KEYS = (
    "tiny",
    "large-v3",
    "large-v3-turbo",
    "qwen3-asr-1.7b",
    "qwen3-forced-aligner-0.6b",
)


class ModelsNotProvisioned(RuntimeError):
    """Fail-fast boot error: one or more locked models are not provisioned."""

    exit_code = EXIT_CONFIG


def _models_root(models_root: str | Path | None = None) -> Path:
    if models_root is not None:
        return Path(models_root)
    return Path(os.environ.get("MODELS_DIR", BAKED_MODELS_ROOT))


def _lock_path(lock_path: str | Path | None = None) -> Path:
    if lock_path is not None:
        return Path(lock_path)
    env = os.environ.get("MODELS_LOCK_PATH")
    if env:
        return Path(env)
    return DEFAULT_LOCK_PATH


def _parse_simple(text: str) -> dict[str, dict]:
    """Minimal YAML-subset parser for models.lock (list of flat mappings).

    Avoids a PyYAML runtime dependency in the Cog image: the lock uses only
    nested mappings/lists with scalar values and block/flow lists.
    """
    entries: dict[str, dict] = {}
    current: dict | None = None
    current_repo: str | None = None
    last_key: str | None = None
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip())
        stripped = line.strip()
        if stripped == "models:":
            continue
        # A list item starting at indent <= 2 is a NEW model entry ("- repo: …");
        # deeper list items ("- config.json") belong to expected_files.
        if stripped.startswith("- ") and indent <= 2:
            if current is not None and current_repo:
                entries[current_repo] = current
            current = {}
            current_repo = None
            last_key = None
            stripped = stripped[2:]
        if current is None:
            continue
        if stripped.startswith("- "):
            if last_key:
                item = stripped[2:].strip().strip("'\"")
                current.setdefault(last_key, []).append(item)
            continue
        if ":" in stripped:
            key, _, value = stripped.partition(":")
            key = key.strip()
            value = value.strip()
            if (
                indent >= 4
                and last_key
                and last_key not in current
            ):
                # First nested map entry under an empty-valued parent (sizes:).
                current[last_key] = {key: value}
                continue
            if (
                last_key
                and isinstance(current.get(last_key), dict)
            ):
                # Subsequent entries of the same nested map.
                current[last_key][key] = value
                continue
            if value.startswith("[") and value.endswith("]"):
                items = [
                    v.strip().strip("'\"")
                    for v in value[1:-1].split(",")
                    if v.strip()
                ]
                current[key] = items
            elif value == "":
                # Block sequence/map follows ("- item" or deeper "k: v" lines).
                last_key = key
            else:
                current[key] = value.strip("'\"")
                last_key = key
                if key == "repo" and current_repo is None:
                    current_repo = value.strip("'\"").split("/", 1)[-1]
    if current is not None and current_repo:
        entries[current_repo] = current
    return entries


# models.lock keys are the repo basenames stripped of org + the
# faster-whisper- prefix (e.g. Systran/faster-whisper-tiny -> "tiny"), so
# they match the registry keys AND the provisioned /models/<key>/ layout.
_REPO_TO_KEY = {
    "Qwen/Qwen3-ASR-1.7B": "qwen3-asr-1.7b",
    "Qwen/Qwen3-ForcedAligner-0.6B": "qwen3-forced-aligner-0.6b",
    "Systran/faster-whisper-tiny": "tiny",
    "Systran/faster-whisper-large-v3": "large-v3",
    "mobiuslabsgmbh/faster-whisper-large-v3-turbo": "large-v3-turbo",
}


def _lock_key(repo: str) -> str:
    if repo in _REPO_TO_KEY:
        return _REPO_TO_KEY[repo]
    base = repo.split("/", 1)[-1]
    return base[len("faster-whisper-") :] if base.startswith("faster-whisper-") else base


def _parse_v2(text: str) -> dict[str, dict]:
    """Minimal YAML-subset parser for models.lock schema v2.

    Schema (T5, one source of truth, sha256 per file):
        version: 2
        models:
        - repo: <hf repo>
          revision: <40-hex>
          required: true
          files:
          - path: <relative path>
            size: <bytes>
            sha256: <64-hex>
            lfs: <bool>

    Returns the same shape as _parse_simple: {model_key: {repo, revision,
    files: [{path, size, sha256, lfs}]}}. Files keep their declared order
    (the lock writes them sorted by path).
    """
    entries: dict[str, dict] = {}
    current: dict | None = None
    current_repo: str | None = None
    last_key: str | None = None
    in_files = False
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip())
        stripped = line.strip()
        if stripped.startswith("#") or stripped in ("models:", "version: 2"):
            continue
        if stripped.startswith("- ") and indent == 0:
            if current is not None and current_repo:
                entries[current_repo] = current
            current = {}
            current_repo = None
            last_key = None
            in_files = False
            stripped = stripped[2:]
        if current is None:
            continue
        if stripped.startswith("- "):
            item = stripped[2:].strip()
            if in_files and indent >= 2:
                # New file spec inside files:.
                current.setdefault("files", []).append(_parse_file_item(item))
                continue
            if last_key:
                current.setdefault(last_key, []).append(item)
            continue
        if ":" not in stripped:
            continue
        key, _, value = stripped.partition(":")
        key = key.strip()
        value = value.strip()
        if (
            indent >= 2
            and in_files
            and current.get("files")
            and not value.startswith("[")
        ):
            # Field of the current file spec (size:, sha256:, lfs:).
            fname = key
            current["files"][-1][fname] = value
            continue
        if value.startswith("[") and value.endswith("]"):
            items = [
                v.strip().strip("'\"")
                for v in value[1:-1].split(",")
                if v.strip()
            ]
            current[key] = items
        elif value == "":
            if key == "files":
                in_files = True
            last_key = key
        else:
            current[key] = value.strip("'\"")
            if key == "repo" and current_repo is None:
                current_repo = value.strip("'\"").split("/", 1)[-1]
            if key == "path" and in_files is False and current.get("files"):
                pass
    if current is not None and current_repo:
        entries[current_repo] = current
    return entries


def _parse_file_item(item: str) -> dict:
    """First field of a v2 file list item: '- path: config.json'."""
    if item.startswith("path:"):
        return {"path": item[len("path:"):].strip().strip("'\"")}
    return {"path": item}


def parse_lock(lock_file: str | Path = DEFAULT_LOCK_PATH) -> dict[str, dict]:
    """Parse models.lock -> {model_key: {repo, revision, expected_files[, sizes]}}.

    Schema detection by `version:` field:
    - v2 (files: [{path, size, sha256, lfs}]): the adapter derives
      expected_files from files[].path and sizes from files[].size — the
      boot contract (presence + exact size + .complete marker, NEVER
      sha256) is unchanged, only the lock shape moved (E5-LOCK-SYNC).
    - v1 (expected_files: list + sizes: map): still supported for backward
      compatibility with pre-T5 pinned locks and legacy fixtures.
    - any other version: explicit ValueError with the migration command
      (`uv run scripts/lock_audit.py --regenerate`), never a silent skip.

    Key = repo basename (the registry lock_key). Raises ValueError on a
    malformed entry (missing repo/revision/files or a non-40-hex revision).
    """
    path = Path(lock_file)
    text = path.read_text()
    version = _lock_version(text)
    if version is not None and version != 1 and version != 2:
        raise ValueError(
            f"{path}: unsupported models.lock schema version {version}; "
            "supported: 1 (legacy, expected_files) and 2 (files with sha256). "
            "Migrate: uv run scripts/lock_audit.py --regenerate"
        )
    entries = _parse_v2(text) if version == 2 else _parse_simple(text)
    lock: dict[str, dict] = {}
    for repo_key, entry in entries.items():
        repo = entry.get("repo")
        revision = entry.get("revision")
        files = entry.get("files")
        expected = entry.get("expected_files")
        sizes = entry.get("sizes")
        if files:
            # v2 → v1-shape adapter: expected_files/sizes derived from files[].
            expected = [f["path"] for f in files]
            sizes = {}
            for f in files:
                if f.get("size") not in (None, ""):
                    sizes[f["path"]] = int(f["size"])
        if not repo or not revision or not expected:
            raise ValueError(
                f"{path}: malformed models.lock entry for {repo_key!r}: "
                "repo, revision and expected_files/files are required"
            )
        if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
            raise ValueError(
                f"{path}: revision for {repo_key!r} is not a 40-hex sha: {revision!r}"
            )
        lock[_lock_key(repo)] = {
            "repo": repo,
            "revision": revision,
            "expected_files": list(expected),
        }
        if sizes:
            lock[_lock_key(repo)]["sizes"] = {
                name: int(size) for name, size in sizes.items()
            }
        if not lock[_lock_key(repo)]["expected_files"]:
            raise ValueError(
                f"{path}: expected_files must not be empty for {repo_key!r}"
            )
    return lock


def _lock_version(text: str) -> int | None:
    """Detect the models.lock schema version (v1 locks carry no version)."""
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if line.startswith("version:"):
            try:
                return int(line[len("version:"):].strip())
            except ValueError:
                return None
    return None


def expected_paths(
    models_root: str | Path, lock: dict[str, dict] | None = None
) -> dict[str, str]:
    """Revisioned layout /models/<key>/<sha40>/ per lock entry."""
    root = Path(models_root)
    lock = lock if lock is not None else parse_lock(_lock_path())
    return {
        key: str(root / key / entry["revision"]) for key, entry in lock.items()
    }


def expected_sizes(lock: dict[str, dict]) -> dict[tuple[str, str], int]:
    """{(key, filename): size} for lock entries that carry a size field.

    Lock v1 (no size) returns {}: fast_validate then checks presence +
    .complete only, never a byte count.
    """
    sizes: dict[tuple[str, str], int] = {}
    for key, entry in lock.items():
        for name, size in (entry.get("sizes") or {}).items():
            sizes[(key, name)] = int(size)
    return sizes


def _not_provisioned(
    key: str, entry: dict, models_root: Path, detail: str
) -> ModelsNotProvisioned:
    model_path = str(models_root / key / entry["revision"])
    remediation = (
        f"docker run --rm -v {PROVISIONER_HOST_DIR}:/models "
        f"{PROVISIONER_IMAGE} provision --model {key}"
    )
    return ModelsNotProvisioned(
        f"E_MODEL_NOT_PROVISIONED model={key} rev={entry['revision']} "
        f"path={model_path} ({detail}). "
        f"Remediation: {remediation}. "
        "If the model repo is gated/private, pass -e HF_TOKEN "
        "(token value never logged)."
    )


def fast_validate(
    models_root: str | Path, lock: dict[str, dict] | None = None
) -> None:
    """Boot-time fail-fast validation (fast mode).

    Per locked model: os.stat each expected file (presence) + exact size
    when the lock carries sizes + the .complete marker. NEVER sha256 at
    boot. Raises ModelsNotProvisioned (exit code 78) on the first missing
    model, listing every missing file for that model.
    """
    root = _models_root(models_root)
    lock = lock if lock is not None else parse_lock(_lock_path())
    sizes = expected_sizes(lock)
    for key, entry in lock.items():
        model_dir = root / key / entry["revision"]
        if not model_dir.is_dir():
            raise _not_provisioned(key, entry, root, "revision directory missing")
        missing = []
        for name in entry["expected_files"]:
            fpath = model_dir / name
            try:
                st = os.stat(fpath)
            except OSError:
                missing.append(name)
                continue
            want = sizes.get((key, name))
            if want is not None and st.st_size != want:
                missing.append(f"{name} (size {st.st_size} != {want})")
        if missing:
            raise _not_provisioned(
                key, entry, root, "missing/mismatched files: " + ", ".join(missing)
            )
        if not os.path.isfile(model_dir / COMPLETE_MARKER):
            raise _not_provisioned(
                key, entry, root, f"{COMPLETE_MARKER} marker absent (incomplete provisioning)"
            )


def assert_snapshot_weights(snapshot_dir: str | Path, weight_files) -> None:
    """Single check point for a snapshot's weight files (non-empty).

    Used by the fail-hard resolver contract; the runtime error keeps the
    ModelsNotProvisioned type (exit 78 semantics) for consistency.
    """
    missing = [
        name
        for name in weight_files
        if not os.path.isfile(os.path.join(snapshot_dir, name))
        or os.path.getsize(os.path.join(snapshot_dir, name)) == 0
    ]
    if missing:
        raise ModelsNotProvisioned(
            f"E_MODEL_WEIGHTS_MISSING path={snapshot_dir} "
            f"missing={', '.join(missing)}. "
            f"Remediation: docker run --rm -v {PROVISIONER_HOST_DIR}:/models "
            f"{PROVISIONER_IMAGE} provision."
        )


def boot_validate(
    models_root: str | Path | None = None,
    lock_file: str | Path | None = None,
) -> None:
    """Entry point used by Predictor.setup(): parse + fast_validate, logging
    a one-line success summary. Fails fast (exit 78 semantics) otherwise."""
    import logging

    logger = logging.getLogger(__name__)
    root = _models_root(models_root)
    lock = parse_lock(_lock_path(lock_file))
    fast_validate(root, lock)
    logger.info(
        "Model boot validation OK: %d provisioned models under %s",
        len(lock),
        root,
    )


def main() -> int:
    """CLI: python models_lock.py [--models-root DIR] [--lock PATH].

    Exit 78 (EX_CONFIG) on validation failure so the container orchestrator
    sees a distinct provisioning error.
    """
    args = list(sys.argv[1:])
    models_root = None
    lock_file = None
    it = iter(args)
    for arg in it:
        if arg == "--models-root":
            models_root = next(it, None)
        elif arg == "--lock":
            lock_file = next(it, None)
    try:
        boot_validate(models_root, lock_file)
    except ModelsNotProvisioned as exc:
        print(str(exc), file=sys.stderr)
        return exc.exit_code
    return 0


if __name__ == "__main__":
    sys.exit(main())
