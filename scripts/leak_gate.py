#!/usr/bin/env python3
"""Repo leak gate (qwen3-asr-remote-vllm change, group 1).

Exhaustive regex scan of the worktree for personal IP/port leaks. The real
deploy address of the remote engine must NEVER appear anywhere in the
repository: code, tests, docs, compose, k8s, openspec.

Rules (flagged): any dotted-quad IPv4 literal outside the loopback
whitelist (private, public, link-local...); URL userinfo credentials
(scheme containing a credential pair); and host:port pairs that name an
address, either behind a URL scheme or with a dotted host, unless the
host is whitelisted or RFC 2606 reserved. The documented placeholder
tokens `<host>` and `<port>` are accepted everywhere.

Whitelist (bare literals AND any port): 127.0.0.1, 0.0.0.0, localhost —
loopback/any-interface usage is generic, not a personal address. Test
fixtures and docs use RFC 2606 hosts (*.invalid, *.test, *.example,
*.localhost, example.com/net/org): guaranteed non-routable, never flagged.

Usage: python3 scripts/leak_gate.py [repo_root]
Exit 0 = clean; exit 1 = leak, each hit printed as file:line:reason:match.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

# Loopback whitelist: the only allowed IPv4 literals and host names.
WHITELIST_V4 = {"127.0.0.1", "0.0.0.0"}
WHITELIST_NAMES = {"localhost"}

# RFC 2606 reserved names: mandated fixture placeholders, never flagged.
RFC2606_SUFFIXES = (".invalid", ".test", ".example", ".localhost")
RFC2606_NAMES_EXACT = {"example.com", "example.net", "example.org"}

# URL userinfo with credentials: scheme:// + user/password pair before @.
_USERINFO = re.compile(r"[a-zA-Z][a-zA-Z0-9+.-]*://[^\s/:@]+:[^\s/@]+@")
# Any dotted-quad IPv4 literal.
_IPV4 = re.compile(r"(?<![\w.])(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})(?![\w.])")
# host:port behind a URL scheme (real address usage).
_URL_HOSTPORT = re.compile(
    r"[a-zA-Z][a-zA-Z0-9+.-]*://(?P<host>[^\s/:@]+):(?P<port>\d{1,5})(?![\w.])"
)
# Dotted host (FQDN-like) with a port, outside any URL scheme: still an
# address. Host part must contain a dot; bare version tokens (v1.2, no
# port) never match. The numbered example below in prose is illustrative
# only and uses the reserved documentation address.
_DOTTED_HOSTPORT = re.compile(
    r"(?<![\w.:/])(?P<host>[A-Za-z0-9][A-Za-z0-9-]*(?:\.[A-Za-z0-9][A-Za-z0-9-]*)+):"
    r"(?P<port>\d{1,5})(?![\w.])"
)

_SKIP_DIRS = {
    ".git", ".venv", "__pycache__", ".ruff_cache", "node_modules",
    ".mypy_cache", ".pytest_cache",
}
_TEXT_SUFFIXES = {
    ".py", ".md", ".txt", ".yml", ".yaml", ".json", ".toml", ".cfg",
    ".ini", ".sh", ".html", ".css", ".js", ".provision",
}
_TEXT_FILES = {
    "Dockerfile", "Makefile", "Makefile.harness", "cog.yaml", ".gitignore",
    ".dockerignore", "models.lock", "AGENTS.md",
}


def _iter_files(root: Path):
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if any(part in _SKIP_DIRS for part in path.relative_to(root).parts):
            continue
        if path.name in _TEXT_FILES or path.suffix in _TEXT_SUFFIXES or not path.suffix:
            yield path


def _host_ok(host: str) -> bool:
    h = host.lower().rstrip(".")
    if h in WHITELIST_V4 or h in WHITELIST_NAMES:
        return True
    if h in RFC2606_NAMES_EXACT or h.endswith(RFC2606_SUFFIXES):
        return True
    # Documented placeholder tokens: the neutral doc placeholders mandated
    # by the spec (exact tokens, not substrings).
    if h in {"<host>", "<port>", "host", "port", "your_host", "my_host"}:
        return True
    return False


def scan(root: Path) -> list[tuple[str, int, str, str]]:
    findings: list[tuple[str, int, str, str]] = []
    for path in sorted(_iter_files(root)):
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            findings.append((str(path), 0, "unreadable file", str(exc)))
            continue
        rel = str(path.relative_to(root))
        for lineno, line in enumerate(text.splitlines(), start=1):
            for m in _USERINFO.finditer(line):
                findings.append((rel, lineno, "URL userinfo credentials", m.group()))
            for m in _IPV4.finditer(line):
                value = m.group(1)
                if value in WHITELIST_V4:
                    continue
                findings.append((rel, lineno, "non-whitelisted IPv4 literal", value))
            for pattern, label in (
                (_URL_HOSTPORT, "address in URL (host:port)"),
                (_DOTTED_HOSTPORT, "dotted host with port"),
            ):
                for m in pattern.finditer(line):
                    if _host_ok(m.group("host")):
                        continue
                    findings.append(
                        (rel, lineno, label, "%s:%s" % (m.group("host"), m.group("port")))
                    )
    return findings


def main(argv: list[str]) -> int:
    root = Path(argv[1]) if len(argv) > 1 else Path(__file__).resolve().parent.parent
    findings = scan(root)
    if findings:
        print("FAIL leak_gate: %d finding(s)" % len(findings))
        for rel, lineno, reason, value in findings:
            print("  %s:%s: %s: %s" % (rel, lineno, reason, value))
        return 1
    print("OK leak_gate: no personal IP/port in the repository")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
