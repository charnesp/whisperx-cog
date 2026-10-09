"""GPU-free unit tests for the repo leak gate (qwen3-asr-remote-vllm, group 1).

Two defense layers are unit-tested:
- wiring: the gate MUST be part of `make -f Makefile.harness check` (the
  harness gate is theoretical otherwise — a script nobody runs protects
  nobody);
- behavior: exhaustive-scan semantics proven SABOTAGE-PROOF. A clean tree
  yields 0 findings, and dynamically injecting an address into a tmpdir
  tree MUST be caught (the real RED criterion: a gate that catches nothing
  fails these tests).

Sabotage self-check paradox: the sabotage strings must NOT appear verbatim
in this file (the gate scans the whole repo, this test included), so every
sabotage address is assembled at runtime from fragments. Fixtures use RFC
2606 hosts (vllm.internal.invalid, example.test) per tasks.md group 1.
"""
from __future__ import annotations
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
LEAK_GATE_PATH = REPO_ROOT / "scripts" / "leak_gate.py"
MAKEFILE_PATH = REPO_ROOT / "Makefile.harness"

# Runtime-assembled sabotage fragments (never verbatim in this file).
IP_PRIVATE = "192.168." + "1.50"
IP_LINKLOCAL = "169.254." + "10.99"
IP_PUBLIC = "203.0.113." + "77"
HOST_CORP = "vllm.internal." + "corp"
HOST_ENGINE_CORP = "vllm-engine.internal." + "corp"


def _run_gate(root: Path):
    proc = subprocess.run(
        [sys.executable, str(LEAK_GATE_PATH), str(root)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    return proc.returncode, proc.stdout + proc.stderr


def _make_tree(files) -> Path:
    root = Path(tempfile.mkdtemp(prefix="leakgate-"))
    for rel, content in files:
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    return root

class TestLeakGateWiring(unittest.TestCase):
    """The gate is wired in the harness `check` target (task 1.2)."""

    def test_gate_script_exists(self):
        self.assertTrue(LEAK_GATE_PATH.is_file(), "scripts/leak_gate.py must exist")

    def test_check_target_invokes_leak_gate(self):
        text = MAKEFILE_PATH.read_text(encoding="utf-8")
        self.assertIn("leak_gate", text, "Makefile.harness check must call scripts/leak_gate.py")
        recipe = [
            line for line in text.splitlines()
            if line.startswith("\t") and "leak_gate" in line
        ]
        self.assertTrue(recipe, "check recipe must contain a leak_gate invocation line")
        self.assertIn("leak_gate.py", recipe[0])

    def test_gate_clean_repo_rc0(self):
        rc, out = _run_gate(REPO_ROOT)
        self.assertEqual(
            rc, 0,
            "gate on the clean repo returned rc=%s: %s" % (rc, out[-2000:]),
        )
        self.assertIn("OK leak_gate", out)

class TestLeakGateSabotageProof(unittest.TestCase):
    """Dynamic address injection in a tmpdir MUST be caught."""

    def test_clean_tmpdir_has_no_findings(self):
        root = _make_tree([
            ("README.md", "Remote engine: docs use placeholders only.\n"),
            (
                "src/clean.py",
                "BASE_URL = 'http://vllm.internal.invalid:9000/v1'  # RFC 2606\n",
            ),
        ])
        rc, out = _run_gate(root)
        self.assertEqual(rc, 0, "clean tmpdir flagged: %s" % out[-2000:])
        self.assertIn("OK leak_gate", out)

    def test_private_ipv4_caught(self):
        root = _make_tree([
            ("docs/config.md", "engine at " + IP_PRIVATE + ":9000 per ops\n"),
        ])
        rc, out = _run_gate(root)
        self.assertEqual(rc, 1)
        self.assertIn("FAIL leak_gate", out)
        self.assertIn(IP_PRIVATE, out)
        self.assertIn("docs/config.md", out)

    def test_link_local_ipv4_caught(self):
        root = _make_tree([
            ("k8s/inline.yaml", "endpoint: " + IP_LINKLOCAL + ":9000\n"),
        ])
        rc, out = _run_gate(root)
        self.assertEqual(rc, 1)
        self.assertIn(IP_LINKLOCAL, out)

    def test_public_nonwhitelisted_ipv4_caught(self):
        root = _make_tree([
            ("src/x.py", "addr = '" + IP_PUBLIC + ":443'\n"),
        ])
        rc, out = _run_gate(root)
        self.assertEqual(rc, 1)
        self.assertIn(IP_PUBLIC, out)

    def test_fqdn_with_port_caught(self):
        root = _make_tree([
            ("compose/x.yaml", "url: http://" + HOST_CORP + ":9000/v1  # sabotaged\n"),
        ])
        rc, out = _run_gate(root)
        self.assertEqual(rc, 1)
        self.assertIn(HOST_CORP + ":9000", out)

    def test_url_scheme_hostport_caught(self):
        root = _make_tree([
            ("src/x.py", "BASE = 'http://" + HOST_ENGINE_CORP + ":9000/v1'\n"),
        ])
        rc, out = _run_gate(root)
        self.assertEqual(rc, 1)
        self.assertIn(HOST_ENGINE_CORP + ":9000", out)

    def test_placeholders_accepted(self):
        root = _make_tree([
            (
                "README.md",
                "Set QWEN_REMOTE_BASE_URL=http://<host>:<port> (placeholder tokens).\n",
            ),
        ])
        rc, out = _run_gate(root)
        self.assertEqual(rc, 0, "placeholders flagged: %s" % out[-2000:])

    def test_whitelist_loopback_bare_and_port(self):
        root = _make_tree([
            (
                "tests/h.py",
                "loopback = ['127.0.0.1', '0.0.0.0', 'localhost', 'localhost:8080']\n",
            ),
        ])
        rc, out = _run_gate(root)
        self.assertEqual(rc, 0, "whitelisted loopback flagged: %s" % out[-2000:])

    def test_rfc2606_hosts_accepted(self):
        root = _make_tree([
            (
                "tests/h.py",
                "fixtures = ['example.invalid:9000', 'example.test:1',"
                " 'example.com:8080', 'vllm.example.test:9000']\n",
            ),
        ])
        rc, out = _run_gate(root)
        self.assertEqual(rc, 0, "RFC 2606 flags: %s" % out[-2000:])

    def test_version_tokens_not_flagged(self):
        root = _make_tree([
            (
                "docs/changelog.md",
                "vllm 0.30.0, cog 0.16.1, torch 2.8.0 (versions, no addresses)\n",
            ),
        ])
        rc, out = _run_gate(root)
        self.assertEqual(rc, 0, "version tokens flagged: %s" % out[-2000:])

    def test_userinfo_caught(self):
        scheme = "http://"
        cred = "saturnin:p4" + "ssw0rd@"
        host = "vllm-engine.corp" + ".example"
        root = _make_tree([
            ("src/x.py", "url = '" + scheme + cred + host + ":9000/v1'\n"),
        ])
        rc, out = _run_gate(root)
        self.assertEqual(rc, 1)
        self.assertIn("userinfo", out)

if __name__ == "__main__":
    unittest.main()
