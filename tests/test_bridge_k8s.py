"""Unit tests for bridge ↔ k8s ConfigMap sync helpers."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from bridge_k8s import embed_bridge_in_k8s, extract_configmap_bridge, read_standalone_bridge, K8S


class TestBridgeK8sRoundtrip(unittest.TestCase):
    def test_extract_matches_standalone(self):
        standalone = read_standalone_bridge()
        embedded = extract_configmap_bridge(K8S.read_text())
        self.assertIsNotNone(embedded)
        self.assertEqual(standalone, embedded)

    def test_embed_roundtrip(self):
        bridge = read_standalone_bridge()
        k8s = K8S.read_text()
        updated = embed_bridge_in_k8s(bridge, k8s)
        self.assertIsNotNone(updated)
        self.assertEqual(extract_configmap_bridge(updated), bridge)


if __name__ == "__main__":
    unittest.main()
