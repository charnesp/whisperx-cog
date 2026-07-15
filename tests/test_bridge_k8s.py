"""Unit tests for bridge ↔ k8s deployment checks."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from bridge_k8s import (
    BRIDGE_IMAGE,
    K8S,
    bridge_container_image,
    bridge_source_files_present,
    k8s_uses_bridge_configmap,
)


class TestBridgeK8sDeployment(unittest.TestCase):
    def test_manifest_exists(self):
        self.assertTrue(K8S.is_file())

    def test_uses_ghcr_bridge_image(self):
        k8s = K8S.read_text()
        self.assertEqual(bridge_container_image(k8s), BRIDGE_IMAGE)

    def test_no_configmap_embed(self):
        k8s = K8S.read_text()
        self.assertFalse(k8s_uses_bridge_configmap(k8s))

    def test_bridge_source_files_present(self):
        self.assertEqual(bridge_source_files_present(), [])


if __name__ == "__main__":
    unittest.main()
