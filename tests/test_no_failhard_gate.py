"""E5-LEGACY-HF — no fail-hard boot gate, no MODELS_MODE, no remediation.

Final Charles decision: the container starts and downloads whatever is
missing. The boot validator (E_MODEL_NOT_PROVISIONED crash-loop) and the
MODELS_MODE dev/online switch are REMOVED from the runtime path:
- predict.Predictor.setup() must NOT call any boot validation;
- model_paths must not raise when a model is absent (HF repo id fallback);
- MODELS_MODE must not influence resolution anymore.
RED until the implementation drops those guards.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tests"))

from _predict_stub import install  # noqa: E402

predict = install()


class TestNoBootGate(unittest.TestCase):
    def test_setup_does_not_call_boot_validator(self):
        """predict.py must not import/call the boot fail-fast validator."""
        src = (REPO_ROOT / "predict.py").read_text()
        self.assertNotIn(
            "validate_boot_models",
            src,
            "Predictor.setup must not gate boot on provisioned weights: the "
            "container starts and downloads what is missing",
        )
        self.assertNotIn("E_MODEL_NOT_PROVISIONED", src)

    def test_model_paths_has_no_models_mode_switch(self):
        src = (REPO_ROOT / "model_paths.py").read_text()
        self.assertNotIn("MODELS_MODE", src)
        self.assertNotIn("ModelNotProvisioned", src)

    def test_model_paths_has_no_remediation_prose(self):
        src = (REPO_ROOT / "model_paths.py").read_text()
        self.assertNotIn("Remediation:", src)
        self.assertNotIn("E_MODEL_NOT_PROVISIONED", src)


if __name__ == "__main__":
    unittest.main()
