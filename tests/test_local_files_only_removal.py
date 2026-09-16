"""E5-LEGACY-HF — predict.py must not pin local_files_only anywhere.

Final Charles decision: NO anti-download guard at all. The container may
download whatever it misses (pyannote included), as long as the main
models resolve from /models when provisioned (prefer-local-then-HF).
RED: every `local_files_only=` keyword in predict.py / golden_set.py must
be gone.
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


class TestNoLocalFilesOnlyPins(unittest.TestCase):
    def test_predict_has_no_local_files_only(self):
        src = (REPO_ROOT / "predict.py").read_text()
        self.assertNotIn(
            "local_files_only",
            src,
            "predict.py must not pin local_files_only: the container is "
            "allowed to download what it misses (legacy HF behavior)",
        )

    def test_golden_set_has_no_local_files_only(self):
        src = (REPO_ROOT / "scripts" / "golden_set.py").read_text()
        self.assertNotIn("local_files_only", src)


if __name__ == "__main__":
    unittest.main()
