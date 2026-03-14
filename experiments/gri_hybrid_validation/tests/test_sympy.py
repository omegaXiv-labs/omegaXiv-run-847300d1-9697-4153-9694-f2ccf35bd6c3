from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from gri_validation.sympy_checks import run_sympy_validations  # noqa: E402


def test_sympy_report(tmp_path: Path) -> None:
    workspace_root = ROOT.parent.parent
    spec = workspace_root / "phase_outputs" / "SYMPY.md"
    out = tmp_path / "sympy_report.json"
    result = run_sympy_validations(spec, out)
    assert out.exists()
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["symbolic_identity_pass_rate"] >= 0.8
    assert result["checks"]["spec_exists"] is True
