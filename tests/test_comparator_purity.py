# SPDX-License-Identifier: MIT
"""Comparator-purity test: the comparator package is MIT and CivRealm-free
(design-plan-freeciv-v1.1.md §7.1 — comparator/ is MIT; the adapter is the
ONLY package that imports civrealm, and importing the GPL adapter would taint
the MIT comparator).

Two layers of enforcement:
1. a clean subprocess import of lechonk.comparator must leave zero civrealm
   modules AND zero lechonk.adapter modules in sys.modules;
2. a source scan: no file under src/lechonk/comparator may import civrealm or
   lechonk.adapter (docstring mentions of the invariant are fine; imports are
   not), and every file carries the MIT SPDX header.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
COMPARATOR_DIR = REPO / "src" / "lechonk" / "comparator"

IMPORT_PROBE = (
    "import sys\n"
    "import lechonk.comparator\n"
    "import lechonk.comparator.scripted\n"
    "import lechonk.comparator.defaults\n"
    "bad_civ = sorted(m for m in sys.modules\n"
    "                 if m == 'civrealm' or m.startswith('civrealm.'))\n"
    "bad_adapter = sorted(m for m in sys.modules\n"
    "                     if m.startswith('lechonk.adapter'))\n"
    "print('CIVREALM_MODULES:', bad_civ)\n"
    "print('ADAPTER_MODULES:', bad_adapter)\n"
    "sys.exit(1 if (bad_civ or bad_adapter) else 0)\n"
)


def test_importing_comparator_never_imports_civrealm_or_adapter():
    proc = subprocess.run(
        [sys.executable, "-c", IMPORT_PROBE],
        capture_output=True, text=True, timeout=120,
        cwd=str(REPO),
    )
    assert proc.returncode == 0, (
        "lechonk.comparator transitively imported civrealm or lechonk.adapter:\n"
        f"stdout: {proc.stdout}\nstderr: {proc.stderr}")
    assert "CIVREALM_MODULES: []" in proc.stdout
    assert "ADAPTER_MODULES: []" in proc.stdout


def test_comparator_sources_never_import_civrealm_or_adapter():
    import_re = re.compile(
        r"^\s*(import|from)\s+(civrealm\b|lechonk\.adapter\b)", re.MULTILINE)
    offenders = []
    for path in sorted(COMPARATOR_DIR.rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        if import_re.search(text):
            offenders.append(str(path))
    assert not offenders, (
        f"comparator files importing civrealm or lechonk.adapter: {offenders}")


def test_comparator_package_declares_mit_spdx():
    for path in sorted(COMPARATOR_DIR.rglob("*.py")):
        first = path.read_text(encoding="utf-8").splitlines()[0]
        assert first == "# SPDX-License-Identifier: MIT", (
            f"{path} lost its MIT SPDX header")
