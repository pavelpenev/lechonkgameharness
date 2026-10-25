# SPDX-License-Identifier: MIT
"""Core-purity test: importing lechonk.core must never transitively import
civrealm (design-plan-freeciv-v1.1.md §7.1: core is MIT and CivRealm-free;
adapter is the ONLY package that imports civrealm).

Two layers of enforcement:
1. a clean subprocess import of lechonk / lechonk.core / lechonk.core.schemas
   (and lechonk.adapter.obs_extract, which must itself stay civrealm-free for
   offline testing) must leave zero civrealm modules in sys.modules;
2. a source scan: no file under src/lechonk/core may reference civrealm, and
   the adapter's extractor must not import civrealm at module level (the runner
   imports the adapter lazily, only after the Docker stack is up, because
   ``import civrealm`` performs a network fetch at import time).
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CORE_DIR = REPO / "src" / "lechonk" / "core"
ADAPTER_DIR = REPO / "src" / "lechonk" / "adapter"

IMPORT_PROBE = (
    "import sys\n"
    "import lechonk\n"
    "import lechonk.core\n"
    "import lechonk.core.schemas\n"
    "import lechonk.adapter.obs_extract\n"
    "bad = sorted(m for m in sys.modules\n"
    "             if m == 'civrealm' or m.startswith('civrealm.'))\n"
    "print('CIVREALM_MODULES:', bad)\n"
    "sys.exit(1 if bad else 0)\n"
)


def test_importing_lechonk_core_never_imports_civrealm():
    proc = subprocess.run(
        [sys.executable, "-c", IMPORT_PROBE],
        capture_output=True, text=True, timeout=120,
        cwd=str(REPO),
    )
    assert proc.returncode == 0, (
        "lechonk.core (or the extractor) transitively imported civrealm:\n"
        f"stdout: {proc.stdout}\nstderr: {proc.stderr}")
    assert "CIVREALM_MODULES: []" in proc.stdout


def test_core_sources_never_reference_civrealm():
    """No import of civrealm anywhere under src/lechonk/core (docstring
    mentions of the invariant are fine; imports are not)."""
    import_re = re.compile(r"^\s*(import|from)\s+civrealm\b", re.MULTILINE)
    offenders = []
    for path in sorted(CORE_DIR.rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        if import_re.search(text):
            offenders.append(str(path))
    assert not offenders, f"core files importing civrealm: {offenders}"


def test_adapter_extractor_has_no_module_level_civrealm_import():
    """obs_extract must stay importable without civrealm (offline tests); the
    session module (later package) is the piece that owns the civrealm env."""
    for path in sorted(ADAPTER_DIR.rglob("*.py")):
        for lineno, line in enumerate(
                path.read_text(encoding="utf-8").splitlines(), start=1):
            stripped = line.strip()
            if (stripped.startswith("import civrealm")
                    or stripped.startswith("from civrealm")):
                raise AssertionError(
                    f"module-level civrealm import at {path}:{lineno}: {line}")


def test_core_package_declares_mit_spdx():
    for path in sorted(CORE_DIR.rglob("*.py")):
        first = path.read_text(encoding="utf-8").splitlines()[0]
        assert first == "# SPDX-License-Identifier: MIT", (
            f"{path} lost its MIT SPDX header")


def test_adapter_package_declares_gpl_spdx():
    """Plan decision 7: the adapter package is GPLv3+ and marked as such
    in-repo."""
    for path in sorted(ADAPTER_DIR.rglob("*.py")):
        first = path.read_text(encoding="utf-8").splitlines()[0]
        assert first == "# SPDX-License-Identifier: GPL-3.0-or-later", (
            f"{path} lost its GPL-3.0-or-later SPDX header")
