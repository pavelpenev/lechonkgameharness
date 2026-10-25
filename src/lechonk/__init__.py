# SPDX-License-Identifier: MIT
"""lechonk — Le Chonk Plays Freeciv harness package.

Subpackages:
- ``lechonk.core``    — MIT. CivRealm-free schemas and (later) ledger/metrics.
- ``lechonk.adapter`` — GPL-3.0-or-later. The ONLY package that imports
  civrealm (vendored, GPLv3+). Imported lazily by the runner, only after the
  Docker stack is up (``import civrealm`` performs a network fetch at import
  time).
"""

__version__ = "0.1.0"
