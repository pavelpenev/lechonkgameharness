# SPDX-License-Identifier: GPL-3.0-or-later
"""lechonk.adapter — GPL-3.0-or-later (design-plan-freeciv-v1.1.md decision 7).

This is the ONLY lechonk package that imports civrealm (the vendored GPLv3+
CivRealm stack). Keep this package free of module-level civrealm imports where
possible: ``import civrealm`` performs an HTTP pubstatus fetch at import time,
so the runner imports the adapter lazily, only after the Docker stack is up.
"""
