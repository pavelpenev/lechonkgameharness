# SPDX-License-Identifier: GPL-3.0-or-later
"""Canonical freeciv unit-activity id/name tables (version/v14.py:50-65).

Single source of truth for the wire-level activity ids: the name table and
the per-id constants live here; obs_extract.py, execute.py, and runner.py
all import from this module (C3a). Core stays civrealm-free — these ids are
source-verified wire constants, not civrealm imports.
"""

ACTIVITY_NAMES = {
    0: "idle", 1: "cultivate", 2: "mine", 3: "irrigate", 4: "fortified",
    5: "sentry", 6: "pillage", 7: "goto", 8: "explore", 9: "transform",
    10: "fortifying", 11: "clean", 12: "base", 13: "road", 14: "convert",
    15: "plant",
}

# Per-id constants for activity read-backs (execute.py).
ACTIVITY_IDLE = 0
ACTIVITY_MINE = 2
ACTIVITY_IRRIGATE = 3
ACTIVITY_FORTIFIED = 4
ACTIVITY_SENTRY = 5
ACTIVITY_FORTIFYING = 10
ACTIVITY_GEN_ROAD = 13
