"""Load the frozen compaction snapshot as ``recovered_dobby``.

``tests/vendor/recovered_dobby`` is the historical SDK at
``94b5a8f1fd28e6257cac84237072e23760acc54f``. Importing it under its own
package name leaves the installed ``dobby`` package untouched.
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

_VENDOR = Path(__file__).resolve().parent / "vendor"


def load_recovered() -> Any:
    """Import the in-tree snapshot without replacing installed ``dobby``."""
    loaded = sys.modules.get("recovered_dobby")
    if loaded is not None:
        return loaded
    entry = str(_VENDOR)
    if entry not in sys.path:
        sys.path.insert(0, entry)
    import recovered_dobby

    return recovered_dobby
