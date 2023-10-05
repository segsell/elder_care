from __future__ import annotations

import parental_care


def test_import():
    assert hasattr(parental_care, "__version__")
