import os

import pytest

from dpeva.utils.security import normalize_sys_name, safe_join


def test_safe_join_blocks_traversal(tmp_path):
    base = tmp_path / "base"
    base.mkdir()
    with pytest.raises(ValueError):
        safe_join(str(base), "..", "outside")


def test_safe_join_blocks_prefix_escape(tmp_path):
    base = tmp_path / "base"
    base.mkdir()
    sibling = tmp_path / "base_other"
    sibling.mkdir()
    with pytest.raises(ValueError):
        safe_join(str(base), "..", "base_other", "x")


def test_normalize_sys_name_preserves_hierarchy():
    normalized = normalize_sys_name("alex-2d-1d-FeCOH/Fe0O0C1H5")
    assert normalized == os.path.join("alex-2d-1d-FeCOH", "Fe0O0C1H5")


def test_normalize_sys_name_rejects_traversal():
    with pytest.raises(ValueError):
        normalize_sys_name("../../etc/passwd")
