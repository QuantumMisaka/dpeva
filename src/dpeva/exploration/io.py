"""I/O helpers for exploration backends."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from ase import Atoms
from ase.io import read


def read_structure_paths(paths: Iterable[str | Path]) -> list[Atoms]:
    structures: list[Atoms] = []
    for path in paths:
        loaded = read(Path(path))
        if isinstance(loaded, list):
            structures.extend(loaded)
        else:
            structures.append(loaded)
    return structures
