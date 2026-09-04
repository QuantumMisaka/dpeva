#!/usr/bin/env python3
"""Prepare the small, periodic input and immutable model references.

This script deliberately records research-pipeline models in place.  It never
copies a checkpoint into the DP-EVA repository or qualification directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_fixture(root: Path) -> dict[str, Any]:
    data = root / "input"
    set_dir = data / "set.000"
    set_dir.mkdir(parents=True, exist_ok=True)
    # Four atoms, one periodic frame.  This is an execution fixture only; it
    # is not a scientific quality or composition claim.
    np.save(set_dir / "coord.npy", np.array([[0.0, 0.0, 0.0, 1.5, 1.5, 1.5, 3.0, 3.0, 3.0, 4.5, 4.5, 4.5]], dtype=np.float64))
    np.save(set_dir / "box.npy", np.array([[6.0, 0.0, 0.0, 0.0, 6.0, 0.0, 0.0, 0.0, 6.0]], dtype=np.float64))
    np.save(set_dir / "energy.npy", np.array([[0.0]], dtype=np.float64))
    np.save(set_dir / "force.npy", np.zeros((1, 12), dtype=np.float64))
    np.save(set_dir / "virial.npy", np.zeros((1, 9), dtype=np.float64))
    # DeepMD/npy expects type.raw at the system root (not inside set.000).
    (data / "type.raw").write_text("0 1 2 3\n", encoding="utf-8")
    (data / "type_map.raw").write_text("Fe\nC\nH\nO\n", encoding="utf-8")
    return {
        "path": str(data.resolve()),
        "relative_path": "input",
        "format": "deepmd/npy",
        "periodic": True,
        "type_map": ["Fe", "C", "H", "O"],
        "frame_count": 1,
        "atom_count": 4,
        "qualification_only": True,
        "model_input_semantics": "periodic DPA4 FT2DP type-map fixture; execution qualification only",
    }


def prepare(model_root: Path, output: Path) -> dict[str, Any]:
    model_root = model_root.expanduser().resolve()
    output = output.expanduser().resolve()
    if not model_root.is_dir():
        raise FileNotFoundError(f"model root does not exist: {model_root}")
    regular = model_root / "model.ckpt.pt"
    ema = model_root / "model_ema.ckpt.pt"
    missing = [str(path) for path in (regular, ema) if not path.is_file()]
    if missing:
        raise FileNotFoundError("required research model artifact is missing: " + ", ".join(missing))

    root = output.parent
    root.mkdir(parents=True, exist_ok=True)
    fixture = _write_fixture(root)
    payload: dict[str, Any] = {
        "schema_version": "1.0",
        "qualification": "deepmd-3.2-sai-v100",
        "model_root": str(model_root),
        "models": {
            "regular": {"path": str(regular), "sha256": sha256(regular)},
            "ema": {"path": str(ema), "sha256": sha256(ema)},
        },
        "fixture": fixture,
        "dpa4c_model_path": os.environ.get("DPEVA_DEEPMD_DPA4C_MODEL"),
        "dpa4c_model_sha256": (
            sha256(Path(os.environ["DPEVA_DEEPMD_DPA4C_MODEL"]).expanduser().resolve())
            if os.environ.get("DPEVA_DEEPMD_DPA4C_MODEL")
            and Path(os.environ["DPEVA_DEEPMD_DPA4C_MODEL"]).expanduser().is_file()
            else None
        ),
        "required_cases": [
            "pip-freeze", "deepmd-version", "torch-cuda", "gpu",
            "pt-test", "pt-test-ema", "pt-eval-desc", "pt-eval-desc-ema",
            "pt-embed", "pt-embed-ema", "dpa4c-periodic-eval-desc",
        ],
    }
    # Atomic only within the caller-owned build directory; never replace an
    # existing input declaration.
    if output.exists():
        raise FileExistsError(f"refusing to overwrite prepared input: {output}")
    fd, name = tempfile.mkstemp(prefix=f".{output.name}.", dir=str(root), text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(name, output)
    finally:
        try:
            Path(name).unlink()
        except FileNotFoundError:
            pass
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    payload = prepare(args.model_root, args.output)
    print(json.dumps({"output": str(args.output.resolve()), "model_root": payload["model_root"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
