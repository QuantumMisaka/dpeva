import os
import logging
import numpy as np
import dpdata
from typing import List, Literal, Optional

logger = logging.getLogger(__name__)

LMDB_FORMAT = "deepmd/lmdb"
# dpdata registers ``lmdb`` as a backward-compatible alias, but under dpdata
# 1.0.x that name resolves to a *different* legacy plugin.  DP-EVA therefore
# only ever uses the canonical name and normalizes user input to it.
LMDB_FORMAT_ALIASES = frozenset({"lmdb"})
EMPTY_POLICIES = ("error", "warn")
NPY_SYSTEM_MARKERS = ("type.raw", "type_map.raw", "set.000")
DEFAULT_FORMATS_TO_TRY = ("deepmd/npy/mixed", "deepmd/npy")


class DatasetLoadError(ValueError):
    """Raised when a dataset path cannot be loaded as any supported format."""


def is_lmdb_path(path) -> bool:
    """Return whether *path* addresses a DeepMD LMDB dataset.

    Mirrors deepmd-kit's own rule (``dpmodel/utils/lmdb_data.py::is_lmdb``) so
    that DP-EVA and the DeepMD CLI agree on what counts as an LMDB input:
    a ``.lmdb`` suffix, or a directory containing ``data.mdb``.
    """
    if path is None:
        return False
    try:
        text = os.fspath(path)
    except TypeError:
        return False
    if text.endswith(".lmdb"):
        return True
    return os.path.isfile(os.path.join(text, "data.mdb"))


def is_lmdb_detail_label(label: str) -> bool:
    """Return whether a ``dp test -d`` header names an LMDB data source.

    ``dp test`` on LMDB writes one header per nloc group in the form
    ``# <lmdb path> [nloc=<N>]: ...``, so either marker identifies the source.
    """
    text = str(label)
    return ".lmdb" in text or "[nloc=" in text


def detect_dataset_kind(path) -> Literal["lmdb", "single_system", "container", "missing", "unknown"]:
    """Classify a dataset path using the markers each format actually needs."""
    if path is None or not os.path.exists(path):
        return "missing"
    if is_lmdb_path(path):
        return "lmdb"
    if not os.path.isdir(path):
        return "unknown"
    if any(os.path.exists(os.path.join(path, marker)) for marker in NPY_SYSTEM_MARKERS):
        return "single_system"
    return "container"


def normalize_format(fmt: str) -> str:
    """Map accepted aliases onto the canonical dpdata format name."""
    if fmt in LMDB_FORMAT_ALIASES:
        logger.info("Normalized legacy LMDB format name %r to %r", fmt, LMDB_FORMAT)
        return LMDB_FORMAT
    return fmt


def _lmdb_stored_frame_count(path) -> Optional[int]:
    """Read ``nframes`` from an LMDB dataset's metadata, or ``None`` if unavailable."""
    try:
        import lmdb
        import msgpack
    except ImportError:  # pragma: no cover - dpdata>=1.1 always ships both
        return None
    try:
        env = lmdb.open(os.fspath(path), readonly=True, lock=False, subdir=True)
        try:
            with env.begin() as txn:
                raw = txn.get(b"__metadata__")
            if raw is None:
                return None
            metadata = msgpack.unpackb(raw, raw=False, strict_map_key=False)
        finally:
            env.close()
    except Exception as exc:  # pragma: no cover - defensive metadata probe
        logger.debug("Could not read LMDB metadata at %s: %s", path, exc)
        return None
    nframes = metadata.get("nframes")
    return int(nframes) if isinstance(nframes, int) else None


def _lmdb_group_name(system, path, index: int) -> str:
    """Build a readable label for one composition group read out of an LMDB."""
    names = list(system.data.get("atom_names", []))
    numbs = list(system.data.get("atom_numbs", []))
    formula = "".join(
        f"{name}{int(count)}" for name, count in zip(names, numbs) if int(count)
    )
    base = os.path.basename(os.path.normpath(os.fspath(path)))
    if formula:
        return f"{base}[{formula}]"
    return f"{base}[group{index}]"


def _load_lmdb_systems(path) -> List[dpdata.System]:
    """Load an LMDB dataset exclusively through ``dpdata.MultiSystems``.

    A single-system read (``dpdata.LabeledSystem``) silently returns only the
    first composition group of a multi-composition LMDB, so it must never be
    used for this format.
    """
    version = getattr(dpdata, "__version__", "unknown")
    try:
        multisystems = dpdata.MultiSystems.from_file(
            os.fspath(path), fmt=LMDB_FORMAT, max_frames=None
        )
    except Exception as exc:
        raise DatasetLoadError(
            f"Failed to load LMDB dataset at {path!r} with format {LMDB_FORMAT!r} "
            f"(dpdata {version}): {exc}. Reading {LMDB_FORMAT!r} requires dpdata>=1.1."
        ) from exc

    systems = list(multisystems)
    loaded_frames = sum(len(system) for system in systems)
    stored_frames = _lmdb_stored_frame_count(path)
    if stored_frames is not None and loaded_frames != stored_frames:
        raise DatasetLoadError(
            f"LMDB frame-count mismatch at {path!r}: the file records {stored_frames} "
            f"frames but {loaded_frames} were loaded. Refusing to continue with "
            "partial data."
        )
    for index, system in enumerate(systems):
        system.target_name = _lmdb_group_name(system, path, index)
    logger.info(
        "Loaded LMDB dataset %s: %s composition groups, %s frames.",
        path,
        len(systems),
        loaded_frames,
    )
    return systems


def _enforce_non_empty(
    systems: List, data_dir, formats_tried, on_empty: str
) -> List:
    """Apply the configured zero-system policy and keep the message actionable."""
    if systems:
        return systems
    message = (
        f"No systems could be loaded from {data_dir!r} "
        f"(tried formats: {list(formats_tried)}; dpdata {getattr(dpdata, '__version__', 'unknown')}). "
        f"Supported inputs are {DEFAULT_FORMATS_TO_TRY[0]}, {DEFAULT_FORMATS_TO_TRY[1]} and "
        f"{LMDB_FORMAT} (LMDB requires dpdata>=1.1)."
    )
    if on_empty == "error":
        raise DatasetLoadError(message)
    logger.warning(message)
    return []


def _resolve_target_system_dir(data_dir: str, sys_name: str):
    data_dir_abs = os.path.abspath(data_dir)
    data_base = os.path.basename(os.path.normpath(data_dir_abs))
    normalized = os.path.normpath(sys_name)
    normalized = normalized.lstrip("/\\")
    candidates_rel = []
    seen = set()

    def add_candidate(path_rel: str):
        path_rel_norm = os.path.normpath(path_rel).lstrip("/\\")
        if not path_rel_norm or path_rel_norm in seen:
            return
        seen.add(path_rel_norm)
        candidates_rel.append(path_rel_norm)

    add_candidate(normalized)
    prefix = f"{data_base}{os.sep}"
    if normalized.startswith(prefix):
        add_candidate(normalized[len(prefix):])

    parts = normalized.split(os.sep)
    if len(parts) > 1 and parts[0] == data_base:
        add_candidate(os.path.join(*parts[1:]))

    tried_paths = []
    for rel_path in candidates_rel:
        abs_path = os.path.join(data_dir_abs, rel_path)
        tried_paths.append(abs_path)
        if os.path.isdir(abs_path):
            return abs_path, tried_paths

    return None, tried_paths

def _load_single_path(path, sys_name, fmt="auto"):
    """
    Helper to load a single path with format retry.
    """
    # LMDB is a flat frame store: a single-system read would silently return
    # only the first composition group, so this path is reserved for
    # directory-per-system formats.
    if fmt in LMDB_FORMAT_ALIASES or fmt == LMDB_FORMAT or is_lmdb_path(path):
        raise DatasetLoadError(
            f"Refusing to load {path!r} through the single-system path. LMDB datasets must be "
            f"read with load_systems(), which uses dpdata.MultiSystems(fmt={LMDB_FORMAT!r})."
        )

    formats_to_try = list(DEFAULT_FORMATS_TO_TRY) if fmt == "auto" else [fmt]
    
    for f in formats_to_try:
        try:
            try:
                sys = dpdata.LabeledSystem(path, fmt=f)
            except Exception:
                sys = dpdata.System(path, fmt=f)
            
            # Attach target name for reference
            sys.target_name = sys_name
            # Fix duplicate atom names
            _fix_duplicate_atom_names(sys, sys_name)
            return sys
        except Exception:
            continue
    raise ValueError(f"Failed to load system at {path} with formats {formats_to_try}")

def load_systems(
    data_dir: str, 
    fmt: str = "auto", 
    target_systems: Optional[List[str]] = None,
    on_empty: Literal["error", "warn"] = "error",
) -> List[dpdata.System]:
    """
    Load systems from a directory using dpdata with auto-format detection.
    
    Args:
        data_dir (str): Path to the data directory.
        fmt (str): Format of the data. 
                   If "auto" (default), attempts to detect "deepmd/npy/mixed",
                   "deepmd/npy", or "deepmd/lmdb" (the latter requires dpdata>=1.1).
        target_systems (list, optional): List of specific system names to load. 
                                         If None, tries to load all subdirectories or the dir itself.
                                         Not supported for LMDB input, which has no directory-level names.
        on_empty (str): "error" (default) raises DatasetLoadError when nothing loads;
                        "warn" preserves the historical empty-list behaviour for callers
                        that deliberately tolerate an empty dataset.
    
    Returns:
        List[dpdata.System]: List of loaded dpdata Systems.
    """
    if on_empty not in EMPTY_POLICIES:
        raise ValueError(f"on_empty must be one of {EMPTY_POLICIES}, got {on_empty!r}")

    fmt = normalize_format(fmt)

    # LMDB is a single flat frame store, so it is detected before any
    # directory-by-directory discovery and never enters the single-system path.
    if fmt in ("auto", LMDB_FORMAT) and is_lmdb_path(data_dir):
        if target_systems:
            raise DatasetLoadError(
                f"target_systems={list(target_systems)!r} cannot be applied to LMDB input "
                f"{data_dir!r}: an LMDB file stores frames, not directory-level system names. "
                "Load the dataset without target_systems, or use its deepmd/npy copy."
            )
        return _enforce_non_empty(
            _load_lmdb_systems(data_dir), data_dir, [LMDB_FORMAT], on_empty
        )

    loaded_systems = []
    
    # 1. Determine directories to load
    if target_systems:
        dirs_to_load = []
        for sys_name in target_systems:
            resolved_dir, tried_paths = _resolve_target_system_dir(data_dir, sys_name)
            if resolved_dir is None:
                logger.warning(
                    f"Data directory not found for system: {sys_name}. Tried: {tried_paths}"
                )
                continue
            dirs_to_load.append((sys_name, resolved_dir))
    else:
        # Optimization: Check if data_dir is itself a system
        try:
            sys = _load_single_path(data_dir, os.path.basename(data_dir), fmt)
            return [sys]
        except Exception:
            pass

        # Try to load as MultiSystems first if no target specified (Auto-Discovery Mode)
        try:
            formats_to_try = list(DEFAULT_FORMATS_TO_TRY) if fmt == "auto" else [fmt]
            for f in formats_to_try:
                try:
                    ms = dpdata.MultiSystems.from_file(data_dir, fmt=f)
                    if len(ms) > 0:
                        logger.info(f"Loaded {len(ms)} systems using {f} format.")
                        fixed_systems = []
                        for s in ms:
                            _fix_duplicate_atom_names(s)
                            fixed_systems.append(s)
                        return fixed_systems
                except Exception:
                    continue
            
            # Fallback to scanning directories manually if MultiSystems fails
            subdirs = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
            # Filter out internal deepmd directories (set.*) to avoid false positives
            subdirs = [d for d in subdirs if not d.startswith("set.")]
            dirs_to_load = [(d, os.path.join(data_dir, d)) for d in subdirs]
            
        except Exception as e:
            if on_empty == "warn":
                logger.warning(f"Failed to auto-discover systems in {data_dir}: {e}")
                return []
            raise DatasetLoadError(
                f"Failed to auto-discover systems in {data_dir!r}: {e}. "
                f"Requested format: {fmt!r}."
            ) from e

    # 2. Load each directory
    for name, path in dirs_to_load:
        try:
            sys = _load_single_path(path, name, fmt)
            loaded_systems.append(sys)
        except Exception as e:
            logger.warning(f"Failed to load system at {path}: {e}")
            
    formats_tried = (
        list(DEFAULT_FORMATS_TO_TRY) if fmt == "auto" else [fmt]
    )
    return _enforce_non_empty(loaded_systems, data_dir, formats_tried, on_empty)

def _fix_duplicate_atom_names(sys: dpdata.System, sys_name: str = "Unknown"):
    """
    Detects and merges duplicate atom names in a dpdata System.
    Modifies the system in-place.

    Args:
        sys (dpdata.System): The dpdata System object to fix.
        sys_name (str, optional): Name of the system for logging. Defaults to "Unknown".
    """
    atom_names = sys['atom_names']
    if len(atom_names) != len(set(atom_names)):
        logger.warning(f"Duplicate atom names detected in {sys_name}: {atom_names}. Merging duplicate types.")
        
        # 1. Determine unique names (preserve order)
        new_atom_names = []
        seen = set()
        for name in atom_names:
            if name not in seen:
                new_atom_names.append(name)
                seen.add(name)
        
        # 2. Map old index to new index
        old_to_new_map = {}
        for old_idx, name in enumerate(atom_names):
            new_idx = new_atom_names.index(name)
            old_to_new_map[old_idx] = new_idx
        
        # 3. Update atom_types
        old_atom_types = sys['atom_types']
        new_atom_types = np.array([old_to_new_map[t] for t in old_atom_types], dtype=int)
        
        # 4. Update atom_numbs
        new_atom_numbs = []
        for i in range(len(new_atom_names)):
            count = np.sum(new_atom_types == i)
            new_atom_numbs.append(int(count))
        
        logger.info(f"Merged atom names to: {new_atom_names}")
        
        # 5. Apply changes
        sys.data['atom_names'] = new_atom_names
        sys.data['atom_numbs'] = new_atom_numbs
        sys.data['atom_types'] = new_atom_types
