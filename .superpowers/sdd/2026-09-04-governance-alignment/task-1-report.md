# Plan E Task 1 report

## Scope

Defined the single executable TOML gate catalog and a safe argv-only runner.
The catalog records the existing local, PR, docs, integration, optional-extra,
DeepMD contract, qualification, and release profiles.  DeepMD qualification
remains an explicitly named evidence profile and is not included in `release`.

## TDD evidence

- RED: `pytest tests/unit/scripts/test_run_gate.py -q` failed during collection
  with `ModuleNotFoundError: No module named 'scripts.run_gate'`.
- GREEN: the focused suite passed with `11 passed`.

## Implementation decisions

- `load_manifest()` strictly validates the `1.0` schema, rejects unknown
  top-level/gate fields, missing fields, malformed argv, empty metadata, and
  unknown or duplicate profile references. TOML duplicate keys are rejected by
  the parser.
- Gate execution passes a list argv to `subprocess.run`, explicitly sets
  `shell=False`, uses the repository root as `cwd`, stops at the first
  non-zero return code, and returns `127` when an executable cannot be started.
- The CLI accepts one gate or profile, while `--list` only prints the catalog
  and never starts a subprocess. Its manifest path is anchored to the script,
  so invocation from another working directory remains stable.
- The manifest uses the existing `scripts/check_docs_freshness.py` filename.

## Verification

- `conda run -n dpeva-dpa4 pytest tests/unit/scripts/test_run_gate.py -q` — `11 passed`.
- `conda run -n dpeva-dpa4 ruff check scripts/run_gate.py tests/unit/scripts/test_run_gate.py` — passed.
- `conda run -n dpeva-dpa4 pytest tests/unit -q` — `834 passed`, 5 warnings.
- `python scripts/run_gate.py --list` — listed all 14 gates and 7 profiles without execution.
- `git diff --check` — passed.

## Changed files

- `scripts/gates.toml`
- `scripts/run_gate.py`
- `tests/unit/scripts/test_run_gate.py`

## Remaining boundary

Task 2 must migrate local and CI entry points to invoke profile/gate names and
remove duplicated command definitions. This task does not change CI or
`scripts/gate.sh`.

## Fix round 1

- The Python 3.10 documentation workflow installation now includes `tomli`,
  matching the runner's import fallback.
- The unit suite reloads the runner in an isolated module while simulating a
  missing `tomllib` and supplying a `tomli` stub, then exercises manifest
  loading through the fallback path. This verifies executable behavior rather
  than only checking source text.
- Removed the extra EOF blank line from `scripts/run_gate.py`.

Verification for this round:

- `conda run -n dpeva-dpa4 pytest tests/unit/scripts/test_run_gate.py -q` — `12 passed`.
- `conda run -n dpeva-dpa4 ruff check scripts/run_gate.py tests/unit/scripts/test_run_gate.py` — passed.
- `conda run -n dpeva-dpa4 pytest tests/unit -q` — `835 passed`, 5 warnings.
- `git diff --check` — passed.
