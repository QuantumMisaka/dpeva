# FP11 SAI-1344 campaign archive

These are historical reproduction assets for the completed July 2026 FP11
labeling campaign. They were retired from the general `scripts/` directory on
2026-11-17. They are not current labeling recipes: the wrappers retain the old
workspace, environment, tmux session names, and recovery job **581295**. Do not
use those defaults to operate a new campaign.

The completion record is in the [v0.8.1 campaign
plan](../../../docs/archive/v0.8.1/plans/2026-07-02-fp11-first-principles-labeling.md).
That frozen record retains its original paths and commands. Their relocation is:

| Former path | Archived path (relative to repository root) |
| --- | --- |
| `scripts/fp11_1344_make_config.py` | `practices/archive/fp11_1344/fp11_1344_make_config.py` |
| `scripts/fp11_1344_backend_report.py` | `practices/archive/fp11_1344/fp11_1344_backend_report.py` |
| `scripts/fp11_1344_recover_after_false_finish.py` | `practices/archive/fp11_1344/fp11_1344_recover_after_false_finish.py` |
| `scripts/fp11_1344_recover_wrapper.sh` | `practices/archive/fp11_1344/fp11_1344_recover_wrapper.sh` |
| `scripts/fp11_1344_finalize_after_execute.sh` | `practices/archive/fp11_1344/fp11_1344_finalize_after_execute.sh` |
| `scripts/fp11_submission_helpers.py` | `practices/archive/fp11_1344/fp11_submission_helpers.py` |

The Python implementation is unchanged by relocation. The two shell wrappers
call the archived Python paths. Existing tests under `tests/unit/scripts/test_fp11*`
load these files at their new location; they preserve config generation,
benchmark selection, recovery config normalization, and report parsing checks.

For a current campaign, use `dpeva label` and its `--stage` interface with an
explicit labeling config. Configure class-aware resources and Slurm arrays
through the supported configuration fields rather than reviving the fixed
recovery watcher. See the [CLI guide](../../../docs/guides/cli.md)
and [recipe index](../../../examples/recipes/README.md).

For exact historical reconstruction, use repository revision `cb666ba` (the
pre-retirement baseline), or follow the mapping above and explicitly audit the
old environment, resource policy and job identifiers. No forwarding scripts are
left at the old locations. This relocation does not change the installed
`dpeva` package, CLI, configuration aliases, or its v0.8 compatibility helpers.
