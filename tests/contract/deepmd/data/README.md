# DeepMD 3.2 CPU contract fixtures

The contract suite intentionally does not commit model weights or scientific
data.  The qualified environment supplies three explicit paths:

| Variable | Required content | Owner |
| --- | --- | --- |
| `DPEVA_DEEPMD_PT_MODEL` | A small DeepMD 3.2 PyTorch checkpoint usable by `dp --pt test`, `eval-desc`, and `embed` | Compatibility Owner |
| `DPEVA_DEEPMD_DPA4C_MODEL` | A periodic DPA4C exportable model usable by `dp --pt-expt eval-desc` | Compatibility Owner |
| `DPEVA_DEEPMD_PERIODIC_DATA` | A tiny labelled periodic DeepMD `npy` system (`coord.npy`, `box.npy`, labels and `type.raw`) | Compatibility Owner |
| `DPEVA_DEEPMD_PT_HEAD` | Optional non-sensitive PT model head; required by multitask checkpoints and never defaulted | Compatibility Owner |
| `DPEVA_DEEPMD_DPA4C_HEAD` | Optional non-sensitive DPA4C model head; required by multitask checkpoints and never defaulted | Compatibility Owner |

Paths are resolved before any DeepMD command starts.  If one explicitly named
fixture is not supplied, only tests requiring that fixture are skipped and the
skip message names the variable and owner.  Missing paths after a variable is
set are failures.  The DPA4C fixture is optional for local exploration but its
absence cannot promote the periodic capability.

The two head variables are optional so that single-task checkpoints remain
compatible.  When a checkpoint is multitask, the caller must set the matching
head explicitly; whitespace-only values are treated as unset and no head is
ever guessed.  The PT and DPA4C commands append `--head` only when their
corresponding variable is non-empty.

The public pretrained PT model may be prepared outside pytest with the
qualified environment's documented DeepMD download command.  The ordinary
contract test run never downloads weights or accesses the network.

## Protected CI bundle

The protected CI environment provides `DPEVA_DEEPMD_CONTRACT_FIXTURE_URL` and
`DPEVA_DEEPMD_CONTRACT_FIXTURE_SHA256` as secrets or environment variables.
The first is a URL to a gzip-compressed tar archive; the second is its exact
SHA-256 digest.  CI downloads into the runner's temporary directory, verifies
the digest before extraction, and exports the three fixture variables with
`DPEVA_DEEPMD_CONTRACT_REQUIRED=1`.

The archive must have this layout (with no model or data files outside this
root):

```text
dpeva-deepmd-contract/
├── pt-model/model.pt
├── dpa4c-model/model.pt2
└── periodic-data/
    ├── type.raw
    ├── set.000/coord.npy
    ├── set.000/box.npy
    ├── set.000/energy.npy
    ├── set.000/force.npy
    └── set.000/virial.npy
```

The bundle is a protected research fixture.  Its URL, archive, weights and
labels must not be committed, printed, or redistributed through repository
artifacts; CI uploads only command logs and the pytest summary.  The fixture
owner is responsible for license review and for rotating the digest whenever
the bundle changes.  A changed digest is a deliberate review event, not an
automatic upgrade.

Outputs are created under pytest temporary directories.  CI retains command
stdout/stderr and the session summary in `build/deepmd-cpu-contract`.
