# DeepMD 3.2 CPU contract fixtures

The contract suite intentionally does not commit model weights or scientific
data.  The qualified environment supplies three explicit paths:

| Variable | Required content | Owner |
| --- | --- | --- |
| `DPEVA_DEEPMD_PT_MODEL` | A small DeepMD 3.2 PyTorch checkpoint usable by `dp --pt test`, `eval-desc`, and `embed` | Compatibility Owner |
| `DPEVA_DEEPMD_DPA4C_MODEL` | A periodic DPA4C exportable model usable by `dp --pt-expt eval-desc` | Compatibility Owner |
| `DPEVA_DEEPMD_PERIODIC_DATA` | A tiny labelled periodic DeepMD `npy` system (`coord.npy`, `box.npy`, labels and `type.raw`) | Compatibility Owner |

Paths are resolved before any DeepMD command starts.  If one explicitly named
fixture is not supplied, only tests requiring that fixture are skipped and the
skip message names the variable and owner.  Missing paths after a variable is
set are failures.  The DPA4C fixture is optional for local exploration but its
absence cannot promote the periodic capability.

The public pretrained PT model may be prepared outside pytest with the
qualified environment's documented DeepMD download command.  The ordinary
contract test run never downloads weights or accesses the network.

Outputs are created under pytest temporary directories.  CI retains command
stdout/stderr and the session summary in `build/deepmd-cpu-contract`.
