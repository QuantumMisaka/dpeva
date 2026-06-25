# DP embed vs eval-desc 对比验证报告

日期：2026-06-24

## 结论

1. **数值一致性成立。** 在同一最新 DeepMD/DP-EVA 环境 `dpeva-dpa4-embed-test` 中，`dp --pt embed` 写出的 HDF5 `descriptor` 与 `dp --pt eval-desc` 写出的 `.npy` descriptor 在 `test/deepmd-kit/examples/water/data/data_0..3` 上最大绝对误差为 `1.49e-7` 到 `1.79e-7`；在 DPA4-Plus 压力样本 `216` 上最大绝对误差为 `8.58e-6`。这满足本次按 fp32/数值推理路径设定的有效数字要求。
2. **旧环境 legacy `eval-desc` 的资源问题被复现。** 在 `dpeva-dpa4-test` 环境中，`eval-desc` 对 DPA4-Plus OOM 复现样本仍然触发 CUDA OOM：batch size 降到 1 后仍失败。相同样本在 `dpeva-dpa4-embed-test` 中用 fixed `eval-desc` 与 `embed` 都能完成。
3. **效率/显存结论需要分两层表述。** 相比旧环境 legacy `eval-desc`，当前 `embed` 显存使用显著更低或至少避免 OOM；但相比最新 DeepMD 中已经修复的 `eval-desc`，`embed` 与 fixed `eval-desc` 的 wall time 和显存峰值基本相同。也就是说，真正的资源收益来自 DeepMD 修复后的推理路径；`embed` 的额外价值是一次性输出 HDF5 中的 `descriptor`、`atomic_feature`、`structural_feature`，便于 DP-EVA descriptor、2-DIRECT 和 LLPR last-layer 统一消费。

## 测试范围

测试脚本：

- [compare_eval_desc_embed.py](../../../../scripts/validation/compare_eval_desc_embed.py)
- [run_compare_eval_desc_embed_4v100px.slurm](../../../../scripts/validation/run_compare_eval_desc_embed_4v100px.slurm)

原始结果：

- `practices/dpeva-dpa4-test/dpa4-dpeva-test/embed_eval_desc_comparison/results_v4/results.json`
- `practices/dpeva-dpa4-test/dpa4-dpeva-test/embed_eval_desc_comparison/results_v4/summary.md`
- Slurm job：`560130`
- 硬件：`4V100PX`，单卡，`--nodes=1 --ntasks=1 --gpus-per-node=1 --qos=rush-1o2gpu`

对比对象：

- legacy old：`dpeva-dpa4-test` 环境中的 `dp --pt eval-desc`
- fixed eval-desc：`dpeva-dpa4-embed-test` 环境中的 `dp --pt eval-desc`
- embed：`dpeva-dpa4-embed-test` 环境中的 `dp --pt embed`

## 数据集

| case | 来源 | 帧数 | 原子数 | 模型 |
| --- | --- | ---: | ---: | --- |
| `water_data_0` | `test/deepmd-kit/examples/water/data/data_0` | 80 | 192 | `test/deepmd-kit/examples/water/dpa4/lmp/pretrained.pt` |
| `water_data_1` | `test/deepmd-kit/examples/water/data/data_1` | 160 | 192 | `test/deepmd-kit/examples/water/dpa4/lmp/pretrained.pt` |
| `water_data_2` | `test/deepmd-kit/examples/water/data/data_2` | 80 | 192 | `test/deepmd-kit/examples/water/dpa4/lmp/pretrained.pt` |
| `water_data_3` | `test/deepmd-kit/examples/water/data/data_3` | 80 | 192 | `test/deepmd-kit/examples/water/dpa4/lmp/pretrained.pt` |
| `dpa4_plus_oom_216` | historical OOM reproducer | 59 | 216 | `practices/.../DPA4-Plus-OMat24-16M.pt` |

前 4 个是本次按要求使用的 `test/` 内测试集；最后一个是之前 issue/OOM 复现用压力样本，用于确认旧方法资源问题。

## 数值对比

同一最新环境内 `embed` vs fixed `eval-desc`：

| case | dtype | shape | max abs | mean abs | `allclose(rtol=1e-5, atol=1e-6)` |
| --- | --- | --- | ---: | ---: | --- |
| `water_data_0` | native | `80x192x16` | `1.788e-7` | `7.718e-9` | true |
| `water_data_0` | fp32 | `80x192x16` | `1.490e-7` | `7.686e-9` | true |
| `water_data_1` | native | `160x192x16` | `1.788e-7` | `7.681e-9` | true |
| `water_data_1` | fp32 | `160x192x16` | `1.788e-7` | `7.691e-9` | true |
| `water_data_2` | native | `80x192x16` | `1.788e-7` | `7.559e-9` | true |
| `water_data_2` | fp32 | `80x192x16` | `1.788e-7` | `7.537e-9` | true |
| `water_data_3` | native | `80x192x16` | `1.490e-7` | `7.577e-9` | true |
| `water_data_3` | fp32 | `80x192x16` | `1.490e-7` | `7.514e-9` | true |
| `dpa4_plus_oom_216` | native | `59x216x64` | `8.583e-6` | `4.595e-7` | false |
| `dpa4_plus_oom_216` | fp32 | `59x216x64` | `8.583e-6` | `4.589e-7` | false |

压力样本的最大绝对误差仍低于 `1e-5`，按 `allclose(rtol=1e-5, atol=3e-5)` 通过。其 `rtol=1e-5, atol=1e-6` 不通过的原因是 descriptor 中存在接近零的元素，最大相对误差会被近零分母放大；绝对误差仍处在可接受有效数字范围。

legacy old `eval-desc` vs 当前 `embed`：

- `test/` water 四个 case 的最大绝对误差为 `2.24e-5` 到 `2.72e-5`，平均绝对误差约 `8.1e-7` 到 `8.3e-7`。
- 这些差异混入了 DeepMD 版本/实现差异，不应作为纯方法误差；同环境对比才是判断 `embed` 和 `eval-desc` 数值一致性的主要证据。

## 性能与显存

`test/` water 数据集，fp32 路径：

| case | legacy old eval-desc GPU MiB | fixed eval-desc GPU MiB | embed GPU MiB | legacy/embed 显存比 | fixed eval-desc s | embed s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `water_data_0` | 10952 | 3388 | 3388 | 3.23x | 7.660 | 8.165 |
| `water_data_1` | 15770 | 6288 | 6288 | 2.51x | 7.657 | 8.166 |
| `water_data_2` | 10952 | 3386 | 3386 | 3.23x | 7.659 | 7.654 |
| `water_data_3` | 10952 | 3388 | 3388 | 3.23x | 7.658 | 7.658 |

平均值：

- legacy old `eval-desc`：GPU 峰值增量 `12156.5 MiB`，平均 wall time `7.785 s`
- fixed `eval-desc fp32`：GPU 峰值增量 `4112.5 MiB`，平均 wall time `7.659 s`
- `embed fp32`：GPU 峰值增量 `4112.5 MiB`，平均 wall time `7.911 s`

压力样本：

| method | status | wall s | CPU max RSS GiB | GPU peak delta MiB | output size MiB |
| --- | --- | ---: | ---: | ---: | ---: |
| legacy old `eval-desc` | failed, CUDA OOM | 8.183 | 1.718 | 16028 | 0.000 |
| fixed `eval-desc native` | success | 17.862 | 1.498 | 15388 | 6.223 |
| fixed `eval-desc fp32` | success | 17.394 | 1.492 | 15388 | 3.111 |
| `embed native` | success | 17.871 | 1.493 | 15388 | 11.084 |
| `embed fp32` | success | 17.919 | 1.519 | 15388 | 10.706 |

legacy old `eval-desc` 的 stderr 中包含：

```text
torch.OutOfMemoryError: CUDA out of memory.
deepmd.utils.errors.OutOfMemoryError: The callable still throws an out-of-memory (OOM) error even when batch size is 1!
```

这复现了旧路径的内存爆炸问题。当前 fixed `eval-desc` 与 `embed` 都能完成同一压力样本，且 GPU 峰值相同。

## 文件大小说明

HDF5 `embed` 产物并不等价于 descriptor-only `.npy`：

- `eval-desc fp32` 只保存 descriptor。
- `embed fp32` 同时保存 `descriptor`、`atomic_feature`、`structural_feature`、`atom_types` 等数据集。

因此本轮测试中 `embed_fp32.hdf5` 文件比 descriptor-only `.npy` 更大是正常现象。它的优势不是 descriptor-only 文件体积更小，而是一个压缩 HDF5 文件同时承载 DP-EVA feature、2-DIRECT 和 LLPR last-layer 所需数据，避免额外多次推理与格式拼装。

## 最终判断

- 对于 **数值正确性**：通过。最新环境中 `embed` 与 `eval-desc` descriptor 在 `test/` water 数据集上达到 `1e-7` 量级一致；DPA4-Plus 压力样本达到 `1e-5` 以内一致。
- 对于 **资源效率**：相对旧环境 legacy `eval-desc`，通过，且压力样本复现旧 OOM；相对最新 fixed `eval-desc`，不应宣称 `embed` 明显更省显存，二者在本测试中基本相同。
- 对于 **DP-EVA 适配路线**：继续采用 `embed` HDF5 是合理的，因为它覆盖 descriptor、2-DIRECT atomic descriptor 读取和 LLPR `atomic_feature`，并保持与最新 DeepMD 推荐接口一致。
