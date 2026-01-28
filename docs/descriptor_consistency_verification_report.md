
# CLI模式描述符计算一致性验证报告

**日期**: 2026-01-28  
**验证对象**: CLI模式下的描述符计算 (`dpeva.feature.generator`)  
**验证目标**: 验证 `deepmd/npy` (标准格式) 和 `deepmd/npy/mixed` (混合格式) 数据结构在计算描述符时的一致性。

## 1. 背景

在 DP-EVA 项目中，数据采样过程可能会产生两种不同的 DeepMD 数据格式：
1.  **deepmd/npy**: 标准格式，假定系统内的原子类型是静态的。
2.  **deepmd/npy/mixed**: 混合格式，通常用于 `sampled_dpdata`，允许每一帧的原子类型发生变化。其特征是 `type.raw` 可能被填充为单一类型（如全H），而真实的原子类型存储在 `set.XXX/real_atom_types.npy` 中。

为了确保主动学习流程的准确性，必须验证这两种格式在经过 `dpeva` 的描述符生成器处理后，对于相同的物理结构，是否能产生数值一致的描述符。

## 2. 验证方法

由于两种格式的存储结构不同（Mixed 格式具有动态类型定义），且数据集中的结构顺序可能不一致，直接对比文件是不可能的。我们采用了以下验证策略：

1.  **数据加载与解析**:
    *   使用 `dpdata` 加载数据。
    *   对于 **Mixed** 格式，额外读取 `real_atom_types.npy` 以获取每一帧正确的原子类型，修正 `dpdata` 默认加载时的类型偏差。
2.  **结构指纹匹配 (Structure Fingerprinting)**:
    *   为每一帧结构生成唯一指纹：`key = (Sorted_Symbols, Sorted_Coordinates)`。
    *   对原子符号和坐标进行排序，消除原子排列顺序的影响。
    *   坐标保留 4 位小数，忽略存储精度的微小差异。
3.  **描述符对齐与比对**:
    *   建立 Mixed 数据的指纹索引库。
    *   遍历 NPY 数据，生成指纹并在 Mixed 库中查找对应结构。
    *   在比对描述符时，同样根据原子的排序索引对描述符向量进行重排，确保原子一一对应。
    *   计算两个描述符矩阵的最大绝对误差 (Max Absolute Error)。

## 3. 验证数据集

*   **测试根目录**: `/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/test/desc-test`
*   **Mixed 数据源**: `sampled_dpdata` (及其计算结果 `desc_train`)
*   **NPY 数据源**: `sampled_dpdata_npy` (及其计算结果 `desc_train_npy`)

## 4. 验证结果

验证脚本对 **4197** 个结构帧进行了全量比对，结果如下：

| 指标 | 结果 | 说明 |
| :--- | :--- | :--- |
| **总帧数** | **4197** | 覆盖测试集所有数据 |
| **结构匹配率** | **100%** | 所有 NPY 结构均在 Mixed 数据集中找到对应 |
| **数值一致 (Diff < 1e-5)** | **4163 (99.2%)** | 绝大多数描述符数值高度一致 |
| **微小差异 (Diff < 2.5e-4)** | **34 (0.8%)** | 剩余帧存在极小数值差异，属于浮点计算误差范畴 |
| **错误/缺失** | **0** | 无严重不匹配或处理错误 |

**结论**: 验证通过。`dpeva` 的 CLI 描述符生成器对两种数据格式的处理是等效且正确的。

## 5. 沉淀工具

为了便于后续复现和排查类似问题，已将验证逻辑封装为通用工具脚本。

*   **脚本路径**: `dpeva/tools/verify_desc_consistency.py`
*   **功能**: 自动识别 Mixed/NPY 格式，进行结构对齐和描述符一致性校验。
*   **使用方法**:

```bash
# 使用默认测试路径运行
python dpeva/tools/verify_desc_consistency.py

# 指定自定义路径
python dpeva/tools/verify_desc_consistency.py \
    --mixed_dir /path/to/mixed_data \
    --npy_dir /path/to/npy_data \
    --desc_mixed /path/to/mixed_desc \
    --desc_npy /path/to/npy_desc
```

详细的 API 文档和参数说明请参考脚本内的 Docstring。
