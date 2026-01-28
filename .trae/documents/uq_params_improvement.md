# UQ 参数配置改进说明

## 概述
本次改进主要增强了 `CollectionWorkflow` 中关于不确定度 (UQ) 筛选参数的灵活性。现在支持对 `uq_qbc` 和 `uq_rnd_rescaled` 分别指定 `ratio` 和 `width`，同时保持对全局默认值和旧版配置方式的兼容性。

## 配置优先级规则

系统按照以下优先级解析参数：
1.  **特定参数 (Specific)**: 如 `uq_qbc_trust_ratio`
2.  **全局参数 (Global)**: 如 `uq_trust_ratio`
3.  **默认值 (Default)**: `ratio=0.33`, `width=0.25`

### 1. 全局配置（推荐）
如果仅指定全局参数，两个 UQ 方法将共享相同的设置。

```json
{
    "uq_trust_mode": "auto",
    "uq_trust_ratio": 0.33,
    "uq_trust_width": 0.25
}
```

### 2. 分别配置（高级）
可以为 QbC 和 RND 分别指定不同的筛选比例和宽度。

```json
{
    "uq_trust_mode": "auto",
    "uq_trust_ratio": 0.33,      // 全局默认
    "uq_trust_width": 0.25,      // 全局默认
    
    // QbC 特有配置 (覆盖全局)
    "uq_qbc_trust_ratio": 0.40,
    "uq_qbc_trust_width": 0.20,
    
    // RND 特有配置 (覆盖全局)
    "uq_rnd_rescaled_trust_ratio": 0.50
    // RND width 未指定，将沿用全局 width=0.25
}
```

### 3. Manual 模式下的 Lo/Hi/Width 逻辑
在 `manual` 模式下，可以直接指定 `lo` 和 `hi`。系统会自动进行一致性校验。

*   **规则 1**: `lo + width` 必须等于 `hi`。
*   **规则 2**: 如果同时指定了 `lo` 和 `hi`，系统会自动推导 `width`。
*   **规则 3**: 如果同时指定了 `lo`, `hi` **以及** `width`，且三者不满足 `lo + width = hi`，系统将报错退出。
*   **规则 4**: 如果只指定 `lo` 和 `width`，系统自动计算 `hi`。
*   **规则 5**: 如果只指定 `lo` 和 `hi`，系统自动更新内部的 `width`。

**冲突示例（将报错）：**
```json
{
    "uq_qbc_trust_lo": 0.1,
    "uq_qbc_trust_hi": 0.4,     // 隐含 width = 0.3
    "uq_qbc_trust_width": 0.25  // 冲突！抛出 ValueError
}
```

## 参数列表

| 参数名 | 说明 | 默认值 |
| :--- | :--- | :--- |
| `uq_trust_mode` | 模式选择: `auto` 或 `manual` | `manual` |
| `uq_trust_ratio` | 全局比例 (Auto模式) | 0.33 |
| `uq_trust_width` | 全局宽度 | 0.25 |
| `uq_qbc_trust_ratio` | QbC 专用比例 | 继承全局 |
| `uq_qbc_trust_width` | QbC 专用宽度 | 继承全局 |
| `uq_qbc_trust_lo` | QbC 下限 (Manual) | 0.12 |
| `uq_qbc_trust_hi` | QbC 上限 (Manual) | 计算得出 |
| `uq_rnd_rescaled_trust_ratio` | RND 专用比例 | 继承全局 |
| `uq_rnd_rescaled_trust_width` | RND 专用宽度 | 继承全局 |
| `uq_rnd_rescaled_trust_lo` | RND 下限 (Manual) | 继承 QbC lo |
| `uq_rnd_rescaled_trust_hi` | RND 上限 (Manual) | 计算得出 |
