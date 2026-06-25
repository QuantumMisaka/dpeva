---
title: DP-EVA Feature Embed/HDF5 Adaptation Plan
status: archived
audience: Developers
last-updated: 2026-06-25
owner: Maintainers
---

# DP-EVA feature Embed/HDF5 适配与 SAI DPA4 验证计划

  ## Confirmed Facts

  - DeepMD 当前 dp embed 输出 HDF5，并且在本地 test/deepmd-kit 实现中对 dataset 使用 gzip level 9 + byte-shuffle 压缩；同一类数组通常会比未压缩 .npy 更紧凑。
  - 不能绝对承诺整个 HDF5 一定比 .npy 目录小：dp embed 同时保存 descriptor、atomic_feature、structural_feature、atom_types，如果拿完整 HDF5 和单一 descriptor .npy 比，大小取决于保留的数据范围。
  - DeepMD 新接口已经把旧能力收敛到 embed：
      - eval_descriptor 对应 HDF5 descriptor。
      - eval_fitting_last_layer 对应 HDF5 atomic_feature。
      - structural_feature 是 atomic_feature 的按原子求和结果。

  - 因此 DP-EVA 应把 HDF5 作为 embed 路线的一等存储格式，而不是默认转回 .npy。

  ## Summary

  - 直接在当前 develop 工作区开发，不切到 main。
  - dpeva feature 新增 embed 行为：feature_exporter="embed" 时调用 dp --pt embed，保留 HDF5 主输出。
  - 旧行为兼容：默认仍为 feature_exporter="eval_desc"，继续生成 .npy。
  - DP-EVA IO 层新增 HDF5 读取能力，使 collect、2-DIRECT atomic feature 读取、LLPR last-layer 特征读取都能直接消费 embedding.hdf5。

  ## Public Interface Changes

  - 修改 FeatureConfig：
      - feature_exporter: Literal["eval_desc", "embed"] = "eval_desc"
      - embedding_dtype: Literal["fp32", "fp64", "native"] = "fp32"

  - 行为约束：
      - feature_exporter="eval_desc" 只支持 feature_kind="descriptor"，保持旧 CLI 约束。
      - feature_exporter="embed" 支持 feature_kind="descriptor" 和 feature_kind="fitting_last_layer"。
      - embed 输出始终保留完整 HDF5；output_mode 不裁剪 HDF5 内容，下游读取时决定使用原子特征、均值池化或 LLPR 求和。

  - 修改 CollectionConfig.desc_dir 语义：既可指向旧 .npy descriptor 目录，也可指向 HDF5 文件或包含 embedding.hdf5 的目录。

  ## Implementation Changes

  - FeatureExecutionManager.submit_cli_job() 增加 embed 分支：
      - 生成 dp --pt embed -s <data_path> -m <model_path> -o <savedir>/embedding.hdf5 --dtype <embedding_dtype>。
      - Slurm/local 日志名改为能区分 eval_desc 与 embed。
      - 不默认物化 .npy，HDF5 是 embed 路线主产物。
      - feature_kind="descriptor" 返回 descriptor。
      - feature_kind="fitting_last_layer" 返回 atomic_feature。
      - 对旧模型或旧 DeepMD 保留 eval_descriptor / eval_fitting_last_layer fallback。

  - CollectionIOManager 增加统一 feature store 读取：
      - .npy 路径保持现状。
      - HDF5 读取 descriptor 作为普通 desc_dir 默认数据源。
      - HDF5 读取 atomic_feature 作为 LLPR feature 数据源。
      - load_atomic_features() 对 HDF5 直接按 frame 读取原子级 descriptor。
      - 系统名优先由 HDF5 group 的 system attribute 相对 testdata_dir 归一化；失败时回退 group name。

  - LLPR 读取逻辑改造：
      - _load_llpr_feature_sums() 支持 HDF5。
      - 默认保持 DP-EVA 现有语义：按配置选择 sum 或 mean 聚合 atomic_feature。
      - DeepMD 原生 structural_feature 作为可选优化/校验对象，不默认替换现有 DP-EVA 聚合语义。

  - 项目依赖与文档：
      - 确认并显式记录 h5py 依赖。
      - 更新配置文档和 feature recipe，说明 HDF5 输出、dataset 映射、兼容 .npy 的读取规则。

  ## Test Plan

  - 单元测试：
      - 默认配置仍生成 dp --pt eval-desc。
      - feature_exporter="embed" 对 descriptor 和 fitting-last-layer 都生成 dp --pt embed。
      - fake HDF5 测试 descriptor、atomic_feature、structural_feature、atom_types 读取。
      - load_descriptors() 同时覆盖 .npy 目录、HDF5 文件、包含 embedding.hdf5 的目录。
      - load_atomic_features() 可从 HDF5 按 candidate frame 读取原子级 descriptor。
      - LLPR feature sums 可从 HDF5 atomic_feature 读取，并保持 sum/mean 归一化语义。

  - 环境验证：
      - 克隆 dpeva-dpa4-test 为 dpeva-dpa4-embed-test。
      - 安装本地 test/deepmd-kit 和本地 DP-EVA editable。
      - 验证 dp --pt embed --help、DeepPot.eval_embedding、h5py。

  - SAI DPA4 实战：
      - 在 4V100 Slurm 计算节点运行，--nodes=1 --ntasks=1 --gpus-per-node=1，不设置 --mem/--cpus-per-task。
      - 先对旧 OOM 样本 sampled_dpdata/216 跑 dp embed，确认不再 OOM 且 HDF5 datasets 存在。
      - 再跑 DP-EVA feature_exporter="embed"，产出 embedding.hdf5。
      - 用 dpeva collect 直接读取 HDF5，确认普通 descriptor 路线、2-DIRECT atomic feature 路线、LLPR last-layer 路线至少各有 smoke test。

  ## Assumptions

  - 当前 develop 是本次允许直接开发的分支。
  - test/deepmd-kit 已包含 issue #5507 对应修复和 dp embed 能力。
  - HDF5 是 embed 路线主格式；.npy 是历史兼容格式，不再作为 embed 的默认中间产物。
  - 第一阶段不改变主动学习算法，只替换/扩展 feature 存储与读取后端。
