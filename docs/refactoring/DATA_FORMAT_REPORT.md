### 研判报告：关于 DEFAULT_DATA_FORMAT 及相关变量的必要性分析 1. 变量适用范围确认
- 常量 : DEFAULT_DATA_FORMAT ( "auto" )
- 适用范围 : 该常量仅用于定义 FeatureConfig 类中 data_format (别名 format ) 字段的默认值。
- 配置类 : FeatureConfig (位于 src/dpeva/config.py )
- 受影响工作流 : FeatureWorkflow (及其调用的 DescriptorGenerator ) 2. 深入分析与综合研判
在 load_systems 全面接管数据加载且默认采用 auto 模式的背景下，我们需要评估显式配置 data_format 的必要性。

A. 现状分析

1. load_systems 的能力 :
   
   - 该函数通过 fmt="auto" 实现了对 deepmd/npy/mixed 和 deepmd/npy 两种主流格式的智能识别与回退机制。
   - 在 InferenceWorkflow 和 CollectionWorkflow 中， load_systems 已经被直接调用且大多未暴露格式配置给用户（或者硬编码了 auto ），运行良好。
2. FeatureWorkflow 的特殊性 :
   
   - 目前只有 FeatureWorkflow (通过 FeatureConfig ) 向用户暴露了 format (即 data_format ) 参数。
   - 在 Python 模式下，该参数被传递给 DescriptorGenerator ，最终传给 load_systems 。
   - 在 CLI 模式下，该参数 完全未被使用 （CLI 命令 dp eval-desc 自身负责格式识别，不接受此参数）。
B. 存在的理由 (Pros)

1. 强制指定 (Override) : 在极少数情况下，如果自动检测失效（例如目录结构异常导致误判），用户可能需要显式指定 deepmd/npy 来强制使用特定解析器。
2. 兼容性 (Compatibility) : format 参数在旧版本配置文件中广泛存在。保留该字段并将其默认值设为 auto 可以最大程度保障向后兼容性，避免旧的配置文件报错。
3. 扩展性 : 未来如果支持更多非标准格式（如 vasp/poscar 等），显式的格式参数是必要的接口。
C. 移除的风险 (Cons)

1. 破坏配置兼容性 : 如果直接从 FeatureConfig 中移除该字段，所有包含 "format": "..." 的旧配置文件在加载时都会因 Extra fields not permitted (若配置了 extra='forbid' ) 或被忽略而产生行为变更。
2. 灵活性降低 : 丧失了在自动检测失败时的手动干预能力。
D. 综合研判结论

保留该常量及配置入口，但将其定位为“高级/调试选项”。

- 默认行为 : 绝大多数用户应使用默认值 "auto" ，无需感知此参数。
- 配置入口 : 应保留 FeatureConfig 中的 data_format 字段，以维持向后兼容性并提供逃生舱（Escape Hatch）。
- 常量定义 : DEFAULT_DATA_FORMAT = "auto" 是合理的，它准确描述了系统的默认行为（自动探测）。 3. 建议措施
1. 保持现状 : 不需要删除 DEFAULT_DATA_FORMAT 或 FeatureConfig 中的相关字段。
2. 文档引导 : 在文档中弱化该参数的存在感，说明仅在自动检测失败时才需设置。
3. 代码清理 : 确认所有调用 load_systems 的地方（如 InferenceWorkflow , CollectionWorkflow ）是否需要统一暴露此参数。目前来看， Inference 和 Collect 均未暴露，这导致了配置接口的不一致性（Feature 暴露了，其他没暴露）。
   - 长远建议 : 考虑在所有涉及数据加载的 Config 中统一添加 data_format: str = "auto" 字段，或统一移除/隐藏，以保持接口一致性。但鉴于 auto 模式的高鲁棒性，目前“默认隐藏，Feature 保留兼容”的策略是务实且低风险的。
最终结论 : DEFAULT_DATA_FORMAT 需要存在 ，作为配置系统的默认值基准；相关用户定义入口（ FeatureConfig.data_format ）也 需要存在 ，主要用于兼容性和特殊场景下的强制指定。