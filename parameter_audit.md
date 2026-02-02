# DP-EVA User Configurable Parameters Audit

## Summary
- **Total Parameters**: 273
- **Parameters with Defaults**: 147
- **Implicit (config.get) Parameters**: 60
- **Missing Docstrings (Explicit)**: 116

## Implicit Dependency Risks (Top 10)
- `data_path` in `InferenceWorkflow.__init__` (Default: `None`)
- `output_basedir` in `InferenceWorkflow.__init__` (Default: `'./'`)
- `task_name` in `InferenceWorkflow.__init__` (Default: `'test'`)
- `head` in `InferenceWorkflow.__init__` (Default: `'Hybrid_Perovskite'`)
- `submission` in `InferenceWorkflow.__init__` (Default: `{}`)
- `omp_threads` in `InferenceWorkflow.__init__` (Default: `2`)
- `ref_energies` in `InferenceWorkflow.analyze_results` (Default: `None`)
- `project` in `CollectionWorkflow.__init__` (Default: `'stage9-2'`)
- `uq_select_scheme` in `CollectionWorkflow.__init__` (Default: `'tangent_lo'`)
- `backend` in `CollectionWorkflow.__init__` (Default: `'local'`)

## Detailed Parameter List

| Context | Parameter | Type | Default | Required | Description | Source |
|---|---|---|---|---|---|---|
| [ParallelTrainer.__init__](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/training/trainer.py) | `base_config_path` | Any | `None` | True | ... | explicit |
| [ParallelTrainer.__init__](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/training/trainer.py) | `work_dir` | Any | `None` | True | ... | explicit |
| [ParallelTrainer.__init__](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/training/trainer.py) | `num_models` | Any | `4` | False | ... | explicit |
| [ParallelTrainer.__init__](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/training/trainer.py) | `backend` | Any | `'local'` | False | ... | explicit |
| [ParallelTrainer.__init__](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/training/trainer.py) | `template_path` | Any | `None` | False | ... | explicit |
| [ParallelTrainer.__init__](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/training/trainer.py) | `slurm_config` | Any | `None` | False | ... | explicit |
| [ParallelTrainer.__init__](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/training/trainer.py) | `training_data_path` | Any | `None` | False | ... | explicit |
| [ParallelTrainer.prepare_configs](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/training/trainer.py) | `seeds` | Any | `None` | True | ... | explicit |
| [ParallelTrainer.prepare_configs](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/training/trainer.py) | `training_seeds` | Any | `None` | True | ... | explicit |
| [ParallelTrainer.prepare_configs](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/training/trainer.py) | `finetune_heads` | Any | `None` | True | ... | explicit |
| [ParallelTrainer.setup_workdirs](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/training/trainer.py) | `base_models` | Any | `None` | True | ... | explicit |
| [ParallelTrainer.setup_workdirs](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/training/trainer.py) | `omp_threads` | Any | `12` | False | ... | explicit |
| [ParallelTrainer.train](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/training/trainer.py) | `blocking` | Any | `True` | False | If True, wait for all tasks to complete (Only vali... | explicit |
| [DescriptorGenerator.__init__](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/feature/generator.py) | `model_path` | Any | `None` | True | Path to the frozen DeepMD model file.... | explicit |
| [DescriptorGenerator.__init__](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/feature/generator.py) | `head` | Any | `'OC20M'` | False | Head type for multi-head models (default: "OC20M")... | explicit |
| [DescriptorGenerator.__init__](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/feature/generator.py) | `batch_size` | Any | `1000` | False | Batch size for inference (default: 1000). Ignored ... | explicit |
| [DescriptorGenerator.__init__](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/feature/generator.py) | `omp_threads` | Any | `24` | False | Number of OMP threads (default: 24).... | explicit |
| [DescriptorGenerator.__init__](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/feature/generator.py) | `mode` | Any | `'cli'` | False | 'cli' or 'python'. 'cli' uses `dp eval-desc`.... | explicit |
| [DescriptorGenerator.__init__](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/feature/generator.py) | `backend` | Any | `'local'` | False | 'local' or 'slurm'. Only used in 'cli' mode.... | explicit |
| [DescriptorGenerator.__init__](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/feature/generator.py) | `slurm_config` | Any | `None` | False | Configuration for Slurm submission.... | explicit |
| [DescriptorGenerator.__init__](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/feature/generator.py) | `env_setup` | Any | `None` | False | Environment setup script for job execution.... | explicit |
| [DescriptorGenerator.run_cli_generation](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/feature/generator.py) | `data_path` | Any | `None` | True | Path to the dataset.... | explicit |
| [DescriptorGenerator.run_cli_generation](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/feature/generator.py) | `output_dir` | Any | `None` | True | Directory to save descriptors.... | explicit |
| [DescriptorGenerator.run_cli_generation](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/feature/generator.py) | `blocking` | Any | `True` | False | Whether to wait for the job to complete (local mod... | explicit |
| [DescriptorGenerator._descriptor_from_model](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/feature/generator.py) | `sys` | dpdata.System | `None` | True | ... | explicit |
| [DescriptorGenerator._descriptor_from_model](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/feature/generator.py) | `nopbc` | Any | `False` | False | ... | explicit |
| [DescriptorGenerator._get_desc_by_batch](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/feature/generator.py) | `sys` | dpdata.System | `None` | True | ... | explicit |
| [DescriptorGenerator._get_desc_by_batch](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/feature/generator.py) | `nopbc` | Any | `False` | False | ... | explicit |
| [DescriptorGenerator.run_python_generation](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/feature/generator.py) | `data_path` | Any | `None` | True | ... | explicit |
| [DescriptorGenerator.run_python_generation](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/feature/generator.py) | `output_dir` | Any | `None` | True | ... | explicit |
| [DescriptorGenerator.run_python_generation](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/feature/generator.py) | `data_format` | Any | `'deepmd/npy'` | False | ... | explicit |
| [DescriptorGenerator.run_python_generation](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/feature/generator.py) | `output_mode` | Any | `'atomic'` | False | ... | explicit |
| [DescriptorGenerator.compute_descriptors_python](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/feature/generator.py) | `data_path` | Any | `None` | True | Path to the dataset (system directory).... | explicit |
| [DescriptorGenerator.compute_descriptors_python](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/feature/generator.py) | `data_format` | Any | `'deepmd/npy'` | False | Format of the dataset (default: "deepmd/npy").... | explicit |
| [DescriptorGenerator.compute_descriptors_python](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/feature/generator.py) | `output_mode` | Any | `'atomic'` | False | "atomic" (per atom) or "structural" (per frame, me... | explicit |
| [DPTestResultParser.__init__](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/io/dataproc.py) | `result_dir` | str | `None` | True | Directory containing the test results (e.g. *.e.ou... | explicit |
| [DPTestResultParser.__init__](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/io/dataproc.py) | `head` | str | `'results'` | False | The head name used in dp test output files (e.g. "... | explicit |
| [DPTestResultParser.__init__](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/io/dataproc.py) | `type_map` | List[str] | `None` | False | List of atom types. If None, defaults to ["H", "C"... | explicit |
| [DPTestResultParser._get_dataname_info](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/io/dataproc.py) | `filename` | str | `None` | True | ... | explicit |
| [DPTestResultParser._get_natom_from_name](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/io/dataproc.py) | `dataname` | str | `None` | True | ... | explicit |
| [load_systems](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/io/dataset.py) | `data_dir` | str | `None` | True | Path to the data directory.... | explicit |
| [load_systems](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/io/dataset.py) | `fmt` | str | `'auto'` | False | Format of the data. If "auto" (default), attempts ... | explicit |
| [load_systems](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/io/dataset.py) | `target_systems` | Optional[List[str]] | `None` | False | List of specific system names to load. If None, tr... | explicit |
| [_fix_duplicate_atom_names](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/io/dataset.py) | `sys` | dpdata.System | `None` | True | ... | explicit |
| [_fix_duplicate_atom_names](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/io/dataset.py) | `sys_name` | str | `'Unknown'` | False | ... | explicit |
| [set_visual_style](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/utils/visual_style.py) | `font_size` | int | `12` | False | Base font size.... | explicit |
| [set_visual_style](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/utils/visual_style.py) | `context` | str | `'paper'` | False | Seaborn context ('paper', 'notebook', 'talk', 'pos... | explicit |
| [set_visual_style](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/utils/visual_style.py) | `style` | str | `'whitegrid'` | False | Seaborn style ('whitegrid', 'darkgrid', 'white', '... | explicit |
| [InferenceWorkflow.__init__](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/workflows/infer.py) | `config` | Any | `None` | True | ... | explicit |
| [InferenceWorkflow.__init__](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/workflows/infer.py) | `data_path` | Unknown | `None` | False | ... | implicit |
| [InferenceWorkflow.__init__](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/workflows/infer.py) | `output_basedir` | Unknown | `'./'` | False | ... | implicit |
| [InferenceWorkflow.__init__](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/workflows/infer.py) | `task_name` | Unknown | `'test'` | False | ... | implicit |
| [InferenceWorkflow.__init__](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/workflows/infer.py) | `head` | Unknown | `'Hybrid_Perovskite'` | False | ... | implicit |
| [InferenceWorkflow.__init__](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/workflows/infer.py) | `submission` | Unknown | `{}` | False | ... | implicit |
| [InferenceWorkflow.__init__](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/workflows/infer.py) | `omp_threads` | Unknown | `2` | False | ... | implicit |
| [InferenceWorkflow.analyze_results](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/workflows/infer.py) | `output_dir_suffix` | Any | `'analysis'` | False | ... | explicit |
| [InferenceWorkflow.analyze_results](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/workflows/infer.py) | `ref_energies` | Unknown | `None` | False | ... | implicit |
| [CollectionWorkflow.__init__](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/workflows/collect.py) | `config` | Any | `None` | True | Configuration dictionary containing:... | explicit |
| [CollectionWorkflow.__init__](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/workflows/collect.py) | `config_path` | Any | `None` | False | Path to the configuration file. Used for optimized... | explicit |
| [CollectionWorkflow.__init__](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/workflows/collect.py) | `project` | Unknown | `'stage9-2'` | False | ... | implicit |
| [CollectionWorkflow.__init__](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/workflows/collect.py) | `uq_select_scheme` | Unknown | `'tangent_lo'` | False | ... | implicit |
| [CollectionWorkflow.__init__](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/workflows/collect.py) | `backend` | Unknown | `'local'` | False | ... | implicit |
| [CollectionWorkflow.__init__](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/workflows/collect.py) | `slurm_config` | Unknown | `{}` | False | ... | implicit |
| [CollectionWorkflow.__init__](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/workflows/collect.py) | `testing_dir` | Unknown | `'test-val-npy'` | False | ... | implicit |
| [CollectionWorkflow.__init__](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/workflows/collect.py) | `testing_head` | Unknown | `'results'` | False | ... | implicit |
| [CollectionWorkflow.__init__](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/workflows/collect.py) | `desc_dir` | Unknown | `None` | False | ... | implicit |
| [CollectionWorkflow.__init__](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/workflows/collect.py) | `desc_filename` | Unknown | `'desc.npy'` | False | ... | implicit |
| [CollectionWorkflow.__init__](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/workflows/collect.py) | `testdata_dir` | Unknown | `None` | False | ... | implicit |
| [CollectionWorkflow.__init__](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/workflows/collect.py) | `training_data_dir` | Unknown | `None` | False | ... | implicit |
| [CollectionWorkflow.__init__](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/workflows/collect.py) | `training_desc_dir` | Unknown | `None` | False | ... | implicit |
| [CollectionWorkflow.__init__](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/workflows/collect.py) | `root_savedir` | Unknown | `'dpeva_uq_post'` | False | ... | implicit |
| [CollectionWorkflow.__init__](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/workflows/collect.py) | `uq_trust_mode` | Unknown | `None` | False | ... | implicit |
| [CollectionWorkflow.__init__](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/workflows/collect.py) | `uq_trust_ratio` | Unknown | `0.33` | False | ... | implicit |
| [CollectionWorkflow.__init__](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/workflows/collect.py) | `uq_trust_width` | Unknown | `0.25` | False | ... | implicit |
| [CollectionWorkflow.__init__](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/workflows/collect.py) | `uq_qbc_trust_ratio` | Unknown | `self.global_trust_ratio` | False | ... | implicit |
| [CollectionWorkflow.__init__](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/workflows/collect.py) | `uq_qbc_trust_width` | Unknown | `self.global_trust_width` | False | ... | implicit |
| [CollectionWorkflow.__init__](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/workflows/collect.py) | `uq_qbc_trust_lo` | Unknown | `None` | False | ... | implicit |
| [CollectionWorkflow.__init__](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/workflows/collect.py) | `uq_qbc_trust_hi` | Unknown | `None` | False | ... | implicit |
| [CollectionWorkflow.__init__](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/workflows/collect.py) | `uq_rnd_rescaled_trust_ratio` | Unknown | `self.global_trust_ratio` | False | ... | implicit |
| [CollectionWorkflow.__init__](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/workflows/collect.py) | `uq_rnd_rescaled_trust_width` | Unknown | `self.global_trust_width` | False | ... | implicit |
| [CollectionWorkflow.__init__](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/workflows/collect.py) | `uq_rnd_rescaled_trust_lo` | Unknown | `None` | False | ... | implicit |
| [CollectionWorkflow.__init__](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/workflows/collect.py) | `uq_rnd_rescaled_trust_hi` | Unknown | `None` | False | ... | implicit |
| [CollectionWorkflow.__init__](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/workflows/collect.py) | `uq_auto_bounds` | Unknown | `{}` | False | ... | implicit |
| [CollectionWorkflow.__init__](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/workflows/collect.py) | `num_selection` | Unknown | `100` | False | ... | implicit |
| [CollectionWorkflow.__init__](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/workflows/collect.py) | `direct_k` | Unknown | `1` | False | ... | implicit |
| [CollectionWorkflow.__init__](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/workflows/collect.py) | `direct_thr_init` | Unknown | `0.5` | False | ... | implicit |
| [CollectionWorkflow._validate_manual_params](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/workflows/collect.py) | `params` | Any | `None` | True | ... | explicit |
| [CollectionWorkflow._validate_manual_params](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/workflows/collect.py) | `name` | Any | `None` | True | ... | explicit |
| [CollectionWorkflow._clamp_trust_lo](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/workflows/collect.py) | `value` | Any | `None` | True | The auto-calculated value.... | explicit |
| [CollectionWorkflow._clamp_trust_lo](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/workflows/collect.py) | `bounds` | Any | `None` | True | Dictionary containing 'lo_min' and/or 'lo_max'.... | explicit |
| [CollectionWorkflow._clamp_trust_lo](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/workflows/collect.py) | `name` | Any | `'UQ'` | False | Name of the parameter for logging.... | explicit |
| [CollectionWorkflow._validate_config](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/workflows/collect.py) | `project` | Unknown | `'.'` | False | ... | implicit |
| [CollectionWorkflow._validate_config](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/workflows/collect.py) | `project` | Unknown | `None` | False | ... | implicit |
| [CollectionWorkflow._validate_config](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/workflows/collect.py) | `project` | Unknown | `None` | False | ... | implicit |
| [CollectionWorkflow._validate_config](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/workflows/collect.py) | `unknown` | Unknown | `None` | False | ... | implicit |
| [CollectionWorkflow._load_descriptors](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/workflows/collect.py) | `desc_dir` | Any | `None` | True | Path to the descriptor directory.... | explicit |
| [CollectionWorkflow._load_descriptors](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/workflows/collect.py) | `desc_filename` | Any | `'desc.npy'` | False | Name of the descriptor file (default: "desc.npy") ... | explicit |
| [CollectionWorkflow._load_descriptors](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/workflows/collect.py) | `label` | Any | `'descriptors'` | False | Label for logging purposes.... | explicit |
| [CollectionWorkflow._load_descriptors](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/workflows/collect.py) | `target_names` | Any | `None` | False | List of system names (without index) to load speci... | explicit |
| [CollectionWorkflow._load_descriptors](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/workflows/collect.py) | `expected_frames` | Any | `None` | False | Optional dict {sys_name: n_frames} to enforce cons... | explicit |
| [CollectionWorkflow._count_frames_in_data](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/workflows/collect.py) | `data_dir` | Any | `None` | True | ... | explicit |
| [CollectionWorkflow._count_frames_in_data](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/workflows/collect.py) | `fmt` | Any | `'auto'` | False | ... | explicit |
| [CollectionWorkflow._prepare_features_for_direct](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/workflows/collect.py) | `df_candidate` | Any | `None` | True | ... | explicit |
| [CollectionWorkflow._prepare_features_for_direct](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/workflows/collect.py) | `df_desc` | Any | `None` | True | ... | explicit |
| [CollectionWorkflow.run](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/workflows/collect.py) | `fig_dpi` | Unknown | `150` | False | ... | implicit |
| [TrainingWorkflow.__init__](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/workflows/train.py) | `config` | Any | `None` | True | ... | explicit |
| [TrainingWorkflow.__init__](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/workflows/train.py) | `work_dir` | Unknown | `os.getcwd()` | False | ... | implicit |
| [TrainingWorkflow.__init__](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/workflows/train.py) | `input_json_path` | Unknown | `'input.json'` | False | ... | implicit |
| [TrainingWorkflow.__init__](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/workflows/train.py) | `num_models` | Unknown | `4` | False | ... | implicit |
| [TrainingWorkflow.__init__](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/workflows/train.py) | `mode` | Unknown | `'cont'` | False | ... | implicit |
| [TrainingWorkflow.__init__](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/workflows/train.py) | `base_model_path` | Unknown | `None` | False | ... | implicit |
| [TrainingWorkflow.__init__](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/workflows/train.py) | `omp_threads` | Unknown | `12` | False | ... | implicit |
| [TrainingWorkflow.__init__](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/workflows/train.py) | `backend` | Unknown | `'local'` | False | ... | implicit |
| [TrainingWorkflow.__init__](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/workflows/train.py) | `slurm_config` | Unknown | `{}` | False | ... | implicit |
| [TrainingWorkflow.__init__](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/workflows/train.py) | `template_path` | Unknown | `None` | False | ... | implicit |
| [TrainingWorkflow.__init__](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/workflows/train.py) | `finetune_head_name` | Unknown | `'Hybrid_Perovskite'` | False | ... | implicit |
| [TrainingWorkflow.__init__](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/workflows/train.py) | `training_data_path` | Unknown | `None` | False | ... | implicit |
| [FeatureWorkflow.__init__](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/workflows/feature.py) | `config` | Any | `None` | True | ... | explicit |
| [FeatureWorkflow.__init__](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/workflows/feature.py) | `data_path` | Unknown | `None` | False | ... | implicit |
| [FeatureWorkflow.__init__](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/workflows/feature.py) | `modelpath` | Unknown | `None` | False | ... | implicit |
| [FeatureWorkflow.__init__](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/workflows/feature.py) | `format` | Unknown | `'deepmd/npy'` | False | ... | implicit |
| [FeatureWorkflow.__init__](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/workflows/feature.py) | `output_mode` | Unknown | `'atomic'` | False | ... | implicit |
| [FeatureWorkflow.__init__](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/workflows/feature.py) | `savedir` | Unknown | `f"desc-{os.path.basename(self.modelpath).split('.')[0]}-{os.path.basename(self.data_path)}"` | False | ... | implicit |
| [FeatureWorkflow.__init__](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/workflows/feature.py) | `head` | Unknown | `'OC20M'` | False | ... | implicit |
| [FeatureWorkflow.__init__](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/workflows/feature.py) | `batch_size` | Unknown | `1000` | False | ... | implicit |
| [FeatureWorkflow.__init__](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/workflows/feature.py) | `omp_threads` | Unknown | `24` | False | ... | implicit |
| [FeatureWorkflow.__init__](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/workflows/feature.py) | `mode` | Unknown | `'cli'` | False | ... | implicit |
| [FeatureWorkflow.__init__](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/workflows/feature.py) | `submission` | Unknown | `{}` | False | ... | implicit |
| [InferenceVisualizer.__init__](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/inference/visualizer.py) | `output_dir` | str | `None` | True | ... | explicit |
| [InferenceVisualizer.__init__](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/inference/visualizer.py) | `dpi` | int | `150` | False | ... | explicit |
| [InferenceVisualizer.plot_parity](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/inference/visualizer.py) | `y_true` | np.ndarray | `None` | True | ... | explicit |
| [InferenceVisualizer.plot_parity](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/inference/visualizer.py) | `y_pred` | np.ndarray | `None` | True | ... | explicit |
| [InferenceVisualizer.plot_parity](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/inference/visualizer.py) | `label` | str | `None` | True | ... | explicit |
| [InferenceVisualizer.plot_parity](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/inference/visualizer.py) | `unit` | str | `None` | True | ... | explicit |
| [InferenceVisualizer.plot_parity](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/inference/visualizer.py) | `title` | str | `None` | False | ... | explicit |
| [InferenceVisualizer.plot_distribution](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/inference/visualizer.py) | `data` | np.ndarray | `None` | True | ... | explicit |
| [InferenceVisualizer.plot_distribution](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/inference/visualizer.py) | `label` | str | `None` | True | ... | explicit |
| [InferenceVisualizer.plot_distribution](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/inference/visualizer.py) | `unit` | str | `None` | True | ... | explicit |
| [InferenceVisualizer.plot_distribution](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/inference/visualizer.py) | `color` | str | `'blue'` | False | ... | explicit |
| [InferenceVisualizer.plot_distribution](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/inference/visualizer.py) | `title` | str | `None` | False | ... | explicit |
| [InferenceVisualizer.plot_distribution](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/inference/visualizer.py) | `highlight_outliers` | bool | `False` | False | ... | explicit |
| [InferenceVisualizer.plot_distribution](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/inference/visualizer.py) | `outlier_mask` | Optional[np.ndarray] | `None` | False | ... | explicit |
| [InferenceVisualizer.plot_error_distribution](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/inference/visualizer.py) | `error` | np.ndarray | `None` | True | ... | explicit |
| [InferenceVisualizer.plot_error_distribution](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/inference/visualizer.py) | `label` | str | `None` | True | ... | explicit |
| [InferenceVisualizer.plot_error_distribution](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/inference/visualizer.py) | `unit` | str | `None` | True | ... | explicit |
| [StatsCalculator.__init__](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/inference/stats.py) | `energy_per_atom` | np.ndarray | `None` | True | Predicted energy per atom.... | explicit |
| [StatsCalculator.__init__](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/inference/stats.py) | `force_flat` | np.ndarray | `None` | True | Predicted forces (flattened).... | explicit |
| [StatsCalculator.__init__](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/inference/stats.py) | `virial_per_atom` | Optional[np.ndarray] | `None` | False | Predicted virial per atom (optional).... | explicit |
| [StatsCalculator.__init__](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/inference/stats.py) | `energy_true` | Optional[np.ndarray] | `None` | False | Ground truth energy per atom (optional).... | explicit |
| [StatsCalculator.__init__](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/inference/stats.py) | `force_true` | Optional[np.ndarray] | `None` | False | Ground truth forces (flattened) (optional).... | explicit |
| [StatsCalculator.__init__](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/inference/stats.py) | `virial_true` | Optional[np.ndarray] | `None` | False | Ground truth virial per atom (optional).... | explicit |
| [StatsCalculator.__init__](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/inference/stats.py) | `atom_counts_list` | Optional[List[Dict[str, int]]] | `None` | False | List of atom counts dict for each frame (e.g. [{"H... | explicit |
| [StatsCalculator.__init__](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/inference/stats.py) | `atom_num_list` | Optional[List[int]] | `None` | False | List of total atom numbers for each frame.... | explicit |
| [StatsCalculator.__init__](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/inference/stats.py) | `ref_energies` | Optional[Dict[str, float]] | `None` | False | Dictionary of atomic reference energies (e.g. {"H"... | explicit |
| [StatsCalculator.get_distribution_stats](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/inference/stats.py) | `data` | Optional[np.ndarray] | `None` | True | ... | explicit |
| [StatsCalculator.get_distribution_stats](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/inference/stats.py) | `label` | str | `None` | True | ... | explicit |
| [StatsCalculator.compute_force_magnitude](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/inference/stats.py) | `force_flat` | np.ndarray | `None` | True | ... | explicit |
| [StatsCalculator.compute_relative_energy](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/inference/stats.py) | `energy` | np.ndarray | `None` | True | ... | explicit |
| [SelectKFromClusters.__init__](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/sampling/stratified_sampling.py) | `k` | int | `1` | False | Select k structures from each cluster.... | explicit |
| [SelectKFromClusters.__init__](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/sampling/stratified_sampling.py) | `allow_duplicate` | Any | `False` | False | Whether structures are allowed to be selected over... | explicit |
| [SelectKFromClusters.__init__](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/sampling/stratified_sampling.py) | `selection_criteria` | Any | `'center'` | False | The criteria to do stratified sampling from each c... | explicit |
| [SelectKFromClusters.__init__](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/sampling/stratified_sampling.py) | `n_sites` | Any | `None` | False | The number of sites in all the structures to sampl... | explicit |
| [SelectKFromClusters.fit](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/sampling/stratified_sampling.py) | `X` | Any | `None` | True | Input features... | explicit |
| [SelectKFromClusters.fit](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/sampling/stratified_sampling.py) | `y` | Any | `None` | False | Target.... | explicit |
| [SelectKFromClusters.transform](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/sampling/stratified_sampling.py) | `clustering_data` | dict | `None` | True | Results from clustering in a dict. The dict should... | explicit |
| [BirchClustering.__init__](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/sampling/clustering.py) | `n` | Any | `None` | False | Clustering the PCs into n clusters. When n is None... | explicit |
| [BirchClustering.__init__](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/sampling/clustering.py) | `threshold_init` | Any | `0.5` | False | The initial radius of the subcluster obtained by m... | explicit |
| [BirchClustering.fit](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/sampling/clustering.py) | `X` | Any | `None` | True | Any inputs... | explicit |
| [BirchClustering.fit](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/sampling/clustering.py) | `y` | Any | `None` | False | Any outputs... | explicit |
| [BirchClustering.transform](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/sampling/clustering.py) | `PCAfeatures` | Any | `None` | True | An array of PCA features.... | explicit |
| [DIRECTSampler.__init__](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/sampling/direct.py) | `structure_encoder` | Any | `None` | False | Structure featurizer. It can be any encoder that t... | explicit |
| [DIRECTSampler.__init__](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/sampling/direct.py) | `scaler` | Any | `'StandardScaler'` | False | StandardScaler to perform normalization before PCA... | explicit |
| [DIRECTSampler.__init__](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/sampling/direct.py) | `pca` | Any | `'PrincipalComponentAnalysis'` | False | PCA for dimensionality reduction.... | explicit |
| [DIRECTSampler.__init__](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/sampling/direct.py) | `weighting_PCs` | Any | `True` | False | Whether to weight PC with their explained variance... | explicit |
| [DIRECTSampler.__init__](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/sampling/direct.py) | `clustering` | Any | `'Birch'` | False | Clustering method to clustering based on PCs.... | explicit |
| [DIRECTSampler.__init__](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/sampling/direct.py) | `select_k_from_clusters` | Any | `'select_k_from_clusters'` | False | Straitified sampling of k structures from each clu... | explicit |
| [PrincipalComponentAnalysis.__init__](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/sampling/pca.py) | `weighting_PCs` | Any | `True` | False | Whether to weight PCs with explained variances.... | explicit |
| [PrincipalComponentAnalysis.fit](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/sampling/pca.py) | `normalized_features` | Any | `None` | True | An array of normalized features with fixed dimensi... | explicit |
| [PrincipalComponentAnalysis.transform](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/sampling/pca.py) | `normalized_features` | Any | `None` | True | An array of normalized features with fixed dimensi... | explicit |
| [JobManager.__init__](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/submission/manager.py) | `mode` | Literal['local', 'slurm'] | `'local'` | False | ... | explicit |
| [JobManager.__init__](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/submission/manager.py) | `custom_template_path` | Optional[str] | `None` | False | ... | explicit |
| [JobManager.generate_script](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/submission/manager.py) | `config` | JobConfig | `None` | True | ... | explicit |
| [JobManager.generate_script](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/submission/manager.py) | `output_path` | str | `None` | True | ... | explicit |
| [JobManager.submit](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/submission/manager.py) | `script_path` | str | `None` | True | ... | explicit |
| [JobManager.submit](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/submission/manager.py) | `working_dir` | str | `'.'` | False | ... | explicit |
| [JobManager.submit_python_script](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/submission/manager.py) | `script_content` | str | `None` | True | The Python code to run.... | explicit |
| [JobManager.submit_python_script](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/submission/manager.py) | `script_name` | str | `None` | True | Name of the python file (e.g. "run_task.py").... | explicit |
| [JobManager.submit_python_script](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/submission/manager.py) | `job_config` | JobConfig | `None` | True | Configuration for the job. The 'command' field wil... | explicit |
| [JobManager.submit_python_script](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/submission/manager.py) | `working_dir` | str | `'.'` | False | Directory to write script and submit job.... | explicit |
| [TemplateEngine.__init__](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/submission/templates.py) | `template_content` | str | `None` | True | ... | explicit |
| [TemplateEngine.from_file](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/submission/templates.py) | `cls` | Any | `None` | True | ... | explicit |
| [TemplateEngine.from_file](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/submission/templates.py) | `filepath` | str | `None` | True | ... | explicit |
| [TemplateEngine.from_default](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/submission/templates.py) | `cls` | Any | `None` | True | ... | explicit |
| [TemplateEngine.from_default](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/submission/templates.py) | `mode` | str | `None` | True | ... | explicit |
| [TemplateEngine.render](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/submission/templates.py) | `config` | JobConfig | `None` | True | ... | explicit |
| [UQFilter.__init__](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/uncertain/filter.py) | `scheme` | Any | `'tangent_lo'` | False | Filtering scheme name (e.g., "tangent_lo", "strict... | explicit |
| [UQFilter.__init__](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/uncertain/filter.py) | `trust_lo` | Any | `0.12` | False | Lower bound for QbC trust region.... | explicit |
| [UQFilter.__init__](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/uncertain/filter.py) | `trust_hi` | Any | `0.22` | False | Upper bound for QbC trust region.... | explicit |
| [UQFilter.__init__](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/uncertain/filter.py) | `rnd_trust_lo` | Any | `None` | False | Lower bound for RND trust region. Defaults to trus... | explicit |
| [UQFilter.__init__](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/uncertain/filter.py) | `rnd_trust_hi` | Any | `None` | False | Upper bound for RND trust region. Defaults to trus... | explicit |
| [UQFilter.filter](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/uncertain/filter.py) | `df_uq` | Any | `None` | True | DataFrame containing UQ metrics.... | explicit |
| [UQFilter.filter](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/uncertain/filter.py) | `qbc_col` | Any | `'uq_qbc_for'` | False | Column name for QbC UQ.... | explicit |
| [UQFilter.filter](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/uncertain/filter.py) | `rnd_col` | Any | `'uq_rnd_for_rescaled'` | False | Column name for RND UQ (usually rescaled).... | explicit |
| [UQFilter.get_identity_labels](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/uncertain/filter.py) | `df_uq` | Any | `None` | True | ... | explicit |
| [UQFilter.get_identity_labels](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/uncertain/filter.py) | `df_candidate` | Any | `None` | True | ... | explicit |
| [UQFilter.get_identity_labels](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/uncertain/filter.py) | `df_accurate` | Any | `None` | True | ... | explicit |
| [UQVisualizer.__init__](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/uncertain/visualization.py) | `save_dir` | Any | `None` | True | Directory to save plots.... | explicit |
| [UQVisualizer.__init__](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/uncertain/visualization.py) | `dpi` | Any | `150` | False | DPI for saved figures (default 150).... | explicit |
| [UQVisualizer._filter_uq](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/uncertain/visualization.py) | `data` | Any | `None` | True | ... | explicit |
| [UQVisualizer._filter_uq](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/uncertain/visualization.py) | `name` | Any | `'UQ'` | False | ... | explicit |
| [UQVisualizer.plot_uq_distribution](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/uncertain/visualization.py) | `uq_qbc` | Any | `None` | True | ... | explicit |
| [UQVisualizer.plot_uq_distribution](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/uncertain/visualization.py) | `uq_rnd` | Any | `None` | True | ... | explicit |
| [UQVisualizer.plot_uq_distribution](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/uncertain/visualization.py) | `uq_rnd_rescaled` | Any | `None` | False | ... | explicit |
| [UQVisualizer.plot_uq_with_trust_range](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/uncertain/visualization.py) | `uq_data` | Any | `None` | True | ... | explicit |
| [UQVisualizer.plot_uq_with_trust_range](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/uncertain/visualization.py) | `label` | Any | `None` | True | ... | explicit |
| [UQVisualizer.plot_uq_with_trust_range](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/uncertain/visualization.py) | `filename` | Any | `None` | True | ... | explicit |
| [UQVisualizer.plot_uq_with_trust_range](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/uncertain/visualization.py) | `trust_lo` | Any | `None` | True | ... | explicit |
| [UQVisualizer.plot_uq_with_trust_range](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/uncertain/visualization.py) | `trust_hi` | Any | `None` | True | ... | explicit |
| [UQVisualizer.plot_uq_vs_error](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/uncertain/visualization.py) | `uq_qbc` | Any | `None` | True | ... | explicit |
| [UQVisualizer.plot_uq_vs_error](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/uncertain/visualization.py) | `uq_rnd` | Any | `None` | True | ... | explicit |
| [UQVisualizer.plot_uq_vs_error](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/uncertain/visualization.py) | `diff_maxf` | Any | `None` | True | ... | explicit |
| [UQVisualizer.plot_uq_vs_error](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/uncertain/visualization.py) | `rescaled` | Any | `False` | False | ... | explicit |
| [UQVisualizer.plot_uq_diff_parity](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/uncertain/visualization.py) | `uq_qbc` | Any | `None` | True | ... | explicit |
| [UQVisualizer.plot_uq_diff_parity](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/uncertain/visualization.py) | `uq_rnd_rescaled` | Any | `None` | True | ... | explicit |
| [UQVisualizer.plot_uq_diff_parity](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/uncertain/visualization.py) | `diff_maxf` | Any | `None` | False | ... | explicit |
| [UQVisualizer.plot_uq_fdiff_scatter](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/uncertain/visualization.py) | `df_uq` | Any | `None` | True | ... | explicit |
| [UQVisualizer.plot_uq_fdiff_scatter](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/uncertain/visualization.py) | `scheme` | Any | `None` | True | ... | explicit |
| [UQVisualizer.plot_uq_fdiff_scatter](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/uncertain/visualization.py) | `trust_lo` | Any | `None` | True | ... | explicit |
| [UQVisualizer.plot_uq_fdiff_scatter](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/uncertain/visualization.py) | `trust_hi` | Any | `None` | True | ... | explicit |
| [UQVisualizer.plot_uq_fdiff_scatter](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/uncertain/visualization.py) | `rnd_trust_lo` | Any | `None` | True | ... | explicit |
| [UQVisualizer.plot_uq_fdiff_scatter](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/uncertain/visualization.py) | `rnd_trust_hi` | Any | `None` | True | ... | explicit |
| [UQVisualizer.plot_uq_identity_scatter](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/uncertain/visualization.py) | `df_uq` | Any | `None` | True | ... | explicit |
| [UQVisualizer.plot_uq_identity_scatter](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/uncertain/visualization.py) | `scheme` | Any | `None` | True | ... | explicit |
| [UQVisualizer.plot_uq_identity_scatter](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/uncertain/visualization.py) | `trust_lo` | Any | `None` | True | ... | explicit |
| [UQVisualizer.plot_uq_identity_scatter](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/uncertain/visualization.py) | `trust_hi` | Any | `None` | True | ... | explicit |
| [UQVisualizer.plot_uq_identity_scatter](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/uncertain/visualization.py) | `rnd_trust_lo` | Any | `None` | True | ... | explicit |
| [UQVisualizer.plot_uq_identity_scatter](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/uncertain/visualization.py) | `rnd_trust_hi` | Any | `None` | True | ... | explicit |
| [UQVisualizer.plot_candidate_vs_error](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/uncertain/visualization.py) | `df_uq` | Any | `None` | True | ... | explicit |
| [UQVisualizer.plot_candidate_vs_error](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/uncertain/visualization.py) | `df_candidate` | Any | `None` | True | ... | explicit |
| [UQVisualizer.plot_pca_analysis](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/uncertain/visualization.py) | `explained_variance` | Any | `None` | True | ... | explicit |
| [UQVisualizer.plot_pca_analysis](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/uncertain/visualization.py) | `selected_PC_dim` | Any | `None` | True | ... | explicit |
| [UQVisualizer.plot_pca_analysis](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/uncertain/visualization.py) | `all_features` | Any | `None` | True | PCA features for all samples (Joint if joint sampl... | explicit |
| [UQVisualizer.plot_pca_analysis](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/uncertain/visualization.py) | `direct_indices` | Any | `None` | True | Indices selected by DIRECT (in all_features).... | explicit |
| [UQVisualizer.plot_pca_analysis](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/uncertain/visualization.py) | `random_indices` | Any | `None` | True | Indices selected by Random (in all_features).... | explicit |
| [UQVisualizer.plot_pca_analysis](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/uncertain/visualization.py) | `scores_direct` | Any | `None` | True | Coverage scores for DIRECT.... | explicit |
| [UQVisualizer.plot_pca_analysis](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/uncertain/visualization.py) | `scores_random` | Any | `None` | True | Coverage scores for Random.... | explicit |
| [UQVisualizer.plot_pca_analysis](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/uncertain/visualization.py) | `df_uq` | Any | `None` | True | Dataframe of candidates (for Final_sampled_PCAview... | explicit |
| [UQVisualizer.plot_pca_analysis](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/uncertain/visualization.py) | `final_indices` | Any | `None` | True | Indices of finally selected candidates (relative t... | explicit |
| [UQVisualizer.plot_pca_analysis](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/uncertain/visualization.py) | `n_candidates` | Any | `None` | False | Number of candidate samples. If provided, assumes ... | explicit |
| [UQVisualizer.plot_pca_analysis](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/uncertain/visualization.py) | `full_features` | Any | `None` | False | PCA features for the entire dataset (including fil... | explicit |
| [UQVisualizer._plot_coverage](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/uncertain/visualization.py) | `all_features` | Any | `None` | True | ... | explicit |
| [UQVisualizer._plot_coverage](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/uncertain/visualization.py) | `selected_indices` | Any | `None` | True | ... | explicit |
| [UQVisualizer._plot_coverage](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/uncertain/visualization.py) | `method` | Any | `None` | True | ... | explicit |
| [UQVisualizer._plot_coverage](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/uncertain/visualization.py) | `n_candidates` | Any | `None` | False | ... | explicit |
| [UQVisualizer._setup_2d_plot_axes](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/uncertain/visualization.py) | `x_lo` | Any | `None` | True | ... | explicit |
| [UQVisualizer._setup_2d_plot_axes](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/uncertain/visualization.py) | `x_hi` | Any | `None` | True | ... | explicit |
| [UQVisualizer._setup_2d_plot_axes](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/uncertain/visualization.py) | `y_lo` | Any | `None` | True | ... | explicit |
| [UQVisualizer._setup_2d_plot_axes](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/uncertain/visualization.py) | `y_hi` | Any | `None` | True | ... | explicit |
| [UQVisualizer._draw_boundary](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/uncertain/visualization.py) | `scheme` | Any | `None` | True | ... | explicit |
| [UQVisualizer._draw_boundary](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/uncertain/visualization.py) | `uq_x_lo` | Any | `None` | True | ... | explicit |
| [UQVisualizer._draw_boundary](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/uncertain/visualization.py) | `uq_x_hi` | Any | `None` | True | ... | explicit |
| [UQVisualizer._draw_boundary](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/uncertain/visualization.py) | `uq_y_lo` | Any | `None` | True | ... | explicit |
| [UQVisualizer._draw_boundary](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/uncertain/visualization.py) | `uq_y_hi` | Any | `None` | True | ... | explicit |
| [UQCalculator.compute_qbc_rnd](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/uncertain/calculator.py) | `predictions_0` | PredictionData | `None` | True | Predictions from model 0 (main model).... | explicit |
| [UQCalculator.compute_qbc_rnd](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/uncertain/calculator.py) | `predictions_1` | PredictionData | `None` | True | Predictions from model 1.... | explicit |
| [UQCalculator.compute_qbc_rnd](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/uncertain/calculator.py) | `predictions_2` | PredictionData | `None` | True | Predictions from model 2.... | explicit |
| [UQCalculator.compute_qbc_rnd](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/uncertain/calculator.py) | `predictions_3` | PredictionData | `None` | True | Predictions from model 3.... | explicit |
| [UQCalculator.align_scales](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/uncertain/calculator.py) | `uq_qbc` | np.ndarray | `None` | True | QbC uncertainty values.... | explicit |
| [UQCalculator.align_scales](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/uncertain/calculator.py) | `uq_rnd` | np.ndarray | `None` | True | RND uncertainty values.... | explicit |
| [UQCalculator.calculate_trust_lo](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/uncertain/calculator.py) | `data` | np.ndarray | `None` | True | The uncertainty data (1D array).... | explicit |
| [UQCalculator.calculate_trust_lo](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/uncertain/calculator.py) | `ratio` | float | `0.5` | False | The ratio of peak density to define the cutoff (de... | explicit |
| [UQCalculator.calculate_trust_lo](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/uncertain/calculator.py) | `grid_size` | int | `1000` | False | Number of points for KDE evaluation grid.... | explicit |
| [UQCalculator.calculate_trust_lo](/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/src/dpeva/uncertain/calculator.py) | `bound` | Tuple[float, float] | `(0, 2.0)` | False | Tuple of (min, max) for the grid range.... | explicit |
