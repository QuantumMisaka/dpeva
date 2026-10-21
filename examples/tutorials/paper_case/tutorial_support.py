"""Notebook presentation and file handoff; algorithms live in dpeva.*.

The CPU tutorial imports the source checkout, so it does not require DeepMD,
GPU libraries, a model checkpoint, or an editable installation of the package.
"""
from __future__ import annotations

import json
import sys
from functools import lru_cache
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ase import Atoms
from ase.io import write
from ase.visualize.plot import plot_atoms

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]
sys.path.insert(0, str(REPO / 'src'))
from dpeva.io.types import PredictionData  # noqa: E402
from dpeva.sampling.manager import SamplingManager  # noqa: E402
from dpeva.uncertain.calculator import UQCalculator  # noqa: E402
from dpeva.uncertain.filter import UQFilter  # noqa: E402

PAPER = json.loads((ROOT / 'configs/paper_sampling.json').read_text())

COLORS = {'accurate': '#4c956c', 'candidate': '#e9a23b', 'failed': '#d1495b'}


@lru_cache(maxsize=1)
def load_bundle():
    data = ROOT / 'data'
    def arrays(name):
        with np.load(data / name, allow_pickle=False) as archive:
            return {key: archive[key] for key in archive.files}
    manifest = pd.read_csv(data / 'manifest.csv.gz')
    cold = arrays('features.npz')
    features = cold['features']
    iteration = arrays('iteration_features.npz')
    uq = pd.read_csv(data / 'uq.csv.gz')
    assert len(features) == len(manifest) == 30656
    assert manifest.frame_id.is_unique
    assert np.array_equal(manifest.loc[manifest.role == 'pool', 'dataname'], uq.dataname)
    assert np.isfinite(features).all()
    structures = arrays('structures.npz')
    np.testing.assert_array_equal(structures['feature_rows'], manifest.feature_row)
    np.testing.assert_array_equal(np.diff(structures['offsets']), manifest.n_atoms)
    assert manifest.source_dataname.is_unique
    return {'manifest': manifest, 'features': features, 'cold_start_order': cold['paper_order'],
            'iteration_features': iteration['features'], 'training_order': iteration['training_order'], 'uq': uq,
            'live': arrays('small_subset_indices.npz'), 'structures': structures,
            'probe': arrays('committee_probe.npz'),
            'paper_selected': pd.read_csv(data / 'paper_selected.csv'),
            'provenance': json.loads((data / 'provenance.json').read_text())}


@lru_cache(maxsize=1)
def paper_thresholds():
    """Recompute both KDE bounds using the full pool, never error labels."""
    uq = load_bundle()['uq']
    calc = UQCalculator()
    bounds = {}
    for name, column in [('qbc', 'uq_qbc_for'), ('rnd', 'uq_rnd_for_rescaled')]:
        low = calc.calculate_trust_lo(uq[column].to_numpy(), ratio=PAPER['uq_trust_ratio'])
        if low is None:
            raise ValueError('论文全池的自动 UQ 阈值计算失败。')
        bounds[name] = (low, low + PAPER['uq_trust_width'])
    return bounds


def uq_thresholds(low=None, high=None):
    if low is None and high is None:
        return paper_thresholds().copy()
    if low is None or high is None:
        raise ValueError('手动阈值必须同时指定 low 和 high。')
    return {'qbc': (low, high), 'rnd': (low, high)}


def paper_parameters():
    return dict(joint=True, n_clusters=PAPER['joint_clusters'], low=None, high=None,
                scheme=PAPER['uq_select_scheme'], use_uq=True, scope='full')


def classify(uq, low=None, high=None, scheme='tangent_lo'):
    """Use full-pool paper auto thresholds unless manual bounds are supplied."""
    bounds = uq_thresholds(low, high)
    candidate, accurate, failed = UQFilter(scheme, *bounds['qbc'], *bounds['rnd']).filter(uq)
    labels = pd.Series('failed', index=uq.index)
    labels.loc[accurate.index] = 'accurate'
    labels.loc[candidate.index] = 'candidate'
    assert len(candidate) + len(accurate) + len(failed) == len(uq)
    return candidate, accurate, failed, labels


def plot_uq(low=None, high=None, scheme='tangent_lo', reveal_error=False):
    uq = load_bundle()['uq']
    candidate, accurate, failed, labels = classify(uq, low, high, scheme)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    ax = axes[0]
    # Fixed viewport for comparison; decisions always use all frames.
    limit = 0.6
    if reveal_error:
        points = ax.scatter(uq.uq_qbc_for, uq.uq_rnd_for_rescaled,
                            c=np.log10(uq.diff_maxf_0_frame.clip(lower=1e-5)), s=5, cmap='magma')
        fig.colorbar(points, ax=ax, label='log10 max force-vector error [eV/A]')
    else:
        for label in COLORS:
            frame = uq.loc[labels == label]
            ax.scatter(frame.uq_qbc_for, frame.uq_rnd_for_rescaled, s=5,
                       color=COLORS[label], label=f'{label}: {len(frame):,}', alpha=0.55)
        ax.legend(markerscale=3, fontsize=8)
    grid = np.linspace(0, limit, 150)
    xx, yy = np.meshgrid(grid, grid)
    mesh = pd.DataFrame({'uq_qbc_for': xx.ravel(), 'uq_rnd_for_rescaled': yy.ravel()})
    mesh_labels = classify(mesh, low, high, scheme)[3]
    mask = (mesh_labels == 'candidate').to_numpy().reshape(xx.shape)
    if mask.any() and not mask.all():
        ax.contour(xx, yy, mask, levels=[0.5], colors='black', linewidths=1)
    ax.set(xlim=(0, limit), ylim=(0, limit), xlabel='QbC [eV/A]', ylabel='Rescaled RND [eV/A]', title=scheme)
    outside = ((uq.uq_qbc_for > limit) | (uq.uq_rnd_for_rescaled > limit) |
               (uq.uq_qbc_for < 0) | (uq.uq_rnd_for_rescaled < 0)).sum()
    ax.text(0.02, 0.98, f'{outside} points outside viewport; counted in totals', transform=ax.transAxes,
            va='top', fontsize=7)
    if reveal_error:
        for label in COLORS:
            values = uq.loc[labels == label, 'diff_maxf_0_frame']
            if len(values):
                axes[1].hist(values, bins=np.geomspace(1e-4, max(10, uq.diff_maxf_0_frame.max()), 65),
                             density=True, histtype='step', color=COLORS[label], label=label)
        axes[1].set(xscale='log', xlabel='Max force-vector error [eV/A]', ylabel='Density', title='Labels revealed AFTER selection')
        axes[1].legend()
    else:
        counts = [len(accurate), len(candidate), len(failed)]
        axes[1].bar(list(COLORS), counts, color=list(COLORS.values()))
        axes[1].set(ylabel='Frames', title='All 27,656 pool frames')
    plt.show()
    return {'accurate': len(accurate), 'candidate': len(candidate), 'failed': len(failed)}


def recompute_probe():
    probe = load_bundle()['probe']
    names = ['data_fx', 'data_fy', 'data_fz', 'pred_fx', 'pred_fy', 'pred_fz']
    metadata = [[str(key), i, int(n)] for i, (key, n) in enumerate(zip(probe['datanames'], probe['n_atoms']))]
    predictions = []
    for raw in probe['force']:
        force = np.empty(len(raw), dtype=[(name, 'f8') for name in names])
        for i, name in enumerate(names):
            force[name] = raw[:, i]
        predictions.append(PredictionData(force=force, has_ground_truth=True, dataname_list=metadata))
    result = UQCalculator().compute_qbc_rnd(predictions)
    frame = pd.DataFrame({key: result[key] for key in ['uq_qbc_for', 'uq_rnd_for', 'diff_maxf_0_frame']},
                         index=probe['datanames'])
    original = load_bundle()['uq'].set_index('dataname').loc[frame.index]
    np.testing.assert_allclose(frame.to_numpy(), original[frame.columns].to_numpy(), rtol=1e-6, atol=1e-8)
    return frame


def scope_indices(scope='full'):
    bundle = load_bundle()
    if scope == 'full':
        return np.arange(3000), np.arange(27656)
    if scope == 'small':
        return bundle['live']['train'], bundle['live']['pool']
    raise ValueError('scope must be full or small')


def cold_start(n_clusters=None, scope='full'):
    n_clusters = PAPER['cold_start_clusters'] if n_clusters is None else n_clusters
    bundle = load_bundle()
    train, pool = scope_indices(scope)
    rows = bundle['cold_start_order']
    rows = rows[np.isin(rows, np.r_[train, 3000 + pool])]
    features = bundle['features'][rows]
    np.random.seed(42)
    result = SamplingManager({'direct_n_clusters': int(n_clusters), 'direct_k': PAPER['direct_k'],
                              'direct_thr_init': PAPER['direct_thr_init']}).execute_sampling(features)
    print(f'冷启动 DIRECT：输入 {len(features):,} 帧 × {features.shape[1]} 维；'
          f'聚类预算 {int(n_clusters):,}；实际选择 {len(result["selected_indices"]):,} 帧；'
          f'保留 {result["pca_features"].shape[1]} 个主成分；随机种子 42。')
    selected_rows = rows[np.asarray(result['selected_indices'], dtype=int)]
    print(f'与论文初始训练名单交集：{np.count_nonzero(selected_rows < 3000):,} / 3,000；名单仅用于事后对照。')
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    pcs = result['pca_features']
    for ax, key, title in zip(axes, ['random_indices', 'selected_indices'], ['Random (same count)', 'DIRECT']):
        selected = np.asarray(result[key], dtype=int)
        ax.scatter(pcs[:, 0], pcs[:, 1], s=6, color='#c6c9ce')
        ax.scatter(pcs[selected, 0], pcs[selected, 1], s=18, color='#157f93')
        ax.set(xlabel='Weighted PC1', ylabel='Weighted PC2', title=f'{title}: {len(selected)} frames')
    plt.show()
    return pd.DataFrame({'DIRECT coverage': result['scores_direct'], 'Random coverage': result['scores_random']}).rename_axis('PC')


def sample_live(joint=True, n_clusters=None, low=None, high=None, scheme='tangent_lo', use_uq=True,
                scope='full'):
    """Selection is label-blind. Returned pool indices map to packed structures."""
    bundle = load_bundle()
    train, pool = scope_indices(scope)
    n_clusters = (PAPER['joint_clusters'] if joint else PAPER['normal_clusters']) if n_clusters is None else n_clusters
    candidates = bundle['uq'].iloc[pool].copy()
    if use_uq:
        candidates = classify(candidates, low, high, scheme)[0]
    if candidates.empty:
        return {'selected_pool_indices': np.array([], dtype=int), 'candidate_count': 0,
                'selected_training_count': 0, 'selected_total': 0, 'retained_pcs': 0,
                'cluster_budget': int(n_clusters)}
    desc_columns = [f'desc_stru_{i}' for i in range(128)]
    candidate_features = bundle['iteration_features'][3000 + candidates.index.to_numpy()]
    frame = pd.DataFrame(candidate_features, columns=desc_columns, index=candidates.index)
    ordered_train = bundle['training_order']
    ordered_train = ordered_train[np.isin(ordered_train, train)]
    training = bundle['iteration_features'][ordered_train] if joint else np.empty((0, 128))
    manager = SamplingManager({'direct_n_clusters': int(n_clusters), 'direct_k': PAPER['direct_k'], 'direct_thr_init': PAPER['direct_thr_init']})
    features, _, n_candidates = manager.prepare_features(frame, frame, training)
    if len(features) < 3:
        raise ValueError('用于 PCA 的样本少于 3 帧；请扩大 UQ 区间或启用 joint。')
    np.random.seed(42)
    result = manager.execute_sampling(features)
    selected = np.asarray(result['selected_indices'], dtype=int)
    new = selected[selected < n_candidates]
    return {'selected_pool_indices': candidates.index.to_numpy()[new], 'candidate_count': n_candidates,
            'selected_training_count': int((selected >= n_candidates).sum()), 'selected_total': len(selected),
            'retained_pcs': result['pca_features'].shape[1], 'pca_features': result['pca_features'],
            'selected_indices': selected, 'n_candidates': n_candidates, 'cluster_budget': int(n_clusters)}


def plot_sampling(joint=True, n_clusters=None, use_uq=True, low=None, high=None, scheme='tangent_lo',
                  scope='full'):
    result = sample_live(joint, n_clusters, low, high, scheme, use_uq, scope)
    if not result['candidate_count']:
        print('当前阈值下没有候选；请扩大区间。')
        return result
    pcs, n = result['pca_features'], result['n_candidates']
    # A narrow selection may retain only one PC; zero-pad only the plot.
    if pcs.shape[1] == 1:
        pcs = np.column_stack([pcs[:, 0], np.zeros(len(pcs))])
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    ax.scatter(pcs[:n, 0], pcs[:n, 1], s=15, color='#a6c9ce', label=f'Candidates ({n})')
    if joint:
        ax.scatter(pcs[n:, 0], pcs[n:, 1], marker='x', s=13, color='#888888', label=f'Existing train ({len(pcs) - n})')
    new = result['selected_indices'][result['selected_indices'] < n]
    ax.scatter(pcs[new, 0], pcs[new, 1], s=45, facecolors='none', edgecolors='#d1495b', label=f'New ({len(new)})')
    ax.set(xlabel='Weighted PC1', ylabel='Weighted PC2', title=f'{"Joint" if joint else "Normal"} DIRECT; {result["retained_pcs"]} PCs used')
    ax.legend(fontsize=8)
    plt.show()
    print(f'聚类预算 {result["cluster_budget"]}；聚类选中 {result["selected_total"]}；已有训练点 {result["selected_training_count"]}；新增 {len(new)}。')
    return result


def compare_paper_selection(result):
    """Post-selection audit only; historical membership is never a selector."""
    bundle = load_bundle()
    actual = set(bundle['uq'].iloc[result['selected_pool_indices']].dataname)
    paper = set(bundle['paper_selected'].dataname)
    return pd.Series({'本次新增': len(actual), '历史新增': len(paper),
                      '名单交集': len(actual & paper), '仅本次': len(actual - paper),
                      '仅历史': len(paper - actual), '名单完全一致': actual == paper},
                     name='论文名单对照（选数后）').to_frame()


def get_frame(feature_row):
    """Read any of the 30,656 frames in feature/manifest order, without labels."""
    bundle = load_bundle()
    structures = bundle['structures']
    i = int(feature_row)
    if not 0 <= i < len(bundle['manifest']):
        raise ValueError('feature_row must be between 0 and 30655')
    lo, hi = structures['offsets'][i:i + 2]
    # Do not attach reference labels to the candidate used for labeling export.
    atoms = Atoms(numbers=structures['numbers'][lo:hi], positions=structures['positions'][lo:hi],
                  cell=structures['cells'][i], pbc=True)
    row = bundle['manifest'].iloc[i]
    atoms.info.update(paper_dataname=row.dataname, source_dataname=row.source_dataname,
                      feature_row=i, role=row.role)
    return atoms


def get_structure(pool_index):
    if not 0 <= int(pool_index) < 27656:
        raise ValueError('pool_index must be between 0 and 27655')
    return get_frame(3000 + int(pool_index))


def show_structure(feature_row=0):
    atoms = get_frame(feature_row)
    fig, ax = plt.subplots(figsize=(7, 4))
    plot_atoms(atoms, ax=ax, rotation='15x,25y', show_unit_cell=2)
    ax.set_title(f'{atoms.get_chemical_formula()} | {atoms.info["role"]} | feature row {feature_row}')
    ax.set_axis_off()
    plt.show()
    return atoms.info['paper_dataname']


def export_selection(pool_indices):
    indices = np.asarray(pool_indices, dtype=int)
    if not len(indices):
        raise ValueError('No candidates selected; no export was written.')
    if len(np.unique(indices)) != len(indices):
        raise ValueError('Duplicate selected indices.')
    atoms = [get_structure(i) for i in indices]  # Validate before writing anything.
    # A new run directory prevents stale systems from a prior larger export.
    from tempfile import mkdtemp
    work = ROOT / 'work'
    work.mkdir(exist_ok=True)
    target = Path(mkdtemp(prefix='selection-', dir=work))
    write(target / 'selected.extxyz', atoms, format='extxyz', write_results=False)
    mapping = []
    for i, (pool_index, structure) in enumerate(zip(indices, atoms)):
        system = target / 'label_input' / f'frame_{i:04d}'
        block = system / 'set.000'
        block.mkdir(parents=True)
        symbols = structure.get_chemical_symbols()
        type_map = sorted(set(symbols))
        (system / 'type_map.raw').write_text('\n'.join(type_map) + '\n')
        np.savetxt(system / 'type.raw', [type_map.index(s) for s in symbols], fmt='%d')
        np.save(block / 'coord.npy', structure.positions.reshape(1, -1))
        np.save(block / 'box.npy', structure.cell.array.reshape(1, 9))
        mapping.append({'export_system': system.name, 'pool_index': int(pool_index),
                        'dataname': structure.info['paper_dataname'],
                        'source_dataname': structure.info['source_dataname']})
    pd.DataFrame(mapping).to_csv(target / 'selected.csv', index=False)
    return target
