#!/usr/bin/env python3
"""Verify real-data identities, selection handoff and independent notebooks."""
from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
os.environ.setdefault('MPLBACKEND', 'Agg')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import nbformat
import numpy as np
import dpdata
from ase.io import read
from jupyter_client import KernelManager
from nbclient import NotebookClient
import tutorial_support as t


def check_paper_alignment():
    """Compare recomputed defaults with independent archived labels and names."""
    bundle = t.load_bundle()
    uq = bundle['uq']
    candidate, accurate, failed, labels = t.classify(uq)
    assert (len(candidate), len(accurate), len(failed)) == (5339, 22217, 100)
    np.testing.assert_array_equal(labels.to_numpy(), uq.paper_identity.to_numpy())
    result = t.sample_live()
    assert result['cluster_budget'] == result['selected_total'] == 3000
    actual = set(uq.iloc[result['selected_pool_indices']].dataname)
    historical = set(bundle['paper_selected'].dataname)
    # Report membership drift independently; never substitute the historical list.
    assert actual <= set(candidate.dataname)
    return {'candidate_frames': len(candidate), 'historical_classifications_match': True,
            'selected_total': result['selected_total'], 'new_frames': len(actual),
            'historical_new_frames': len(historical),
            'historical_selection_matches': len(actual & historical),
            'selection_identical': actual == historical,
            'thresholds': t.paper_thresholds()}


def check_data_and_handoff():
    b = t.load_bundle()
    assert b['features'].shape == (30656, 128)
    assert b['iteration_features'].shape == (30656, 128)
    assert b['features'].dtype == b['iteration_features'].dtype == np.dtype('float64')
    np.testing.assert_array_equal(np.sort(b['cold_start_order']), np.arange(30656))
    np.testing.assert_array_equal(np.sort(b['training_order']), np.arange(3000))
    np.testing.assert_allclose(np.linalg.norm(b['iteration_features'], axis=1), 1, atol=1e-7)
    structures = b['structures']
    assert structures['positions'].shape == structures['forces'].shape == (1251299, 3)
    assert structures['cells'].shape == structures['virials'].shape == (30656, 3, 3)
    assert len(structures['energies']) == 30656
    assert b['manifest'].source_dataname.is_unique
    assert b['provenance']['full_structures']['frames'] == 30656
    # Recompute independent physical summaries from the complete raw labels.
    stats = json.loads((ROOT / 'data/dataset_stats.json').read_text())
    np.testing.assert_allclose((structures['energies'] / b['manifest'].n_atoms).mean(),
                               stats['energy_per_atom']['mean'])
    np.testing.assert_allclose(np.linalg.norm(structures['forces'], axis=1).mean(),
                               stats['force_magnitude']['mean'])
    np.testing.assert_allclose(np.linalg.norm(b['features'], axis=1), 1, atol=1e-7)
    assert b['uq'].paper_identity.value_counts().to_dict() == {
        'accurate': 22217, 'candidate': 5339, 'failed': 100}
    assert len(b['paper_selected']) == 843
    assert b['paper_selected'].dataname.is_unique
    assert set(b['paper_selected'].dataname) <= set(
        b['uq'].loc[b['uq'].paper_identity == 'candidate', 'dataname'])
    probe = t.recompute_probe()
    aligned = t.UQCalculator().align_scales(b['uq'].uq_qbc_for, b['uq'].uq_rnd_for)
    np.testing.assert_allclose(aligned, b['uq'].uq_rnd_for_rescaled, atol=1e-8)
    for scheme in ['tangent_lo', 'circle_lo', 'strict']:
        t.classify(b['uq'], scheme=scheme)
    assert t.sample_live(low=10000, high=10001)['candidate_count'] == 0

    cases = []
    for joint in [False, True]:
        for use_uq in [False, True]:
            for budget in [16, 48, 96]:
                result = t.sample_live(joint=joint, use_uq=use_uq, n_clusters=budget, scope='small')
                selected = result['selected_pool_indices']
                assert len(selected) == len(np.unique(selected))
                assert set(selected) <= set(b['live']['pool'])
                assert len(selected) + result['selected_training_count'] == result['selected_total']
                cases.append({'scope': 'small', 'joint': joint, 'use_uq': use_uq, 'budget': budget,
                              'new_frames': len(selected)})
    selected = t.sample_live()['selected_pool_indices']
    assert len(selected) == len(set(selected))
    assert set(selected) <= set(t.classify(b['uq'])[0].index)
    full = t.sample_live(joint=False)
    assert full['cluster_budget'] == full['selected_total'] == 1000
    assert len(full['selected_pool_indices']) == full['selected_total']
    assert set(full['selected_pool_indices']) <= set(t.classify(b['uq'])[0].index)
    # Include frames beyond the old small bundle and both ends of the pool.
    full_export_probe = np.array([0, 27655, int(selected[0])])
    np.testing.assert_array_equal(selected, t.sample_live()['selected_pool_indices'])
    # Removing every ground-truth diagnostic must leave selection unchanged.
    original = b['uq']
    try:
        b['uq'] = original.drop(columns=['diff_maxf_0_frame', 'paper_identity'])
        t.paper_thresholds.cache_clear()
        np.testing.assert_array_equal(selected, t.sample_live()['selected_pool_indices'])
    finally:
        b['uq'] = original
        t.paper_thresholds.cache_clear()

    first, second = t.export_selection(full_export_probe), t.export_selection(selected[:1])
    assert first != second
    assert len(list((second / 'label_input').iterdir())) == 1
    loaded = read(first / 'selected.extxyz', index=':')
    assert len(loaded) == len(full_export_probe)
    for i, (pool_index, atoms) in enumerate(zip(full_export_probe, loaded)):
        reference = t.get_structure(pool_index)
        assert atoms.calc is None
        np.testing.assert_array_equal(atoms.numbers, reference.numbers)
        np.testing.assert_allclose(atoms.positions, reference.positions, atol=1e-8)
        block = first / 'label_input' / f'frame_{i:04d}' / 'set.000'
        assert {p.name for p in block.iterdir()} == {'coord.npy', 'box.npy'}
        np.testing.assert_array_equal(np.load(block / 'coord.npy'), reference.positions.reshape(1, -1))
        np.testing.assert_array_equal(np.load(block / 'box.npy'), reference.cell.array.reshape(1, 9))
        system = dpdata.System(str(block.parent), fmt='deepmd/npy')
        assert len(system) == 1
        assert 'energies' not in system.data and 'forces' not in system.data
        np.testing.assert_allclose(system.data['coords'][0], reference.positions)
        symbols = [system.data['atom_names'][j] for j in system.data['atom_types']]
        assert symbols == reference.get_chemical_symbols()
    for invalid in [[], [selected[0], selected[0]]]:
        try:
            t.export_selection(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError('Empty or duplicated selection accepted')
    return {'full_structure_frames': 30656, 'full_pool_normal_and_joint': True,
            'probe_frames': len(probe), 'selection_cases': cases,
            'default_new_frames': len(selected), 'label_blind_selection': True,
            'unlabeled_handoff': True, 'separate_stage_features': True}


def execute_notebook(path, write_back):
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    manager = KernelManager(kernel_name='python3')
    # Use this interpreter, independent of globally registered kernelspecs.
    manager.kernel_spec.argv = [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}']
    started = time.monotonic()
    client = NotebookClient(notebook, km=manager, timeout=180,
                            resources={'metadata': {'path': str(ROOT)}},
                            store_widget_state=True, shutdown_kernel='immediate')
    client.execute(cleanup_kc=True)
    # Widget callbacks capture exceptions instead of raising a cell error.
    state = notebook.metadata.get('widgets', {})
    for widget in state.get('application/vnd.jupyter.widget-state+json', {}).get('state', {}).values():
        for output in widget.get('state', {}).get('outputs', []):
            assert output.get('output_type') != 'error', output
    # Retain static plots/tables; browser widgets are rebuilt when run locally.
    notebook.metadata.pop('widgets', None)
    for cell in notebook.cells:
        if cell.cell_type == 'code':
            cell.outputs = [output for output in cell.outputs
                            if 'application/vnd.jupyter.widget-view+json' not in output.get('data', {})]
        cell.get('metadata', {}).pop('execution', None)
    figures = sum('image/png' in output.get('data', {})
                  for cell in notebook.cells for output in cell.get('outputs', []))
    assert figures >= 1, f'{path.name}: missing static figures'
    nbformat.validate(notebook)
    target = path if write_back else ROOT / 'work' / 'executed' / path.name
    target.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(notebook, target)
    elapsed = round(time.monotonic() - started, 2)
    print(f'{path.name}: passed ({elapsed}s)', flush=True)
    return {'name': path.name, 'seconds': elapsed,
            'code_cells': sum(c.cell_type == 'code' for c in notebook.cells), 'static_figures': figures}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write-notebooks', action='store_true')
    args = parser.parse_args()
    report = {'status': 'running', 'python': sys.version.split()[0],
              'versions': {p: version(p) for p in ['numpy', 'pandas', 'scikit-learn', 'ase', 'nbclient']}}
    report_path = ROOT / 'work' / 'verification.json'
    report_path.parent.mkdir(exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + '\n')
    try:
        report['paper_alignment'] = check_paper_alignment()
        report['checks'] = check_data_and_handoff()
        print('Data, historical UQ and selection/export checks: passed', flush=True)
        report['notebooks'] = [execute_notebook(p, args.write_notebooks) for p in sorted(ROOT.glob('*.ipynb'))]
        assert len(report['notebooks']) == 4
        sources = [ROOT / 'tutorial_support.py', Path(__file__), *ROOT.glob('*.ipynb'), *ROOT.joinpath('data').iterdir(), ROOT / 'configs/paper_sampling.json']
        report['sha256'] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in sorted(sources) if p.is_file()}
        report['status'] = 'passed'
    except Exception as error:
        report['status'] = 'failed'
        report['error'] = repr(error)
        raise
    finally:
        report_path.write_text(json.dumps(report, indent=2) + '\n')
    print('All tutorial checks passed; report: work/verification.json')


if __name__ == '__main__':
    main()
