"""Build the portable teaching bundle from the author's paper experiment archive.

Usage: python scripts/prepare_data.py --source /path/to/test-for-paper
Readers of the tutorial do not need the archive or this preparation step.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from pack_full_structures import pack_structures
from pack_sampling_features import pack_sampling_features

ROOT = Path(__file__).resolve().parents[1]
SEED = 20261021


def composition(name):
    system = name.rsplit('/', 1)[-1].rsplit('-', 1)[0]
    return {element: int(count) for element, count in re.findall(r'([A-Z][a-z]*)(\d+)', system)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--output', type=Path, default=ROOT / 'data')
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    sources = []

    def read_csv(relative):
        path = source / relative
        sources.append({'path': relative, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
        return pd.read_csv(path, index_col=0)

    train = read_csv('dpeva-data/dpeva-init/dpeva_sampling_direct/dataframe/final_df.csv')
    pool = read_csv('dpeva-data/dpeva-iter1/dpeva_post_joint/dataframe/df_uq_desc.csv')
    uq = read_csv('dpeva-data/dpeva-iter1/dpeva_post_joint/dataframe/df_uq.csv')
    selected = read_csv('dpeva-data/dpeva-iter1/dpeva_post_joint/dataframe/final_df.csv')
    assert len(train) == 3000 and len(pool) == len(uq) == 27656 and len(selected) == 843
    assert pool.dataname.is_unique and train.dataname.is_unique
    uq = uq.set_index('dataname').loc[pool.dataname].reset_index()
    assert set(selected.dataname) <= set(uq.loc[uq.uq_identity == 'candidate', 'dataname'])
    columns = [f'desc_stru_{i}' for i in range(128)]
    features = np.concatenate([train[columns].to_numpy(), pool[columns].to_numpy()])
    assert np.isfinite(features).all()
    np.testing.assert_allclose(np.linalg.norm(features, axis=1), 1, atol=1e-6)
    # float64 preserves the archived values; no two-dimensional projection is
    # substituted for the features used by DIRECT.
    names = train.dataname.tolist() + pool.dataname.tolist()
    manifest = pd.DataFrame({'feature_row': np.arange(len(names)), 'dataname': names,
                             'role': ['train'] * len(train) + ['pool'] * len(pool)})
    manifest['frame_id'] = manifest.role + '/' + manifest.dataname
    for element in ['Fe', 'C', 'H', 'O']:
        manifest[element] = [composition(name).get(element, 0) for name in names]
    manifest['n_atoms'] = manifest[['Fe', 'C', 'H', 'O']].sum(axis=1)
    manifest.to_csv(output / 'manifest.csv.gz', index=False, compression={'method': 'gzip', 'mtime': 0})
    uq.rename(columns={'uq_identity': 'paper_identity'}).to_csv(
        output / 'uq.csv.gz', index=False, compression={'method': 'gzip', 'mtime': 0})
    selected[['dataname']].to_csv(output / 'paper_selected.csv', index=False)
    rng = np.random.default_rng(SEED)
    live_pool = np.sort(rng.choice(len(pool), 512, replace=False))
    live_train = np.sort(rng.choice(len(train), 256, replace=False))
    np.savez_compressed(output / 'small_subset_indices.npz', pool=live_pool, train=live_train)

    print('Packing all 30,656 structures and verifying split/feature identity...', flush=True)
    full_structures = pack_structures(source, output)
    sampling_inputs = pack_sampling_features(source, output)

    # The first eight frames are a small raw committee example. Verify system
    # identity and reference forces across all four files before packing.
    probe = uq.iloc[:8]
    probe_systems = [n.split('/')[-1].rsplit('-', 1)[0] for n in probe.dataname]
    assert len(set(probe_systems)) == 1
    assert [int(n.rsplit('-', 1)[1]) for n in probe.dataname] == list(range(8))
    n_atoms = sum(composition(probe.dataname.iloc[0]).values())
    raw_forces = []
    for model in range(4):
        rel = f'dpeva-data/dpeva-iter1/{model}/test_val/results.f.out'
        with (source / rel).open() as stream:
            header = stream.readline()
            assert header.startswith('# ') and header.split(':')[0].endswith('/' + probe_systems[0])
            values = np.array([[float(v) for v in stream.readline().split()] for _ in range(8 * n_atoms)])
        assert values.shape == (8 * n_atoms, 6)
        if raw_forces:
            np.testing.assert_array_equal(values[:, :3], raw_forces[0][:, :3])
        raw_forces.append(values)
    np.savez_compressed(output / 'committee_probe.npz', force=np.stack(raw_forces),
                        datanames=probe.dataname.to_numpy(dtype=str), n_atoms=np.full(8, n_atoms))
    summary = json.loads((source / 'alldata-analysis/analysis_dataset/dataset_stats.json').read_text())
    summary.pop('dataset_dir', None)
    (output / 'dataset_stats.json').write_text(json.dumps(summary, indent=2) + '\n')
    provenance = {
        'paper_doi': '10.1093/ce/zkag029', 'model': 'DPA-3.1-3M', 'model_head': 'OC22',
        'dataset': 'FT2DP-dataset-FeCHO-v1', 'dataset_aissquare_id': 306,
        'source_root_description': 'Author archive: CE-2025-DPEVA-paper/test-for-paper',
        'sources': sources, 'seed': SEED, 'full_structures': full_structures,
        'sampling_inputs': sampling_inputs,
        'storage': {'arrays': 'features/iteration_features/structures.npz: float64 ZIP LZMA; no precision reduction',
                    'large_tables': 'manifest.csv.gz and uq.csv.gz: gzip'},
        'live_selection': 'Uniform without replacement from pool (512), then train (256); no label/error filtering.',
        'rows': {'train': 3000, 'pool': 27656, 'paper_new': 843},
        'precision': 'float64 raw atomic descriptors, mean pooling + L2; separate init and iter1 arrays',
        'identity': 'feature_row indexes features.npz; dataname frame index is local to its source dataset, not global.',
        'checks': full_structures['checks'] + [
            'All eight raw committee probe frames share identical reference forces across four models.'],
        'limitations': ['Committee model.ckpt.pt files in this archive are zero-byte placeholders; predictions are real archived outputs.',
                        'No new training, DFT calculation or accuracy benchmark is performed by the notebooks.',
                        'Historical validation sets are the remaining structures after each selection, not a common holdout.',
                        'SI Figure S2 uses an earlier DPA-2.3.1 experiment; this tutorial uses DPA3 iteration data.'],
    }
    (output / 'provenance.json').write_text(json.dumps(provenance, indent=2, ensure_ascii=False) + '\n')
    print('Data bundle ready:', output, flush=True)


if __name__ == '__main__':
    main()
