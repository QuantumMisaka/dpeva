"""Pack stage-specific raw descriptors in the paper's sampling input order."""
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from archive_io import save_npz


def pack_sampling_features(source, output):
    source, output = Path(source), Path(output)
    manifest = pd.read_csv(output / 'manifest.csv.gz')
    original = manifest.source_dataname.str.rsplit('-', n=1)
    systems, frames = original.str[0], original.str[1].astype(int)
    cold = np.empty((len(manifest), 128))
    iteration = np.empty_like(cold)
    train_rows, cold_order, sources = [], [], []

    def pooled(path):
        sources.append({'path': str(path.relative_to(source)),
                        'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
        atomic = np.load(path)
        mean = atomic.mean(axis=1)
        return mean / (np.linalg.norm(mean, axis=1, keepdims=True) + 1e-12)

    base = source / 'dpeva-data'
    for system in sorted(systems.unique()):
        rows = manifest.index[systems == system].to_numpy()
        ordered = rows[np.argsort(frames.iloc[rows].to_numpy())]
        cold_order.extend(ordered)
        cold[rows] = pooled(base / 'dpeva-init/desc_pool' / f'{system}.npy')[frames.iloc[rows]]
        selected = ordered[manifest.iloc[ordered].role.to_numpy() == 'train']
        remaining = rows[manifest.iloc[rows].role.to_numpy() == 'pool']
        if len(selected):
            values = pooled(base / 'dpeva-iter1/desc_train' / f'{system}.npy')
            assert len(values) == len(selected)
            iteration[selected] = values
            train_rows.extend(selected)
        if len(remaining):
            local = manifest.iloc[remaining].dataname.str.rsplit('-', n=1).str[1].astype(int)
            iteration[remaining] = pooled(base / 'dpeva-iter1/desc_pool' / f'{system}.npy')[local]
    assert len(train_rows) == 3000 and len(cold_order) == 30656
    assert set(train_rows) == set(range(3000))
    np.testing.assert_allclose(np.linalg.norm(cold, axis=1), 1, atol=1e-7)
    np.testing.assert_allclose(np.linalg.norm(iteration, axis=1), 1, atol=1e-7)
    save_npz(output / 'features.npz', features=cold, paper_order=np.array(cold_order))
    save_npz(output / 'iteration_features.npz', features=iteration,
             training_order=np.array(train_rows))
    return {'cold_start': 'features.npz: init raw descriptors, manifest row order; paper_order restores loader order',
            'iteration': 'iteration_features.npz: iter1 raw descriptors, manifest row order; training_order restores loader order',
            'pool_order': 'manifest pool order, identical to archived df_uq_desc_sampled-UQ candidate order',
            'sources': sources}
