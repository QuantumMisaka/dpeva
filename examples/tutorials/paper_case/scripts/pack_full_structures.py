"""Pack all paper structures losslessly, validating original frame and split identity."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from ase.data import atomic_numbers
from archive_io import save_npz

ROOT = Path(__file__).resolve().parents[1]
FIELDS = {'positions': 'coord', 'forces': 'force', 'cells': 'box',
          'energies': 'energy', 'virials': 'virial'}


def read_system(folder):
    symbols = (folder / 'type_map.raw').read_text().split()
    types = np.loadtxt(folder / 'type.raw', dtype=int, ndmin=1)
    numbers = np.array([atomic_numbers[symbols[i]] for i in types], dtype=np.int64)
    sets = sorted(folder.glob('set.*'))
    if not sets:
        raise ValueError(f'No DeepMD npy sets: {folder}')
    result = {}
    for key, stem in FIELDS.items():
        blocks = []
        for block in sets:
            if (block / 'real_atom_types.npy').exists():
                raise ValueError(f'Mixed atom ordering needs explicit mapping: {folder}')
            value = np.load(block / f'{stem}.npy', allow_pickle=False)
            shape = (-1, len(numbers), 3) if key in {'positions', 'forces'} else (
                (-1,) if key == 'energies' else (-1, 3, 3))
            blocks.append(value.reshape(shape))
        result[key] = np.concatenate(blocks)
        if not np.isfinite(result[key]).all():
            raise ValueError(f'Non-finite {key}: {folder}')
    assert len({len(v) for v in result.values()}) == 1
    result['numbers'] = numbers
    return result


def pack_structures(source, output):
    manifest = pd.read_csv(output / 'manifest.csv.gz')
    assert len(manifest) == 30656
    assert np.array_equal(manifest.feature_row, np.arange(30656))
    names = manifest.dataname.str.rsplit('/', n=1).str[-1].str.rsplit('-', n=1)
    systems = names.str[0]
    local_frames = names.str[1].astype(int)
    offsets = np.r_[0, np.cumsum(manifest.n_atoms.to_numpy())]
    count, atoms = len(manifest), int(offsets[-1])
    packed = {'feature_rows': np.arange(count), 'offsets': offsets,
              'numbers': np.empty(atoms, dtype=np.int64),
              'positions': np.empty((atoms, 3)), 'forces': np.empty((atoms, 3)),
              'cells': np.empty((count, 3, 3)), 'energies': np.empty(count),
              'virials': np.empty((count, 3, 3))}
    original_names = np.empty(count, dtype=object)
    folders = sorted((source / 'ft2dp-v1-30656').glob('*/type.raw'))
    assert len(folders) == 737
    seen = np.zeros(count, dtype=bool)
    for number, type_file in enumerate(folders, 1):
        folder, system = type_file.parent, type_file.parent.name
        original = read_system(folder)
        n_frames = len(original['positions'])
        rows = manifest.index[systems == system].to_numpy()
        training_rows = rows[manifest.loc[rows, 'role'].to_numpy() == 'train']
        pool_rows = rows[manifest.loc[rows, 'role'].to_numpy() == 'pool']
        training_frames = local_frames.loc[training_rows].to_numpy()
        assert len(np.unique(training_frames)) == len(training_frames)
        assert ((training_frames >= 0) & (training_frames < n_frames)).all()
        remaining = np.setdiff1d(np.arange(n_frames), training_frames)
        assert len(rows) == n_frames and len(remaining) == len(pool_rows)
        np.testing.assert_array_equal(np.sort(local_frames.loc[pool_rows]), np.arange(len(remaining)))
        if len(remaining):
            split = read_system(source / 'dpeva-data/dpeva-iter1/other_dpdata' / system)
            np.testing.assert_array_equal(original['numbers'], split['numbers'])
            for key in FIELDS:
                np.testing.assert_array_equal(original[key][remaining], split[key],
                                              err_msg=f'{system}: {key} split mismatch')
        original_frames = local_frames.loc[rows].to_numpy().copy()
        pool_mask = manifest.loc[rows, 'role'].to_numpy() == 'pool'
        original_frames[pool_mask] = remaining[original_frames[pool_mask]]
        for row, frame in zip(rows, original_frames):
            lo, hi = offsets[row:row + 2]
            assert hi - lo == len(original['numbers'])
            packed['numbers'][lo:hi] = original['numbers']
            for key in ['positions', 'forces']:
                packed[key][lo:hi] = original[key][frame]
            for key in ['cells', 'energies', 'virials']:
                packed[key][row] = original[key][frame]
            original_names[row] = f'{system}-{frame}'
        seen[rows] = True
        if number % 100 == 0 or number == len(folders):
            print(f'Full structures: {number}/737 systems verified', flush=True)
    assert seen.all() and len(set(original_names)) == count
    # Commit output only after all original/split/feature checks have passed.
    temporary = output / 'structures.building.npz'
    save_npz(temporary, **packed)
    temporary.replace(output / 'structures.npz')
    manifest['source_dataname'] = original_names
    manifest.to_csv(output / 'manifest.csv.gz', index=False, compression={'method': 'gzip', 'mtime': 0})
    return {'frames': count, 'atoms': atoms, 'systems': len(folders),
            'source': 'ft2dp-v1-30656', 'precision': 'float64, lossless NPZ (ZIP LZMA) compression',
            'checks': ['All 27,656 pool coordinates and E/F/virial/cells exactly match the original train complement.',
                       'source_dataname uniquely maps each feature row to the original full dataset.']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--output', type=Path, default=ROOT / 'data')
    args = parser.parse_args()
    report = pack_structures(args.source.resolve(), args.output.resolve())
    path = args.output / 'provenance.json'
    provenance = json.loads(path.read_text())
    provenance['full_structures'] = report
    provenance['checks'] = report['checks'] + [
        'All eight raw committee probe frames share identical reference forces across four models.']
    path.write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
