"""Validate the released data, paired row provenance and archived checksums."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import numpy as np
ROOT=Path(__file__).resolve().parents[1]


def check(data_dir):
    data_dir=Path(data_dir)
    pairs=[('parameter_geometry_5200.npy','label_5200.npy',3840),('parameters-10000.npy','performance-10000.npy',10000),('merged_geometry.npy','merged_label.npy',13840)]
    arrays=[]
    for gn,yn,n in pairs:
        g,y=[np.load(data_dir/p,allow_pickle=False) for p in [gn,yn]]
        assert g.shape==(n,21) and y.shape==(n,3),(gn,g.shape,y.shape)
        assert np.isfinite(g).all() and np.isfinite(y).all(),gn
        arrays.append(np.column_stack([g,y]))
    with (data_dir/'sample_sources.csv').open(encoding='utf8') as f:
        rows=list(csv.DictReader(f))
    assert len(rows)==13840
    for i,row in enumerate(rows):
        assert int(row['merged_row_0based'])==i
        source=arrays[0 if row['label_source']=='CFD' else 1]
        assert np.array_equal(arrays[2][i],source[int(row['source_row_0based'])])
    for name,a in zip(['dataset_I_CFD','additional_TeM','dataset_II_merged'],arrays):
        assert np.array_equal(np.loadtxt(data_dir/f'{name}.csv',delimiter=',',skiprows=1),a),name
    print('PASS: 3,840 CFD + 10,000 TeM samples; exact paired mapping and CSV roundtrip')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir',type=Path,default=ROOT/'data')
    p.add_argument('--check-results',action='store_true')
    args=p.parse_args()
    try:
        check(args.data_dir)
        if args.check_results:
            entries=json.loads((ROOT/'paper_results/source_manifest.json').read_text())
            for row in entries:
                path=ROOT/row['path']
                assert hashlib.sha256(path.read_bytes()).hexdigest()==row['sha256'],row['path']
            print(f'PASS: {len(entries)} archived source checksums')
    except (AssertionError,FileNotFoundError,ValueError) as e:p.exit(2,f'Data verification failed: {e}\n')


if __name__=='__main__':main()
