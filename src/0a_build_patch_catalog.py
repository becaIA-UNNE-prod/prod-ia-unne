'''
Builds/updates an incremental catalog of patch-level metadata (tile, year,
maize prevalence) used by the group-stratified split
(src/0_data_split.py --how group_stratified).

Only the 'labels' group of each netCDF patch is read (not the spectral
bands), so this stays cheap even as the number of tiles grows. Re-running
this script only processes patches that are not already present in the
catalog, so cataloging a large, ever-growing dataset never requires
re-scanning files that were already processed.
'''
import argparse
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import netCDF4
import xarray as xr
from tqdm.contrib.concurrent import process_map

from utils.settings.config import LINEAR_ENCODER
from utils.split_tools import load_catalog, save_catalog

netCDF4.default_encoding = 'utf-8'

# Every non-background class in LINEAR_ENCODER is a "target" class to track
# prevalence for (currently just maize, id 120 - see utils/settings/config.py).
TARGET_CLASSES = sorted(k for k in LINEAR_ENCODER if k != 0)

CATALOG_COLUMNS = ['file_name', 'tile', 'year', 'has_target', 'n_target_px', 'n_valid_px']


def _process_one(data_path, rel_path):
    path = data_path / rel_path
    base_name = path.stem.split('_')
    year, tile = base_name[0], base_name[1]

    try:
        patch_netcdf = netCDF4.Dataset(path, 'r')
        labels = xr.open_dataset(xr.backends.NetCDF4DataStore(patch_netcdf['labels'])).labels.data
        labels = np.asarray(labels)
    except Exception as e:
        print(f'[WARNING] Saltando "{rel_path}": {e}', flush=True)
        return None

    n_target_px = int(np.isin(labels, TARGET_CLASSES).sum())

    return {
        'file_name': rel_path,
        'tile': tile,
        'year': year,
        'has_target': n_target_px > 0,
        'n_target_px': n_target_px,
        'n_valid_px': int(labels.size),
    }


def build_catalog(data_path, catalog_path, num_workers=8):
    data_path = Path(data_path)
    catalog_path = Path(catalog_path)

    if catalog_path.exists():
        existing = load_catalog(catalog_path)
    else:
        existing = pd.DataFrame(columns=CATALOG_COLUMNS)

    known = set(existing['file_name'])

    all_paths = sorted(f'{p.parts[-2]}/{p.parts[-1]}' for p in data_path.rglob('*.nc'))
    new_paths = [p for p in all_paths if p not in known]

    print(f'Catalogo: {len(existing)} patches ya catalogados, {len(new_paths)} nuevos por procesar.', flush=True)

    if not new_paths:
        return existing

    rows = process_map(
        partial(_process_one, data_path),
        new_paths,
        max_workers=num_workers,
        chunksize=16,
    )
    rows = [r for r in rows if r is not None]

    catalog = pd.concat([existing, pd.DataFrame(rows, columns=CATALOG_COLUMNS)], ignore_index=True)
    save_catalog(catalog, catalog_path)

    print(f'Catalogo actualizado: {len(catalog)} patches totales -> "{catalog_path}"', flush=True)

    return catalog


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path', type=str, required=True,
                        help='Ruta a los archivos .nc de patches (misma que --data_path de 0_data_split.py).')
    parser.add_argument('--catalog_path', type=str, default='data/patch_catalog.csv', required=False,
                        help='Ruta del catalogo (csv). Se actualiza incrementalmente. Default "data/patch_catalog.csv".')
    parser.add_argument('--num_workers', type=int, default=8, required=False)

    args = parser.parse_args()

    build_catalog(args.data_path, args.catalog_path, args.num_workers)
