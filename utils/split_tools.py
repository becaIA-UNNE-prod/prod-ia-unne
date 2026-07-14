'''
Utilities for the group-stratified train/val/test split ("--how group_stratified"
in src/0_data_split.py).

Design (see doc/0_data_split.md for the full rationale):

- The split unit is a (tile, year) group, not an individual patch. All
  sub-patches of a given tile-year always land in the same split, which
  avoids spatial leakage between neighboring patches (they share fields,
  weather and acquisition dates).
- Groups are bucketed by their target-class prevalence (e.g. % of patches
  containing maize) so that train/val/test keep a similar class balance.
- Assignment is incremental and stable: once a group has been assigned to a
  split it never moves. New groups (new tiles/years) are only added to
  whichever split needs them to keep the target ratio, so metrics stay
  comparable across experiments as more data is downloaded.
'''
import json
from pathlib import Path

import numpy as np
import pandas as pd

SPLIT_NAMES = ['train', 'val', 'test']


def load_catalog(path):
    '''Loads the patch catalog (see src/0a_build_patch_catalog.py) with consistent dtypes.'''
    return pd.read_csv(path, dtype={'file_name': str, 'tile': str, 'year': str})


def save_catalog(catalog, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    catalog.to_csv(path, index=False)


def compute_group_prevalence(catalog):
    '''
    Aggregates the patch catalog into one row per (tile, year) group, with
    the number of patches and the fraction of patches containing the target
    class.
    '''
    catalog = catalog.copy()
    catalog['group'] = catalog['tile'] + '_' + catalog['year']

    grouped = catalog.groupby('group').agg(
        tile=('tile', 'first'),
        year=('year', 'first'),
        n_patches=('has_target', 'size'),
        pct_has_target=('has_target', 'mean'),
    ).reset_index()

    return grouped


def bucketize_prevalence(group_df, n_buckets=3, min_groups_per_bucket=2):
    '''
    Splits groups into prevalence buckets (quantile-based) so that the
    incremental assignment keeps a similar class balance across splits.
    Falls back to fewer buckets automatically if there are too few groups
    (e.g. early on, with few tiles downloaded) or too many ties (e.g. many
    groups with 0% prevalence).
    '''
    group_df = group_df.copy()
    n = len(group_df)

    n_buckets = max(1, min(n_buckets, n // max(min_groups_per_bucket, 1)))

    if n_buckets <= 1:
        group_df['bucket'] = 0
        return group_df

    buckets = pd.qcut(group_df['pct_has_target'], q=n_buckets, labels=False, duplicates='drop')
    group_df['bucket'] = buckets.fillna(0).astype(int)

    return group_df


def load_assignments(path):
    path = Path(path)
    if path.exists():
        with open(path, encoding='utf-8') as f:
            return json.load(f)

    return {}


def save_assignments(path, assignments):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, 'wt', encoding='utf-8') as f:
        json.dump(assignments, f, indent=2, sort_keys=True)


def assign_new_groups(group_df, assignments, ratio=(60, 20, 20), seed=16):
    '''
    Extends `assignments` (a dict of group -> split) with any group present
    in `group_df` that isn't assigned yet. Previously assigned groups are
    never changed.

    Each new group is assigned, within its own prevalence bucket, to the
    split whose current patch count is furthest below its target ratio -
    this keeps both the overall train/val/test ratio and the per-bucket
    (i.e. per class-prevalence) balance close to the requested one, without
    ever reshuffling groups that were already assigned in a previous run.

    Parameters
    ----------
    group_df: DataFrame
        Output of `bucketize_prevalence`, with columns
        ['group', 'bucket', 'n_patches'].
    assignments: dict
        Existing group -> split assignments (possibly empty).
    ratio: tuple of 3 numbers
        Relative train/val/test ratio. Does not need to sum to 100.
    seed: int
        Used only to break ties deterministically among same-size groups.

    Returns
    -------
    dict: the updated group -> split assignments.
    '''
    ratio = np.asarray(ratio, dtype=float)
    ratio = ratio / ratio.sum()

    assignments = dict(assignments)

    # Patch counts already committed per (bucket, split), from prior runs.
    counts = {}
    for row in group_df.itertuples(index=False):
        split = assignments.get(row.group)
        if split is not None:
            key = (row.bucket, split)
            counts[key] = counts.get(key, 0) + row.n_patches

    pending = group_df[~group_df['group'].isin(assignments)].copy()

    if pending.empty:
        return assignments

    # Deterministic order: shuffle for reproducibility, then place bigger
    # groups first within each bucket (they have the largest impact on the
    # final ratio, so placing them first minimizes drift for the smaller
    # ones that follow).
    pending = pending.sample(frac=1, random_state=seed)
    pending = pending.sort_values(['bucket', 'n_patches'], ascending=[True, False])

    for row in pending.itertuples(index=False):
        bucket_total = sum(counts.get((row.bucket, s), 0) for s in SPLIT_NAMES) + row.n_patches

        deficits = {
            split: ratio[i] * bucket_total - counts.get((row.bucket, split), 0)
            for i, split in enumerate(SPLIT_NAMES)
        }
        chosen = max(SPLIT_NAMES, key=lambda s: deficits[s])

        assignments[row.group] = chosen
        counts[(row.bucket, chosen)] = counts.get((row.bucket, chosen), 0) + row.n_patches

    return assignments


def export_coco_from_catalog(catalog, assignments, coco_path, prefix):
    '''
    Writes '{prefix}_coco_{train,val,test}.json' from the patch catalog and
    the group -> split assignments.
    '''
    from utils.coco_tools import init_coco
    from utils.settings.config import IMG_SIZE, CROP_ENCODING, LINEAR_ENCODER

    coco_path = Path(coco_path)
    coco_path.mkdir(parents=True, exist_ok=True)

    catalog = catalog.copy()
    catalog['group'] = catalog['tile'] + '_' + catalog['year']
    catalog['split'] = catalog['group'].map(assignments)

    n_unmatched = int(catalog['split'].isna().sum())
    if n_unmatched:
        print(f'[WARNING] {n_unmatched} patches belong to a group with no split assignment, skipping them.')

    categories = [
        {'supercategory': 'Crop', 'name': crop_name, 'id': LINEAR_ENCODER[crop_id]}
        for crop_name, crop_id in CROP_ENCODING.items() if crop_id in LINEAR_ENCODER
    ]

    for split in SPLIT_NAMES:
        coco = init_coco()
        coco['categories'] = categories

        subset = catalog[catalog['split'] == split]
        for image_id, row in enumerate(subset.itertuples(index=False), start=1):
            coco['images'].append({
                'license': 1,
                'file_name': row.file_name,
                'height': IMG_SIZE,
                'width': IMG_SIZE,
                'date_captured': row.year,
                'id': image_id,
            })

        out_path = coco_path / f'{prefix}_coco_{split}.json'
        with open(out_path, 'wt', encoding='utf-8') as f:
            json.dump(coco, f)

        n_groups = catalog.loc[catalog['split'] == split, 'group'].nunique()
        print(f'{split}: {len(subset)} patches from {n_groups} tile-years -> "{out_path}"')
