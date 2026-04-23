import argparse
from pathlib import Path
from tqdm import tqdm
import numpy as np
from numpy.lib.stride_tricks import as_strided
import pandas as pd
from tqdm.contrib.concurrent import process_map
from functools import partial

import xarray as xr
from pycocotools.coco import COCO
import netCDF4


IMG_SIZE = 366
BANDS = {
    'B02': 10, 'B03': 10, 'B04': 10, 'B08': 10,
    'B05': 20, 'B07': 20, 'B06': 20, 'B8A': 20, 'B11': 20, 'B12': 20,
    'B01': 60, 'B09': 60, 'B10': 60
}

# Extract patches based on this band
REFERENCE_BAND = 'B02'

def process_patch(out_path, mode, num_buckets, data_path, bands, padded_patch_height,
                  padded_patch_width, medians_dtype, label_dtype, group_freq, output_size,
                  pad_top, pad_bot, pad_left, pad_right, patch):
    patch_id, patch_info = patch
    patch_dir = out_path / mode / f'{patch_id}'
    patch_dir.mkdir(exist_ok=True, parents=True)

    # if len(list(patch_dir.iterdir())) == num_buckets + 1:
    #     return

    # Extraer rutas y prefijos desde el JSON
    file_path = Path(patch_info['file_name']) # ej: 31TCG/2019_31TCG_patch_21_17_B11.nc
    year = file_path.name.split('_')[0]

    # Directorio base del parche: ej. /mnt/yacy_3/prod/.../2019/31TCG
    patch_data_dir = data_path / year / file_path.parent

    # Nombre base sin la banda: ej. 2019_31TCG_patch_21_17
    base_name = "_".join(file_path.stem.split('_')[:-1])

    try:
        # Calculate medians
        medians = get_medians(patch_data_dir, base_name, 0, num_buckets, group_freq, bands,
                              padded_patch_height, padded_patch_width, output_size,
                              pad_top, pad_bot, pad_left, pad_right, medians_dtype)

        # Extraemos labels aquí adentro también, por si el archivo _labels.nc es el corrupto
        labels = get_labels(patch_data_dir, base_name, output_size, pad_top, pad_bot, pad_left, pad_right)

    except Exception as e:
        # Si un archivo está roto, lo reportamos y salimos de la función sin crashear el programa entero
        print(f"\n[WARNING] Saltando el parche {patch_id} por archivo corrupto: {e}")
        return

    num_bins, num_bands = medians.shape[:2]

    # Usamos [0, 0] para descartar las dimensiones de salto (steps) de tiempo y banda,
    # conservando intactas las dimensiones de la grilla espacial (sub_row, sub_col) aunque sean 1.
    medians = sliding_window_view(medians, [num_bins, num_bands, output_size[0], output_size[1]], [1, 1, output_size[0], output_size[1]])[0, 0]
    # shape: (subpatches_in_row, subpatches_in_col, bins, bands, height, width)

    # Save medians
    bins_pad = len(str(medians.shape[-4]))
    subs_pad = len(str(medians.shape[0] * medians.shape[1]))
    sub_idx = 0
    for i in range(medians.shape[0]):
        for j in range(medians.shape[1]):
            for t in range(num_bins):
                np.save(patch_dir / f'sub{str(sub_idx).rjust(subs_pad, "0")}_bin{str(t).rjust(bins_pad, "0")}', medians[i, j, t, :, :, :].astype(medians_dtype))
            sub_idx += 1

    # Save labels
    labels = get_labels(patch_data_dir, base_name, output_size, pad_top, pad_bot, pad_left, pad_right)
    # Al no usar squeeze, forzamos a que conserve la forma 4D: (subpatches_in_row, subpatches_in_col, height, width)
    labels = sliding_window_view(labels, output_size, output_size)

    lbl_idx = 0
    lbl_pad = len(str(labels.shape[0] * labels.shape[1]))
    for i in range(labels.shape[0]):
        for j in range(labels.shape[1]):
            np.save(patch_dir / f'labels_sub{str(lbl_idx).rjust(lbl_pad, "0")}', labels[i, j, :, :].astype(label_dtype))
            lbl_idx += 1

def sliding_window_view(arr, window_shape, steps):
    '''
    Code taken from:
        https://gist.github.com/meowklaski/4bda7c86c6168f3557657d5fb0b5395a

    Produce a view from a sliding, striding window over `arr`.
        The window is only placed in 'valid' positions - no overlapping
        over the boundary.
        Parameters
        ----------
        arr : numpy.ndarray, shape=(...,[x, (...), z])
            The array to slide the window over.
        window_shape : Sequence[int]
            The shape of the window to raster: [Wx, (...), Wz],
            determines the length of [x, (...), z]
        steps : Sequence[int]
            The step size used when applying the window
            along the [x, (...), z] directions: [Sx, (...), Sz]
        Returns
        -------
        view of `arr`, shape=([X, (...), Z], ..., [Wx, (...), Wz])
            Where X = (x - Wx) // Sx + 1
    '''
    in_shape = np.array(arr.shape[-len(steps):])  # [x, (...), z]
    window_shape = np.array(window_shape)  # [Wx, (...), Wz]
    steps = np.array(steps)  # [Sx, (...), Sz]
    nbytes = arr.strides[-1]  # size (bytes) of an element in `arr`

    # number of per-byte steps to take to fill window
    window_strides = tuple(np.cumprod(arr.shape[:0:-1])[::-1]) + (1,)
    # number of per-byte steps to take to place window
    step_strides = tuple(window_strides[-len(steps):] * steps)
    # number of bytes to step to populate sliding window view
    strides = tuple(int(i) * nbytes for i in step_strides + window_strides)

    outshape = tuple((in_shape - window_shape) // steps + 1)
    # outshape: ([X, (...), Z], ..., [Wx, (...), Wz])
    outshape = outshape + arr.shape[:-len(steps)] + tuple(window_shape)
    return as_strided(arr, shape=outshape, strides=strides, writeable=False)


def get_medians(patch_data_dir, base_name, start_bin, window, group_freq, bands,
                padded_patch_height, padded_patch_width, output_size,
                pad_top, pad_bot, pad_left, pad_right, medians_dtype):

    # Extraer el año directamente del nombre base
    year = base_name.split('_')[0]

    # Intervalos de salida
    date_range = pd.date_range(start=f'{year}-01-01', end=f'{int(year) + 1}-01-01', freq=group_freq)

    # Array de salida
    medians = np.empty((len(bands), window, padded_patch_height, padded_patch_width), dtype=medians_dtype)

    for band_id, band in enumerate(bands):
        # Construir la ruta exacta para esta banda
        band_file = patch_data_dir / f"{base_name}_{band}.nc"
        ds = xr.open_dataset(band_file)

        # 1. Extraer los offsets de tiempo del atributo global
        time_attr = ds.attrs.get('GDAL_NETCDF_DIM_time_VALUES', [])
        if isinstance(time_attr, str):
            # Limpiar corchetes y extraer números
            time_attr = time_attr.replace('[', '').replace(']', '').replace(',', ' ')
            time_offsets = [float(x) for x in time_attr.split()]
        else:
            time_offsets = list(time_attr)

        # Crear objetos datetime asumiendo que los offsets son segundos desde el inicio del año
        times = pd.to_datetime(f'{year}-01-01') + pd.to_timedelta(time_offsets, unit='s')

        # 2. Extraer y ordenar las variables (Band1, Band2... Band27)
        band_vars = [v for v in ds.data_vars if v.startswith('Band')]
        band_vars = sorted(band_vars, key=lambda x: int(x.replace('Band', '')))

        # Apilar las matrices 2D en un solo tensor 3D (time, lat, lon)
        data_3d = np.stack([ds[v].values for v in band_vars], axis=0)

        # 3. Construir un DataArray estándar con la dimensión 'time'
        da = xr.DataArray(
            data_3d,
            dims=['time', 'lat', 'lon'],
            coords={'time': times}
        )

        # 4. Ahora sí, aplicar la lógica original de agrupación temporal
        da = da.groupby_bins(
            'time',
            bins=date_range,
            right=True,
            include_lowest=False,
            labels=date_range[:-1]
        ).median(dim='time')

        da = da.resample(time_bins=group_freq).median(dim='time_bins')
        da = da.interpolate_na(dim='time_bins', method='linear', fill_value='extrapolate')
        da = da.isel(time_bins=slice(start_bin, start_bin + window))

        # Convertir de vuelta a NumPy
        band_data_np = da.values

        # Cortar valores extremos (como NoData=65535) para que entren bien en float16 (max 65504)
        band_data_np = np.clip(band_data_np, -65500, 65500)

        # Calcular el ratio dinámicamente dividiendo el tamaño objetivo (IMG_SIZE) por el ancho actual de la banda
        expand_ratio = IMG_SIZE // band_data_np.shape[1]

        # Expandir si la matriz es más pequeña que la banda de referencia
        if expand_ratio > 1:
            band_data_np = np.repeat(band_data_np, expand_ratio, axis=1)
            band_data_np = np.repeat(band_data_np, expand_ratio, axis=2)

        # Padding si es necesario
        if (output_size[0] < band_data_np.shape[1]) or (output_size[1] < band_data_np.shape[2]):
            band_data_np = np.pad(band_data_np,
                                  pad_width=((0, 0), (pad_top, pad_bot), (pad_left, pad_right)),
                                  mode='constant',
                                  constant_values=0)

        medians[band_id, :, :, :] = np.expand_dims(band_data_np, axis=0)

    # (T, B, H, W)
    return medians.transpose(1, 0, 2, 3)


def get_labels(patch_data_dir, base_name, output_size, pad_top, pad_bot, pad_left, pad_right):
    # Construir ruta y cargar array de NumPy
    label_file = patch_data_dir / f"{base_name}_labels.nc"
    labels = xr.open_dataset(label_file)['Band1'].values

    # Add padding if needed
    if (output_size[0] < labels.shape[0]) or (output_size[1] < labels.shape[1]):
        labels = np.pad(labels,
                        pad_width=((pad_top, pad_bot), (pad_left, pad_right)),
                        mode='constant',
                        constant_values=0
                        )

    return labels


def get_padding_offset(patch_height, patch_width, output_size):
    img_size_x = patch_height
    img_size_y = patch_width

    output_size_x = output_size[0]
    output_size_y = output_size[1]

    # Calculate padding offset
    if img_size_x >= output_size_x:
        pad_x = int(output_size_x - img_size_x % output_size_x)
    else:
        # For bigger images, is just the difference
        pad_x = output_size_x - img_size_x

    if img_size_y >= output_size_y:
        pad_y = int(output_size_y - img_size_y % output_size_y)
    else:
        # For bigger images, is just the difference
        pad_y = output_size_y - img_size_y

    # Number of rows that need to be padded (top and bot)
    if not pad_x == output_size_x:
        pad_top = int(pad_x // 2)
        pad_bot = int(pad_x // 2)

        # if padding is not equally divided, pad +1 row to the top
        if not pad_x % 2 == 0:
            pad_top += 1
    else:
        pad_top = 0
        pad_bot = 0

    # Number of rows that need to be padded (left and right)
    if not pad_y == output_size_y:
        pad_left = int(pad_y // 2)
        pad_right = int(pad_y // 2)

        # if padding is not equally divided, pad +1 row to the left
        if not pad_y % 2 == 0:
            pad_left += 1
    else:
        pad_left = 0
        pad_right = 0

    return pad_top, pad_bot, pad_left, pad_right


def calculate_subpatches(output_size):
    assert output_size[0] == output_size[1], \
        f'Only square sub-patch size is supported. Mismatch: {output_size[0]} != {output_size[1]}.'

    patch_width, patch_height = IMG_SIZE, IMG_SIZE
    padded_patch_width, padded_patch_height = IMG_SIZE, IMG_SIZE

    # Calculate number of sub-patches in each dimension, check if image needs to be padded
    if (output_size[0] == patch_height) or (output_size[1] == patch_width):
        return patch_height, patch_width, 0, 0, 0, 0

    # Calculating padding offsets if there is a need to
    if (patch_height % output_size[0] != 0) or (patch_width % output_size[1] != 0):
        requires_pad = True
        pad_top, pad_bot, pad_left, pad_right = get_padding_offset(patch_height, patch_width, output_size)

        # patch_height should always match patch_width because we have square images,
        # but doing it like this ensures expandability
        padded_patch_height += (pad_top + pad_bot)
        padded_patch_width += (pad_left + pad_right)
    else:
        pad_top, pad_bot, pad_left, pad_right = 0, 0, 0, 0

    # num_subpatches = (padded_patch_height // output_size[0]) * (padded_patch_width // output_size[1])

    return padded_patch_height, padded_patch_width, pad_top, pad_bot, pad_left, pad_right



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute and export median files for a given S2 dataset')
    parser.add_argument('--data', type=str, default='dataset/netcdf', required=False,
                        help='Path to the netCDF files. Default "dataset/netcdf/".')
    parser.add_argument('--root_coco_path', type=str, default='coco_files/', required=False,
                        help='Root path for coco file. Default "coco_files/".')
    parser.add_argument('--prefix_coco', type=str, default=None, required=False,
                        help='The prefix to use for the COCO file. Default none.')
    parser.add_argument('--out_path', type=str, default='logs/medians', required=False,
                        help='Path to export the medians into. Default "logs/medians/".')
    parser.add_argument('--group_freq', type=str, default='1MS', required=False,
                        help='The frequency to aggregate medians with. Default "1MS".')
    parser.add_argument('--output_size', nargs='+', default=None, required=False,
                        help='The size of the medians. If none given, the output will be of the same size.')
    parser.add_argument('--bands', nargs='+', default=None, required=False,
                        help='The bands to use. Default all.')
    parser.add_argument('--num_workers', type=int, default=8, required=False,
                        help='The number of workers to use for parallel computation. Default 8.')
    args = parser.parse_args()

    data_path = Path(args.data)
    out_path = Path(args.out_path)
    root_coco_path = Path(args.root_coco_path)

    medians_dtype = np.float16
    label_dtype = np.int16

    if args.bands is None:
        bands = BANDS.keys()
    else:
        bands = args.bands

    bands = sorted(bands)

    if args.output_size is None:
        output_size = [366, 366]
    else:
        output_size = [int(x) for x in args.output_size]

    num_buckets = len(pd.date_range(start=f'2020-01-01', end=f'2021-01-01', freq=args.group_freq)) - 1

    padded_patch_height, padded_patch_width, pad_top, pad_bot, pad_left, pad_right = calculate_subpatches(output_size)

    # Create medians folder if it doesn't exist
    out_path.mkdir(exist_ok=True, parents=True)

    print(f'Saving into: {out_path}.')

    print(f'\nStart process...')

    for mode in ['train', 'val', 'test']:
        if args.prefix_coco is not None:
            coco_path = root_coco_path / f'{args.prefix_coco}_coco_{mode}.json'
        else:
            coco_path = root_coco_path / f'coco_{mode}.json'
        coco = COCO(coco_path)

        func = partial(process_patch, out_path, mode, num_buckets, data_path,
                       bands, padded_patch_height, padded_patch_width, medians_dtype,
                       label_dtype, args.group_freq, output_size, pad_top, pad_bot, pad_left, pad_right)

        process_map(func, list(coco.imgs.items()), max_workers=args.num_workers, chunksize=10)

    print('Medians saved.\n')
