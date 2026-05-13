import os
from pathlib import Path
import xarray as xr
import rioxarray
from rasterio.enums import Resampling
import netCDF4

INPUT_DIR = Path("/mnt/yacy_1/prod/dat/S4A/")
OUTPUT_DIR = Path("/mnt/yacy_1/prod/ferreyra/S4A_30m/")
TARGET_RESOLUTION = 30

def process_file(input_path, output_path):
    try:
        with netCDF4.Dataset(input_path) as src:
            groups = list(src.groups.keys())
            root_attrs = {k: src.getncattr(k) for k in src.ncattrs()}

        with netCDF4.Dataset(output_path, 'w', format='NETCDF4') as dst:
            dst.setncatts(root_attrs)

        for grp in groups:
            with xr.open_dataset(input_path, group=grp) as ds_grp:
                # Guardar TODOS los atributos antes del reproject
                original_attrs = ds_grp.attrs.copy()
                original_var_attrs = {v: ds_grp[v].attrs.copy() for v in ds_grp.data_vars}

                if grp in ['labels', 'parcels']:
                    resampling = Resampling.nearest
                else:
                    resampling = Resampling.bilinear

                ds_30m = ds_grp.rio.reproject(
                    ds_grp.rio.crs,
                    resolution=TARGET_RESOLUTION,
                    resampling=resampling
                )

                # Restaurar atributos del dataset
                ds_30m.attrs = original_attrs

                # Restaurar atributos de cada variable
                for v in ds_30m.data_vars:
                    if v in original_var_attrs:
                        ds_30m[v].attrs = original_var_attrs[v]

                ds_30m.to_netcdf(output_path, group=grp, mode='a', engine='netcdf4')

        print(f"OK: {output_path.name}")

    except Exception as e:
        print(f"ERROR procesando {input_path.name}: {e}")
        if output_path.exists():
            output_path.unlink()

def main():
    all_nc_files = list(INPUT_DIR.rglob("*.nc"))
    total_files = len(all_nc_files)
    print(f"Se encontraron {total_files} archivos .nc para procesar.")

    for i, nc_file in enumerate(all_nc_files, 1):
        rel_path = nc_file.relative_to(INPUT_DIR)
        out_file = OUTPUT_DIR / rel_path
        out_file.parent.mkdir(parents=True, exist_ok=True)

        if not out_file.exists():
            print(f"[{i}/{total_files}] Procesando {nc_file.name}...")
            process_file(nc_file, out_file)
        else:
            print(f"[{i}/{total_files}] Ya existe: {nc_file.name}")

if __name__ == "__main__":
    main()
