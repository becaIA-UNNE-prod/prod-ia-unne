import os
import sys
from pathlib import Path
from datetime import datetime
import xarray as xr
import rioxarray
from rasterio.enums import Resampling
import netCDF4
import warnings

# Ignorar warnings molestos de xarray/pyproj durante el procesamiento masivo
warnings.filterwarnings("ignore")

# --- CONFIGURACIÓN ---
INPUT_DIR = Path("/mnt/yacy_1/prod/dat/S4A/")
OUTPUT_DIR = Path("/mnt/yacy_1/prod/ferreyra/S4A_30m/")
TARGET_RESOLUTION = 30
# Si rioxarray no logra deducir el CRS de las coordenadas, usará este por defecto.
# (Ejemplo: EPSG:32631 corresponde a la zona UTM del tile 31TBF).
FALLBACK_CRS = "EPSG:32631"

def log(mensaje):
    """Imprime mensajes con timestamp para un logueo limpio con nohup."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {mensaje}")
    sys.stdout.flush() # Fuerza la escritura al log inmediatamente

def process_file(input_path, output_path):
    try:
        # 1. Leer grupos y atributos globales del archivo original
        with netCDF4.Dataset(input_path) as src:
            groups = list(src.groups.keys())
            root_attrs = {k: src.getncattr(k) for k in src.ncattrs()}

        # 2. Crear el archivo de salida con los atributos globales
        with netCDF4.Dataset(output_path, 'w', format='NETCDF4') as dst:
            dst.setncatts(root_attrs)

        # 3. Procesar cada grupo de forma independiente
        for grp in groups:
            with xr.open_dataset(input_path, group=grp) as ds_grp:
                # Resguardar atributos originales
                original_attrs = ds_grp.attrs.copy()
                original_var_attrs = {v: ds_grp[v].attrs.copy() for v in ds_grp.data_vars}

                # Resguardar tipos de datos y valores nulos
                dtypes = {v: ds_grp[v].dtype for v in ds_grp.data_vars}

                # Chequeo y asignación de CRS si rioxarray no lo detecta automáticamente
                if ds_grp.rio.crs is None:
                    ds_grp = ds_grp.rio.write_crs(FALLBACK_CRS)

                # Definir método de remuestreo
                if grp in ['labels', 'parcels']:
                    resampling = Resampling.nearest
                else:
                    resampling = Resampling.bilinear

                # Aplicar reproyección / reescalado a 30m
                ds_30m = ds_grp.rio.reproject(
                    ds_grp.rio.crs,
                    resolution=TARGET_RESOLUTION,
                    resampling=resampling
                )

                # Restaurar estrictamente los tipos de datos (vital para los enteros)
                for v in ds_30m.data_vars:
                    orig_dtype = dtypes[v]
                    orig_nodata = original_var_attrs.get(v, {}).get('_FillValue', 0)

                    # Rellenar posibles NaNs generados en los bordes y forzar tipo
                    ds_30m[v] = ds_30m[v].fillna(orig_nodata).astype(orig_dtype)

                # Limpiar variables inyectadas por rioxarray para mantener estructura exacta
                if 'spatial_ref' in ds_30m.variables:
                    ds_30m = ds_30m.drop_vars('spatial_ref')

                # Restaurar atributos del grupo y de cada variable
                ds_30m.attrs = original_attrs
                for v in ds_30m.data_vars:
                    if v in original_var_attrs:
                        ds_30m[v].attrs = original_var_attrs[v]

                # Anexar el grupo al NetCDF final
                ds_30m.to_netcdf(output_path, group=grp, mode='a', engine='netcdf4')

        log(f"✅ OK: {output_path.name}")

    except Exception as e:
        log(f"❌ ERROR procesando {input_path.name}: {e}")
        # Limpieza: Si falla a la mitad, borramos el archivo corrupto
        if output_path.exists():
            output_path.unlink()

def main():
    log("Iniciando escaneo de directorio...")
    all_nc_files = list(INPUT_DIR.rglob("*.nc"))
    total_files = len(all_nc_files)

    log(f"Se encontraron {total_files} archivos .nc para procesar.")
    log("-" * 50)

    for i, nc_file in enumerate(all_nc_files, 1):
        # Mantener la estructura de carpetas (Año / Tile / archivo.nc)
        rel_path = nc_file.relative_to(INPUT_DIR)
        out_file = OUTPUT_DIR / rel_path
        out_file.parent.mkdir(parents=True, exist_ok=True)

        if not out_file.exists():
            # Descomentar la siguiente línea si querés un log por CADA inicio de archivo
            # log(f"[{i}/{total_files}] Procesando {nc_file.name}...")
            process_file(nc_file, out_file)
        else:
            log(f"⏭️ [{i}/{total_files}] Saltando (Ya existe): {nc_file.name}")

    log("-" * 50)
    log("Proceso finalizado por completo.")

if __name__ == "__main__":
    main()
