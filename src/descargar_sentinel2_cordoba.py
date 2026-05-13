import requests
import time
import json
import zipfile
import io
from pathlib import Path

USERNAME  = "micaflorlauuu13@gmail.com"
PASSWORD  = "florAcopernicus1."
BANDS     = ["B02", "B03", "B04", "B08"]
OUT_DIR   = Path("/mnt/yacy_1/prod/ferreyra/sentinel2_cordoba")
PRODUCTS_CACHE = Path("/home1/ferreyra/prod_ia_unne/data/productos_encontrados.json")
TMP_ZIP   = Path("/tmp/tmp_sentinel_download.zip")

TOKEN_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
DOWNLOAD  = "https://download.dataspace.copernicus.eu/odata/v1/Products"

PROXIES = {
    "http":  "http://10.40.1.254:3128",
    "https": "http://10.40.1.254:3128",
}

def get_token():
    r = requests.post(TOKEN_URL, data={
        "client_id":  "cdse-public",
        "grant_type": "password",
        "username":   USERNAME,
        "password":   PASSWORD,
    }, proxies=PROXIES)
    r.raise_for_status()
    return r.json()["access_token"]

def make_session(token):
    s = requests.Session()
    s.headers.update({"Authorization": f"Bearer {token}"})
    s.proxies.update(PROXIES)
    return s

def ya_descargado(tile, date):
    for band in BANDS:
        dest = OUT_DIR / tile / date / f"{tile}_{date}_{band}.jp2"
        if not dest.exists():
            return False
    return True

def descargar_y_extraer(session, pid, tile, date):
    url = f"{DOWNLOAD}({pid})/$value"
    dest_dir = OUT_DIR / tile / date
    dest_dir.mkdir(parents=True, exist_ok=True)

    print(f"  Descargando zip...", flush=True)
    try:
        with session.get(url, stream=True, timeout=600) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            downloaded = 0
            with open(TMP_ZIP, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 512):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        pct = downloaded / total * 100
                        print(f"  {pct:.0f}%  ({downloaded/1024/1024:.0f} MB)", end="\r", flush=True)
        print(flush=True)
    except Exception as e:
        print(f"  ERROR descarga: {e}", flush=True)
        if TMP_ZIP.exists():
            TMP_ZIP.unlink()
        return

    print(f"  Extrayendo bandas...", flush=True)
    try:
        with zipfile.ZipFile(TMP_ZIP) as zf:
            for name in zf.namelist():
                fname = Path(name).name
                for band in BANDS:
                    if f"_{band}_" in fname or f"_{band}." in fname:
                        if "R10m" in name and fname.endswith(".jp2"):
                            dest = dest_dir / f"{tile}_{date}_{band}.jp2"
                            if not dest.exists():
                                dest.write_bytes(zf.read(name))
                                mb = dest.stat().st_size / 1024 / 1024
                                print(f"    OK {dest.name}  ({mb:.1f} MB)", flush=True)
    except Exception as e:
        print(f"  ERROR extraccion: {e}", flush=True)
    finally:
        if TMP_ZIP.exists():
            TMP_ZIP.unlink()
            print(f"  Zip borrado.", flush=True)

def main():
    OUT_DIR.mkdir(exist_ok=True)

    with open(PRODUCTS_CACHE) as f:
        products = json.load(f)
    print(f"Cache cargada: {len(products)} productos.", flush=True)

    print("Obteniendo token...", flush=True)
    token = get_token()
    session = make_session(token)
    token_time = time.time()

    for i, prod in enumerate(products):
        if time.time() - token_time > 480:
            print("  Refrescando token...", flush=True)
            token = get_token()
            session = make_session(token)
            token_time = time.time()

        name  = prod["Name"]
        pid   = prod["Id"]
        parts = name.split("_")
        date  = parts[2][:8]
        tile  = [p for p in parts if p.startswith("T") and len(p) == 6]
        tile  = tile[0] if tile else "UNKNOWN"

        print(f"\n[{i+1}/{len(products)}] {tile} - {date}", flush=True)

        if ya_descargado(tile, date):
            print(f"  Ya existe, saltando.", flush=True)
            continue

        descargar_y_extraer(session, pid, tile, date)

    print("\nDescarga completa.", flush=True)

if __name__ == "__main__":
    main()
