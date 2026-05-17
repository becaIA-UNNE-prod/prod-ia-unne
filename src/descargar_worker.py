import sys
sys.argv[1]  # worker_id

import requests
import time
import json
import zipfile
from pathlib import Path

WORKER_ID = int(sys.argv[1])
USERNAME  = "micaflorlauuu13@gmail.com"
PASSWORD  = "florAcopernicus1."
BANDS     = ["B02", "B03", "B04", "B08"]
OUT_DIR   = Path("/mnt/yacy_1/prod/ferreyra/sentinel2_cordoba")
CACHE     = Path(f"/home1/ferreyra/prod_ia_unne/data/productos_worker_{WORKER_ID}.json")
TMP_ZIP   = Path(f"/tmp/tmp_sentinel_worker_{WORKER_ID}.zip")
TOKEN_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
DOWNLOAD  = "https://download.dataspace.copernicus.eu/odata/v1/Products"
PROXIES   = {"http": "http://10.40.1.254:3128", "https": "http://10.40.1.254:3128"}

def get_token():
    r = requests.post(TOKEN_URL, data={"client_id":"cdse-public","grant_type":"password","username":USERNAME,"password":PASSWORD}, proxies=PROXIES)
    r.raise_for_status()
    return r.json()["access_token"]

def make_session(token):
    s = requests.Session()
    s.headers.update({"Authorization": f"Bearer {token}"})
    s.proxies.update(PROXIES)
    return s

def ya_descargado(tile, date):
    return all((OUT_DIR / tile / date / f"{tile}_{date}_{b}.jp2").exists() for b in BANDS)

def descargar(session, pid, tile, date):
    url = f"{DOWNLOAD}({pid})/$value"
    dest_dir = OUT_DIR / tile / date
    dest_dir.mkdir(parents=True, exist_ok=True)
    try:
        with session.get(url, stream=True, timeout=600) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            downloaded = 0
            with open(TMP_ZIP, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024*512):
                    f.write(chunk)
                    downloaded += len(chunk)
            print(f"  W{WORKER_ID} zip OK ({downloaded/1024/1024:.0f}MB)", flush=True)
    except Exception as e:
        print(f"  W{WORKER_ID} ERROR descarga: {e}", flush=True)
        if TMP_ZIP.exists(): TMP_ZIP.unlink()
        return
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
                                print(f"  W{WORKER_ID} OK {dest.name}", flush=True)
    except Exception as e:
        print(f"  W{WORKER_ID} ERROR extraccion: {e}", flush=True)
    finally:
        if TMP_ZIP.exists(): TMP_ZIP.unlink()

def main():
    with open(CACHE) as f:
        products = json.load(f)
    print(f"Worker {WORKER_ID}: {len(products)} productos", flush=True)
    token = get_token()
    session = make_session(token)
    token_time = time.time()
    for i, prod in enumerate(products):
        if time.time() - token_time > 480:
            token = get_token()
            session = make_session(token)
            token_time = time.time()
        name = prod["Name"]
        pid  = prod["Id"]
        parts = name.split("_")
        date  = parts[2][:8]
        tile  = [p for p in parts if p.startswith("T") and len(p) == 6]
        tile  = tile[0] if tile else "UNKNOWN"
        print(f"W{WORKER_ID} [{i+1}/{len(products)}] {tile} - {date}", flush=True)
        if ya_descargado(tile, date):
            print(f"  W{WORKER_ID} ya existe", flush=True)
            continue
        descargar(session, pid, tile, date)
    print(f"Worker {WORKER_ID} terminado.", flush=True)

if __name__ == "__main__":
    main()
