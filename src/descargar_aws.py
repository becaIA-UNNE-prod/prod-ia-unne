import sys
import requests
from pathlib import Path
import json

BANDS = ["B02", "B03", "B04", "B08"]
OUT_DIR = Path("/mnt/yacy_1/prod/ferreyra/sentinel2_cordoba")
WORKER_ID = int(sys.argv[1]) if len(sys.argv) > 1 else 0
N_WORKERS = int(sys.argv[2]) if len(sys.argv) > 2 else 4
CACHE = Path("data/productos_aws.json")

def descargar_banda(url, dest_path):
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        r = requests.get(url, stream=True, timeout=180)
        r.raise_for_status()
        with open(dest_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=1024*512):
                f.write(chunk)
        return True
    except Exception as e:
        print(f"  ERROR: {e}", flush=True)
        return False

def main():
    with open(CACHE) as f:
        all_items = json.load(f)

    mis_items = [item for i, item in enumerate(all_items) if i % N_WORKERS == WORKER_ID]
    print(f"Worker {WORKER_ID}: {len(mis_items)} productos", flush=True)

    for i, item in enumerate(mis_items):
        tile = item["tile"]
        date = item["date"]
        print(f"W{WORKER_ID} [{i+1}/{len(mis_items)}] {tile} - {date}", flush=True)

        for band, url in item["bands"].items():
            dest = OUT_DIR / tile / date / f"{tile}_{date}_{band}.jp2"
            if dest.exists():
                print(f"  W{WORKER_ID} ya existe {band}", flush=True)
                continue
            print(f"  W{WORKER_ID} descargando {band}...", flush=True)
            ok = descargar_banda(url, dest)
            if ok:
                print(f"  W{WORKER_ID} OK {dest.name}", flush=True)

    print(f"Worker {WORKER_ID} terminado.", flush=True)

if __name__ == "__main__":
    main()
