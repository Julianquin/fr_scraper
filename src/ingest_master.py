"""
Upsert de lo scrapeado en el store maestro incremental.

Lee los CSV de data/raw/ (las corridas del scraper) y los inserta/actualiza en
data/master/listings.parquet por `id_inmueble`. Pensado para correr una sola vez
en el job `merge` (con los CSV de todos los departamentos ya consolidados), o en
local tras `python -m src.scraper --incremental`.

Uso:
    python -m src.ingest_master
"""
from pathlib import Path
import pandas as pd

from src.config import DATA_RAW
from src import master as M


def run() -> None:
    files = sorted(Path(DATA_RAW).rglob("*.csv"))
    if not files:
        print(f"⚠️  No hay CSV en {DATA_RAW}; nada que upsertar.")
        return
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

    # Los CSV viejos no traen id_inmueble → lo derivamos de la URL (último segmento),
    # idéntico a lo que hace el scraper/limpieza. Así se pueden sembrar en el master.
    if "URL detalle" in df.columns:
        df["id_inmueble"] = (
            df["URL detalle"].astype(str).str.rstrip("/").str.split("/").str[-1]
        )

    run_date = pd.Timestamp.now().strftime("%Y-%m-%d")
    stats = M.upsert(df, run_date)
    print(f"🗄️  Master actualizado → {M.MASTER_PATH}")
    print(f"   +{stats['nuevos']} nuevos · {stats['actualizados']} actualizados · "
          f"{stats['total']} inmuebles en total")


if __name__ == "__main__":
    run()
