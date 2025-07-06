# src/ingest.py
import pandas as pd, os, joblib
from pathlib import Path
from src.config import DATA_RAW, DATA_INT
from src.preprocessing import preprocesar_datos_finca_raiz   # lo movemos a preprocessing.py

def run():
    """Lee todos los CSV de data/raw, aplica limpieza y deja un Parquet en data/interim/."""
    files = sorted(DATA_RAW.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No hay CSV en {DATA_RAW}")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df = preprocesar_datos_finca_raiz(df)

    out = DATA_INT / "housing_clean.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"✅ Datos limpios guardados en {out} — {len(df):,} filas")

if __name__ == "__main__":
    run()






