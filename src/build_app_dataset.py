"""
Genera el dataset curado y liviano que consume el dashboard
(data/app/housing_clean.parquet) a partir del parquet limpio de `ingest`.

Uso:
    python -m src.ingest            # produce data/processed/housing_clean.parquet
    python -m src.build_app_dataset # produce data/app/housing_clean.parquet
"""
from pathlib import Path
import pandas as pd
from src.config import DATA_PROC, BASE_DIR

# Solo las columnas que el dashboard necesita (mantiene el parquet pequeño)
APP_COLS = ["id_inmueble", "Título", "URL detalle", "Precio", "Area_m2",
            "Habitaciones", "Baños", "Tipo_propiedad", "Ciudad",
            "Departamento", "Barrio", "Latitud", "Longitud"]


def run() -> None:
    src = DATA_PROC / "housing_clean.parquet"
    if not src.exists():
        raise FileNotFoundError(f"No existe {src}. Corre `python -m src.ingest` primero.")
    df = pd.read_parquet(src)
    out = BASE_DIR / "data" / "app" / "housing_clean.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df[[c for c in APP_COLS if c in df.columns]].to_parquet(out, index=False)
    print(f"✅ {len(df):,} filas → {out}")


if __name__ == "__main__":
    run()
