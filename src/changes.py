"""
Deriva cambios en el mercado a partir de la historia de snapshots.

Requiere haber corrido antes:
    python -m src.scraper ...   (varias veces, en días distintos)
    python -m src.ingest        (genera data/processed/housing_history.parquet)

Uso:
    python -m src.changes
    python -m src.changes --save     # guarda los 3 CSV en data/processed/
"""
from __future__ import annotations
import argparse
import pandas as pd
from src.config import DATA_PROC


def cargar_historia() -> pd.DataFrame:
    path = DATA_PROC / "housing_history.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"No existe {path}. Corre `python -m src.ingest` primero."
        )
    df = pd.read_parquet(path)
    df["fecha_recoleccion"] = pd.to_datetime(df["fecha_recoleccion"], errors="coerce")
    return df


def cambios_de_precio(hist: pd.DataFrame) -> pd.DataFrame:
    """Un registro por cada vez que el precio de un inmueble cambió entre fechas."""
    h = hist.dropna(subset=["fecha_recoleccion"]).sort_values(
        ["id_inmueble", "fecha_recoleccion"]
    )
    h["precio_anterior"] = h.groupby("id_inmueble")["Precio"].shift(1)
    cambios = h[h["Precio"] != h["precio_anterior"]].dropna(subset=["precio_anterior"])
    cambios = cambios.assign(
        variacion=cambios["Precio"] - cambios["precio_anterior"],
        variacion_pct=(cambios["Precio"] / cambios["precio_anterior"] - 1) * 100,
    )
    return cambios[
        ["id_inmueble", "Título", "fecha_recoleccion",
         "precio_anterior", "Precio", "variacion", "variacion_pct"]
    ].reset_index(drop=True)


def nuevas_y_eliminadas(hist: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Nuevas   : inmuebles cuya PRIMERA aparición no es en la primera corrida.
    Eliminadas: inmuebles que NO aparecen en la corrida (fecha) más reciente.
    """
    h = hist.dropna(subset=["fecha_recoleccion"])
    if h.empty:
        vacio = pd.DataFrame()
        return vacio, vacio

    fechas = sorted(h["fecha_recoleccion"].unique())
    primera, ultima = fechas[0], fechas[-1]

    primera_aparicion = h.groupby("id_inmueble")["fecha_recoleccion"].min()
    ultima_aparicion = h.groupby("id_inmueble")["fecha_recoleccion"].max()

    ids_nuevas = primera_aparicion[primera_aparicion > primera].index
    ids_elim = ultima_aparicion[ultima_aparicion < ultima].index

    ultimo_por_id = (
        h.sort_values("fecha_recoleccion").groupby("id_inmueble", as_index=False).tail(1)
    ).set_index("id_inmueble")

    cols = ["Título", "Precio", "Ciudad", "fecha_recoleccion"]
    nuevas = ultimo_por_id.loc[ids_nuevas, cols].reset_index()
    eliminadas = ultimo_por_id.loc[ids_elim, cols].reset_index().rename(
        columns={"fecha_recoleccion": "ultima_vez_vista"}
    )
    return nuevas, eliminadas


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", action="store_true", help="Guardar CSVs en data/processed/")
    args = parser.parse_args()

    hist = cargar_historia()
    n_fechas = hist["fecha_recoleccion"].nunique(dropna=True)
    if n_fechas < 2:
        print(
            f"⚠️  Solo hay {n_fechas} fecha(s) de recolección en la historia. "
            "Necesitas al menos 2 corridas en días distintos para detectar cambios."
        )

    precios = cambios_de_precio(hist)
    nuevas, eliminadas = nuevas_y_eliminadas(hist)

    print(f"\n📈 Cambios de precio: {len(precios)}")
    print(precios.head(10).to_string(index=False))
    print(f"\n🆕 Nuevas: {len(nuevas)}")
    print(nuevas.head(10).to_string(index=False))
    print(f"\n❌ Eliminadas (no vistas en la última corrida): {len(eliminadas)}")
    print(eliminadas.head(10).to_string(index=False))

    if args.save:
        precios.to_csv(DATA_PROC / "cambios_precio.csv", index=False)
        nuevas.to_csv(DATA_PROC / "nuevas.csv", index=False)
        eliminadas.to_csv(DATA_PROC / "eliminadas.csv", index=False)
        print(f"\n💾 Guardado en {DATA_PROC}/ (cambios_precio.csv, nuevas.csv, eliminadas.csv)")


if __name__ == "__main__":
    main()
