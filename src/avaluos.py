"""
El avalúo catastral de Cali (E9.4): el archivo de Hacienda «Información Histórica de Predios» (datos.cali.gov.co), que
el dueño baja a mano una vez al año (el portal rechaza a los programas y no se disfraza el cliente). En R2:
catastro/avaluos.parquet: la vigencia más reciente de cada predio (2024; 66.689 ya con la 2025) y el avalúo de 2023
(DATOS.md). Cruza con el catastro de la IDESC (src/catastro_cali.py) por `id_predio`: el 97,6 %.

    python -m src.avaluos resumir [--subir]     # por barrio, comuna y ciudad → catastro/avaluos_barrios.parquet
    python -m src.avaluos predio <id_predio>    # el de un predio (el informe de captación)

Medido el 20-nov: el avalúo de un apartamento, 1,53 M/m² construido (mediana); el precio pedido, ~2,8 veces. De 2023 a
2024 subió un 4,5 % parejo (reajuste por decreto): no se cuenta «cuánto subió», sino cuánto se pide frente a él.

En público, siempre por barrio o comuna y de un mismo tipo (apartamentos con apartamentos): nunca un predio. El de un
predio va solo al informe de captación, que pide un agente sobre la vivienda que le encargan.
"""
from __future__ import annotations

import argparse
import json

import pandas as pd

from src import catastro_cali

CARPETA = catastro_cali.CARPETA
ARCHIVO = CARPETA / "avaluos.parquet"
RESUMEN = CARPETA / "avaluos_barrios.parquet"
R2_ARCHIVO, R2_RESUMEN = catastro_cali.R2 + "avaluos.parquet", catastro_cali.R2 + "avaluos_barrios.parquet"
MIN_AREA = 20                   # m² construidos: por debajo, depósitos o datos raros
MIN_PREDIOS = 30                # un barrio y tipo cuenta con al menos estos predios con avalúo
TIPOS = ("apartamento", "casa")


def _traer(clave: str) -> None:
    """De R2 a data/ si no está (con claves de R2; en local, lo que haya)."""
    if catastro_cali._con_r2() and not (catastro_cali.BASE_DIR / "data" / clave).exists():
        from src import almacen
        almacen.traer(clave, catastro_cali.BASE_DIR / "data")


def leer() -> pd.DataFrame | None:
    _traer(R2_ARCHIVO)
    return pd.read_parquet(ARCHIVO) if ARCHIVO.exists() else None


def resumir(predios: pd.DataFrame | None = None) -> pd.DataFrame | None:
    """Avalúo por m² construido (mediana) de apartamentos y casas, por barrio oficial, comuna y la ciudad (`nivel`).
    `predios`: el catastro clasificado (si no, catastro/predios.parquet). None sin avalúos o sin catastro."""
    av = leer()
    if predios is None:
        _traer(catastro_cali.R2 + "predios.parquet")
        predios = catastro_cali.leer("predios")
    if av is None or predios is None:
        return None
    p = predios[["id_predio", "tipo", "comuna", "barrio_oficial"]].assign(id_predio=lambda x: x["id_predio"].astype(str))
    v = av.merge(p, on="id_predio")
    v = v[v["tipo"].isin(TIPOS) & (v["area_construccion"] > MIN_AREA) & (v["avaluo"] > 0)].copy()
    v["avaluo_m2"] = v["avaluo"] / v["area_construccion"]

    def agrupar(clave: list[str], nivel: str) -> pd.DataFrame:
        g = v.dropna(subset=clave).groupby([*clave, "tipo"])
        t = pd.DataFrame({"predios": g.size(), "avaluo_m2": g["avaluo_m2"].median(), "avaluo": g["avaluo"].median(),
                          "vigencia": g["vigencia"].max()}).reset_index()
        return t.assign(nivel=nivel)

    t = pd.concat([agrupar(["comuna", "barrio_oficial"], "barrio"), agrupar(["comuna"], "comuna"),
                   agrupar([], "ciudad")], ignore_index=True)
    t = t[t["predios"] >= MIN_PREDIOS].reset_index(drop=True)
    RESUMEN.parent.mkdir(parents=True, exist_ok=True)
    t.to_parquet(RESUMEN, index=False)
    print(f"Avalúos: {len(v):,} viviendas con avalúo y área · {int((t['nivel'] == 'barrio').sum())} barrios y tipo con "
          f"≥{MIN_PREDIOS} · ciudad: " + ", ".join(f"{r.tipo} {r.avaluo_m2 / 1e6:.2f} M/m²"
                                                    for r in t[t["nivel"] == "ciudad"].itertuples()))
    return t


def subir() -> None:
    from src import almacen
    almacen.subir_archivo(RESUMEN, R2_RESUMEN)


def actualizar(predios: pd.DataFrame) -> None:
    """Rehace el resumen con el catastro recién rehecho (src/catastro_cali.py, una vez al mes) y lo sube: así los
    barrios del avalúo son los mismos que los del catastro. Sin el archivo de avalúos en R2, nada."""
    if resumir(predios) is not None and catastro_cali._con_r2():
        subir()


def por_barrio() -> pd.DataFrame | None:
    """El resumen (data/catastro/avaluos_barrios.parquet, que los flujos bajan de R2 con el catastro). None si no
    está."""
    return pd.read_parquet(RESUMEN) if RESUMEN.exists() else None


def del_predio(id_predio: str) -> dict | None:
    """El avalúo de un predio: avalúo, vigencia, terreno, construcción y área construida. None si no está."""
    av = leer()
    if av is None:
        return None
    x = av[av["id_predio"] == str(id_predio)]
    if x.empty:
        return None
    r = x.iloc[0]
    return {"avaluo": float(r["avaluo"]), "vigencia": int(r["vigencia"]) if pd.notna(r["vigencia"]) else None,
            "valor_terreno": float(r["valor_terreno"]), "valor_construccion": float(r["valor_construccion"]),
            "area_construccion": float(r["area_construccion"]) if pd.notna(r["area_construccion"]) else None}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="accion", required=True)
    r = sub.add_parser("resumir")
    r.add_argument("--subir", action="store_true")
    pr = sub.add_parser("predio")
    pr.add_argument("id_predio")
    a = p.parse_args()
    if a.accion == "resumir":
        if resumir() is not None and a.subir:
            subir()
    else:
        print(json.dumps(del_predio(a.id_predio), ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
