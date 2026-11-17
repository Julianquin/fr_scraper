"""
Las compraventas registradas (E9.5, BACKLOG.md): cuántas hubo cada año en Cali y en su área metropolitana, y
cuántas hipotecas.

Fuente: «Registro de transacciones inmobiliarias en Colombia» del IGAC con los actos de la Superintendencia de
Notariado y Registro, en datos.gov.co (`7y2j-43cv`, abierto, sin clave): cada acto registrado en un folio de
matrícula, de 2015 a 2023 (publicado en abril de 2025). Se agrega en el servidor por municipio, año y acto:
    compraventas    acto 125 (casas, apartamentos, parqueaderos, locales, lotes: el registro no dice qué es)
    hipotecas       acto 219, hipoteca abierta sin límite de cuantía (la del crédito de vivienda)
Sin ubicación dentro del municipio, sin área y sin tipo de inmueble; el valor de la escritura (desde 2021) no es
el precio real. Por eso aquí solo se cuentan actos.

Lo que no sirve (revisado el 17-nov): «Actos de Naturaleza Jurídica Registral» de la SNR (`3icx-ndrp`, hasta
2026): por departamento, no por municipio, y con huecos enormes desde 2024 (el Valle: 112 compraventas en julio
de 2024 y 118 en noviembre, frente a 5.000–7.000 de costumbre).

Salida (R2 `transacciones/`, en local data/transacciones/):
    municipios.parquet   municipio, año, compraventas, hipotecas
    meta.json            cuándo se bajó y hasta qué año llega

Uso:
    python -m src.transacciones actualizar [--si-viejo 90]
    python -m src.transacciones resumen
"""
from __future__ import annotations

import argparse
import json
import os

import pandas as pd

from src.config import BASE_DIR
from src.credito import MUNICIPIOS, _consulta

DATASET = "7y2j-43cv"
ACTOS = {"125": "compraventas", "219": "hipotecas"}
CARPETA = BASE_DIR / "data" / "transacciones"
R2 = "transacciones/"
ARCHIVOS = ("municipios.parquet", "meta.json")


def bajar() -> pd.DataFrame:
    """Compraventas e hipotecas por municipio (Cali y su área) y año."""
    codigos = ", ".join(f"'{c}'" for c in MUNICIPIOS)
    actos = ", ".join(ACTOS)
    t = _consulta(DATASET, "divipola, year_radica, cod_natujur, count(*) as n",
                  f"divipola in ({codigos}) AND cod_natujur in ({actos})", "divipola, year_radica, cod_natujur")
    t["acto"] = t["cod_natujur"].astype(str).map(ACTOS)
    t["n"] = pd.to_numeric(t["n"]).astype(int)
    m = (t.pivot_table(index=["divipola", "year_radica"], columns="acto", values="n", aggfunc="sum", fill_value=0)
         .reset_index().rename(columns={"divipola": "municipio", "year_radica": "año"}))
    m["año"] = m["año"].astype(int)
    m["nombre"] = m["municipio"].map(MUNICIPIOS)
    for c in ACTOS.values():
        if c not in m:
            m[c] = 0
    return m[["municipio", "nombre", "año", "compraventas", "hipotecas"]].sort_values(["municipio", "año"])


def _con_r2() -> bool:
    return all(os.environ.get(v) for v in ("R2_ACCOUNT_ID", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "R2_BUCKET"))


def _traer() -> None:
    from src import almacen
    for f in ARCHIVOS:
        almacen.traer(R2 + f, BASE_DIR / "data")


def actualizar(si_viejo: int | None = None) -> None:
    """Baja, guarda y sube a R2. Con `si_viejo`, solo si lo de R2 tiene más de esos días (el IGAC publica un año
    nuevo de vez en cuando); si no, baja lo de R2. Si datos.gov.co falla, también: se usa lo de la vez anterior."""
    if si_viejo is not None and _con_r2():
        _traer()
        if (m := meta()) and (pd.Timestamp.now() - pd.Timestamp(m["bajado"])).days <= si_viejo:
            print(f"✅ Transacciones del {m['bajado']} (hasta {m['hasta']}): nada que hacer")
            return
    try:
        t = bajar()
    except Exception as e:                  # noqa: BLE001 — sin la fuente, lo de la vez anterior
        print(f"⚠️  datos.gov.co: {type(e).__name__}; se usa lo de la vez anterior")
        raise
    CARPETA.mkdir(parents=True, exist_ok=True)
    t.to_parquet(CARPETA / "municipios.parquet", index=False)
    m = {"bajado": f"{pd.Timestamp.now():%Y-%m-%d}", "desde": int(t["año"].min()), "hasta": int(t["año"].max()),
         "fuente": "IGAC y SNR, datos.gov.co 7y2j-43cv"}
    (CARPETA / "meta.json").write_text(json.dumps(m, ensure_ascii=False, indent=1))
    print(f"✅ {len(t)} filas · {m['desde']}–{m['hasta']}")
    if _con_r2():
        from src import almacen
        for f in ARCHIVOS:
            almacen.subir_archivo(CARPETA / f, R2 + f)


def leer() -> pd.DataFrame | None:
    p = CARPETA / "municipios.parquet"
    return pd.read_parquet(p) if p.exists() else None


def meta() -> dict | None:
    p = CARPETA / "meta.json"
    return json.loads(p.read_text()) if p.exists() else None


def resumen(t: pd.DataFrame | None = None) -> dict | None:
    """Por municipio: las compraventas de cada año, las hipotecas del último, y cuánto cambió frente al primero y
    frente al pico. Para las cifras de la semana (src/contenido.py) y el carrusel «Cali, año a año»."""
    t = leer() if t is None else t
    if t is None or t.empty:
        return None
    desde, hasta = int(t["año"].min()), int(t["año"].max())
    municipios = []
    for cod, x in t.groupby("municipio"):
        x = x.set_index("año").sort_index()
        ultimo, primero = x.loc[hasta], x.loc[desde]
        pico = int(x["compraventas"].idxmax())
        municipios.append({
            "codigo": cod, "nombre": MUNICIPIOS.get(cod, cod),
            "compraventas": {str(a): int(v) for a, v in x["compraventas"].items()},
            "hipotecas": {str(a): int(v) for a, v in x["hipotecas"].items()},
            "ultimo": int(ultimo["compraventas"]), "veces": round(float(ultimo["compraventas"] / primero["compraventas"]), 2)
            if primero["compraventas"] else None,
            "pico": pico, "pico_compraventas": int(x.loc[pico, "compraventas"]),
            "hipotecas_parte": round(float(ultimo["hipotecas"] / ultimo["compraventas"]), 3)
            if ultimo["compraventas"] else None})
    return {"desde": desde, "hasta": hasta, "fuente": "IGAC y SNR",
            "municipios": sorted(municipios, key=lambda m: -m["ultimo"])}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("accion", choices=["actualizar", "resumen"])
    p.add_argument("--si-viejo", type=int, help="solo si lo de R2 tiene más de estos días")
    a = p.parse_args()
    try:                                    # en local, las claves vienen de .env
        from src.almacen import _cargar_env
        _cargar_env()
    except ImportError:
        pass
    if a.accion == "actualizar":
        actualizar(a.si_viejo)
    r = resumen()
    if r:
        for m in r["municipios"]:
            print(f"{m['nombre']:11} {m['ultimo']:>7,} en {r['hasta']} (×{m['veces']} frente a {r['desde']}; pico "
                  f"{m['pico_compraventas']:,} en {m['pico']}) · hipotecas {m['hipotecas_parte']:.0%} de las compraventas")


if __name__ == "__main__":
    main()
