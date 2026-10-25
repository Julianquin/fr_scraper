"""
Exploración de FincaRaíz por HTTP simple (bloque E1). SOLO LECTURA: no toca el master ni R2.

Hallazgos que motivan este módulo (25-oct-2026):
  - Las páginas de listado traen en <script id="__NEXT_DATA__"> los datos ESTRUCTURADOS de
    sus 21 avisos (precio, administración, fecha de publicación, habitaciones, baños, áreas,
    estrato, amenidades, ubicación jerárquica…). No hace falta navegador ni visitar el detalle.
  - El buscador solo deja recorrer los primeros 10.000 resultados de cada consulta (página 476).
    Desde la 477 repite siempre la misma página. Para barrer todo hay que segmentar.

Acciones (resultados en data/analisis/e1/, que no se versiona):
    python -m src.explorar_venta ventana                 # confirma dónde se congela la paginación
    python -m src.explorar_venta segmentos               # total de avisos por ciudad (urls/*.txt)
    python -m src.explorar_venta muestra --paginas 30    # recientes: tiempo, campos, cobertura
"""
from __future__ import annotations
import argparse
import time

import pandas as pd
import requests

from src.config import BASE_DIR
from src.fincaraiz_http import MAX_RESULTADOS, POR_PAGINA, aplanar, pedir

OUT_DIR = BASE_DIR / "data" / "analisis" / "e1"


def ventana(sesion: requests.Session) -> None:
    """Muestra en qué página deja de cambiar el contenido (tope de 10.000 resultados)."""
    ultima = MAX_RESULTADOS // POR_PAGINA
    for n in (ultima - 1, ultima, ultima + 1, ultima + 100):
        data, pi, dt = pedir(sesion, "venta", n)
        print(f"página {n}: primer id {data[0]['id'] if data else '-'} · "
              f"{data[0]['created_at'] if data else '-'} · total {pi.get('total')} · {dt:.2f}s")
        time.sleep(1)


def rutas_ciudad() -> list[str]:
    """Las rutas ciudad/departamento de urls/*.txt, sin el tipo de inmueble."""
    rutas = set()
    for f in (BASE_DIR / "urls").glob("*.txt"):
        for u in f.read_text().split():
            partes = u.split("fincaraiz.com.co/")[-1].strip("/").split("/")
            if partes[0] == "venta" and len(partes) >= 3:
                rutas.add(f"venta/{partes[-2]}/{partes[-1]}")
    return sorted(rutas)


def segmentos(sesion: requests.Session, pausa: float) -> pd.DataFrame:
    """Total de avisos por ciudad: ¿cuáles pasan de 10.000 y hay que partir más?"""
    filas = []
    for ruta in ["venta"] + rutas_ciudad():
        try:
            _, pi, dt = pedir(sesion, ruta, 1)
            filas.append({"ruta": ruta, "total": pi.get("total"), "paginas": pi.get("lastPage"), "segundos": round(dt, 2)})
        except Exception as e:  # noqa: BLE001
            filas.append({"ruta": ruta, "total": None, "error": str(e)[:80]})
        time.sleep(pausa)
    df = pd.DataFrame(filas).sort_values("total", ascending=False)
    df["supera_10k"] = df["total"] > MAX_RESULTADOS
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_DIR / "segmentos.csv", index=False)
    ciudades = df[df["ruta"] != "venta"]
    print(df.head(15).to_string(index=False))
    print(f"\n{len(ciudades)} ciudades · suman {ciudades['total'].sum():,.0f} avisos "
          f"(el total nacional dice {df.loc[df['ruta'] == 'venta', 'total'].iloc[0]:,.0f}) · "
          f"{ciudades['supera_10k'].sum()} pasan de 10.000")
    return df


def muestra(sesion: requests.Session, paginas: int, pausa: float) -> pd.DataFrame:
    """Recorre las primeras `paginas` de /venta (más recientes) y mide tiempo, campos y cobertura."""
    filas, tiempos = [], []
    for n in range(1, paginas + 1):
        data, _, dt = pedir(sesion, "venta", n)
        tiempos.append(dt)
        filas += [aplanar(a) | {"pagina": n} for a in data]
        time.sleep(pausa)
    df = pd.DataFrame(filas)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_DIR / "muestra_venta.parquet", index=False)

    print(f"{paginas} páginas · {len(df):,} avisos · {df['id_inmueble'].nunique():,} únicos · "
          f"{sum(tiempos) / len(tiempos):.2f} s por página (sin contar la pausa de {pausa}s)")
    print(f"Publicados entre {df['fecha_publicacion'].min()} y {df['fecha_publicacion'].max()}")

    print("\nCampos con dato (% de avisos):")
    campos = ["precio", "administracion", "fecha_publicacion", "habitaciones", "banos", "m2_construido",
              "estrato", "piso", "antiguedad", "latitud", "barrio", "comuna", "amenidades", "descripcion"]
    llenos = {c: df[c].notna() & (df[c].astype(str).str.strip() != "") & (df[c].astype(str) != "0") for c in campos}
    print(pd.Series({c: f"{v.mean():.0%}" for c, v in llenos.items()}).to_string())
    proyectos = df["etiquetas"].str.contains('"Proyecto"|"Unidad de proyecto"', regex=True)
    print(f"\nProyectos o unidades de proyecto: {proyectos.mean():.0%}")
    print("Tipos:", df["tipo"].value_counts().head(6).to_dict())

    master = BASE_DIR / "data" / "master" / "listings.parquet"
    if master.exists():
        ids = set(pd.read_parquet(master, columns=["id_inmueble"])["id_inmueble"].astype(str))
        df["en_master"] = df["id_inmueble"].isin(ids)
        print(f"\nCobertura: {df['en_master'].mean():.0%} de estos avisos ya están en el master")
        por_fecha = df.groupby("fecha_publicacion")["en_master"].agg(["size", "mean"]).sort_index(ascending=False)
        print(por_fecha.head(8).rename(columns={"size": "avisos", "mean": "en master"})
              .to_string(formatters={"en master": "{:.0%}".format}))
        fuera = df[~df["en_master"]]
        print("Ciudades de los que faltan:", fuera["ciudad"].value_counts().head(8).to_dict())
    return df


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Exploración de /venta por HTTP (solo lectura)")
    p.add_argument("--pausa", type=float, default=1.0, help="segundos entre peticiones (cortesía)")
    sub = p.add_subparsers(dest="accion", required=True)
    sub.add_parser("ventana")
    sub.add_parser("segmentos")
    m = sub.add_parser("muestra")
    m.add_argument("--paginas", type=int, default=30)
    a = p.parse_args()
    s = requests.Session()
    {"ventana": lambda: ventana(s), "segmentos": lambda: segmentos(s, a.pausa),
     "muestra": lambda: muestra(s, a.paginas, a.pausa)}[a.accion]()
