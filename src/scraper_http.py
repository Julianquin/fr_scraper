"""
Barrido diario de FincaRaíz por HTTP y por segmentos (bloque E1). Corre EN PARALELO al scraper
actual y NO toca el master, la bitácora ni el dataset activo: escribe solo su propio crudo.

Por qué por segmentos: el buscador solo deja recorrer 10.000 resultados por consulta, y las
búsquedas grandes se saltan avisos (la caché del portal "corre" las páginas). Recorrer entero
cada segmento chico capta ~98% de sus avisos, y con eso: nuevos, cambios de precio y bajas.

Tres pasos (en el CI, tres jobs; ver .github/workflows/barrido_http.yml):
    python -m src.scraper_http plan --grupos 4      # árbol de segmentos repartido en grupos
    python -m src.scraper_http barrer --grupo 0     # recorre los segmentos de un grupo
    python -m src.scraper_http juntar [--subir]     # une, deduplica, mide cobertura, sube a R2

Archivos de trabajo en data/http/ y crudo final en data/raw_http/<fecha>/venta.parquet
(en R2: raw/venta_http/fecha=<fecha>/scrape.parquet y reporte.json). Git ignora data/.
"""
from __future__ import annotations
import argparse
import json
import math
import time
from pathlib import Path

import pandas as pd
import requests

from src.config import BASE_DIR
from src.fincaraiz_http import POR_PAGINA, ULTIMA_PAGINA, aplanar, guardar_parquet, pedir

TRABAJO = BASE_DIR / "data" / "http"
PLAN = TRABAJO / "plan.csv"
SALIDA = BASE_DIR / "data" / "raw_http"
ORDENES = {"barrer": [3], "barrer_dos_ordenes": [1, 0]}   # recientes; o menor y mayor precio


def hoy() -> str:
    return pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d")


# ── 1. Plan ──────────────────────────────────────────────────────────────────
def plan(grupos: int, pausa: float) -> pd.DataFrame:
    """Arma el árbol de segmentos y reparte las hojas en `grupos` de páginas parecidas."""
    from src import segmentos
    arbol = segmentos.construir(pausa)
    hojas = arbol[arbol["accion"] != "partido"].copy()
    hojas["paginas_estimadas"] = hojas["paginas"] * hojas["accion"].map(lambda a: len(ORDENES[a]))
    hojas = hojas.sort_values("paginas_estimadas", ascending=False).reset_index(drop=True)
    carga = [0] * grupos
    asignado = []
    for p in hojas["paginas_estimadas"]:                 # al grupo con menos carga (reparto voraz)
        g = carga.index(min(carga))
        asignado.append(g)
        carga[g] += p
    hojas["grupo"] = asignado
    TRABAJO.mkdir(parents=True, exist_ok=True)
    hojas.to_csv(PLAN, index=False)
    nacional = segmentos.Arbol(pausa).total("venta")
    (TRABAJO / "nacional.json").write_text(json.dumps({"total_nacional": nacional, "fecha": hoy()}))
    print(f"Plan: {len(hojas)} segmentos en {grupos} grupos · páginas por grupo: {carga}")
    return hojas


# ── 2. Barrer ────────────────────────────────────────────────────────────────
def barrer_segmento(sesion: requests.Session, ruta: str, accion: str, pausa: float) -> tuple[list[dict], dict]:
    filas, errores, paginas, declarado = [], 0, 0, 0
    for orden in ORDENES[accion]:
        try:
            data, info, _ = pedir(sesion, ruta, 1, orden=orden)
        except Exception:  # noqa: BLE001
            errores += 1
            continue
        declarado = max(declarado, int(info.get("total") or 0))
        ultima = min(ULTIMA_PAGINA, math.ceil(declarado / POR_PAGINA))
        for n in range(1, ultima + 1):
            if n > 1:
                try:
                    data, _, _ = pedir(sesion, ruta, n, orden=orden)
                except Exception:  # noqa: BLE001  una página caída no frena el segmento
                    errores += 1
                    continue
                time.sleep(pausa)
            paginas += 1
            filas += [aplanar(a) | {"segmento": ruta, "pagina": n} for a in data]
    unicos = len({f["id_inmueble"] for f in filas})
    return filas, {"segmento": ruta, "declarado": declarado,
                   "paginas": paginas, "avisos": len(filas), "unicos": unicos, "errores": errores}


def barrer(grupo: int, pausa: float, solo: int | None = None) -> pd.DataFrame:
    hojas = pd.read_csv(PLAN)
    hojas = hojas[hojas["grupo"] == grupo]
    if solo:
        hojas = hojas.tail(solo)                         # para pruebas: los segmentos más chicos
    sesion = requests.Session()
    filas, informe, t0 = [], [], time.time()
    for i, h in enumerate(hojas.itertuples(), 1):
        f, inf = barrer_segmento(sesion, h.ruta, h.accion, pausa)
        filas += f
        informe.append(inf)
        print(f"[{i}/{len(hojas)}] {h.ruta:48} {inf['unicos']:>6}/{inf['declarado']:<6} "
              f"({inf['unicos'] / max(inf['declarado'], 1):.0%}) · {inf['paginas']} págs · "
              f"{inf['errores']} errores · {(time.time() - t0) / 60:.0f} min", flush=True)
    df = pd.DataFrame(filas)
    TRABAJO.mkdir(parents=True, exist_ok=True)
    guardar_parquet(df, TRABAJO / f"barrido_{grupo}.parquet")
    pd.DataFrame(informe).to_csv(TRABAJO / f"informe_{grupo}.csv", index=False)
    print(f"Grupo {grupo}: {len(df):,} filas · {df['id_inmueble'].nunique() if len(df) else 0:,} avisos únicos · "
          f"{(time.time() - t0) / 60:.1f} min")
    return df


# ── 3. Juntar ────────────────────────────────────────────────────────────────
def juntar(subir: bool) -> dict:
    partes = sorted(TRABAJO.glob("barrido_*.parquet"))
    df = pd.concat([pd.read_parquet(p) for p in partes], ignore_index=True)
    informe = pd.concat([pd.read_csv(p) for p in sorted(TRABAJO.glob("informe_*.csv"))], ignore_index=True)
    filas = len(df)
    df = df.drop_duplicates("id_inmueble", keep="first").reset_index(drop=True)
    fecha = hoy()
    df["fecha_recoleccion"] = fecha

    nacional = json.loads((TRABAJO / "nacional.json").read_text())["total_nacional"]
    reporte = {
        "fecha": fecha,
        "segmentos": int(len(informe)),
        "paginas": int(informe["paginas"].sum()),
        "errores_pagina": int(informe["errores"].sum()),
        "filas_leidas": int(filas),
        "avisos_unicos": int(len(df)),
        "total_nacional_declarado": int(nacional),
        "cobertura": round(len(df) / nacional, 4),
        "segmentos_bajo_90pct": informe.loc[informe["unicos"] < 0.9 * informe["declarado"], "segmento"].tolist(),
    }
    destino = SALIDA / fecha / "venta.parquet"
    destino.parent.mkdir(parents=True, exist_ok=True)
    from src.almacen import COMPRESION
    guardar_parquet(df, destino, compression=COMPRESION)
    (destino.parent / "reporte.json").write_text(json.dumps(reporte, ensure_ascii=False, indent=2))
    print(json.dumps(reporte, ensure_ascii=False, indent=2))
    print(f"💾 {destino.relative_to(BASE_DIR)} ({destino.stat().st_size / 1e6:.1f} MB)")
    if subir:
        from src import almacen
        almacen.subir_archivo(destino, f"raw/venta_http/fecha={fecha}/scrape.parquet")
        almacen.subir_archivo(destino.parent / "reporte.json", f"raw/venta_http/fecha={fecha}/reporte.json")
    return reporte


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Barrido diario de FincaRaíz por segmentos (en paralelo al actual)")
    p.add_argument("--pausa", type=float, default=0.4, help="segundos entre páginas (cortesía)")
    sub = p.add_subparsers(dest="accion", required=True)
    a = sub.add_parser("plan")
    a.add_argument("--grupos", type=int, default=4)
    b = sub.add_parser("barrer")
    b.add_argument("--grupo", type=int, required=True)
    b.add_argument("--solo", type=int, help="solo los N segmentos más chicos del grupo (pruebas)")
    j = sub.add_parser("juntar")
    j.add_argument("--subir", action="store_true")
    args = p.parse_args()

    if args.accion == "plan":
        plan(args.grupos, args.pausa)
    elif args.accion == "barrer":
        barrer(args.grupo, args.pausa, args.solo)
    else:
        juntar(args.subir)
