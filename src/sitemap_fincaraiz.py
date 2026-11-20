"""
FincaRaíz por su sitemap oficial de avisos (desde el 20-nov-2026).

Ese día el portal empezó a pedir un desafío del cortafuegos de AWS (HTTP 202 vacío, `x-amzn-waf-action: challenge`)
en sus páginas de resultados a partir de la 2, y el barrido por segmentos (src/scraper_http.py) ya no puede
recorrerlas. No se salta el desafío: se usa lo que el portal publica para los buscadores.

El sitemap de avisos (cde-sitemap-listings-index.xml, anunciado en su robots.txt) trae un archivo por tipo, operación
y departamento, rehecho cada madrugada, con la URL de cada aviso publicado y la fecha de su último cambio (lastmod).
De ahí sale:
  - qué avisos siguen publicados. El que no está cuenta una ausencia en `consolidar`, como en el barrido; las bajas
    siguen pidiendo 5 ausencias seguidas;
  - cuáles hay que leer: los nuevos, los que cambiaron desde la última vez que se vieron (lastmod; medido el 20-nov:
    las bajadas de precio mueven el lastmod) y los que vuelven tras una baja. La página de cada aviso trae el mismo
    JSON que los listados: `aplanar` da las mismas columnas. Lo que no cambió se toma del estado de ayer.

El resultado tiene la forma del barrido (barrido_0.parquet, informe_0.csv y nacional.json en data/http[_arriendo]/),
así que `scraper_http juntar` y `consolidar` siguen igual.

    python -m src.sitemap_fincaraiz [--raiz arriendo] medir                  # cuántos hay que leer, sin leerlos
    python -m src.sitemap_fincaraiz [--raiz arriendo] barrer [--pausa 0.4]   # el barrido del día
"""
from __future__ import annotations

import argparse
import io
import json
import re
import time

import pandas as pd
import requests

from src.config import BASE_DIR
from src.fincaraiz_http import BASE, UA, _NEXT, aplanar, guardar_parquet, pedir
from src.scraper_http import carpetas, hoy

INDICE = f"{BASE}/cde-sitemap-listings-index.xml"
OPERACION = {"venta": "-en-venta-", "arriendo": "-en-alquiler-"}
ESTADO = {"venta": "estado/avisos.parquet", "arriendo": "estado_arriendo/avisos.parquet"}
_URL = re.compile(r"<url>(.*?)</url>", re.S)
_LOC = re.compile(r"<loc>(.*?)</loc>")
_LASTMOD = re.compile(r"<lastmod>(.*?)</lastmod>")
_ID = re.compile(r"/(\d{6,})/?$")
# Los tipos del portal, como van en sus URLs (los mismos 15 que los archivos del sitemap). La cobertura se mide
# frente a la SUMA de lo que declara cada tipo: el contador general (`venta`, 185.611 el 20-nov) cuenta ~3.000 más que
# esa suma (182.591), entradas que no son avisos sueltos (probablemente unidades de proyectos). Medido el 20-nov: el
# sitemap trae el 100 % de la suma por tipos (182.570).
TIPOS = ["apartamentos", "apartaestudios", "bodegas", "cabanas", "casas", "casas-campestres", "casas-lotes",
         "consultorios", "edificios", "fincas", "habitaciones", "locales", "lotes", "oficinas", "parqueaderos"]


class Bloqueado(RuntimeError):
    """El portal pidió el desafío del cortafuegos: no se reintenta ni se sigue (no se salta)."""


def _get(sesion: requests.Session, url: str, reintentos: int = 3) -> requests.Response | None:
    """La página, o None si no existe (404/410). Bloqueado si el portal pide el desafío."""
    for intento in range(1, reintentos + 1):
        try:
            r = sesion.get(url, headers=UA, timeout=30)
        except requests.RequestException:
            if intento == reintentos:
                raise
            time.sleep(5 * intento)
            continue
        if r.headers.get("x-amzn-waf-action"):
            raise Bloqueado(f"{r.status_code} {r.headers.get('x-amzn-waf-action')} en {url}")
        if r.status_code in (404, 410):
            return None
        if r.status_code >= 500 and intento < reintentos:
            time.sleep(5 * intento)
            continue
        r.raise_for_status()
        return r
    return None


# ── 1. El sitemap ────────────────────────────────────────────────────────────
def sitemaps(raiz: str, pausa: float, sesion: requests.Session | None = None) -> pd.DataFrame:
    """(id_inmueble, url, lastmod en UTC, archivo) de cada aviso del sitemap de `raiz`."""
    sesion = sesion or requests.Session()
    indice = _get(sesion, INDICE)
    if indice is None:
        raise RuntimeError(f"No se pudo leer el índice del sitemap ({INDICE})")
    archivos = [u for u in _LOC.findall(indice.text) if OPERACION[raiz] in u]
    filas, fallidos = [], []

    def leer(archivo: str) -> bool:
        try:
            r = _get(sesion, archivo)
        except Bloqueado:
            raise
        except Exception:  # noqa: BLE001
            r = None
        if r is None or "<urlset" not in r.text:
            return False
        nombre = archivo.rsplit("cde-sitemap-listings-", 1)[-1].removesuffix(".xml")
        for bloque in _URL.findall(r.text):
            loc, lm = _LOC.search(bloque), _LASTMOD.search(bloque)
            m = _ID.search(loc.group(1)) if loc else None
            if m:
                filas.append((m.group(1), loc.group(1), lm.group(1) if lm else None, nombre))
        return True

    for k, archivo in enumerate(archivos, 1):
        if not leer(archivo):
            fallidos.append(archivo)
        if k % 25 == 0:
            print(f"   sitemap {k}/{len(archivos)} · {len(filas):,} avisos", flush=True)
        time.sleep(pausa)
    # Un archivo que falta haría contar una ausencia a todos sus avisos (y, cinco días así, darlos de baja sin serlo):
    # se reintenta al final y, si sigue sin bajar, el barrido para. Mejor un día sin datos que bajas falsas.
    if fallidos:
        time.sleep(30)
        fallidos = [a for a in fallidos if not leer(a)]
    if fallidos:
        raise RuntimeError(f"{len(fallidos)} archivos del sitemap no se pudieron leer (p. ej. {fallidos[0]}): "
                           "no se sigue para no contar ausencias falsas")
    df = pd.DataFrame(filas, columns=["id_inmueble", "url", "lastmod", "archivo"])
    df["lastmod"] = pd.to_datetime(df["lastmod"], utc=True, errors="coerce", format="ISO8601")
    print(f"Sitemap de {raiz}: {len(archivos)} archivos · {len(df):,} avisos · {df['id_inmueble'].nunique():,} únicos")
    return df.drop_duplicates("id_inmueble").reset_index(drop=True)


# ── 2. Qué hay que leer ──────────────────────────────────────────────────────
def estado_y_columnas(raiz: str) -> tuple[pd.DataFrame, list[str]]:
    """El estado de ayer (R2) y las columnas del crudo (las del último scrape.parquet de esa raíz)."""
    import pyarrow.parquet as pq
    from src import almacen
    s3, bucket = almacen._cliente()
    estado = pd.read_parquet(io.BytesIO(s3.get_object(Bucket=bucket, Key=ESTADO[raiz])["Body"].read()))
    claves = sorted(o["Key"] for pagina in s3.get_paginator("list_objects_v2").paginate(
        Bucket=bucket, Prefix=f"raw/{raiz}_http/") for o in pagina.get("Contents", [])
        if o["Key"].endswith("/scrape.parquet"))
    columnas = pq.read_schema(io.BytesIO(s3.get_object(Bucket=bucket, Key=claves[-1])["Body"].read())).names
    estado["id_inmueble"] = estado["id_inmueble"].astype(str)
    return estado, columnas


def repartir(sm: pd.DataFrame, estado: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """leer: nuevos, cambiados desde que se vieron por última vez o que vuelven tras una baja; igual: los que se toman
    del estado tal cual; ausentes: activos en el estado que el sitemap ya no trae."""
    e = estado.set_index("id_inmueble")
    en_estado = sm["id_inmueble"].isin(e.index)
    visto = pd.to_datetime(sm["id_inmueble"].map(e["ultima_vez"]), errors="coerce").dt.tz_localize("UTC")
    activo = sm["id_inmueble"].map(e["estado_aviso"]).eq("activo")
    # Cambió el mismo día que se vio o después (lastmod en UTC; el barrido corría a las 00:00–02:00 de Colombia)
    cambio = sm["lastmod"].isna() | (sm["lastmod"] >= visto)
    leer = ~en_estado | ~activo | cambio
    activos = estado.loc[estado["estado_aviso"].eq("activo"), "id_inmueble"]
    return {"leer": sm[leer], "igual": sm[~leer],
            "ausentes": estado[estado["id_inmueble"].isin(set(activos) - set(sm["id_inmueble"]))]}


def medir(raiz: str, pausa: float) -> dict[str, pd.DataFrame]:
    sm = sitemaps(raiz, pausa)
    estado, _ = estado_y_columnas(raiz)
    partes = repartir(sm, estado)
    nuevos = (~partes["leer"]["id_inmueble"].isin(estado["id_inmueble"])).sum()
    print(f"{raiz}: en el sitemap {len(sm):,} · leer {len(partes['leer']):,} (nuevos {nuevos:,}) · del estado "
          f"{len(partes['igual']):,} · activos ayer que ya no están {len(partes['ausentes']):,} "
          f"(de {estado['estado_aviso'].eq('activo').sum():,} activos)")
    return partes


# ── 3. El barrido del día ────────────────────────────────────────────────────
def ficha(sesion: requests.Session, url: str) -> dict | None:
    """El JSON del aviso (el mismo de los listados), o None si ya no está publicado."""
    r = _get(sesion, url)
    if r is None:
        return None
    m = _NEXT.search(r.text)
    if not m:
        return None
    d = (json.loads(m.group(1)).get("props") or {}).get("pageProps", {}).get("data")
    return d if d and d.get("active", True) and not d.get("deleted") else None


def barrer(raiz: str, pausa: float, max_fichas: int | None = None) -> pd.DataFrame:
    trabajo, _ = carpetas(raiz)
    sesion = requests.Session()
    t0 = time.time()
    sm = sitemaps(raiz, pausa, sesion)
    estado, columnas = estado_y_columnas(raiz)
    partes = repartir(sm, estado)
    leer = partes["leer"].head(max_fichas) if max_fichas else partes["leer"]
    print(f"A leer: {len(leer):,} avisos (nuevos, cambiados o que vuelven) · del estado: {len(partes['igual']):,}",
          flush=True)

    filas, errores, idos, archivo_de = [], 0, 0, dict(zip(sm["id_inmueble"], sm["archivo"]))
    for k, a in enumerate(leer.itertuples(), 1):
        try:
            d = ficha(sesion, a.url)
        except Bloqueado:
            raise
        except Exception:  # noqa: BLE001 — un aviso caído no frena el resto (cuenta como error)
            errores += 1
            d = None
        if d is None:
            idos += 1
        else:
            filas.append(aplanar(d) | {"segmento": f"sitemap/{a.archivo}"})
        if k % 500 == 0:
            print(f"   {k:,}/{len(leer):,} leídos · {errores} errores · {(time.time() - t0) / 60:.0f} min", flush=True)
        time.sleep(pausa)

    igual = estado.set_index("id_inmueble").loc[partes["igual"]["id_inmueble"]].reset_index()
    igual["segmento"] = igual["id_inmueble"].map(archivo_de).map(lambda x: f"sitemap/{x}")
    df = pd.concat([pd.DataFrame(filas), igual[[c for c in columnas if c in igual.columns]]], ignore_index=True)
    df = df[[c for c in columnas if c in df.columns]]

    informe = (sm.assign(leido=sm["id_inmueble"].isin(df["id_inmueble"]))
               .groupby("archivo").agg(declarado=("id_inmueble", "size"), unicos=("leido", "sum")).reset_index()
               .rename(columns={"archivo": "segmento"}))
    informe["segmento"] = "sitemap/" + informe["segmento"]
    informe["paginas"] = informe["segmento"].map(
        pd.Series([f"sitemap/{x}" for x in leer["archivo"]]).value_counts()).fillna(0).astype(int)
    informe["avisos"] = informe["unicos"]
    informe["errores"] = 0
    if errores:
        informe.loc[0, "errores"] = errores

    por_tipo = {}                                                     # la página 1 no pide el desafío
    for tipo in TIPOS:
        try:
            por_tipo[tipo] = int(pedir(sesion, f"{raiz}/{tipo}", 1)[1].get("total") or 0)
        except Exception:  # noqa: BLE001 — un tipo sin avisos en esa raíz (la ruta no existe) cuenta 0
            por_tipo[tipo] = 0
        time.sleep(pausa)
    nacional = sum(por_tipo.values())
    general = int(pedir(sesion, raiz, 1)[1].get("total") or 0)
    trabajo.mkdir(parents=True, exist_ok=True)
    for viejo in trabajo.glob("barrido_*.parquet"):                   # que juntar no mezcle otro barrido
        viejo.unlink()
    for viejo in trabajo.glob("informe_*.csv"):
        viejo.unlink()
    guardar_parquet(df, trabajo / "barrido_0.parquet")
    informe[["segmento", "declarado", "paginas", "avisos", "unicos", "errores"]].to_csv(trabajo / "informe_0.csv",
                                                                                        index=False)
    (trabajo / "nacional.json").write_text(json.dumps({"total_nacional": nacional, "por_tipo": por_tipo,
                                                       "contador_general": general, "fecha": hoy(), "fuente": "sitemap"}))
    print(f"Sitemap {raiz}: {len(df):,} avisos ({len(filas):,} leídos, {len(igual):,} del estado) · {idos:,} ya no "
          f"publicados · {errores} errores · declarado por tipos {nacional:,} (contador general {general:,}) · "
          f"{(time.time() - t0) / 60:.1f} min")
    return df


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="FincaRaíz por su sitemap oficial de avisos")
    p.add_argument("--raiz", choices=tuple(OPERACION), default="venta")
    p.add_argument("--pausa", type=float, default=0.4, help="segundos entre peticiones (cortesía)")
    sub = p.add_subparsers(dest="accion", required=True)
    sub.add_parser("medir")
    b = sub.add_parser("barrer")
    b.add_argument("--max-fichas", type=int, help="leer como mucho N avisos (pruebas)")
    a = p.parse_args()
    try:
        medir(a.raiz, a.pausa) if a.accion == "medir" else barrer(a.raiz, a.pausa, a.max_fichas)
    except Bloqueado as e:
        raise SystemExit(f"❌ FincaRaíz pidió el desafío del cortafuegos también aquí: {e}. No se salta: parar.")
