"""
Cifras neutras para las piezas de redes: solo números y nombres de zonas, sin textos, diseño ni marca (el repositorio
es público). Las piezas (vídeos, carruseles, historias) se hacen fuera del repositorio a partir de estas cifras.

Dos productos:
  - `ficha(r)`: una oportunidad en cifras. canal.py la guarda con cada propuesta en canal/estado.json y pasa a
    «publicados» cuando se aprueba: de ahí sale el vídeo de la oportunidad del día.
  - `semana(ciudad)`: el mercado de los últimos 7 días (pulso, precio por m² y rentabilidad por comuna, cifras por
    barrio, el barrio de la semana, mayores bajadas, captación en agregado). canal.yml lo deja cada día, tras
    proponer, en
        contenido/semana.json              la última semana (siempre la misma clave)
        contenido/semana/<hasta>.json      copia con la fecha del último barrido: el historial para comparar meses
        contenido/barrios_semana.json      los barrios de la semana ya elegidos, para no repetirlos
    Sin claves de R2 (pruebas en local), en data/contenido/.

Uso:
    python -m src.contenido semana [--ciudad Cali] [--no-subir]    # --no-subir: imprime el resumen y no guarda
"""
from __future__ import annotations

import argparse
import json
import math
import os

import numpy as np
import pandas as pd

from src.config import BASE_DIR

LOCAL = BASE_DIR / "data" / "contenido"
COMPARABLES = 3                 # similares en la ficha (los mismos que enseña el canal)


def _num(x, dec: int = 0):
    """Número limpio para JSON: None si falta; entero si dec=0."""
    if x is None or (isinstance(x, float) and math.isnan(x)) or (not isinstance(x, str) and pd.isna(x)):
        return None
    return int(round(float(x))) if dec == 0 else round(float(x), dec)


def _fecha(x) -> str | None:
    return None if x is None or pd.isna(x) else f"{pd.Timestamp(x):%Y-%m-%d}"


def _texto(x) -> str | None:
    return None if x is None or (not isinstance(x, str) and pd.isna(x)) else str(x)


def _limpio(v):
    """Lo que json no sabe escribir (numpy, Timestamp, NaN) a tipos de Python."""
    if isinstance(v, dict):
        return {str(k): _limpio(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_limpio(x) for x in v]
    if isinstance(v, (pd.Timestamp, np.datetime64)):
        return _fecha(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        return None if math.isnan(v) else float(v)
    if isinstance(v, np.bool_):
        return bool(v)
    return v


# ── La oportunidad en cifras ──────────────────────────────────────────────────────────────────────────────
def ficha(r: pd.Series) -> dict:
    """Una oportunidad (o una bajada) en cifras: lo que hace falta para contarla en un vídeo o una imagen.
    `r`: una fila de oportunidades.enriquecer (con similares y modelo). Lo que falte queda en None."""
    g = r.get
    comps = g("comps")
    comparables = [] if not isinstance(comps, pd.DataFrame) else [
        {"precio": _num(c.get("Precio")), "area_m2": _num(c.get("Area_m2")), "precio_m2": _num(c.get("precio_m2")),
         "habitaciones": _num(c.get("Habitaciones")), "barrio": _texto(c.get("barrio_oficial")) or _texto(c.get("Barrio")),
         "url": _texto(c.get("URL detalle"))}
        for _, c in comps.head(COMPARABLES).iterrows()]
    bajada = None
    if _num(g("baja"), 3):
        bajada = {"pct": _num(g("baja"), 3), "precio_antes": _num(g("precio_antes")),
                  "fecha": _fecha(g("fecha_baja")), "al_republicar": bool(g("al_republicar") is True)}
    return {
        "id": _texto(g("id_inmueble")), "url": _texto(g("URL detalle")), "fuente": _texto(g("fuente")),
        "ciudad": _texto(g("Ciudad")), "tipo": _texto(g("Tipo_propiedad")),
        "area_m2": _num(g("Area_m2")), "habitaciones": _num(g("Habitaciones")), "banos": _num(g("Baños")),
        "parqueaderos": _num(g("Parqueaderos")), "estrato": _num(g("Estrato")),
        "barrio": _texto(g("barrio_oficial")) or _texto(g("Barrio")), "barrio_aviso": _texto(g("Barrio")),
        "comuna": _texto(g("comuna")), "zona": _texto(g("zona_nombre")),
        "lat": _num(g("Latitud"), 5), "lon": _num(g("Longitud"), 5),
        "publicada": _fecha(g("first_seen")),
        "precio": _num(g("Precio")), "precio_m2": _num(g("precio_m2")),
        "similares": {"n": _num(g("n_comps")), "nivel": _texto(g("nivel_comps")),
                      "mediana_m2": _num(g("med_comps")), "p25_m2": _num(g("p25_comps")), "p75_m2": _num(g("p75_comps")),
                      "descuento": _num(g("descuento"), 3)},
        "modelo": {"estimado": _num(g("modelo_precio")), "p10": _num(g("modelo_p10")), "p90": _num(g("modelo_p90")),
                   "bajo": _num(g("bajo_modelo"), 3)},
        "bajada": bajada,
        "comparables": comparables,
    }


# ── Por barrio, y el barrio de la semana ──────────────────────────────────────────────────────────────────
MIN_BARRIO_SEMANA = 15          # el barrio de la semana sale de los que tienen al menos estas viviendas en venta
MIN_ARRIENDOS_BARRIO = 15       # canon y rentabilidad de un barrio, con al menos estos arriendos
DIAS_BAJADAS = 30               # «bajaron de precio en el último mes»
OLVIDO = 26                     # cuando ya salieron todos, vuelve uno que no haya salido en estas semanas
BARRIOS_SEMANA = "contenido/barrios_semana.json"    # {ciudad: {"2026-W47": {"comuna", "barrio", "precio_m2"}}}
# (campo, decimales) de cada barrio
CAMPOS_BARRIO = [("viviendas", 0), ("precio_m2", 0), ("p25_m2", 0), ("p75_m2", 0), ("precio", 0), ("precio_p25", 0),
                 ("precio_p75", 0), ("area", 0), ("habitaciones", 1), ("apartamentos", 2), ("estrato", 0),
                 ("en_fincaraiz", 0), ("bajaron_mes", 0), ("dias_publicadas", 0), ("arriendos", 0), ("canon_m2", 0),
                 ("rentabilidad", 4)]


def _bajaron(ids: pd.Series, desde: pd.Timestamp, hasta: pd.Timestamp) -> pd.Series:
    """Si cada aviso bajó de precio entre `desde` y `hasta`, contado como las bajadas del canal (≥3 %, sin los
    errores de digitación)."""
    from src import oportunidades as op
    ev = op._eventos()
    b = ev[ev["evento"].isin(["cambio_precio", "republicado"]) & (ev["precio_despues"] < ev["precio_antes"])
           & ev["fecha"].between(f"{desde:%Y-%m-%d}", f"{hasta:%Y-%m-%d}")]
    b = b[(1 - b["precio_despues"] / b["precio_antes"]).between(op.BAJA_MIN, 0.5)]
    return ids.astype(str).isin(set(b["id_inmueble"].astype(str)))


def por_barrio(d: dict) -> pd.DataFrame:
    """Una fila por barrio oficial con al menos MIN_BARRIO viviendas en venta (los del mapa del informe), del más
    caro al más económico por m². Precio, área y tipo: de las viviendas sin repetir de los dos portales; días
    publicadas y bajadas: de Fincaraíz (la fecha de publicación y el historial de precios son suyos)."""
    from src import informe_semanal as s
    clave = ["comuna", "barrio_oficial"]
    u = d["inmuebles"].dropna(subset=clave)
    u = u[~u["barrio_oficial"].str.contains(s.NO_BARRIO, na=False)]
    g = u.groupby(clave)
    t = pd.DataFrame({
        "viviendas": g.size(), "precio_m2": g["precio_m2"].median(),
        "p25_m2": g["precio_m2"].quantile(0.25), "p75_m2": g["precio_m2"].quantile(0.75),
        "precio": g["precio_mediano"].median(), "precio_p25": g["precio_mediano"].quantile(0.25),
        "precio_p75": g["precio_mediano"].quantile(0.75), "area": g["m2"].median(),
        "habitaciones": g["habitaciones"].median(),
        "apartamentos": g["tipo"].agg(lambda x: x.astype(str).str.lower().str.startswith("aparta").mean())})
    t = t[t["viviendas"] >= s.MIN_BARRIO]
    v = d["venta"].dropna(subset=clave)
    v = v.assign(_bajo=_bajaron(v["id_inmueble"], d["hasta"] - pd.Timedelta(days=DIAS_BAJADAS - 1), d["hasta"]))
    gv = v.groupby(clave)
    t["en_fincaraiz"] = gv.size()
    t["bajaron_mes"] = gv["_bajo"].sum()
    t["dias_publicadas"] = gv["dias_pub"].median()
    t["estrato"] = gv["Estrato"].agg(lambda x: x.mode().iloc[0] if x.notna().any() else np.nan)
    ga = d["arriendos_ok"].dropna(subset=clave).groupby(clave)
    t["arriendos"] = ga.size()
    t["canon_m2"] = ga["precio_m2"].median()
    t["canon_m2"] = t["canon_m2"].where(t["arriendos"] >= MIN_ARRIENDOS_BARRIO)
    t["rentabilidad"] = 12 * t["canon_m2"] / t["precio_m2"]
    return t.sort_values("precio_m2", ascending=False)


def candidatos(t: pd.DataFrame) -> pd.DataFrame:
    """Los barrios que pueden ser el de la semana: con muestra y en la ciudad (comunas 01–22 y la zona de expansión,
    81); los sectores de los corregimientos no son barrios."""
    urbano = np.array([c <= "22" or c == "81" for c in t.index.get_level_values("comuna")], dtype=bool)
    return t[urbano & (t["viviendas"] >= MIN_BARRIO_SEMANA)]


def _campos(x: pd.Series) -> dict:
    return {k: _num(x.get(k), dec) for k, dec in CAMPOS_BARRIO}


def _semana_iso(f: pd.Timestamp) -> str:
    a, s, _ = f.isocalendar()
    return f"{a}-W{s:02d}"


def elegir_barrio(t: pd.DataFrame, hasta: pd.Timestamp, ciudad_m2: float, vistos: dict) -> tuple[str, str] | None:
    """El barrio de la semana de `hasta`: el mismo de lunes a domingo, anotado en `vistos` ({"2026-W47": {...}}).
    Uno nuevo: de los `candidatos` que aún no han salido, el de más viviendas, alternando
    uno por encima y otro por debajo del m² de la ciudad (que no salgan seguidos los caros del sur).
    Solo se anota de lunes a miércoles: de jueves a domingo esa semana ya no tiene carrusel (sale los miércoles), y
    anotarlo lo gastaría sin publicarse (pasaba con el primer domingo: Pance). Esos días el elegido es provisional
    y el lunes se vuelve a elegir."""
    semana = _semana_iso(hasta)
    if (x := vistos.get(semana)) and (x["comuna"], x["barrio"]) in t.index:
        return x["comuna"], x["barrio"]
    cand = candidatos(t)
    if cand.empty:
        return None
    antes = sorted(k for k in vistos if k < semana)
    libres = cand[~cand.index.isin([(vistos[k]["comuna"], vistos[k]["barrio"]) for k in antes])]
    if libres.empty:                    # ya salieron todos: vuelve uno que no haya salido hace poco
        libres = cand[~cand.index.isin([(vistos[k]["comuna"], vistos[k]["barrio"]) for k in antes[-OLVIDO:]])]
        libres = libres if len(libres) else cand
    if antes and vistos[antes[-1]].get("precio_m2") is not None:
        caro = vistos[antes[-1]]["precio_m2"] >= ciudad_m2
        otro_lado = libres[(libres["precio_m2"] < ciudad_m2) if caro else (libres["precio_m2"] >= ciudad_m2)]
        libres = otro_lado if len(otro_lado) else libres
    comuna, barrio = libres["viviendas"].idxmax()
    if hasta.weekday() <= 2:
        vistos[semana] = {"comuna": str(comuna), "barrio": barrio, "precio_m2": _num(t.loc[(comuna, barrio), "precio_m2"])}
    return comuna, barrio


def ficha_barrio(t: pd.DataFrame, comuna: str, barrio: str, d: dict) -> dict:
    """El barrio de la semana en cifras: lo suyo, su puesto en la ciudad, su comuna y Cali para comparar, y los
    demás barrios de su comuna."""
    r = t.loc[(comuna, barrio)]
    cand = candidatos(t)
    todas = d["todas"]
    c = todas.loc[comuna] if todas is not None and comuna in todas.index else None
    vecinos = cand.xs(comuna, level="comuna") if comuna in cand.index.get_level_values("comuna") else cand.iloc[0:0]
    return {
        "semana": _semana_iso(d["hasta"]), "comuna": str(comuna), "comuna_nombre": d["nombres"].get(comuna, comuna),
        "barrio": barrio, **_campos(r), "dias_bajadas": DIAS_BAJADAS,
        "puesto_m2": int((cand["precio_m2"] > r["precio_m2"]).sum()) + 1, "de": len(cand),
        "comuna_precio_m2": _num(c["precio_m2"]) if c is not None else None,
        "comuna_rentabilidad": _num(c["rentab"], 4) if c is not None else None,
        "ciudad_precio_m2": _num(d["precio_m2"]), "ciudad_rentabilidad": _num(d["rentab"], 4),
        "su_comuna": [{"barrio": b, "precio_m2": _num(x["precio_m2"]), "viviendas": _num(x["viviendas"])}
                      for b, x in vecinos.iterrows()],
    }


# ── El mes ────────────────────────────────────────────────────────────────────────────────────────────────
# Lo que cambia de un mes a otro. No la mediana del m² por comuna o barrio: en 3 semanas, solo por qué viviendas
# entran y salen, el 15–20 % de las comunas y barrios ya cambia más de un 3 %, mientras el precio pedido de las
# mismas viviendas se movía un −0,01 % (medido el 14-nov). Así que el precio se mide con las MISMAS viviendas (las
# que estuvieron en venta todo el mes) y, aparte, lo que entra (publicadas), lo que sale (retiradas) y cuántas
# bajaron o subieron de precio.
BARRIDO_COMPLETO = pd.Timestamp("2026-10-25")   # desde aquí el barrido ve toda la oferta: antes, el mes no se cuenta
MIN_COMUNA_MES = 300            # una comuna entra en el mes con al menos estas viviendas en venta todo el mes (con
                                # menos, «el 3 % bajó» son 3 o 4 avisos)
NOMBRES_MES = ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto", "septiembre", "octubre",
               "noviembre", "diciembre"]


def mes_anterior(hasta: pd.Timestamp) -> tuple[pd.Timestamp, pd.Timestamp]:
    """El último mes completo antes de `hasta`: (primer día, último día)."""
    fin = pd.Timestamp(hasta).normalize().replace(day=1) - pd.Timedelta(days=1)
    return fin.replace(day=1), fin


def es_semana_del_mes(hasta: pd.Timestamp) -> bool:
    """La semana que lleva el mes: la primera de cada mes (el informe, el canal y el carrusel de ese lunes)."""
    return pd.Timestamp(hasta).day <= 7


def _movimientos(ciudad: str, desde: pd.Timestamp, hasta: pd.Timestamp) -> pd.DataFrame:
    """Viviendas de Fincaraíz de la ciudad publicadas o retiradas en el periodo (columnas `publicada`, `retirada`),
    con su comuna. Del estado completo (también las ya retiradas), no solo de lo que sigue en venta."""
    from src import oportunidades as op, zonas
    from src.consolidar import rutas
    r = rutas("venta")
    a = pd.read_parquet(r.avisos, columns=["id_inmueble", "ciudad", "tipo", "latitud", "longitud", "barrio",
                                           "estado_aviso", "fecha_publicacion"])
    aqui = (zonas.en_area(ciudad, a["latitud"], a["longitud"], a["ciudad"]) if ciudad in zonas.AREAS
            else a["ciudad"].map(zonas.canonica) == ciudad)
    a = a[aqui & a["tipo"].isin(op.TIPOS)].assign(id_inmueble=lambda x: x["id_inmueble"].astype(str))
    publicada = pd.to_datetime(a["fecha_publicacion"], errors="coerce").between(desde, hasta)
    dias = pd.date_range(desde, hasta).strftime("%Y-%m-%d")
    partes = [pd.read_parquet(p, columns=["id_inmueble", "evento"]) for d in dias
              if (p := r.historial / f"fecha={d}.parquet").exists()]
    bajas = set()
    if partes:
        h = pd.concat(partes, ignore_index=True)
        bajas = set(h.loc[h["evento"] == "baja", "id_inmueble"].astype(str))
    # Como src/informe_semanal._retiradas: una baja que volvió a publicarse no fue una retirada
    retirada = a["id_inmueble"].isin(bajas) & a["estado_aviso"].eq("retirado")
    m = a[publicada | retirada].assign(publicada=publicada, retirada=retirada)
    if zonas.tiene_zonas(ciudad):
        m = zonas.asignar_zonas(m.rename(columns={"latitud": "Latitud", "longitud": "Longitud", "barrio": "Barrio"}),
                                ciudad)
    else:
        m = m.assign(comuna=None)
    return m


def _mismas(venta: pd.DataFrame, desde: pd.Timestamp, hasta: pd.Timestamp) -> pd.DataFrame:
    """Las viviendas en venta todo el periodo (publicadas antes y siguen) y cuánto cambió su precio en él
    (`cambio`: producto de sus cambios de precio menos 1; sin las correcciones de digitación)."""
    from src import oportunidades as op
    ev = op._eventos()
    e = ev[(ev["evento"] == "cambio_precio") & ev["fecha"].between(f"{desde:%Y-%m-%d}", f"{hasta:%Y-%m-%d}")]
    factor = (e["precio_despues"] / e["precio_antes"]).groupby(e["id_inmueble"].astype(str)).prod()
    v = venta[venta["first_seen"] <= desde]
    return v.assign(cambio=v["id_inmueble"].astype(str).map(factor).fillna(1.0) - 1)


def _resumen_mes(m: pd.DataFrame, mov: pd.DataFrame) -> dict:
    """Cuenta como bajada o subida un cambio de al menos BAJA_MIN (3 %), como las bajadas del canal."""
    from src import oportunidades as op
    bajo, subio = m["cambio"] <= -op.BAJA_MIN, m["cambio"] >= op.BAJA_MIN
    return {"en_venta_todo": len(m), "bajaron": int(bajo.sum()), "subieron": int(subio.sum()),
            "bajada_mediana": _num(-m.loc[bajo, "cambio"].median(), 3) if bajo.any() else None,
            "cambio_mismas": _num(m["cambio"].mean(), 4) if len(m) else None,
            "publicadas": int(mov["publicada"].sum()), "retiradas": int(mov["retirada"].sum())}


def datos_mes(d: dict, desde: pd.Timestamp, hasta: pd.Timestamp) -> dict:
    """Un mes (de `desde` a `hasta`) en cifras: lo de la ciudad y, por comuna, lo de las que tienen muestra."""
    m = _mismas(d["venta"], desde, hasta)
    mov = _movimientos(d["ciudad"], desde, hasta)
    comunas = []
    for c, x in m.dropna(subset=["comuna"]).groupby("comuna"):
        if len(x) < MIN_COMUNA_MES:
            continue
        comunas.append({"codigo": str(c), "nombre": d["nombres"].get(c, c),
                        **_resumen_mes(x, mov[mov["comuna"] == c])})
    return {"mes": f"{desde:%Y-%m}", "nombre": NOMBRES_MES[desde.month - 1], "desde": desde, "hasta": hasta,
            "completo": bool(desde >= BARRIDO_COMPLETO), **_resumen_mes(m, mov), "comunas": comunas}


def mes(d: dict) -> dict:
    """El último mes completo y, si también se vio entero, el anterior (para comparar). Lo de cada mes se calcula con
    lo que sigue hoy en venta: las del mes anterior que se vendieron después no cuentan (una aproximación)."""
    desde, hasta = mes_anterior(d["hasta"])
    actual = datos_mes(d, desde, hasta)
    a_desde, a_hasta = mes_anterior(desde)
    anterior = datos_mes(d, a_desde, a_hasta) if actual["completo"] and a_desde >= BARRIDO_COMPLETO else None
    return {**actual, "anterior": anterior, "es_semana_del_mes": es_semana_del_mes(d["hasta"]),
            "min_comuna": MIN_COMUNA_MES}


# ── El mercado de la semana ───────────────────────────────────────────────────────────────────────────────
def semana(ciudad: str = "Cali", vistos: dict | None = None) -> dict:
    """El mercado de los últimos 7 días en cifras, del mismo cálculo que el informe semanal. `vistos`: los barrios
    de la semana ya elegidos en la ciudad (BARRIOS_SEMANA); si esta semana no tiene, se elige y se anota ahí."""
    from src import informe_semanal as s
    d = s.datos(ciudad)
    nombre = lambda c: d["nombres"].get(c, c)
    t = d["todas"]
    comunas = [] if t is None or not len(t) else [
        {"codigo": str(c), "nombre": nombre(c), "viviendas": _num(x["viviendas"]), "precio_m2": _num(x["precio_m2"]),
         "dias_publicadas": _num(x["dias_pub"]), "arriendos": _num(x["arriendos"]), "canon_m2": _num(x["canon_m2"]),
         "rentabilidad": _num(x["rentab"], 4), "barrios": x["barrios"]}
        for c, x in t.iterrows()]
    tb = por_barrio(d)
    barrios = [{"comuna": str(c), "barrio": b, **_campos(x)} for (c, b), x in tb.iterrows()]
    elegido = elegir_barrio(tb, d["hasta"], d["precio_m2"], {} if vistos is None else vistos)
    barrio_semana = ficha_barrio(tb, *elegido, d) if elegido else None
    cc = d["captar_comunas"]
    captacion = {
        "dias": list(s.CAPTAR_DIAS), "sobre_similares": list(s.CAPTAR_SOBRE),
        "viejas": d["viejas"], "piden_de_mas": d["captar_n"],
        "por_comuna": [{"codigo": str(c), "nombre": nombre(c), "viejas": int(x["viejas"]), "piden_de_mas": int(x["sobre"])}
                       for c, x in cc.iterrows()],
    }
    sel_bj = d["sel_bj"]
    bajadas = [ficha(r) for _, r in sel_bj.iterrows()] if len(sel_bj) else []
    return _limpio({
        "ciudad": ciudad, "desde": d["desde"], "hasta": d["hasta"], "generado": pd.Timestamp.now(tz="America/Bogota"),
        "mercado": {
            "viviendas": d["viviendas"], "avisos_fincaraiz": d["avisos_fr"], "avisos_metrocuadrado": d["avisos_m2"],
            "en_los_dos": d["en_los_dos"], "publicadas": d["publicadas"], "retiradas": d["retiradas"],
            "bajadas": d["bajadas"], "bajada_mediana": d["bajada_mediana"], "precio_m2": d["precio_m2"],
            "arriendos": d["arriendos"], "canon_mediano": d["canon_mediano"], "rentabilidad": d["rentab"],
        },
        "oportunidades": {"evaluadas": d["evaluadas"], "bajo_similares_20": d["bajo_umbral"]},
        "comunas": comunas, "barrios": barrios, "barrio_semana": barrio_semana, "mes": mes(d), "bajadas": bajadas,
        "captacion": captacion,
    })


def _con_r2() -> bool:
    return all(os.environ.get(v) for v in ("R2_ACCOUNT_ID", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "R2_BUCKET"))


def leer(clave: str) -> dict | None:
    """Lo que guardó `guardar` (de R2, o de data/contenido/ sin claves); None si no existe."""
    if not _con_r2():
        p = LOCAL / clave.removeprefix("contenido/")
        return json.loads(p.read_text()) if p.exists() else None
    from src import almacen
    s3, bucket = almacen._cliente()
    try:
        return json.loads(s3.get_object(Bucket=bucket, Key=clave)["Body"].read())
    except s3.exceptions.NoSuchKey:
        return None


def guardar(clave: str, valor: dict) -> None:
    texto = json.dumps(valor, ensure_ascii=False, indent=1)
    if not _con_r2():
        p = LOCAL / clave.removeprefix("contenido/")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(texto)
        print(f"   (sin R2) {p}")
        return
    from src import almacen
    s3, bucket = almacen._cliente()
    s3.put_object(Bucket=bucket, Key=clave, Body=texto.encode(), ContentType="application/json")
    print(f"   R2: {clave}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("accion", choices=["semana"])
    p.add_argument("--ciudad", default="Cali")
    p.add_argument("--no-subir", action="store_true")
    a = p.parse_args()
    todos = leer(BARRIOS_SEMANA) or {}
    s = semana(a.ciudad, todos.setdefault(a.ciudad, {}))
    m = s["mercado"]
    b = s["barrio_semana"]
    print(f"{a.ciudad} {s['desde']} → {s['hasta']}: {m['viviendas']} viviendas · {m['publicadas']} publicadas · "
          f"{m['bajadas']} bajadas · {len(s['comunas'])} comunas · {len(s['bajadas'])} bajadas destacadas · "
          f"captación {s['captacion']['piden_de_mas']}/{s['captacion']['viejas']} · {len(s['barrios'])} barrios")
    print(f"   barrio de la semana {b['semana']}: {b['barrio']} ({b['viviendas']} viviendas, puesto {b['puesto_m2']} "
          f"de {b['de']} por m²)" if b else "   sin barrio de la semana")
    if not a.no_subir:
        guardar("contenido/semana.json", s)
        guardar(f"contenido/semana/{s['hasta']}.json", s)
        guardar(BARRIOS_SEMANA, todos)


if __name__ == "__main__":
    main()
