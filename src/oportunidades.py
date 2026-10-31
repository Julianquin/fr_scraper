"""
Alerta semanal de oportunidades, con dos secciones:

  A) Avisos NUEVOS por debajo del precio de sus comparables. Para cada aviso de la semana
     (vivienda: apartamento, apartaestudio, casa):
       1. Busca comparables: mismo tipo y estrato, área ±30%, ±1 habitación.
          Primero en su barrio oficial; si no alcanzan, a menos de 1 km; luego en su comuna.
       2. Descuento = 1 − (precio/m² del aviso ÷ mediana de los comparables).
       3. Descarta probables errores (área incoherente, descuento extremo, proyectos sobre planos).
  B) Avisos que BAJARON DE PRECIO en la semana (bitácora de src/eventos.py), con su
     historial (días publicado, bajadas acumuladas) y dónde quedan frente a sus comparables.

Si hay estado de arriendos en disco (E3), cada aviso trae además su arriendo estimado y su
rentabilidad bruta (src/rentabilidad.py).

Salidas en reports/oportunidades/ (no se versionan; son para revisión humana):
  - <ciudad>_<fecha>.csv           todos los avisos nuevos con métricas y motivo de descarte
  - <ciudad>_<fecha>_bajadas.csv   todas las bajadas de la semana
  - <ciudad>_<fecha>.html          informe para compartir
  - <ciudad>_<fecha>_whatsapp.txt  mensaje listo para pegar

Uso:
    python -m src.oportunidades                       # Cali, semana que termina en el último dato
    python -m src.oportunidades --fecha 2026-10-23 --umbral 0.2 --top 10
    python -m src.oportunidades --fuente master       # con el scraper viejo, para comparar
"""
from __future__ import annotations
import argparse
import ast
import html
from pathlib import Path

import numpy as np
import pandas as pd

from src import eventos, master, zonas
from src.config import BASE_DIR
from src.viz_map import cargar_ciudad

OUT_DIR = BASE_DIR / "reports" / "oportunidades"
TIPOS = ["Apartamento", "Apartaestudio", "Casa"]
MIN_COMPS = 8           # comparables mínimos para calificar un aviso
RADIO_M = 1000          # radio de búsqueda si el barrio no alcanza
TOL_AREA = 0.30         # área ±30%
DESC_VERIFICAR = 0.35   # desde aquí se marca "verificar"
DESC_DESCARTAR = 0.45   # desde aquí se asume error de publicación
BAJA_MIN = 0.03         # bajadas menores a esto no se reportan (ajustes de redondeo)
TOP_BAJADAS = 5
INICIO_DATOS = pd.Timestamp("2026-08-09")   # antes: siembra del master con avisos ya publicados


def _etiquetas(s) -> list[str]:
    try:
        return list(ast.literal_eval(s)) if isinstance(s, str) else []
    except (ValueError, SyntaxError):
        return []


FUENTE = "http"       # "http": barrido diario (src/fuente_http.py) · "master": scraper viejo, para comparar


def _eventos() -> pd.DataFrame:
    if FUENTE == "http":
        from src import fuente_http
        return fuente_http.eventos()
    return eventos.cargar()


def cargar(ciudad: str) -> pd.DataFrame:
    """Avisos de vivienda de la ciudad, con zona oficial y first_seen/etiquetas.
    Con la fuente http, first_seen es la fecha real de publicación y ya viene en el dataset."""
    df = cargar_ciudad(ciudad)
    df = df[df["Tipo_propiedad"].isin(TIPOS)].copy()
    if FUENTE != "http":
        m = master.cargar()[["id_inmueble", "first_seen", "Etiquetas"]]
        df = df.merge(m.astype({"id_inmueble": str}), on="id_inmueble", how="left")
    df["first_seen"] = pd.to_datetime(df["first_seen"])
    df["etiquetas"] = df["Etiquetas"].map(_etiquetas)
    df["zona_nombre"] = df["comuna"].map(zonas.nombres(ciudad)) if zonas.tiene_zonas(ciudad) else None
    for c in ["Estrato", "Habitaciones", "Baños"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.reset_index(drop=True)


def historial(hasta: pd.Timestamp) -> pd.DataFrame:
    """Por inmueble, cambios de precio reales hasta `hasta`: bajadas y variación acumulada.
    Las correcciones de digitación (src/eventos.py) no cuentan."""
    ev = _eventos()
    c = ev[(ev["evento"] == "cambio_precio") & (ev["fecha"] <= f"{hasta:%Y-%m-%d}")].copy()
    c["factor"] = c["precio_despues"] / c["precio_antes"]
    g = c.sort_values("fecha").groupby("id_inmueble")
    return pd.DataFrame({"n_bajadas": g["factor"].apply(lambda f: int((f < 1).sum())),
                         "var_total": g["factor"].prod() - 1})


def _comparables(c: pd.Series, pool: pd.DataFrame, minimo: int = MIN_COMPS) -> tuple[pd.DataFrame, str | None]:
    """Comparables de `c` y el nivel donde se encontraron ('barrio' | '1 km' | 'comuna').
    Si ningún nivel alcanza `minimo` devuelve los del barrio (para diagnóstico) y None.
    También busca arriendos comparables a un aviso de venta (src/rentabilidad.py)."""
    base = pool[(pool["Tipo_propiedad"] == c["Tipo_propiedad"])
                & (pool["id_inmueble"] != c["id_inmueble"])
                & pool["Area_m2"].between(c["Area_m2"] * (1 - TOL_AREA), c["Area_m2"] * (1 + TOL_AREA))]
    if pd.notna(c["Habitaciones"]):
        base = base[(base["Habitaciones"] - c["Habitaciones"]).abs() <= 1]
    if pd.notna(c["Estrato"]):
        base = base[base["Estrato"] == c["Estrato"]]

    en_barrio = base[base["barrio_oficial"] == c["barrio_oficial"]]
    if len(en_barrio) >= minimo:
        return en_barrio, "barrio"
    if pd.notna(c["Latitud"]):
        dy = (base["Latitud"] - c["Latitud"]) * 110_540
        dx = (base["Longitud"] - c["Longitud"]) * 111_320 * np.cos(np.radians(c["Latitud"]))
        cerca = base[np.hypot(dx, dy) <= RADIO_M]
        if len(cerca) >= minimo:
            return cerca, "1 km"
    # La comuna solo con estrato conocido: sin él mezclaría zonas muy distintas
    if pd.notna(c["Estrato"]) and pd.notna(c.get("comuna")):
        en_comuna = base[base["comuna"] == c["comuna"]]
        if len(en_comuna) >= minimo:
            return en_comuna, "comuna"
    return en_barrio, None


def _motivo_descarte(c: pd.Series) -> str | None:
    """Señales de que el 'descuento' es un error de publicación y no una oportunidad."""
    if "Proyecto" in c["etiquetas"]:
        return "proyecto sobre planos"
    if c["descuento"] >= DESC_DESCARTAR:
        return f"descuento extremo ({c['descuento']:.0%}): probable error de área o precio"
    # Una comuna mezcla barrios muy distintos: un gran "descuento" suele ser ese efecto
    if c["nivel_comps"] == "comuna" and c["descuento"] >= DESC_VERIFICAR:
        return f"descuento alto ({c['descuento']:.0%}) con comparables de toda la comuna"
    if pd.notna(c["Habitaciones"]) and c["Habitaciones"] > 0 and c["Area_m2"] / c["Habitaciones"] < 12:
        return "área incoherente con las habitaciones"
    if c["Tipo_propiedad"] != "Casa" and c["Area_m2"] > 500:
        return "área atípica para apartamento"
    return None


def calificar(df: pd.DataFrame, candidatos: pd.DataFrame, hist: pd.DataFrame) -> pd.DataFrame:
    """Compara cada candidato con sus comparables del pool `df` y le añade su historial."""
    filas = []
    for _, c in candidatos.iterrows():
        comps, nivel = _comparables(c, df)
        fila = c.to_dict()
        fila.update(nivel_comps=nivel, n_comps=len(comps), med_comps=np.nan,
                    p25_comps=np.nan, p75_comps=np.nan, descuento=np.nan)
        if nivel:
            pm = comps["precio_m2"]
            fila.update(med_comps=pm.median(), p25_comps=pm.quantile(.25), p75_comps=pm.quantile(.75))
            fila["descuento"] = 1 - c["precio_m2"] / fila["med_comps"]
        filas.append(fila)
    ev = pd.DataFrame(filas)
    if ev.empty:
        return ev

    ev = ev.join(hist, on="id_inmueble")
    ev["n_bajadas"] = ev["n_bajadas"].fillna(0).astype(int)
    ev["motivo_descarte"] = ev.apply(_motivo_descarte, axis=1)
    ev["verificar"] = ev["descuento"].between(DESC_VERIFICAR, DESC_DESCARTAR, inclusive="left")
    ev["confianza"] = np.select(
        [(ev["nivel_comps"] == "barrio") & (ev["n_comps"] >= 15), ev["nivel_comps"] == "comuna"],
        ["Alta", "Baja"], default="Media")
    return ev.sort_values("descuento", ascending=False)


def seleccionar(ev: pd.DataFrame, umbral: float, top: int) -> pd.DataFrame:
    """Las mejores por confianza (Alta → Media → Baja) y, dentro de cada una, por descuento."""
    ok = ev[(ev["descuento"] >= umbral) & ev["motivo_descarte"].isna()]
    rango = ok["confianza"].map({"Alta": 0, "Media": 1, "Baja": 2})
    return ok.assign(_rango=rango).sort_values(["_rango", "descuento"], ascending=[True, False]) \
             .head(top).drop(columns="_rango")


def bajadas(df: pd.DataFrame, hist: pd.DataFrame, desde: pd.Timestamp,
            hasta: pd.Timestamp) -> pd.DataFrame:
    """Avisos de la ciudad que bajaron de precio en la semana (última bajada de cada uno)."""
    ev = _eventos()
    b = ev[(ev["evento"] == "cambio_precio") & (ev["precio_despues"] < ev["precio_antes"])
           & ev["fecha"].between(f"{desde:%Y-%m-%d}", f"{hasta:%Y-%m-%d}")]
    b = b.sort_values("fecha").drop_duplicates("id_inmueble", keep="last")
    b = b.assign(baja=1 - b["precio_despues"] / b["precio_antes"])
    b = b[b["baja"] >= BAJA_MIN][["id_inmueble", "fecha", "precio_antes", "baja"]]
    candidatos = df.merge(b.rename(columns={"fecha": "fecha_baja"}), on="id_inmueble")
    return calificar(df, candidatos, hist)


def seleccionar_bajadas(bj: pd.DataFrame, excluir: pd.Index, top: int) -> pd.DataFrame:
    """Primero las que tras bajar quedan ≥10% bajo sus comparables; luego, por tamaño de la bajada."""
    ok = bj[bj["motivo_descarte"].isna() & ~bj["id_inmueble"].isin(excluir)]
    return ok.assign(_bajo=ok["descuento"] >= 0.10) \
             .sort_values(["_bajo", "baja"], ascending=False).head(top).drop(columns="_bajo")


# ── Presentación ──────────────────────────────────────────────────────────────
def _millones(x: float) -> str:
    return f"${x / 1e6:,.0f} M".replace(",", ".")


def _m2(x: float) -> str:
    return f"{x / 1e6:.1f}".replace(".", ",")


def _ficha(r: pd.Series) -> str:
    hab = f" · {r['Habitaciones']:.0f} hab" if pd.notna(r["Habitaciones"]) else ""
    banos = f" · {r['Baños']:.0f} baños" if pd.notna(r["Baños"]) else ""
    return f"{r['Tipo_propiedad']} · {r['Area_m2']:.0f} m²{hab}{banos}"


def _zona(r: pd.Series) -> str:
    barrio = r["barrio_oficial"] if pd.notna(r["barrio_oficial"]) else r["Barrio"]
    estrato = f" · estrato {r['Estrato']:.0f}" if pd.notna(r["Estrato"]) else ""
    return f"{barrio}{estrato}"


def _en_zona(r: pd.Series) -> str:
    """'en la comuna 2', 'en Chapinero', 'en el corregimiento Pance', 'en la zona de expansión (…)'."""
    nombre = r.get("zona_nombre") if pd.notna(r.get("zona_nombre")) else f"Comuna {r['comuna']}"
    articulo = {"Comuna": "la comuna", "Corregimiento": "el corregimiento", "Zona": "la zona", "Localidad": "la localidad"}
    primera, _, resto = nombre.partition(" ")
    return f"en {articulo[primera]} {resto}" if primera in articulo else f"en {nombre}"


def _comps_txt(r: pd.Series) -> str:
    donde = {"barrio": "en el barrio", "1 km": "a menos de 1 km", "comuna": _en_zona(r)}[r["nivel_comps"]]
    return (f"{r['n_comps']} similares {donde}: mediana {_m2(r['med_comps'])} M/m² "
            f"(rango {_m2(r['p25_comps'])}–{_m2(r['p75_comps'])})")


def _frente_comps(r: pd.Series) -> str | None:
    """'ahora 12% bajo sus comparables' (o sobre, o en línea); None si no hay comparables."""
    if pd.isna(r["descuento"]):
        return None
    if abs(r["descuento"]) < 0.03:
        return "ahora en línea con sus comparables"
    lado = "bajo" if r["descuento"] > 0 else "sobre"
    return f"ahora {abs(r['descuento']):.0%} {lado} sus comparables"


def _historia_txt(r: pd.Series, hasta: pd.Timestamp) -> str | None:
    """'publicado hace 45 días · bajó 2 veces (−8% en total)'; None si no hay nada que contar."""
    partes = []
    if pd.notna(r["first_seen"]) and (hasta - r["first_seen"]).days >= 7:
        partes.append("publicado desde antes de agosto" if r["first_seen"] < INICIO_DATOS
                      else f"publicado hace {(hasta - r['first_seen']).days} días")
    if r["n_bajadas"] > 0:
        veces = "1 vez" if r["n_bajadas"] == 1 else f"{r['n_bajadas']} veces"
        partes.append(f"bajó {veces} ({r['var_total']:+.0%} en total)".replace("-", "−"))
    return " · ".join(partes) or None


def _señales(r: pd.Series) -> list[str]:
    s = [e for e in ("Bajó de precio", "Negociable") if e in r["etiquetas"]]
    if r["verificar"]:
        s.append("⚠️ Descuento muy alto: verificar área y precio")
    return s


def _tarjeta(r: pd.Series, i: int, insignia: str, clase: str, precio: str, lineas: list[str],
             señales: list[str]) -> str:
    extra = "".join(f'<p class="comps">{html.escape(l)}</p>' for l in lineas if l)
    tags = "".join(f'<span class="tag">{html.escape(s)}</span>' for s in señales)
    return f"""
      <article class="card">
        <div class="top"><span class="n">{i}</span><span class="badge {clase}">{insignia}</span></div>
        <h3>{html.escape(_ficha(r))}</h3>
        <p class="zona">{html.escape(_zona(r))}</p>
        <p class="precio">{precio}</p>{extra}
        <div class="tags">{tags}</div>
        <a href="{html.escape(str(r['URL detalle']))}" target="_blank" rel="noopener">Ver aviso ↗</a>
      </article>"""


def informe_html(sel: pd.DataFrame, bj: pd.DataFrame, resumen: dict, hasta: pd.Timestamp) -> str:
    from src.rentabilidad import texto as renta_txt
    nuevas = [_tarjeta(r, i, f"{r['descuento']:.0%} bajo comparables", "ok",
                       f"<b>{_millones(r['Precio'])}</b> · {_m2(r['precio_m2'])} M/m²",
                       [f"{_comps_txt(r)} · confianza {r['confianza'].lower()}", renta_txt(r),
                        _historia_txt(r, hasta)],
                       _señales(r))
              for i, (_, r) in enumerate(sel.iterrows(), 1)]
    bajaron = [_tarjeta(r, i, f"Bajó {r['baja']:.0%}", "baja",
                        f"{_millones(r['precio_antes'])} → <b>{_millones(r['Precio'])}</b> · "
                        f"{_m2(r['precio_m2'])} M/m²",
                        [_frente_comps(r), renta_txt(r), _historia_txt(r, hasta)], [])
               for i, (_, r) in enumerate(bj.iterrows(), 1)]
    nota_renta = (" Arriendo estimado: mediana del canon/m² de arriendos similares publicados (sin amoblados "
                  "ni por días); la rentabilidad bruta es ese canon × 12 ÷ precio, antes de administración, "
                  "impuestos y vacancia." if resumen.get("con_arriendos") else "")
    seccion_bajadas = (f"""
  <h2>Bajaron de precio esta semana</h2>
  <p class="sub">{resumen['bajadas']} avisos de vivienda bajaron {BAJA_MIN:.0%} o más; estas son las mayores bajadas.</p>
  {''.join(bajaron)}""" if bajaron else "")
    return f"""<!doctype html>
<html lang="es"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Oportunidades {resumen['ciudad']} · {resumen['periodo']}</title>
<style>
  :root{{--bg:#f6f7fb;--card:#fff;--ink:#1a1a2e;--muted:#5b6472;--line:#e6e9f0;--ok:#1a7f4b;--okbg:#e6f4ec;--baja:#9a5b00;--bajabg:#fdf1dc;--tag:#eef2ff}}
  @media (prefers-color-scheme: dark){{:root{{--bg:#0f1220;--card:#171a2b;--ink:#eef1f7;--muted:#9aa3b2;--line:#262a3d;--ok:#5fd394;--okbg:#14301f;--baja:#f5b95f;--bajabg:#33260f;--tag:#232a4a}}}}
  *{{box-sizing:border-box}} body{{margin:0;font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;background:var(--bg);color:var(--ink);line-height:1.5}}
  .wrap{{max-width:760px;margin:0 auto;padding:28px 16px 48px}}
  h1{{margin:0 0 4px;font-size:26px}} h2{{margin:32px 0 4px;font-size:21px}} .sub{{color:var(--muted);margin:0 0 20px}}
  .resumen{{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:14px 18px;margin-bottom:20px;font-size:14px;color:var(--muted)}}
  .resumen b{{color:var(--ink)}}
  .card{{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:18px;margin-bottom:14px}}
  .top{{display:flex;justify-content:space-between;align-items:center;margin-bottom:8px}}
  .n{{font-weight:700;color:var(--muted)}} .badge{{font-weight:700;border-radius:999px;padding:3px 12px;font-size:14px}}
  .badge.ok{{background:var(--okbg);color:var(--ok)}} .badge.baja{{background:var(--bajabg);color:var(--baja)}}
  h3{{margin:0;font-size:18px}} .zona{{margin:2px 0 8px;color:var(--muted)}}
  .precio{{margin:0 0 4px;font-size:16px}} .comps{{margin:0 0 4px;font-size:14px;color:var(--muted)}}
  .tags{{display:flex;gap:6px;flex-wrap:wrap;margin:6px 0 10px}} .tag{{background:var(--tag);border-radius:6px;padding:2px 8px;font-size:12.5px}}
  a{{color:#2563eb;font-weight:600;text-decoration:none}}
  footer{{font-size:12.5px;color:var(--muted);margin-top:24px}}
</style></head><body><div class="wrap">
  <h1>Oportunidades de la semana — {resumen['ciudad']}</h1>
  <p class="sub">{resumen['periodo']}</p>
  <div class="resumen">Revisamos <b>{resumen['nuevos']}</b> avisos nuevos de vivienda.
    <b>{resumen['calificados']}</b> tenían comparables suficientes,
    <b>{resumen['bajo_umbral']}</b> estaban más de un {resumen['umbral']:.0%} bajo sus comparables
    y <b>{resumen['descartados']}</b> de ellos se descartaron como probables errores.
    Te mostramos los <b>{len(sel)}</b> mejores. Además, <b>{resumen['bajadas']}</b> avisos bajaron de precio.</div>
  <h2>Nuevos bajo sus comparables</h2>
  {''.join(nuevas) or '<p>Esta semana no hubo oportunidades que pasaran los filtros.</p>'}
  {seccion_bajadas}
  <footer>Comparables: mismo tipo y estrato, área ±30% y ±1 habitación, en el mismo barrio oficial
    (o a menos de 1 km, o en la comuna). Es una herramienta para priorizar, no un avalúo: verifica siempre el
    inmueble y su documentación.{nota_renta} Fuente: avisos publicados en portales; barrios: IDESC – Alcaldía de Cali.</footer>
</div></body></html>"""


def mensaje_whatsapp(sel: pd.DataFrame, bj: pd.DataFrame, resumen: dict, hasta: pd.Timestamp) -> str:
    from src.rentabilidad import texto as renta_txt
    lineas = [f"*Oportunidades de la semana — {resumen['ciudad']}*", resumen["periodo"], "",
              "*🆕 Nuevos bajo sus comparables*", ""]
    for i, (_, r) in enumerate(sel.iterrows(), 1):
        lineas += [f"*{i}. {r['descuento']:.0%} bajo comparables* — {_ficha(r)}",
                   f"📍 {_zona(r)}",
                   f"💰 {_millones(r['Precio'])} ({_m2(r['precio_m2'])} M/m²)",
                   f"📊 {_comps_txt(r)} · confianza {r['confianza'].lower()}"]
        if renta := renta_txt(r):
            lineas.append(f"💵 {renta}")
        if historia := _historia_txt(r, hasta):
            lineas.append(f"🕒 {historia}")
        lineas += [f"🏷️ {s}" for s in _señales(r)]
        lineas += [f"🔗 {r['URL detalle']}", ""]
    if not bj.empty:
        lineas += [f"*📉 Bajaron de precio esta semana* ({resumen['bajadas']} en total, las mayores)", ""]
        for i, (_, r) in enumerate(bj.iterrows(), 1):
            lineas += [f"*{i}. Bajó {r['baja']:.0%}* — {_ficha(r)}",
                       f"📍 {_zona(r)}",
                       f"💰 {_millones(r['precio_antes'])} → {_millones(r['Precio'])} ({_m2(r['precio_m2'])} M/m²)"]
            if frente := _frente_comps(r):
                lineas.append(f"📊 {frente}")
            if renta := renta_txt(r):
                lineas.append(f"💵 {renta}")
            if historia := _historia_txt(r, hasta):
                lineas.append(f"🕒 {historia}")
            lineas += [f"🔗 {r['URL detalle']}", ""]
    lineas.append("_Herramienta para priorizar, no un avalúo: verifica siempre el inmueble._")
    return "\n".join(lineas)


def main() -> None:
    global FUENTE, INICIO_DATOS
    p = argparse.ArgumentParser()
    p.add_argument("--ciudad", default="Cali")
    p.add_argument("--fecha", default=None, help="Último día de la semana (YYYY-MM-DD); por defecto, el último dato")
    p.add_argument("--dias", type=int, default=7)
    p.add_argument("--umbral", type=float, default=0.20)
    p.add_argument("--top", type=int, default=10)
    p.add_argument("--fuente", choices=["http", "master"], default="http",
                   help="datos del barrido diario (http) o del scraper viejo (master), para comparar")
    args = p.parse_args()
    FUENTE = args.fuente
    import src.viz_map as viz_map
    viz_map.FUENTE = args.fuente
    if FUENTE == "http":       # first_seen es la fecha real de publicación: no hay "siembra" que ocultar
        INICIO_DATOS = pd.Timestamp("1900-01-01")

    df = cargar(args.ciudad)
    hasta = pd.Timestamp(args.fecha) if args.fecha else df["first_seen"].max()
    desde = hasta - pd.Timedelta(days=args.dias - 1)
    hist = historial(hasta)

    ev = calificar(df, df[df["first_seen"].between(desde, hasta)], hist)
    bj = bajadas(df, hist, desde, hasta)
    if ev.empty and bj.empty:
        print("No hay avisos nuevos ni bajadas de precio en el periodo.")
        return
    from src import rentabilidad
    con_arriendos = rentabilidad.disponible()
    if con_arriendos:
        arr = rentabilidad.arriendos(args.ciudad)
        ev = rentabilidad.estimar(ev, arr) if not ev.empty else ev
        bj = rentabilidad.estimar(bj, arr) if not bj.empty else bj
    sel = seleccionar(ev, args.umbral, args.top) if not ev.empty else ev
    sel_bj = seleccionar_bajadas(bj, sel.get("id_inmueble", pd.Index([])), TOP_BAJADAS) \
        if not bj.empty else bj

    bajo = ev[ev["descuento"] >= args.umbral] if not ev.empty else ev
    meses = ["ene", "feb", "mar", "abr", "may", "jun", "jul", "ago", "sep", "oct", "nov", "dic"]
    resumen = {
        "ciudad": args.ciudad, "umbral": args.umbral,
        "periodo": f"Avisos publicados del {desde.day} {meses[desde.month - 1]} "
                   f"al {hasta.day} {meses[hasta.month - 1]} {hasta.year}",
        "nuevos": len(ev), "calificados": int(ev["nivel_comps"].notna().sum()) if not ev.empty else 0,
        "bajo_umbral": len(bajo),
        "descartados": int(bajo["motivo_descarte"].notna().sum()) if not bajo.empty else 0,
        "bajadas": len(bj),
        "con_arriendos": con_arriendos,
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base = OUT_DIR / (f"{args.ciudad.lower()}_{hasta:%Y-%m-%d}" + ("_http" if FUENTE == "http" else ""))
    cols = ["id_inmueble", "first_seen", "Tipo_propiedad", "barrio_oficial", "Barrio", "Estrato",
            "Area_m2", "Habitaciones", "Precio", "precio_m2", "nivel_comps", "n_comps", "med_comps",
            "descuento", "confianza", "verificar", "motivo_descarte", "n_bajadas", "var_total",
            "nivel_arr", "n_arr", "confianza_arr", "canon_est", "rentab_bruta", "Etiquetas", "URL detalle"]
    ev[[c for c in cols if c in ev]].to_csv(f"{base}.csv", index=False)
    cols_bj = ["fecha_baja", "precio_antes", "baja"] + cols
    bj[[c for c in cols_bj if c in bj]].to_csv(f"{base}_bajadas.csv", index=False)
    Path(f"{base}.html").write_text(informe_html(sel, sel_bj, resumen, hasta), encoding="utf-8")
    Path(f"{base}_whatsapp.txt").write_text(mensaje_whatsapp(sel, sel_bj, resumen, hasta),
                                           encoding="utf-8")

    print(f"{resumen['periodo']}: {resumen['nuevos']} nuevos · {resumen['calificados']} con comparables · "
          f"{resumen['bajo_umbral']} bajo {args.umbral:.0%} · {resumen['descartados']} descartados · "
          f"{len(sel)} seleccionados · {resumen['bajadas']} bajadas ({len(sel_bj)} en la alerta)")
    print(f"✅ {base}.html  (+ .csv, _bajadas.csv, _whatsapp.txt)")


if __name__ == "__main__":
    main()
