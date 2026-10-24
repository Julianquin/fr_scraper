"""
Alerta semanal de oportunidades: avisos nuevos por debajo del precio de sus comparables.

Para cada aviso publicado en la semana (vivienda: apartamento, apartaestudio, casa):
  1. Busca comparables: mismo tipo y estrato, área ±30%, ±1 habitación.
     Primero en su barrio oficial; si no alcanzan, a menos de 1 km; luego en su comuna.
  2. Descuento = 1 − (precio/m² del aviso ÷ mediana de los comparables).
  3. Descarta probables errores (área incoherente, descuento extremo, proyectos sobre planos).

Salidas en reports/oportunidades/ (no se versionan; son para revisión humana):
  - <ciudad>_<fecha>.csv           todos los candidatos con métricas y motivo de descarte
  - <ciudad>_<fecha>.html          informe para compartir
  - <ciudad>_<fecha>_whatsapp.txt  mensaje listo para pegar

Uso:
    python -m src.oportunidades                       # Cali, semana que termina en el último dato
    python -m src.oportunidades --fecha 2026-10-23 --umbral 0.2 --top 10
"""
from __future__ import annotations
import argparse
import ast
import html
from pathlib import Path

import numpy as np
import pandas as pd

from src import master
from src.config import BASE_DIR
from src.viz_map import cargar_ciudad

OUT_DIR = BASE_DIR / "reports" / "oportunidades"
TIPOS = ["Apartamento", "Apartaestudio", "Casa"]
MIN_COMPS = 8           # comparables mínimos para calificar un aviso
RADIO_M = 1000          # radio de búsqueda si el barrio no alcanza
TOL_AREA = 0.30         # área ±30%
DESC_VERIFICAR = 0.35   # desde aquí se marca "verificar"
DESC_DESCARTAR = 0.45   # desde aquí se asume error de publicación


def _etiquetas(s) -> list[str]:
    try:
        return list(ast.literal_eval(s)) if isinstance(s, str) else []
    except (ValueError, SyntaxError):
        return []


def cargar(ciudad: str) -> pd.DataFrame:
    """Avisos de vivienda de la ciudad, con zona oficial y first_seen/etiquetas del master."""
    df = cargar_ciudad(ciudad)
    df = df[df["Tipo_propiedad"].isin(TIPOS)].copy()
    m = master.cargar()[["id_inmueble", "first_seen", "Etiquetas"]]
    df = df.merge(m.astype({"id_inmueble": str}), on="id_inmueble", how="left")
    df["first_seen"] = pd.to_datetime(df["first_seen"])
    df["etiquetas"] = df["Etiquetas"].map(_etiquetas)
    for c in ["Estrato", "Habitaciones", "Baños"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.reset_index(drop=True)


def _comparables(c: pd.Series, pool: pd.DataFrame) -> tuple[pd.DataFrame, str | None]:
    """Comparables de `c` y el nivel donde se encontraron ('barrio' | '1 km' | 'comuna').
    Si ningún nivel alcanza MIN_COMPS devuelve los del barrio (para diagnóstico) y None."""
    base = pool[(pool["Tipo_propiedad"] == c["Tipo_propiedad"])
                & (pool["id_inmueble"] != c["id_inmueble"])
                & pool["Area_m2"].between(c["Area_m2"] * (1 - TOL_AREA), c["Area_m2"] * (1 + TOL_AREA))]
    if pd.notna(c["Habitaciones"]):
        base = base[(base["Habitaciones"] - c["Habitaciones"]).abs() <= 1]
    if pd.notna(c["Estrato"]):
        base = base[base["Estrato"] == c["Estrato"]]

    en_barrio = base[base["barrio_oficial"] == c["barrio_oficial"]]
    if len(en_barrio) >= MIN_COMPS:
        return en_barrio, "barrio"
    if pd.notna(c["Latitud"]):
        dy = (base["Latitud"] - c["Latitud"]) * 110_540
        dx = (base["Longitud"] - c["Longitud"]) * 111_320 * np.cos(np.radians(c["Latitud"]))
        cerca = base[np.hypot(dx, dy) <= RADIO_M]
        if len(cerca) >= MIN_COMPS:
            return cerca, "1 km"
    # La comuna solo con estrato conocido: sin él mezclaría zonas muy distintas
    if pd.notna(c["Estrato"]) and pd.notna(c.get("comuna")):
        en_comuna = base[base["comuna"] == c["comuna"]]
        if len(en_comuna) >= MIN_COMPS:
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


def evaluar(df: pd.DataFrame, desde: pd.Timestamp, hasta: pd.Timestamp) -> pd.DataFrame:
    """Califica los avisos nuevos entre `desde` y `hasta` (incluidos) contra todo el pool."""
    nuevos = df[df["first_seen"].between(desde, hasta)]
    filas = []
    for _, c in nuevos.iterrows():
        comps, nivel = _comparables(c, df)
        fila = c.to_dict()
        fila["nivel_comps"] = nivel
        fila["n_comps"] = len(comps)
        if nivel:
            pm = comps["precio_m2"]
            fila.update(med_comps=pm.median(), p25_comps=pm.quantile(.25), p75_comps=pm.quantile(.75))
            fila["descuento"] = 1 - c["precio_m2"] / fila["med_comps"]
        filas.append(fila)
    ev = pd.DataFrame(filas)
    if ev.empty or "descuento" not in ev:
        return ev

    ev["motivo_descarte"] = ev.apply(lambda c: _motivo_descarte(c) if pd.notna(c["descuento"]) else None, axis=1)
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


def _comps_txt(r: pd.Series) -> str:
    donde = {"barrio": "en el barrio", "1 km": "a menos de 1 km",
             "comuna": f"en la comuna {r['comuna']}"}[r["nivel_comps"]]
    return (f"{r['n_comps']} similares {donde}: mediana {_m2(r['med_comps'])} M/m² "
            f"(rango {_m2(r['p25_comps'])}–{_m2(r['p75_comps'])})")


def _señales(r: pd.Series) -> list[str]:
    s = [e for e in ("Bajó de precio", "Negociable") if e in r["etiquetas"]]
    if r["verificar"]:
        s.append("⚠️ Descuento muy alto: verificar área y precio")
    return s


def informe_html(sel: pd.DataFrame, resumen: dict) -> str:
    tarjetas = []
    for i, (_, r) in enumerate(sel.iterrows(), 1):
        señales = "".join(f'<span class="tag">{html.escape(s)}</span>' for s in _señales(r))
        tarjetas.append(f"""
      <article class="card">
        <div class="top"><span class="n">{i}</span>
          <span class="desc">{r['descuento']:.0%} bajo comparables</span></div>
        <h2>{html.escape(_ficha(r))}</h2>
        <p class="zona">{html.escape(_zona(r))}</p>
        <p class="precio"><b>{_millones(r['Precio'])}</b> · {_m2(r['precio_m2'])} M/m²</p>
        <p class="comps">{html.escape(_comps_txt(r))} · confianza {r['confianza'].lower()}</p>
        <div class="tags">{señales}</div>
        <a href="{html.escape(str(r['URL detalle']))}" target="_blank" rel="noopener">Ver aviso ↗</a>
      </article>""")
    return f"""<!doctype html>
<html lang="es"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Oportunidades {resumen['ciudad']} · {resumen['periodo']}</title>
<style>
  :root{{--bg:#f6f7fb;--card:#fff;--ink:#1a1a2e;--muted:#5b6472;--line:#e6e9f0;--ok:#1a7f4b;--okbg:#e6f4ec;--tag:#eef2ff}}
  @media (prefers-color-scheme: dark){{:root{{--bg:#0f1220;--card:#171a2b;--ink:#eef1f7;--muted:#9aa3b2;--line:#262a3d;--ok:#5fd394;--okbg:#14301f;--tag:#232a4a}}}}
  *{{box-sizing:border-box}} body{{margin:0;font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;background:var(--bg);color:var(--ink);line-height:1.5}}
  .wrap{{max-width:760px;margin:0 auto;padding:28px 16px 48px}}
  h1{{margin:0 0 4px;font-size:26px}} .sub{{color:var(--muted);margin:0 0 20px}}
  .resumen{{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:14px 18px;margin-bottom:20px;font-size:14px;color:var(--muted)}}
  .resumen b{{color:var(--ink)}}
  .card{{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:18px;margin-bottom:14px}}
  .top{{display:flex;justify-content:space-between;align-items:center;margin-bottom:8px}}
  .n{{font-weight:700;color:var(--muted)}} .desc{{background:var(--okbg);color:var(--ok);font-weight:700;border-radius:999px;padding:3px 12px;font-size:14px}}
  h2{{margin:0;font-size:18px}} .zona{{margin:2px 0 8px;color:var(--muted)}}
  .precio{{margin:0 0 4px;font-size:16px}} .comps{{margin:0 0 10px;font-size:14px;color:var(--muted)}}
  .tags{{display:flex;gap:6px;flex-wrap:wrap;margin-bottom:10px}} .tag{{background:var(--tag);border-radius:6px;padding:2px 8px;font-size:12.5px}}
  a{{color:#2563eb;font-weight:600;text-decoration:none}}
  footer{{font-size:12.5px;color:var(--muted);margin-top:24px}}
</style></head><body><div class="wrap">
  <h1>Oportunidades de la semana — {resumen['ciudad']}</h1>
  <p class="sub">{resumen['periodo']}</p>
  <div class="resumen">Revisamos <b>{resumen['nuevos']}</b> avisos nuevos de vivienda.
    <b>{resumen['calificados']}</b> tenían comparables suficientes,
    <b>{resumen['bajo_umbral']}</b> estaban más de un {resumen['umbral']:.0%} bajo sus comparables
    y <b>{resumen['descartados']}</b> de ellos se descartaron como probables errores.
    Te mostramos los <b>{len(sel)}</b> mejores.</div>
  {''.join(tarjetas) or '<p>Esta semana no hubo oportunidades que pasaran los filtros.</p>'}
  <footer>Comparables: mismo tipo y estrato, área ±30% y ±1 habitación, en el mismo barrio oficial
    (o a menos de 1 km, o en la comuna). Es una herramienta para priorizar, no un avalúo: verifica siempre el
    inmueble y su documentación. Fuente: avisos publicados en portales; barrios: IDESC – Alcaldía de Cali.</footer>
</div></body></html>"""


def mensaje_whatsapp(sel: pd.DataFrame, resumen: dict) -> str:
    lineas = [f"*Oportunidades de la semana — {resumen['ciudad']}*", resumen["periodo"], ""]
    for i, (_, r) in enumerate(sel.iterrows(), 1):
        lineas += [f"*{i}. {r['descuento']:.0%} bajo comparables* — {_ficha(r)}",
                   f"📍 {_zona(r)}",
                   f"💰 {_millones(r['Precio'])} ({_m2(r['precio_m2'])} M/m²)",
                   f"📊 {_comps_txt(r)} · confianza {r['confianza'].lower()}"]
        lineas += [f"🏷️ {s}" for s in _señales(r)]
        lineas += [f"🔗 {r['URL detalle']}", ""]
    lineas.append("_Herramienta para priorizar, no un avalúo: verifica siempre el inmueble._")
    return "\n".join(lineas)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ciudad", default="Cali")
    p.add_argument("--fecha", default=None, help="Último día de la semana (YYYY-MM-DD); por defecto, el último dato")
    p.add_argument("--dias", type=int, default=7)
    p.add_argument("--umbral", type=float, default=0.20)
    p.add_argument("--top", type=int, default=10)
    args = p.parse_args()

    df = cargar(args.ciudad)
    hasta = pd.Timestamp(args.fecha) if args.fecha else df["first_seen"].max()
    desde = hasta - pd.Timedelta(days=args.dias - 1)
    ev = evaluar(df, desde, hasta)
    if ev.empty or "descuento" not in ev:
        print("No hay avisos nuevos con comparables en el periodo.")
        return
    sel = seleccionar(ev, args.umbral, args.top)

    bajo = ev[ev["descuento"] >= args.umbral]
    meses = ["ene", "feb", "mar", "abr", "may", "jun", "jul", "ago", "sep", "oct", "nov", "dic"]
    resumen = {
        "ciudad": args.ciudad, "umbral": args.umbral,
        "periodo": f"Avisos publicados del {desde.day} {meses[desde.month - 1]} "
                   f"al {hasta.day} {meses[hasta.month - 1]} {hasta.year}",
        "nuevos": len(ev), "calificados": int(ev["nivel_comps"].notna().sum()),
        "bajo_umbral": len(bajo), "descartados": int(bajo["motivo_descarte"].notna().sum()),
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base = OUT_DIR / f"{args.ciudad.lower()}_{hasta:%Y-%m-%d}"
    cols = ["id_inmueble", "first_seen", "Tipo_propiedad", "barrio_oficial", "Barrio", "Estrato",
            "Area_m2", "Habitaciones", "Precio", "precio_m2", "nivel_comps", "n_comps", "med_comps",
            "descuento", "confianza", "verificar", "motivo_descarte", "Etiquetas", "URL detalle"]
    ev[[c for c in cols if c in ev]].to_csv(f"{base}.csv", index=False)
    Path(f"{base}.html").write_text(informe_html(sel, resumen), encoding="utf-8")
    Path(f"{base}_whatsapp.txt").write_text(mensaje_whatsapp(sel, resumen), encoding="utf-8")

    print(f"{resumen['periodo']}: {resumen['nuevos']} nuevos · {resumen['calificados']} con comparables · "
          f"{resumen['bajo_umbral']} bajo {args.umbral:.0%} · {resumen['descartados']} descartados · "
          f"{len(sel)} seleccionados")
    print(f"✅ {base}.html  (+ .csv, _whatsapp.txt)")


if __name__ == "__main__":
    main()
