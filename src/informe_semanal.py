"""
Informe semanal para los agentes: el pulso del mercado de vivienda de una ciudad y las oportunidades de la
semana, en un PDF de dos páginas fácil de leer y de reenviar.

  Página 1 · Pulso del mercado
    - viviendas en venta sin repetidos entre FincaRaíz y Metrocuadrado (src/unificar.py); publicadas,
      retiradas y que bajaron de precio en la semana (de FincaRaíz: Metrocuadrado todavía no tiene una
      semana de historia); arriendos de vivienda y su canon
    - por comuna: viviendas en venta, precio mediano por m², canon mediano por m² y rentabilidad bruta
      típica (canon por m² × 12 ÷ precio por m², con las medianas de la comuna)
    - el resumen en tres frases
  Página 2 · Oportunidades: las de la alerta (src/oportunidades.py, mismos criterios): avisos nuevos bajo
    sus comparables y las mayores bajadas de precio, con el enlace a cada aviso.

Es una estimación del precio de mercado, no un avalúo (Ley 1673 de 2013). Sin fotos ni datos de los
anunciantes: solo cifras y enlaces a los avisos públicos.

Uso:
    python -m src.informe_semanal [--ciudad Cali] [--fecha AAAA-MM-DD] [--enviar admin|agentes]
    → reports/informe/<ciudad>_<fecha>.pdf (y el .html del que sale); con --enviar, por Telegram al
      administrador (TELEGRAM_ADMIN_CHAT_ID, para revisarlo) o a los agentes (TELEGRAM_CHAT_ID)
"""
from __future__ import annotations
import argparse
import html
import re

import pandas as pd

from src import oportunidades as op, rentabilidad, unificar, viz_map, zonas
from src.config import BASE_DIR

OUT_DIR = BASE_DIR / "reports" / "informe"
UMBRAL = 0.20               # descuento mínimo frente a los comparables (el de la alerta)
TOP = 6                     # oportunidades nuevas en la página 2
TARJETAS = 8                # tarjetas que caben en la página 2: las bajadas llenan el resto (al menos 2)
MIN_VIVIENDAS = 50          # una comuna entra en la tabla con al menos estas viviendas en venta
MAX_FILAS = 20              # y, si son más, las que tienen más oferta (la tabla cabe en la página 1)
MIN_ARRIENDOS = 15          # y su canon, con al menos estos arriendos
BARRIOS_POR_COMUNA = 2
PREFIJO = re.compile(r"^(urbanizaci[oó]n|parcelaci[oó]n(es)?|ciudadela|conjunto residencial)\s+", re.I)
MESES = ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto", "septiembre", "octubre",
         "noviembre", "diciembre"]


# ── Datos ─────────────────────────────────────────────────────────────────────

def _retiradas(ciudad: str, desde: pd.Timestamp, hasta: pd.Timestamp) -> int:
    """Viviendas de la ciudad con la baja confirmada en la semana (historial de src/consolidar.py)."""
    from src.consolidar import rutas
    r = rutas("venta")
    todos = pd.read_parquet(r.avisos, columns=["id_inmueble", "ciudad", "tipo"])
    viv = todos.loc[(todos["ciudad"] == ciudad) & todos["tipo"].isin(op.TIPOS), "id_inmueble"].astype(str)
    dias = pd.date_range(desde, hasta).strftime("%Y-%m-%d")
    partes = [pd.read_parquet(p) for d in dias if (p := r.historial / f"fecha={d}.parquet").exists()]
    if not partes:
        return 0
    h = pd.concat(partes, ignore_index=True)
    return int(((h["evento"] == "baja") & h["id_inmueble"].astype(str).isin(viv)).sum())


def _por_comuna(inm: pd.DataFrame, arr: pd.DataFrame) -> pd.DataFrame:
    """Viviendas en venta, precio mediano por m², canon mediano por m² y rentabilidad bruta típica."""
    g = inm.dropna(subset=["comuna"]).groupby("comuna")
    t = pd.DataFrame({"viviendas": g.size(), "precio_m2": g["precio_m2"].median()})
    ga = arr.dropna(subset=["comuna"]).groupby("comuna")
    t["arriendos"] = ga.size()
    t["canon_m2"] = ga["precio_m2"].median().where(t["arriendos"] >= MIN_ARRIENDOS)
    t["rentab"] = 12 * t["canon_m2"] / t["precio_m2"]
    barrios = (inm.dropna(subset=["comuna", "barrio_oficial"]).groupby("comuna")["barrio_oficial"]
               .agg(lambda s: ", ".join(PREFIJO.sub("", b) for b in s.value_counts().index[:BARRIOS_POR_COMUNA])))
    t["barrios"] = barrios
    t = t[t["viviendas"] >= MIN_VIVIENDAS].nlargest(MAX_FILAS, "viviendas")
    return t.sort_values("precio_m2", ascending=False)


def datos(ciudad: str = "Cali", fecha: str | None = None, dias: int = 7) -> dict:
    """Todo lo que lleva el informe de la semana que termina en `fecha` (por defecto, el último dato)."""
    op.FUENTE = viz_map.FUENTE = "http"
    op.INICIO_DATOS = pd.Timestamp("1900-01-01")       # con el barrido, first_seen es la publicación real
    venta = op.cargar(ciudad)
    hasta = pd.Timestamp(fecha) if fecha else venta["first_seen"].max().normalize()
    desde = hasta - pd.Timedelta(days=dias - 1)

    # Oportunidades: el mismo cálculo de la alerta
    hist = op.historial(hasta)
    nuevos = venta[venta["first_seen"].between(desde, hasta)]
    ev = op.calificar(venta, nuevos, hist) if len(nuevos) else nuevos
    bj = op.bajadas(venta, hist, desde, hasta)
    arr = rentabilidad.arriendos(ciudad)
    ev = rentabilidad.estimar(ev, arr) if len(ev) else ev
    bj = rentabilidad.estimar(bj, arr) if len(bj) else bj
    sel = op.seleccionar(ev, UMBRAL, TOP) if len(ev) else ev
    sel_bj = op.seleccionar_bajadas(bj, sel.get("id_inmueble", pd.Index([])), max(2, TARJETAS - len(sel))) \
        if len(bj) else bj

    # Oferta sin repetidos (los dos portales), con su comuna y un precio por m² saneado como el de venta
    avisos, inm, _ = unificar.unificar(ciudad)
    inm = zonas.asignar_zonas(inm.rename(columns={"latitud": "Latitud", "longitud": "Longitud", "barrio": "Barrio"}))
    inm["precio_m2"] = inm["precio_mediano"] / inm["m2"].where(inm["m2"] > 10)
    lo, hi = venta["precio_m2"].quantile([0.01, 0.99])
    con_precio = inm[inm["precio_m2"].between(lo, hi) & (inm["precio_mediano"] > 1e7)]
    arr_ok = arr[arr["motivo_exclusion"].isna()]
    tabla = _por_comuna(con_precio, arr_ok)

    precio_m2 = con_precio["precio_m2"].median()
    canon_m2 = arr_ok["precio_m2"].median()
    return {
        "ciudad": ciudad, "desde": desde, "hasta": hasta,
        "viviendas": len(inm), "avisos_fr": int((avisos["portal"] == "fincaraiz").sum()),
        "avisos_m2": int((avisos["portal"] == "metrocuadrado").sum()),
        "solo_m2": int((inm["portales"] == "metrocuadrado").sum()),
        "en_los_dos": int((inm["portales"] == "fincaraiz+metrocuadrado").sum()),
        "publicadas": len(nuevos), "retiradas": _retiradas(ciudad, desde, hasta),
        "bajadas": len(bj), "bajada_mediana": float(bj["baja"].median()) if len(bj) else None,
        "arriendos": len(arr_ok), "canon_mediano": float(arr_ok["Precio"].median()),
        "precio_m2": float(precio_m2), "rentab": float(12 * canon_m2 / precio_m2),
        "tabla": tabla,
        "evaluadas": len(ev), "con_comparables": int(ev["nivel_comps"].notna().sum()) if len(ev) else 0,
        "bajo_umbral": int((ev["descuento"] >= UMBRAL).sum()) if len(ev) else 0,
        "sel": sel, "sel_bj": sel_bj,
    }


# ── Presentación ──────────────────────────────────────────────────────────────

def _n(x: float) -> str:
    return f"{x:,.0f}".replace(",", ".")


def _pct(x: float, dec: int = 0) -> str:
    return f"{x * 100:.{dec}f}%".replace(".", ",")


def _mill(x: float, dec: int = 1) -> str:
    """Millones de pesos: 4,5 M; los grandes sin decimales."""
    return (f"{x / 1e6:,.0f}" if x >= 1e8 else f"{x / 1e6:.{dec}f}").replace(",", "X").replace(".", ",") \
        .replace("X", ".") + " M"


def _miles(x: float) -> str:
    return f"{x / 1e3:,.0f}".replace(",", ".") + " mil"


def _fecha(d: pd.Timestamp, año: bool = True) -> str:
    return f"{d.day} de {MESES[d.month - 1]}" + (f" de {d.year}" if año else "")


def _periodo(desde: pd.Timestamp, hasta: pd.Timestamp) -> str:
    inicio = (str(desde.day) if (desde.month, desde.year) == (hasta.month, hasta.year)
              else _fecha(desde, año=desde.year != hasta.year))
    return f"Semana del {inicio} al {_fecha(hasta)}"


def _barrio(r: pd.Series) -> str:
    """El barrio oficial; si no hay, el primero del aviso (algunos anunciantes listan todos los conjuntos)."""
    b = r["barrio_oficial"] if pd.notna(r.get("barrio_oficial")) else str(r["Barrio"]).split(",")[0].strip()
    partes = [PREFIJO.sub("", b)]
    if pd.notna(r.get("Estrato")):
        partes.append(f"estrato {r['Estrato']:.0f}")
    if pd.notna(r.get("comuna")):
        partes.append(_comuna(r["comuna"]))
    return " · ".join(partes)


def _comuna(c: str) -> str:
    return f"Comuna {int(c)}" if str(c).isdigit() else str(c)


def frases(d: dict) -> list[str]:
    t = d["tabla"]
    cara = t.index[0]
    rent = t["rentab"].dropna()
    lineas = [
        f"En {d['ciudad']} hay <b>{_n(d['viviendas'])} viviendas en venta</b> sin contar repetidos: "
        f"{_n(d['avisos_fr'])} avisos en FincaRaíz y {_n(d['avisos_m2'])} en Metrocuadrado. "
        f"{_n(d['en_los_dos'])} están en los dos portales y {_n(d['solo_m2'])} solo en Metrocuadrado.",
        f"Esta semana se publicaron <b>{_n(d['publicadas'])}</b> viviendas, se confirmó el retiro de "
        f"<b>{_n(d['retiradas'])}</b> y <b>{_n(d['bajadas'])}</b> bajaron de precio"
        + (f" (rebaja mediana del {_pct(d['bajada_mediana'])})." if d["bajada_mediana"] else "."),
        f"La más cara por m² es la <b>{_comuna(cara)}</b> ({html.escape(str(t.loc[cara, 'barrios']))}), "
        f"con {_mill(t.loc[cara, 'precio_m2'])}/m²"
        + (f"; la de mayor rentabilidad bruta típica, la <b>{_comuna(rent.idxmax())}</b> "
           f"({html.escape(str(t.loc[rent.idxmax(), 'barrios']))}), con {_pct(rent.max(), 1)}." if len(rent) else "."),
    ]
    return lineas


def _tile(etiqueta: str, valor: str, nota: str = "") -> str:
    return (f'<div class="tile"><div class="t-label">{etiqueta}</div><div class="t-value">{valor}</div>'
            f'<div class="t-note">{nota}</div></div>')


def _fila(c: str, r: pd.Series, pmax: float, rmax: float) -> str:
    barra = lambda v, m, clase: (f'<span class="bar {clase}" style="width:{v / m * 100:.1f}%"></span>'
                                 if pd.notna(v) else "")
    canon = _miles(r["canon_m2"]) if pd.notna(r["canon_m2"]) else "—"
    rent = _pct(r["rentab"], 1) if pd.notna(r["rentab"]) else "—"
    return f"""
      <tr>
        <td class="c-name"><b>{_comuna(c)}</b> <span>{html.escape(str(r['barrios']))}</span></td>
        <td class="num">{_n(r['viviendas'])}</td>
        <td class="num">{_mill(r['precio_m2'])}</td>
        <td class="bar-cell">{barra(r['precio_m2'], pmax, 'b-precio')}</td>
        <td class="num">{canon}</td>
        <td class="num">{rent}</td>
        <td class="bar-cell narrow">{barra(r['rentab'], rmax, 'b-rentab')}</td>
      </tr>"""


def _tarjeta(r: pd.Series, insignia: str, clase: str, precio: str, lineas: list[str | None]) -> str:
    extra = "".join(f"<p>{html.escape(l)}</p>" for l in lineas if l)
    return f"""
      <article class="card">
        <div class="card-top"><a class="ver" href="{html.escape(str(r['URL detalle']))}">Ver aviso ↗</a>
          <span class="badge {clase}">{insignia}</span></div>
        <h3>{html.escape(op._ficha(r))}</h3>
        <p class="zona">{html.escape(_barrio(r))}</p>
        <p class="precio">{precio}</p>{extra}
      </article>"""


def _renta_corta(r: pd.Series) -> str | None:
    if pd.isna(r.get("nivel_arr")):
        return None
    return (f"Arriendo estimado {_mill(r['canon_est'])}/mes → rentabilidad bruta {_pct(r['rentab_bruta'], 1)} "
            f"({r['n_arr']:.0f} arriendos similares)")


def informe_html(d: dict) -> str:
    t = d["tabla"]
    filas = "".join(_fila(c, r, t["precio_m2"].max(), t["rentab"].max()) for c, r in t.iterrows())
    periodo = _periodo(d["desde"], d["hasta"])
    nuevas = "".join(
        _tarjeta(r, f"{_pct(r['descuento'])} bajo similares", "ok",
                 f"<b>{_mill(r['Precio'], 0)}</b> · {_mill(r['precio_m2'])}/m²",
                 [f"{r['n_comps']} similares, mediana {_mill(r['med_comps'])}/m² · confianza {r['confianza'].lower()}",
                  _renta_corta(r), "⚠️ Descuento muy alto: verificar área y precio" if r["verificar"] else None])
        for _, r in d["sel"].iterrows())
    bajaron = "".join(
        _tarjeta(r, f"Bajó {_pct(r['baja'])}", "baja",
                 f"{_mill(r['precio_antes'], 0)} → <b>{_mill(r['Precio'], 0)}</b> · {_mill(r['precio_m2'])}/m²",
                 [op._frente_comps(r), _renta_corta(r)])
        for _, r in d["sel_bj"].iterrows())
    resumen = "".join(f"<li>{f}</li>" for f in frases(d))
    return f"""<!doctype html>
<html lang="es"><head><meta charset="utf-8">
<title>Pulso del mercado · {d['ciudad']}</title>
<style>
  @page {{ size: A4; margin: 12mm 14mm 14mm;
    @bottom-left {{ content: "Informe semanal · {d['ciudad']} · {_fecha(d['hasta'])}"; font: 7pt Inter, sans-serif; color: #898781; }}
    @bottom-right {{ content: "Página " counter(page) " de " counter(pages); font: 7pt Inter, sans-serif; color: #898781; }} }}
  :root {{ --ink:#0b0b0b; --ink-2:#52514e; --muted:#898781; --line:#e1e0d9; --plane:#f9f9f7;
           --blue:#2a78d6; --orange:#eb6834; --good:#006300; --good-bg:#e8f3e8; --neutral-bg:#f0efec; }}
  * {{ box-sizing: border-box; }}
  body {{ font-family: Inter, "DejaVu Sans", sans-serif; font-size: 8.6pt; line-height: 1.45; color: var(--ink); margin: 0; }}
  .kicker {{ font-size: 7.2pt; letter-spacing: .08em; text-transform: uppercase; color: var(--muted); margin: 0 0 2pt; }}
  h1 {{ font-size: 21pt; font-weight: 600; margin: 0; letter-spacing: -.01em; }}
  .periodo {{ color: var(--ink-2); margin: 2pt 0 10pt; }}
  h2 {{ font-size: 11.5pt; font-weight: 600; margin: 12pt 0 2pt; }}
  .sub {{ color: var(--ink-2); margin: 0 0 7pt; }}
  .tiles {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 6pt; margin-bottom: 6pt; }}
  .tile {{ background: var(--plane); border-radius: 6pt; padding: 6pt 9pt 7pt; }}
  .t-label {{ color: var(--ink-2); font-size: 7.6pt; }}
  .t-value {{ font-size: 16pt; font-weight: 600; margin: 0; }}
  .t-note {{ color: var(--muted); font-size: 7pt; line-height: 1.3; }}
  .resumen {{ border-left: 2.5pt solid var(--blue); padding: 1pt 0 1pt 9pt; margin: 8pt 0 0; }}
  .resumen ul {{ margin: 0; padding-left: 10pt; }} .resumen li {{ margin-bottom: 2pt; }}
  table {{ width: 100%; border-collapse: collapse; font-variant-numeric: tabular-nums; }}
  th {{ text-align: left; font-weight: 500; color: var(--ink-2); font-size: 7.4pt; padding: 0 4pt 4pt;
        border-bottom: 1px solid var(--line); vertical-align: bottom; }}
  th.num, td.num {{ text-align: right; }}
  tr {{ break-inside: avoid; }}
  td {{ padding: 2.5pt 4pt; border-bottom: 1px solid var(--line); vertical-align: middle; }}
  td.c-name {{ white-space: nowrap; }} td.c-name span {{ color: var(--muted); font-size: 7.4pt; }}
  td.bar-cell {{ width: 21%; padding-left: 2pt; }} td.bar-cell.narrow {{ width: 12%; }}
  .bar {{ display: block; height: 6pt; border-radius: 0 3pt 3pt 0; }}
  .b-precio {{ background: var(--blue); }} .b-rentab {{ background: var(--orange); }}
  .nota {{ color: var(--muted); font-size: 7pt; line-height: 1.4; margin-top: 8pt; }}
  .salto {{ break-before: page; }}
  .cards {{ display: grid; grid-template-columns: 1fr 1fr; gap: 7pt; }}
  .card {{ border: 1px solid var(--line); border-radius: 6pt; padding: 7pt 10pt 8pt; break-inside: avoid; }}
  .card-top {{ overflow: hidden; margin-bottom: 4pt; }}   /* sin flex: WeasyPrint pierde los enlaces dentro de flex o grid */
  .card-top .ver {{ float: right; margin-top: 1pt; }}
  .badge {{ font-weight: 600; font-size: 7.6pt; border-radius: 99pt; padding: 1.5pt 7pt; }}
  .badge.ok {{ background: var(--good-bg); color: var(--good); }}
  .badge.baja {{ background: var(--neutral-bg); color: var(--ink); }}
  .card a {{ color: var(--blue); font-weight: 600; text-decoration: none; font-size: 7.8pt; }}
  .card h3 {{ font-size: 9.4pt; margin: 0; font-weight: 600; }}
  .card .zona {{ color: var(--ink-2); margin: 1pt 0 4pt; }}
  .card .precio {{ font-size: 9.4pt; margin: 0 0 3pt; }}
  .card p {{ margin: 0 0 2pt; }} .card p:not(.zona):not(.precio) {{ color: var(--ink-2); font-size: 7.6pt; }}
  .vacio {{ color: var(--ink-2); }}
</style></head><body>

<p class="kicker">Informe semanal · vivienda en venta y arriendo</p>
<h1>Pulso del mercado · {d['ciudad']}</h1>
<p class="periodo">{periodo} · datos de FincaRaíz y Metrocuadrado</p>

<div class="tiles">
  {_tile("Viviendas en venta", _n(d['viviendas']), f"sin repetidos · {_n(d['avisos_fr'] + d['avisos_m2'])} avisos")}
  {_tile("Publicadas esta semana", _n(d['publicadas']), "viviendas nuevas en FincaRaíz")}
  {_tile("Retiradas esta semana", _n(d['retiradas']), "confirmadas tras 3 días sin verse")}
  {_tile("Bajaron de precio", _n(d['bajadas']), f"rebaja mediana del {_pct(d['bajada_mediana'])}" if d['bajada_mediana'] else "")}
</div>
<div class="tiles">
  {_tile("Precio mediano por m²", _mill(d['precio_m2']), "vivienda en venta, toda la ciudad")}
  {_tile("Arriendos de vivienda", _n(d['arriendos']), "sin amoblados ni por días")}
  {_tile("Canon mediano", _mill(d['canon_mediano']), "por mes")}
  {_tile("Rentabilidad bruta típica", _pct(d['rentab'], 1), "antes de gastos e impuestos")}
</div>

<div class="resumen"><ul>{resumen}</ul></div>

<h2>Por comuna</h2>
<p class="sub">De la más cara a la más económica por m². Comunas con al menos {MIN_VIVIENDAS} viviendas en venta.</p>
<table>
  <thead><tr><th>Comuna y barrios con más oferta</th><th class="num">Viviendas<br>en venta</th>
    <th class="num">Precio mediano<br>por m²</th><th></th><th class="num">Canon mediano<br>por m²</th>
    <th class="num">Rentabilidad<br>bruta típica</th><th></th></tr></thead>
  <tbody>{filas}</tbody>
</table>
<p class="nota">Rentabilidad bruta típica: canon mediano por m² × 12 ÷ precio mediano por m² de la comuna, antes de
  administración, impuestos y vacancia; «—»: menos de {MIN_ARRIENDOS} arriendos. Definiciones y fuentes, al final.</p>

<div class="salto"></div>
<p class="kicker">Informe semanal · {d['ciudad']}</p>
<h1>Oportunidades de la semana</h1>
<p class="periodo">{periodo}</p>

<h2>Nuevas, bajo el precio de inmuebles similares</h2>
<p class="sub">De {_n(d['evaluadas'])} viviendas publicadas esta semana, {_n(d['con_comparables'])} tenían suficientes
  similares para compararlas y {_n(d['bajo_umbral'])} estaban al menos un {_pct(UMBRAL)} por debajo. Estas son las
  más claras.</p>
<div class="cards">{nuevas or '<p class="vacio">Esta semana no hubo oportunidades que pasaran los filtros.</p>'}</div>

<h2>Bajaron de precio</h2>
<p class="sub">{_n(d['bajadas'])} viviendas bajaron {_pct(op.BAJA_MIN)} o más esta semana; estas son las mayores bajadas.</p>
<div class="cards">{bajaron or '<p class="vacio">Esta semana no hubo bajadas de precio.</p>'}</div>

<p class="nota">Viviendas: apartamentos, apartaestudios y casas. Sin repetidos: un inmueble publicado en los dos
  portales o por varias inmobiliarias cuenta una vez. Los movimientos de la semana (publicadas, retiradas, bajadas)
  son de FincaRaíz. Arriendos: sin amoblados ni por días. Comunas y barrios oficiales: IDESC, Alcaldía de Cali.
  Similares: mismo tipo y estrato, área ±30% y ±1 habitación, en el mismo barrio oficial (o a menos de
  1 km, o en la comuna); el descuento compara el precio por m² con la mediana de esos similares. Es una estimación del
  precio de mercado, no un avalúo (Ley 1673 de 2013): verifica siempre el inmueble y su documentación. Arriendo
  estimado: mediana del canon por m² de arriendos similares publicados. Fuente: avisos públicos de FincaRaíz y
  Metrocuadrado.</p>
</body></html>"""


def pie(d: dict) -> str:
    """El texto que acompaña al PDF en Telegram (formato de src/notificar.py: *negrita*)."""
    return (f"*Informe semanal · {d['ciudad']}*\n{_periodo(d['desde'], d['hasta'])}: "
            f"{_n(d['viviendas'])} viviendas en venta, {_n(d['publicadas'])} nuevas, {_n(d['bajadas'])} bajaron "
            f"de precio y {_n(len(d['sel']))} oportunidades bajo el precio de inmuebles similares.")


def generar(ciudad: str = "Cali", fecha: str | None = None) -> tuple[str, dict]:
    """Escribe el .html y el .pdf; devuelve la ruta del PDF y los datos."""
    from weasyprint import HTML
    d = datos(ciudad, fecha)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base = OUT_DIR / f"{ciudad.lower()}_{d['hasta']:%Y-%m-%d}"
    texto = informe_html(d)
    base.with_suffix(".html").write_text(texto, encoding="utf-8")
    HTML(string=texto, base_url=str(BASE_DIR)).write_pdf(base.with_suffix(".pdf"))
    print(f"✅ {base.with_suffix('.pdf')}")
    return str(base.with_suffix(".pdf")), d


def enviar(pdf: str, d: dict, a_quien: str) -> None:
    """Por Telegram: 'admin' (TELEGRAM_ADMIN_CHAT_ID) o 'agentes' (TELEGRAM_CHAT_ID, separados por coma)."""
    import os
    from src.notificar import documento
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chats = os.environ.get("TELEGRAM_ADMIN_CHAT_ID" if a_quien == "admin" else "TELEGRAM_CHAT_ID", "")
    if not (token and chats):
        raise SystemExit("❌ Falta TELEGRAM_BOT_TOKEN o el chat de destino")
    for chat in filter(None, map(str.strip, chats.split(","))):
        documento(pdf, token, chat, pie(d))
    print(f"✅ Enviado ({a_quien})")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Informe semanal para los agentes (PDF)")
    p.add_argument("--ciudad", default="Cali")
    p.add_argument("--fecha", help="último día de la semana (AAAA-MM-DD); por defecto, el último dato")
    p.add_argument("--enviar", choices=["admin", "agentes"], help="lo envía por Telegram")
    a = p.parse_args()
    pdf, d = generar(a.ciudad, a.fecha)
    if a.enviar:
        try:                                    # en local, las claves vienen de .env
            from src.almacen import _cargar_env
            _cargar_env()
        except ImportError:
            pass
        enviar(pdf, d, a.enviar)
