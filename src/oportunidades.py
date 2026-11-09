"""
Alerta semanal de oportunidades, con dos secciones:

  A) Avisos NUEVOS por debajo del precio de sus comparables. Para cada aviso de la semana
     (vivienda: apartamento, apartaestudio, casa):
       1. Busca comparables: mismo tipo y estrato, área ±30%, ±1 habitación.
          Primero en su barrio oficial; si no alcanzan, a menos de 1 km; luego en su comuna.
       2. Descuento = 1 − (precio/m² del aviso ÷ mediana de los comparables).
       3. Descarta probables errores (área incoherente, descuento extremo, proyectos sobre planos).
  B) Avisos que BAJARON DE PRECIO en la semana (bitácora de src/eventos.py), con su
     historial (días publicado, bajadas acumuladas) y dónde quedan frente a sus comparables. Cuenta
     también los que se republicaron con otro número y un precio más bajo ("bajó al republicar").

Cada aviso de la alerta trae además (enriquecer()):
  - el precio que estima el modelo (src/modelo_precio.py) y cuánto pide por debajo o por encima, siempre de un
    modelo que no vio el aviso (con_modelo);
  - 3 de sus comparables, los de área más parecida, con su enlace;
  - si el mismo inmueble está también en Metrocuadrado y a qué precio (registro único, src/unificar.py);
  - si es un aviso republicado: desde cuándo estaba publicado con su número anterior y a qué precio.

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


def cargar(ciudad: str, df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Avisos de vivienda de la ciudad, con zona oficial y first_seen/etiquetas.
    Con la fuente http, first_seen es la fecha real de publicación y ya viene en el dataset.
    `df`: otros avisos ya limpios con viz_map.limpiar_ciudad (p. ej. los de Metrocuadrado); por defecto,
    los de FincaRaíz."""
    df = cargar_ciudad(ciudad) if df is None else df
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
    b = ev[ev["evento"].isin(["cambio_precio", "republicado"]) & (ev["precio_despues"] < ev["precio_antes"])
           & ev["fecha"].between(f"{desde:%Y-%m-%d}", f"{hasta:%Y-%m-%d}")]
    b = b.sort_values("fecha").drop_duplicates("id_inmueble", keep="last")
    b = b.assign(baja=1 - b["precio_despues"] / b["precio_antes"], al_republicar=b["evento"] == "republicado")
    b = b[b["baja"].between(BAJA_MIN, 0.5)][["id_inmueble", "fecha", "precio_antes", "baja", "al_republicar"]]
    candidatos = df.merge(b.rename(columns={"fecha": "fecha_baja"}), on="id_inmueble")
    return calificar(df, candidatos, hist)


def seleccionar_bajadas(bj: pd.DataFrame, excluir: pd.Index, top: int) -> pd.DataFrame:
    """Primero las que tras bajar quedan ≥10% bajo sus comparables; luego, por tamaño de la bajada."""
    ok = bj[bj["motivo_descarte"].isna() & ~bj["id_inmueble"].isin(excluir)]
    return ok.assign(_bajo=ok["descuento"] >= 0.10) \
             .sort_values(["_bajo", "baja"], ascending=False).head(top).drop(columns="_bajo")


# ── Lo que acompaña a cada aviso de la alerta ─────────────────────────────────
COLS_MODELO = ["modelo_precio", "modelo_p10", "modelo_p90", "modelo_conf", "bajo_modelo"]


def con_modelo(sel: pd.DataFrame) -> pd.DataFrame:
    """Precio que estima el modelo (src/modelo_precio.py) para cada aviso, su rango del 80% y cuánto pide por debajo
    (`bajo_modelo` > 0) o por encima. Solo estimados de un modelo que NO vio el aviso (estimar_sin_trampa: la tabla de
    estimados sin ver para los avisos con que se entrenó, el modelo vigente para el resto): uno que ya conoce su
    precio pedido tiende a darle la razón (medido el 8-nov: de los que piden ≥15% más de lo que valen, reconoce el
    27%). Sin modelo en disco (data/modelos/), o si no se puede saber si lo vio, la línea no sale: la alerta sí."""
    sel = sel.assign(**{c: np.nan for c in COLS_MODELO})
    if sel.empty or FUENTE != "http":
        return sel
    try:
        from src.modelo_precio import ARCHIVO, estimar_sin_trampa, preparar
    except ImportError as e:                    # sin lightgbm instalado
        print(f"⚠️  Sin el precio del modelo ({e})")
        return sel
    if not ARCHIVO.exists():
        print(f"⚠️  Sin el precio del modelo: no está {ARCHIVO}")
        return sel
    from src.consolidar import rutas
    a = pd.read_parquet(rutas("venta").avisos)
    a = a[a["id_inmueble"].astype(str).isin(sel["id_inmueble"])].drop_duplicates("id_inmueble")
    a = zonas.corregir_ubicacion(a, solo_imposibles=True)
    p = estimar_sin_trampa(preparar(a))
    p.index = a["id_inmueble"].astype(str).to_numpy()
    if (~p["sin_ver"]).any():
        print(f"⚠️  {(~p['sin_ver']).sum()} avisos sin el precio del modelo: el modelo los vio y no hay tabla de "
              f"estimados sin ver ({SIN_VER_NOTA})")
    p = p[p["sin_ver"]]
    ids = sel["id_inmueble"]
    sel["modelo_precio"] = ids.map(p["precio_estimado"])
    sel["modelo_p10"] = ids.map(p["precio_p10"])
    sel["modelo_p90"] = ids.map(p["precio_p90"])
    sel["modelo_conf"] = ids.map(p["confianza"])
    sel["bajo_modelo"] = 1 - sel["Precio"] / sel["modelo_precio"]
    return sel


SIN_VER_NOTA = "modelo.yml la calcula los domingos: data/modelos/estimados_sin_ver.parquet"


def parecido(c: pd.Series, comps: pd.DataFrame) -> pd.Series:
    """Menor es más parecido a `c`: diferencia de área, de habitaciones y distancia."""
    area = np.abs(np.log(comps["Area_m2"] / c["Area_m2"]))
    hab = (comps["Habitaciones"] - c["Habitaciones"]).abs().fillna(1) if pd.notna(c["Habitaciones"]) else 0
    if pd.notna(c["Latitud"]):
        dy = (comps["Latitud"] - c["Latitud"]) * 110.54
        dx = (comps["Longitud"] - c["Longitud"]) * 111.32 * np.cos(np.radians(c["Latitud"]))
        km = np.hypot(dx, dy).fillna(1)
    else:
        km = 0
    return area + 0.15 * hab + 0.2 * km


def tres_comparables(r: pd.Series, pool: pd.DataFrame, n: int = 3) -> pd.DataFrame:
    """`n` de los comparables que dan la mediana, para mostrar: los más parecidos entre los típicos (la mitad
    central de precio/m²), sin copias del mismo inmueble (mismo precio y área que el aviso, o entre ellos)."""
    comps, _ = _comparables(r, pool)
    copia = (comps["Precio"] == r["Precio"]) & (comps["Area_m2"].round() == round(r["Area_m2"]))
    comps = comps[~copia].drop_duplicates(["Precio", "Area_m2"])
    lo, hi = comps["precio_m2"].quantile([.25, .75])
    tipicos = comps[comps["precio_m2"].between(lo, hi)]
    comps = tipicos if len(tipicos) >= n else comps
    return comps.assign(_d=parecido(r, comps)).nsmallest(n, "_d")


def en_metrocuadrado(sel: pd.DataFrame, avisos: pd.DataFrame) -> pd.DataFrame:
    """Para cada aviso de FincaRaíz, el mismo inmueble en Metrocuadrado (registro único), si está:
    `m2_precio` y `m2_url`. `avisos` es el primero de lo que devuelve src/unificar.unificar(ciudad)."""
    grupo = avisos.set_index("clave")["id_unificado"]
    m2 = avisos[avisos["portal"] == "metrocuadrado"].drop_duplicates("id_unificado").set_index("id_unificado")
    g = ("fincaraiz:" + sel["id_inmueble"].astype(str)).map(grupo)
    return pd.DataFrame({"m2_precio": g.map(m2["precio"]).to_numpy(), "m2_url": g.map(m2["url"]).to_numpy()},
                        index=sel.index)


def republicados(hasta: pd.Timestamp) -> pd.DataFrame:
    """Por aviso republicado con otro número (src/consolidar.py): el precio que tenía antes (`rep_precio_antes`)
    y desde cuándo estaba publicado con sus números anteriores (`rep_desde`; sigue la cadena si se republicó
    varias veces)."""
    vacio = pd.DataFrame(columns=["rep_precio_antes", "rep_desde"])
    if FUENTE != "http":
        return vacio
    ev = _eventos()
    r = ev[(ev["evento"] == "republicado") & (ev["fecha"] <= f"{hasta:%Y-%m-%d}")]
    if r.empty:
        return vacio
    r = r.sort_values("fecha").drop_duplicates("id_inmueble", keep="last").set_index("id_inmueble")
    viejo = r["id_viejo"].astype(str)
    primero = viejo.copy()
    for _ in range(10):                                   # el más antiguo de la cadena
        siguiente = primero.map(viejo)
        if siguiente.isna().all():
            break
        primero = siguiente.fillna(primero)
    from src.consolidar import rutas
    a = pd.read_parquet(rutas("venta").avisos, columns=["id_inmueble", "fecha_publicacion"])
    pub = pd.to_datetime(a.set_index(a["id_inmueble"].astype(str))["fecha_publicacion"], errors="coerce")
    pub = pub[~pub.index.duplicated()]
    return pd.DataFrame({"rep_precio_antes": r["precio_antes"],
                         "rep_desde": primero.map(pub).fillna(viejo.map(pub))})


def enriquecer(sel: pd.DataFrame, pool: pd.DataFrame, hasta: pd.Timestamp,
               avisos: pd.DataFrame | None = None) -> pd.DataFrame:
    """El modelo (sin trampa), 3 comparables, el otro portal (si se pasan los `avisos` del registro único) y la
    republicación."""
    if sel.empty:
        return sel
    sel = con_modelo(sel)
    sel["comps"] = [tres_comparables(r, pool) for _, r in sel.iterrows()]
    if avisos is not None:
        sel = sel.join(en_metrocuadrado(sel, avisos))
    return sel.join(republicados(hasta), on="id_inmueble")


# ── Presentación ──────────────────────────────────────────────────────────────
def _mayus(t: str) -> str:
    return t[:1].upper() + t[1:]


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


def _modelo_txt(r: pd.Series, fmt=None) -> str | None:
    """'modelo: $540 M (entre $480 M y $600 M) → 22% bajo el modelo'; None sin modelo. `fmt` da formato a
    los precios (por defecto, _millones)."""
    if pd.isna(r.get("modelo_precio")):
        return None
    fmt = fmt or _millones
    b = r["bajo_modelo"]
    lado = "en línea con el modelo" if abs(b) < 0.03 else f"{abs(b):.0%} {'bajo' if b > 0 else 'sobre'} el modelo"
    return f"modelo: {fmt(r['modelo_precio'])} (entre {fmt(r['modelo_p10'])} y {fmt(r['modelo_p90'])}) → {lado}"


def _comp_txt(c: pd.Series) -> str:
    """'El Ingenio · 82 m² · $520 M · 6,3 M/m²'"""
    barrio = c["barrio_oficial"] if pd.notna(c["barrio_oficial"]) else c["Barrio"]
    return f"{str(barrio).title()} · {c['Area_m2']:.0f} m² · {_millones(c['Precio'])} · {_m2(c['precio_m2'])} M/m²"


def _otro_portal_txt(r: pd.Series) -> str | None:
    """'también en Metrocuadrado a $430 M (2% más)'; None si no está en Metrocuadrado."""
    if pd.isna(r.get("m2_precio")):
        return None
    dif = r["m2_precio"] / r["Precio"] - 1
    cuanto = "el mismo precio" if abs(dif) < .005 else f"{abs(dif):.0%} {'más' if dif > 0 else 'menos'}"
    return f"también en Metrocuadrado a {_millones(r['m2_precio'])} ({cuanto})"


def _republicado_txt(r: pd.Series, hasta: pd.Timestamp, fmt=None) -> str | None:
    """'republicado con otro número: estaba publicado desde hace 80 días, a $450 M (al republicar bajó 7%)';
    None si no lo es."""
    if pd.isna(r.get("rep_precio_antes")):
        return None
    t = "republicado con otro número"
    if pd.notna(r.get("rep_desde")):
        t += f": estaba publicado desde hace {(hasta - r['rep_desde']).days} días"
    cambio = r["Precio"] / r["rep_precio_antes"] - 1
    if not r.get("al_republicar", False) and abs(cambio) >= .01:   # si bajó al republicar ya lo dice el título
        t += (f", a {(fmt or _millones)(r['rep_precio_antes'])} (al republicar {'bajó' if cambio < 0 else 'subió'} "
              f"{abs(cambio):.0%})")
    return t


def _historia_txt(r: pd.Series, hasta: pd.Timestamp) -> str | None:
    """'publicado hace 45 días · bajó 2 veces (−8% en total)'; None si no hay nada que contar.
    Si es un aviso republicado, cuenta desde cuándo estaba publicado con su número anterior."""
    partes = [rep] if (rep := _republicado_txt(r, hasta)) else []
    if not partes and pd.notna(r["first_seen"]) and (hasta - r["first_seen"]).days >= 7:
        partes.append("publicado desde antes de agosto" if r["first_seen"] < INICIO_DATOS
                      else f"publicado hace {(hasta - r['first_seen']).days} días")
    if r["n_bajadas"] > 0:
        veces = "1 vez" if r["n_bajadas"] == 1 else f"{r['n_bajadas']} veces"
        partes.append(f"bajó {veces} ({r['var_total']:+.0%} en total)".replace("-", "−"))
    return " · ".join(partes) or None


def _bajo_txt(r: pd.Series) -> str:
    """'Bajó 8%' o 'Bajó 8% al republicar'."""
    return f"Bajó {r['baja']:.0%}" + (" al republicar" if r.get("al_republicar", False) else "")


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
    comps = lambda r: [f"· {_comp_txt(c)}" for _, c in r["comps"].iterrows()] if "comps" in r else []
    nuevas = [_tarjeta(r, i, f"{r['descuento']:.0%} bajo comparables", "ok",
                       f"<b>{_millones(r['Precio'])}</b> · {_m2(r['precio_m2'])} M/m²",
                       [f"{_comps_txt(r)} · confianza {r['confianza'].lower()}", *comps(r), _modelo_txt(r),
                        _otro_portal_txt(r), renta_txt(r), _historia_txt(r, hasta)],
                       _señales(r))
              for i, (_, r) in enumerate(sel.iterrows(), 1)]
    bajaron = [_tarjeta(r, i, _bajo_txt(r), "baja",
                        f"{_millones(r['precio_antes'])} → <b>{_millones(r['Precio'])}</b> · "
                        f"{_m2(r['precio_m2'])} M/m²",
                        [_frente_comps(r), _modelo_txt(r), _otro_portal_txt(r), renta_txt(r),
                         _historia_txt(r, hasta)], [])
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


def _enlaces(r: pd.Series) -> list[str]:
    """El aviso en FincaRaíz y, si está, el mismo inmueble en Metrocuadrado con su precio."""
    lineas = [f"🔗 [Ver aviso en FincaRaíz]({r['URL detalle']})"]
    if otro := _otro_portal_txt(r):
        lineas.append(f"🔁 {_mayus(otro)} · [ver]({r['m2_url']})")
    return lineas


def mensaje_whatsapp(sel: pd.DataFrame, bj: pd.DataFrame, resumen: dict, hasta: pd.Timestamp) -> str:
    from src.rentabilidad import texto as renta_txt
    lineas = [f"*Oportunidades de la semana — {resumen['ciudad']}*", resumen["periodo"], "",
              "*🆕 Nuevos bajo sus comparables*", ""]
    for i, (_, r) in enumerate(sel.iterrows(), 1):
        lineas += [f"*{i}. {r['descuento']:.0%} bajo comparables* — {_ficha(r)}",
                   f"📍 {_zona(r)}",
                   f"💰 {_millones(r['Precio'])} ({_m2(r['precio_m2'])} M/m²)",
                   f"📊 {_comps_txt(r)} · confianza {r['confianza'].lower()}"]
        if "comps" in r:          # [texto](enlace): en Telegram el enlace va dentro de "ver"
            lineas += [f"   ↳ {_comp_txt(c)} · [ver]({c['URL detalle']})" for _, c in r["comps"].iterrows()]
        if modelo := _modelo_txt(r):
            lineas.append(f"🤖 {_mayus(modelo)}")
        if renta := renta_txt(r):
            lineas.append(f"💵 {renta}")
        if historia := _historia_txt(r, hasta):
            lineas.append(f"🕒 {historia}")
        lineas += [f"🏷️ {s}" for s in _señales(r)]
        lineas += _enlaces(r) + [""]
    if not bj.empty:
        lineas += [f"*📉 Bajaron de precio esta semana* ({resumen['bajadas']} en total, las mayores)", ""]
        for i, (_, r) in enumerate(bj.iterrows(), 1):
            lineas += [f"*{i}. {_bajo_txt(r)}* — {_ficha(r)}",
                       f"📍 {_zona(r)}",
                       f"💰 {_millones(r['precio_antes'])} → {_millones(r['Precio'])} ({_m2(r['precio_m2'])} M/m²)"]
            if frente := _frente_comps(r):
                lineas.append(f"📊 {frente}")
            if modelo := _modelo_txt(r):
                lineas.append(f"🤖 {_mayus(modelo)}")
            if renta := renta_txt(r):
                lineas.append(f"💵 {renta}")
            if historia := _historia_txt(r, hasta):
                lineas.append(f"🕒 {historia}")
            lineas += _enlaces(r) + [""]
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
    avisos = None
    if FUENTE == "http" and (len(sel) or len(sel_bj)):
        from src import unificar
        try:
            avisos = unificar.unificar(args.ciudad)[0]
        except FileNotFoundError as e:           # sin el estado de Metrocuadrado: la alerta sale sin él
            print(f"⚠️  Sin Metrocuadrado ({e})")
    sel = enriquecer(sel, df, hasta, avisos)
    sel_bj = enriquecer(sel_bj, df, hasta, avisos)

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
            "al_republicar",
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
