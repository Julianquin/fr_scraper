"""
Cifras neutras para las piezas de redes: solo números y nombres de zonas, sin textos, diseño ni marca (el repositorio
es público). Las piezas (vídeos, carruseles, historias) se hacen fuera del repositorio a partir de estas cifras.

Dos productos:
  - `ficha(r)`: una oportunidad en cifras. canal.py la guarda con cada propuesta en canal/estado.json y pasa a
    «publicados» cuando se aprueba: de ahí sale el vídeo de la oportunidad del día.
  - `semana(ciudad)`: el mercado de los últimos 7 días (pulso, precio por m² y rentabilidad por comuna, cifras por
    barrio, el barrio de la semana, mayores bajadas, captación en agregado, el crédito de vivienda y la cuota del
    apartamento típico de cada comuna, y Cali y su área año a año). canal.yml lo deja cada día, tras proponer, en
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

from src import credito, zonas
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
        # La cuota aproximada con la tasa media de las últimas semanas (src/credito.py); None sin la serie
        "cuota": credito.para_ficha(g("Precio"), _texto(g("Ciudad")) or "Cali"),
    }


# ── Cali y su área, año a año (E9.5) ──────────────────────────────────────────────────────────────────────
def area_metropolitana() -> dict | None:
    """Las compraventas e hipotecas registradas por año en Cali, Jamundí, Palmira, Yumbo y Candelaria
    (src/transacciones.py, IGAC: 2015–2023) y el crédito de vivienda de cada uno en las últimas 8 semanas
    (src/credito.py; el número de créditos, solo si es fiable: en los municipios pequeños las semanas con hueco
    son frecuentes). None sin las transacciones."""
    from src import transacciones
    r = transacciones.resumen()
    if not r:
        return None
    serie = credito.leer()
    for m in r["municipios"]:
        c = credito.resumen(serie, m["codigo"], semanas=8) if not serie.empty else None
        m["credito"] = None if not c else {
            "hasta": c["hasta"], "creditos": c["creditos"] if c["fiable"] else None, "tasa_pesos": c["tasa_pesos"],
            "parte_vis": c["parte_vis"]}
    return _limpio(r)


# ── El crédito de vivienda ────────────────────────────────────────────────────────────────────────────────
MIN_APTOS_CUOTA = 30            # una comuna entra en «¿cuánto hay que ganar?» con al menos estos apartamentos en venta


def credito_semana(d: dict) -> dict | None:
    """El crédito de vivienda de la ciudad en cifras (src/credito.py): la tasa y los créditos de las últimas 4
    semanas (el volumen, solo si es `fiable`), el último mes completo, la cuota del apartamento típico de la ciudad y
    de cada comuna, y la serie de 26 semanas para el gráfico. None sin la serie o si la ciudad no se baja.
    El ejemplo es el APARTAMENTO típico (precio pedido mediano): con las casas, la mediana de Cali sube de 400 a 560
    millones (17-nov) y el ejemplo deja de parecerse a lo que compra la mayoría."""
    municipio = credito.CIUDADES.get(d["ciudad"])
    serie = credito.leer()
    r = credito.resumen(serie, municipio) if municipio else None
    if not r or not r["tasa_pesos"]:
        return None
    tasa = r["tasa_pesos"]
    u = d["inmuebles"]
    aptos = u[u["tipo"].astype(str).str.lower().str.startswith("aparta")]

    def ejemplo(x: pd.DataFrame) -> dict:
        """El apartamento típico y su cuota en pesos (fija) y en UVR (la primera; luego sube con la inflación)."""
        precio = x["precio_mediano"].median()
        c = credito.cuota(precio, tasa)
        u = credito.cuota(precio, r["tasa_uvr"]) if r["tasa_uvr"] else None
        return {"precio": _num(precio), "area": _num(x["m2"].median()), "apartamentos": len(x), "cuota": _num(c),
                "ingreso": _num(credito.ingreso(c)), "cuota_uvr": _num(u),
                "ingreso_uvr": _num(credito.ingreso(u)) if u else None}

    comunas = [{"codigo": str(c), "nombre": d["nombres"].get(c, c), **ejemplo(x)}
               for c, x in aptos.dropna(subset=["comuna"]).groupby("comuna") if len(x) >= MIN_APTOS_CUOTA]
    return _limpio({**r, "mes": credito.mes(serie, municipio), "financiado": credito.FINANCIADO, "años": credito.AÑOS,
                    "cuota_max": credito.CUOTA_MAX, "ciudad_ejemplo": ejemplo(aptos),
                    "min_apartamentos": MIN_APTOS_CUOTA, "por_comuna": comunas})


# ── Por barrio, y el barrio de la semana ──────────────────────────────────────────────────────────────────
MIN_BARRIO_SEMANA = 15          # el barrio de la semana sale de los que tienen al menos estas viviendas en venta
MIN_ARRIENDOS_BARRIO = 15       # canon y rentabilidad de un barrio, con al menos estos arriendos
DIAS_BAJADAS = 30               # «bajaron de precio en el último mes»
OLVIDO = 26                     # cuando ya salieron todos, vuelve uno que no haya salido en estas semanas
BARRIOS_SEMANA = "contenido/barrios_semana.json"    # {ciudad: {"2026-W47": {"comuna", "barrio", "precio_m2"}}}
# (campo, decimales) de cada barrio
CAMPOS_BARRIO = [("viviendas", 0), ("precio_m2", 0), ("p25_m2", 0), ("p75_m2", 0), ("precio", 0), ("precio_p25", 0),
                 ("precio_p75", 0), ("area", 0), ("habitaciones", 1), ("apartamentos", 2), ("estrato", 0),
                 ("en_venta_mes", 0), ("bajaron_mes", 0), ("dias_publicadas", 0), ("arriendos", 0), ("canon_m2", 0),
                 ("rentabilidad", 4),
                 # del catastro (E9.2): viviendas que hay, la parte de apartamentos, pisos y área típicos, cuántos
                 # edificios o conjuntos, la parte de coordenadas genéricas y qué parte está en venta (o None)
                 ("viviendas_catastro", 0), ("apartamentos_parte_catastro", 2), ("pisos_edificio", 0),
                 ("area_apartamento_catastro", 0), ("edificios", 0), ("genericos", 2), ("en_venta_parte", 4),
                 # del avalúo catastral (E9.4): el avalúo por m² construido de sus apartamentos y casas, y cuántas veces
                 # se pide (precio pedido por m² ÷ avalúo por m², el mismo tipo)
                 ("avaluo_m2_apartamento", 0), ("veces_avaluo_apartamento", 2), ("avaluo_m2_casa", 0),
                 ("veces_avaluo_casa", 2)]


def por_barrio(d: dict) -> pd.DataFrame:
    """Una fila por barrio oficial con al menos MIN_BARRIO viviendas en venta (los del mapa del informe), del más
    caro al más económico por m². Precio, área y tipo: de las viviendas sin repetir de los dos portales; cuántas
    bajaron de precio en el último mes: por inmueble en los dos portales (src/movimientos.py; `en_venta_mes`, las que
    estuvieron en venta todo ese mes); días publicadas: de las que tienen fecha de publicación (FincaRaíz; E8.5)."""
    from src import movimientos
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
    mb = movimientos.periodo(d["ciudad"], d["hasta"] - pd.Timedelta(days=DIAS_BAJADAS - 1), d["hasta"],
                             por="barrio_oficial")
    nombre = t.index.get_level_values("barrio_oficial")
    t["en_venta_mes"] = nombre.map(mb["en_venta_todo"]).to_numpy()
    t["bajaron_mes"] = nombre.map(mb["bajaron"]).to_numpy()
    gv = d["venta"].dropna(subset=clave).groupby(clave)
    t["dias_publicadas"] = gv["dias_pub"].median()
    t["estrato"] = gv["Estrato"].agg(lambda x: x.mode().iloc[0] if x.notna().any() else np.nan)
    ga = d["arriendos_ok"].dropna(subset=clave).groupby(clave)
    t["arriendos"] = ga.size()
    t["canon_m2"] = ga["precio_m2"].median()
    t["canon_m2"] = t["canon_m2"].where(t["arriendos"] >= MIN_ARRIENDOS_BARRIO)
    t["rentabilidad"] = 12 * t["canon_m2"] / t["precio_m2"]
    t = con_catastro(t, u, clave)
    return t.sort_values("precio_m2", ascending=False)


# ── El catastro: cuántas viviendas hay y qué parte está en venta (E9.2) ─────────────────────────────────────
GENERICO = 5                    # un punto con al menos estos inmuebles en venta es una coordenada genérica (el centro
                                # de un barrio o de la ciudad), no la del inmueble: el 21,5 % en Cali (17-nov)
MAX_GENERICOS = 0.25            # «1 de cada N en venta» solo donde menos de esta parte de lo que está en venta tiene
                                # coordenada genérica (en Pance cabecera, el 97 %: salía un 75 % en venta; con las
                                # precisas, un 2 %)
MIN_VIVIENDAS_CATASTRO = 100    # y con al menos estas viviendas en el catastro


def genericos(u: pd.DataFrame) -> pd.Series:
    """Por inmueble: ¿su coordenada es genérica (compartida con GENERICO o más)?"""
    punto = u["Latitud"].round(5).astype(str) + "," + u["Longitud"].round(5).astype(str)
    return punto.map(punto.value_counts()) >= GENERICO


def con_catastro(t: pd.DataFrame, u: pd.DataFrame, clave: list[str]) -> pd.DataFrame:
    """Añade a la tabla por zona (índice `clave`) lo del catastro (src/catastro_cali.py): viviendas, la parte de
    apartamentos, los pisos y el área típicos, y `en_venta_parte`: las que están en venta frente a las del catastro
    (solo si la zona tiene pocas coordenadas genéricas: MAX_GENERICOS). Sin el catastro, la tabla igual."""
    from src import catastro_cali
    cat = catastro_cali.leer("barrios")
    if cat is None or not len(t):
        return t
    if clave[-1] == "comuna":           # por comuna: la suma de sus barrios
        g = cat.groupby("comuna")
        cat = pd.DataFrame({"viviendas": g["viviendas"].sum(), "apartamentos": g["apartamentos"].sum(),
                            "pisos_edificio": g.apply(lambda x: np.average(x["pisos_edificio"].fillna(0),
                                                                          weights=x["apartamentos"])
                                                      if x["apartamentos"].sum() else np.nan),
                            "area_apartamento": g["area_apartamento"].median(), "edificios": g["edificios"].sum()})
        cat["parte_apartamentos"] = cat["apartamentos"] / cat["viviendas"]
    else:
        cat = cat.set_index(clave)
    cat = cat.reindex(t.index)
    t = t.copy()
    t["viviendas_catastro"] = cat["viviendas"]
    t["apartamentos_parte_catastro"] = cat["parte_apartamentos"]
    t["pisos_edificio"] = cat["pisos_edificio"]
    t["area_apartamento_catastro"] = cat["area_apartamento"]
    t["edificios"] = cat["edificios"]
    gen = u.assign(_g=genericos(u)).dropna(subset=clave).groupby(clave)["_g"].mean()
    t["genericos"] = gen.reindex(t.index)
    t["en_venta_parte"] = (t["viviendas"] / t["viviendas_catastro"]).where(
        (t["genericos"] < MAX_GENERICOS) & (t["viviendas_catastro"] >= MIN_VIVIENDAS_CATASTRO))
    return con_avaluo(t, u, clave)


# ── El avalúo catastral: cuántas veces el avalúo se pide (E9.4) ──────────────────────────────────────────────
MIN_TIPO_ZONA = 10              # viviendas en venta de un tipo en la zona para comparar su precio con el avalúo


def tipo_simple(tipo: pd.Series) -> pd.Series:
    """«apartamento» (también apartaestudios), «casa» (también campestres) o vacío: los dos tipos del avalúo."""
    t = tipo.astype(str).str.lower()
    return pd.Series(pd.NA, index=tipo.index, dtype="object").mask(t.str.startswith("aparta"), "apartamento").mask(
        t.str.startswith("casa"), "casa")


def con_avaluo(t: pd.DataFrame, u: pd.DataFrame, clave: list[str]) -> pd.DataFrame:
    """Añade el avalúo por m² construido de apartamentos y casas de cada zona (src/avaluos.py) y `veces_avaluo_…`:
    el precio pedido por m² de ese tipo en la zona ÷ su avalúo por m² (con al menos MIN_TIPO_ZONA en venta). Medido
    el 20-nov: en apartamentos, ~2,8 veces en Cali. Sin el resumen de avalúos, la tabla igual."""
    from src import avaluos
    a = avaluos.por_barrio()
    if a is None or not len(t):
        return t
    nivel = "comuna" if clave[-1] == "comuna" and len(clave) == 1 else "barrio"
    a = a[a["nivel"] == nivel]
    u = u.dropna(subset=clave).assign(_tipo=lambda x: tipo_simple(x["tipo"]))
    t = t.copy()
    for tipo in avaluos.TIPOS:
        ped = u[u["_tipo"] == tipo].groupby(clave)["precio_m2"].agg(["median", "size"])
        av = a[a["tipo"] == tipo].set_index(clave)["avaluo_m2"]
        av.index = av.index if len(clave) > 1 else av.index.astype(str)
        t[f"avaluo_m2_{tipo}"] = av.reindex(t.index)
        veces = (ped["median"] / av.reindex(ped.index)).where(ped["size"] >= MIN_TIPO_ZONA)
        t[f"veces_avaluo_{tipo}"] = veces.reindex(t.index)
    return t


def ciudad_avaluo(d: dict) -> dict:
    """El avalúo por m² de los apartamentos de la ciudad y cuántas veces se pide (para comparar el barrio). Vacío sin
    el resumen de avalúos."""
    from src import avaluos
    a = avaluos.por_barrio()
    if a is None or d["ciudad"] != "Cali":
        return {}
    c = a[(a["nivel"] == "ciudad") & (a["tipo"] == "apartamento")]
    u = d["inmuebles"]
    ped = u.loc[tipo_simple(u["tipo"]) == "apartamento", "precio_m2"].median()
    if c.empty or not ped:
        return {}
    av = float(c["avaluo_m2"].iloc[0])
    return {"ciudad_avaluo_m2_apartamento": _num(av), "ciudad_veces_avaluo_apartamento": _num(ped / av, 2)}


def conjuntos_barrio(comuna: str, barrio: str, n: int = 5) -> list[dict]:
    """Los edificios y conjuntos con más apartamentos del barrio, según el catastro (dirección, apartamentos, pisos):
    para los agentes (informe semanal), nunca en lo público. Vacío sin el catastro."""
    from src import catastro_cali
    c = catastro_cali.leer("conjuntos")
    if c is None:
        return []
    c = c[(c["comuna"] == comuna) & (c["barrio_oficial"] == barrio)].head(n)
    return [{"direccion": x["direccion"], "apartamentos": _num(x["apartamentos"]), "pisos": _num(x["pisos"]),
             "area": _num(x["area"])} for _, x in c.iterrows()]


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
        **ciudad_catastro(d), **ciudad_avaluo(d),
        "conjuntos": conjuntos_barrio(comuna, barrio),          # para los agentes (el informe), no para lo público
    }


def ciudad_catastro(d: dict) -> dict:
    """Las viviendas de la ciudad en el catastro y qué parte está en venta (todas: en la ciudad entera las coordenadas
    genéricas no cambian el total). Vacío sin el catastro."""
    from src import catastro_cali
    m = catastro_cali.meta()
    if not m or d["ciudad"] != "Cali":
        return {}
    return {"ciudad_viviendas_catastro": m["viviendas"], "ciudad_en_venta_parte": _num(d["viviendas"] / m["viviendas"], 4),
            "catastro_bajado": m["bajado"]}


# ── El mes ────────────────────────────────────────────────────────────────────────────────────────────────
# Lo que cambia de un mes a otro. No la mediana del m² por comuna o barrio: en 3 semanas, solo por qué viviendas
# entran y salen, el 15–20 % de las comunas y barrios ya cambia más de un 3 %, mientras el precio pedido de las
# mismas viviendas se movía un −0,01 % (medido el 14-nov). Así que el precio se mide con las MISMAS viviendas (las
# que estuvieron en venta todo el mes) y, aparte, lo que entra (publicadas), lo que sale (retiradas) y cuántas
# bajaron o subieron de precio.
BARRIDO_COMPLETO = pd.Timestamp("2026-11-01")   # desde aquí se ven enteros los dos portales y el estado unificado
                                                # (src/movimientos.py): antes, el mes no se cuenta
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


def _resumen_mes(x: pd.Series) -> dict:
    return {"en_venta_todo": int(x["en_venta_todo"]), "bajaron": int(x["bajaron"]), "subieron": int(x["subieron"]),
            "bajada_mediana": _num(x["bajada_mediana_mismas"], 3), "cambio_mismas": _num(x["cambio_mismas"], 4),
            "publicadas": int(x["publicadas"]), "retiradas": int(x["retiradas"])}


def datos_mes(d: dict, desde: pd.Timestamp, hasta: pd.Timestamp) -> dict:
    """Un mes (de `desde` a `hasta`) en cifras, por inmueble en los dos portales (src/movimientos.py): lo de la ciudad
    y, por comuna, lo de las que tienen muestra."""
    from src import movimientos
    total = movimientos.periodo(d["ciudad"], desde, hasta).loc["total"]
    por = movimientos.periodo(d["ciudad"], desde, hasta, por="comuna")
    comunas = [{"codigo": str(c), "nombre": d["nombres"].get(c, c), **_resumen_mes(x)}
               for c, x in por.iterrows() if x["en_venta_todo"] >= MIN_COMUNA_MES]
    return {"mes": f"{desde:%Y-%m}", "nombre": NOMBRES_MES[desde.month - 1], "desde": desde, "hasta": hasta,
            "completo": bool(desde >= BARRIDO_COMPLETO), **_resumen_mes(total), "comunas": comunas}


def mes(d: dict) -> dict:
    """El último mes completo y, si también se vio entero, el anterior (para comparar)."""
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
    if t is not None and len(t):
        t = con_catastro(t, d["inmuebles"], ["comuna"])
    cat = lambda x: {k: _num(x.get(k), dec) for k, dec in (("viviendas_catastro", 0), ("apartamentos_parte_catastro", 2),
                                                           ("pisos_edificio", 0), ("genericos", 2), ("en_venta_parte", 4),
                                                           ("avaluo_m2_apartamento", 0), ("veces_avaluo_apartamento", 2),
                                                           ("avaluo_m2_casa", 0), ("veces_avaluo_casa", 2))}
    comunas = [] if t is None or not len(t) else [
        {"codigo": str(c), "nombre": nombre(c), "viviendas": _num(x["viviendas"]), "precio_m2": _num(x["precio_m2"]),
         "dias_publicadas": _num(x["dias_pub"]), "arriendos": _num(x["arriendos"]), "canon_m2": _num(x["canon_m2"]),
         "rentabilidad": _num(x["rentab"], 4), "barrios": x["barrios"], **cat(x)}
        for c, x in t.iterrows()]
    tb = por_barrio(d)
    barrios = [{"comuna": str(c), "barrio": b, **_campos(x)} for (c, b), x in tb.iterrows()]
    elegido = elegir_barrio(tb, d["hasta"], d["precio_m2"], {} if vistos is None else vistos)
    barrio_semana = ficha_barrio(tb, *elegido, d) if elegido else None
    cc = d["captar_comunas"]
    ext = cc[cc.index.isin(zonas.DE_EXTENSION)] if len(cc) else cc     # E10: lo público de captación, solo la ciudad
    captacion = {
        "dias": list(s.CAPTAR_DIAS), "sobre_similares": list(s.CAPTAR_SOBRE),
        "viejas": d["viejas"] - int(ext["viejas"].sum()) if len(ext) else d["viejas"],
        "piden_de_mas": d["captar_n"] - int(ext["sobre"].sum()) if len(ext) else d["captar_n"],
        "por_comuna": [{"codigo": str(c), "nombre": nombre(c), "viejas": int(x["viejas"]), "piden_de_mas": int(x["sobre"])}
                       for c, x in cc.iterrows()],
    }
    sel_bj = d["sel_bj"]
    bajadas = [ficha(r) for _, r in sel_bj.iterrows()] if len(sel_bj) else []
    return _limpio({
        "ciudad": ciudad, "desde": d["desde"], "hasta": d["hasta"], "generado": pd.Timestamp.now(tz="America/Bogota"),
        "mercado": {
            "viviendas": d["viviendas"], "publicadas": d["publicadas"], "retiradas": d["retiradas"],
            "bajadas": d["bajadas"], "bajada_mediana": d["bajada_mediana"], "precio_m2": d["precio_m2"],
            "arriendos": d["arriendos"], "canon_mediano": d["canon_mediano"], "rentabilidad": d["rentab"],
            **ciudad_catastro(d),
        },
        "oportunidades": {"evaluadas": d["evaluadas"], "bajo_similares_20": d["bajo_umbral"]},
        "comunas": comunas, "barrios": barrios, "barrio_semana": barrio_semana, "mes": mes(d), "bajadas": bajadas,
        "captacion": captacion, "credito": credito_semana(d),
        "area_metropolitana": area_metropolitana() if ciudad == "Cali" else None,
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
    c = s["credito"]
    print(f"   crédito al {c['hasta']}: tasa {credito.pct(c['tasa_pesos'])} · {c['creditos']} créditos "
          f"({'fiable' if c['fiable'] else 'no fiable'}) · {len(c['por_comuna'])} comunas con cuota" if c else
          "   sin crédito de vivienda (data/credito/semanas.parquet)")
    if not a.no_subir:
        guardar("contenido/semana.json", s)
        guardar(f"contenido/semana/{s['hasta']}.json", s)
        guardar(BARRIOS_SEMANA, todos)


if __name__ == "__main__":
    main()
