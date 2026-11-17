"""
El crédito de vivienda: lo que prestan los bancos cada semana y a qué tasa, por municipio (E9.1, BACKLOG.md).

Fuente: Superintendencia Financiera, «Tasas de interés activas por tipo de crédito», en datos.gov.co (datos abiertos,
sin clave). Un corte por semana (los viernes, publicado con ~1 semana de retraso):
    qzsc-9esp   los dos últimos meses (lo que se pide cada día)
    w9zh-vetq   el histórico, desde el 29-sep-2023 (se carga una vez, con --historico)
Cada fila: banco, producto (compra de vivienda VIS o no VIS, en pesos o en UVR, leasing, libranza…), municipio, tasa
efectiva anual promedio, monto desembolsado y número de créditos. Desde el 29-sep-2023 la tasa en UVR NO incluye la
variación de la UVR (se paga además la inflación).

Se agrega en el servidor por semana, municipio y producto (créditos, monto y tasa × monto, para ponderar la tasa por lo
prestado) y se guarda la serie en
    credito/semanas.parquet      R2 (en local, data/credito/semanas.parquet)

Qué se puede decir y qué no (medido el 17-nov con sep-2023 a nov-2026):
  - La TASA es estable (Cali, compra en pesos: 14,3–15,2 % semana a semana): se publica la media de las últimas 4
    semanas, ponderada por monto.
  - El NÚMERO de créditos por municipio no es fiable semana a semana: entre el 14-ago y el 4-sep los grandes bancos
    casi desaparecen de Cali (Davivienda: 2, 1; Bancolombia: 3, 0) mientras en el país desembolsaban como siempre; y
    los desembolsos se amontonan en la última semana del mes. El volumen va en 4 semanas o por mes y solo si ninguna
    de sus semanas tiene hueco (`huecos`).
  - El volumen de Cali NO se compara con el del año pasado: su parte en el país baja de ~7 % (2025) a ~5 % (2026) y
    una parte es cómo reportan los bancos (el Fondo Nacional del Ahorro deja de aparecer en Cali desde agosto). En 8
    semanas al 6-nov: país −22 % frente a hace un año, Cali −44 %. Lo que se compara con el año pasado es la tasa
    (Cali, en pesos: 15,0 % frente a 12,3 %) y el volumen del país.
  - La cuota de una vivienda es una aproximación con supuestos fijos y a la vista (FINANCIADO, AÑOS, en pesos):
    no es una oferta de crédito.

Uso:
    python -m src.credito actualizar [--historico] [--avisar]   # baja lo nuevo, lo junta con la serie y la sube
    python -m src.credito resumen [--municipio 76001]           # la tasa y los créditos de las últimas 4 semanas
"""
from __future__ import annotations

import argparse
import os
import re
import time
import unicodedata
from functools import lru_cache

import numpy as np
import pandas as pd

from src.config import BASE_DIR

URL = "https://www.datos.gov.co/resource/{}.json"
RECIENTE, HISTORICO = "qzsc-9esp", "w9zh-vetq"
INICIO_HISTORICO = pd.Timestamp("2023-09-29")    # primer corte del histórico (desde ahí la UVR va aparte)
RUTA = BASE_DIR / "data" / "credito" / "semanas.parquet"
CLAVE = "credito/semanas.parquet"
PAIS = "pais"                                    # el total del país, en la columna `municipio`
CALI = "76001"
# La ciudad y su área metropolitana (las que se bajan por municipio; las demás ciudades usan la tasa del país)
MUNICIPIOS = {"76001": "Cali", "76364": "Jamundí", "76520": "Palmira", "76892": "Yumbo", "76130": "Candelaria"}
CIUDADES = {"Cali": CALI}                        # ciudad del producto → municipio
SEMANAS = 4                                      # la ventana de la tasa y del volumen
MIN_CREDITOS = 30                                # una tasa, con al menos estos créditos en la ventana
HUECO = 0.50                                     # una semana con hueco: la parte de la ciudad en el país, bajo la mitad
                                                 # de la habitual (ver `huecos`)
FINANCIADO, AÑOS, CUOTA_MAX = 0.70, 20, 0.30    # la cuota de ejemplo: 70 % financiado a 20 años; cuota ≤ 30 % del ingreso
FRENO = 0.70                                     # aviso al administrador: 8 semanas del país bajo el 70 % de hace un año
DIAS_SIN_DATOS = 21                              # aviso: la fuente no trae un corte nuevo en tres semanas
COMPRA = ("compra", "leasing", "libranza")       # «créditos para comprar vivienda» (sin construcción ni empleados)
MESES = ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto", "septiembre", "octubre",
         "noviembre", "diciembre"]


# ── Bajar de datos.gov.co ────────────────────────────────────────────────────────────────────────────────
def _consulta(dataset: str, select: str, where: str, group: str, pagina: int = 50_000) -> pd.DataFrame:
    """Una consulta SoQL agregada en el servidor, por páginas y con reintentos (el portal a veces tarda o falla)."""
    import requests
    filas, offset = [], 0
    while True:
        params = {"$select": select, "$where": where, "$group": group, "$order": group, "$limit": pagina,
                  "$offset": offset}
        for intento in range(4):
            try:
                r = requests.get(URL.format(dataset), params=params, timeout=900)
                r.raise_for_status()
                lote = r.json()
                break
            except Exception as e:          # noqa: BLE001 — se reintenta; al cuarto, el error sube
                if intento == 3:
                    raise
                print(f"   ⚠️  datos.gov.co: {type(e).__name__}; reintento en {20 * (intento + 1)} s")
                time.sleep(20 * (intento + 1))
        filas += lote
        if len(lote) < pagina:
            return pd.DataFrame(filas)
        offset += pagina


def _bajar(dataset: str, desde: pd.Timestamp | None = None, hasta: pd.Timestamp | None = None) -> pd.DataFrame:
    """Los créditos de vivienda de `dataset` por semana y producto: de los MUNICIPIOS y del país entero."""
    suma = ("sum(numero_de_creditos) as creditos, sum(montos_desembolsados) as monto, "
            "sum(tasa_efectiva_promedio * montos_desembolsados) as tasa_monto")
    where = "tipo_de_cr_dito = 'Vivienda'"
    if desde is not None:
        where += f" AND fecha_corte >= '{desde:%Y-%m-%d}'"
    if hasta is not None:
        where += f" AND fecha_corte < '{hasta:%Y-%m-%d}'"
    codigos = ", ".join(f"'{c}'" for c in MUNICIPIOS)
    mun = _consulta(dataset, f"fecha_corte, codigo_municipio, producto_de_cr_dito, {suma}",
                    f"{where} AND codigo_municipio in ({codigos})", "fecha_corte, codigo_municipio, producto_de_cr_dito")
    pais = _consulta(dataset, f"fecha_corte, producto_de_cr_dito, {suma}", where, "fecha_corte, producto_de_cr_dito")
    pais["codigo_municipio"] = PAIS
    t = pd.concat([mun, pais], ignore_index=True)
    if t.empty:
        return _vacia()
    return pd.DataFrame({
        "fecha": pd.to_datetime(t["fecha_corte"]).dt.normalize(),
        "municipio": t["codigo_municipio"].astype(str),
        "producto": t["producto_de_cr_dito"].astype(str),
        "creditos": pd.to_numeric(t["creditos"]).fillna(0).astype(int),
        "monto": pd.to_numeric(t["monto"]).fillna(0.0),
        "tasa_monto": pd.to_numeric(t["tasa_monto"]).fillna(0.0),
    })


def _vacia() -> pd.DataFrame:
    return pd.DataFrame({"fecha": pd.Series(dtype="datetime64[ns]"), "municipio": pd.Series(dtype=str),
                         "producto": pd.Series(dtype=str), "creditos": pd.Series(dtype=int),
                         "monto": pd.Series(dtype=float), "tasa_monto": pd.Series(dtype=float)})


def _juntar(serie: pd.DataFrame, nuevo: pd.DataFrame) -> pd.DataFrame:
    """Lo nuevo reemplaza, semana a semana, lo que ya había de esa semana (en el país y en los municipios por
    separado): si la Superfinanciera corrige un corte, gana la corrección."""
    if nuevo.empty:
        return serie
    ambito = lambda t: np.where(t["municipio"] == PAIS, PAIS, "municipios")
    clave_nueva = set(zip(nuevo["fecha"], ambito(nuevo)))
    quedan = serie[[k not in clave_nueva for k in zip(serie["fecha"], ambito(serie))]]
    return (pd.concat([quedan, nuevo], ignore_index=True)
            .sort_values(["fecha", "municipio", "producto"]).reset_index(drop=True))


def _con_r2() -> bool:
    return all(os.environ.get(v) for v in ("R2_ACCOUNT_ID", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "R2_BUCKET"))


def leer() -> pd.DataFrame:
    """La serie guardada (data/credito/; los workflows la bajan de R2 antes). Vacía si no hay."""
    return pd.read_parquet(RUTA) if RUTA.exists() else _vacia()


def actualizar(historico: bool = False) -> pd.DataFrame:
    """Baja los dos últimos meses (y, con `historico`, el histórico por trimestres: ~13 consultas de varios minutos,
    nunca en el flujo diario), los junta con la serie y la guarda (y la sube a R2 si hay claves)."""
    if _con_r2() and not RUTA.exists():
        from src import almacen
        try:
            almacen.traer(CLAVE, BASE_DIR / "data")
        except Exception as e:              # noqa: BLE001 — la primera vez no existe
            print(f"   (sin serie en R2: {type(e).__name__})")
    serie = leer()
    if historico:
        inicio = INICIO_HISTORICO
        while inicio < pd.Timestamp.now():
            fin = inicio + pd.DateOffset(months=3)
            print(f"⬇️  Histórico {inicio:%Y-%m-%d} → {fin:%Y-%m-%d}")
            serie = _juntar(serie, _bajar(HISTORICO, inicio, fin))
            inicio = fin
    print("⬇️  Los dos últimos meses")
    serie = _juntar(serie, _bajar(RECIENTE))
    RUTA.parent.mkdir(parents=True, exist_ok=True)
    serie.to_parquet(RUTA, index=False, compression="zstd")
    print(f"✅ {len(serie):,} filas · {serie['fecha'].min():%Y-%m-%d} → {serie['fecha'].max():%Y-%m-%d} → {RUTA}")
    if _con_r2():
        from src import almacen
        almacen.subir_archivo(RUTA, CLAVE)
    return serie


# ── Clasificar los productos ─────────────────────────────────────────────────────────────────────────────
def _sin_tildes(t: str) -> str:
    return unicodedata.normalize("NFKD", t).encode("ascii", "ignore").decode().lower()


@lru_cache(maxsize=64)
def clase(producto: str) -> tuple[str, str | None, str]:
    """(destino, segmento, moneda) de un producto de la Superfinanciera:
    destino: compra | leasing | libranza | construccion | empleados | otro; segmento: vis | no_vis | None;
    moneda: pesos | uvr."""
    p = _sin_tildes(producto)
    moneda = "uvr" if "uvr" in p else "pesos"
    segmento = "no_vis" if ("no vis" in p or "diferente de vis" in p) else ("vis" if re.search(r"\bvis\b", p) else None)
    if p.startswith("adquisicion leasing"):
        destino = "leasing"
    elif p.startswith("adquisicion de vivienda"):
        destino = "compra"
    elif p.startswith("libranza"):
        destino = "libranza"
    elif p.startswith("construccion"):
        destino = "construccion"
    elif "empleados" in p:
        destino = "empleados"
    else:
        destino = "otro"
    return destino, segmento, moneda


def clasificar(serie: pd.DataFrame) -> pd.DataFrame:
    c = pd.DataFrame([clase(p) for p in serie["producto"]], columns=["destino", "segmento", "moneda"],
                     index=serie.index)
    return serie.join(c)


# ── Lo que se cuenta ─────────────────────────────────────────────────────────────────────────────────────
def _tasa(t: pd.DataFrame, moneda: str = "pesos", segmento: str | None = None) -> float | None:
    """La tasa efectiva anual media de los créditos de compra (sin leasing), ponderada por monto, como fracción
    (0,149); None con menos de MIN_CREDITOS créditos."""
    x = t[(t["destino"] == "compra") & (t["moneda"] == moneda)]
    if segmento:
        x = x[x["segmento"] == segmento]
    if x["creditos"].sum() < MIN_CREDITOS or x["monto"].sum() <= 0:
        return None
    return float(x["tasa_monto"].sum() / x["monto"].sum() / 100)


def _ventana(s: pd.DataFrame, fin: pd.Timestamp, semanas: int = SEMANAS) -> pd.DataFrame:
    return s[(s["fecha"] > fin - pd.Timedelta(weeks=semanas)) & (s["fecha"] <= fin)]


def _compra(t: pd.DataFrame) -> pd.DataFrame:
    return t[t["destino"].isin(COMPRA)]


def huecos(s: pd.DataFrame, municipio: str) -> pd.Series:
    """Por semana, ¿le faltan créditos al municipio? Su parte en los créditos de compra del país por debajo de HUECO
    veces la habitual (la mediana de las 26 semanas anteriores). Medido el 17-nov con sep-2023 a nov-2026: marca las
    cuatro semanas en que los grandes bancos casi desaparecen de Cali (14-ago a 4-sep de 2026: 0,07–0,40) y ninguna
    más; fuera de ellas, el 1 % más bajo queda en 0,7. True = semana con hueco."""
    c = _compra(s)
    pais = c[c["municipio"] == PAIS].groupby("fecha")["creditos"].sum()
    mun = c[c["municipio"] == municipio].groupby("fecha")["creditos"].sum().reindex(pais.index, fill_value=0)
    parte = mun / pais.where(pais > 0)
    habitual = parte.shift(1).rolling(26, min_periods=8).median()
    return (parte < HUECO * habitual).fillna(False)


def _fiable(s: pd.DataFrame, municipio: str, fin: pd.Timestamp, semanas: int = SEMANAS) -> dict:
    """¿El volumen de las `semanas` que acaban en `fin` es creíble? Si ninguna tiene hueco (`huecos`)."""
    h = huecos(s, municipio)
    v = h[(h.index > fin - pd.Timedelta(weeks=semanas)) & (h.index <= fin)]
    return {"fiable": bool(len(v) and not v.any()), "semanas_con_hueco": int(v.sum())}


def _bloque(t: pd.DataFrame) -> dict:
    """Las cifras de un trozo de la serie (un municipio o el país, unas semanas)."""
    compra = _compra(t)
    n = int(compra["creditos"].sum())
    return {
        "tasa_pesos": _tasa(t, "pesos"), "tasa_uvr": _tasa(t, "uvr"),
        "tasa_vis_pesos": _tasa(t, "pesos", "vis"), "tasa_no_vis_pesos": _tasa(t, "pesos", "no_vis"),
        "creditos": n, "monto": float(compra["monto"].sum()),
        "parte_vis": float(compra.loc[compra["segmento"] == "vis", "creditos"].sum() / n) if n else None,
        "parte_uvr": float(compra.loc[compra["moneda"] == "uvr", "creditos"].sum() / n) if n else None,
    }


def resumen(serie: pd.DataFrame, municipio: str = CALI, hasta: pd.Timestamp | None = None,
            semanas: int = SEMANAS) -> dict | None:
    """Las últimas `semanas` semanas con datos (hasta `hasta`) del municipio: tasas, créditos y si el volumen es
    fiable; las mismas cifras de las `semanas` anteriores y de hace un año; la tasa del país (y su volumen de 8
    semanas frente a hace un año); y la serie del último año (53 cortes, tasa móvil de `semanas` semanas) para el
    gráfico. None si no hay datos."""
    if serie.empty:
        return None
    s = clasificar(serie)
    fechas = s.loc[s["municipio"] == municipio, "fecha"]
    if hasta is not None:
        fechas = fechas[fechas <= hasta]
    if fechas.empty:
        return None
    fin = fechas.max()
    mun, pais = s[s["municipio"] == municipio], s[s["municipio"] == PAIS]
    actual = {**_bloque(_ventana(mun, fin, semanas)), **_fiable(s, municipio, fin, semanas)}
    antes_fin = fin - pd.Timedelta(weeks=semanas)
    antes = {**_bloque(_ventana(mun, antes_fin, semanas)), **_fiable(s, municipio, antes_fin, semanas)}
    año_fin = fin - pd.Timedelta(weeks=52)
    año = ({**_bloque(_ventana(mun, año_fin, semanas)), **_fiable(s, municipio, año_fin, semanas)}
           if año_fin - pd.Timedelta(weeks=semanas) >= s["fecha"].min() else None)
    p = _bloque(_ventana(pais, fin, semanas))
    # El volumen del país frente a hace un año, en 8 semanas (dos finales de mes: los desembolsos se amontonan ahí)
    ocho = int(_compra(_ventana(pais, fin, 8))["creditos"].sum())
    ocho_año = int(_compra(_ventana(pais, año_fin, 8))["creditos"].sum()) if año is not None else 0
    graf = []
    for f in sorted(mun.loc[mun["fecha"] <= fin, "fecha"].unique())[-53:]:
        v = _ventana(mun, pd.Timestamp(f), semanas)
        graf.append({"fecha": f"{pd.Timestamp(f):%Y-%m-%d}", "tasa_pesos": _tasa(v, "pesos"), "tasa_uvr": _tasa(v, "uvr"),
                     "creditos": int(_compra(mun[mun["fecha"] == f])["creditos"].sum())})
    return {"municipio": municipio, "nombre": MUNICIPIOS.get(municipio, "Colombia"), "hasta": f"{fin:%Y-%m-%d}",
            "desde": f"{fin - pd.Timedelta(weeks=semanas) + pd.Timedelta(days=1):%Y-%m-%d}", "semanas": semanas,
            **actual, "antes": antes, "hace_un_año": año,
            "pais": {**{k: p[k] for k in ("tasa_pesos", "tasa_uvr", "creditos")}, "creditos_8": ocho,
                     "creditos_8_año": ocho_año or None, "cambio_año": ocho / ocho_año - 1 if ocho_año else None},
            "serie": graf}


def _mitad(f: pd.Series) -> pd.Series:
    """El mes de cada semana: el de su martes (el corte es el viernes y la semana va de sábado a viernes)."""
    return (f - pd.Timedelta(days=3)).dt.to_period("M")


def _mes(s: pd.DataFrame, municipio: str, periodo: pd.Period) -> dict | None:
    m = s[(s["municipio"] == municipio) & (_mitad(s["fecha"]) == periodo)]
    if m.empty:
        return None
    semanas = int(m["fecha"].nunique())
    b = _bloque(m)
    # El mes es fiable si su parte en el país está cerca de la habitual (con ventanas del tamaño del mes)
    fin = m["fecha"].max()
    fia = _fiable(s, municipio, fin, semanas)
    return {"mes": f"{periodo}", "nombre": MESES[periodo.month - 1], "año": periodo.year, "semanas": semanas,
            **b, "creditos_semana": b["creditos"] / semanas, **fia}


def mes(serie: pd.DataFrame, municipio: str = CALI, hasta: pd.Timestamp | None = None) -> dict | None:
    """El último mes completo con datos antes de `hasta` (sus semanas: las de corte en el mes, por su martes), con
    el anterior y el mismo mes del año pasado. El volumen se compara por semana (un mes tiene 4 o 5 cortes)."""
    if serie.empty:
        return None
    s = clasificar(serie)
    hasta = pd.Timestamp(hasta) if hasta is not None else s["fecha"].max() + pd.Timedelta(days=1)
    ultimo = s.loc[(s["municipio"] == municipio) & (s["fecha"] < hasta), "fecha"].max()
    if pd.isna(ultimo):
        return None
    # El mes de la última semana está completo si el martes de la semana siguiente ya es del mes siguiente
    martes = ultimo - pd.Timedelta(days=3)
    periodo = martes.to_period("M")
    if (ultimo + pd.Timedelta(days=4)).month == martes.month:
        periodo -= 1
    actual = _mes(s, municipio, periodo)
    if actual is None:
        return None
    return {**actual, "anterior": _mes(s, municipio, periodo - 1), "hace_un_año": _mes(s, municipio, periodo - 12)}


def alertas(serie: pd.DataFrame, hoy: pd.Timestamp | None = None) -> list[str]:
    """Avisos para el administrador: la fuente no trae cortes nuevos; la última semana de Cali tiene hueco (su
    volumen no se publica); o los créditos de compra del país en 8 semanas caen bajo el FRENO de hace un año (el de
    Cali no se compara: ver arriba)."""
    if serie.empty:
        return ["La serie del crédito de vivienda está vacía"]
    hoy = pd.Timestamp(hoy or pd.Timestamp.now()).normalize()
    s = clasificar(serie)
    avisos = []
    ultimo = s["fecha"].max()
    if (hoy - ultimo).days > DIAS_SIN_DATOS:
        avisos.append(f"El crédito de vivienda (Superfinanciera) no trae cortes nuevos desde el {ultimo:%d-%m-%Y}")
    h = huecos(s, CALI)
    if len(h) and h.iloc[-1]:
        avisos.append(f"La semana al {h.index[-1]:%d-%m} los bancos casi no reportan créditos en Cali (hueco): su "
                      f"volumen no se publica hasta que pasen 4 semanas sin hueco")
    pais = s[s["municipio"] == PAIS]
    a = _compra(_ventana(pais, ultimo, 8))["creditos"].sum()
    b = _compra(_ventana(pais, ultimo - pd.Timedelta(weeks=52), 8))["creditos"].sum()
    if b and a < FRENO * b:
        avisos.append(f"Los créditos para comprar vivienda en el país: {a:,} en 8 semanas, un {1 - a / b:.0%} menos "
                      f"que hace un año ({b:,})".replace(",", "."))
    return avisos


# ── La cuota ─────────────────────────────────────────────────────────────────────────────────────────────
def cuota(precio: float, tasa: float, financiado: float = FINANCIADO, años: int = AÑOS) -> float:
    """La cuota mensual de un crédito en pesos a cuota fija: `financiado` del precio, a `años`, con la tasa efectiva
    anual `tasa` (0,149)."""
    i = (1 + tasa) ** (1 / 12) - 1
    k = años * 12
    return precio * financiado * i / (1 - (1 + i) ** -k)


def ingreso(cuota_mes: float) -> float:
    """El ingreso del hogar para que la cuota no pase de CUOTA_MAX."""
    return cuota_mes / CUOTA_MAX


@lru_cache(maxsize=8)
def tasa_actual(ciudad: str = "Cali") -> dict | None:
    """La tasa para la cuota: la media de compra en pesos de las últimas semanas en la ciudad (o en el país, si la
    ciudad no se baja o no tiene créditos suficientes). {"tasa", "hasta", "donde"} o None sin serie."""
    serie = leer()
    municipio = CIUDADES.get(ciudad)
    r = resumen(serie, municipio) if municipio else None
    if r and r["tasa_pesos"]:
        return {"tasa": r["tasa_pesos"], "hasta": r["hasta"], "donde": r["nombre"]}
    r = resumen(serie, PAIS)
    return {"tasa": r["tasa_pesos"], "hasta": r["hasta"], "donde": "Colombia"} if r and r["tasa_pesos"] else None


def para_ficha(precio: float | None, ciudad: str = "Cali") -> dict | None:
    """La cuota de una oportunidad en cifras (la ficha del canal y de las piezas de redes); None sin datos."""
    t = tasa_actual(ciudad or "Cali")
    if not t or precio is None or pd.isna(precio):
        return None
    c = cuota(float(precio), t["tasa"])
    return {"cuota": int(round(c)), "ingreso": int(round(ingreso(c))), "tasa": round(t["tasa"], 4),
            "financiado": FINANCIADO, "años": AÑOS, "hasta": t["hasta"], "donde": t["donde"]}


# ── Textos ───────────────────────────────────────────────────────────────────────────────────────────────
def pct(x: float, dec: int = 1) -> str:
    return f"{x * 100:.{dec}f}%".replace(".", ",")


def millones(x: float, dec: int = 1) -> str:
    return f"${x / 1e6:,.{dec}f} M".replace(",", "X").replace(".", ",").replace("X", ".")


def fecha_corta(f: str) -> str:
    d = pd.Timestamp(f)
    return f"{d.day} {MESES[d.month - 1][:3]}"


def texto_cuota(precio: float, ciudad: str = "Cali") -> str | None:
    """'Cuota aproximada: $3,0 M al mes (70 % financiado a 20 años al 14,9 %, la tasa media de Cali)'."""
    f = para_ficha(precio, ciudad)
    if not f:
        return None
    return (f"Cuota aproximada: {millones(f['cuota'])} al mes ({f['financiado']:.0%} financiado a {f['años']} años al "
            f"{pct(f['tasa'])}, la tasa media de {f['donde']})").replace("70%", "70 %")


def linea_tasa(ciudad: str = "Cali") -> str | None:
    """'Crédito de vivienda en Cali: tasa media 15,0% en pesos y 8,3% + la inflación en UVR (4 semanas al 6 nov)';
    None si la ciudad no tiene serie."""
    municipio = CIUDADES.get(ciudad)
    r = resumen(leer(), municipio) if municipio else None
    if not r or not r["tasa_pesos"]:
        return None
    uvr = f" y {pct(r['tasa_uvr'])} + la inflación en UVR" if r["tasa_uvr"] else ""
    return (f"Crédito de vivienda en {ciudad}: tasa media {pct(r['tasa_pesos'])} en pesos{uvr} "
            f"({r['semanas']} semanas al {fecha_corta(r['hasta'])}, Superintendencia Financiera)")


def cambio_pts(a: float | None, b: float | None) -> str | None:
    """'−0,2 puntos' / '+0,3 puntos' / 'igual'."""
    if a is None or b is None:
        return None
    v = round((a - b) * 100, 1) + 0.0
    return "igual" if v == 0 else f"{v:+.1f} puntos".replace(".", ",").replace("-", "−")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("accion", choices=["actualizar", "resumen"])
    p.add_argument("--historico", action="store_true", help="vuelve a bajar el histórico (desde sep-2023)")
    p.add_argument("--avisar", action="store_true", help="los lunes, avisa al administrador si algo va mal")
    p.add_argument("--municipio", default=CALI)
    a = p.parse_args()
    try:                                    # en local, las claves vienen de .env
        from src.almacen import _cargar_env
        _cargar_env()
    except ImportError:
        pass
    serie = actualizar(a.historico) if a.accion == "actualizar" else leer()
    r = resumen(serie, a.municipio)
    if r:
        f = lambda x: pct(x) if x is not None else "—"
        año = (r["hace_un_año"] or {}).get("tasa_pesos")
        cambio = r["pais"]["cambio_año"]
        print(f"{r['nombre']}, {r['semanas']} semanas al {r['hasta']}: tasa en pesos {f(r['tasa_pesos'])} "
              f"(antes {f(r['antes']['tasa_pesos'])}; hace un año {f(año)}; país {f(r['pais']['tasa_pesos'])}) · UVR "
              f"{f(r['tasa_uvr'])} + UVR · {r['creditos']:,} créditos de compra "
              f"({'fiable' if r['fiable'] else 'NO fiable'}) · país en 8 semanas frente a hace un año: "
              f"{f'{cambio:+.0%}' if cambio is not None else '—'}")
        m = mes(serie, a.municipio)
        if m:
            print(f"   {m['nombre']} {m['año']}: tasa {f(m['tasa_pesos'])} · {m['creditos']:,} créditos en {m['semanas']} "
                  f"semanas ({'fiable' if m['fiable'] else 'NO fiable'})")
    avisos = alertas(serie)
    for x in avisos:
        print(f"⚠️  {x}")
    hoy = pd.Timestamp.now(tz="America/Bogota")
    if a.avisar and avisos and hoy.weekday() == 0:
        from src import monitor
        monitor.enviar("*🏦 Crédito de vivienda*\n" + "\n".join(f"• {x}" for x in avisos))


if __name__ == "__main__":
    main()
