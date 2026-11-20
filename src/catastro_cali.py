"""
El catastro de Cali (E9.2, BACKLOG.md): cuántas viviendas hay en cada barrio, de qué tipo y en qué edificios, para
decir qué parte está en venta y dónde están los conjuntos grandes.

Fuente: IDESC (Alcaldía de Cali; Cali lleva su propio catastro, el IGAC no lo cubre), el mismo servidor WFS de los
barrios (src/zonas.py), abierto y sin clave:
    catastro:cat_bas_terrenos    756.277 registros, un predio por registro (las unidades de un edificio comparten el
                                 lote); sin geometría se baja en CSV en ~1,5 min. Sin avalúo y sin propietario.
    catastro:cat_bas_manzanas    15.743 manzanas con su polígono (~7 s)

Qué es una vivienda (medido el 17-nov con toda la ciudad): destino A (habitacional) con uso de vivienda y área
construida. El destino A incluye también parqueaderos (usos 8, 411, 412; ~12 m²), depósitos (6; ~3 m²) y lotes
sin construir (0). Usos de vivienda: 1 y 2 apartamentos en propiedad horizontal (~60 m², edificios de ~5 pisos);
3, 4 y 12 casas (93–248 m²); 5 y 13 apartamentos o casas en conjunto (según la dirección: «AP …» o «CA …»).
Resultado: 511.099 predios de vivienda (312.414 casas y 198.685 apartamentos). Un predio no es un hogar: una casa
con varios pisos arrendados cuenta una vez.

Barrio: el catastro usa sus códigos viejos (Valle del Lili es uno solo; el Acuerdo 0636 de 2026 lo parte), así que
casar por código solo cubre el 57 % de los predios. Cada predio va a la manzana de su número predial y cada manzana
al barrio oficial donde cae su punto interior: 95 % de los predios (lo demás es rural).

Salida (R2 `catastro/`, en local data/catastro/):
    predios.parquet      los predios, con su tipo y su barrio oficial (para E9.3, la captación por dirección)
    barrios.parquet      por barrio oficial: viviendas, apartamentos y casas, pisos y área típicos, conjuntos
    conjuntos.parquet    los lotes con apartamentos: dirección, apartamentos, pisos (para los agentes)
    meta.json            cuándo se bajó y cuántos

Direcciones (E9.3, `buscar`): la del catastro y la que escribe el agente se llevan a la misma forma (`canon`):
«Carrera 98B No. 48-127, Torre 3 Apto 436» y «KR 98 B # 48 - 127 TO 3 AP 436» son la misma. Con la unidad, el
predio (área construida, pisos, número predial, plano y uso del suelo); sin ella, el edificio (cuántos
apartamentos, sus áreas). El punto es el de la manzana (para buscar los similares).

Uso:
    python -m src.catastro_cali actualizar [--si-viejo 30]   # baja, resume y sube (solo si lo de R2 tiene > N días)
    python -m src.catastro_cali resumen                       # lo guardado, en pocas líneas
    python -m src.catastro_cali buscar "KR 98 B # 48-127 AP 436"
"""
from __future__ import annotations

import argparse
import io
import json
import os
import re
import unicodedata

import numpy as np
import pandas as pd

from src.config import BASE_DIR

WFS = "https://ws-idesc.cali.gov.co/geoserver/wfs"
PREDIOS, MANZANAS = "catastro:cat_bas_terrenos", "catastro:cat_bas_manzanas"
CAMPOS = ["id_predio", "comuna", "barrio", "idbarrio", "nom_barrio", "manzana", "terreno", "condicion", "npn",
          "direpred", "destinacio", "uso_princi", "total_pis1", "uso_id1", "area_cons1", "last_edite"]
PAGINA = 100_000
CARPETA = BASE_DIR / "data" / "catastro"
R2 = "catastro/"
USOS_VIVIENDA = {1, 2, 3, 4, 5, 12, 13}
USOS_APARTAMENTO = {1, 2}           # siempre apartamentos
USOS_PH_MIXTOS = {5, 13}            # apartamentos o casas en conjunto: según la dirección («AP …» / «CA …»)
VERSION = 2                         # 2: con la dirección normalizada y el punto de la manzana (E9.3); si lo de R2
                                    # es de otra versión, se rehace aunque sea reciente
MIN_CONJUNTO = 2                    # un lote es un edificio o conjunto con al menos estos apartamentos
CLAVE_MANZANA = slice(5, 17)        # en el número predial: zona, sector, comuna, barrio y manzana
CLAVE_LOTE = slice(5, 21)           # … y el terreno
UNIDAD = re.compile(r"\s+(TO|TORRE|BL|BLQ|BLOQUE|ET|ETAPA|AP|APTO|CA|CASA|PQ|GA|DP|LC|OF|PH|CS|ED|EDIF|MZ|LT|INT|"
                    r"UND|UNIDAD|CONJ|URB|SECTOR)\b.*$")


# ── Bajar ────────────────────────────────────────────────────────────────────────────────────────────────
def _wfs(params: dict, timeout: int = 600) -> str:
    import requests
    for intento in range(3):
        try:
            r = requests.get(WFS, params={"service": "WFS", "version": "2.0.0", "request": "GetFeature", **params},
                             timeout=timeout)
            r.raise_for_status()
            return r.text
        except Exception as e:              # noqa: BLE001 — se reintenta; al tercero, el error sube
            if intento == 2:
                raise
            print(f"   ⚠️  IDESC: {type(e).__name__}; reintento")


def bajar_predios() -> pd.DataFrame:
    """Todos los predios, sin geometría, por páginas. Sin repetidos: un predio en varias piezas cuenta una vez y los
    registros sin datos (id 0) se descartan."""
    partes, inicio = [], 0
    while True:
        texto = _wfs({"typeNames": PREDIOS, "outputFormat": "csv", "propertyName": ",".join(CAMPOS),
                      "sortBy": "id_predio", "count": PAGINA, "startIndex": inicio})
        t = pd.read_csv(io.StringIO(texto), dtype=str)
        partes.append(t)
        print(f"   ⬇️  predios {inicio:,}–{inicio + len(t):,}")
        if len(t) < PAGINA:
            break
        inicio += PAGINA
    t = pd.concat(partes, ignore_index=True).drop(columns=["FID"], errors="ignore")
    t = t[(t["id_predio"] != "0") & t["uso_id1"].notna()].drop_duplicates("id_predio")
    for c in ("total_pis1", "uso_id1", "area_cons1"):
        t[c] = pd.to_numeric(t[c], errors="coerce")
    return t.reset_index(drop=True)


def barrio_de_manzanas() -> pd.DataFrame:
    """Cada manzana del catastro con el barrio oficial y la comuna donde cae su punto interior (src/zonas.py). Si
    una manzana tiene varias piezas, manda la más grande."""
    from shapely.geometry import shape
    from src import zonas
    g = json.loads(_wfs({"typeNames": MANZANAS, "outputFormat": "application/json", "srsName": "EPSG:4326"}))
    k = ("tipo_avalu", "sector", "comuna", "barrio", "manzana")
    filas = []
    for f in g["features"]:
        p = f["properties"]
        if not f["geometry"] or any(p.get(c) is None for c in k):
            continue
        pt = shape(f["geometry"]).representative_point()
        filas.append({"manzana": "".join(p[c] for c in k), "Latitud": pt.y, "Longitud": pt.x,
                      "area": p.get("shape_area") or 0})
    m = zonas.asignar_zonas(pd.DataFrame(filas).assign(Barrio=None), "Cali")
    print(f"   ⬇️  {len(m):,} manzanas")
    return (m.sort_values("area", ascending=False).drop_duplicates("manzana")
            [["manzana", "barrio_oficial", "comuna", "Latitud", "Longitud"]]
            .rename(columns={"Latitud": "lat", "Longitud": "lon"}).reset_index(drop=True))


# ── Clasificar y resumir ─────────────────────────────────────────────────────────────────────────────────
def clasificar(t: pd.DataFrame, manzanas: pd.DataFrame) -> pd.DataFrame:
    """Añade `tipo` (apartamento | casa | None si no es vivienda), `lote`, `direccion` (la del lote, sin la unidad) y
    el barrio oficial y la comuna de su manzana."""
    t = t.copy()
    sufijo = t["direpred"].fillna("").str.extract(r"\b([A-Z]{2,4})\s*[\w-]*\s*$", expand=False).fillna("")
    vivienda = (t["destinacio"] == "A") & t["uso_id1"].isin(USOS_VIVIENDA) & (t["area_cons1"] > 0)
    apto = t["uso_id1"].isin(USOS_APARTAMENTO) | (t["uso_id1"].isin(USOS_PH_MIXTOS) & sufijo.str.startswith("AP"))
    t["tipo"] = np.where(~vivienda, None, np.where(apto, "apartamento", "casa"))
    t["lote"] = t["npn"].str[CLAVE_LOTE]
    t["direccion"] = t["direpred"].fillna("").str.replace(r"\s+", " ", regex=True).str.strip().str.replace(
        UNIDAD, "", regex=True)
    t["manzana_clave"] = t["npn"].str[CLAVE_MANZANA]
    partes = [canon(x) for x in t["direpred"].fillna("")]
    t["dir_base"] = [p[0] for p in partes]
    t["dir_unidad"] = [";".join(f"{k}={v}" for k, v in p[1].items()) for p in partes]
    t = t.drop(columns=["comuna", "barrio"]).rename(columns={"nom_barrio": "barrio_catastro"})
    return t.merge(manzanas.rename(columns={"manzana": "manzana_clave"}), on="manzana_clave", how="left")


# ── Las direcciones (E9.3): la misma forma para la del catastro y la que escribe el agente ─────────────────
VIAS = {"CALLE": "CL", "CLL": "CL", "CL": "CL", "C": "CL", "CALL": "CL",
        "CARRERA": "KR", "CRA": "KR", "KRA": "KR", "KR": "KR", "K": "KR", "CR": "KR", "CARR": "KR", "CRR": "KR",
        "AVENIDA": "AV", "AV": "AV", "AVE": "AV", "AVDA": "AV", "A": "AV",
        "DIAGONAL": "DG", "DG": "DG", "DIAG": "DG", "TRANSVERSAL": "TV", "TV": "TV", "TR": "TV", "TRANSV": "TV",
        "TRV": "TV"}
PUNTOS = {"NORTE": "N", "OESTE": "O", "W": "O", "SUR": "S", "ESTE": "E"}
# Torre, bloque, edificio e interior son lo mismo: el catastro los mezcla en un mismo conjunto («TO 1 AP 102» y
# «BLQ 1 AP 103»)
UNIDADES = {"APARTAMENTO": "AP", "APTO": "AP", "APT": "AP", "AP": "AP", "TORRE": "TO", "TO": "TO",
            "BLOQUE": "TO", "BLQ": "TO", "BL": "TO", "BLOQ": "TO", "EDIFICIO": "TO", "ED": "TO", "EDIF": "TO",
            "CASA": "CA", "CA": "CA", "INTERIOR": "TO", "INT": "TO", "IN": "TO", "ETAPA": "ET", "ET": "ET",
            "LOCAL": "LC", "LC": "LC", "OFICINA": "OF", "OF": "OF", "MANZANA": "MZ", "MZ": "MZ", "LOTE": "LT",
            "LT": "LT", "PISO": "PISO"}


def _tokens(texto: str) -> list[str]:
    t = unicodedata.normalize("NFKD", str(texto).upper()).encode("ascii", "ignore").decode()
    t = re.sub(r"\b(N[O°º]|NRO|NUM|NUMERO)\b\.?", " # ", t)                # «No. 48», «No. AV 2A» → «#»
    t = t.replace("#", " # ").replace("-", " - ")
    t = re.sub(r"[^A-Z0-9#\- ]", " ", t)
    t = re.sub(r"(\d)([A-Z])", r"\1 \2", t)
    t = re.sub(r"([A-Z])(\d)", r"\1 \2", t)
    return t.split()


def canon(texto: str) -> tuple[str | None, dict]:
    """La dirección en una forma única: (base, unidad). 'Carrera 98B No. 48-127, Torre 3 Apto 436' y
    'KR 98 B # 48 - 127 TO 3 AP 436' → ('KR 98 B # 48 127', {'TO': '3', 'AP': '436'}). None si no empieza
    por una vía (calle, carrera, avenida, diagonal, transversal)."""
    tk = _tokens(texto)
    if len(tk) < 3 or tk[0] not in VIAS or not any(x.isdigit() for x in tk[1:3]):
        return None, {}
    tk = [VIAS[tk[0]]] + [PUNTOS.get(x, x) for x in tk[1:]]
    if "#" in tk and "-" in tk[tk.index("#"):]:
        i = tk.index("-", tk.index("#"))
        base, resto = tk[:i + 2], tk[i + 2:]
    else:                                       # sin «-»: hasta la primera palabra de unidad
        j = next((k for k, x in enumerate(tk) if k > 1 and x in UNIDADES), len(tk))
        base, resto = tk[:j], tk[j:]
    unidad, clave = {}, None
    for x in resto:
        if x in UNIDADES:
            clave = UNIDADES[x]
            if unidad.get(clave):               # «ED 3 TO 2»: los dos, separados
                unidad[clave] += "/"
            unidad.setdefault(clave, "")
        elif clave:
            unidad[clave] += x
    unidad = {k: v.lstrip("0") or v for k, v in unidad.items() if v and k != "PISO"}
    return " ".join(x for x in base if x != "-"), unidad        # sin el guion: hay direcciones que no lo llevan


def es_direccion(texto: str | None) -> bool:
    """¿Parece una dirección («KR 98 B # 48 - 127») y no un barrio o unas coordenadas?"""
    return bool(texto) and "#" in _tokens(texto) and canon(texto)[0] is not None


def _unidad(s: str) -> dict:
    return dict(p.split("=", 1) for p in s.split(";") if "=" in p) if s else {}


def enlaces(predio: dict) -> dict:
    """El plano del predio (planoteca de la IDESC) y la consulta de su uso del suelo (Alcaldía)."""
    return {"plano": f"https://idesc.cali.gov.co/servicios/planoteca/planos.php?id_predio={predio['id_predio']}",
            "uso_suelo": f"https://www.cali.gov.co/tramites/uso-de-suelo/consulta/{predio['npn']}"}


def _predio(r: pd.Series) -> dict:
    p = {"id_predio": str(r["id_predio"]), "npn": r["npn"], "direccion": r["direpred"], "tipo": r["tipo"],
         "area": None if pd.isna(r["area_cons1"]) else float(r["area_cons1"]),
         "pisos": None if pd.isna(r["total_pis1"]) else int(r["total_pis1"]), "uso": r["uso_princi"],
         "destino": r["destinacio"], "editado": str(r["last_edite"])[:10]}
    return {**p, **enlaces(p)}


def buscar(texto: str, predios: pd.DataFrame | None = None) -> dict:
    """El predio de una dirección de Cali en el catastro. `estado`:
      unidad      un predio (el apartamento o la casa): `predio`, y el `edificio` si es uno
      edificio    la dirección es la de un edificio o conjunto y no se dijo cuál unidad (o no se encontró): el
                  `edificio` (apartamentos, pisos, áreas) y algunas `opciones`
      casas       varias casas con la misma dirección (el lote y una mejora, o pisos registrados aparte): `casas`
      varias      la unidad que se dijo puede ser varias («AP 101» en varias torres): `opciones`
      no          no está: `sugerencias` (direcciones parecidas de la misma vía)
    Siempre que se encuentra: `barrio_oficial`, `comuna`, `lat`, `lon` (el punto de la manzana)."""
    import difflib
    base, unidad = canon(texto)
    if base is None:
        return {"estado": "no", "sugerencias": [], "motivo": "no parece una dirección"}
    p = predios if predios is not None else leer("predios")
    en = p[p["dir_base"] == base]
    if en.empty:
        via = " ".join(base.split("#")[0].split())                     # «KR 98 B»
        mismas = p.loc[p["dir_base"].fillna("").str.startswith(via + " #"), "dir_base"].dropna().unique()
        if not len(mismas):
            mismas = p.loc[p["dir_base"].fillna("").str.startswith(base.split()[0] + " "), "dir_base"].dropna().unique()
        cerca = difflib.get_close_matches(base, list(mismas), n=6, cutoff=0.6)
        como_se_escribe = p[p["dir_base"].isin(cerca)].groupby("dir_base")["direccion"].agg(lambda x: x.mode().iloc[0])
        return {"estado": "no", "base": base, "sugerencias": [como_se_escribe.get(b, b) for b in cerca]}
    viv = en[en["tipo"].notna()]
    aptos = viv[viv["tipo"] == "apartamento"]
    fila = (viv if len(viv) else en).iloc[0]
    comun = {"base": base, "direccion": fila["direccion"], "barrio_oficial": fila["barrio_oficial"],
             "comuna": fila["comuna"], "lat": fila["lat"], "lon": fila["lon"]}
    edificio = None
    if len(aptos) >= MIN_CONJUNTO:
        a = aptos["area_cons1"]
        edificio = {"apartamentos": len(aptos), "casas": int((viv["tipo"] == "casa").sum()),
                    "pisos": int(aptos["total_pis1"].median()) if aptos["total_pis1"].notna().any() else None,
                    "area_p10": float(a.quantile(.1)), "area_mediana": float(a.median()), "area_p90": float(a.quantile(.9))}
    if unidad:
        casan = viv[[all(_unidad(u).get(k) == v for k, v in unidad.items()) for u in viv["dir_unidad"]]]
        if len(casan) == 1:
            return {"estado": "unidad", **comun, "predio": _predio(casan.iloc[0]), "edificio": edificio}
        if len(casan) > 1:
            return {"estado": "varias", **comun, "edificio": edificio,
                    "opciones": casan["direpred"].head(8).tolist()}
    if len(viv) == 1:                           # una casa (o un solo predio de vivienda en la dirección)
        return {"estado": "unidad", **comun, "predio": _predio(viv.iloc[0]), "edificio": None}
    if edificio is None and len(viv) > 1:       # varias casas con la misma dirección (el lote y una mejora, pisos…)
        return {"estado": "casas", **comun, "edificio": None,
                "casas": [{"area": float(r["area_cons1"]), "pisos": None if pd.isna(r["total_pis1"]) else int(r["total_pis1"])}
                          for _, r in viv.sort_values("area_cons1").iterrows()]}
    return {"estado": "edificio", **comun, "edificio": edificio,
            "opciones": viv["direpred"].head(6).tolist(), "unidad_buscada": unidad or None}


def conjuntos(t: pd.DataFrame) -> pd.DataFrame:
    """Los lotes con apartamentos (edificios y conjuntos): dirección, barrio, apartamentos, pisos (mediana de sus
    apartamentos: el máximo arrastra errores, como un conjunto de 5 pisos con una unidad «de 33») y área típica."""
    a = t[t["tipo"] == "apartamento"]
    g = a.groupby("lote")
    c = pd.DataFrame({
        "direccion": g["direccion"].agg(lambda x: x.mode().iloc[0] if len(x.mode()) else None),
        "barrio_oficial": g["barrio_oficial"].first(), "comuna": g["comuna"].first(),
        "apartamentos": g.size(), "pisos": g["total_pis1"].median(), "area": g["area_cons1"].median()})
    return c[c["apartamentos"] >= MIN_CONJUNTO].sort_values("apartamentos", ascending=False).reset_index()


def barrios(t: pd.DataFrame) -> pd.DataFrame:
    """Por barrio oficial: viviendas (predios de vivienda), apartamentos y casas, la parte de apartamentos, los pisos
    típicos de sus edificios (mediana por apartamento: el edificio donde vive el apartamento típico), el área
    típica, cuántos edificios o conjuntos y cuántos predios no son vivienda (comercio, oficinas…)."""
    v = t[t["tipo"].notna()]
    aptos, casas = v[v["tipo"] == "apartamento"], v[v["tipo"] == "casa"]
    clave = ["comuna", "barrio_oficial"]
    edificios = conjuntos(t).groupby(clave).size()
    otros = t[(t["destinacio"] != "A")].groupby(clave).size()
    b = pd.DataFrame({
        "viviendas": v.groupby(clave).size(), "apartamentos": aptos.groupby(clave).size(),
        "casas": casas.groupby(clave).size(),
        "pisos_edificio": aptos.groupby(clave)["total_pis1"].median(),
        "area_apartamento": aptos.groupby(clave)["area_cons1"].median(),
        "area_casa": casas.groupby(clave)["area_cons1"].median(),
        "edificios": edificios, "no_vivienda": otros}).fillna({"apartamentos": 0, "casas": 0, "edificios": 0,
                                                               "no_vivienda": 0})
    b = b[b["viviendas"].notna()]
    b["parte_apartamentos"] = b["apartamentos"] / b["viviendas"]
    return b.reset_index()


# ── Guardar y leer ───────────────────────────────────────────────────────────────────────────────────────
def _con_r2() -> bool:
    return all(os.environ.get(v) for v in ("R2_ACCOUNT_ID", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "R2_BUCKET"))


def _meta_r2() -> dict | None:
    from src import almacen
    s3, bucket = almacen._cliente()
    try:
        return json.loads(s3.get_object(Bucket=bucket, Key=f"{R2}meta.json")["Body"].read())
    except s3.exceptions.NoSuchKey:
        return None


RESUMEN = ("barrios.parquet", "conjuntos.parquet", "meta.json",    # lo que leen el canal y el informe (~250 KB)
           "avaluos_barrios.parquet")                                # y el avalúo por barrio (src/avaluos.py, E9.4)


def traer_resumen() -> None:
    """Los resúmenes de R2 a data/catastro/ (sin los predios, que solo hacen falta para E9.3)."""
    from src import almacen
    for f in RESUMEN:
        almacen.traer(R2 + f, BASE_DIR / "data")


def actualizar(si_viejo: int | None = None) -> None:
    """Baja el catastro, lo resume y lo guarda (y lo sube a R2). Con `si_viejo`, solo si lo de R2 tiene más de esos
    días (el catastro cambia despacio: el canal lo mira cada madrugada y lo rehace una vez al mes); si no, baja los
    resúmenes de R2. Si la IDESC falla, también: se usan los de la vez anterior."""
    if si_viejo is not None and _con_r2() and (meta := _meta_r2()):
        dias = (pd.Timestamp.now() - pd.Timestamp(meta["bajado"])).days
        if dias <= si_viejo and meta.get("version") == VERSION:
            print(f"✅ Catastro de hace {dias} días (se rehace a los {si_viejo}): se bajan sus resúmenes")
            traer_resumen()
            return
    try:
        _rehacer()
    except Exception as e:
        if _con_r2():
            print("⚠️  No se pudo rehacer el catastro: se usan los resúmenes de la vez anterior")
            traer_resumen()
            if pd.Timestamp.now(tz="America/Bogota").weekday() == 0:      # los lunes, al administrador
                from src import monitor
                monitor.enviar(f"*⚠️ Catastro de Cali*\nNo se pudo rehacer desde GitHub ({type(e).__name__}): las cifras "
                               f"van con el de la vez anterior (o sin él). Se puede rehacer desde el PC con "
                               f"`python -m src.catastro_cali actualizar`.")
        raise


def _rehacer() -> None:
    print("⬇️  Catastro de Cali (IDESC)")
    t = clasificar(bajar_predios(), barrio_de_manzanas())
    b, c = barrios(t), conjuntos(t)
    v = t[t["tipo"].notna()]
    meta = {"version": VERSION, "bajado": f"{pd.Timestamp.now():%Y-%m-%d}", "editado_hasta": str(t["last_edite"].max())[:10],
            "predios": len(t), "viviendas": len(v), "apartamentos": int((v["tipo"] == "apartamento").sum()),
            "casas": int((v["tipo"] == "casa").sum()), "con_barrio": round(float(v["barrio_oficial"].notna().mean()), 4),
            "barrios": len(b), "conjuntos": len(c)}
    CARPETA.mkdir(parents=True, exist_ok=True)
    columnas = ["id_predio", "npn", "lote", "direccion", "direpred", "dir_base", "dir_unidad", "tipo", "destinacio",
                "uso_id1", "uso_princi", "condicion", "total_pis1", "area_cons1", "barrio_catastro", "barrio_oficial",
                "comuna", "lat", "lon", "last_edite"]
    t[columnas].to_parquet(CARPETA / "predios.parquet", index=False, compression="zstd")
    b.to_parquet(CARPETA / "barrios.parquet", index=False, compression="zstd")
    c.to_parquet(CARPETA / "conjuntos.parquet", index=False, compression="zstd")
    (CARPETA / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))
    print(f"✅ {meta['viviendas']:,} viviendas ({meta['apartamentos']:,} apartamentos, {meta['casas']:,} casas) · "
          f"{meta['con_barrio']:.1%} con barrio oficial · {len(b)} barrios · {len(c):,} edificios o conjuntos")
    if _con_r2():
        from src import almacen
        for f in ("predios.parquet", "barrios.parquet", "conjuntos.parquet", "meta.json"):
            almacen.subir_archivo(CARPETA / f, R2 + f)
    try:                                    # el avalúo por barrio, con los barrios de este catastro (E9.4)
        from src import avaluos
        avaluos.actualizar(t[columnas])
    except Exception as e:  # noqa: BLE001 — sin el avalúo, el catastro sale igual
        print(f"⚠️  Avalúos sin rehacer: {type(e).__name__}: {e}")


def leer(cual: str = "barrios") -> pd.DataFrame | None:
    """barrios | conjuntos | predios (data/catastro/; los workflows lo bajan de R2). None si no está."""
    p = CARPETA / f"{cual}.parquet"
    return pd.read_parquet(p) if p.exists() else None


def meta() -> dict | None:
    p = CARPETA / "meta.json"
    return json.loads(p.read_text()) if p.exists() else None


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("accion", choices=["actualizar", "resumen", "buscar"])
    p.add_argument("direccion", nargs="?", help="(buscar) una dirección de Cali")
    p.add_argument("--si-viejo", type=int, help="solo si lo de R2 tiene más de estos días")
    a = p.parse_args()
    try:                                    # en local, las claves vienen de .env
        from src.almacen import _cargar_env
        _cargar_env()
    except ImportError:
        pass
    if a.accion == "actualizar":
        actualizar(a.si_viejo)
    elif a.accion == "buscar":
        print(json.dumps(buscar(a.direccion or ""), ensure_ascii=False, indent=1, default=str))
    else:
        print(meta())
        b = leer()
        if b is not None:
            print(b.sort_values("viviendas", ascending=False).head(10).round(2).to_string())


if __name__ == "__main__":
    main()
