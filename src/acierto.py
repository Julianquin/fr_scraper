"""
¿Acierta la alerta? (E2): las viviendas que la alerta marca (nuevas y ≥20 % bajo sus similares) ¿salen del mercado
antes que las demás?

Cada lunes desde que hay barrido, se reconstruye el mercado de ese día: los avisos publicados entonces (también los
que después se retiraron) con el precio de ese día. Sobre los avisos nuevos de esa semana se aplica la misma regla de
la alerta (src/oportunidades.py: similares, umbral del 20 % y descartes). Luego cada uno se sigue hasta hoy:
  - se retiró (y no volvió con otro número), o
  - sigue publicado, y cuenta como «aún sin saber» (censurado).
Se miden los días en el mercado desde la publicación: Kaplan-Meier por grupo y un log-rank estratificado por ciudad y
quinto de precio total («las marcadas se retiran ×1,12 de lo esperado en viviendas de precio parecido»). Se iguala el
precio total porque lo caro se retira mucho más despacio en todas partes y las marcadas casi nunca son caras: sin
igualarlo, parte de su ventaja es solo eso.

Se mide en las 9 ciudades con barrios oficiales porque la regla es la misma y Cali sola da pocos casos (~10 marcadas
por semana). Una baja se confirma tras AUSENCIAS_PARA_BAJA días sin ver el aviso: hasta ese día del final se sabe.
Solo FincaRaíz: Metrocuadrado no da la fecha de publicación. **Retirado no es vendido**: también se retira lo que el
dueño deja de vender o cambia de inmobiliaria sin republicar.

Medido el 20-nov (bajas conocidas hasta el 15-nov; 9.903 avisos nuevos con similares, 724 retirados): a los 21 días
se retiró el 15 % de las marcadas, el 13 % de las que están en línea y el 10 % de las más caras que sus similares.
Igualando el precio total: marcadas ×1,12 (p = 0,24), más caras ×0,95 (p = 0,40); sin igualarlo, ×1,21 y ×0,86. Por
precio total: el 20 % más caro de cada ciudad, 6 % a los 21 días, frente al 12–15 % del resto (lo barato no se retira
antes). Aún no se puede decir que la alerta acierte: hacen falta más semanas.

    python -m src.acierto                  # mide y lo muestra
    python -m src.acierto --avisar         # y se lo manda al administrador (oportunidades.yml, los lunes)
    python -m src.acierto --subir          # deja acierto/resumen.json en R2 (cifras agregadas, para el PDF)
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd

from src.config import BASE_DIR
from src.consolidar import AUSENCIAS_PARA_BAJA, rutas

CIUDADES = ["Cali", "Bogotá", "Medellín", "Barranquilla", "Bucaramanga", "Manizales", "Armenia", "Pereira", "Cartagena"]
UMBRAL = 0.20                    # el de la alerta (src/oportunidades.py --umbral)
DIAS = (7, 14, 21, 28)
MARCADAS, ENTRE, EN_LINEA, CARAS = "marcadas", "10–20 % bajo", "en línea", "más caras"
GRUPOS = {MARCADAS: "Marcadas por la alerta (≥20 % bajo sus similares)", ENTRE: "Entre 10 y 20 % bajo",
          EN_LINEA: "En línea (±10 %)", CARAS: "Más caras (≥10 % sobre sus similares)"}
SALIDA = BASE_DIR / "reports" / "acierto"
R2_RESUMEN = "acierto/resumen.json"


# ── El mercado de cada lunes ─────────────────────────────────────────────────────────────────────────────────
def _avisos() -> pd.DataFrame:
    """Todos los avisos de venta (activos y retirados) con cuándo se vieron por primera vez y si se republicaron."""
    from src import fuente_http
    t = pd.concat([fuente_http.dataset_activo(), fuente_http.dataset_activo(retirados=True)], ignore_index=True)
    ctrl = pd.read_parquet(rutas("venta").avisos, columns=["id_inmueble", "primera_vez", "republicado_como"])
    t = t.merge(ctrl.astype({"id_inmueble": str}), on="id_inmueble", how="left")
    return t.assign(primera_vez=pd.to_datetime(t["primera_vez"]))


def _en(df: pd.DataFrame, cambios: pd.DataFrame, dia: pd.Timestamp) -> pd.DataFrame:
    """Los avisos publicados el `dia` (vistos ya y no retirados), con el precio de ese día: el de antes del primer
    cambio posterior."""
    d = df[(df["primera_vez"] <= dia) & (df["fecha_baja"].isna() | (df["fecha_baja"] > dia))].copy()
    despues = cambios[cambios["fecha"] > f"{dia:%Y-%m-%d}"].drop_duplicates("id_inmueble", keep="first")
    d["Precio"] = d["id_inmueble"].map(despues.set_index("id_inmueble")["precio_antes"]).fillna(d["Precio"])
    d["precio_m2"] = d["Precio"] / d["Area_m2"]
    return d


def cohortes(ciudades: list[str] = CIUDADES) -> tuple[pd.DataFrame, pd.Timestamp]:
    """Los avisos nuevos de cada semana con su descuento frente a sus similares de entonces, su grupo y su
    seguimiento (`retirado`, `dias`). Devuelve también el último día con las bajas ya conocidas."""
    from src import fuente_http, oportunidades as op
    from src.viz_map import limpiar_ciudad
    meta = json.loads(rutas("venta").meta.read_text())
    ultimo, primero = pd.Timestamp(meta["ultimo_barrido"]), pd.Timestamp(meta["barridos"][0])
    fin = ultimo - pd.Timedelta(days=AUSENCIAS_PARA_BAJA)
    op.FUENTE = "http"
    todo = _avisos()
    ev = fuente_http.eventos(("venta",))
    cambios = ev[ev["evento"].isin(["cambio_precio", "correccion_precio"])].sort_values("fecha")
    # Hasta el último barrido: la semana que acaba después de `fin` trae también avisos publicados antes de él
    lunes = pd.date_range(primero + pd.Timedelta(days=1), ultimo, freq="W-MON")
    partes = []
    for ciudad in ciudades:
        t0 = time.time()
        df = op.cargar(ciudad, limpiar_ciudad(todo, ciudad))
        for dia in lunes:
            pool = _en(df, cambios, dia)
            nuevos = pool[pool["first_seen"].between(dia - pd.Timedelta(days=6), dia)]
            c = op.calificar(pool, nuevos, op.historial(dia))
            if len(c):
                partes.append(c.assign(ciudad=ciudad, lunes=dia))
        print(f"   {ciudad}: {sum(len(p) for p in partes if p['ciudad'].iat[0] == ciudad):,} avisos nuevos "
              f"({time.time() - t0:.0f} s)")
    c = pd.concat(partes, ignore_index=True).drop_duplicates("id_inmueble", keep="first")
    c = c[c["first_seen"] < fin].copy()
    c["retirado"] = c["fecha_baja"].notna() & (c["fecha_baja"] <= fin) & c["republicado_como"].isna()
    c["dias"] = np.where(c["retirado"], (c["fecha_baja"] - c["first_seen"]).dt.days, (fin - c["first_seen"]).dt.days)
    d = c["descuento"]
    c["grupo"] = np.select([d.isna(), (d >= UMBRAL) & c["motivo_descarte"].isna(), d >= UMBRAL, d >= .1, d > -.1],
                           [None, MARCADAS, None, ENTRE, EN_LINEA], default=CARAS)
    # El quinto de precio total de su ciudad (1, lo más barato): lo caro se retira mucho más despacio en todas partes
    # (el 20-nov, el 6 % a los 21 días frente al 12–15 %) y las marcadas casi nunca lo son; se compara dentro de él
    con = c["grupo"].notna()
    c.loc[con, "quinto"] = c[con].groupby("ciudad")["Precio"].transform(
        lambda p: pd.qcut(p, 5, labels=False, duplicates="drop")) + 1
    # El top 10 de cada lunes: lo que la alerta habría enviado en esa ciudad
    top = set()
    for _, g in c.groupby(["ciudad", "lunes"]):
        top |= set(op.seleccionar(g, UMBRAL, 10)["id_inmueble"])
    c["top10"] = c["id_inmueble"].isin(top)
    return c, fin


# ── La medida ────────────────────────────────────────────────────────────────────────────────────────────────
def retirada_a(dias: np.ndarray, evento: np.ndarray, h: int) -> float:
    """Kaplan-Meier: la parte retirada a los `h` días de publicada (lo que sigue publicado cuenta hasta hoy)."""
    s = 1.0
    for t in np.unique(dias[evento & (dias <= h)]):
        s *= 1 - ((dias == t) & evento).sum() / (dias >= t).sum()
    return 1 - s


def frente_al_resto(c: pd.DataFrame, grupo: pd.Series, resto: pd.Series,
                    por: tuple[str, ...] = ("ciudad", "quinto")) -> dict:
    """Log-rank estratificado (por defecto, por ciudad y quinto de precio total: viviendas de precio parecido):
    retiradas del grupo frente a las esperadas si se retirara como el resto."""
    from scipy import stats
    O = E = V = 0.0
    for _, g in c.assign(_g=grupo, _r=resto).groupby(list(por)):
        a, b = g[g["_g"]], g[g["_r"]]
        d1, e1, d2, e2 = a["dias"].to_numpy(), a["retirado"].to_numpy(), b["dias"].to_numpy(), b["retirado"].to_numpy()
        for t in np.unique(np.r_[d1[e1], d2[e2]]):
            n1, n2 = (d1 >= t).sum(), (d2 >= t).sum()
            o1 = ((d1 == t) & e1).sum()
            o, n = o1 + ((d2 == t) & e2).sum(), n1 + n2
            if n1 and n2 and n > 1:
                O, E = O + o1, E + o * n1 / n
                V += o * (n1 / n) * (n2 / n) * (n - o) / (n - 1)
    p = float(2 * (1 - stats.norm.cdf(abs(O - E) / np.sqrt(V)))) if V else None
    return {"n": int(grupo.sum()), "retiradas": int(O), "esperadas": round(E, 1),
            "veces": round(O / E, 2) if E else None, "p": round(p, 3) if p is not None else None}


def medir(c: pd.DataFrame, fin: pd.Timestamp) -> dict:
    con = c["grupo"].notna()
    r = {"fecha": f"{pd.Timestamp.now(tz='America/Bogota'):%Y-%m-%d}", "bajas_hasta": f"{fin:%Y-%m-%d}",
         "desde": f"{c['first_seen'].min():%Y-%m-%d}", "ciudades": sorted(c["ciudad"].unique()),
         "avisos": int(con.sum()), "retirados": int(c.loc[con, "retirado"].sum()), "grupos": {}}
    for g in GRUPOS:
        x = c[c["grupo"] == g]
        r["grupos"][g] = {"n": len(x), "retiradas": int(x["retirado"].sum()),
                          **{f"a_{h}_dias": round(retirada_a(x["dias"].to_numpy(), x["retirado"].to_numpy(), h), 4)
                             for h in DIAS}}
    resto = lambda g: con & (c["grupo"] != g)
    r["frente_al_resto"] = {g: frente_al_resto(c, c["grupo"] == g, resto(g)) for g in GRUPOS}
    m = c["grupo"] == MARCADAS
    r["marcadas_cali"] = frente_al_resto(c, m & (c["ciudad"] == "Cali"), resto(MARCADAS) & (c["ciudad"] == "Cali"))
    r["top10"] = frente_al_resto(c, c["top10"], con & ~c["top10"] & (c["grupo"] != MARCADAS))
    # Sin igualar el precio total (solo por ciudad): para ver cuánto pesa el precio
    r["marcadas_sin_igualar_precio"] = frente_al_resto(c, m, resto(MARCADAS), por=("ciudad",))
    r["por_quinto"] = {int(q): round(retirada_a(x["dias"].to_numpy(), x["retirado"].to_numpy(), 21), 4)
                       for q, x in c[con].groupby("quinto")}
    return r


def _n(x: int) -> str:
    return f"{x:,}".replace(",", ".")


def _dec(x: float, d: int = 2) -> str:
    return f"{x:.{d}f}".replace(".", ",")


def _veces(x: dict) -> str:
    if x["veces"] is None:
        return "sin datos"
    p = x["p"]
    certeza = "claro" if p < 0.01 else "probable" if p < 0.05 else "aún no claro"
    return f"×{_dec(x['veces'])} ({x['retiradas']} retiradas de {_n(x['n'])}; p = {_dec(p, 3)}: {certeza})"


def texto(r: dict) -> str:
    """El mensaje al administrador (solo cifras agregadas)."""
    f = r["frente_al_resto"]
    g, h = r["grupos"], 21
    fila = lambda k: f"• {GRUPOS[k]}: {g[k][f'a_{h}_dias']:.0%}"
    return "\n".join([
        f"*🎯 ¿Acierta la alerta? · {r['fecha']}*",
        f"Avisos nuevos desde el {r['desde']} en {len(r['ciudades'])} ciudades ({_n(r['avisos'])}; "
        f"{_n(r['retirados'])} ya retirados). Bajas conocidas hasta el {r['bajas_hasta']}.",
        "",
        f"*Retiradas a los {h} días de publicadas:*",
        *[fila(k) for k in GRUPOS],
        "",
        "*Frente a viviendas de precio total parecido en su ciudad* (cuántas se retiran de las que se esperaría):",
        f"• Marcadas: {_veces(f[MARCADAS])}",
        f"• El top 10 que se envía: {_veces(r['top10'])}",
        f"• Más caras que sus similares: {_veces(f[CARAS])}",
        f"• Solo Cali, marcadas: {_veces(r['marcadas_cali'])}",
        f"• Marcadas sin igualar el precio total: {_veces(r['marcadas_sin_igualar_precio'])}",
        "",
        "*Por precio total* (retiradas a los 21 días; 1 = el 20 % más barato de su ciudad): "
        + " · ".join(f"{q}: {v:.0%}" for q, v in r["por_quinto"].items()),
        "",
        "_Retirado no es vendido. Por encima de ×1 se retiran antes; con p < 0,05 es poco probable que sea azar._",
    ])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--avisar", action="store_true", help="manda el resultado al administrador")
    p.add_argument("--subir", action="store_true", help="deja acierto/resumen.json en R2")
    a = p.parse_args()
    print("🎯 ¿Acierta la alerta?")
    c, fin = cohortes()
    r = medir(c, fin)
    SALIDA.mkdir(parents=True, exist_ok=True)
    c[["ciudad", "lunes", "id_inmueble", "first_seen", "Tipo_propiedad", "barrio_oficial", "descuento", "confianza",
       "motivo_descarte", "grupo", "top10", "fecha_baja", "retirado", "dias"]].to_csv(
        SALIDA / f"cohortes_{r['fecha']}.csv", index=False)
    (SALIDA / "resumen.json").write_text(json.dumps(r, ensure_ascii=False, indent=1))
    print(texto(r))
    if a.subir:
        from src import almacen
        almacen.subir_archivo(SALIDA / "resumen.json", R2_RESUMEN)
    if a.avisar:
        from src import monitor
        monitor.enviar(texto(r))


if __name__ == "__main__":
    main()
