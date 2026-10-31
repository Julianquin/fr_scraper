"""
Monitor de salud del pipeline: un resumen diario por Telegram SOLO para el administrador
(secret TELEGRAM_ADMIN_CHAT_ID; los agentes nunca lo reciben), con avisos si algo se ve mal.

  resumen   al final del barrido diario (barrido_http.yml): por venta y arriendo, cobertura,
            errores, altas, bajas, reaparecen y cambios de precio del día; el resultado de cada
            job y el del scraper viejo (que es el que dispara el barrido).
  fallo     un aviso corto cuando falla otro workflow (p. ej. la alerta de los lunes).
  modelo    tras el entrenamiento semanal del modelo de precios (modelo.yml): el error de esta semana
            frente a la anterior (fichas en R2, modelos/historial/), con aviso si empeora.

Metrocuadrado (metrocuadrado.yml) manda su propio resumen al terminar, con
`--raices m2_venta --titulo Metrocuadrado --unificado`: el último añade el registro único por inmueble
(E8, estado_unificado/meta.json).

Si un día no llega el resumen, algo no corrió (GitHub no lanzó el scraper, se canceló…).

Uso:
    python -m src.monitor resumen [--fecha AAAA-MM-DD] [--jobs plan=success,barrer=failure…]
                                  [--scraper success|failure] [--url URL] [--local] [--no-enviar]
                                  [--raices venta,arriendo] [--titulo "Barrido diario"] [--unificado]
    python -m src.monitor fallo "La alerta semanal falló" [--url URL]
    python -m src.monitor modelo [--fecha AAAA-MM-DD] [--url URL] [--no-enviar]
"""
from __future__ import annotations
import argparse
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from src.config import BASE_DIR

COBERTURA_MIN = 0.90    # del total que declara el portal
ERRORES_MAX = 0.01      # páginas con error / páginas leídas
BAJAS_MAX = 0.05        # bajas del día / activos: más que esto es un barrido roto, no el mercado
CAMBIOS_MAX = 0.03      # cambios de precio del día / activos
MODELO_SUBE_MAX = 0.01  # el error mediano en avisos nuevos no debe subir más de 1 punto en una semana
COBERTURA_RANGO = (0.75, 0.85)   # el rango p10–p90 debe contener ~80% de los precios
RAICES = {"venta": "Venta", "arriendo": "Arriendo", "m2_venta": "Metrocuadrado venta"}
ESTABILIDAD_MIN = 0.98  # registro único: avisos que conservan su inmueble de un día a otro (29-oct: 99,6%)
SALEN_MAX = 0.05        # registro único: avisos que salen en un día / activos (un portal sin estado)


def _n(x) -> str:
    return f"{int(x):,}".replace(",", ".")


def _pct(x: float) -> str:
    return f"{x * 100:.1f}%".replace(".", ",")


def leer(raiz: str, fecha: str, local: bool) -> tuple[dict | None, dict | None]:
    """(meta.json del estado, reporte.json del barrido de `fecha`): de R2, o de data/ con `local`."""
    from src.consolidar import rutas
    from src.scraper_http import carpetas
    r = rutas(raiz)
    if local:
        meta, rep = r.meta, carpetas(raiz)[1] / fecha / "reporte.json"
    else:
        from src import almacen
        tmp = BASE_DIR / "data" / "_monitor"
        shutil.rmtree(tmp / r.r2, ignore_errors=True)       # traer() no rebaja lo de igual tamaño
        almacen.traer(f"{r.r2}/meta.json", tmp)
        almacen.traer(f"raw/{raiz}_http/fecha={fecha}/reporte.json", tmp)
        meta, rep = tmp / r.r2 / "meta.json", tmp / "raw" / f"{raiz}_http" / f"fecha={fecha}" / "reporte.json"
    cargar = lambda p: json.loads(p.read_text()) if p.exists() else None
    return cargar(meta), cargar(rep)


def revisar(raiz: str, fecha: str, meta: dict | None, rep: dict | None) -> tuple[list[str], list[str]]:
    """(líneas del resumen, avisos) de una raíz."""
    nombre = RAICES[raiz]
    if rep is None:
        return [f"*{nombre}:* no hay barrido del {fecha}"], [f"{nombre}: no hay barrido del {fecha} en R2"]
    avisos = []
    lineas = [f"*{nombre}* · {_n(rep['avisos_unicos'])} avisos ({_pct(rep['cobertura'])} del portal) · "
              f"{_n(rep['paginas'])} págs · {rep['errores_pagina']} errores"]
    if rep["cobertura"] < COBERTURA_MIN:
        avisos.append(f"{nombre}: cobertura {_pct(rep['cobertura'])} (mínimo {_pct(COBERTURA_MIN)})")
    if rep["errores_pagina"] > ERRORES_MAX * max(rep["paginas"], 1):
        avisos.append(f"{nombre}: {rep['errores_pagina']} páginas con error de {_n(rep['paginas'])}")
    if rep.get("segmentos_bajo_90pct"):
        lineas.append(f"Segmentos bajo 90%: {', '.join(rep['segmentos_bajo_90pct'])}")
    if rep.get("escalones_sin_resolver"):                # Metrocuadrado: empates de precio sin partir
        lineas.append(f"Escalones de precio sin resolver: {rep['escalones_sin_resolver']}")

    if not meta or meta.get("ultimo_barrido") != fecha:
        ultimo = meta.get("ultimo_barrido") if meta else "ninguno"
        avisos.append(f"{nombre}: el estado no tiene el barrido del {fecha} (último: {ultimo}); ¿falló consolidar?")
        return lineas, avisos
    res = meta.get("ultimo_resumen")
    if not res:                             # consolidado antes de que existiera el resumen (26-oct)
        return lineas + ["(sin resumen del estado: consolidado con una versión anterior)"], avisos
    activos = max(res["activos"], 1)
    lineas += [f"🆕 {_n(res.get('alta', 0))} altas · {_n(res.get('descubierto', 0))} descubiertos · "
               f"❌ {_n(res.get('baja', 0))} bajas · ↩️ {_n(res.get('reaparece', 0))} reaparecen"
               + (f" · {_n(res['reactivado'])} rescatados vuelven" if res.get("reactivado") else "")
               + (f" · 🔁 {_n(res['republicado'])} republicados" if res.get("republicado") else ""),
               f"💲 {_n(res.get('precio_baja', 0))} bajaron y {_n(res.get('precio_sube', 0))} subieron de precio · "
               f"{_n(res['activos'])} activos"]
    acu = meta.get("acumulado", {})
    if acu.get("baja"):
        lineas.append(f"Desde el inicio: {_n(acu['baja'])} bajas, de las que reaparecieron "
                      f"{_n(acu.get('reaparece', 0))} ({_pct(acu.get('reaparece', 0) / acu['baja'])})")
    if len(meta["barridos"]) > 1 and not res["ausencias_contadas"]:
        avisos.append(f"{nombre}: barrido incompleto ({_pct(res['cobertura_portal'])} de lo que declara el portal, "
                      f"{_pct(res['vistos'] / activos)} de los activos); hoy no se contaron ausencias"
                      if res.get("cobertura_portal") is not None else
                      f"{nombre}: el barrido trajo menos del 90% de los activos; hoy no se contaron ausencias")
    if res.get("baja", 0) > BAJAS_MAX * activos:
        avisos.append(f"{nombre}: {_n(res['baja'])} bajas en un día ({_pct(res['baja'] / activos)} de los activos)")
    if res.get("cambio_precio", 0) > CAMBIOS_MAX * activos:
        avisos.append(f"{nombre}: {_n(res['cambio_precio'])} cambios de precio en un día "
                      f"({_pct(res['cambio_precio'] / activos)} de los activos)")
    return lineas, avisos


def unificado(fecha: str, local: bool) -> tuple[list[str], list[str]]:
    """(líneas, avisos) del registro único por inmueble (src/unificar.py)."""
    from src.unificar import ESTADO, R2
    if local:
        ruta = ESTADO / "meta.json"
    else:
        from src import almacen
        tmp = BASE_DIR / "data" / "_monitor"
        shutil.rmtree(tmp / R2, ignore_errors=True)
        almacen.traer(f"{R2}/meta.json", tmp)
        ruta = tmp / R2 / "meta.json"
    meta = json.loads(ruta.read_text()) if ruta.exists() else {}
    if meta.get("ultimo_dia") != fecha:
        return ([f"*Registro único:* sin actualizar (último: {meta.get('ultimo_dia', 'ninguno')})"],
                [f"Registro único: no se actualizó el {fecha}"])
    r, avisos = meta["ultimo_resumen"], []
    lineas = [f"*Registro único* · {_n(r['inmuebles_activos'])} inmuebles de {_n(r['avisos_activos'])} avisos · "
              f"{_n(r['en_los_dos_portales'])} en los dos portales",
              f"🆕 {_n(r['inmuebles_nuevos'])} nuevos · {_n(r['inmuebles_cerrados'])} cerrados · "
              f"{_n(r['fusionados'])} fusionados · {_n(r['partidos'])} partidos"
              + (f" · {_pct(r['estabilidad'])} conservan su inmueble" if r["estabilidad"] is not None else "")]
    if r["estabilidad"] is not None and r["estabilidad"] < ESTABILIDAD_MIN:
        avisos.append(f"Registro único: solo el {_pct(r['estabilidad'])} de los avisos conservó su inmueble")
    if r["avisos_salen"] > SALEN_MAX * max(r["avisos_activos"], 1):
        avisos.append(f"Registro único: salieron {_n(r['avisos_salen'])} avisos en un día")
    return lineas, avisos


def resumen(fecha: str, jobs: dict[str, str], scraper: str | None, url: str | None, local: bool,
            raices: tuple[str, ...] = ("venta", "arriendo"), titulo: str = "Barrido diario",
            con_unificado: bool = False) -> str:
    bloques, avisos = [], []
    fallidos = [f"{j} ({r})" for j, r in jobs.items() if r != "success"]
    if fallidos:
        avisos.append(f"Jobs con problemas: {', '.join(fallidos)}")
    if scraper and scraper != "success":
        lunes = datetime.now(ZoneInfo("America/Bogota")).weekday() == 0
        avisos.append(f"El scraper viejo terminó en '{scraper}'"
                      + (": hoy es lunes y la alerta NO salió" if lunes else ""))
    for raiz in raices:
        lineas, av = revisar(raiz, fecha, *leer(raiz, fecha, local))
        bloques.append("\n".join(lineas))
        avisos += av
    if con_unificado:
        lineas, av = unificado(fecha, local)
        bloques.append("\n".join(lineas))
        avisos += av
    cabeza = (f"*⚠️ {titulo} · {fecha} · {len(avisos)} aviso{'s' if len(avisos) > 1 else ''}*\n"
              + "\n".join(f"• {a}" for a in avisos)) if avisos else f"*✅ {titulo} · {fecha}*"
    return "\n\n".join([cabeza, *bloques] + ([f"🔗 {url}"] if url else []))


def modelo(fecha: str, url: str | None) -> str:
    """Resumen del entrenamiento semanal: la ficha de `fecha` frente a la de la semana anterior."""
    from src import almacen
    tmp = BASE_DIR / "data" / "_monitor"
    shutil.rmtree(tmp / "modelos", ignore_errors=True)
    almacen.traer("modelos/historial/", tmp)
    fichas = {p.parent.name.split("=")[1]: json.loads(p.read_text())
              for p in (tmp / "modelos" / "historial").glob("fecha=*/*.json")}
    enlace = [f"🔗 {url}"] if url else []
    if fecha not in fichas:
        return "\n\n".join([f"*⚠️ Modelo de precios · {fecha}*\n• No hay ficha de hoy en R2: ¿falló el entrenamiento?"] + enlace)
    f = fichas[fecha]
    previas = sorted(k for k in fichas if k < fecha)
    antes = fichas[previas[-1]] if previas else None
    vs = lambda clave: f" (antes {_pct(antes[clave]['MdAPE'])})" if antes else ""
    pi, nt = f["por_inmueble"], f["temporal"]
    lineas = [f"{_n(f['avisos'])} avisos · publicados hasta {f['fecha_datos']}",
              f"Por inmueble: error mediano {_pct(pi['MdAPE'])}{vs('por_inmueble')} · {_pct(pi['±10%'])} a ±10% · "
              f"el rango p10–p90 contiene el {_pct(pi['cobertura_p10_p90'])}",
              f"Avisos nuevos (desde {nt['publicados_desde']}): error mediano {_pct(nt['MdAPE'])}{vs('temporal')} · "
              f"el rango contiene el {_pct(nt['cobertura_p10_p90'])}",
              " · ".join(f"{t} {_pct(m['MdAPE'])}" for t, m in f["por_tipo"].items())]
    avisos = []
    if antes and nt["MdAPE"] > antes["temporal"]["MdAPE"] + MODELO_SUBE_MAX:
        avisos.append(f"El error en avisos nuevos subió de {_pct(antes['temporal']['MdAPE'])} a {_pct(nt['MdAPE'])}")
    for nombre, m in (("por inmueble", pi), ("en avisos nuevos", nt)):
        if not COBERTURA_RANGO[0] <= m["cobertura_p10_p90"] <= COBERTURA_RANGO[1]:
            avisos.append(f"El rango p10–p90 {nombre} contiene el {_pct(m['cobertura_p10_p90'])} (debería ~80%)")
    if antes and f["avisos"] < 0.8 * antes["avisos"]:
        avisos.append(f"Entrenó con {_n(f['avisos'])} avisos, frente a {_n(antes['avisos'])} la semana anterior")
    titulo = (f"*⚠️ Modelo de precios · {fecha} · {len(avisos)} aviso{'s' if len(avisos) > 1 else ''}*\n"
              + "\n".join(f"• {a}" for a in avisos)) if avisos else f"*🧠 Modelo de precios · {fecha}*"
    return "\n\n".join([titulo, "\n".join(lineas)] + enlace)


def enviar(texto: str) -> None:
    """Al chat del administrador. Sin credenciales, el mensaje queda solo en el log."""
    from src.notificar import telegram
    token, chat = os.environ.get("TELEGRAM_BOT_TOKEN"), os.environ.get("TELEGRAM_ADMIN_CHAT_ID")
    if not (token and chat):
        print("⚠️  Falta TELEGRAM_BOT_TOKEN o TELEGRAM_ADMIN_CHAT_ID: no se envía")
        return
    telegram(texto, token, chat)
    print("✅ Enviado al administrador")


def main() -> None:
    p = argparse.ArgumentParser(description="Monitor de salud del pipeline (Telegram al administrador)")
    sub = p.add_subparsers(dest="accion", required=True)
    r = sub.add_parser("resumen", help="resumen diario del barrido")
    r.add_argument("--fecha", default=datetime.now(timezone.utc).strftime("%Y-%m-%d"))
    r.add_argument("--jobs", default="", help="job=resultado separados por coma (needs.*.result)")
    r.add_argument("--scraper", help="conclusión del scraper viejo (workflow_run)")
    r.add_argument("--url", help="enlace a la corrida")
    r.add_argument("--local", action="store_true", help="lee data/ en vez de R2")
    r.add_argument("--no-enviar", action="store_true", help="solo imprime")
    r.add_argument("--raices", default="venta,arriendo", help=f"separadas por coma: {', '.join(RAICES)}")
    r.add_argument("--titulo", default="Barrido diario")
    r.add_argument("--unificado", action="store_true", help="añade el registro único por inmueble (E8)")
    f = sub.add_parser("fallo", help="aviso corto de que algo falló")
    f.add_argument("texto")
    f.add_argument("--url")
    m = sub.add_parser("modelo", help="resumen del entrenamiento semanal del modelo de precios")
    m.add_argument("--fecha", default=datetime.now(timezone.utc).strftime("%Y-%m-%d"))
    m.add_argument("--url", help="enlace a la corrida")
    m.add_argument("--no-enviar", action="store_true", help="solo imprime")
    a = p.parse_args()

    try:                                    # en local, las claves vienen de .env
        from src.almacen import _cargar_env
        _cargar_env()
    except ImportError:
        pass
    if a.accion == "fallo":
        texto = f"*⚠️ {a.texto}*" + (f"\n\n🔗 {a.url}" if a.url else "")
    elif a.accion == "modelo":
        texto = modelo(a.fecha, a.url)
    else:
        jobs = dict(j.split("=", 1) for j in a.jobs.split(",") if "=" in j)
        texto = resumen(a.fecha, jobs, a.scraper, a.url, a.local, tuple(a.raices.split(",")), a.titulo,
                        a.unificado)
    print(texto)
    if a.accion == "fallo" or not a.no_enviar:
        try:
            enviar(texto)
        except RuntimeError as e:
            sys.exit(f"❌ No se pudo enviar: {e}")


if __name__ == "__main__":
    main()
