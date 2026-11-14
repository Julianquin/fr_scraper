"""
Canal público de Telegram: publicaciones diarias que el administrador aprueba antes de que salgan.

Cada día, tras el barrido de Metrocuadrado (canal.yml), `proponer` manda al chat del administrador las
publicaciones del día, cada una EXACTAMENTE como saldrá en el canal y con dos botones, «Publicar» y «Descartar»:
  - todos los días, la oportunidad del día (1–2, con criterios más estrictos que la alerta: CRITERIOS);
  - los lunes, el pulso del mercado de la semana; los jueves, las mayores bajadas de precio.
Al pulsar un botón, Telegram avisa al Worker de Cloudflare (cloudflare/bot/), que contesta y lanza `aprobar`: la
publicación queda en cola, o sale ya si son más de las 07:00 de Colombia. A las 07:00 el Worker lanza `publicar`, que
copia al canal lo aprobado (copyMessage: mismo texto y formato, sin «reenviado de»).

Nada de la marca vive en el código ni en los registros (el repositorio es público): el canal sale del secret
TELEGRAM_CANAL_ID y la línea de contacto de marca/config.json en R2 ({"contacto": "…"}). Estado en R2:
canal/estado.json (qué se propuso, qué está en cola, qué se publicó).

Uso:
    python -m src.canal proponer [--ciudad Cali] [--no-enviar]   # --no-enviar: solo imprime las propuestas
    python -m src.canal aprobar --mensaje 1234                    # lo lanza el Worker al pulsar «Publicar»
    python -m src.canal publicar                                  # lo lanza el Worker a las 07:00 de Colombia
"""
from __future__ import annotations

import argparse
import json
import os
import urllib.parse
from pathlib import Path

import pandas as pd

from src import oportunidades as op
from src.config import BASE_DIR

# Criterios de la oportunidad del día (más estrictos que la alerta: es pública)
CRITERIOS = {
    "dias": 2,                  # publicada en los últimos 2 días
    "bajo_similares": 0.20,     # ≥ 20% bajo sus similares…
    "bajo_modelo": 0.10,        # …y ≥ 10% bajo el modelo (estimado sin ver): las dos medidas coinciden
    "confianza": "Alta",        # similares en el barrio, 15 o más
    "maximo": 2,                # oportunidades por día
}
TOP_BAJADAS = 3
HORA_PUBLICAR = 7               # hora de Colombia
COLOMBIA = "America/Bogota"
LOCAL = BASE_DIR / "data" / "canal"       # sin claves de R2 (pruebas en local), el estado vive aquí


# ── Estado y marca en R2 ──────────────────────────────────────────────────────────────────────────────────
def _hay_r2() -> bool:
    return all(os.environ.get(v) for v in ("R2_ACCOUNT_ID", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "R2_BUCKET"))


def _leer(clave: str, defecto: dict) -> dict:
    if not _hay_r2():
        p = LOCAL / clave.replace("/", "_")
        return json.loads(p.read_text()) if p.exists() else defecto
    from src import almacen
    s3, bucket = almacen._cliente()
    try:
        return json.loads(s3.get_object(Bucket=bucket, Key=clave)["Body"].read())
    except s3.exceptions.NoSuchKey:
        return defecto


def _escribir(clave: str, valor: dict) -> None:
    texto = json.dumps(valor, ensure_ascii=False, indent=1)
    if not _hay_r2():
        LOCAL.mkdir(parents=True, exist_ok=True)
        (LOCAL / clave.replace("/", "_")).write_text(texto)
        return
    from src import almacen
    s3, bucket = almacen._cliente()
    s3.put_object(Bucket=bucket, Key=clave, Body=texto.encode(), ContentType="application/json")


def estado() -> dict:
    return _leer("canal/estado.json", {"propuestas": {}, "propuestos": {}, "cola": [], "publicados": []})


def guardar_estado(e: dict) -> None:
    _escribir("canal/estado.json", e)


def contacto() -> str:
    """La línea de contacto de la marca (marca/config.json en R2); vacía si no está. Nunca se imprime."""
    return _leer("marca/config.json", {}).get("contacto", "")


def ahora_colombia() -> pd.Timestamp:
    return pd.Timestamp.now(tz=COLOMBIA)


# ── Lo que se propone ─────────────────────────────────────────────────────────────────────────────────────
def _preparar(ciudad: str):
    op.FUENTE = "http"
    op.INICIO_DATOS = pd.Timestamp("1900-01-01")
    import src.viz_map as viz_map
    viz_map.FUENTE = "http"
    df = op.cargar(ciudad)
    hasta = df["first_seen"].max().normalize()
    return df, hasta, op.historial(hasta)


def oportunidades_del_dia(ciudad: str, ya: set[str]) -> pd.DataFrame:
    """Las oportunidades publicadas en los últimos CRITERIOS['dias'] que cumplen todos los criterios y no se
    propusieron antes, las de mayor descuento primero (como mucho CRITERIOS['maximo'])."""
    df, hasta, hist = _preparar(ciudad)
    nuevos = df[df["first_seen"] > hasta - pd.Timedelta(days=CRITERIOS["dias"])]
    nuevos = nuevos[~nuevos["id_inmueble"].isin(ya)]
    ev = op.calificar(df, nuevos, hist) if len(nuevos) else nuevos
    if ev.empty:
        return ev
    ev = ev[(ev["descuento"] >= CRITERIOS["bajo_similares"]) & ev["motivo_descarte"].isna() & ~ev["verificar"]
            & (ev["confianza"] == CRITERIOS["confianza"])]
    if ev.empty:
        return ev
    ev = op.enriquecer(ev.sort_values("descuento", ascending=False).head(10), df, hasta)
    ev = ev[(ev["bajo_modelo"] >= CRITERIOS["bajo_modelo"]) & ev["rep_precio_antes"].isna()]   # sin republicados
    return ev.head(CRITERIOS["maximo"])


def texto_oportunidad(r: pd.Series, pie: str) -> str:
    comps = "\n".join(f"   ↳ {op._comp_txt(c)} · [ver]({c['URL detalle']})" for _, c in r["comps"].iterrows())
    return "\n".join(x for x in [
        f"*🔎 Oportunidad del día · {r.get('Ciudad') or 'Cali'}*",
        "",
        f"*{op._ficha(r)}*",
        f"📍 {op._zona(r)}",
        f"💰 {op._millones(r['Precio'])} ({op._m2(r['precio_m2'])} M/m²)",
        "",
        f"📊 *{r['descuento']:.0%} bajo sus {r['n_comps']} similares* {'del barrio' if r['nivel_comps'] == 'barrio' else 'cercanos'} "
        f"(mediana {op._m2(r['med_comps'])} M/m²)",
        f"🤖 *{r['bajo_modelo']:.0%} bajo nuestro modelo de precios* (estima {op._millones(r['modelo_precio'])})",
        "",
        "Tres de sus similares:",
        comps,
        "",
        f"🔗 [Ver el aviso]({r['URL detalle']})",
        "",
        "_Herramienta para priorizar, no un avalúo: verifica siempre el inmueble y su documentación._",
        pie,
    ] if x is not None).strip()


def texto_pulso(ciudad: str, pie: str) -> str:
    """El pulso del mercado de la semana (los lunes), del informe semanal."""
    from src import informe_semanal as s
    d = s.datos(ciudad)
    meses = ["ene", "feb", "mar", "abr", "may", "jun", "jul", "ago", "sep", "oct", "nov", "dic"]
    periodo = f"{d['desde'].day}–{d['hasta'].day} {meses[d['hasta'].month - 1]}"
    n = lambda x: f"{x:,.0f}".replace(",", ".")
    t = d["todas"]
    t = t[[c <= "22" or c == "81" for c in t.index]].dropna(subset=["precio_m2"]) if len(t) else t
    nombre = lambda c: d["nombres"].get(c, c).replace("Comuna 0", "Comuna ").replace(
        "Zona de expansión (corredor Cali–Jamundí)", "Zona de expansión")
    lineas = [f"*📈 El mercado de la semana · {ciudad}* ({periodo})", "",
              f"🏠 {n(d['viviendas'])} viviendas en venta",
              f"🆕 {n(d['publicadas'])} publicadas · ❌ {n(d['retiradas'])} retiradas",
              f"📉 {n(d['bajadas'])} bajaron de precio" + (f" (mediana −{d['bajada_mediana']:.0%})" if d.get("bajada_mediana") else ""),
              f"💰 Precio pedido mediano: {op._m2(d['precio_m2'])} M/m²",
              f"💵 Rentabilidad bruta típica: {d['rentab']:.1%}".replace(".", ",")]
    if len(t) >= 2:
        caro, barato = t["precio_m2"].idxmax(), t["precio_m2"].idxmin()
        lineas += ["", f"La más cara: {nombre(caro)} ({op._m2(t.loc[caro, 'precio_m2'])} M/m²) · "
                       f"la más barata: {nombre(barato)} ({op._m2(t.loc[barato, 'precio_m2'])} M/m²)"]
    lineas += ["", "_Precios pedidos en avisos públicos de vivienda en venta._", pie]
    return "\n".join(lineas).strip()


def texto_bajadas(ciudad: str, pie: str) -> str | None:
    """Las mayores bajadas de precio de la semana (los jueves), con dónde quedan frente a sus similares."""
    df, hasta, hist = _preparar(ciudad)
    bj = op.bajadas(df, hist, hasta - pd.Timedelta(days=6), hasta)
    bj = op.seleccionar_bajadas(bj, pd.Index([]), TOP_BAJADAS) if len(bj) else bj
    if bj.empty:
        return None
    bj = op.enriquecer(bj, df, hasta)
    bloques = [f"*📉 Bajaron de precio esta semana · {ciudad}*"]
    for _, r in bj.iterrows():
        bloques.append("\n".join(x for x in [
            f"*{op._bajo_txt(r)}* · {op._ficha(r)}",
            f"📍 {op._zona(r)}",
            f"💰 {op._millones(r['precio_antes'])} → {op._millones(r['Precio'])}",
            f"📊 {op._frente_comps(r).replace('comparables', 'similares')}" if op._frente_comps(r) else None,
            f"🔗 [Ver el aviso]({r['URL detalle']})",
        ] if x))
    bloques += ["_Herramienta para priorizar, no un avalúo._\n" + pie]
    return "\n\n".join(bloques).strip()


# ── Telegram ──────────────────────────────────────────────────────────────────────────────────────────────
def _botones() -> str:
    """El Worker (cloudflare/bot/) lee de cada clic la acción (pub | des) y el id del mensaje pulsado."""
    return json.dumps({"inline_keyboard": [[{"text": "✅ Publicar", "callback_data": "pub"},
                                            {"text": "✖️ Descartar", "callback_data": "des"}]]})


def enviar_propuesta(texto: str, token: str, chat: str) -> int:
    """La propuesta al administrador, tal como saldrá en el canal, con los botones. Devuelve el id del mensaje."""
    from src.notificar import _a_html, _api
    r = _api("sendMessage", token, urllib.parse.urlencode({
        "chat_id": chat, "text": _a_html(texto), "parse_mode": "HTML", "disable_web_page_preview": "true",
        "reply_markup": _botones()}).encode())
    return int(r["result"]["message_id"])


def copiar_al_canal(mensaje_id: int, token: str, admin: str, canal: str) -> None:
    from src.notificar import _api
    _api("copyMessage", token, urllib.parse.urlencode({"chat_id": canal, "from_chat_id": admin,
                                                       "message_id": mensaje_id}).encode())


def _credenciales(canal: bool = True) -> tuple[str, str, str | None]:
    """Token del bot, chat del administrador y (para aprobar y publicar) el canal."""
    env = os.environ.get
    necesarias = ("TELEGRAM_BOT_TOKEN", "TELEGRAM_ADMIN_CHAT_ID") + (("TELEGRAM_CANAL_ID",) if canal else ())
    faltan = [v for v in necesarias if not env(v)]
    if faltan:
        raise SystemExit(f"❌ Faltan {', '.join(faltan)}")
    return env("TELEGRAM_BOT_TOKEN"), env("TELEGRAM_ADMIN_CHAT_ID"), env("TELEGRAM_CANAL_ID")


# ── Comandos ──────────────────────────────────────────────────────────────────────────────────────────────
def proponer(ciudad: str = "Cali", enviar: bool = True) -> list[tuple[str, str]]:
    """Las publicaciones del día al administrador, con botones. Cada oportunidad se propone una sola vez."""
    e = estado()
    pie = contacto()
    hoy = ahora_colombia()
    propuestas: list[tuple[str, str, str | None]] = []          # (tipo, texto, id del aviso)
    for _, r in oportunidades_del_dia(ciudad, set(e["propuestos"])).iterrows():
        propuestas.append(("oportunidad", texto_oportunidad(r, pie), str(r["id_inmueble"])))
    if hoy.day_of_week == 0:
        propuestas.append(("pulso", texto_pulso(ciudad, pie), None))
    if hoy.day_of_week == 3 and (t := texto_bajadas(ciudad, pie)):
        propuestas.append(("bajadas", t, None))
    n_op = sum(t == "oportunidad" for t, _, _ in propuestas)
    print(f"{hoy:%Y-%m-%d}: {n_op} oportunidades · {len(propuestas) - n_op} más ({[t for t, _, _ in propuestas]})")
    if not enviar:
        for _, texto, _ in propuestas:
            print("\n" + "─" * 60 + "\n" + texto)
        return [(t, x) for t, x, _ in propuestas]
    token, admin, _ = _credenciales(canal=False)
    from src.notificar import telegram
    cabecera = (f"*📝 Propuestas para el canal · {hoy.day}-{hoy.month:02d}*\n"
                + (f"{n_op} oportunidad{'es' if n_op != 1 else ''} del día" if n_op else
                   "Hoy ninguna oportunidad cumple los criterios del canal")
                + (" · y " + " y ".join(t for t, _, _ in propuestas if t != "oportunidad") if len(propuestas) > n_op else "")
                + (f".\nLas que apruebes salen a las 0{HORA_PUBLICAR}:00 de Colombia (si apruebas más tarde, al momento)."
                   if propuestas else "."))
    telegram(cabecera, token, admin)
    for tipo, texto, id_aviso in propuestas:
        mensaje = enviar_propuesta(texto, token, admin)
        e["propuestas"][str(mensaje)] = {"tipo": tipo, "id": id_aviso, "fecha": f"{hoy:%Y-%m-%d}"}
        if id_aviso:
            e["propuestos"][id_aviso] = f"{hoy:%Y-%m-%d}"
    guardar_estado(e)
    return [(t, x) for t, x, _ in propuestas]


def aprobar(mensaje: int) -> None:
    """«Publicar» pulsado: a la cola, o al canal ya si son más de las HORA_PUBLICAR de Colombia."""
    e = estado()
    if any(p["mensaje"] == mensaje for p in e["publicados"]) or mensaje in e["cola"]:
        print("Ya estaba aprobada")
        return
    if ahora_colombia().hour >= HORA_PUBLICAR:
        token, admin, canal = _credenciales()
        copiar_al_canal(mensaje, token, admin, canal)
        e["publicados"].append({"mensaje": mensaje, "fecha": f"{ahora_colombia():%Y-%m-%d}",
                                **e["propuestas"].get(str(mensaje), {})})
        print("✅ Publicada ya (pasadas las 07:00 de Colombia)")
    else:
        e["cola"].append(mensaje)
        print(f"✅ En cola para las 0{HORA_PUBLICAR}:00 de Colombia")
    guardar_estado(e)


def publicar() -> None:
    """Las 07:00 de Colombia: copia al canal lo aprobado, en el orden en que se propuso."""
    e = estado()
    if not e["cola"]:
        print("Nada en cola")
        return
    token, admin, canal = _credenciales()
    for mensaje in sorted(e["cola"]):
        copiar_al_canal(mensaje, token, admin, canal)
        e["publicados"].append({"mensaje": mensaje, "fecha": f"{ahora_colombia():%Y-%m-%d}",
                                **e["propuestas"].get(str(mensaje), {})})
    print(f"✅ {len(e['cola'])} publicadas")
    e["cola"] = []
    guardar_estado(e)


def main() -> None:
    p = argparse.ArgumentParser(description="Canal público de Telegram con aprobación del administrador")
    sub = p.add_subparsers(dest="accion", required=True)
    pr = sub.add_parser("proponer")
    pr.add_argument("--ciudad", default="Cali")
    pr.add_argument("--no-enviar", action="store_true", help="solo imprime las propuestas")
    pr.add_argument("--dias", type=int, help=f"ventana de publicación (por defecto {CRITERIOS['dias']}; para pruebas)")
    pr.add_argument("--maximo", type=int, help=f"oportunidades como mucho (por defecto {CRITERIOS['maximo']})")
    ap = sub.add_parser("aprobar")
    ap.add_argument("--mensaje", type=int, required=True)
    sub.add_parser("publicar")
    a = p.parse_args()
    if a.accion == "proponer":
        CRITERIOS.update({k: v for k, v in (("dias", a.dias), ("maximo", a.maximo)) if v})
        proponer(a.ciudad, not a.no_enviar)
    elif a.accion == "aprobar":
        aprobar(a.mensaje)
    else:
        publicar()


if __name__ == "__main__":
    main()
