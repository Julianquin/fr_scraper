"""
El bot de consulta por Telegram: un agente escribe al bot y recibe el informe de precio para captar
(src/informe_captacion.py) en su chat.

Qué entiende (un mensaje, en texto libre):
  - una dirección de Cali, con la torre y el apartamento si es un edificio: «KR 98 B # 48-127 TO 3 AP 436»
    (y, si quiere, más datos separados por comas: «…, 3 hab, 2 baños»);
  - el enlace de un aviso publicado (FincaRaíz o Metrocuadrado);
  - una descripción: «apto 80 m², 3 hab, 2 baños, estrato 4, Ciudad Jardín» (tipo, área y barrio hacen falta; lo
    demás, si está). Otra ciudad, nombrándola: «casa 120 m2 Laureles Medellín».
Si falta algo o la dirección no es una sola vivienda, le responde qué necesita (las mismas preguntas del informe).

Cómo llega (el repositorio es público y los registros de GitHub Actions también: la dirección del cliente de un
agente no puede quedar en ellos):
  1. El Worker del bot (cloudflare/bot/worker.js) recibe el mensaje, comprueba que el chat es de un agente
     autorizado, guarda la consulta en R2 (consultas/pendientes/<id>.json: chat y texto) y lanza consulta.yml con
     el id, nada más.
  2. consulta.yml corre `python -m src.consulta --id <id>`: lee la consulta, la borra de R2, prepara el informe y
     lo manda a ese chat. En el registro solo quedan el id y el resultado; en R2, consultas/registro/<fecha>/<id>.json
     con el chat, el tipo de consulta y el resultado (sin la dirección ni el texto), para saber cuánto se usa.

    python -m src.consulta --id <id>                      # lo que corre el flujo
    python -m src.consulta --texto "…" --chat <chat_id>    # a mano (pruebas)
    python -m src.consulta --entender "…"                 # solo cómo lo entiende
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import re
import time
import traceback
from dataclasses import dataclass, field

from src import zonas

PENDIENTES, REGISTRO = "consultas/pendientes/", "consultas/registro/"
ENLACE = re.compile(r"https?://(?:www\.)?(?:fincaraiz\.com\.co|metrocuadrado\.com)/\S+", re.I)
TIPOS = [(r"apa?rta[- ]?estudios?|aptoestudio", "Apartaestudio"), (r"apartamentos?|aptos?\.?|apto|apt\.?", "Apartamento"),
         (r"casas?", "Casa")]
NUMEROS = {
    "m2": r"(\d+(?:[.,]\d+)?)\s*(?:m2|m²|mts2?|mt2|metros(?:\s+cuadrados)?|m\b)",
    "habitaciones": r"(\d+)\s*(?:habitaci[oó]n(?:es)?|habs?\.?|alcobas?|cuartos?)",
    "banos": r"(\d+)\s*(?:baños?|banos?)",
    "parqueaderos": r"(\d+)\s*(?:parqueaderos?|parq\.?|garajes?)",
    "estrato": r"(?:estrato|est\.?)\s*(\d)",
}
# Palabras que sobran alrededor del barrio («casa de 120 m² EN El Ingenio CON…»): solo se quitan de los extremos, y
# nunca los artículos, que son parte de muchos nombres (El Ingenio, La Flora, Los Cristales, Valle del Lili)
RELLENO = {"en", "de", "del", "barrio", "sector", "urbanizacion", "con", "y", "un", "una", "vendo", "venta", "precio",
           "cuanto", "vale", "informe", "para", "mi", "hola", "por", "favor", "un", "a"}
# Otras ciudades del informe de captación (nombrándolas en el texto); Jamundí y Yumbo van con Cali (E10)
CIUDADES = {"bogota": "Bogotá", "medellin": "Medellín", "barranquilla": "Barranquilla", "bucaramanga": "Bucaramanga",
            "pereira": "Pereira", "cartagena": "Cartagena", "manizales": "Manizales", "armenia": "Armenia",
            "jamundi": "Cali y alrededores", "yumbo": "Cali y alrededores", "cali": "Cali"}
AYUDA = ("Escríbeme una de estas tres cosas y te mando el informe de precio para captar (en unos minutos):\n\n"
         "• La dirección (en Cali), con la torre y el apartamento si es un edificio:\n"
         "  KR 98 B # 48-127 TO 3 AP 436\n"
         "• El enlace de un aviso publicado.\n"
         "• Una descripción con el tipo, el área y el barrio (y si quieres, habitaciones, baños y estrato):\n"
         "  apto 80 m², 3 hab, 2 baños, estrato 4, Ciudad Jardín\n\n"
         "Es una estimación del precio de mercado pedido, no un avalúo.")


@dataclass
class Consulta:
    modo: str                                  # direccion | aviso | descripcion | ayuda | incompleta
    campos: dict = field(default_factory=dict)
    falta: list = field(default_factory=list)


def _num(x: str) -> float:
    return float(x.replace(",", "."))


def entender(texto: str) -> Consulta:
    """De un mensaje libre a lo que necesita el informe de captación."""
    from src import catastro_cali
    t = " ".join((texto or "").split())
    if not t or t.lower().lstrip("/").split()[0] in ("start", "ayuda", "help", "hola"):
        return Consulta("ayuda")
    if m := ENLACE.search(t):
        return Consulta("aviso", {"aviso": m.group(0)})

    campos: dict = {}
    partes = [p.strip() for p in re.split(r"[,;\n]", texto) if p.strip()]
    direccion = next((p for p in partes if catastro_cali.es_direccion(p)), None)
    resto = " ".join(p for p in partes if p != direccion)
    for clave, patron in NUMEROS.items():
        if m := re.search(patron, resto, re.I):
            campos[clave] = _num(m.group(1))
            resto = resto[:m.start()] + " " + resto[m.end():]
    for patron, tipo in TIPOS:
        if m := re.search(rf"\b(?:{patron})\b", resto, re.I):
            campos["tipo"] = tipo
            resto = resto[:m.start()] + " " + resto[m.end():]
            break
    if direccion:
        return Consulta("direccion", {**campos, "direccion": direccion})

    ciudad = "Cali"
    for palabra in re.findall(r"[\wáéíóúñ]+", resto.lower()):
        if (c := CIUDADES.get(zonas.normalizar(palabra))) and c != "Cali":
            ciudad = c
            if c != "Cali y alrededores":       # Jamundí y Yumbo se quedan: sirven para encontrar el barrio
                resto = re.sub(rf"\b{re.escape(palabra)}\b", " ", resto, flags=re.I)
    resto = re.sub(r"\bcali\b", " ", resto, flags=re.I)
    resto = re.sub(r"\b(?:el|la)\s+(?:barrio|sector|urbanizaci[oó]n|conjunto)\b", " ", resto, flags=re.I)
    palabras = resto.replace("·", " ").replace(".", " ").split()
    while palabras and zonas.normalizar(palabras[0]) in RELLENO:
        palabras.pop(0)
    while palabras and zonas.normalizar(palabras[-1]) in RELLENO:
        palabras.pop()
    barrio = " ".join(palabras).strip(" -")
    campos.update(ciudad=ciudad, barrio=barrio or None)
    falta = [n for n, k in (("el tipo (apto, casa o apartaestudio)", "tipo"), ("el área en m²", "m2"),
                            ("el barrio", "barrio")) if not campos.get(k)]
    return Consulta("incompleta" if falta else "descripcion", campos, falta)


def _respuesta_de_error(texto: str) -> str:
    """Los mensajes del informe dicen «vuelve a lanzarlo» (el formulario de GitHub): en el bot, «escríbelo»."""
    return (texto.replace("Vuelve a lanzarlo", "Escríbemelo de nuevo").replace("vuelve a lanzarlo", "escríbemelo de nuevo")
            .replace("❓ Informe de precio: ", "❓ "))


def atender(texto: str, chat: str) -> str:
    """Responde a una consulta en `chat`: el PDF, una pregunta o la ayuda. Devuelve el resultado (para el registro):
    informe | pregunta | ayuda | error."""
    from src import informe_captacion as ic
    from src.notificar import telegram
    token = os.environ["TELEGRAM_BOT_TOKEN"]
    c = entender(texto)
    if c.modo == "ayuda":
        telegram(AYUDA, token, chat)
        return "ayuda"
    if c.modo == "incompleta":
        telegram(f"❓ Me falta {' y '.join(c.falta)}. Por ejemplo:\n  apto 80 m², 3 hab, 2 baños, estrato 4, Ciudad "
                 f"Jardín\n… o la dirección en Cali: KR 98 B # 48-127 TO 3 AP 436", token, chat)
        return "pregunta"
    k = c.campos
    try:
        if c.modo == "aviso":
            ficha = ic.ficha_de_aviso(k["aviso"])
        elif c.modo == "direccion":
            ficha = ic.ficha_por_direccion(k["direccion"], k.get("tipo"), k.get("m2"), k.get("habitaciones"),
                                           k.get("banos"), k.get("estrato"), k.get("parqueaderos"))
        else:
            ficha = ic.ficha_manual(zonas.canonica(k["ciudad"]), k["tipo"], k["m2"], k.get("habitaciones"),
                                    k.get("banos"), k.get("estrato"), k["barrio"], None, None, k.get("parqueaderos"))
    except ic.BarrioNoEncontrado as e:
        telegram(_respuesta_de_error(e.mensaje()), token, chat)
        return "pregunta"
    pdf, d = ic.generar(ficha)
    ic.enviar_a(pdf, d, chat)
    return "informe"


# ── La consulta en R2 (la deja el Worker del bot) ────────────────────────────────────────────────────────────
def _r2():
    from src import almacen
    return almacen._cliente()


def leer_pendiente(id_: str) -> dict:
    s3, bucket = _r2()
    clave = f"{PENDIENTES}{id_}.json"
    c = json.loads(s3.get_object(Bucket=bucket, Key=clave)["Body"].read())
    s3.delete_object(Bucket=bucket, Key=clave)        # la dirección del cliente no se queda guardada
    return c


def anotar(id_: str, chat: str, modo: str, resultado: str, segundos: float) -> None:
    """Para saber cuánto se usa y quién: sin la dirección ni el texto."""
    s3, bucket = _r2()
    fecha = time.strftime("%Y-%m-%d", time.gmtime())
    s3.put_object(Bucket=bucket, Key=f"{REGISTRO}{fecha}/{id_}.json", ContentType="application/json",
                  Body=json.dumps({"chat": chat, "modo": modo, "resultado": resultado, "segundos": round(segundos),
                                   "fecha": fecha}).encode())


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--id", help="la consulta que dejó el Worker en R2 (consultas/pendientes/<id>.json)")
    g.add_argument("--texto", help="(pruebas) el mensaje, con --chat")
    g.add_argument("--entender", help="solo muestra cómo entiende el mensaje")
    p.add_argument("--chat", help="(con --texto) el chat al que responder")
    a = p.parse_args()
    if a.entender:
        c = entender(a.entender)
        print(json.dumps({"modo": c.modo, "campos": c.campos, "falta": c.falta}, ensure_ascii=False))
        return
    if a.id:
        pendiente = leer_pendiente(a.id)
        texto, chat = pendiente["texto"], str(pendiente["chat"])
    else:
        texto, chat = a.texto, a.chat
    t0, modo, resultado = time.time(), entender(texto).modo, "error"
    # En el flujo, nada de lo que imprime el informe (la ruta del PDF lleva el barrio y el área; algún mensaje, la
    # dirección) va al registro, que es público: se descarta. A mano (pruebas), se ve todo.
    silencio = contextlib.redirect_stdout(io.StringIO()) if a.id else contextlib.nullcontext()
    try:
        with silencio:
            resultado = atender(texto, chat)
    except Exception as e:  # noqa: BLE001 — al agente, un aviso; al administrador, el fallo (consulta.yml)
        from src.notificar import telegram
        telegram("⚠️ No pude preparar el informe esta vez. Ya avisé para revisarlo; prueba de nuevo en un rato.",
                 os.environ["TELEGRAM_BOT_TOKEN"], chat)
        if a.id:                       # solo el tipo de error y dónde: el mensaje puede llevar la dirección
            print(f"❌ {type(e).__name__} en " + " ← ".join(f"{f.name}:{f.lineno}"
                                                          for f in traceback.extract_tb(e.__traceback__)[-4:]))
            raise SystemExit(1) from None
        raise
    finally:
        print(f"Consulta {a.id or '(prueba)'}: {modo} → {resultado} ({time.time() - t0:.0f} s)")
        if a.id:
            anotar(a.id, chat, modo, resultado, time.time() - t0)


if __name__ == "__main__":
    main()
