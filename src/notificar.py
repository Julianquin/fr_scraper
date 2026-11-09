"""
Envía un mensaje de texto (p. ej. la alerta de src/oportunidades.py) por Telegram y/o WhatsApp.

Usa los canales que tengan credenciales en variables de entorno (secrets en GitHub Actions):
  - Telegram:  TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID (uno o varios, separados por coma;
               un canal privado con el bot como admin también es un chat id, -100…)
  - WhatsApp:  CALLMEBOT_PHONE + CALLMEBOT_APIKEY   (CallMeBot, solo uso personal)

El texto viene con formato de WhatsApp (*negrita*, _cursiva_) y enlaces como [texto](url); para
Telegram se convierte a HTML (el enlace queda dentro del texto, sin ocupar espacio) y para WhatsApp
el enlace va escrito después del texto. Los mensajes largos se parten por bloques (una oportunidad
por bloque).

Cada llamada a Telegram se reintenta (REINTENTOS) si falla la red o el servidor: el 9-nov la alerta no
salió porque la conexión se cortó una vez («Connection reset by peer») y el programa se detuvo. Lo que no
se arregla reintentando (chat no encontrado, bot bloqueado) no se reintenta. Un destino que falla no
detiene a los demás; al final se dice cuáles fallaron y el programa termina con error.

Uso:
    python -m src.notificar reports/oportunidades/cali_2026-10-23_whatsapp.txt
"""
from __future__ import annotations
import html
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

LIMITE = {"telegram": 3500, "whatsapp": 1500}   # caracteres por mensaje (con margen)
REINTENTOS = (5, 20, 60)    # segundos de espera antes de cada nuevo intento (4 intentos en total)
PAUSA_PARTES = 1            # segundos entre los mensajes de una misma alerta (Telegram: ~1 por segundo por chat)


ENLACE = re.compile(r"\[([^\]\n]+)\]\((https?://[^)\s]+)\)")


def _visible(texto: str) -> int:
    """Largo del texto como lo cuenta Telegram: de un [texto](url) solo cuenta el texto."""
    return len(ENLACE.sub(r"\1", texto))


def partir(texto: str, limite: int) -> list[str]:
    """Agrupa los bloques (separados por línea en blanco) en mensajes de hasta `limite` (visibles)."""
    partes, actual = [], ""
    for bloque in texto.strip().split("\n\n"):
        if actual and _visible(actual) + _visible(bloque) + 2 > limite:
            partes.append(actual)
            actual = bloque
        else:
            actual = f"{actual}\n\n{bloque}" if actual else bloque
    return partes + [actual] if actual else partes


def _a_html(texto: str) -> str:
    t = html.escape(texto)
    t = ENLACE.sub(r'<a href="\2">\1</a>', t)
    t = re.sub(r"\*(.+?)\*", r"<b>\1</b>", t)
    return re.sub(r"(?m)^_(.+)_$", r"<i>\1</i>", t)


def _get(url: str) -> str:
    with urllib.request.urlopen(url, timeout=60) as r:
        return r.read().decode("utf-8", "replace")


def _api(metodo: str, token: str, datos: bytes, cabeceras: dict | None = None, timeout: int = 60) -> dict:
    """Llama a la API de Telegram (`metodo`: sendMessage, sendDocument…) con reintentos: si se corta la conexión,
    se agota el tiempo, el servidor falla (5xx) o pide esperar (429, respetando su `retry_after`). Lo que no se
    arregla reintentando (400 chat no encontrado, 403 bot bloqueado…) es un RuntimeError inmediato. La URL lleva el
    token: nunca se imprime."""
    url = f"https://api.telegram.org/bot{token}/{metodo}"
    esperas = [0, *REINTENTOS]
    for intento, espera in enumerate(esperas, 1):
        time.sleep(espera)
        try:
            pedido = urllib.request.Request(url, data=datos, headers=cabeceras or {})
            with urllib.request.urlopen(pedido, timeout=timeout) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            cuerpo = json.loads(e.read() or b"{}")
            detalle = cuerpo.get("description", f"HTTP {e.code}")
            if e.code < 500 and e.code != 429:
                raise RuntimeError(detalle) from None          # p. ej. 403 bot bloqueado, 400 chat not found
            if e.code == 429 and intento < len(esperas):        # Telegram dice cuánto esperar
                esperas[intento] = max(esperas[intento], cuerpo.get("parameters", {}).get("retry_after", 0))
            error = detalle
        except (urllib.error.URLError, OSError) as e:          # conexión cortada o rechazada, tiempo agotado
            error = str(getattr(e, "reason", e))
        print(f"⚠️  Telegram {metodo}: {error} (intento {intento} de {len(esperas)})", file=sys.stderr)
    raise RuntimeError(f"sin respuesta de Telegram tras {len(esperas)} intentos ({error})")


def telegram(texto: str, token: str, chat_id: str) -> None:
    for i, parte in enumerate(partir(texto, LIMITE["telegram"])):
        time.sleep(PAUSA_PARTES if i else 0)
        _api("sendMessage", token, urllib.parse.urlencode({
            "chat_id": chat_id, "text": _a_html(parte), "parse_mode": "HTML",
            "disable_web_page_preview": "true"}).encode())


def documento(ruta: str, token: str, chat_id: str, texto: str = "") -> None:
    """Un archivo (p. ej. el PDF de src/informe_semanal.py) por Telegram, con `texto` como pie."""
    import mimetypes
    import uuid
    from pathlib import Path
    p, limite = Path(ruta), uuid.uuid4().hex
    campos = {"chat_id": chat_id, "caption": _a_html(texto)[:1024], "parse_mode": "HTML"}
    cuerpo = b"".join(f'--{limite}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode()
                      for k, v in campos.items())
    tipo = mimetypes.guess_type(p.name)[0] or "application/octet-stream"
    cuerpo += (f'--{limite}\r\nContent-Disposition: form-data; name="document"; filename="{p.name}"\r\n'
               f"Content-Type: {tipo}\r\n\r\n").encode() + p.read_bytes() + f"\r\n--{limite}--\r\n".encode()
    _api("sendDocument", token, cuerpo, {"Content-Type": f"multipart/form-data; boundary={limite}"}, timeout=120)


def documento_a_todos(ruta: str, token: str, chats: str, texto: str = "") -> list[str]:
    """El archivo a cada chat de `chats` (separados por coma), sin detenerse si uno falla: la lista de fallos."""
    fallos = []
    for chat in filter(None, map(str.strip, chats.split(","))):
        try:
            documento(ruta, token, chat, texto)
        except RuntimeError as e:
            fallos.append(f"Telegram {chat}: {e}")
    return fallos


def whatsapp(texto: str, phone: str, apikey: str) -> None:
    for parte in partir(ENLACE.sub(r"\1: \2", texto), LIMITE["whatsapp"]):
        q = urllib.parse.urlencode({"phone": phone, "text": parte, "apikey": apikey})
        respuesta = _get(f"https://api.callmebot.com/whatsapp.php?{q}")
        if "error" in respuesta.lower():
            raise RuntimeError(f"CallMeBot: {respuesta[:200]}")
        time.sleep(3)   # CallMeBot no documenta límites: espaciamos los envíos


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit("Uso: python -m src.notificar <archivo.txt>")
    texto = open(sys.argv[1], encoding="utf-8").read()
    env = os.environ.get

    enviados, fallos = [], []
    if env("TELEGRAM_BOT_TOKEN") and env("TELEGRAM_CHAT_ID"):
        # Varios destinos separados por coma: tu chat, cada agente, un canal…
        # Si uno falla (p. ej. bloqueó el bot) se sigue con los demás.
        chats = [c.strip() for c in env("TELEGRAM_CHAT_ID").split(",") if c.strip()]
        ok = 0
        for chat in chats:
            try:
                telegram(texto, env("TELEGRAM_BOT_TOKEN"), chat)
                ok += 1
            except RuntimeError as e:
                fallos.append(f"Telegram {chat}: {e}")
        enviados.append(f"Telegram ({ok}/{len(chats)} destinos)")
    if env("CALLMEBOT_PHONE") and env("CALLMEBOT_APIKEY"):
        try:
            whatsapp(texto, env("CALLMEBOT_PHONE"), env("CALLMEBOT_APIKEY"))
            enviados.append("WhatsApp")
        except (RuntimeError, OSError) as e:
            fallos.append(f"WhatsApp: {e}")
    if not enviados and not fallos:
        sys.exit("❌ No hay credenciales: define TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID "
                 "o CALLMEBOT_PHONE/CALLMEBOT_APIKEY.")
    if enviados:
        print(f"✅ Enviado por {' y '.join(enviados)}")
    if fallos:
        sys.exit("❌ Fallaron algunos envíos:\n  " + "\n  ".join(fallos))


if __name__ == "__main__":
    main()
