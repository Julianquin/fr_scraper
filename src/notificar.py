"""
Envía un mensaje de texto (p. ej. la alerta de src/oportunidades.py) por Telegram y/o WhatsApp.

Usa los canales que tengan credenciales en variables de entorno (secrets en GitHub Actions):
  - Telegram:  TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID (uno o varios, separados por coma;
               un canal privado con el bot como admin también es un chat id, -100…)
  - WhatsApp:  CALLMEBOT_PHONE + CALLMEBOT_APIKEY   (CallMeBot, solo uso personal)

El texto viene con formato de WhatsApp (*negrita*, _cursiva_); para Telegram se convierte
a HTML. Los mensajes largos se parten por bloques (una oportunidad por bloque).

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


def partir(texto: str, limite: int) -> list[str]:
    """Agrupa los bloques (separados por línea en blanco) en mensajes de hasta `limite`."""
    partes, actual = [], ""
    for bloque in texto.strip().split("\n\n"):
        if actual and len(actual) + len(bloque) + 2 > limite:
            partes.append(actual)
            actual = bloque
        else:
            actual = f"{actual}\n\n{bloque}" if actual else bloque
    return partes + [actual] if actual else partes


def _a_html(texto: str) -> str:
    t = html.escape(texto)
    t = re.sub(r"\*(.+?)\*", r"<b>\1</b>", t)
    return re.sub(r"(?m)^_(.+)_$", r"<i>\1</i>", t)


def _get(url: str) -> str:
    with urllib.request.urlopen(url, timeout=60) as r:
        return r.read().decode("utf-8", "replace")


def telegram(texto: str, token: str, chat_id: str) -> None:
    for parte in partir(texto, LIMITE["telegram"]):
        datos = urllib.parse.urlencode({"chat_id": chat_id, "text": _a_html(parte),
                                        "parse_mode": "HTML",
                                        "disable_web_page_preview": "true"}).encode()
        try:
            with urllib.request.urlopen(f"https://api.telegram.org/bot{token}/sendMessage",
                                        data=datos, timeout=60) as r:
                json.load(r)
        except urllib.error.HTTPError as e:
            # p. ej. 403 "bot was blocked by the user", 400 "chat not found"
            detalle = json.loads(e.read() or b"{}").get("description", str(e))
            raise RuntimeError(detalle) from None


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
    pedido = urllib.request.Request(f"https://api.telegram.org/bot{token}/sendDocument", data=cuerpo,
                                    headers={"Content-Type": f"multipart/form-data; boundary={limite}"})
    try:
        with urllib.request.urlopen(pedido, timeout=120) as r:
            json.load(r)
    except urllib.error.HTTPError as e:
        detalle = json.loads(e.read() or b"{}").get("description", str(e))
        raise RuntimeError(detalle) from None


def whatsapp(texto: str, phone: str, apikey: str) -> None:
    for parte in partir(texto, LIMITE["whatsapp"]):
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
        whatsapp(texto, env("CALLMEBOT_PHONE"), env("CALLMEBOT_APIKEY"))
        enviados.append("WhatsApp")
    if not enviados:
        sys.exit("❌ No hay credenciales: define TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID "
                 "o CALLMEBOT_PHONE/CALLMEBOT_APIKEY.")
    print(f"✅ Enviado por {' y '.join(enviados)}")
    if fallos:
        sys.exit("❌ Fallaron algunos envíos:\n  " + "\n  ".join(fallos))


if __name__ == "__main__":
    main()
