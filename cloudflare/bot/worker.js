// Botones del canal público (Cloudflare Worker): recibe los clics de «Publicar» / «Descartar» que el
// administrador pulsa en las propuestas del día (src/canal.py proponer) y lanza la publicación a su hora.
//
//   fetch      Telegram llama aquí (webhook) en cada clic. Se comprueba la clave secreta de la cabecera
//              X-Telegram-Bot-Api-Secret-Token y que el clic venga del chat del administrador. «Publicar» →
//              GitHub lanza canal.yml con accion=aprobar y el id del mensaje (queda en cola o sale ya);
//              «Descartar» → nada más. En los dos casos el botón cambia para que se vea que ya se pulsó.
//   scheduled  "0 12 * * *" todos los días 12:00 UTC = 07:00 Colombia → canal.yml con accion=publicar.
//
// Va aparte del disparador (cloudflare/disparador/) para que un fallo aquí no toque el barrido.
//
// Configuración (Cloudflare → Workers → este Worker → Settings):
//   Variables: REPO, RAMA ("main")
//   Secretos:  GITHUB_TOKEN (el mismo tipo de token que el disparador: SOLO este repo, "Actions: Read and
//              write"), TELEGRAM_BOT_TOKEN, TELEGRAM_ADMIN_CHAT_ID, WEBHOOK_SECRET (una clave al azar; la misma
//              que se le da a Telegram con setWebhook)
//   Domains & Routes: la URL workers.dev activada (Telegram necesita una dirección a la que llamar).
//
// Ojo: con el webhook activo, el bot ya no sirve para getUpdates (averiguar el id de un chat nuevo): antes
// hay que quitarlo (deleteWebhook) y volver a ponerlo después.

const WORKFLOW = "canal.yml";

export default {
  async fetch(peticion, env, ctx) {
    if (peticion.method !== "POST") return new Response("ok");
    if (peticion.headers.get("X-Telegram-Bot-Api-Secret-Token") !== env.WEBHOOK_SECRET) {
      return new Response("no", { status: 403 });
    }
    const u = await peticion.json();
    const clic = u.callback_query;
    if (!clic || !clic.message) return new Response("ok");             // otros mensajes al bot: se ignoran
    const chat = String(clic.message.chat.id);
    if (chat !== String(env.TELEGRAM_ADMIN_CHAT_ID)) {
      ctx.waitUntil(telegram(env, "answerCallbackQuery", { callback_query_id: clic.id, text: "No autorizado" }));
      return new Response("ok");
    }
    const mensaje = clic.message.message_id;
    let etiqueta, aviso;
    if (clic.data === "pub") {
      etiqueta = "✅ Aprobada · sale a las 07:00 (o ya, si es más tarde)";
      aviso = "Aprobada";
      ctx.waitUntil(lanzar(env, { accion: "aprobar", mensaje: String(mensaje) }));
    } else if (clic.data === "des") {
      etiqueta = "✖️ Descartada";
      aviso = "Descartada";
    } else {                                                            // un botón ya pulsado
      ctx.waitUntil(telegram(env, "answerCallbackQuery", { callback_query_id: clic.id }));
      return new Response("ok");
    }
    ctx.waitUntil(Promise.all([
      telegram(env, "answerCallbackQuery", { callback_query_id: clic.id, text: aviso }),
      telegram(env, "editMessageReplyMarkup", {
        chat_id: chat, message_id: mensaje,
        reply_markup: { inline_keyboard: [[{ text: etiqueta, callback_data: "hecho" }]] },
      }),
    ]));
    return new Response("ok");
  },

  async scheduled(evento, env, ctx) {
    ctx.waitUntil(lanzar(env, { accion: "publicar" }));
  },
};

async function lanzar(env, inputs) {
  let detalle;
  try {
    const r = await fetch(`https://api.github.com/repos/${env.REPO}/actions/workflows/${WORKFLOW}/dispatches`, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${env.GITHUB_TOKEN}`,
        Accept: "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "bot-canal",                   // GitHub rechaza llamadas sin User-Agent
      },
      body: JSON.stringify({ ref: env.RAMA || "main", inputs }),
    });
    if (r.status === 204) return;
    detalle = `GitHub respondió ${r.status}: ${(await r.text()).slice(0, 300)}`;
  } catch (e) {
    detalle = `Error de red: ${e}`;
  }
  await telegram(env, "sendMessage", {
    chat_id: env.TELEGRAM_ADMIN_CHAT_ID,
    text: `⚠️ No pude lanzar ${WORKFLOW} (${inputs.accion})\n${detalle}`,
  });
  throw new Error(detalle);
}

function telegram(env, metodo, cuerpo) {
  return fetch(`https://api.telegram.org/bot${env.TELEGRAM_BOT_TOKEN}/${metodo}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(cuerpo),
  });
}
