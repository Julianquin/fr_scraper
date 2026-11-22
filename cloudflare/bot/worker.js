// Botones del canal público (Cloudflare Worker): recibe los clics de «Publicar» / «Descartar» que el
// administrador pulsa en las propuestas del día (src/canal.py proponer) y lanza la publicación a su hora.
//
//   fetch      Telegram llama aquí (webhook) en cada clic. Se comprueba la clave secreta de la cabecera
//              X-Telegram-Bot-Api-Secret-Token y que el clic venga del chat del administrador. «Publicar» →
//              GitHub lanza canal.yml con accion=aprobar y el id del mensaje (queda en cola o sale ya);
//              «Descartar» → nada más. En los dos casos el botón cambia para que se vea que ya se pulsó.
//              Y los mensajes de los agentes (bot de consulta, src/consulta.py): la consulta va a R2 y se lanza
//              consulta.yml con su id; quien no tiene acceso recibe un aviso y el administrador, su número de chat.
//              Y «📲 Publicar en Instagram» (callback ig:<pieza>) en las piezas de la fábrica de contenido: lanza
//              instagram.yml en el repositorio privado de la fábrica (REPO_MARCA) con la pieza y el id del mensaje;
//              el botón pasa a «⏳ Publicando…» y ese flujo lo deja en «✅ Publicado» (o «🔁 Reintentar»).
//   scheduled  "0 12 * * *" todos los días 12:00 UTC = 07:00 Colombia → canal.yml con accion=publicar.
//
// Va aparte del disparador (cloudflare/disparador/) para que un fallo aquí no toque el barrido.
//
// Configuración (Cloudflare → Workers → este Worker → Settings):
//   Variables: REPO, RAMA ("main")
//   Secretos:  GITHUB_TOKEN (el mismo tipo de token que el disparador: SOLO este repo, "Actions: Read and
//              write"), TELEGRAM_BOT_TOKEN, TELEGRAM_ADMIN_CHAT_ID, WEBHOOK_SECRET (una clave al azar; la misma
//              que se le da a Telegram con setWebhook), AGENTES (los chats de los agentes con acceso al bot de
//              consulta, separados por coma; el administrador siempre lo tiene), REPO_MARCA y
//              GITHUB_TOKEN_MARCA (el repositorio privado de la fábrica, «usuario/nombre», y un token SOLO de ese
//              repo con "Actions: Read and write"; secretos y no variables porque este repositorio es público)
//   Bindings:  R2 bucket «fr-scraper-datos» con el nombre DATOS (las consultas pendientes)
//   Domains & Routes: la URL workers.dev activada (Telegram necesita una dirección a la que llamar).
//
// El webhook recibe callback_query (los botones) y message (las consultas): setWebhook con
// allowed_updates=["callback_query","message"].
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
    if (u.message) return consulta(u.message, env, ctx);              // un agente escribe al bot (src/consulta.py)
    const clic = u.callback_query;
    if (!clic || !clic.message) return new Response("ok");             // otros avisos de Telegram: se ignoran
    const chat = String(clic.message.chat.id);
    if (chat !== String(env.TELEGRAM_ADMIN_CHAT_ID)) {
      ctx.waitUntil(telegram(env, "answerCallbackQuery", { callback_query_id: clic.id, text: "No autorizado" }));
      return new Response("ok");
    }
    const mensaje = clic.message.message_id;
    let etiqueta, aviso;
    if (clic.data === "pub") {
      // Lo mismo que decide src/canal.py aprobar: antes de las 07:00 de Colombia, a la cola; después, al momento
      const horaColombia = (new Date().getUTCHours() + 24 - 5) % 24;
      etiqueta = horaColombia >= 7 ? "✅ Publicada en el canal" : "✅ Aprobada · sale a las 07:00 (Colombia)";
      aviso = "Aprobada";
      ctx.waitUntil(lanzar(env, { accion: "aprobar", mensaje: String(mensaje) }));
    } else if (clic.data === "des") {
      etiqueta = "✖️ Descartada";
      aviso = "Descartada";
    } else if (/^ig:[\w-]{1,60}$/.test(clic.data || "")) {
      return instagram(clic, chat, mensaje, env, ctx);
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

// ── El bot de consulta: un agente escribe una dirección, un enlace o «apto 80 m², 3 hab, Ciudad Jardín» ───────────
// El texto y el chat se guardan en R2 (consultas/pendientes/<id>.json) y consulta.yml recibe solo el id: el
// repositorio es público y sus registros también, y la dirección del cliente de un agente no puede quedar en ellos.
const AYUDA = [                                        // la misma que src/consulta.py (AYUDA)
  "Escríbeme una de estas tres cosas y te mando el informe de precio para captar (en unos minutos):",
  "",
  "• La dirección (en Cali), con la torre y el apartamento si es un edificio:",
  "  KR 98 B # 48-127 TO 3 AP 436",
  "• El enlace de un aviso publicado.",
  "• Una descripción con el tipo, el área y el barrio (y si quieres, habitaciones, baños y estrato):",
  "  apto 80 m², 3 hab, 2 baños, estrato 4, Ciudad Jardín",
  "",
  "Es una estimación del precio de mercado pedido, no un avalúo.",
].join("\n");

async function consulta(m, env, ctx) {
  if (m.chat?.type !== "private" || !m.text) return new Response("ok");      // solo chats privados y con texto
  const chat = String(m.chat.id);
  const autorizados = new Set([String(env.TELEGRAM_ADMIN_CHAT_ID),
    ...String(env.AGENTES || "").split(",").map((x) => x.trim()).filter(Boolean)]);
  const responder = (texto) => telegram(env, "sendMessage", { chat_id: chat, text: texto });
  if (!autorizados.has(chat)) {
    // Al administrador, quién fue: así, para dar de alta a un agente basta con que escriba al bot
    const quien = [m.from?.first_name, m.from?.last_name].filter(Boolean).join(" ") || "(sin nombre)";
    ctx.waitUntil(Promise.all([
      responder("Este servicio es solo para los agentes del piloto. Si quieres probarlo, escríbenos."),
      telegram(env, "sendMessage", { chat_id: env.TELEGRAM_ADMIN_CHAT_ID,
        text: `🔒 Escribió al bot alguien sin acceso: ${quien}${m.from?.username ? " (@" + m.from.username + ")" : ""} · ` +
              `chat ${chat}. Para darle acceso, añade ese número a AGENTES en el Worker.` }),
    ]));
    return new Response("ok");
  }
  if (/^\/(start|ayuda|help)\b/i.test(m.text.trim())) {
    ctx.waitUntil(responder(AYUDA));
    return new Response("ok");
  }
  const id = crypto.randomUUID();
  ctx.waitUntil((async () => {
    await env.DATOS.put(`consultas/pendientes/${id}.json`, JSON.stringify({
      chat, texto: m.text.slice(0, 500), recibido: new Date().toISOString(),
    }), { httpMetadata: { contentType: "application/json" } });
    await responder("⏳ Recibido. En unos minutos te llega el informe.");
    await lanzar(env, { id }, "consulta.yml");
  })().catch(() => responder("⚠️ No pude recibir la consulta. Prueba de nuevo en un rato.")));
  return new Response("ok");
}

// ── «📲 Publicar en Instagram»: una pieza de la fábrica de contenido (callback ig:<pieza>) ─────────────────────────
// La publica instagram.yml del repositorio privado de la fábrica, que al terminar cambia el botón otra vez. Aquí
// solo se lanza y se marca el botón como en marcha (así un segundo clic no hace nada).
function instagram(clic, chat, mensaje, env, ctx) {
  const botones = (texto, data) => telegram(env, "editMessageReplyMarkup", {
    chat_id: chat, message_id: mensaje, reply_markup: { inline_keyboard: [[{ text: texto, callback_data: data }]] },
  });
  ctx.waitUntil((async () => {
    await Promise.all([
      telegram(env, "answerCallbackQuery", { callback_query_id: clic.id, text: "Publicando en Instagram…" }),
      botones("⏳ Publicando en Instagram…", "hecho"),
    ]);
    try {
      await lanzar(env, { accion: "publicar", pieza: clic.data.slice(3), mensaje: String(mensaje) },
                   "instagram.yml", env.REPO_MARCA, env.GITHUB_TOKEN_MARCA);
    } catch {
      await botones("🔁 Reintentar publicar en Instagram", clic.data);   // lanzar() ya avisó del motivo
    }
  })());
  return new Response("ok");
}

async function lanzar(env, inputs, workflow = WORKFLOW, repo = env.REPO, token = env.GITHUB_TOKEN) {
  let detalle;
  try {
    if (!repo || !token) throw new Error("falta el repositorio o su token en la configuración del Worker");
    const r = await fetch(`https://api.github.com/repos/${repo}/actions/workflows/${workflow}/dispatches`, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${token}`,
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
    text: `⚠️ No pude lanzar ${workflow} (${inputs.accion || "consulta"})\n${detalle}`,
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
