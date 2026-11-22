// Disparador del pipeline de fr_scraper (Cloudflare Worker con Cron Triggers).
//
// GitHub atrasa sus tareas programadas 4–6 h; los Cron Triggers de Cloudflare son puntuales.
// Este Worker le pide a GitHub (workflow_dispatch) que lance los workflows a la hora exacta:
//
//   "0 5 * * *"   todos los días 05:00 UTC = 00:00 Colombia → env.WORKFLOW
//                 (barrido_http.yml desde el cambio de fuente del 30-oct; encadena Metrocuadrado,
//                 el mapa y, los domingos, el modelo)
//   "0 12 * * MON" los lunes 12:00 UTC = 07:00 Colombia → oportunidades.yml (la alerta). Ojo: en
//                 Cloudflare el día 1 es DOMINGO (no lunes como en GitHub): por eso MON y no 1
//   "30 12 * * *" todos los días 12:30 UTC = 07:30 Colombia → la fábrica de contenido (fabrica.yml en su
//                 repositorio privado, REPO_MARCA, con accion=diario y enviar=true), después de la publicación
//                 del canal de las 07:00
//
// Configuración (Cloudflare → Workers → fr-scraper-disparador → Settings):
//   Variables: REPO ("Julianquin/fr_scraper"), WORKFLOW, RAMA ("main")
//   Secretos:  GITHUB_TOKEN (token fine-grained SOLO de este repo, permiso "Actions: Read and
//              write"), TELEGRAM_BOT_TOKEN y TELEGRAM_ADMIN_CHAT_ID (aviso si algo falla),
//              REPO_MARCA y GITHUB_TOKEN_MARCA (el repositorio privado de la fábrica, «usuario/nombre», y un
//              token SOLO de ese repo con "Actions: Read and write"; secretos porque este repositorio es público)
//   PRUEBA="1" (opcional): en vez de lanzar, comprueba token y workflows y avisa por Telegram.
//
// El código se pega en el editor del panel; los secretos viven solo en Cloudflare.

// La alerta se reconoce por la HORA del horario (12 UTC), no por el texto completo: así da igual si
// Cloudflare escribe el día como MON o como número. Si se cambia la hora de la alerta, cambiar aquí.
const HORA_ALERTA = "12";
const ALERTA = "oportunidades.yml";
// La fábrica, por minuto y hora ("30 12"): va antes que la alerta, que comparte la hora 12
const MINUTO_FABRICA = "30", HORA_FABRICA = "12";
const FABRICA = "fabrica.yml";

export default {
  async scheduled(evento, env, ctx) {
    const [minuto, hora] = evento.cron.trim().split(/\s+/);
    if (minuto === MINUTO_FABRICA && hora === HORA_FABRICA) {
      const marca = { repo: env.REPO_MARCA, token: env.GITHUB_TOKEN_MARCA, nombre: "el repositorio de la fábrica" };
      console.log(`cron "${evento.cron}" · fábrica · REPO_MARCA ${env.REPO_MARCA ? "sí" : "NO"} · ` +
                  `GITHUB_TOKEN_MARCA ${env.GITHUB_TOKEN_MARCA ? "sí" : "NO"}`);
      ctx.waitUntil(env.PRUEBA ? probar(env, [FABRICA], marca)
                               : disparar(env, FABRICA, { accion: "diario", enviar: "true" }, marca));
      return;
    }
    const workflow = hora === HORA_ALERTA ? ALERTA : env.WORKFLOW;
    // Solo si cada variable existe (nunca su valor): para diagnosticar en Observability → Logs
    const hay = (k) => (env[k] ? "sí" : "NO");
    console.log(`cron "${evento.cron}" · PRUEBA ${hay("PRUEBA")} · REPO ${env.REPO} · WORKFLOW ${env.WORKFLOW} · ` +
                `GITHUB_TOKEN ${hay("GITHUB_TOKEN")} · TELEGRAM_BOT_TOKEN ${hay("TELEGRAM_BOT_TOKEN")} · ` +
                `TELEGRAM_ADMIN_CHAT_ID ${hay("TELEGRAM_ADMIN_CHAT_ID")}`);
    ctx.waitUntil(env.PRUEBA ? probar(env, [env.WORKFLOW, ALERTA]) : disparar(env, workflow));
  },
};

// `destino`: el repositorio y su token (por defecto, los de fr_scraper); `nombre`, cómo se llama en los avisos
// (el de la fábrica no se escribe: es privado)
function propio(env) {
  return { repo: env.REPO, token: env.GITHUB_TOKEN, nombre: env.REPO };
}

function github(env, ruta, opciones = {}, destino = propio(env)) {
  if (!destino.repo || !destino.token) {
    return Promise.resolve(new Response("falta el repositorio o su token en el Worker", { status: 499 }));
  }
  return fetch(`https://api.github.com/repos/${destino.repo}/actions/workflows/${ruta}`, {
    ...opciones,
    headers: {
      Authorization: `Bearer ${destino.token}`,
      Accept: "application/vnd.github+json",
      "X-GitHub-Api-Version": "2022-11-28",
      "User-Agent": "fr-scraper-disparador",       // GitHub rechaza llamadas sin User-Agent
    },
  });
}

async function disparar(env, workflow, inputs = undefined, destino = propio(env)) {
  let detalle;
  try {
    const r = await github(env, `${workflow}/dispatches`, {
      method: "POST",
      body: JSON.stringify({ ref: env.RAMA || "main", ...(inputs ? { inputs } : {}) }),
    }, destino);
    if (r.status === 204) {                           // lanzado
      console.log(`${workflow} lanzado en ${destino.nombre} (${env.RAMA || "main"})`);
      return;
    }
    // 401: token vencido o inválido · 404: repo o workflow mal escrito, o token sin acceso
    // 422: el workflow no tiene workflow_dispatch o la rama no existe
    detalle = `GitHub respondió ${r.status}: ${(await r.text()).slice(0, 300)}`;
  } catch (e) {
    detalle = `Error de red: ${e}`;
  }
  await avisar(env, `⚠️ El disparador de Cloudflare no pudo lanzar ${workflow}\n${detalle}`);
  throw new Error(detalle);                           // queda como fallido en los logs del Worker
}

async function probar(env, workflows, destino = propio(env)) {
  const lineas = [];
  for (const w of workflows) {
    const r = await github(env, w, {}, destino);
    lineas.push(`${r.status === 200 ? "✅" : "❌"} ${w}: ${r.status}`);
  }
  console.log(`prueba: ${lineas.join(" · ")}`);
  await avisar(env, `🧪 Prueba del disparador (${destino.nombre}, rama ${env.RAMA || "main"})\n${lineas.join("\n")}`);
}

async function avisar(env, texto) {
  if (!env.TELEGRAM_BOT_TOKEN || !env.TELEGRAM_ADMIN_CHAT_ID) {
    console.error("Sin TELEGRAM_BOT_TOKEN o TELEGRAM_ADMIN_CHAT_ID: no se avisa");
    return;
  }
  const r = await fetch(`https://api.telegram.org/bot${env.TELEGRAM_BOT_TOKEN}/sendMessage`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ chat_id: env.TELEGRAM_ADMIN_CHAT_ID, text: texto }),
  });
  // 401/404: token del bot mal copiado · 400 "chat not found": chat_id mal copiado
  if (!r.ok) throw new Error(`Telegram respondió ${r.status}: ${(await r.text()).slice(0, 200)}`);
}
