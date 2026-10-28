// Disparador del pipeline de fr_scraper (Cloudflare Worker con Cron Triggers).
//
// GitHub atrasa sus tareas programadas 4–6 h; los Cron Triggers de Cloudflare son puntuales.
// Este Worker le pide a GitHub (workflow_dispatch) que lance los workflows a la hora exacta:
//
//   "0 5 * * *"   todos los días 05:00 UTC = 00:00 Colombia → env.WORKFLOW
//                 (scraper.yml hasta el cambio de fuente; desde el 2-nov, barrido_http.yml,
//                 que encadena el mapa y Metrocuadrado)
//   "0 12 * * 1"  los lunes 12:00 UTC = 07:00 Colombia → oportunidades.yml (la alerta)
//
// Configuración (Cloudflare → Workers → fr-scraper-disparador → Settings):
//   Variables: REPO ("Julianquin/fr_scraper"), WORKFLOW, RAMA ("main")
//   Secretos:  GITHUB_TOKEN (token fine-grained SOLO de este repo, permiso "Actions: Read and
//              write"), TELEGRAM_BOT_TOKEN y TELEGRAM_ADMIN_CHAT_ID (aviso si algo falla)
//   PRUEBA="1" (opcional): en vez de lanzar, comprueba token y workflows y avisa por Telegram.
//
// El código se pega en el editor del panel; los secretos viven solo en Cloudflare.

const CRON_ALERTA = "0 12 * * 1";
const ALERTA = "oportunidades.yml";

export default {
  async scheduled(evento, env, ctx) {
    const workflow = evento.cron === CRON_ALERTA ? ALERTA : env.WORKFLOW;
    ctx.waitUntil(env.PRUEBA ? probar(env, [env.WORKFLOW, ALERTA]) : disparar(env, workflow));
  },
};

function github(env, ruta, opciones = {}) {
  return fetch(`https://api.github.com/repos/${env.REPO}/actions/workflows/${ruta}`, {
    ...opciones,
    headers: {
      Authorization: `Bearer ${env.GITHUB_TOKEN}`,
      Accept: "application/vnd.github+json",
      "X-GitHub-Api-Version": "2022-11-28",
      "User-Agent": "fr-scraper-disparador",       // GitHub rechaza llamadas sin User-Agent
    },
  });
}

async function disparar(env, workflow) {
  let detalle;
  try {
    const r = await github(env, `${workflow}/dispatches`, {
      method: "POST",
      body: JSON.stringify({ ref: env.RAMA || "main" }),
    });
    if (r.status === 204) return;                     // lanzado
    // 401: token vencido o inválido · 404: repo o workflow mal escrito, o token sin acceso
    // 422: el workflow no tiene workflow_dispatch o la rama no existe
    detalle = `GitHub respondió ${r.status}: ${(await r.text()).slice(0, 300)}`;
  } catch (e) {
    detalle = `Error de red: ${e}`;
  }
  await avisar(env, `⚠️ El disparador de Cloudflare no pudo lanzar ${workflow}\n${detalle}`);
  throw new Error(detalle);                           // queda como fallido en los logs del Worker
}

async function probar(env, workflows) {
  const lineas = [];
  for (const w of workflows) {
    const r = await github(env, w);
    lineas.push(`${r.status === 200 ? "✅" : "❌"} ${w}: ${r.status}`);
  }
  await avisar(env, `🧪 Prueba del disparador (${env.REPO}, rama ${env.RAMA || "main"})\n${lineas.join("\n")}`);
}

async function avisar(env, texto) {
  if (!env.TELEGRAM_BOT_TOKEN || !env.TELEGRAM_ADMIN_CHAT_ID) return;
  await fetch(`https://api.telegram.org/bot${env.TELEGRAM_BOT_TOKEN}/sendMessage`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ chat_id: env.TELEGRAM_ADMIN_CHAT_ID, text: texto }),
  });
}
