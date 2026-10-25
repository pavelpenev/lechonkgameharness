// WP0-6: zero-perturbation observer attach to the running harness game on
// civserver port 6300 (proxy 7300), via the webclient's own observe URL:
//   /webclient/?action=observe&civserverport=6300&civserverhost=unknown&multi=true&type=multiplayer
// (pattern from WEB-INF/jsp/index.jsp:262; trailing slash after /webclient/ REQUIRED)
//
// HARD CONSTRAINT: never click "Play" or any join link. With action=observe the
// client removes the "Start Game" button and relabels the next button
// "Observe Game" (pregame.js:1162-1165); we only fill the username and click
// that. The client then sends the /observe chat command after login
// (civclient.js:440-443, packhand.js:129-130), attaching as an observer with
// client.conn.playing == null — no player slot is consumed.
const { chromium } = require('playwright');
const fs = require('fs');
const path = require('path');

const OUT = path.join(__dirname, '..', 'artifacts', 'runs', 'wp0-observe');
// usage: node scripts/wp0_observe_attach.js [civserverport]  (default 6300)
const PORT = process.argv[2] || '6300';
const URL = `http://localhost:8080/webclient/?action=observe&civserverport=${PORT}&civserverhost=unknown&multi=true&type=multiplayer`;
const USERNAME = 'wp0observer';
const SOAK_MS = 45000; // keep the observer connected to build video material

function ts() { return new Date().toISOString().slice(11, 19); }

(async () => {
  fs.mkdirSync(OUT, { recursive: true });
  const browser = await chromium.launch();
  const context = await browser.newContext({
    viewport: { width: 1280, height: 900 },
    recordVideo: { dir: OUT, size: { width: 1280, height: 900 } },
  });
  const page = await context.newPage();
  page.on('console', m => {
    const t = m.text();
    if (/observe|connect|error|login/i.test(t)) console.log(`[${ts()}] console:`, t.slice(0, 160));
  });

  console.log(`[${ts()}] goto observe URL`);
  await page.goto(URL, { waitUntil: 'load', timeout: 30000 });
  await page.waitForTimeout(8000);
  await page.screenshot({ path: path.join(OUT, 'step1-intro-dialog.png') });

  // Fill username and click the "Observe Game" button (only button, observe-labeled).
  const dialogInput = page.locator('#username_req');
  if (await dialogInput.count() > 0) {
    await dialogInput.fill(USERNAME);
    console.log(`[${ts()}] filled username:`, USERNAME);
  } else {
    console.log(`[${ts()}] WARN: intro dialog not found`);
  }
  const btn = page.getByRole('button', { name: /observe game/i }).first();
  if (await btn.count() === 0) throw new Error('Observe Game button not found');
  console.log(`[${ts()}] clicking Observe Game button`);
  await btn.click();

  console.log(`[${ts()}] waiting 25s for websocket login + /observe ...`);
  await page.waitForTimeout(25000);

  const probe = () => page.evaluate(() => {
    const txt = (id) => { const e = document.getElementById(id); return e ? e.textContent.slice(0, 900) : null; };
    return {
      observing: typeof observing !== 'undefined' ? observing : null,
      clientState: typeof civclient_state !== 'undefined' ? civclient_state : null,
      clientPlaying: (typeof client !== 'undefined' && client.conn) ? (client.conn.playing == null ? 'null (observer)' : 'PLAYER SLOT TAKEN') : 'no client.conn',
      clientObserver: (typeof client !== 'undefined' && client.conn) ? client.conn.observer : null,
      username: typeof username !== 'undefined' ? username : null,
      gameInfo: (typeof game_info !== 'undefined' && game_info) ? { turn: game_info.turn, year: game_info.year, phase: game_info.phase } : null,
      metaMessage: (typeof server_settings !== 'undefined' && server_settings && server_settings.metamessage) ? server_settings.metamessage.val : null,
      chat: txt('game_message_area') || txt('pregame_message_area') || null,
      roster: (typeof players !== 'undefined') ? Object.values(players).map(p => p.name).filter(Boolean) : null,
      canvasCount: document.querySelectorAll('canvas').length,
      statusPanel: txt('game_status_panel_top') || txt('game_status_panel_bottom') || null,
    };
  });

  const state1 = await probe();
  console.log(`[${ts()}] state after connect:`, JSON.stringify(state1, null, 1));
  await page.screenshot({ path: path.join(OUT, 'step2-attached.png') });

  // Proxy-side evidence while attached: logged-in user count on the proxy.
  const proxy = await page.evaluate((pport) =>
    fetch(`/civsocket/${pport}/status`, { credentials: 'omit' }).then(r => r.text()).then(t => {
      const m = t.match(/Logged in users\s*\(count (\d+)\)[\s\S]*?<\/h3>([\s\S]*)/);
      return m ? { count: m[1], users: m[2].slice(0, 400) } : t.slice(0, 200);
    })
  , String(1000 + Number(PORT)));
  console.log(`[${ts()}] proxy 7300 while attached:`, JSON.stringify(proxy));
  await page.screenshot({ path: path.join(OUT, 'step3-observer-view.png') });

  // Soak to let the map render and build video material.
  console.log(`[${ts()}] soaking ${SOAK_MS / 1000}s for map render + video ...`);
  await page.waitForTimeout(SOAK_MS);

  const state2 = await probe();
  console.log(`[${ts()}] state after soak:`, JSON.stringify(state2, null, 1));
  await page.screenshot({ path: path.join(OUT, 'step4-observer-soaked.png') });

  await context.close();
  await browser.close();
  for (const f of fs.readdirSync(OUT)) {
    if (f.endsWith('.webm')) console.log('[video]', path.join(OUT, f));
  }
  console.log(`[${ts()}] done`);
})().catch(e => { console.error('[fatal]', e); process.exit(1); });
