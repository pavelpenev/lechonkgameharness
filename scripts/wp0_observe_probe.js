// WP0-6 pass A: probe the game-details page for the running harness game
// (server port 6300) and dump every link/button/handler we can find.
// Read-only probe: never click anything that could join a game.
const { chromium } = require('playwright');
const fs = require('fs');
const path = require('path');

const OUT = path.join(__dirname, '..', 'artifacts', 'runs', 'wp0-spectate');
const DETAILS = 'http://localhost:8080/game/details?host=unknown&port=6300';

(async () => {
  fs.mkdirSync(OUT, { recursive: true });
  const browser = await chromium.launch();
  const context = await browser.newContext({
    viewport: { width: 1280, height: 900 },
  });
  const page = await context.newPage();
  await page.goto(DETAILS, { waitUntil: 'load', timeout: 30000 });
  await page.waitForTimeout(3000);
  console.log('[url]', page.url());
  await page.screenshot({ path: path.join(OUT, 'details-probe.png') });

  const text = await page.evaluate(() => document.body.innerText);
  console.log('[body]\n' + text.slice(0, 4000));

  const links = await page.evaluate(() =>
    [...document.querySelectorAll('a')].map(a => ({
      text: (a.innerText || '').trim().slice(0, 60),
      href: a.href,
      onclick: a.getAttribute('onclick'),
    })).filter(x => x.text || x.onclick)
  );
  console.log('[links]', JSON.stringify(links, null, 1));

  const buttons = await page.evaluate(() =>
    [...document.querySelectorAll('button, input[type=submit], input[type=button]')].map(b => ({
      tag: b.tagName, text: (b.innerText || b.value || '').trim().slice(0, 60),
      onclick: b.getAttribute('onclick'), id: b.id,
    })).filter(x => x.text || x.onclick)
  );
  console.log('[buttons]', JSON.stringify(buttons, null, 1));

  // any element whose class/id mentions observe
  const observeEls = await page.evaluate(() =>
    [...document.querySelectorAll('[class*=observe i], [id*=observe i], [href*=observe i]')].map(e => ({
      tag: e.tagName, id: e.id, cls: e.className, href: e.href || null,
      onclick: e.getAttribute('onclick'), text: (e.innerText || '').trim().slice(0, 80),
    }))
  );
  console.log('[observe-els]', JSON.stringify(observeEls, null, 1));

  await context.close();
  await browser.close();
})().catch(e => { console.error('[fatal]', e); process.exit(1); });
