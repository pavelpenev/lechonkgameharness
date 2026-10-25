// WP0 spectate smoke, pass 2: load the freeciv-web client headlessly,
// record video, dump visible text and links to locate the running game.
const { chromium } = require('playwright');
const fs = require('fs');
const path = require('path');

const OUT = path.join(__dirname, '..', 'artifacts', 'runs', 'wp0-spectate');

(async () => {
  fs.mkdirSync(OUT, { recursive: true });
  const browser = await chromium.launch();
  const context = await browser.newContext({
    viewport: { width: 1280, height: 900 },
    recordVideo: { dir: OUT, size: { width: 1280, height: 900 } },
  });
  const page = await context.newPage();
  await page.goto('http://localhost:8080', { waitUntil: 'load', timeout: 30000 });
  await page.waitForTimeout(8000);
  console.log('[title]', await page.title());
  await page.screenshot({ path: path.join(OUT, 'pass2.png') });
  const text = await page.evaluate(() => document.body.innerText);
  console.log('[body]\n' + text.slice(0, 2500));
  const links = await page.evaluate(() =>
    [...document.querySelectorAll('a')].map(a => a.href).filter(Boolean).slice(0, 25));
  console.log('[links]', JSON.stringify(links, null, 1));
  // save the video
  await context.close();
  await browser.close();
  for (const f of fs.readdirSync(OUT)) {
    if (f.endsWith('.webm')) console.log('[video]', path.join(OUT, f));
  }
})();
