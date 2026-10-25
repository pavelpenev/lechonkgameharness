// WP0 spectate smoke, pass 3: find the running harness game in the online
// games list and attempt a zero-perturbation observer attach.
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
  await page.waitForTimeout(3000);
  // find the "Online Games" link and follow it
  const online = page.locator('a', { hasText: 'Online Games' }).first();
  const href = await online.getAttribute('href');
  console.log('[online-games link]', href);
  await online.click();
  await page.waitForTimeout(6000);
  console.log('[url after click]', page.url());
  await page.screenshot({ path: path.join(OUT, 'online-games.png') });
  const text = await page.evaluate(() => document.body.innerText);
  console.log('[body]\n' + text.slice(0, 3000));
  await context.close();
  await browser.close();
})();
