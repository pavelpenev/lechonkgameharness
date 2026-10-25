// WP0 spectate smoke, pass 4: attempt observer attach to the running harness
// game (server port 6300 / proxy 7300) via the webclient observe URL.
const { chromium } = require('playwright');
const fs = require('fs');
const path = require('path');

const OUT = path.join(__dirname, '..', 'artifacts', 'runs', 'wp0-spectate');
const CANDIDATES = [
  'http://localhost:8080/webclient?action=observe&port=6300',
  'http://localhost:8080/webclient?action=observe&port=7300',
];

(async () => {
  fs.mkdirSync(OUT, { recursive: true });
  const browser = await chromium.launch();
  const context = await browser.newContext({
    viewport: { width: 1280, height: 900 },
    recordVideo: { dir: OUT, size: { width: 1280, height: 900 } },
  });
  const page = await context.newPage();
  for (const url of CANDIDATES) {
    try {
      await page.goto(url, { waitUntil: 'load', timeout: 30000 });
      await page.waitForTimeout(12000);
      const text = await page.evaluate(() => document.body.innerText);
      console.log('[url]', url);
      console.log('[title]', await page.title());
      console.log('[body]\n' + text.slice(0, 1200));
      const shot = path.join(OUT, `observe-${url.split('port=')[1]}.png`);
      await page.screenshot({ path: shot });
      console.log('[screenshot]', shot);
      if (/observer|observe|myagent|turn/i.test(text)) {
        console.log('[verdict] OBSERVE-SIGNALS-FOUND for', url);
        break;
      }
    } catch (e) {
      console.log('[error]', url, String(e).slice(0, 200));
    }
  }
  await context.close();
  await browser.close();
})();
