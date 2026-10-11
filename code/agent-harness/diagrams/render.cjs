// Render an HTML diagram as a full-size PNG with the site's local font.
const path = require('node:path');
const { pathToFileURL } = require('node:url');
const puppeteer = require('../../../node_modules/puppeteer');

async function main() {
  const [input, output] = process.argv.slice(2);
  if (!input || !output) {
    throw new Error('usage: node render.cjs INPUT.html OUTPUT.png');
  }
  const browser = await puppeteer.launch({
    headless: 'new',
    executablePath: '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
    args: ['--allow-file-access-from-files'],
  });
  try {
    const page = await browser.newPage();
    await page.goto(pathToFileURL(path.resolve(input)).href);
    const svg = await page.$('#d');
    const size = await svg.evaluate(element => ({
      width: Number(element.getAttribute('width')),
      height: Number(element.getAttribute('height')),
    }));
    await page.setViewport({ ...size, deviceScaleFactor: 1 });
    await page.waitForFunction(() => document.body.dataset.ready === '1');
    await svg.screenshot({ path: output });
  } finally {
    await browser.close();
  }
}

main().catch(error => { console.error(error); process.exitCode = 1; });
