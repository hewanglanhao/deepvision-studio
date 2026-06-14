const fs = require('fs');
const path = require('path');
const { pathToFileURL } = require('url');
const pptxgen = require('pptxgenjs');
const { chromium } = require('playwright');

const ROOT = path.resolve(__dirname, '..');
const HTML_FILE = path.join(ROOT, 'index.html');
const OUT_FILE = path.resolve(ROOT, process.env.PPTX_OUT_FILE || 'DeepVision-Studio.pptx');
const EXPORT_SCALE = Math.max(1, Math.min(4, Number.parseFloat(process.env.PPTX_EXPORT_SCALE || '2') || 2));

function findBrowserExecutable() {
  const candidates = [
    process.env.PPTX_BROWSER_PATH,
    path.join(process.env.ProgramFiles || '', 'Microsoft', 'Edge', 'Application', 'msedge.exe'),
    path.join(process.env['ProgramFiles(x86)'] || '', 'Microsoft', 'Edge', 'Application', 'msedge.exe'),
    path.join(process.env.LOCALAPPDATA || '', 'Microsoft', 'Edge', 'Application', 'msedge.exe'),
    path.join(process.env.ProgramFiles || '', 'Google', 'Chrome', 'Application', 'chrome.exe'),
    path.join(process.env['ProgramFiles(x86)'] || '', 'Google', 'Chrome', 'Application', 'chrome.exe'),
    path.join(process.env.LOCALAPPDATA || '', 'Google', 'Chrome', 'Application', 'chrome.exe'),
  ].filter(Boolean);

  return candidates.find((candidate) => fs.existsSync(candidate));
}

async function captureSlide(page, index) {
  await page.evaluate(async (slideIndex) => {
    const deck = document.getElementById('deck');
    const slides = Array.from(document.querySelectorAll('.slide'));
    const hud = document.querySelector('.hud');
    const notes = document.querySelector('.speaker-notes');
    if (hud) hud.style.display = 'none';
    if (notes) notes.style.display = 'none';
    document.body.classList.remove('overview');
    deck.style.transform = 'translate(-50%, -50%) scale(1)';
    slides.forEach((slide, currentIndex) => {
      slide.classList.toggle('active', currentIndex === slideIndex);
      slide.style.display = currentIndex === slideIndex ? 'flex' : 'none';
    });
    await new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve)));
  }, index);

  return page.locator('.slide.active').screenshot({ type: 'png' });
}

async function main() {
  const executablePath = findBrowserExecutable();
  const browser = await chromium.launch(executablePath ? { executablePath } : {});
  const page = await browser.newPage({
    viewport: { width: 1920, height: 1080 },
    deviceScaleFactor: EXPORT_SCALE,
  });

  await page.goto(pathToFileURL(HTML_FILE).href, { waitUntil: 'networkidle' });
  await page.evaluate(() => document.fonts?.ready);

  const slideCount = await page.locator('.slide').count();
  const pptx = new pptxgen();
  pptx.defineLayout({ name: 'DEEPVISION_WIDE', width: 13.333333, height: 7.5 });
  pptx.layout = 'DEEPVISION_WIDE';
  pptx.author = 'DeepVision Studio';
  pptx.subject = 'Screenshot export generated from course-presentation/index.html';
  pptx.title = 'DeepVision Studio';
  pptx.lang = 'zh-CN';

  for (let index = 0; index < slideCount; index += 1) {
    const image = await captureSlide(page, index);
    const slide = pptx.addSlide();
    slide.addImage({
      data: `data:image/png;base64,${image.toString('base64')}`,
      x: 0,
      y: 0,
      w: 13.333333,
      h: 7.5,
    });
  }

  await browser.close();
  await pptx.writeFile({ fileName: OUT_FILE });
  console.log(`Wrote ${OUT_FILE} (${slideCount} slides, ${EXPORT_SCALE}x screenshots)`);
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
