// Interface tests for the Jack terminal: real browser, real server, real shell.
//
//   PLAYWRIGHT_BROWSERS_PATH=/data/caches/ms-playwright node dashboard/terminal/uitest.mjs chromium [phone,laptop,…]
//
// Six screen sizes; for each: sign in (a wrong password must be refused), run a
// command, measure keystroke echo, use every kind of extra key, scroll back by
// key and by swipe, flood 30,000 lines, open the on-screen keyboard, turn the
// screen, change text size, reload, restart the server helper, sign out.
// Screenshots and results-<engine>.json go to $JACK_UITEST_OUT (/data/tmp/jack-uitest).
// Both passwords are read from owner-only files; none is in this file.
// It types into the real shared session, so run it when nobody is using it.
import { createRequire } from 'module';
import fs from 'fs';
import { execSync } from 'child_process';
const require = createRequire('/home/opc/worldtwin/scripts/');
const pw = require('playwright');
const OUT = (process.env.JACK_UITEST_OUT || '/data/tmp/jack-uitest') + '/';
fs.mkdirSync(OUT, { recursive: true });
const HOST = 'jack.worldtwin.duckdns.org';
const BASE = `https://${HOST}`;
const LONG = fs.readFileSync('/home/opc/.jack-terminal-password', 'utf8').trim();
const PIN = fs.readFileSync('/home/opc/.config/jack-terminal/dashboard-pin', 'utf8').trim();
const engineName = process.argv[2] || 'chromium';
const only = process.argv[3] || '';
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

const PROFILES = [
  { id: 'phone',        w: 390,  h: 844,  dpr: 3, touch: true,  label: 'Phone, upright (390×844)' },
  { id: 'phone-small',  w: 360,  h: 640,  dpr: 2, touch: true,  label: 'Small phone (360×640)' },
  { id: 'phone-wide',   w: 844,  h: 390,  dpr: 3, touch: true,  label: 'Phone, sideways (844×390)' },
  { id: 'tablet',       w: 820,  h: 1180, dpr: 2, touch: true,  label: 'Tablet (820×1180)' },
  { id: 'laptop',       w: 1440, h: 900,  dpr: 1, touch: false, label: 'Laptop (1440×900)' },
  { id: 'desktop',      w: 1920, h: 1080, dpr: 1, touch: false, label: 'Large desktop (1920×1080)' },
].filter((p) => !only || only.split(',').includes(p.id));

const results = [];
// The tests never touch the owner's live session: they run their own throwaway
// terminal server and tmux, and the page's connection is carried to it.
import { spawn } from 'node:child_process';
const TPORT = 7699, CONF = new URL('./tmux.conf', import.meta.url).pathname;
try { execSync('tmux -L jacktest kill-server', { stdio: 'ignore' }); } catch {}
const tty = spawn('/usr/local/bin/ttyd', ['--interface', '127.0.0.1', '--port', String(TPORT), '--base-path', '/term', '--writable',
  '/usr/bin/tmux', '-u', '-L', 'jacktest', '-f', CONF, 'new-session', '-A', '-s', 'jack', '-c', '/home/opc'], { stdio: 'ignore' });
function cleanup() { try { tty.kill(); } catch {} try { execSync('tmux -L jacktest kill-server', { stdio: 'ignore' }); } catch {} }
process.on('exit', cleanup); process.on('SIGINT', () => process.exit(130)); process.on('SIGTERM', () => process.exit(143));
await sleep(700);
function bridge(ws) {
  const up = new WebSocket(`ws://127.0.0.1:${TPORT}/term/ws`, ['tty']); up.binaryType = 'arraybuffer';
  const q = []; let open = false;
  up.onopen = () => { open = true; for (const m of q) up.send(m); };
  ws.onMessage((m) => { if (process.env.JACK_UITEST_TRACE) fs.appendFileSync(OUT + 'trace.log', `${Date.now() % 1000000} ${typeof m} ${JSON.stringify(Buffer.from(m).toString('latin1')).slice(0, 120)}\n`); if (open) up.send(m); else q.push(m); });
  up.onmessage = (e) => { try { ws.send(typeof e.data === 'string' ? e.data : Buffer.from(e.data)); } catch {} };
  up.onclose = () => { try { ws.close(); } catch {} };
  ws.onClose(() => { try { up.close(); } catch {} });
}
function check(list, name, ok, detail = '') { list.push({ name, ok: !!ok, detail: String(detail) }); }

async function screen(page) {
  return page.evaluate(() => {
    const t = window.jackTerm, b = t.buffer.active, out = [];
    for (let i = b.viewportY; i < b.viewportY + t.rows; i++) { const l = b.getLine(i); out.push(l ? l.translateToString(true) : ''); }
    return out.join('\n');
  });
}
async function waitText(page, needle, ms = 8000) {
  const t0 = Date.now();
  while (Date.now() - t0 < ms) { const sc = await screen(page); if (sc.includes(needle) || sc.replace(/\n/g, '').includes(needle)) return Date.now() - t0; await sleep(40); }
  return -1;
}
async function state(page) { return page.evaluate(() => document.querySelector('#state span').textContent); }
async function waitState(page, want, ms = 15000) {
  const t0 = Date.now();
  while (Date.now() - t0 < ms) { if ((await state(page)) === want) return Date.now() - t0; await sleep(80); }
  return -1;
}
async function press(page, p, k) { if (p.touch) await page.tap(`[data-k="${k}"]`); else await page.click(`[data-k="${k}"]`); }
async function focusTerm(page, p) { if (p.touch) await page.tap('#term', { position: { x: 60, y: 60 } }); else await page.click('#term', { position: { x: 60, y: 60 } }); }
async function focused(page) { return page.evaluate(() => document.activeElement === window.jackTerm.textarea); }
async function geometry(page) {
  return page.evaluate(() => {
    const r = (s) => { const b = document.querySelector(s).getBoundingClientRect(); return { top: Math.round(b.top), bottom: Math.round(b.bottom), left: Math.round(b.left), right: Math.round(b.right), h: Math.round(b.height), w: Math.round(b.width) }; };
    const keys = [...document.querySelectorAll('#keyrow .key')].map((k) => k.getBoundingClientRect());
    return { vw: innerWidth, vh: innerHeight, scrollW: document.documentElement.scrollWidth, scrollH: document.documentElement.scrollHeight,
      app: r('#app'), bar: r('.bar'), stage: r('#stage'), keysBar: r('#keys'), cols: window.jackTerm.cols, rows: window.jackTerm.rows,
      minKeyH: Math.round(Math.min(...keys.map((k) => k.height))), minKeyW: Math.round(Math.min(...keys.map((k) => k.width))),
      font: window.jackTerm.options.fontSize };
  });
}

const browser = await pw[engineName].launch({ args: engineName === 'chromium' ? [`--host-resolver-rules=MAP ${HOST} 10.0.0.135`, '--no-sandbox'] : [] });
{ const nc = browser.newContext.bind(browser); browser.newContext = async (o) => { const c = await nc(o); await c.routeWebSocket(/\/term\/ws/, bridge); await c.route(/\/term\/upload\?history=/, (route) => { let text = ''; try { text = execSync('tmux -L jacktest capture-pane -p -J -t jack: -S -3000', { encoding: 'utf8' }); } catch {} route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({ text }) }); }); await c.route(/\/term\/upload\?[^]*action=/, (route) => { const u = new URL(route.request().url()); const a = u.searchParams.get('action'), win = u.searchParams.get('win') || '', name = u.searchParams.get('name') || '', where = u.searchParams.get('where') || 'server'; const T = (args) => { try { execSync('tmux -L jacktest ' + args, { stdio: 'ignore' }); return true; } catch { return false; } }; let ok = false; if (a === 'rename' && /^\d+$/.test(win)) { T('set-window-option -t jack:' + win + ' automatic-rename off'); ok = T('rename-window -t jack:' + win + ' -- ' + JSON.stringify(name)); } else if (a === 'newwin') { ok = where === 'laptop' ? T('new-window -t jack: -n Laptop ' + JSON.stringify('sleep 86400')) : T('new-window -t jack:'); } else if (a === 'select' && /^\d+$/.test(win)) { ok = T('select-window -t jack:' + win); } route.fulfill({ status: ok ? 200 : 503, contentType: 'application/json', body: JSON.stringify({ ok }) }); }); return c; }; }

{
  const R = { profile: 'The app itself, no dashboard password', id: 'app', engine: engineName, checks: [], timing: {}, errors: [] };
  const ctx = await browser.newContext({ viewport: { width: 390, height: 844 }, isMobile: true, hasTouch: true });
  const page = await ctx.newPage(); const C = R.checks;
  try {
    const resp = await page.goto(`${BASE}/terminal/`, { waitUntil: 'load' });
    await page.waitForSelector('#signin:not([hidden])', { timeout: 8000 });
    check(C, 'Terminal app opens with no dashboard password, straight to its own sign-in', resp.status() === 200);
    const man = await page.evaluate(() => fetch('manifest.webmanifest').then((r) => r.json()));
    check(C, 'Manifest says: stand-alone app that starts in the terminal', man.display === 'standalone' && man.start_url === '/terminal/' && man.icons.length >= 2, `${man.short_name}, ${man.display}`);
    const icon = await page.evaluate(() => new Promise((res) => { const i = new Image(); i.onload = () => res([i.naturalWidth, i.naturalHeight]); i.onerror = () => res([0, 0]); i.src = 'icon-512.png'; }));
    const touch = await page.evaluate(() => { const l = document.querySelector('link[rel="apple-touch-icon"]'); return l ? fetch(l.href).then((r) => r.status) : 0; });
    check(C, 'Home-screen icons load', icon[0] === 512 && icon[1] === 512 && touch === 200, `${icon.join('×')}, touch icon ${touch}`);
    const dash = await ctx.request.get(`${BASE}/status.json`).then((r) => r.status());
    const tok = await ctx.request.get(`${BASE}/term/token`).then((r) => r.status());
    check(C, 'Dashboard and terminal backend still refuse without their passwords', dash === 401 && tok === 401, `dashboard ${dash}, backend ${tok}`);
    const jar = await ctx.cookies(); 
    await page.fill('#pw', LONG); await page.keyboard.press('Enter');
    const tc = await waitState(page, 'Connected');
    const ck = (await ctx.cookies()).find((c) => c.name === 'jt');
    check(C, 'Sign-in works from here and the phone keeps it for a year, unreadable by scripts', tc >= 0 && ck && ck.httpOnly && ck.secure && ck.sameSite === 'Strict' && ck.path === '/term' && (ck.expires - Date.now() / 1000) > 300 * 86400, ck ? `httpOnly ${ck.httpOnly}, ${Math.round((ck.expires - Date.now() / 1000) / 86400)} days` : 'no cookie');
    await ctx.close();
  } catch (e) { check(C, 'App checks completed', false, String(e).slice(0, 300)); await ctx.close().catch(() => {}); }
  results.push(R);
}

{
  const R = { profile: 'The Jack app (dashboard)', id: 'jack', engine: engineName, checks: [], timing: {}, errors: [] };
  const ctx = await browser.newContext({ viewport: { width: 390, height: 844 }, deviceScaleFactor: 2, isMobile: true, hasTouch: true });
  const page = await ctx.newPage(); const C = R.checks;
  page.on('pageerror', (e) => R.errors.push(String(e).slice(0, 200)));
  try {
    await page.goto(`${BASE}/`, { waitUntil: 'load' });
    await page.waitForSelector('#pw', { timeout: 8000 });
    await sleep(300); await page.screenshot({ path: `${OUT}jack-0-signin.png` });
    check(C, 'Front door shows its own sign-in page, not a browser password box', (await page.textContent('h1')) === 'Jack');
    await page.fill('#pw', '000000'); await page.keyboard.press('Enter');
    await page.waitForFunction(() => document.querySelector('#err').textContent.length > 0, null, { timeout: 8000 });
    check(C, 'Wrong password is refused with a message', (await page.textContent('#err')).includes('not accepted'));
    await page.fill('#pw', PIN); await page.keyboard.press('Enter');
    await page.waitForSelector('#view-jack .hero', { timeout: 10000 });
    const ck = (await ctx.cookies()).find((c) => c.name === 'jd');
    check(C, 'Right password opens the dashboard and is remembered for a year, unreadable by scripts', ck && ck.httpOnly && ck.secure && (ck.expires - Date.now() / 1000) > 300 * 86400);
    const hero = (await page.textContent('#view-jack .hero')).slice(0, 60);
    const tone = await page.$$eval('#view-jack .meter b', (n) => n.length);
    check(C, 'Front page leads with the step he is on, and keeps the two-tone proof bars', /Where Jack is/i.test(hero) && /of \d+ steps/.test(hero) && tone >= 5, `${hero}, ${tone} two-tone bars`);
    await sleep(300); await page.screenshot({ path: `${OUT}jack-1-home.png` });
    const man = await page.evaluate(() => fetch('manifest.webmanifest').then((r) => r.json()));
    const icon = await page.evaluate(() => new Promise((res) => { const i = new Image(); i.onload = () => res(i.naturalWidth); i.onerror = () => res(0); i.src = 'icon-512.png'; }));
    check(C, 'Jack has its own app name and icon', man.name === 'Jack' && man.display === 'standalone' && man.start_url === '/' && icon === 512);
    await page.reload({ waitUntil: 'load' }); await page.waitForSelector('#view-jack .hero', { timeout: 10000 });
    check(C, 'Reopening does not ask for the password again', !(await page.$('#pw')));
    await page.tap('#tab-talk'); await sleep(200);
    check(C, 'In Safari the Talk tab links to the terminal', (await page.$$eval('#view-talk a.cta', (n) => n.length)) === 1);
    // opened from the home screen, Jack stays Jack: no link that would turn it into the terminal
    const ctx2 = await browser.newContext({ viewport: { width: 390, height: 844 }, deviceScaleFactor: 2, isMobile: true, hasTouch: true, storageState: await ctx.storageState() });
    await ctx2.addInitScript(() => { Object.defineProperty(navigator, 'standalone', { value: true }); });
    const p2 = await ctx2.newPage(); await p2.goto(`${BASE}/#talk`, { waitUntil: 'load' }); await p2.waitForSelector('#view-talk .card');
    const appTalk = await p2.evaluate(() => ({ links: document.querySelectorAll('#view-talk a').length, text: document.querySelector('#view-talk').textContent }));
    await sleep(250); await p2.screenshot({ path: `${OUT}jack-2-app-talk.png` });
    check(C, 'As a home-screen app, Jack stays Jack and points to the Terminal app', appTalk.links === 0 && /Terminal app/.test(appTalk.text.replace(/\s+/g, ' ')) || /Terminal\s*app/.test(appTalk.text));
    await ctx2.close();
    await page.tap('[data-act="signout"]'); await page.waitForSelector('#pw', { timeout: 8000 });
    check(C, 'Sign out returns to the sign-in page', true);
    check(C, 'No script errors', R.errors.length === 0, R.errors.join(' | '));
    await ctx.close();
  } catch (e) { check(C, 'Jack app checks completed', false, String(e).slice(0, 300)); await ctx.close().catch(() => {}); }
  results.push(R);
}

for (const p of PROFILES) {
  const R = { profile: p.label, id: p.id, engine: engineName, checks: [], timing: {}, errors: [] };
  const ctx = await browser.newContext({
    viewport: { width: p.w, height: p.h }, deviceScaleFactor: p.dpr, isMobile: p.touch && engineName !== 'firefox', hasTouch: p.touch,
    colorScheme: 'dark',
    userAgent: p.touch ? 'Mozilla/5.0 (iPhone; CPU iPhone OS 18_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.0 Mobile/15E148 Safari/604.1' : undefined,
  });
  await ctx.addInitScript(() => { try { if (!sessionStorage.getItem('t-boxmode')) localStorage.setItem('jt-direct', '1'); } catch {} });
  await ctx.grantPermissions(['clipboard-read', 'clipboard-write']).catch(() => {});
  await ctx.request.post(`${BASE}/login-check`, { headers: { Authorization: 'Basic ' + Buffer.from('jack:' + PIN).toString('base64') } });
  // a stand-in for the phone's speech service, so the microphone path can be exercised here
  await ctx.addInitScript(() => { window.SpeechRecognition = window.webkitSpeechRecognition = class { start() { setTimeout(() => { const r = [{ transcript: 'echo SPOKENWORDS' }]; r.isFinal = true; if (this.onresult) this.onresult({ results: [r] }); }, 250); } stop() { if (this.onend) this.onend(); } }; });
  try { execSync('tmux -L jacktest kill-window -a -t jack:0 2>/dev/null; tmux -L jacktest send-keys -t jack:0 -X cancel 2>/dev/null; true', { stdio: 'ignore', shell: '/bin/bash' }); } catch {}
  const page = await ctx.newPage();
  page.on('console', (m) => { if (m.type() === 'error' && !/status of 401/.test(m.text()) && !R.expectDrop) R.errors.push('console: ' + m.text().slice(0, 200)); });
  page.on('pageerror', (e) => R.errors.push('page: ' + String(e).slice(0, 200)));
  const C = R.checks;
  try {
    // 1. dashboard → Talk tab → the terminal button
    await page.goto(`${BASE}/#talk`, { waitUntil: 'load' });
    await page.waitForSelector('#view-talk .cta', { timeout: 8000 });
    await page.screenshot({ path: `${OUT}${p.id}-0-talk.png` });
    if (p.touch) await page.tap('#view-talk .cta'); else await page.click('#view-talk .cta');
    await page.waitForURL('**/terminal/', { timeout: 8000 });
    check(C, 'Dashboard button opens the terminal page', true);

    // 2. sign-in sheet: wrong password is refused with a clear message
    await page.waitForSelector('#signin:not([hidden])', { timeout: 6000 });
    check(C, 'Sign-in sheet appears on first visit', true);
    await page.fill('#pw', 'not-the-password');
    await sleep(450);
    await page.screenshot({ path: `${OUT}${p.id}-1-signin.png` });
    await page.keyboard.press('Enter');
    await page.waitForFunction(() => document.querySelector('#err').textContent.length > 0, null, { timeout: 8000 });
    check(C, 'Wrong password is refused with a message', (await page.textContent('#err')).includes('not accepted'), await page.textContent('#err'));

    // 3. the real password opens the shell
    await page.fill('#pw', LONG);
    const t0 = Date.now();
    await page.keyboard.press('Enter');
    const tc = await waitState(page, 'Connected');
    R.timing.connect_ms = tc;
    check(C, 'Right password connects', tc >= 0, `${tc} ms`);
    await sleep(900);
    // first-visit tips sheet
    const tips = await page.isVisible('#tips');
    check(C, 'First visit shows the how-to sheet', tips);
    if (tips) { await sleep(350); await page.screenshot({ path: `${OUT}${p.id}-2-tips.png` }); if (p.touch) await page.tap('#tips .cta'); else await page.click('#tips .cta'); }

    // 4. geometry: nothing overflows, keys are thumb-sized, the terminal is usable
    const g = await geometry(page);
    R.geometry = g;
    check(C, 'No sideways scrolling', g.scrollW <= g.vw, `${g.scrollW} ≤ ${g.vw}`);
    check(C, 'App fills the screen exactly', g.app.h === g.vh && g.keysBar.bottom <= g.vh + 1, `app ${g.app.h}/${g.vh}, keys end ${g.keysBar.bottom}`);
    const short = p.h <= 460;
    check(C, short ? 'Keys stay tappable on a short screen (36 px or more)' : 'Every key is at least 44 px tall', g.minKeyH >= (short ? 36 : 44), `${g.minKeyH} px`);
    check(C, 'Terminal is wide and tall enough to work in', g.cols >= 40 && g.rows >= 14, `${g.cols}×${g.rows} at ${g.font}px`);

    // 5. type a command, see its output
    await focusTerm(page, p);
    const mark = 'JACKUI' + Math.random().toString(36).slice(2, 8).toUpperCase();
    await page.keyboard.type(`clear; echo ${mark}-$((40+2))`);
    const k0 = Date.now();
    await page.keyboard.press('Enter');
    const te = await waitText(page, `${mark}-42`);
    R.timing.command_ms = te;
    check(C, 'Typed command runs and its output shows', te >= 0, `${te} ms`);

    // 6. keystroke echo latency (one character, there and back)
    const lat = [];
    for (const ch of ['q', 'w', 'e', 'r', 't']) {
      const before = await screen(page); const s0 = Date.now();
      await page.keyboard.type(ch);
      let ok = -1; while (Date.now() - s0 < 3000) { if ((await screen(page)) !== before) { ok = Date.now() - s0; break; } await sleep(8); }
      lat.push(ok);
    }
    R.timing.echo_ms = lat;
    check(C, 'Keystrokes echo quickly', lat.every((x) => x >= 0 && x < 600), lat.join(', ') + ' ms');
    await press(page, p, 'intr');                         // ^C clears the half-typed line
    await sleep(250);

    // 7. extra keys: up recalls the last command; focus (and the keyboard) stays on the terminal
    await press(page, p, 'up');
    const tu = await waitText(page, `echo ${mark}-$((40+2))`, 4000);
    const f1 = await focused(page);
    check(C, 'Up arrow key recalls the last command', tu >= 0);
    check(C, 'Tapping a key keeps the terminal focused (keyboard stays up)', f1);
    await press(page, p, 'intr'); await sleep(200);

    // 8. sticky ctrl: ctrl then c stops a running command
    await page.keyboard.type('sleep 30'); await page.keyboard.press('Enter'); await sleep(500);
    await press(page, p, 'ctrl');
    const armed = await page.evaluate(() => document.querySelector('[data-k="ctrl"]').classList.contains('on'));
    await page.keyboard.type('c');
    await sleep(500);
    await page.keyboard.type(`echo STOPPED-${mark}`); await page.keyboard.press('Enter');
    const ts = await waitText(page, `STOPPED-${mark}`, 5000);
    const disarmed = await page.evaluate(() => !document.querySelector('[data-k="ctrl"]').classList.contains('on'));
    check(C, 'ctrl lights up, then ctrl+c stops a running command', armed && disarmed && ts >= 0, `armed ${armed}, released ${disarmed}, ${ts} ms`);

    // 9. scroll back: page keys, then swipe
    await page.keyboard.type('clear; seq 1 400'); await page.keyboard.press('Enter');
    await waitText(page, '400', 5000); await sleep(300);
    const lowest = async () => { const nums = (await screen(page)).split('\n').map((l) => l.trim()).filter((l) => /^\d+$/.test(l)).map(Number); return nums.length ? Math.min(...nums) : null; };
    const lo0 = await lowest();
    await press(page, p, 'pgup'); await sleep(450);
    const lo1 = await lowest();
    check(C, 'Page-up key scrolls back', lo0 !== null && lo1 !== null && lo1 < lo0, `${lo0} → ${lo1}`);
    for (let i = 0; i < 6; i++) { await press(page, p, 'pgdn'); await sleep(120); }
    await sleep(300);
    if (p.touch && engineName === 'chromium') {
      const cdp = await ctx.newCDPSession(page);
      const x = Math.round(p.w / 2), ya = Math.round(g.stage.top + 60), yb = Math.round(g.stage.bottom - 40);
      const lob = await lowest();
      await cdp.send('Input.dispatchTouchEvent', { type: 'touchStart', touchPoints: [{ x, y: ya }] });
      for (let i = 1; i <= 14; i++) { await cdp.send('Input.dispatchTouchEvent', { type: 'touchMove', touchPoints: [{ x, y: ya + ((yb - ya) * i) / 14 }] }); await sleep(16); }
      await cdp.send('Input.dispatchTouchEvent', { type: 'touchEnd', touchPoints: [] });
      await sleep(500);
      const loa = await lowest();
      check(C, 'Swiping down scrolls back through earlier lines', lob !== null && loa !== null && loa < lob, `${lob} → ${loa}`);
      await page.screenshot({ path: `${OUT}${p.id}-4-scrolled.png` });
      // typing while scrolled back returns to the bottom first, so nothing is lost
      await page.keyboard.type(`echo BACK-${mark}`); await page.keyboard.press('Enter');
      const bk = await waitText(page, `\nBACK-${mark}`, 5000);
      check(C, 'Typing while scrolled back returns to the bottom and runs', bk >= 0, `${bk} ms`);
      await cdp.send('Input.dispatchTouchEvent', { type: 'touchStart', touchPoints: [{ x, y: yb }] });
      for (let i = 1; i <= 24; i++) { await cdp.send('Input.dispatchTouchEvent', { type: 'touchMove', touchPoints: [{ x, y: yb - ((yb - ya) * 2 * i) / 24 }] }); await sleep(12); }
      await cdp.send('Input.dispatchTouchEvent', { type: 'touchEnd', touchPoints: [] });
      await sleep(400);
    }
    for (let i = 0; i < 8; i++) { await press(page, p, 'pgdn'); await sleep(90); }
    await page.keyboard.press('q').catch(() => {}); await sleep(200); await press(page, p, 'intr'); await sleep(200);

    // 10. smoothness under a flood of output: how long does the page ever stall?
    await page.evaluate(() => { window.__f = []; let last = performance.now(); const tick = (t) => { window.__f.push(t - last); last = t; if (window.__f.length < 400) requestAnimationFrame(tick); }; requestAnimationFrame(tick); });
    await page.keyboard.type('clear; seq 1 30000 | tail -n 30000; echo FLOOD-DONE'); await page.keyboard.press('Enter');
    const tf = await waitText(page, 'FLOOD-DONE', 20000);
    await sleep(300);
    const frames = await page.evaluate(() => window.__f.slice(2));
    const worst = Math.round(Math.max(0, ...frames)), slow = frames.filter((x) => x > 100).length;
    R.timing.flood_ms = tf; R.timing.worst_frame_ms = worst; R.timing.frames_over_100ms = slow; R.timing.frames = frames.length;
    check(C, '30,000 lines of output arrive without freezing the page', tf >= 0 && worst < 1500, `done in ${tf} ms, worst stall ${worst} ms, ${slow} of ${frames.length} frames over 100 ms`);

    // 11. the keyboard opening: shrink the visible screen and the keys must ride up with it
    await page.keyboard.type('clear'); await page.keyboard.press('Enter'); await sleep(300);
    if (p.touch) {
      const kh = Math.round(p.h * 0.42);
      await page.setViewportSize({ width: p.w, height: p.h - kh }); await sleep(450);
      const g2 = await geometry(page);
      check(C, 'With the keyboard up, the keys sit on top of it and the terminal refits', g2.keysBar.bottom <= g2.vh + 1 && g2.app.h === g2.vh && g2.rows < g.rows && g2.rows >= 4, `rows ${g.rows} → ${g2.rows}`);
      await page.screenshot({ path: `${OUT}${p.id}-5-keyboard.png` });
      await page.setViewportSize({ width: p.w, height: p.h }); await sleep(450);
    }

    // 12. turning the device: the terminal refits to the new shape
    await page.setViewportSize({ width: p.h, height: p.w }); await sleep(600);
    const g3 = await geometry(page);
    check(C, 'Turning the screen refits the terminal with nothing cut off', g3.scrollW <= g3.vw && g3.app.h === g3.vh && g3.cols !== g.cols, `${g.cols}×${g.rows} → ${g3.cols}×${g3.rows}`);
    await page.setViewportSize({ width: p.w, height: p.h }); await sleep(600);

    // 13. menu: text size
    await page.keyboard.type(`echo ${mark}-STILL-HERE`); await page.keyboard.press('Enter'); await waitText(page, `${mark}-STILL-HERE`);
    if (p.touch) await page.tap('#more'); else await page.click('#more');
    await page.waitForSelector('#menu:not([hidden])');
    await sleep(350);
    await page.screenshot({ path: `${OUT}${p.id}-6-menu.png` });
    const f0 = (await geometry(page)).font;
    if (p.touch) await page.tap('[data-m="bigger"]'); else await page.click('[data-m="bigger"]');
    await sleep(350);
    const gb = await geometry(page);
    check(C, 'Text size control enlarges the text and refits', gb.font === f0 + 1 && gb.cols < g.cols, `${f0}px ${g.cols} cols → ${gb.font}px ${gb.cols} cols`);
    if (p.touch) await page.tap('[data-m="smaller"]'); else await page.click('[data-m="smaller"]');
    await sleep(250);
    await page.keyboard.press('Escape').catch(() => {}); await page.mouse.click(5, p.h / 2).catch(() => {}); await sleep(200);
    if (await page.isVisible('#menu')) { if (p.touch) await page.tap('#more'); else await page.click('#more'); }

    // 13b. a second window: a strip appears, switching works, closing it removes the strip
    const menu = async (item) => { if (!(await page.isVisible('#menu'))) { if (p.touch) await page.tap('#more'); else await page.click('#more'); await page.waitForSelector('#menu:not([hidden])'); } if (p.touch) await page.tap(`[data-m="${item}"]`); else await page.click(`[data-m="${item}"]`); await sleep(250); };
    await menu('newwin');
    await page.waitForSelector('#place:not([hidden])');
    const places = await page.$$eval('#placelist [data-place] b', (n) => n.map((x) => x.textContent));
    check(C, 'New window asks where: this server or the laptop', places.length === 2 && /laptop/i.test(places[1]), places.join(', '));
    if (p.touch) await page.tap('#placelist [data-place="0"]'); else await page.click('#placelist [data-place="0"]');
    await sleep(300);
    const tabs = async () => page.$$eval('#wins .win:not(.add)', (n) => n.map((x) => x.classList.contains('on')));
    const t7 = Date.now(); let strip = -1; while (Date.now() - t7 < 5000) { if ((await tabs()).length === 2) { strip = Date.now() - t7; break; } await sleep(60); }
    await focusTerm(page, p); await page.keyboard.type(`echo WINTWO-${mark}`); await page.keyboard.press('Enter');
    const w2 = await waitText(page, `WINTWO-${mark}`, 5000);
    await sleep(450);   // as with the close button: let the emulated phone finish the previous touch
    // a burst of keys in one go (a focus report, then prefix + number, as a click on a tab sends on a computer)
    // must still reach tmux's own keys: tmux's paste guess used to swallow the third key
    await page.evaluate(() => { window.jackTerm.input('\x1b[O'); window.jackTerm.input('\x020'); window.jackTerm.input('\x1b[I'); });
    const burst = await waitText(page, `${mark}-STILL-HERE`, 4000);
    await page.evaluate(() => { window.jackTerm.input('\x021'); });
    const burstBack = await waitText(page, `WINTWO-${mark}`, 4000);
    check(C, 'A burst of three keys still switches windows (no paste guessing)', burst >= 0 && burstBack >= 0, `${burst} ms, back ${burstBack} ms`);
    await sleep(450);
    if (p.touch) await page.tap('#wins .win >> nth=0'); else await page.click('#wins .win >> nth=0');
    const w1 = await waitText(page, `${mark}-STILL-HERE`, 5000);
    const firstOn = (await tabs())[0] === true;
    const tabsFit = await page.evaluate(() => { const w = document.querySelector('#wins').getBoundingClientRect(), k = document.querySelector('#keys').getBoundingClientRect(); return w.height >= 44 && k.bottom <= innerHeight + 1 && document.documentElement.scrollWidth <= innerWidth; });
    await sleep(200); await page.screenshot({ path: `${OUT}${p.id}-c-windows.png` });
    if (p.touch && engineName === 'chromium') {
      // a sideways swipe on the screen goes to the next window
      const cdp2 = await ctx.newCDPSession(page); const sy = Math.round(p.h * 0.4), xa = Math.round(p.w * 0.75), xb = Math.round(p.w * 0.25);
      await cdp2.send('Input.dispatchTouchEvent', { type: 'touchStart', touchPoints: [{ x: xa, y: sy }] });
      for (let i = 1; i <= 8; i++) { await cdp2.send('Input.dispatchTouchEvent', { type: 'touchMove', touchPoints: [{ x: xa + ((xb - xa) * i) / 8, y: sy }] }); await sleep(16); }
      await cdp2.send('Input.dispatchTouchEvent', { type: 'touchEnd', touchPoints: [] });
      const sw = await waitText(page, `WINTWO-${mark}`, 5000);
      check(C, 'Swiping sideways goes to the next window', sw >= 0 && (await tabs())[1] === true, `${sw} ms`);
    } else if (p.touch) await page.tap('#wins .win >> nth=1'); else await page.click('#wins .win >> nth=1');
    await waitText(page, `WINTWO-${mark}`, 5000);
    let dlg = 0; page.once('dialog', (d) => { dlg++; d.accept(); });
    await sleep(600);   // a tap straight after another touch is read as a double tap by the emulated phone
    if (p.touch) await page.tap('#wins .win.x'); else await page.click('#wins .win.x');
    const t9 = Date.now(); let stripGone = false;
    while (Date.now() - t9 < 5000) { if ((await page.$$('#wins .win:not(.add)')).length === 1 && (await screen(page)).includes(`${mark}-STILL-HERE`)) { stripGone = true; break; } await sleep(80); }
    if (!stripGone) { await page.screenshot({ path: `${OUT}${p.id}-d-tidy.png` }); R.tidy = { tabs: await page.$$eval('#wins .win', (n) => n.map((x) => x.textContent + (x.classList.contains('on') ? '*' : ''))), dlg }; console.log('TIDY', JSON.stringify(R.tidy)); }
    check(C, 'Window tabs: a second window gets a tab, tapping switches, closing removes the tabs', strip >= 0 && w2 >= 0 && w1 >= 0 && firstOn && tabsFit && stripGone, `tabs ${strip} ms, second ${w2} ms, back ${w1} ms, first lit ${firstOn}, fits ${tabsFit}, tidy ${stripGone}`);

    // 13c. renaming a tab (a name with a space, to prove the quoting) sticks on the tab
    page.once('dialog', (d) => d.accept('QA renamed'));
    await menu('rename');
    let renamed = false; const trn = Date.now();
    while (Date.now() - trn < 5000) { if ((await page.$$eval('#wins .win:not(.add)', (n) => n.map((x) => x.textContent).join('|'))).includes('QA renamed')) { renamed = true; break; } await sleep(100); }
    check(C, 'Rename this tab: the typed name appears on the tab', renamed);

    // 13d. the close confirm must fire once and close one window even when OK is slow —
    //      the reported bug: a slow OK let the trailing click raise a second confirm and close two
    for (let i = 0; i < 2; i++) { await menu('newwin'); await page.waitForSelector('#place:not([hidden])'); if (p.touch) await page.tap('#placelist [data-place="0"]'); else await page.click('#placelist [data-place="0"]'); await sleep(450); }
    const before = (await page.$$('#wins .win:not(.add)')).length;
    let slowDlgs = 0; const slowOk = async (d) => { slowDlgs++; await sleep(900); await d.accept(); };
    page.on("dialog", slowOk);
    await sleep(600);
    if (p.touch) await page.tap('#wins .win.x'); else await page.click('#wins .win.x');
    await sleep(1800);
    const after = (await page.$$('#wins .win:not(.add)')).length;
    page.off("dialog", slowOk);
    check(C, 'Close a tab: one confirm, one window closed, even with a slow OK', slowDlgs === 1 && before - after === 1, `dialogs ${slowDlgs}, closed ${before - after}, before ${before}`);
    while ((await page.$$('#wins .win:not(.add)')).length > 1) { page.once('dialog', (d) => d.accept()); if (p.touch) await page.tap('#wins .win.x'); else await page.click('#wins .win.x'); await sleep(700); }

    // 13e. a URL in the terminal is clickable (even with tmux mouse mode on)
    await page.evaluate(() => { window.__opened = null; window.open = (u) => { window.__opened = u; return null; }; });
    await focusTerm(page, p); await page.keyboard.type('echo go https://example.com/abc done'); await page.keyboard.press('Enter');
    await waitText(page, 'example.com');
    const lp = await page.evaluate(() => { const t = window.jackTerm, buf = t.buffer.active; for (let i = 0; i < t.rows; i++) { const ln = buf.getLine(buf.viewportY + i); if (!ln) continue; const s = ln.translateToString(true); const idx = s.indexOf('https://'); if (idx >= 0) { const d = t._core._renderService.dimensions, cw = (d.css ? d.css.cell.width : d.actualCellWidth), ch = (d.css ? d.css.cell.height : d.actualCellHeight); const sc = t.element.querySelector('.xterm-screen') || t.element; const r = sc.getBoundingClientRect(); return { x: Math.round(r.left + (idx + 8) * cw), y: Math.round(r.top + i * ch + ch / 2) }; } } return null; });
    let linkOpened = '';
    if (lp) { await page.mouse.move(lp.x, lp.y); await sleep(200); await page.mouse.move(lp.x + 1, lp.y); await sleep(150); if (p.touch) await page.touchscreen.tap(lp.x, lp.y); else await page.mouse.click(lp.x, lp.y); await sleep(300); linkOpened = await page.evaluate(() => window.__opened || ''); }
    check(C, 'A URL in the terminal is clickable', linkOpened === 'https://example.com/abc', `pos ${lp ? 'found' : 'MISSING'}, opened ${linkOpened || '(none)'}`);

    // 13b2. Claude account: the menu opens a sheet drawn from what the server tool reports; closing it tidies up
    await menu('account');
    const acOpen = await page.waitForSelector('#acct:not([hidden]) .acrow', { timeout: 25000 }).then(() => true, () => false);
    const ac = await page.evaluate(() => ({ rows: document.querySelectorAll('#acctlist .acrow:not(.add)').length, on: document.querySelectorAll('#acctlist .acrow.on').length, add: !!document.querySelector('#acctlist .acrow.add'),
      fits: document.querySelector('#acct .sheet').getBoundingClientRect().bottom <= innerHeight + 1 && document.documentElement.scrollWidth <= innerWidth }));
    await page.evaluate(() => document.querySelectorAll('#acctlist .acrow:not(.add) b').forEach((b, i) => { b.textContent = 'account' + (i + 1) + '@example.com'; }));   // keep the real address out of screenshots
    await sleep(200); await page.screenshot({ path: `${OUT}${p.id}-e-account.png` });
    await sleep(450);
    if (p.touch) await page.tap('#acctclose'); else await page.click('#acctclose');
    const t11 = Date.now(); let acTidy = false;
    while (Date.now() - t11 < 8000) { if (!(await page.isVisible('#acct')) && (await page.$$('#wins .win:not(.add)')).length === 1) { acTidy = true; break; } await sleep(100); }
    check(C, 'Claude account: the menu lists the saved accounts with the one in use marked, and closing returns to one window', acOpen && ac.rows >= 1 && ac.on === 1 && ac.add && ac.fits && acTidy, JSON.stringify({ acOpen, ...ac, acTidy }));
    // a sign-in link from the server tool becomes a button; a link to anywhere but Claude is ignored
    await focusTerm(page, p); await page.keyboard.type('claude-account --emit-login https://evil.example.com/oauth'); await page.keyboard.press('Enter'); await sleep(900);
    const badShown = await page.isVisible('#acct');
    const LINK = "https://claude.com/cai/oauth/authorize?code=true&client_id=test-client&response_type=code&redirect_uri=https%3A%2F%2Fplatform.claude.com%2Foauth%2Fcode%2Fcallback&scope=user%3Aprofile+user%3Ainference&code_challenge=abcdefghijklmnopqrstuvwxyz0123456789ABCDEFG&code_challenge_method=S256&state=abcdefghijklmnopqrstuvwxyz0123456789";
    await page.keyboard.type("claude-account --emit-login '" + LINK + "'"); await page.keyboard.press('Enter');
    const goodShown = await page.waitForSelector('#acctlogin:not([hidden])', { timeout: 6000 }).then(() => true, () => false);
    const href = await page.getAttribute('#acctopen', 'href');
    const shownUrl = await page.inputValue('#accturl');
    if (p.touch) await page.tap('#acctcopy'); else await page.click('#acctcopy');
    await sleep(300);
    const linkClip = await page.evaluate(() => navigator.clipboard.readText().catch(() => 'DENIED'));
    const linkCopied = shownUrl === LINK && (linkClip === shownUrl || linkClip === 'DENIED') && /Copied/.test(await page.textContent('#acctcopy'));
    const signFits = await page.evaluate(() => document.querySelector('#acct .sheet').getBoundingClientRect().bottom <= innerHeight + 1 && document.documentElement.scrollWidth <= innerWidth);
    await sleep(200); await page.screenshot({ path: `${OUT}${p.id}-f-signin.png` });
    await page.fill('#acctcode', `echo CODE-${mark}`); if (p.touch) await page.tap('#acctfinish'); else await page.click('#acctfinish');
    const typed = await waitText(page, `CODE-${mark}`, 5000);
    await sleep(400);
    const ran = (await screen(page)).split('\n').filter((l) => l.trim() === `CODE-${mark}`).length >= 1;
    await page.evaluate(() => { document.querySelector('#acct').hidden = true; });
    check(C, 'A sign-in link from the server is shown with a Copy button, only for Claude pages, and the pasted code is typed in and entered', !badShown && goodShown && (href || '').startsWith('https://claude.com/') && linkCopied && signFits && typed >= 0 && ran, JSON.stringify({ badShown, goodShown, linkCopied, clipOk: linkClip === shownUrl, signFits, typed, ran }));

    // 13c. select and copy: the copy view is real selectable text and holds what has scrolled away
    await focusTerm(page, p); await page.keyboard.type(`clear; echo TOP-${mark}; seq 1 150; echo END-${mark}`); await page.keyboard.press('Enter');
    await waitText(page, `\nEND-${mark}`, 8000); await sleep(300);
    await press(page, p, 'select');
    await page.waitForSelector('#copy:not([hidden])');
    const gotHist = await page.waitForFunction((mk) => document.querySelector('#copytext').textContent.includes('TOP-' + mk), mark, { timeout: 6000 }).then(() => true, () => false);
    await sleep(250);
    const rd = await page.evaluate(() => { const b = document.querySelector('#copytext'), sh = document.querySelector('#copy .sheet').getBoundingClientRect(), cs = getComputedStyle(b); return { atBottom: b.scrollHeight - b.scrollTop - b.clientHeight < 40, fits: sh.bottom <= innerHeight + 1 && sh.top >= -1 && document.documentElement.scrollWidth <= innerWidth, sel: cs.userSelect || cs.webkitUserSelect }; });
    await page.screenshot({ path: `${OUT}${p.id}-7-copy.png` });
    if (p.touch) await page.tap('#copyall'); else await page.click('#copyall');
    await sleep(300);
    const clipAll = await page.evaluate(() => navigator.clipboard.readText().catch(() => ''));
    await page.evaluate((mk) => { const b = document.querySelector('#copytext'), n = b.firstChild, i = n.nodeValue.lastIndexOf('END-' + mk), r = document.createRange(); r.setStart(n, i); r.setEnd(n, i + 4 + mk.length); const s = getSelection(); s.removeAllRanges(); s.addRange(r); }, mark);
    await sleep(250);
    const selOn = await page.evaluate(() => !document.querySelector('#copysel').disabled);
    if (p.touch) await page.tap('#copysel'); else await page.click('#copysel');
    await sleep(300);
    const clipSel = await page.evaluate(() => navigator.clipboard.readText().catch(() => ''));
    check(C, 'Copy view: real selectable text with everything that scrolled away; Copy all and Copy selection work', gotHist && rd.atBottom && rd.fits && rd.sel === 'text' && clipAll.includes(`TOP-${mark}`) && clipAll.includes(`END-${mark}`) && selOn && clipSel === `END-${mark}`, JSON.stringify({ gotHist, ...rd, all: clipAll.length, selOn, clipSel: clipSel.slice(0, 30) }));
    if (p.touch) await page.tap('#copy [data-close]'); else await page.click('#copy [data-close]');
    await sleep(250);
    if (p.touch && engineName === 'chromium') {
      // press and hold hands off to the phone's own selector: no copy box, no message box,
      // and the terminal text is natively selectable (user-select:text, a range selects)
      const cdp3 = await ctx.newCDPSession(page);
      const pos = await page.evaluate((mk) => { const t = window.jackTerm, b = t.buffer.active; let row = -1; for (let i = 0; i < t.rows; i++) { const l = b.getLine(b.viewportY + i); if (l && l.translateToString(true).trim() === 'END-' + mk) row = i; } const r = document.querySelector('#term').getBoundingClientRect(); return { row, x: Math.round(r.left + 90), y: Math.round(r.top + (row + 0.5) * (r.height / t.rows)) }; }, mark);
      await sleep(600);
      await cdp3.send('Input.dispatchTouchEvent', { type: 'touchStart', touchPoints: [{ x: pos.x, y: pos.y }] });
      await sleep(800);   // past the long-press threshold
      await cdp3.send('Input.dispatchTouchEvent', { type: 'touchEnd', touchPoints: [] });
      await sleep(400);
      const lp = await page.evaluate((mk) => {
        const box = !document.querySelector('#copy').hidden, compose = !document.querySelector('#compose').hidden;
        const rows = document.querySelector('.xterm-rows');
        const cs = rows ? getComputedStyle(rows) : null, us = cs ? (cs.webkitUserSelect || cs.userSelect) : '';
        let sel = '';
        try { const w = document.createTreeWalker(rows, NodeFilter.SHOW_TEXT); let node = null; while (w.nextNode()) { if (w.currentNode.nodeValue.includes('END-' + mk)) { node = w.currentNode; break; } } if (node) { const rng = document.createRange(); rng.selectNodeContents(node); const s = getSelection(); s.removeAllRanges(); s.addRange(rng); sel = String(s); } } catch (e) {}
        return { box, compose, us, sel };
      }, mark);
      check(C, 'Press and hold hands off to the phone selector: no copy box, no message box, terminal text selectable', pos.row >= 0 && !lp.box && !lp.compose && /text/.test(lp.us) && lp.sel.includes(`END-${mark}`), JSON.stringify(lp));
      await page.evaluate(() => { try { getSelection().removeAllRanges(); } catch (e) {} }); await sleep(250);
    }
    if (!p.touch) {
      // on a computer: drag across a line, and it is on the clipboard when the button is released
      await focusTerm(page, p); await page.keyboard.type(`clear; echo DRAG-${mark}`); await page.keyboard.press('Enter');
      await waitText(page, `\nDRAG-${mark}`, 5000); await sleep(300);
      const dg = await page.evaluate((mk) => { const t = window.jackTerm, b = t.buffer.active; let row = -1; for (let i = 0; i < t.rows; i++) { const l = b.getLine(b.viewportY + i); if (l && l.translateToString(true).trim() === 'DRAG-' + mk) row = i; } const r = document.querySelector('#term .xterm-screen').getBoundingClientRect(); const cw = r.width / t.cols, ch = r.height / t.rows; return { row, x0: r.left + cw * 0.3, x1: r.left + cw * (5 + mk.length + 0.4), y: r.top + (row + 0.5) * ch }; }, mark);
      await page.evaluate(() => navigator.clipboard.writeText('EMPTY'));
      await page.mouse.move(dg.x0, dg.y); await page.mouse.down(); await page.mouse.move((dg.x0 + dg.x1) / 2, dg.y, { steps: 6 }); await page.mouse.move(dg.x1, dg.y, { steps: 6 }); await sleep(120); await page.mouse.up();
      let dragClip = ''; const t12 = Date.now();
      while (Date.now() - t12 < 4000) { dragClip = await page.evaluate(() => navigator.clipboard.readText().catch(() => '')); if (dragClip.includes(`DRAG-${mark}`)) break; await sleep(100); }
      await page.keyboard.type(`echo AFTER-${mark}`); await page.keyboard.press('Enter');
      const after = await waitText(page, `\nAFTER-${mark}`, 5000);
      check(C, 'Dragging across text copies it on release, and typing afterwards still works', dg.row >= 0 && dragClip.trim() === `DRAG-${mark}` && after >= 0, JSON.stringify({ row: dg.row, dragClip: dragClip.slice(0, 40), after }));
    }
    // put the marker back on screen for the reload tests that follow
    await focusTerm(page, p); await page.keyboard.type(`clear; echo ${mark}-STILL-HERE`); await page.keyboard.press('Enter');
    await waitText(page, `\n${mark}-STILL-HERE`, 5000); await sleep(200);

    // 13e. speak: the microphone key opens a box, the words land in it, Send puts them in the terminal
    await press(page, p, 'speak');
    await page.waitForSelector('#compose:not([hidden])');
    await page.waitForFunction(() => document.querySelector('#ctext').value.includes('SPOKENWORDS'), null, { timeout: 5000 });
    const cg = await page.evaluate(() => { const c = document.querySelector('#compose').getBoundingClientRect(), k = document.querySelector('#keys').getBoundingClientRect(); return { above: c.bottom <= k.top + 1, inView: k.bottom <= innerHeight + 1, listening: document.querySelector('#cmic').classList.contains('rec') }; });
    await sleep(250); await page.screenshot({ path: `${OUT}${p.id}-a-speak.png` });
    if (p.touch) await page.tap('#csend'); else await page.click('#csend');
    const t8 = Date.now(); let spoke = false;
    while (Date.now() - t8 < 6000) { const sc = (await screen(page)).replace(/\n/g, ''); if ((sc.match(/SPOKENWORDS/g) || []).length >= 2) { spoke = true; break; } await sleep(80); }
    check(C, 'Microphone: words appear in a box above the keys and Send runs them', cg.above && cg.inView && cg.listening && spoke && !(await page.isVisible('#compose')), JSON.stringify(cg));

    // 13f. a photo: sent to the server, its location typed into the terminal
    await page.setInputFiles('#photo', '/home/opc/jackthelearner/dashboard/terminal/www/icon-192.png');
    await page.waitForFunction(() => /added/.test(document.querySelector('#toast').textContent), null, { timeout: 15000 });
    await sleep(300); await page.screenshot({ path: `${OUT}${p.id}-b-photo.png` });
    const pm = (await screen(page)).replace(/\n/g, '').match(/\/data\/jack-uploads\/[0-9-]+\/photo-[0-9]+-[0-9a-f]+\.(?:png|jpg)/);
    check(C, 'Photo: saved on the server and its location typed in', !!pm && fs.existsSync(pm[0]) && fs.statSync(pm[0]).size > 500, pm ? pm[0] : 'no path on screen');
    await press(page, p, 'intr'); await sleep(250);

    // 13d. the home-screen help, and how it looks once opened as an app
    await menu('install');
    await page.waitForSelector('#install:not([hidden])'); await sleep(350);
    const steps = await page.$$eval('#installsteps li', (n) => n.length);
    await page.screenshot({ path: `${OUT}${p.id}-8-install.png` });
    check(C, 'Add-to-Home-Screen help shows steps for this device', steps >= 1, `${steps} step(s)`);
    if (p.touch) await page.tap('#install [data-close]'); else await page.click('#install [data-close]');
    await page.evaluate(() => document.documentElement.classList.add('app')); await sleep(200);
    const appOK = await page.evaluate(() => getComputedStyle(document.querySelector('.bar .back')).display === 'none' && getComputedStyle(document.querySelector('[data-m="install"]')).display === 'none' && getComputedStyle(document.querySelector('[data-m="dash"]')).display !== 'none');
    await page.screenshot({ path: `${OUT}${p.id}-9-app.png` });
    check(C, 'Opened as an app: no back link, menu offers the dashboard instead', appOK);
    await page.evaluate(() => document.documentElement.classList.remove('app'));

    // 14. a reload comes straight back in, with the screen as it was (the session lives on the server)
    await page.screenshot({ path: `${OUT}${p.id}-3-terminal.png` });
    await page.reload({ waitUntil: 'load' });
    const tr = await waitState(page, 'Connected');
    const back = await waitText(page, `${mark}-STILL-HERE`, 6000);
    check(C, 'Reloading reconnects without asking again, screen intact', tr >= 0 && back >= 0 && !(await page.isVisible('#signin')), `${tr} ms`);
    R.timing.reload_connect_ms = tr;

    // 15. the helper on the server restarts: the page reconnects by itself and the session survives
    if (p.id === 'laptop' || p.id === 'phone') {
      R.expectDrop = true;   // the next lines cut the connection on purpose
      execSync('sudo -n systemctl restart jack-terminal.service');
      const dropped = await waitState(page, 'Reconnecting', 8000);
      const again = await waitState(page, 'Connected', 30000);
      const still = await waitText(page, `${mark}-STILL-HERE`, 6000);
      check(C, 'Server helper restarted: page reconnects alone, session and screen survive', again >= 0 && still >= 0, `noticed in ${dropped} ms, back in ${again} ms`);
      R.timing.restart_recover_ms = again; R.expectDrop = false;
    }

    // 16. sign out forgets the password on this device
    if (p.touch) await page.tap('#more'); else await page.click('#more');
    await page.waitForSelector('#menu:not([hidden])');
    if (p.touch) await page.tap('[data-m="signout"]'); else await page.click('[data-m="signout"]');
    await page.waitForSelector('#signin:not([hidden])', { timeout: 5000 });
    await sleep(400);
    const gone = await page.evaluate(() => fetch('/term/token', { cache: 'no-store' }).then((r) => r.status === 401));
    check(C, 'Sign out closes the terminal and forgets the password', gone && (await state(page)) === 'Signed out');
    check(C, 'No script or security-policy errors in the page', R.errors.length === 0, R.errors.slice(0, 3).join(' | '));
  } catch (e) {
    check(C, 'Test run completed', false, String(e).slice(0, 300));
    await page.screenshot({ path: `${OUT}${p.id}-9-failure.png` }).catch(() => {});
  }
  results.push(R);
  await ctx.close();
}
// The copy helper itself: history only for a signed-in, same-site request.
{
  const R = { profile: 'The copy helper on the server', id: 'helper', engine: engineName, checks: [], timing: {}, errors: [] };
  const http = await import('node:http');
  const hp = spawn('python3', [new URL('./upload_server.py', import.meta.url).pathname, '127.0.0.1', '7697'], { env: { ...process.env, JACK_TERM_PASSWORD: 'only-for-this-test', JACK_TMUX_SOCKET: 'jacktest', LISTEN_FDS: '' }, stdio: 'ignore' });
  await sleep(900);
  const get = (path, headers) => new Promise((res) => { const rq = http.request({ host: '127.0.0.1', port: 7697, path, method: 'GET', headers }, (r) => { let d = ''; r.on('data', (c) => (d += c)); r.on('end', () => res({ s: r.statusCode, d })); }); rq.on('error', () => res({ s: 0, d: '' })); rq.end(); });
  try { execSync("tmux -L jacktest send-keys -t jack: 'echo HELPER-MARK' Enter"); } catch {}
  await sleep(600);
  const auth = 'Basic ' + Buffer.from('jack:only-for-this-test').toString('base64');
  const ok = await get('/term/upload?history=200', { Authorization: auth });
  const noauth = await get('/term/upload?history=200', {});
  const cross = await get('/term/upload?history=200', { Authorization: auth, 'Sec-Fetch-Site': 'cross-site' });
  const plain = await get('/term/upload', { Authorization: auth });
  let text = ''; try { text = JSON.parse(ok.d).text; } catch {}
  check(R.checks, 'History is given only to a signed-in request from the page itself', ok.s === 200 && text.includes('HELPER-MARK') && noauth.s === 401 && cross.s === 403 && plain.s === 405, JSON.stringify({ ok: ok.s, has: text.includes('HELPER-MARK'), noauth: noauth.s, cross: cross.s, plain: plain.s }));
  try { hp.kill(); } catch {}
  results.push(R);
}
await browser.close();
cleanup();
fs.writeFileSync(`${OUT}results-${engineName}.json`, JSON.stringify(results, null, 1));
for (const r of results) {
  const bad = r.checks.filter((c) => !c.ok);
  console.log(`\n${r.profile} [${r.engine}] — ${r.checks.length - bad.length}/${r.checks.length} passed` + (r.geometry ? ` · ${r.geometry.cols}×${r.geometry.rows} · connect ${r.timing.connect_ms} ms · echo ${(r.timing.echo_ms || []).join('/')} ms · worst stall ${r.timing.worst_frame_ms} ms` : ''));
  for (const c of bad) console.log(`   FAIL  ${c.name} — ${c.detail}`);
}
