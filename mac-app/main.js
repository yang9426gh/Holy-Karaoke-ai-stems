const { app, BrowserWindow, shell, dialog } = require('electron');
const path = require('path');
const { spawn } = require('child_process');
const fs = require('fs');

let win;
let appWin;
let procs = [];

function resourcesAppDir() {
  // In packaged app, resourcesPath points to .../Holy Karaoke Stems.app/Contents/Resources
  return path.join(process.resourcesPath, 'app');
}

function userAppDir() {
  return path.join(app.getPath('userData'), 'app');
}

function installMarkerPath() {
  return path.join(userAppDir(), '.installed.json');
}

function ensureUserMirror() {
  // Copy packaged resources (code only) into userData so we can create venv/data there.
  // This keeps the DMG small and avoids writing into the .app bundle.
  const src = resourcesAppDir();
  const dst = userAppDir();
  const want = { version: app.getVersion() };

  try {
    const cur = JSON.parse(fs.readFileSync(installMarkerPath(), 'utf-8'));
    if (cur && cur.version === want.version && fs.existsSync(path.join(dst, 'backend')) && fs.existsSync(path.join(dst, 'frontend'))) {
      return;
    }
  } catch {}

  fs.mkdirSync(dst, { recursive: true });

  // Node 22 supports fs.cpSync; Electron ships a modern Node.
  fs.cpSync(src, dst, {
    recursive: true,
    dereference: true,
    // Best-effort: backend/data and backend/venv should not exist in packaged resources,
    // but exclude anyway.
    filter: (p) => {
      const rel = path.relative(src, p);
      if (!rel) return true;
      const norm = rel.split(path.sep).join('/');
      if (norm.startsWith('backend/data/')) return false;
      if (norm.startsWith('backend/venv/')) return false;
      if (norm.startsWith('backend/__pycache__/')) return false;
      return true;
    },
  });

  fs.writeFileSync(installMarkerPath(), JSON.stringify(want, null, 2));
}

function configPath() {
  return path.join(app.getPath('userData'), 'config.json');
}

function loadConfig() {
  try {
    return JSON.parse(fs.readFileSync(configPath(), 'utf-8'));
  } catch {
    return { cookiesFromBrowser: 'chrome', useCookies: true };
  }
}

function saveConfig(patch) {
  const cur = loadConfig();
  const next = { ...cur, ...patch };
  fs.mkdirSync(path.dirname(configPath()), { recursive: true });
  fs.writeFileSync(configPath(), JSON.stringify(next, null, 2));
  return next;
}

function sendLog(line) {
  try {
    if (win && !win.isDestroyed()) win.webContents.send('log', String(line));
  } catch {}
}

function sendPhase(name, step, total) {
  try {
    if (win && !win.isDestroyed()) win.webContents.send('phase', { name, step, total });
  } catch {}
}

function runCmd(cmd, args, opts) {
  return new Promise((resolve, reject) => {
    const p = spawn(cmd, args, { ...opts, stdio: 'pipe' });
    p.stdout.on('data', (d) => sendLog(d));
    p.stderr.on('data', (d) => sendLog(d));
    p.on('error', reject);
    p.on('close', (code) => {
      if (code === 0) resolve({ code });
      else reject(new Error(`${cmd} ${args.join(' ')} exited with code ${code}`));
    });
    procs.push(p);
  });
}

function spawnLogged(cmd, args, opts) {
  const p = spawn(cmd, args, { ...opts, stdio: 'pipe' });
  p.stdout.on('data', (d) => sendLog(d));
  p.stderr.on('data', (d) => sendLog(d));
  procs.push(p);
  return p;
}

async function waitFor(url, timeoutMs = 30000) {
  const start = Date.now();
  while (Date.now() - start < timeoutMs) {
    try {
      const res = await fetch(url);
      if (res.ok) return true;
    } catch {}
    await new Promise((r) => setTimeout(r, 500));
  }
  return false;
}

function platformBins(venvDir) {
  const isWin = process.platform === 'win32';
  return {
    pySystem: isWin ? 'python' : '/usr/bin/python3',
    python: isWin ? path.join(venvDir, 'Scripts', 'python.exe') : path.join(venvDir, 'bin', 'python'),
    pip: isWin ? path.join(venvDir, 'Scripts', 'pip.exe') : path.join(venvDir, 'bin', 'pip'),
    npm: isWin ? 'npm.cmd' : 'npm',
  };
}

async function ensureInstalled(root) {
  // First-run bootstrap: create venv + pip install; npm install.
  // For Windows offline builds we will ship a packaged backend exe and skip this.
  ensureUserMirror();

  const backendExe = path.join(root, 'backend-bin', process.platform === 'win32' ? 'holy-backend.exe' : 'holy-backend');
  if (fs.existsSync(backendExe)) {
    sendLog('[skip] bundled backend exe detected (no install needed)\n');
    return;
  }

  const backend = path.join(root, 'backend');
  const frontend = path.join(root, 'frontend');
  const venv = path.join(backend, 'venv');
  const bins = platformBins(venv);

  if (!fs.existsSync(path.join(backend, 'requirements.txt'))) {
    throw new Error('Missing backend/requirements.txt');
  }

  // 1) venv
  sendPhase('Create Python venv', 1, 4);
  if (!fs.existsSync(venv)) {
    await runCmd(bins.pySystem, ['-m', 'venv', 'venv'], { cwd: backend });
  } else {
    sendLog('[skip] venv already exists\n');
  }

  // 2) pip upgrade
  sendPhase('Upgrade pip', 2, 4);
  await runCmd(bins.pip, ['install', '--upgrade', 'pip', 'wheel'], { cwd: backend });

  // 3) requirements
  sendPhase('Install Python dependencies', 3, 4);
  await runCmd(bins.pip, ['install', '-r', 'requirements.txt'], { cwd: backend });

  // 4) npm install
  sendPhase('Install frontend dependencies', 4, 4);
  await runCmd(bins.npm, ['install'], { cwd: frontend });

  sendPhase('Done', 4, 4);
}

async function startServices(root) {
  ensureUserMirror();
  const backend = path.join(root, 'backend');
  const frontend = path.join(root, 'frontend');

  const dataDir = path.join(app.getPath('userData'), 'data');
  try { fs.mkdirSync(dataDir, { recursive: true }); } catch {}

  const webDir = path.join(frontend, 'out');
  const serveWeb = fs.existsSync(path.join(webDir, 'index.html'));

  // Preferred path for Windows offline builds: spawn bundled backend exe
  const backendExe = path.join(root, 'backend-bin', process.platform === 'win32' ? 'holy-backend.exe' : 'holy-backend');
  if (fs.existsSync(backendExe)) {
    spawnLogged(backendExe, [], {
      cwd: path.dirname(backendExe),
      env: {
        ...process.env,
        HOLY_DATA_DIR: dataDir,
        HOLY_DEMUCS_MODEL: 'htdemucs_6s',
        HOLY_YTDLP_COOKIES_FROM_BROWSER: '',
        HOLY_YTDLP_JS_RUNTIMES: 'node',
        HOLY_YTDLP_REMOTE_COMPONENTS: 'ejs:github',
        HOLY_SERVE_WEB: serveWeb ? '1' : '0',
        HOLY_WEB_DIR: webDir,
      }
    });
    return;
  }

  // Dev/legacy path: create venv and run uvicorn + (optional) next dev
  const venv = path.join(backend, 'venv');
  const bins = platformBins(venv);

  const python = bins.python;
  if (!fs.existsSync(python)) {
    throw new Error('Python venv missing. Run Install in the app first.');
  }

  const cfg = loadConfig();
  const cookiesEnv = cfg.useCookies ? String(cfg.cookiesFromBrowser || 'chrome') : '';

  // backend (stems)

  spawnLogged(python, ['-m', 'uvicorn', 'main:app', '--host', '127.0.0.1', '--port', '8011'], {
    cwd: backend,
    env: {
      ...process.env,
      // Store downloaded audio/video/stems OUTSIDE the app bundle
      HOLY_DATA_DIR: dataDir,
      HOLY_DEMUCS_MODEL: 'htdemucs_6s',
      HOLY_YTDLP_COOKIES_FROM_BROWSER: cookiesEnv,
      HOLY_YTDLP_JS_RUNTIMES: 'node',
      HOLY_YTDLP_REMOTE_COMPONENTS: 'ejs:github',
      // Serve static frontend from backend when available (packaged builds)
      HOLY_SERVE_WEB: serveWeb ? '1' : '0',
      HOLY_WEB_DIR: webDir,
    }
  });

  // frontend (dev mode fallback)
  if (!serveWeb) {
    spawnLogged(bins.npm, ['run', 'dev'], {
      cwd: frontend,
      env: {
        ...process.env,
        PORT: '3011',
        NEXT_PUBLIC_BACKEND_URL: 'http://localhost:8011'
      }
    });
  }
}

function createWindow() {
  win = new BrowserWindow({
    width: 520,
    height: 460,
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false,
    }
  });

  win.loadFile(path.join(__dirname, 'installer', 'index.html'));
}

function openAppWindow() {
  if (appWin && !appWin.isDestroyed()) {
    appWin.focus();
    return;
  }

  appWin = new BrowserWindow({
    width: 1200,
    height: 800,
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false,
    }
  });

  appWin.on('closed', () => {
    appWin = null;
  });

  appWin.loadURL('http://localhost:8011');
}

app.whenReady().then(createWindow);

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') app.quit();
});

app.on('before-quit', () => {
  for (const p of procs) {
    try { p.kill('SIGTERM'); } catch {}
  }
});

// IPC via preload
const { ipcMain } = require('electron');

ipcMain.handle('install', async () => {
  // Install into userData (writable) so we don't bloat the DMG and can update safely.
  const root = userAppDir();
  ensureUserMirror();
  await ensureInstalled(root);
  return { ok: true };
});

ipcMain.handle('start', async () => {
  const root = userAppDir();
  ensureUserMirror();
  await startServices(root);

  const ok = await waitFor('http://127.0.0.1:8011/', 45000);
  if (!ok) {
    dialog.showErrorBox('Holy Karaoke Stems', 'Backend did not start on http://localhost:8011 (timed out). Check logs in Console.');
  }

  // Open the app inside Electron so we can provide native dialogs (Save As).
  openAppWindow();
  return { ok: true };
});

ipcMain.handle('config.get', async () => {
  return loadConfig();
});

ipcMain.handle('config.set', async (_ev, patch) => {
  return saveConfig(patch || {});
});

ipcMain.handle('file.save', async (_ev, payload) => {
  const url = String(payload?.url || '');
  const suggestedName = String(payload?.suggestedName || 'mix.mp3');
  if (!url) throw new Error('missing url');

  const { canceled, filePath } = await dialog.showSaveDialog({
    title: 'Export MP3',
    defaultPath: path.join(app.getPath('downloads'), suggestedName),
    filters: [{ name: 'MP3 Audio', extensions: ['mp3'] }],
  });
  if (canceled || !filePath) return { ok: false, canceled: true };

  const res = await fetch(url);
  if (!res.ok) {
    const t = await res.text().catch(() => '');
    throw new Error(`download failed: ${res.status} ${t}`);
  }
  const buf = Buffer.from(await res.arrayBuffer());
  fs.writeFileSync(filePath, buf);
  return { ok: true, path: filePath };
});
