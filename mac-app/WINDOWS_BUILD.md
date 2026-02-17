# Windows offline build (one-click installer)

## Goal
Ship a Windows 11 installer (.exe) that installs and runs without the user manually installing Python/Node.

## Architecture
- Electron app (NSIS installer)
- Backend packaged as a Windows exe via PyInstaller (`holy-backend.exe`)
- Frontend shipped as static export (Next `output: 'export'`) and served by backend at http://127.0.0.1:8011/
- All runtime data stored in `%APPDATA%/Holy Karaoke Stems/data` (Electron userData)

## Build steps (on Windows)

### 1) Build frontend static
From repo root:
```bat
cd frontend
npm ci
npm run build
```
This produces `frontend/out`.

### 2) Build backend exe
```bat
cd backend
py -m pip install -r requirements.txt
py -m pip install pyinstaller
pyinstaller -y pyinstaller_backend.spec
```
Output: `backend/dist/holy-backend/holy-backend.exe`

> Note: If you use torch/demucs, you may need extra PyInstaller hooks/hiddenimports.

### 3) Copy backend exe into Electron resources
Copy the PyInstaller output folder into `mac-app/installer/win/backend/` (or similar) and update `extraResources` to include it.

### 4) Build Electron installer
```bat
cd mac-app
npm ci
npm run dist:win
```

## TODOs
- Update Electron main.js to prefer running bundled `holy-backend.exe` on win32 (no venv).
- Bundle ffmpeg + (optional) rubberband into backend/vendor so the backend can run fully offline.
- Consider code-signing for Windows SmartScreen.
