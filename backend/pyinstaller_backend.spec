# PyInstaller spec for Windows packaging (offline installer)
# Build on Windows:
#   py -m pip install pyinstaller
#   pyinstaller -y pyinstaller_backend.spec

from PyInstaller.utils.hooks import collect_submodules, collect_data_files
import os


def collect_tree(root: str, prefix: str):
    """Return PyInstaller datas list [(src, dest), ...] for all files under root."""
    out = []
    for dirpath, _dirnames, filenames in os.walk(root):
        for fn in filenames:
            src = os.path.join(dirpath, fn)
            rel = os.path.relpath(src, root)
            dest = os.path.join(prefix, rel)
            out.append((src, dest))
    return out


hiddenimports = []
hiddenimports += collect_submodules('fastapi')
hiddenimports += collect_submodules('uvicorn')

# ytmusicapi uses gettext translations; bundle its locale data
# to avoid: FileNotFoundError: No translation file found for domain: 'base'
yt_data = collect_data_files('ytmusicapi')

# If you use demucs/torch, you may need additional hidden imports.
# hiddenimports += collect_submodules('demucs')

# Bundle static frontend (Next export)
web_datas = collect_tree(os.path.join('..', 'frontend', 'out'), os.path.join('frontend', 'out'))

# Bundle any backend-side templates/assets if needed
backend_datas = collect_data_files('.', includes=['requirements.txt'])

# Vendor binaries (offline)
# - ffmpeg.exe
# - ffprobe.exe
# - yt-dlp.exe
# - rubberband.exe (optional)
vendor_datas = collect_tree('vendor', 'vendor')

block_cipher = None

a = Analysis(
    ['main.py'],
    pathex=['.'],
    binaries=[],
    datas=web_datas + backend_datas + vendor_datas + yt_data,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='holy-backend',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='holy-backend',
)
