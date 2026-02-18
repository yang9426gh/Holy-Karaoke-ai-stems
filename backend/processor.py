import os
import sys
import subprocess
import time
import shutil
from typing import Callable, Optional


def _vendor_dir() -> str | None:
    """Return path to bundled vendor binaries if present (PyInstaller)."""
    try:
        base = getattr(sys, "_MEIPASS", None)
        if base:
            cand = os.path.join(base, "vendor")
            if os.path.isdir(cand):
                return cand
    except Exception:
        pass

    # dev/workspace
    cand2 = os.path.join(os.path.dirname(__file__), "vendor")
    if os.path.isdir(cand2):
        return cand2

    return None


def _with_vendor_env(env: dict | None = None) -> dict:
    out = dict(env or os.environ)
    vd = _vendor_dir()
    if vd:
        out["PATH"] = vd + os.pathsep + out.get("PATH", "")
    return out


def _maybe_add_js_runtime_args(cmd: list[str]) -> list[str]:
    # On Windows offline builds, node might not exist. Only enable js runtimes if we can find it.
    jsr = os.environ.get("HOLY_YTDLP_JS_RUNTIMES", "node").strip()
    if not jsr:
        return cmd

    # If configured as "node" (default), ensure it's actually available
    if jsr == "node" and not shutil.which("node"):
        return cmd

    cmd += ["--js-runtimes", jsr]

    rc = os.environ.get("HOLY_YTDLP_REMOTE_COMPONENTS", "ejs:github").strip()
    if rc:
        cmd += ["--remote-components", rc]

    return cmd


def _vendor_exe(name: str) -> str | None:
    vd = _vendor_dir()
    if not vd:
        return None
    for cand in [os.path.join(vd, name), os.path.join(vd, f"{name}.exe")]:
        if os.path.exists(cand):
            return cand
    return None


def download_youtube_audio(
    url: str,
    output_path: str,
    *,
    cancel_cb: Optional[Callable[[], bool]] = None,
    proc_cb: Optional[Callable[[subprocess.Popen], None]] = None,
    progress_cb: Optional[Callable[[int], None]] = None,
) -> str:
    """Download audio using yt-dlp.

    IMPORTANT for PyInstaller builds: do NOT spawn sys.executable -m yt_dlp,
    because sys.executable points to the packaged backend server exe.
    Prefer vendor yt-dlp.exe when available.
    """
    print(f"[*] Downloading audio from: {url}")

    ytdlp_exe = _vendor_exe("yt-dlp")
    if not ytdlp_exe:
        # fallback: run via python API in-process
        try:
            import yt_dlp

            outtmpl = os.path.join(output_path, "original.%(ext)s")
            opts = {
                "outtmpl": outtmpl,
                "nocheckcertificate": True,
                "format": "bestaudio/best",
                "postprocessors": [
                    {
                        "key": "FFmpegExtractAudio",
                        "preferredcodec": "wav",
                    }
                ],
                "progress_hooks": [
                    lambda d: progress_cb(int(float(d.get("_percent_str", "0").strip("%"))))
                    if progress_cb and d.get("status") == "downloading" and d.get("_percent_str")
                    else None
                ],
            }
            with yt_dlp.YoutubeDL(opts) as ydl:
                ydl.download([url])
        except Exception as e:
            raise RuntimeError(f"yt-dlp python API failed: {e}")
        return os.path.join(output_path, "original.wav")

    cmd = [
        ytdlp_exe,
        "--no-check-certificate",
        "-x",
        "--audio-format",
        "wav",
        "-o",
        os.path.join(output_path, "original.%(ext)s"),
    ]

    cfb = os.environ.get("HOLY_YTDLP_COOKIES_FROM_BROWSER", "").strip()
    if cfb:
        cmd += ["--cookies-from-browser", cfb]

    # JS runtime is optional; enable only if configured and present.
    cmd = _maybe_add_js_runtime_args(cmd)

    vd = _vendor_dir()
    ff = _vendor_exe("ffmpeg")
    if ff:
        cmd += ["--ffmpeg-location", ff]
    elif vd:
        cmd += ["--ffmpeg-location", vd]

    cmd += [url]

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
            env=_with_vendor_env(),
        )
    except PermissionError as e:
        # Some Windows environments block executing bundled .exe (SmartScreen/AV/permissions).
        # Fall back to yt-dlp Python API in-process.
        print(f"[warn] vendor yt-dlp exec blocked ({e}); falling back to Python yt_dlp")
        try:
            import yt_dlp

            outtmpl = os.path.join(output_path, "original.%(ext)s")
            opts = {
                "outtmpl": outtmpl,
                "nocheckcertificate": True,
                "format": "bestaudio/best",
                "ffmpeg_location": ff or vd or None,
                "postprocessors": [
                    {
                        "key": "FFmpegExtractAudio",
                        "preferredcodec": "wav",
                    }
                ],
                "progress_hooks": [
                    lambda d: progress_cb(int(float(d.get("_percent_str", "0").strip("%"))))
                    if progress_cb and d.get("status") == "downloading" and d.get("_percent_str")
                    else None
                ],
            }
            with yt_dlp.YoutubeDL(opts) as ydl:
                ydl.download([url])
        except Exception as ee:
            raise RuntimeError(f"yt-dlp python API failed after PermissionError: {ee}")
        return os.path.join(output_path, "original.wav")

    if proc_cb:
        proc_cb(proc)

    import re

    pct_re = re.compile(r"\[download\]\s+(\d{1,3}(?:\.\d+)?)%")

    while True:
        if cancel_cb and cancel_cb():
            try:
                proc.terminate()
            except Exception:
                pass
            raise RuntimeError("cancelled")

        line = proc.stdout.readline() if proc.stdout else ""
        if line:
            print(line, end="")
            m = pct_re.search(line)
            if m and progress_cb:
                try:
                    progress_cb(int(float(m.group(1))))
                except Exception:
                    pass

        if line == "" and proc.poll() is not None:
            break

        time.sleep(0.01)

    rc = proc.wait()
    if rc != 0:
        raise subprocess.CalledProcessError(rc, cmd)

    return os.path.join(output_path, "original.wav")


def download_youtube_video(
    url: str,
    output_path: str,
    *,
    max_height: int = 720,
    cancel_cb: Optional[Callable[[], bool]] = None,
    proc_cb: Optional[Callable[[subprocess.Popen], None]] = None,
    progress_cb: Optional[Callable[[int], None]] = None,
) -> str:
    """Download a playable MP4 for local <video> playback.

    IMPORTANT for PyInstaller builds: do NOT spawn sys.executable -m yt_dlp.
    Prefer vendor yt-dlp.exe when available.
    """
    out = os.path.join(output_path, "video.mp4")
    if os.path.exists(out) and os.path.getsize(out) > 1024 * 1024:
        if progress_cb:
            progress_cb(100)
        return out

    print(f"[*] Downloading video from: {url}")
    fmt = (
        f"bestvideo[height<={max_height}][ext=mp4]+bestaudio[ext=m4a]/"
        f"best[height<={max_height}][ext=mp4]/best[height<={max_height}]"
    )

    ytdlp_exe = _vendor_exe("yt-dlp")
    if not ytdlp_exe:
        # fallback: python API in-process
        try:
            import yt_dlp

            opts = {
                "outtmpl": out,
                "nocheckcertificate": True,
                "format": fmt,
                "merge_output_format": "mp4",
                "progress_hooks": [
                    lambda d: progress_cb(int(float(d.get("_percent_str", "0").strip("%"))))
                    if progress_cb and d.get("status") == "downloading" and d.get("_percent_str")
                    else None
                ],
            }
            with yt_dlp.YoutubeDL(opts) as ydl:
                ydl.download([url])
        except Exception as e:
            raise RuntimeError(f"yt-dlp python API failed: {e}")

        if progress_cb:
            progress_cb(100)
        return out

    cmd = [
        ytdlp_exe,
        "--no-check-certificate",
    ]

    cfb = os.environ.get("HOLY_YTDLP_COOKIES_FROM_BROWSER", "").strip()
    if cfb:
        cmd += ["--cookies-from-browser", cfb]

    cmd = _maybe_add_js_runtime_args(cmd)

    cmd += [
        "-f",
        fmt,
        "--merge-output-format",
        "mp4",
        "-o",
        out,
    ]

    vd = _vendor_dir()
    ff = _vendor_exe("ffmpeg")
    if ff:
        cmd += ["--ffmpeg-location", ff]
    elif vd:
        cmd += ["--ffmpeg-location", vd]

    cmd += [url]

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
            env=_with_vendor_env(),
        )
    except PermissionError as e:
        print(f"[warn] vendor yt-dlp exec blocked ({e}); falling back to Python yt_dlp")
        try:
            import yt_dlp

            opts = {
                "outtmpl": out,
                "nocheckcertificate": True,
                "format": fmt,
                "merge_output_format": "mp4",
                "ffmpeg_location": ff or vd or None,
                "progress_hooks": [
                    lambda d: progress_cb(int(float(d.get("_percent_str", "0").strip("%"))))
                    if progress_cb and d.get("status") == "downloading" and d.get("_percent_str")
                    else None
                ],
            }
            with yt_dlp.YoutubeDL(opts) as ydl:
                ydl.download([url])
        except Exception as ee:
            raise RuntimeError(f"yt-dlp python API failed after PermissionError: {ee}")

        if progress_cb:
            progress_cb(100)
        return out

    if proc_cb:
        proc_cb(proc)

    import re

    pct_re = re.compile(r"\[download\]\s+(\d{1,3}(?:\.\d+)?)%")

    while True:
        if cancel_cb and cancel_cb():
            try:
                proc.terminate()
            except Exception:
                pass
            raise RuntimeError("cancelled")

        line = proc.stdout.readline() if proc.stdout else ""
        if line:
            print(line, end="")
            m = pct_re.search(line)
            if m and progress_cb:
                try:
                    progress_cb(int(float(m.group(1))))
                except Exception:
                    pass

        if line == "" and proc.poll() is not None:
            break

        time.sleep(0.01)

    rc = proc.wait()
    if rc != 0:
        raise subprocess.CalledProcessError(rc, cmd)

    if progress_cb:
        progress_cb(100)

    return out



def separate_stems(
    input_file: str,
    output_dir: str,
    task_id: str | None = None,
    progress_cb=None,
    *,
    model_name: str = "htdemucs_6s",
    cancel_cb: Optional[Callable[[], bool]] = None,
    proc_cb: Optional[Callable[[subprocess.Popen], None]] = None,
):
    print(f"[*] Separating stems using AI (Demucs model={model_name})...")
    t0 = time.time()

    # In packaged builds (PyInstaller), sys.executable points to the backend server exe.
    # Spawning subprocesses like "sys.executable -m demucs.separate" can accidentally
    # start another server instance. So we run Demucs in-process.
    try:
        from demucs.separate import main as demucs_main

        def _run(device: str, args_base: list[str]):
            # demucs_main expects argv-like list
            print(f"[*] Demucs (in-process) device={device}")
            return demucs_main(args_base + ["--device", device, input_file])

        args_base = [
            "--name",
            model_name,
            "-o",
            output_dir,
        ]

        try:
            _run("mps", args_base)
        except Exception:
            # CPU fallback (works on Windows)
            demucs_main(args_base + [input_file])

        if progress_cb:
            progress_cb(100)

        dt = time.time() - t0
        print(f"[+] Separation completed! ({dt:.1f}s)")
        return

    except Exception as e:
        raise RuntimeError(f"Demucs failed: {e}")

if __name__ == "__main__":
    # 테스트용 (유튜브 URL을 인자로 받음)
    if len(sys.argv) > 1:
        yt_url = sys.argv[1]
        target_dir = "./data/test_run"
        os.makedirs(target_dir, exist_ok=True)

        orig_file = download_youtube_audio(yt_url, target_dir)
        separate_stems(orig_file, target_dir)
    else:
        print("Usage: python3 processor.py <youtube_url>")
