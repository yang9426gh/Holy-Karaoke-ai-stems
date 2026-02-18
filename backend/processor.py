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


def download_youtube_audio(
    url: str,
    output_path: str,
    *,
    cancel_cb: Optional[Callable[[], bool]] = None,
    proc_cb: Optional[Callable[[subprocess.Popen], None]] = None,
    progress_cb: Optional[Callable[[int], None]] = None,
) -> str:
    print(f"[*] Downloading audio from: {url}")
    cmd = [
        sys.executable,
        "-m",
        "yt_dlp",
        "--no-check-certificate",
    ]

    # Optional: use cookies from a local browser profile to avoid bot checks.
    # Enable by setting HOLY_YTDLP_COOKIES_FROM_BROWSER=chrome|edge|firefox|safari
    cfb = os.environ.get("HOLY_YTDLP_COOKIES_FROM_BROWSER", "").strip()
    if cfb:
        cmd += ["--cookies-from-browser", cfb]

    cmd = _maybe_add_js_runtime_args(cmd)

    cmd += [
        "-x",
        "--audio-format",
        "wav",
        "-o",
        f"{output_path}/original.%(ext)s",
        url,
    ]

    vd = _vendor_dir()
    if vd:
        cmd += ["--ffmpeg-location", vd]

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True,
        env=_with_vendor_env(),
    )
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

    return f"{output_path}/original.wav"


def download_youtube_video(
    url: str,
    output_path: str,
    *,
    max_height: int = 720,
    cancel_cb: Optional[Callable[[], bool]] = None,
    proc_cb: Optional[Callable[[subprocess.Popen], None]] = None,
    progress_cb: Optional[Callable[[int], None]] = None,
) -> str:
    """Download a playable MP4 for local <video> playback."""
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

    cmd = [
        sys.executable,
        "-m",
        "yt_dlp",
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
        url,
    ]

    vd = _vendor_dir()
    if vd:
        cmd += ["--ffmpeg-location", vd]

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True,
        env=_with_vendor_env(),
    )
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

    base_cmd = [
        sys.executable,
        "-m",
        "demucs.separate",
        "--name",
        model_name,
        "-o",
        output_dir,
    ]

    # Try Apple Silicon acceleration (MPS) first; if it fails, fall back to CPU.
    cmd = base_cmd + ["--device", "mps", input_file]
    print(f"[*] Demucs cmd (try mps): {' '.join(cmd)}")

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True,
        env=_with_vendor_env(),
    )
    if proc_cb:
        proc_cb(proc)

    import re

    percent_re = re.compile(r"(\d{1,3})%\|")

    last_percent = None
    while True:
        if cancel_cb and cancel_cb():
            try:
                proc.terminate()
            except Exception:
                pass
            raise RuntimeError("cancelled")

        chunk = proc.stdout.readline() if proc.stdout else ""
        if chunk == "" and proc.poll() is not None:
            break
        if not chunk:
            continue

        m = percent_re.search(chunk)
        if m:
            try:
                p = int(m.group(1))
                if progress_cb and p != last_percent and 0 <= p <= 100:
                    progress_cb(p)
                last_percent = p
            except Exception:
                pass

        print(chunk, end="")

    rc = proc.wait()
    if rc != 0:
        # MPS might not be available depending on torch build; retry on CPU once.
        cmd2 = base_cmd + [input_file]
        print(f"[!] Demucs mps failed (rc={rc}). Retrying on CPU: {' '.join(cmd2)}")
        proc2 = subprocess.Popen(
            cmd2,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
        )
        if proc_cb:
            proc_cb(proc2)

        last_percent = None
        while True:
            if cancel_cb and cancel_cb():
                try:
                    proc2.terminate()
                except Exception:
                    pass
                raise RuntimeError("cancelled")

            chunk = proc2.stdout.readline() if proc2.stdout else ""
            if chunk == "" and proc2.poll() is not None:
                break
            if not chunk:
                continue

            m = percent_re.search(chunk)
            if m:
                try:
                    p = int(m.group(1))
                    if progress_cb and p != last_percent and 0 <= p <= 100:
                        progress_cb(p)
                    last_percent = p
                except Exception:
                    pass

            print(chunk, end="")

        rc2 = proc2.wait()
        if rc2 != 0:
            raise subprocess.CalledProcessError(rc2, cmd2)

    if progress_cb:
        progress_cb(100)

    dt = time.time() - t0
    print(f"[+] Separation completed! ({dt:.1f}s)")

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
