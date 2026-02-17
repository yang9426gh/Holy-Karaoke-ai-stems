import os
import subprocess
import threading
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, Tuple


def transpose_wav(src: str, dst: str, semitones: int, *, stem_name: str | None = None) -> None:
    os.makedirs(os.path.dirname(dst), exist_ok=True)

    rb = "/opt/homebrew/bin/rubberband"
    if os.path.exists(rb):
        # Quality-first: use the finer R3 engine + formant preservation.
        # Parallelism (ProcessPool) still provides the speedup.
        args = [
            rb,
            "-3",
            "-F",
            "--centre-focus",
            "-q",
            "-t",
            "1.0",
            "-p",
            str(semitones),
            src,
            dst,
        ]

        subprocess.run(
            args,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return

    # fallback ffmpeg (lower quality)
    factor = 2 ** (float(semitones) / 12.0)
    inv = 1.0 / factor
    at = inv
    at_filters = []
    while at < 0.5:
        at_filters.append("atempo=0.5")
        at /= 0.5
    while at > 2.0:
        at_filters.append("atempo=2.0")
        at /= 2.0
    at_filters.append(f"atempo={at:.6f}")

    af = f"asetrate=44100*{factor:.8f},aresample=44100," + ",".join(at_filters)
    subprocess.run(
        [
            "/opt/homebrew/bin/ffmpeg",
            "-y",
            "-i",
            src,
            "-vn",
            "-ar",
            "44100",
            "-af",
            af,
            dst,
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def ensure_transposed_stems(
    *,
    data_dir: str,
    task_id: str,
    semitones: int,
    stems: Dict[str, str],
    progress_cb=None,
) -> Tuple[str, Dict[str, str]]:
    """Create transposed stems under data_dir/task_id/transpose_{+n}/<stem>.wav

    Returns (out_dir, rel_map) where rel_map is {stem: relative_path_under_data_dir}
    """
    out_dir = os.path.join(data_dir, task_id, f"transpose_{semitones:+d}")
    os.makedirs(out_dir, exist_ok=True)

    rel_map: Dict[str, str] = {}
    items = list(stems.items())
    total = max(1, len(items))

    # Drums: keep original (unpitched)
    for name, src in items:
        if name == "drums":
            rel_map[name] = os.path.relpath(src, data_dir)

    # Work list (skip drums, skip already-done)
    work: list[tuple[str, str, str]] = []  # (name, src, dst)
    for name, src in items:
        if name == "drums":
            continue
        dst = os.path.join(out_dir, f"{name}.wav")
        rel_map[name] = os.path.relpath(dst, data_dir)
        if os.path.exists(dst) and os.path.getsize(dst) > 1024:
            continue
        work.append((name, src, dst))

    # Parallelize external processing across stems.
    done = 0

    if not work:
        if progress_cb:
            try:
                progress_cb(100)
            except Exception:
                pass
        return out_dir, rel_map

    max_workers = min(len(work), max(1, (os.cpu_count() or 4) - 1))

    base_done = total - len(work)  # drums + already-cached outputs
    submit_times: dict[int, float] = {}
    completed_durations: list[float] = []
    stop_tick = threading.Event()

    def tick() -> None:
        # Smooth progress while jobs are running by estimating per-stem completion.
        # This avoids "stuck at 0 then jump" when running in parallel.
        while not stop_tick.is_set():
            if not progress_cb:
                time.sleep(0.25)
                continue
            try:
                now = time.time()
                mean = (sum(completed_durations) / len(completed_durations)) if completed_durations else 20.0
                mean = max(5.0, float(mean))

                est_done = float(base_done + done)
                # Estimate partial progress for in-flight jobs
                for _fid, t0 in list(submit_times.items()):
                    elapsed = now - t0
                    # cap each in-flight stem contribution at 0.95
                    est_done += min(0.95, max(0.0, elapsed / mean))

                pct = int(round((est_done / total) * 100))
                # Don't report 100 until truly done
                pct = min(99, max(0, pct))
                progress_cb(pct)
            except Exception:
                pass
            time.sleep(0.4)

    tick_thread = None
    if progress_cb:
        tick_thread = threading.Thread(target=tick, daemon=True)
        tick_thread.start()

    try:
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            futs = []
            for (name, src, dst) in work:
                f = ex.submit(transpose_wav, src, dst, semitones, stem_name=name)
                submit_times[id(f)] = time.time()
                futs.append(f)

            for f in as_completed(futs):
                t0 = submit_times.pop(id(f), None)
                if t0 is not None:
                    completed_durations.append(time.time() - t0)
                done += 1
                if progress_cb:
                    try:
                        pct = int(round(((base_done + done) / total) * 100))
                        progress_cb(min(99, max(0, pct)))
                    except Exception:
                        pass
    finally:
        stop_tick.set()
        if tick_thread:
            try:
                tick_thread.join(timeout=1.0)
            except Exception:
                pass

    if progress_cb:
        try:
            progress_cb(100)
        except Exception:
            pass

    return out_dir, rel_map
