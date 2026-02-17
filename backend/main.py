from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi import HTTPException
from fastapi import Body
from starlette.concurrency import run_in_threadpool
from pydantic import BaseModel
import os
import uuid
import hashlib
import threading
import traceback
from processor import download_youtube_audio, download_youtube_video, separate_stems
from youtube_transcript_api import YouTubeTranscriptApi
from ytmusicapi import YTMusic

from db import init_db, upsert_track, get_track, list_tracks, upsert_preset, list_presets, delete_preset, delete_track
from lyrics_sync import align_lyrics_to_segments, align_lyrics_to_words, align_lyrics_to_words_dp, align_lyrics_to_words_anchored
from rapidfuzz import fuzz

YT = YouTubeTranscriptApi()
YTM = YTMusic()

import whisper

# Optional: whisperx for better word-level alignment
try:
    import whisperx  # type: ignore
except Exception:  # pragma: no cover
    whisperx = None

WHISPER_MODELS: dict[str, object] = {}
WHISPER_LOCK = threading.Lock()


def wav_duration_seconds(path: str) -> float:
    """Cheap duration reader for WAV files (no ffmpeg dependency)."""
    try:
        import wave
        with wave.open(path, "rb") as w:
            frames = w.getnframes()
            rate = w.getframerate() or 1
            return float(frames) / float(rate)
    except Exception:
        return 0.0


def is_bad_lyric_sync(
    lines: list[dict],
    *,
    max_gap_s: float = 25.0,
    cluster_ratio: float = 0.18,
    min_step_s: float = 0.25,
) -> bool:
    """Heuristics to detect degenerate timestamp outputs.

    Symptoms we want to catch:
      - Huge gaps (lyrics freeze for a long time)
      - Tight clusters (lyrics fly by insanely fast)
      - Duplicate/zero deltas (multiple lines mapped to the same time)

    Note: we run this BEFORE monotonic-normalization.
    """
    if not lines or len(lines) < 8:
        return True

    times: list[float] = []
    for ln in lines:
        try:
            times.append(float(ln.get("time", 0) or 0))
        except Exception:
            times.append(0.0)

    # Must be monotonic-ish
    for i in range(1, len(times)):
        if times[i] < times[i - 1]:
            return True

    deltas = [times[i] - times[i - 1] for i in range(1, len(times))]
    if not deltas:
        return True

    if max(deltas) >= max_gap_s:
        return True

    # duplicates / too-small steps
    zeroish = sum(1 for d in deltas if d <= 0.001)
    if zeroish > 0:
        return True

    small = sum(1 for d in deltas if d <= min_step_s)
    if small / max(1, len(deltas)) >= cluster_ratio:
        return True

    return False


def linear_fallback_sync(lyric_lines: list[str], duration_s: float) -> list[dict]:
    """Fallback: spread lyric lines evenly across the track duration."""
    n = len(lyric_lines)
    if n <= 0:
        return []

    dur = float(duration_s or 0)
    if dur <= 0:
        # sane default
        dur = max(60.0, n * 2.5)

    # leave a small tail so the last line stays visible
    usable = max(5.0, dur - 2.0)
    step = usable / max(1, n)
    step = max(0.35, step)  # prevent too-fast scrolling

    out = []
    t = 0.0
    for ln in lyric_lines:
        out.append({"time": t, "text": ln})
        t += step
    return out


def _norm_text(s: str) -> str:
    s = (s or "").lower().strip()
    # cheap normalization for matching
    for ch in ["\"", "'", "’", ",", ".", "!", "?", "(", ")", "[", "]"]:
        s = s.replace(ch, "")
    s = " ".join(s.split())
    return s


def captions_phrases(video_id: str) -> list[dict]:
    """Fetch YouTube transcript captions (with timestamps) and merge into short phrases."""
    # try English first, then any available
    try_langs = ["en", "en-US", "en-GB"]
    transcript = None
    last_err = None
    # Newer youtube-transcript-api uses .fetch() and .list()
    for lang in try_langs:
        try:
            transcript = YT.fetch(video_id, languages=[lang])
            break
        except Exception as e:
            last_err = e
            transcript = None

    if transcript is None:
        # fallback: try without language (may pick auto)
        try:
            transcript = YT.fetch(video_id)
        except Exception as e:
            raise Exception(f"No YouTube transcript available: {last_err or e}")

    # Merge into phrases ~ up to 70 chars or ~4s span
    phrases: list[dict] = []
    cur = {"start": None, "text": ""}
    cur_end = None

    for item in transcript:
        t = float(item.get("start") or 0)
        d = float(item.get("duration") or 0)
        txt = _norm_text(item.get("text") or "")
        if not txt:
            continue

        if cur["start"] is None:
            cur["start"] = t
            cur["text"] = txt
            cur_end = t + d
            continue

        # decide whether to append or flush
        span = (t + d) - float(cur["start"])
        if len(cur["text"]) < 70 and span < 4.5:
            cur["text"] = (cur["text"] + " " + txt).strip()
            cur_end = t + d
        else:
            phrases.append({"start": float(cur["start"]), "text": cur["text"]})
            cur = {"start": t, "text": txt}
            cur_end = t + d

    if cur["start"] is not None and cur["text"]:
        phrases.append({"start": float(cur["start"]), "text": cur["text"]})

    return phrases


def align_lyrics_to_captions_lines(lyric_lines: list[str], phrases: list[dict]) -> list[dict]:
    """Monotonic greedy match: each lyric line picks best caption phrase ahead."""
    out: list[dict] = []
    if not lyric_lines:
        return out
    if not phrases:
        raise Exception("No caption phrases")

    p_texts = [p.get("text") or "" for p in phrases]

    j = 0
    for ln in lyric_lines:
        q = _norm_text(ln)
        if not q:
            continue

        best_i = None
        best_score = -1

        # search forward window
        win = 120
        for i in range(j, min(len(phrases), j + win)):
            score = fuzz.token_set_ratio(q, p_texts[i])
            if score > best_score:
                best_score = score
                best_i = i
                if score >= 95:
                    break

        if best_i is None:
            best_i = j
            best_score = 0

        # If score is very low, we still advance slowly to keep monotonicity
        # but mark score so we can inspect.
        start_t = float(phrases[best_i].get("start") or 0)
        out.append({"time": start_t, "text": ln, "score": int(best_score)})

        # advance pointer; if we matched confidently, jump to that index+1, else inch forward
        if best_score >= 60:
            j = max(j, best_i + 1)
        else:
            j = min(len(phrases) - 1, j + 1)

    return out


def get_whisper_model(name: str):
    with WHISPER_LOCK:
        m = WHISPER_MODELS.get(name)
        if m is None:
            m = whisper.load_model(name)
            WHISPER_MODELS[name] = m
        return m


def whisperx_words(audio_path: str, model_name: str = "medium", lang: str = "en") -> list[dict]:
    """Return word timestamps using whisperx aligner if available.

    WhisperX can sometimes collapse into filler tokens ("oh", "uh") on singing.
    If the aligned words look degenerate, we fall back to openai-whisper.
    """
    if whisperx is None:
        return []

    device = "cpu"
    try:
        wmodel = whisperx.load_model(model_name, device, compute_type="int8")
        result = wmodel.transcribe(audio_path, language=lang)
        align_model, metadata = whisperx.load_align_model(language_code=lang, device=device)
        aligned = whisperx.align(result["segments"], align_model, metadata, audio_path, device)

        words: list[dict] = []
        toks: list[str] = []
        for w in aligned.get("word_segments") or []:
            ww = (w.get("word") or "").strip().lower()
            words.append({"start": w.get("start"), "end": w.get("end"), "word": ww})
            if ww and any(ch.isalpha() for ch in ww):
                toks.append(ww)

        # Degeneracy check: if too repetitive, don't trust whisperx words
        if len(toks) >= 30:
            from collections import Counter
            c = Counter(toks)
            top_tok, top_cnt = c.most_common(1)[0]
            uniq = len(c)
            top_ratio = top_cnt / max(1, len(toks))
            # if top token dominates or uniqueness is too low, reject
            if top_ratio > 0.20 or uniq < 25:
                return []

        return words
    except Exception:
        return []

app = FastAPI()

# -----------------------------
# Queue / background worker
# -----------------------------
# NOTE: these must be defined BEFORE starting the worker thread.
from collections import deque

QUEUE = deque()  # items: (url, output_dir, task_id)
QUEUE_LOCK = threading.Lock()
QUEUE_COND = threading.Condition(QUEUE_LOCK)
CURRENT: dict | None = None  # {task_id, url}


def _queue_worker():
    global CURRENT
    while True:
        with QUEUE_COND:
            while not QUEUE:
                QUEUE_COND.wait()
            url, output_dir, task_id = QUEUE.popleft()
            CURRENT = {"task_id": task_id, "url": url}

        try:
            run_pipeline(url, output_dir, task_id)
        except Exception:
            # run_pipeline already sets error stage
            pass
        finally:
            with QUEUE_COND:
                if CURRENT and CURRENT.get("task_id") == task_id:
                    CURRENT = None


WORKER_THREAD = threading.Thread(target=_queue_worker, daemon=True)
WORKER_THREAD.start()

# 프론트엔드와 통신을 위한 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Data/cache directory (can override with HOLY_DATA_DIR)
DATA_DIR = os.environ.get("HOLY_DATA_DIR") or os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_DIR, exist_ok=True)

# in-memory task progress (simple dev implementation)
TASKS: dict[str, dict] = {}
TASKS_LOCK = threading.Lock()

# cancellation + running processes (best-effort)
CANCELLED: set[str] = set()
RUNNING_PROCS: dict[str, object] = {}
RUNNING_LOCK = threading.Lock()

# job history
from collections import deque as _deque
JOB_HISTORY = _deque(maxlen=50)  # each: {task_id,url,status,stage,progress,ts}

# init cache DB
init_db(DATA_DIR)

# 분리된 파일들을 웹에서 접근 가능하게 설정
app.mount("/audio", StaticFiles(directory=DATA_DIR), name="audio")


class PresetIn(BaseModel):
    id: str
    name: str
    source_url: str
    video_id: str
    title: str | None = None
    thumbnail_url: str | None = None
    bpm: float | None = None
    key_tonic: str | None = None  # e.g. C, C#, D...
    key_mode: str | None = None   # major|minor
    semitones: int = 0
    master_volume: float = 1.0
    vocal_volume: float = 1.0

def video_id_from_url(url: str) -> str:
    # supports youtube.com/watch?v=... and youtu.be/... and music.youtube.com
    try:
        from urllib.parse import urlparse, parse_qs

        u = urlparse(url)
        if u.netloc.endswith("youtu.be"):
            vid = u.path.strip("/")
            return vid
        qs = parse_qs(u.query)
        vid = (qs.get("v") or [""])[0]
        return vid
    except Exception:
        # fallback best-effort
        if "v=" in url:
            return url.split("v=")[1].split("&")[0]
        return url.split("/")[-1].split("?")[0]


def canonical_watch_url(url: str) -> str:
    vid = video_id_from_url(url)
    if not vid:
        return url
    return f"https://www.youtube.com/watch?v={vid}"


def task_id_for_url(url: str) -> str:
    # deterministic by video id (avoid list/start_radio params breaking cache)
    vid = video_id_from_url(url) or url.strip()
    return hashlib.sha1(vid.encode("utf-8")).hexdigest()[:16]


def output_paths(task_id: str):
    out_dir = os.path.join(DATA_DIR, task_id)
    # legacy paths (2-stem). 6-stem will be discovered dynamically via find_stems().
    vocals = os.path.join(out_dir, "htdemucs", "original", "vocals.wav")
    no_vocals = os.path.join(out_dir, "htdemucs", "original", "no_vocals.wav")
    video = os.path.join(out_dir, "video.mp4")
    return out_dir, vocals, no_vocals, video


def find_stems(out_dir: str) -> dict[str, str]:
    """Return {stem: wav_path} from demucs output."""
    candidates = []
    m = os.environ.get("HOLY_DEMUCS_MODEL", "htdemucs_6s")
    candidates.append(os.path.join(out_dir, m, "original"))
    candidates.append(os.path.join(out_dir, "htdemucs", "original"))
    candidates.append(os.path.join(out_dir, "htdemucs_6s", "original"))
    for c in candidates:
        if os.path.isdir(c):
            stems = {}
            for fn in os.listdir(c):
                if fn.lower().endswith(".wav"):
                    stems[os.path.splitext(fn)[0]] = os.path.join(c, fn)
            if stems:
                return stems
    return {}


@app.post("/process")
async def process_video(url: str, background_tasks: BackgroundTasks, use_test_run: bool = False, queued: bool = False):
    task_id = task_id_for_url(url)
    output_dir, vocals_file, no_vocals_file, video_file = output_paths(task_id)
    os.makedirs(output_dir, exist_ok=True)

    # If we already cached this video_id, return immediately.
    vid = video_id_from_url(url)
    if vid:
        cached = get_track(DATA_DIR, vid)
        if cached and os.path.exists(cached.get("vocals_path", "")) and os.path.exists(cached.get("no_vocals_path", "")):
            tid = cached.get("task_id")
            video_path = os.path.join(DATA_DIR, str(tid), "video.mp4") if tid else ""
            return {"task_id": tid, "status": "completed", "cached": True, "video": f"/audio/{tid}/video.mp4" if tid and os.path.exists(video_path) else None}

    # cache hit -> already completed (multi-stem)
    if find_stems(output_dir):
        return {"task_id": task_id, "status": "completed", "video": f"/audio/{task_id}/video.mp4" if os.path.exists(video_file) else None}

    # legacy cache hit (2-stem)
    if os.path.exists(vocals_file) and os.path.exists(no_vocals_file):
        return {"task_id": task_id, "status": "completed", "video": f"/audio/{task_id}/video.mp4" if os.path.exists(video_file) else None}

    # optional dev shortcut: reuse local test_run outputs
    if use_test_run:
        test_run_dir = os.path.join(DATA_DIR, "test_run")
        test_vocals = os.path.join(test_run_dir, "htdemucs", "original", "vocals.wav")
        test_no_vocals = os.path.join(test_run_dir, "htdemucs", "original", "no_vocals.wav")
        if os.path.exists(test_vocals) and os.path.exists(test_no_vocals):
            try:
                link_path = os.path.join(output_dir, "htdemucs")
                if not os.path.exists(link_path):
                    os.symlink(os.path.join(test_run_dir, "htdemucs"), link_path)
                return {"task_id": task_id, "status": "completed"}
            except Exception:
                pass

    # run in background (can take minutes)
    set_task(task_id, status="processing", stage="queued", progress=0)

    if queued:
        with QUEUE_COND:
            QUEUE.append((url, output_dir, task_id))
            QUEUE_COND.notify()
        return {"task_id": task_id, "status": "processing", "queued": True}

    background_tasks.add_task(run_pipeline, url, output_dir, task_id)
    return {"task_id": task_id, "status": "processing", "queued": False}


def set_task(task_id: str, **patch):
    with TASKS_LOCK:
        cur = TASKS.get(task_id, {})
        cur.update(patch)
        TASKS[task_id] = cur


def is_cancelled(task_id: str) -> bool:
    return task_id in CANCELLED


def register_proc(task_id: str, proc):
    with RUNNING_LOCK:
        RUNNING_PROCS[task_id] = proc


def clear_proc(task_id: str):
    with RUNNING_LOCK:
        RUNNING_PROCS.pop(task_id, None)


def push_history(task_id: str, url: str, status: str, stage: str | None, progress: int | float | None):
    import time
    JOB_HISTORY.appendleft(
        {
            "task_id": task_id,
            "url": url,
            "status": status,
            "stage": stage,
            "progress": progress,
            "ts": int(time.time()),
        }
    )


def run_pipeline(url: str, output_dir: str, task_id: str):
    try:
        if is_cancelled(task_id):
            raise RuntimeError("cancelled")

        vid = video_id_from_url(url)
        set_task(task_id, status="processing", stage="downloading", progress=0, url=url)

        canon = canonical_watch_url(url)

        # Download video for local playback (720p max) - cache reused by task_id
        set_task(task_id, status="processing", stage="downloading_video", progress=0)
        try:
            download_youtube_video(
                canon,
                output_dir,
                max_height=720,
                cancel_cb=lambda: is_cancelled(task_id),
                proc_cb=lambda p: register_proc(task_id, p),
                progress_cb=lambda p: set_task(task_id, progress=p),
            )
        finally:
            clear_proc(task_id)

        if is_cancelled(task_id):
            raise RuntimeError("cancelled")

        set_task(task_id, status="processing", stage="downloading_audio", progress=0)
        orig_file = download_youtube_audio(
            canon,
            output_dir,
            cancel_cb=lambda: is_cancelled(task_id),
            proc_cb=lambda p: register_proc(task_id, p),
            progress_cb=lambda p: set_task(task_id, progress=p),
        )
        clear_proc(task_id)

        if is_cancelled(task_id):
            raise RuntimeError("cancelled")

        set_task(task_id, status="processing", stage="separating", progress=0)
        separate_stems(
            orig_file,
            output_dir,
            task_id=task_id,
            model_name=os.environ.get("HOLY_DEMUCS_MODEL", "htdemucs_6s"),
            progress_cb=lambda p: set_task(task_id, progress=p),
            cancel_cb=lambda: is_cancelled(task_id),
            proc_cb=lambda p: register_proc(task_id, p),
        )
        clear_proc(task_id)

        # attempt lyric sync (A): YT Music lyrics text + Whisper segments
        set_task(task_id, stage="transcribing", progress=0)

        stems = find_stems(output_dir)
        vocals_path = stems.get("vocals") or ""
        no_vocals_path = stems.get("no_vocals") or stems.get("accompaniment") or ""

        # fetch ytmusic lyrics (text)
        lyric_lines = []
        try:
            wp = YTM.get_watch_playlist(vid)
            lyrics_id = wp.get("lyrics")
            if lyrics_id:
                ly = YTM.get_lyrics(lyrics_id)
                text = (ly.get("lyrics") or "").strip()
                lyric_lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        except Exception:
            lyric_lines = []

        synced_path = None
        if lyric_lines and os.path.exists(vocals_path):
            model_name = os.environ.get("HOLY_WHISPER_MODEL", "base")
            model = get_whisper_model(model_name)
            set_task(task_id, stage="transcribing", progress=25)
            result = model.transcribe(
                vocals_path,
                fp16=False,
                language="en",
                task="transcribe",
                temperature=0.0,
                condition_on_previous_text=False,
                word_timestamps=True,
            )
            segments = [{"start": s["start"], "end": s["end"], "text": s["text"]} for s in (result.get("segments") or [])]

            # gather word timestamps if available
            words = []
            for seg in (result.get("segments") or []):
                for w in seg.get("words") or []:
                    words.append({"start": w.get("start"), "end": w.get("end"), "word": w.get("word")})

            set_task(task_id, stage="aligning_lyrics", progress=0)
            if words:
                # Try anchored alignment first (handles repeated choruses better)
                synced = align_lyrics_to_words_anchored(lyric_lines, words)
                if is_bad_lyric_sync(synced):
                    synced = align_lyrics_to_words_dp(lyric_lines, words)
            else:
                synced = align_lyrics_to_segments(lyric_lines, segments)

            # If the sync looks degenerate, do NOT fail the whole job.
            # We'll just skip writing synced lyrics and proceed with stems/video.
            if is_bad_lyric_sync(synced):
                times = [float(x.get('time', 0) or 0) for x in synced]
                deltas = [times[i]-times[i-1] for i in range(1, len(times))]
                max_gap = max(deltas) if deltas else None
                min_delta = min(deltas) if deltas else None
                print(f"[warn] Bad lyric sync detected; skipping lyrics (max_gap={max_gap}, min_delta={min_delta}, words={len(words)}, model={model_name}).")
                synced = []

            # Normalize: enforce monotonically increasing times.
            prev_t = -1.0
            for ln in synced:
                try:
                    t = float(ln.get("time", 0) or 0)
                except Exception:
                    t = 0.0
                if t <= prev_t:
                    t = prev_t + 0.05
                ln["time"] = t
                prev_t = t

            synced_path = os.path.join(output_dir, "lyrics.synced.json")
            import json
            with open(synced_path, "w", encoding="utf-8") as f:
                json.dump({"video_id": vid, "source": "ytmusic+whisper", "lines": synced}, f, ensure_ascii=False)

        # write to db for fast re-use
        if vid and os.path.exists(vocals_path) and os.path.exists(no_vocals_path):
            title = None
            artist = None
            cover_url = None
            try:
                wp = YTM.get_watch_playlist(vid)
                if wp.get("tracks"):
                    t = wp["tracks"][0]
                    title = t.get("title")
                    if t.get("artists"):
                        artist = t["artists"][0].get("name")
                    thumbs = t.get("thumbnail") or []
                    if thumbs:
                        cover_url = sorted(thumbs, key=lambda x: (x.get("width", 0), x.get("height", 0)))[-1].get("url")
            except Exception:
                pass

            upsert_track(
                DATA_DIR,
                video_id=vid,
                source_url=canon,
                task_id=task_id,
                title=title,
                artist=artist,
                cover_url=cover_url,
                vocals_path=vocals_path,
                no_vocals_path=no_vocals_path,
                lyrics_json_path=synced_path,
            )

        set_task(task_id, status="completed", stage="completed", progress=100)
        push_history(task_id, url, "completed", "completed", 100)
    except Exception as e:
        if str(e) == "cancelled" or "cancelled" in str(e).lower():
            set_task(task_id, status="cancelled", stage="cancelled", progress=0, error="cancelled")
            push_history(task_id, url, "cancelled", "cancelled", 0)
            return
        set_task(task_id, status="error", stage="error", error=str(e) + "\n" + traceback.format_exc())
        push_history(task_id, url, "error", "error", 0)
        raise


@app.get("/status/{task_id}")
async def get_status(task_id: str):
    output_dir, vocals_file, no_vocals_file, video_file = output_paths(task_id)

    # fallback: if someone already created a test_run, allow task_id=test_run
    if task_id == "test_run":
        output_dir = os.path.join(DATA_DIR, "test_run")
        vocals_file = os.path.join(output_dir, "htdemucs", "original", "vocals.wav")
        no_vocals_file = os.path.join(output_dir, "htdemucs", "original", "no_vocals.wav")
        video_file = os.path.join(output_dir, "video.mp4")

    stems = find_stems(output_dir)

    with TASKS_LOCK:
        meta = TASKS.get(task_id, {}).copy()

    # prefer multi-stem completion if available
    if stems:
        stage = meta.get("stage")

        # If error happened during post-processing/resync, surface it even though audio exists
        if stage == "error":
            rel = os.path.relpath(os.path.join(output_dir, os.environ.get("HOLY_DEMUCS_MODEL", "htdemucs_6s"), "original"), DATA_DIR)
            if not os.path.isdir(os.path.join(DATA_DIR, rel)):
                # fallback to whatever folder we discovered
                # stems paths are absolute; make relative to DATA_DIR for StaticFiles
                rel = os.path.relpath(os.path.dirname(next(iter(stems.values()))), DATA_DIR)

            return {
                "status": "error",
                "stage": "error",
                "progress": meta.get("progress", 0),
                "error": meta.get("error"),
                "stems": {k: f"/audio/{os.path.relpath(v, DATA_DIR)}" for k, v in stems.items()},
                "video": f"/audio/{os.path.basename(output_dir)}/video.mp4" if os.path.exists(video_file) else None,
            }

        # If we have audio but lyric-sync is still running, keep status=processing.
        if stage and stage != "completed":
            return {
                "status": "processing",
                "stage": stage,
                "progress": meta.get("progress", 0),
                "stems": {k: f"/audio/{os.path.relpath(v, DATA_DIR)}" for k, v in stems.items()},
            }

        return {
            "status": "completed",
            "stage": meta.get("stage", "completed"),
            "progress": meta.get("progress", 100),
            "stems": {k: f"/audio/{os.path.relpath(v, DATA_DIR)}" for k, v in stems.items()},
            "video": f"/audio/{os.path.basename(output_dir)}/video.mp4" if os.path.exists(video_file) else None,
        }

    if meta.get("stage") == "error":
        return {"status": "error", **meta}

    return {
        "status": "processing",
        "stage": meta.get("stage", "processing"),
        "progress": meta.get("progress", 0),
    }

@app.get("/meta/{video_id}")
async def meta(video_id: str):
    """Basic metadata (title/artist/album cover) from YouTube Music."""
    try:
        wp = YTM.get_watch_playlist(video_id)
        if not wp.get("tracks"):
            raise HTTPException(status_code=404, detail="No track metadata")
        t = wp["tracks"][0]
        thumbs = t.get("thumbnail") or []
        cover = None
        if thumbs:
            # pick largest
            cover = sorted(thumbs, key=lambda x: (x.get("width", 0), x.get("height", 0)))[-1].get("url")
        artist = None
        if t.get("artists"):
            artist = t["artists"][0].get("name")
        return {
            "video_id": video_id,
            "title": t.get("title"),
            "artist": artist,
            "cover_url": cover,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"No metadata available: {e}")


@app.get("/queue")
async def queue_status():
    with QUEUE_LOCK:
        items = [{"task_id": t[2], "url": t[0]} for t in list(QUEUE)]
        current = CURRENT
    return {"current": current, "items": items}


@app.get("/jobs")
async def jobs():
    """Unified jobs view: current + queue + recent."""
    with QUEUE_LOCK:
        q_items = [{"task_id": t[2], "url": t[0]} for t in list(QUEUE)]
        cur = CURRENT

    # attach task meta if present
    def meta_for(tid: str):
        with TASKS_LOCK:
            return TASKS.get(tid, {}).copy()

    current = None
    if cur and cur.get("task_id"):
        tid = cur.get("task_id")
        m = meta_for(tid)
        current = {**cur, **m}

    queue = []
    for it in q_items:
        tid = it["task_id"]
        m = meta_for(tid)
        queue.append({**it, **m})

    recent = list(JOB_HISTORY)
    return {"current": current, "queue": queue, "recent": recent}


@app.post("/jobs/{task_id}/cancel")
async def cancel_job(task_id: str):
    """Best-effort cancel: remove from queue and/or terminate running proc."""
    CANCELLED.add(task_id)

    removed = False
    with QUEUE_COND:
        kept = deque([x for x in QUEUE if x[2] != task_id])
        removed = len(kept) != len(QUEUE)
        QUEUE.clear()
        QUEUE.extend(kept)

    # terminate process if running
    with RUNNING_LOCK:
        proc = RUNNING_PROCS.get(task_id)
    if proc is not None:
        try:
            proc.terminate()
        except Exception:
            pass

    set_task(task_id, status="cancelled", stage="cancelled", progress=0, error="cancelled")
    return {"ok": True, "removed_from_queue": removed}


@app.get("/library")
async def library(limit: int = 50):
    """List cached tracks (for UI)."""
    items = list_tracks(DATA_DIR, limit=limit)
    # keep response small
    out = []
    for it in items:
        out.append({
            "video_id": it.get("video_id"),
            "source_url": it.get("source_url"),
            "task_id": it.get("task_id"),
            "title": it.get("title"),
            "artist": it.get("artist"),
            "cover_url": it.get("cover_url"),
            "has_synced_lyrics": bool(it.get("lyrics_json_path")) and os.path.exists(it.get("lyrics_json_path")),
            "updated_at": it.get("updated_at"),
        })
    return {"items": out}


# -----------------------------
# Presets (library for UI)
# -----------------------------

@app.get("/presets")
async def presets(limit: int = 100, sort: str = "recent"):
    items = list_presets(DATA_DIR, limit=limit)

    # Sort options: recent (default), bpm_asc, bpm_desc, key_asc, key_desc
    if sort == "bpm_asc":
        items = sorted(items, key=lambda x: (x.get("bpm") is None, float(x.get("bpm") or 0)))
    elif sort == "bpm_desc":
        items = sorted(items, key=lambda x: (x.get("bpm") is None, -float(x.get("bpm") or 0)))
    elif sort in ("key_asc", "key_desc"):
        order = {"C": 0, "C#": 1, "D": 2, "D#": 3, "E": 4, "F": 5, "F#": 6, "G": 7, "G#": 8, "A": 9, "A#": 10, "B": 11}

        def key_tuple(x):
            tonic = x.get("key_tonic")
            mode = x.get("key_mode")
            # mode order: major first, then minor
            mo = 0 if str(mode) == "major" else 1 if str(mode) == "minor" else 2
            to = order.get(str(tonic), 99)
            missing = tonic is None or mode is None
            return (missing, to, mo)

        items = sorted(items, key=key_tuple, reverse=(sort == "key_desc"))
    # else: keep DB order (updated_at desc)

    out = []

    async def fetch_oembed(vid: str):
        try:
            import aiohttp

            canonical = f"https://www.youtube.com/watch?v={vid}"
            url = f"https://www.youtube.com/oembed?url={canonical}&format=json"
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                    if resp.status != 200:
                        return None
                    return await resp.json()
        except Exception:
            return None

    # quick lookup of queue/current status by task_id
    with QUEUE_LOCK:
        q_tids = {t[2] for t in list(QUEUE)}
        cur_tid = (CURRENT or {}).get("task_id") if CURRENT else None

    for it in items:
        vid = it.get("video_id")

        # Fill missing title/thumbnail via YouTube oEmbed (best-effort), and persist to DB.
        if vid and (not it.get("title") or not it.get("thumbnail_url")):
            oe = await fetch_oembed(str(vid))
            if oe:
                new_title = (oe.get("title") or "").strip() or None
                new_thumb = (oe.get("thumbnail_url") or "").strip() or None
                if new_title and not it.get("title"):
                    it["title"] = new_title
                if new_thumb and not it.get("thumbnail_url"):
                    it["thumbnail_url"] = new_thumb

                try:
                    upsert_preset(
                        DATA_DIR,
                        id=it.get("id"),
                        name=it.get("name") or (it.get("title") or str(vid)),
                        source_url=it.get("source_url") or canonical_watch_url(str(vid)),
                        video_id=str(vid),
                        title=it.get("title"),
                        thumbnail_url=it.get("thumbnail_url"),
                        bpm=it.get("bpm"),
                        semitones=int(it.get("semitones") or 0),
                        master_volume=float(it.get("master_volume") or 1.0),
                        vocal_volume=float(it.get("vocal_volume") or 1.0),
                    )
                except Exception:
                    pass

        tid = hashlib.sha1(str(vid).encode("utf-8")).hexdigest()[:16] if vid else None
        video_path = os.path.join(DATA_DIR, str(tid), "video.mp4") if tid else ""

        meta = {}
        if tid:
            with TASKS_LOCK:
                meta = TASKS.get(tid, {}).copy()

        stage = meta.get("stage")
        progress = meta.get("progress")
        status = meta.get("status")

        out.append(
            {
                **it,
                "task_id": tid,
                "video_ready": bool(tid and os.path.exists(video_path)),
                "job": {
                    "status": status,
                    "stage": stage,
                    "progress": progress,
                    "is_current": bool(tid and cur_tid and tid == cur_tid),
                    "is_queued": bool(tid and tid in q_tids),
                },
            }
        )

    return {"items": out}


@app.get("/presets/{preset_id}")
async def presets_get_one(preset_id: str):
    try:
        from preset_endpoints import get_preset_or_404

        return {"ok": True, "item": get_preset_or_404(DATA_DIR, preset_id)}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"preset lookup failed: {e}")


@app.post("/presets")
async def presets_upsert(p: PresetIn):
    upsert_preset(
        DATA_DIR,
        id=p.id,
        name=p.name,
        source_url=p.source_url,
        video_id=p.video_id,
        title=p.title,
        thumbnail_url=p.thumbnail_url,
        bpm=p.bpm,
        key_tonic=p.key_tonic,
        key_mode=p.key_mode,
        semitones=p.semitones,
        master_volume=p.master_volume,
        vocal_volume=p.vocal_volume,
    )
    return {"ok": True}


@app.post("/presets/{preset_id}/meta")
async def presets_meta(
    preset_id: str,
    payload: dict = Body(default={}),  # noqa: B008
):
    """Patch preset metadata without overwriting the user-defined name."""
    try:
        from presets_meta import update_preset_meta

        bpm = payload.get("bpm")
        key_tonic = payload.get("key_tonic")
        key_mode = payload.get("key_mode")
        title = payload.get("title")
        thumbnail_url = payload.get("thumbnail_url")

        update_preset_meta(
            DATA_DIR,
            id=preset_id,
            bpm=float(bpm) if bpm is not None else None,
            key_tonic=str(key_tonic) if key_tonic is not None else None,
            key_mode=str(key_mode) if key_mode is not None else None,
            title=str(title) if title is not None else None,
            thumbnail_url=str(thumbnail_url) if thumbnail_url is not None else None,
        )
        return {"ok": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"meta update failed: {e}")


@app.post("/bpm/{task_id}/detect")
async def bpm_detect(task_id: str):
    out_dir, _v, _m, _video = output_paths(task_id)
    stems = find_stems(out_dir)
    if not stems:
        raise HTTPException(status_code=404, detail="Stems not ready")

    # Prefer drums stem for detection
    wav = stems.get("drums") or stems.get("other") or stems.get("vocals")
    if not wav:
        raise HTTPException(status_code=404, detail="No stem available")

    source = "drums" if stems.get("drums") else ("other" if stems.get("other") else "vocals")

    try:
        from bpm_detect import detect_bpm_and_offset_from_wav

        bpm, start_offset = detect_bpm_and_offset_from_wav(wav)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"BPM detect failed: {e}")

    if bpm is None:
        raise HTTPException(status_code=422, detail="Could not detect BPM")

    return {"ok": True, "bpm": bpm, "source": source, "start_offset": start_offset}


@app.post("/key/{task_id}/detect")
async def key_detect(task_id: str):
    out_dir, _v, _m, _video = output_paths(task_id)
    stems = find_stems(out_dir)
    if not stems:
        raise HTTPException(status_code=404, detail="Stems not ready")

    # Prefer harmonic content: other/piano/guitar, avoid drums
    wav = stems.get("other") or stems.get("piano") or stems.get("guitar") or stems.get("vocals")
    if not wav:
        raise HTTPException(status_code=404, detail="No stem available")

    source = "other" if stems.get("other") else ("piano" if stems.get("piano") else ("guitar" if stems.get("guitar") else "vocals"))

    try:
        from key_detect import detect_key_from_wav

        res = detect_key_from_wav(wav)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Key detect failed: {e}")

    if not res:
        raise HTTPException(status_code=422, detail="Could not detect key")

    tonic, mode, conf = res
    return {"ok": True, "tonic": tonic, "mode": mode, "confidence": conf, "source": source}


@app.post("/export/{task_id}")
async def export_mix(task_id: str, payload: dict = Body(default={})):  # noqa: B008
    """Export current mix settings to an mp3 and return a URL under /audio.

    Payload example:
      {
        "mode": {"kind":"master"} | {"kind":"vocal"} | {"kind":"instrument","instrument":"drums"},
        "masterVolume": 100,
        "instrumentalMaster": 100,
        "stemVolumes": {"vocals":100, "drums":100, ...},
        "focusVol": 100,
        "otherInstVol": 100,
        "vocalsVol": 100,
        "instrumentsVol": 100
      }
    """
    out_dir, _v, _m, _video = output_paths(task_id)
    stems = find_stems(out_dir)
    if not stems:
        raise HTTPException(status_code=404, detail="Stems not ready")

    mode = payload.get("mode") or {"kind": "master"}
    kind = str(mode.get("kind") or "master")
    instrument = str(mode.get("instrument") or "")

    master = float(payload.get("masterVolume", 100)) / 100.0
    master = max(0.0, min(1.0, master))

    # Compute per-stem linear gains
    gains: dict[str, float] = {}

    if kind == "master":
        inst_master = float(payload.get("instrumentalMaster", 100)) / 100.0
        inst_master = max(0.0, min(1.0, inst_master))
        stem_vols = payload.get("stemVolumes") or {}
        for nm in stems.keys():
            sv = float(stem_vols.get(nm, 100)) / 100.0
            sv = max(0.0, min(1.0, sv))
            if nm == "vocals":
                gains[nm] = master * sv
            else:
                gains[nm] = master * inst_master * sv

    elif kind == "vocal":
        vocals_vol = max(0.0, min(1.0, float(payload.get("vocalsVol", 100)) / 100.0))
        inst_vol = max(0.0, min(1.0, float(payload.get("instrumentsVol", 100)) / 100.0))
        for nm in stems.keys():
            if nm == "vocals":
                gains[nm] = master * vocals_vol
            else:
                gains[nm] = master * inst_vol

    elif kind == "instrument":
        focus_vol = max(0.0, min(1.0, float(payload.get("focusVol", 100)) / 100.0))
        other_vol = max(0.0, min(1.0, float(payload.get("otherInstVol", 100)) / 100.0))
        vocals_vol = max(0.0, min(1.0, float(payload.get("vocalsVol", 100)) / 100.0))
        for nm in stems.keys():
            if nm == "vocals":
                gains[nm] = master * vocals_vol
            elif instrument and nm == instrument:
                gains[nm] = master * focus_vol
            else:
                gains[nm] = master * other_vol
    else:
        raise HTTPException(status_code=400, detail="Invalid mode")

    # Output path (stable for same mode)
    safe_kind = kind.replace("/", "_")
    safe_inst = instrument.replace("/", "_")
    tag = safe_kind + (f"_{safe_inst}" if safe_inst else "")
    export_dir = os.path.join(DATA_DIR, task_id, "exports")
    out_mp3 = os.path.join(export_dir, f"mix_{tag}.mp3")

    try:
        from export_mix import mixdown_to_mp3

        mixdown_to_mp3(stems, gains, out_mp3)
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"ffmpeg failed: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"export failed: {e}")

    return {"ok": True, "mp3": f"/audio/{task_id}/exports/mix_{tag}.mp3"}


@app.delete("/presets/{preset_id}")
async def presets_delete(preset_id: str):
    """Delete a preset and its associated cached files (video/audio/transpose) if present."""
    deleted = delete_preset(DATA_DIR, preset_id)

    if deleted:
        vid = deleted.get("video_id")
        if vid:
            # remove track DB row too (cached separation)
            try:
                delete_track(DATA_DIR, str(vid))
            except Exception:
                pass

            # move files aside (safer than rm) under data/_deleted
            try:
                task_id = hashlib.sha1(str(vid).encode("utf-8")).hexdigest()[:16]
                src_dir = os.path.join(DATA_DIR, task_id)
                if os.path.exists(src_dir):
                    import time, shutil
                    dst_root = os.path.join(DATA_DIR, "_deleted")
                    os.makedirs(dst_root, exist_ok=True)
                    dst_dir = os.path.join(dst_root, f"{task_id}_{int(time.time())}")
                    shutil.move(src_dir, dst_dir)
            except Exception:
                pass

    return {"ok": True, "deleted": bool(deleted)}


# -----------------------------
# Transpose (offline, cached)
# -----------------------------

def transposed_paths(task_id: str, semitones: int):
    out_dir = os.path.join(DATA_DIR, task_id, f"transpose_{semitones:+d}")
    vocals = os.path.join(out_dir, "vocals.wav")
    no_vocals = os.path.join(out_dir, "no_vocals.wav")
    return out_dir, vocals, no_vocals


def transposed_stems_dir(task_id: str, semitones: int) -> str:
    return os.path.join(DATA_DIR, task_id, f"transpose_{semitones:+d}")


def ensure_transposed(task_id: str, semitones: int):
    base_dir, base_v, base_m, _video_file = output_paths(task_id)
    if not os.path.exists(base_v) or not os.path.exists(base_m):
        raise HTTPException(status_code=404, detail="Base audio not ready")

    out_dir, tv, tm = transposed_paths(task_id, semitones)
    os.makedirs(out_dir, exist_ok=True)

    if os.path.exists(tv) and os.path.exists(tm):
        return out_dir, tv, tm

    # Pitch factor (2^(n/12))
    factor = 2 ** (float(semitones) / 12.0)
    inv = 1.0 / factor

    import subprocess

    def run(src: str, dst: str):
        # Prefer Rubber Band (R3 engine + formant preservation) for much better quality.
        rb = "/opt/homebrew/bin/rubberband"
        if os.path.exists(rb):
            # -3 fine engine, -F formant, --centre-focus for stereo, -q quiet
            subprocess.run(
                [
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
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return

        # Fallback: ffmpeg asetrate+atempo (faster, lower quality)
        # Keep within atempo supported range (0.5-2.0) by chaining if needed
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
                "-ac",
                "1",
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

    run(base_v, tv)
    run(base_m, tm)
    return out_dir, tv, tm


@app.post("/transpose/{task_id}")
async def transpose(task_id: str, semitones: int = 0):
    # UI supports only +/-6 to keep quality reasonable and avoid heavy processing
    if abs(int(semitones)) > 6:
        raise HTTPException(status_code=400, detail="semitones out of range (-6..6)")

    out_dir, _v, _m, video_file = output_paths(task_id)
    base_stems = find_stems(out_dir)

    # Prefer multi-stem transpose when available
    if base_stems:
        if semitones == 0:
            stems_rel = {k: os.path.relpath(v, DATA_DIR) for k, v in base_stems.items()}
        else:
            from transpose_stems import ensure_transposed_stems

            set_task(task_id, status="processing", stage="transposing", progress=0)

            def _prog(p: int):
                set_task(task_id, status="processing", stage="transposing", progress=int(p))

            _od, stems_rel = await run_in_threadpool(
                ensure_transposed_stems,
                data_dir=DATA_DIR,
                task_id=task_id,
                semitones=int(semitones),
                stems=base_stems,
                progress_cb=_prog,
            )

            set_task(task_id, status="completed", stage="completed", progress=100)

        # return as API-style urls
        stems_api = {k: f"/audio/{v}" for k, v in stems_rel.items()}
        return {
            "task_id": task_id,
            "semitones": int(semitones),
            "stems": stems_api,
            "video": f"/audio/{task_id}/video.mp4" if os.path.exists(video_file) else None,
        }

    # Fallback: legacy 2-stem transpose
    if semitones == 0:
        output_dir, vocals_file, no_vocals_file, _video_file = output_paths(task_id)
        if not (os.path.exists(vocals_file) and os.path.exists(no_vocals_file)):
            raise HTTPException(status_code=404, detail="Base audio not ready")
        return {
            "task_id": task_id,
            "semitones": 0,
            "vocals": f"/audio/{os.path.basename(output_dir)}/htdemucs/original/vocals.wav",
            "no_vocals": f"/audio/{os.path.basename(output_dir)}/htdemucs/original/no_vocals.wav",
        }

    # ensure_transposed runs rubberband/ffmpeg and can block; run it off the event loop
    out_dir2, tv, tm = await run_in_threadpool(ensure_transposed, task_id, int(semitones))
    rel = os.path.relpath(out_dir2, DATA_DIR)
    return {
        "task_id": task_id,
        "semitones": int(semitones),
        "vocals": f"/audio/{rel}/vocals.wav",
        "no_vocals": f"/audio/{rel}/no_vocals.wav",
    }


@app.post("/resync/{video_id}")
async def resync_lyrics(video_id: str, background_tasks: BackgroundTasks, model: str = "medium"):
    """Re-run whisper+alignment for an already-cached track."""
    cached = get_track(DATA_DIR, video_id)
    if not cached:
        raise HTTPException(status_code=404, detail="Track not found in cache")

    task_id = cached.get("task_id") or task_id_for_url(video_id)
    output_dir = os.path.join(DATA_DIR, task_id)
    vocals_path = cached.get("vocals_path")

    if not vocals_path or not os.path.exists(vocals_path):
        raise HTTPException(status_code=400, detail="Missing vocals file for this track")

    set_task(task_id, stage="transcribing", progress=0)
    background_tasks.add_task(_resync_task, video_id, task_id, output_dir, vocals_path, model, cached)
    return {"task_id": task_id, "status": "processing"}


def _resync_task(video_id: str, task_id: str, output_dir: str, vocals_path: str, model_name: str, cached: dict):
    try:
        set_task(task_id, stage="transcribing", progress=5)

        # ytmusic lyrics
        lyric_lines = []
        wp = YTM.get_watch_playlist(video_id)
        lyrics_id = wp.get("lyrics")
        if lyrics_id:
            ly = YTM.get_lyrics(lyrics_id)
            text = (ly.get("lyrics") or "").strip()
            lyric_lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

        if not lyric_lines:
            raise Exception("No YT Music lyrics text")

        # Captions-based sync (preferred for singing when whisper word-timestamps are unreliable)
        if model_name == "captions":
            set_task(task_id, stage="fetching_captions", progress=20)
            phrases = captions_phrases(video_id)
            set_task(task_id, stage="aligning_lyrics", progress=70)
            synced = align_lyrics_to_captions_lines(lyric_lines, phrases)

            # sanity: require timestamps to be monotonic and not too gappy
            if is_bad_lyric_sync(synced):
                times = [float(x.get('time', 0) or 0) for x in synced]
                deltas = [times[i] - times[i-1] for i in range(1, len(times))]
                max_gap = max(deltas) if deltas else None
                min_delta = min(deltas) if deltas else None
                raise Exception(
                    f"Bad captions sync (max_gap={max_gap}, min_delta={min_delta}, phrases={len(phrases)})."
                )

            # Normalize monotonic
            prev_t = -1.0
            for ln in synced:
                t = float(ln.get('time', 0) or 0)
                if t <= prev_t:
                    t = prev_t + 0.05
                ln['time'] = t
                prev_t = t

            synced_path = os.path.join(output_dir, "lyrics.synced.json")
            import json
            with open(synced_path, "w", encoding="utf-8") as f:
                json.dump({"video_id": video_id, "source": "ytcaptions", "lines": synced}, f, ensure_ascii=False)

            upsert_track(
                DATA_DIR,
                video_id=video_id,
                source_url=cached.get("source_url") or f"https://www.youtube.com/watch?v={video_id}",
                task_id=task_id,
                title=cached.get("title"),
                artist=cached.get("artist"),
                cover_url=cached.get("cover_url"),
                vocals_path=cached.get("vocals_path"),
                no_vocals_path=cached.get("no_vocals_path"),
                lyrics_json_path=synced_path,
            )

            set_task(task_id, stage="completed", progress=100)
            return

        # Prefer ORIGINAL mix for alignment (singing + timing cues are often clearer than demucs vocals)
        set_task(task_id, stage="transcribing", progress=15)
        original_path = os.path.join(output_dir, "original.wav")
        src_audio = original_path if os.path.exists(original_path) else vocals_path

        # Preprocess for Whisper/WhisperX (mono 16k + mild filtering)
        whisper_in = os.path.join(output_dir, "whisper_in.wav")
        try:
            import subprocess
            subprocess.run(
                [
                    "/opt/homebrew/bin/ffmpeg",
                    "-y",
                    "-i",
                    src_audio,
                    "-af",
                    "highpass=f=80,lowpass=f=9000,dynaudnorm",
                    "-ar",
                    "16000",
                    "-ac",
                    "1",
                    whisper_in,
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception:
            whisper_in = src_audio

        # initial prompt: give Whisper a hint of the actual lyrics text
        prompt = "\n".join(lyric_lines[:30])
        if len(prompt) > 1200:
            prompt = prompt[:1200]

        # First try whisperx word alignment (often better on original mix)
        set_task(task_id, stage="transcribing", progress=25)
        words = whisperx_words(whisper_in, model_name=model_name, lang="en")

        segments = []
        if not words:
            # fallback: openai-whisper (segments + optional word timestamps)
            m = get_whisper_model(model_name)
            result = m.transcribe(
                whisper_in,
                fp16=False,
                language="en",
                task="transcribe",
                temperature=0.0,
                condition_on_previous_text=False,
                initial_prompt=prompt,
                carry_initial_prompt=True,
                word_timestamps=True,
                verbose=False,
            )
            segments = [{"start": s["start"], "end": s["end"], "text": s["text"]} for s in (result.get("segments") or [])]
            for seg in (result.get("segments") or []):
                for w in seg.get("words") or []:
                    words.append({"start": w.get("start"), "end": w.get("end"), "word": w.get("word")})

        # Debug artifacts (so we stop guessing)
        try:
            import json
            with open(os.path.join(output_dir, "debug.words.json"), "w", encoding="utf-8") as f:
                json.dump({"video_id": video_id, "model": model_name, "count": len(words), "words": words[:5000]}, f, ensure_ascii=False)
            with open(os.path.join(output_dir, "debug.segments.json"), "w", encoding="utf-8") as f:
                json.dump({"video_id": video_id, "model": model_name, "segments": segments[:2000]}, f, ensure_ascii=False)
        except Exception:
            pass

        set_task(task_id, stage="transcribing", progress=70)

        set_task(task_id, stage="aligning_lyrics", progress=85)
        if words:
            synced = align_lyrics_to_words_anchored(lyric_lines, words)
            if is_bad_lyric_sync(synced):
                synced = align_lyrics_to_words_dp(lyric_lines, words)
        else:
            # segments from original mix tend to be more stable than demucs vocals
            synced = align_lyrics_to_segments(lyric_lines, segments, search_window=40)

        set_task(task_id, stage="aligning_lyrics", progress=95)

        # If the sync looks degenerate, FAIL (do not silently fall back)
        if is_bad_lyric_sync(synced):
            times = [float(x.get('time', 0) or 0) for x in synced]
            deltas = [times[i] - times[i-1] for i in range(1, len(times))]
            max_gap = max(deltas) if deltas else None
            min_delta = min(deltas) if deltas else None
            raise Exception(
                f"Bad lyric sync detected (max_gap={max_gap}, min_delta={min_delta}, words={len(words)}, model={model_name})."
            )

        # Normalize: enforce monotonically increasing times (avoid UI sticking)
        prev_t = -1.0
        for ln in synced:
            try:
                t = float(ln.get("time", 0) or 0)
            except Exception:
                t = 0.0
            if t <= prev_t:
                t = prev_t + 0.05
            ln["time"] = t
            prev_t = t

        synced_path = os.path.join(output_dir, "lyrics.synced.json")
        import json
        with open(synced_path, "w", encoding="utf-8") as f:
            json.dump({"video_id": video_id, "source": f"ytmusic+whisper-{model_name}", "lines": synced}, f, ensure_ascii=False)

        # update DB pointer
        upsert_track(
            DATA_DIR,
            video_id=video_id,
            source_url=cached.get("source_url") or f"https://www.youtube.com/watch?v={video_id}",
            task_id=task_id,
            title=cached.get("title"),
            artist=cached.get("artist"),
            cover_url=cached.get("cover_url"),
            vocals_path=cached.get("vocals_path"),
            no_vocals_path=cached.get("no_vocals_path"),
            lyrics_json_path=synced_path,
        )

        set_task(task_id, stage="completed", progress=100)
    except Exception as e:
        set_task(task_id, stage="error", error=str(e) + "\n" + traceback.format_exc())


@app.get("/lyrics/{video_id}")
async def lyrics(video_id: str, source: str = "ytmusic"):
    """Fetch lyrics.

    source=ytmusic  -> YouTube Music lyrics (usually NOT time-synced)
    source=yt       -> YouTube transcript captions (often auto-translated / not true lyrics)

    Returns: { video_id, source, hasTimestamps, lines: [{time?, text}] }
    """

    if source == "ytmusic":
        # if we previously generated synced lyrics via whisper, return that.
        cached = get_track(DATA_DIR, video_id)
        if cached and cached.get("lyrics_json_path") and os.path.exists(cached.get("lyrics_json_path")):
            import json
            with open(cached.get("lyrics_json_path"), "r", encoding="utf-8") as f:
                data = json.load(f)
            return {
                "video_id": video_id,
                "source": data.get("source", "ytmusic+whisper"),
                "hasTimestamps": True,
                "lines": data.get("lines", []),
            }

        try:
            wp = YTM.get_watch_playlist(video_id)
            lyrics_id = wp.get("lyrics")
            if not lyrics_id:
                raise HTTPException(status_code=404, detail="No YT Music lyrics id for this track")
            ly = YTM.get_lyrics(lyrics_id)
            text = (ly.get("lyrics") or "").strip()
            if not text:
                raise HTTPException(status_code=404, detail="No lyrics text returned")

            # No timestamps exposed here in most cases
            lines = [{"text": ln} for ln in text.splitlines() if ln.strip()]
            return {
                "video_id": video_id,
                "source": "ytmusic",
                "hasTimestamps": bool(ly.get("hasTimestamps")),
                "lines": lines,
            }
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=404, detail=f"No YT Music lyrics available: {e}")

    if source == "yt":
        try:
            transcript_list = YT.list(video_id)

            transcript = None
            for lang in ["ko", "en"]:
                try:
                    transcript = transcript_list.find_transcript([lang])
                    break
                except Exception:
                    pass

            if transcript is None:
                transcript = transcript_list.find_transcript([t.language_code for t in transcript_list])

            items = transcript.fetch()
            lines = []
            for snip in getattr(items, "snippets", []) or []:
                t = float(getattr(snip, "start", 0))
                txt = (getattr(snip, "text", "") or "").replace("\n", " ").strip()
                if txt:
                    lines.append({"time": t, "text": txt})
            return {"video_id": video_id, "source": "yt", "hasTimestamps": True, "lines": lines}
        except Exception as e:
            raise HTTPException(status_code=404, detail=f"No transcript available: {e}")

    raise HTTPException(status_code=400, detail="source must be ytmusic or yt")


# Optional: serve the static frontend bundle from the same backend (desktop packaging).
# Enabled by HOLY_SERVE_WEB=1.
try:
    if os.environ.get("HOLY_SERVE_WEB") == "1":
        web_dir = os.environ.get("HOLY_WEB_DIR")
        if not web_dir:
            # dev/workspace default: ../frontend/out
            web_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "frontend", "out"))
        if os.path.isdir(web_dir):
            # Mount last so API routes still work.
            app.mount(
                "/",
                StaticFiles(directory=web_dir, html=True),
                name="web",
            )
except Exception:
    pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
