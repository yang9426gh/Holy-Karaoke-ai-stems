"use client";

import React, { useEffect, useMemo, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import { Play, Pause } from "lucide-react";

const BACKEND = process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8011";

type Status = "idle" | "processing" | "completed";

type StemsMap = Record<string, string>; // stemName -> absolute URL

// Jobs UI removed

type Preset = {
  id: string;
  name: string;
  source_url: string;
  video_id: string;
  title?: string | null;
  thumbnail_url?: string | null;
  bpm?: number | null;
  key_tonic?: string | null;
  key_mode?: string | null;
  task_id?: string | null;
  video_ready?: boolean;
};

type StemLoadState =
  | { status: "idle" }
  | { status: "loading"; loaded: number; total?: number }
  | { status: "decoding" }
  | { status: "ready"; duration: number }
  | { status: "error"; error: string };

const decodedStemCache = new Map<string, Record<string, AudioBuffer>>();

function stemIcon(stem: string) {
  const s = (stem || "").toLowerCase();
  if (s === "vocals" || s === "vocal") return "🎤";
  if (s === "drums" || s === "drum") return "🥁";
  if (s === "bass") return "🎸";
  if (s === "guitar") return "🎸";
  if (s === "piano") return "🎹";
  if (s === "other" || s === "instruments" || s === "instrumental") return "🎶";
  if (s === "master") return "🎛️";
  if (s === "click" || s === "metronome") return "🔔";
  return "🎵";
}

function VerticalFader({
  value,
  onChange,
  disabled,
  height = 220,
}: {
  value: number;
  onChange: (v: number) => void;
  disabled?: boolean;
  height?: number;
}) {
  const trackRef = React.useRef<HTMLDivElement | null>(null);

  const clamp = (v: number) => Math.max(0, Math.min(100, v));

  const setFromClientY = (clientY: number) => {
    const el = trackRef.current;
    if (!el) return;
    const r = el.getBoundingClientRect();
    const y = Math.max(r.top, Math.min(r.bottom, clientY));
    const t = (y - r.top) / Math.max(1, r.height); // 0..1 top->bottom
    const v = clamp((1 - t) * 100);
    onChange(v);
  };

  const pct = clamp(Number(value || 0));
  const knobY = (1 - pct / 100) * height;

  return (
    <div className="flex flex-col items-center select-none">
      <div
        ref={trackRef}
        className={`relative w-8 rounded-full border ${disabled ? "border-zinc-900 bg-zinc-950" : "border-zinc-700 bg-zinc-900"}`}
        style={{ height }}
        onPointerDown={(e) => {
          if (disabled) return;
          (e.currentTarget as any).setPointerCapture?.(e.pointerId);
          setFromClientY(e.clientY);
        }}
        onPointerMove={(e) => {
          if (disabled) return;
          if ((e.buttons & 1) === 0) return;
          setFromClientY(e.clientY);
        }}
      >
        <div className="absolute inset-x-0 bottom-0 rounded-full bg-blue-600/30" style={{ height: `${pct}%` }} />
        <div
          className={`absolute left-1/2 -translate-x-1/2 w-10 h-6 rounded-full border shadow ${disabled ? "border-zinc-800 bg-zinc-900" : "border-zinc-600 bg-zinc-200"}`}
          style={{ top: Math.max(0, Math.min(height - 24, knobY - 12)) }}
        />
      </div>
      <div className="mt-2 text-xs font-mono text-zinc-300">{Math.round(pct)}%</div>
    </div>
  );
}

function extractYouTubeId(input: string): string {
  try {
    const u = new URL(input);
    if (u.hostname.includes("youtu.be")) return u.pathname.replace("/", "");
    return u.searchParams.get("v") || "";
  } catch {
    const m = input.match(/v=([^&]+)/);
    if (m?.[1]) return m[1];
    const parts = input.split("/");
    return (parts[parts.length - 1] || "").split("?")[0];
  }
}

function clampTime(t: number, duration?: number) {
  const d = Number(duration);
  if (!isFinite(d) || d <= 0) return Math.max(0, t);
  return Math.max(0, Math.min(t, d - 0.01));
}

async function fetchArrayBufferWithProgress(
  url: string,
  onProgress: (loaded: number, total?: number) => void,
  signal?: AbortSignal
): Promise<ArrayBuffer> {
  const resp = await fetch(url, { signal });
  if (!resp.ok) throw new Error(`HTTP ${resp.status}`);

  const total = Number(resp.headers.get("content-length") || "");
  if (!resp.body) {
    const ab = await resp.arrayBuffer();
    onProgress(ab.byteLength, isFinite(total) ? total : undefined);
    return ab;
  }

  const reader = resp.body.getReader();
  const chunks: Uint8Array[] = [];
  let loaded = 0;
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    if (value) {
      chunks.push(value);
      loaded += value.byteLength;
      onProgress(loaded, isFinite(total) ? total : undefined);
    }
  }

  const out = new Uint8Array(loaded);
  let off = 0;
  for (const c of chunks) {
    out.set(c, off);
    off += c.byteLength;
  }
  return out.buffer;
}

export default function KaraokePage() {
  const router = useRouter();
  const handledQueryRef = useRef(false);

  const [url, setUrl] = useState("");
  const [videoId, setVideoId] = useState("");

  const [status, setStatus] = useState<Status>("idle");
  const [stage, setStage] = useState<string>("idle");
  const [progress, setProgress] = useState<number>(0);

  const [nowTitle, setNowTitle] = useState<string>("");
  const [nowThumb, setNowThumb] = useState<string>("");

  const [taskId, setTaskId] = useState<string>("");
  const [videoUrl, setVideoUrl] = useState<string>("");

  const [bpm, setBpm] = useState<string>("");
  const [bpmDetecting, setBpmDetecting] = useState(false);
  const [bpmDetectedAt, setBpmDetectedAt] = useState<number | null>(null);
  const [bpmStartOffset, setBpmStartOffset] = useState<number>(0);

  const NOTE_OPTIONS = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"] as const;
  const [keyTonic, setKeyTonic] = useState<string>("C");
  const [keyMode, setKeyMode] = useState<"major" | "minor">("major");
  const [keyDetecting, setKeyDetecting] = useState(false);

  const [stems, setStems] = useState<StemsMap>({});
  const [stemOrder, setStemOrder] = useState<string[]>([]);

  const [stemVolumes, setStemVolumes] = useState<Record<string, number>>({});
  const [masterVolume, setMasterVolume] = useState(100);
  // instrumental master slider removed

  type MixMode =
    | { kind: "select" }
    | { kind: "master" }
    | { kind: "vocal" }
    | { kind: "instrument"; instrument: string };

  const [mixMode, setMixMode] = useState<MixMode>({ kind: "select" });

  // Simple mode sliders (used in vocal/instrument modes)
  const [focusVol, setFocusVol] = useState(100);
  const [otherInstVol, setOtherInstVol] = useState(100);
  const [vocalsVol, setVocalsVol] = useState(100);
  const [instrumentsVol, setInstrumentsVol] = useState(100);

  const [isPlaying, setIsPlaying] = useState(false);
  const isPlayingRef = useRef(false);

  const [uiPos, setUiPos] = useState(0);
  const [isScrubbing, setIsScrubbing] = useState(false);
  const [scrubPos, setScrubPos] = useState(0);

  // Jobs UI removed

  const [libraryOpen, setLibraryOpen] = useState(false);
  const [presets, setPresets] = useState<Preset[]>([]);
  const [presetName, setPresetName] = useState("");

  const [toastMsg, setToastMsg] = useState<string>("");
  const toastTimerRef = useRef<any>(null);

  const [stemLoad, setStemLoad] = useState<Record<string, StemLoadState>>({});
  const [audioReady, setAudioReady] = useState(false);
  const [audioDuration, setAudioDuration] = useState<number>(0);

  const videoRef = useRef<HTMLVideoElement | null>(null);
  const videoProgrammaticSeekAtRef = useRef<number>(0);

  // WebAudio engine refs
  const audioCtxRef = useRef<AudioContext | null>(null);
  const masterGainRef = useRef<GainNode | null>(null);
  const instGainRef = useRef<GainNode | null>(null);
  const vocalsBusGainRef = useRef<GainNode | null>(null);
  const focusBusGainRef = useRef<GainNode | null>(null);
  const otherBusGainRef = useRef<GainNode | null>(null);
  const stemGainsRef = useRef<Record<string, GainNode>>({});
  const sourcesRef = useRef<Record<string, AudioBufferSourceNode>>({});
  const buffersRef = useRef<Record<string, AudioBuffer>>({});

  const startedAtRef = useRef<number>(0); // audioCtx.currentTime when last started
  const offsetRef = useRef<number>(0); // seconds into track when last started

  const loadAbortRef = useRef<AbortController | null>(null);

  const toast = (msg: string, ms: number = 2200) => {
    setToastMsg(msg);
    if (toastTimerRef.current) clearTimeout(toastTimerRef.current);
    toastTimerRef.current = setTimeout(() => setToastMsg(""), ms);
  };

  const getAudioCtx = () => {
    if (!audioCtxRef.current) audioCtxRef.current = new AudioContext();
    const ctx = audioCtxRef.current;

    if (!masterGainRef.current) {
      masterGainRef.current = ctx.createGain();
      masterGainRef.current.connect(ctx.destination);
    }

    // Master-mode buses
    if (!instGainRef.current) {
      instGainRef.current = ctx.createGain();
      instGainRef.current.connect(masterGainRef.current);
    }

    // Focus-mode buses
    if (!vocalsBusGainRef.current) {
      vocalsBusGainRef.current = ctx.createGain();
      vocalsBusGainRef.current.connect(masterGainRef.current);
    }
    if (!focusBusGainRef.current) {
      focusBusGainRef.current = ctx.createGain();
      focusBusGainRef.current.connect(masterGainRef.current);
    }
    if (!otherBusGainRef.current) {
      otherBusGainRef.current = ctx.createGain();
      otherBusGainRef.current.connect(masterGainRef.current);
    }

    return ctx;
  };

  const currentPosition = () => {
    const ctx = audioCtxRef.current;
    if (!ctx) return offsetRef.current;
    if (!isPlaying) return offsetRef.current;
    return offsetRef.current + Math.max(0, ctx.currentTime - startedAtRef.current);
  };

  const stopSources = () => {
    try {
      for (const s of Object.values(sourcesRef.current)) {
        try {
          s.stop();
        } catch {}
        try {
          s.disconnect();
        } catch {}
      }
    } catch {}
    sourcesRef.current = {};
  };

  const pauseAudio = () => {
    const pos = currentPosition();
    stopSources();
    offsetRef.current = clampTime(pos, audioDuration);
    isPlayingRef.current = false;
    setIsPlaying(false);

    try {
      videoRef.current?.pause();
    } catch {}
  };

  const unloadAudio = async () => {
    try {
      pauseAudio();
    } catch {}

    try {
      loadAbortRef.current?.abort();
    } catch {}
    loadAbortRef.current = null;

    try {
      stopSources();
    } catch {}

    // Fully close AudioContext so audio cannot continue after navigation.
    try {
      const ctx = audioCtxRef.current;
      audioCtxRef.current = null;
      if (ctx) await ctx.close();
    } catch {}

    try {
      // best-effort disconnect buses
      masterGainRef.current?.disconnect();
    } catch {}
    masterGainRef.current = null;
    instGainRef.current = null;
    vocalsBusGainRef.current = null;
    focusBusGainRef.current = null;
    otherBusGainRef.current = null;

    sourcesRef.current = {};
    buffersRef.current = {};
    stemGainsRef.current = {};

    offsetRef.current = 0;
    startedAtRef.current = 0;
    setAudioReady(false);
    setAudioDuration(0);
    setStemLoad({});
  };

  const playAudio = async () => {
    if (!audioReady) return;

    const ctx = getAudioCtx();
    try {
      if (ctx.state !== "running") await ctx.resume();
    } catch {
      // will fail until user gesture
    }

    const startOffset = clampTime(offsetRef.current, audioDuration);
    stopSources();

    // Ensure gain nodes exist / connected for current stem set.
    for (const stem of stemOrder) {
      if (!stemGainsRef.current[stem]) {
        const g = ctx.createGain();
        stemGainsRef.current[stem] = g;
      }
      // Reconnect each time based on current mode (safe: disconnect best-effort)
      try {
        stemGainsRef.current[stem].disconnect();
      } catch {}

      // click track removed

      if (mixMode.kind === "master") {
        // vocals direct -> master, others -> instrumental bus
        if (stem === "vocals") stemGainsRef.current[stem].connect(masterGainRef.current!);
        else stemGainsRef.current[stem].connect(instGainRef.current!);
      } else if (mixMode.kind === "vocal") {
        if (stem === "vocals") stemGainsRef.current[stem].connect(focusBusGainRef.current!);
        else stemGainsRef.current[stem].connect(otherBusGainRef.current!);
      } else if (mixMode.kind === "instrument") {
        if (stem === "vocals") stemGainsRef.current[stem].connect(vocalsBusGainRef.current!);
        else if (stem === mixMode.instrument) stemGainsRef.current[stem].connect(focusBusGainRef.current!);
        else stemGainsRef.current[stem].connect(otherBusGainRef.current!);
      } else {
        // select screen: default to master routing
        if (stem === "vocals") stemGainsRef.current[stem].connect(masterGainRef.current!);
        else stemGainsRef.current[stem].connect(instGainRef.current!);
      }
    }

    // Create fresh sources per stem (BufferSource is one-shot)
    const sources: Record<string, AudioBufferSourceNode> = {};
    for (const stem of stemOrder) {
      const buf = buffersRef.current[stem];
      if (!buf) continue;
      const src = ctx.createBufferSource();
      src.buffer = buf;
      src.connect(stemGainsRef.current[stem]);
      sources[stem] = src;
    }

    sourcesRef.current = sources;
    startedAtRef.current = ctx.currentTime;
    offsetRef.current = startOffset;

    try {
      for (const src of Object.values(sources)) {
        // start immediately with an offset
        src.start(0, startOffset);
      }
    } catch (e: any) {
      toast(`Audio start failed: ${String(e?.message || e)}`);
      stopSources();
      return;
    }

    isPlayingRef.current = true;
    setIsPlaying(true);

    // Best-effort video follow
    try {
      if (videoRef.current) {
        videoRef.current.muted = true;
        videoRef.current.volume = 0;
        videoProgrammaticSeekAtRef.current = Date.now();
        videoRef.current.currentTime = startOffset;
        await videoRef.current.play();
      }
    } catch {
      // ignore
    }
  };

  const togglePlay = async () => {
    if (!audioReady) return;
    if (isPlaying) pauseAudio();
    else await playAudio();
  };

  // Switching mix modes while playing can feel jarring (routing changes / bus gains),
  // so we auto-pause on mode change.
  const modeKey = mixMode.kind === "instrument" ? `${mixMode.kind}:${mixMode.instrument}` : mixMode.kind;
  const lastModeKeyRef = useRef<string>(modeKey);
  useEffect(() => {
    const prev = lastModeKeyRef.current;
    if (prev !== modeKey) {
      lastModeKeyRef.current = modeKey;
      if (isPlayingRef.current) pauseAudio();
    }
  }, [modeKey]);

  const seekTo = async (t: number) => {
    const tt = clampTime(t, audioDuration);
    offsetRef.current = tt;

    try {
      if (videoRef.current) {
        videoProgrammaticSeekAtRef.current = Date.now();
        videoRef.current.currentTime = tt;
      }
    } catch {}

    if (isPlaying) {
      await playAudio();
    }
  };

  const jumpSeconds = async (delta: number) => {
    await seekTo(currentPosition() + delta);
  };

  const fmtTime = (t: number) => {
    if (!isFinite(t) || t < 0) t = 0;
    const m = Math.floor(t / 60);
    const s = Math.floor(t % 60);
    return `${m}:${String(s).padStart(2, "0")}`;
  };

  const [semitones, setSemitones] = useState(0); // applied
  const [semitonesPending, setSemitonesPending] = useState(0);
  const [transposing, setTransposing] = useState(false);
  const [transposePct, setTransposePct] = useState<number>(0);

  const cacheKey = useMemo(() => {
    if (!videoId) return "";
    return `${videoId}|${semitones}`;
  }, [videoId, semitones]);

  const NOTE_OPTIONS_ORDER = useMemo(() => {
    const m: Record<string, number> = {};
    NOTE_OPTIONS.forEach((n, i) => (m[n] = i));
    return m;
  }, []);

  const shiftTonic = (tonic: string, delta: number) => {
    const i = NOTE_OPTIONS_ORDER[tonic];
    if (i == null) return tonic;
    const j = (i + delta + 1200) % 12;
    return NOTE_OPTIONS[j];
  };

  const displayedKeyTonic = useMemo(() => {
    return shiftTonic(keyTonic, semitones);
  }, [keyTonic, semitones]);

  const displayedKeyTonicPending = useMemo(() => {
    return shiftTonic(keyTonic, semitonesPending);
  }, [keyTonic, semitonesPending]);

  const applyTranspose = async (next: number) => {
    if (!taskId) return;
    if (status !== "completed") return;

    setTransposing(true);
    setTransposePct(0);

    // Poll backend progress while transpose is running
    const poll = setInterval(async () => {
      try {
        const sr = await fetch(`${BACKEND}/status/${taskId}`);
        const sj = await sr.json();
        if (sj.stage === "transposing" && typeof sj.progress === "number") {
          setTransposePct(Math.max(0, Math.min(99, Math.round(sj.progress))));
        }
      } catch {}
    }, 400);

    try {
      const tr = await fetch(`${BACKEND}/transpose/${taskId}?semitones=${next}`, { method: "POST" });
      const tj = await tr.json();
      if (!tr.ok) throw new Error(tj?.detail || "transpose failed");

      if (tj?.stems) {
        const stemsRel = tj.stems as Record<string, string>;
        const full: StemsMap = {};
        for (const [k, v] of Object.entries(stemsRel)) full[k] = `${BACKEND}${v}`;

        const names = Object.keys(full);
        names.sort((a, b) => {
          if (a === "vocals") return -1;
          if (b === "vocals") return 1;
          return a.localeCompare(b);
        });

        setStems(full);
        setStemOrder(names);
        resetPlayer();
      }

      setSemitones(next);
      setTransposePct(100);
    } catch (e: any) {
      toast(`Transpose failed: ${String(e?.message || e)}`);
      setSemitonesPending(semitones);
    } finally {
      clearInterval(poll);
      setTransposing(false);
    }
  };

  const canPlay = useMemo(() => {
    return audioReady && stemOrder.length > 0;
  }, [audioReady, stemOrder.length]);

  // UI clock for timeline
  useEffect(() => {
    if (!audioReady) {
      setUiPos(0);
      setScrubPos(0);
      setIsScrubbing(false);
      return;
    }

    const tick = () => {
      if (isScrubbing) return;
      setUiPos(currentPosition());
      setScrubPos(currentPosition());
    };

    tick();
    const t = setInterval(tick, 250);
    return () => clearInterval(t);
  }, [audioReady, isScrubbing, isPlaying, videoUrl, audioDuration]);

  // Overall load/decode percent for the Play button while stems are being fetched/decoded.
  const audioLoadPercent = useMemo(() => {
    if (!stemOrder.length) return null;
    let totalWeight = 0;
    let done = 0;

    for (const stem of stemOrder) {
      const st = stemLoad[stem];
      // Weight each stem equally.
      totalWeight += 1;

      if (!st || st.status === "idle") {
        done += 0;
      } else if (st.status === "ready") {
        done += 1;
      } else if (st.status === "error") {
        done += 0;
      } else if (st.status === "decoding") {
        // no % available; count as mostly done
        done += 0.9;
      } else if (st.status === "loading") {
        if (st.total && st.total > 0) {
          done += Math.max(0, Math.min(0.9, st.loaded / st.total * 0.9));
        } else {
          done += 0.1;
        }
      }
    }

    if (totalWeight <= 0) return null;
    const pct = Math.round((done / totalWeight) * 100);
    return Math.max(0, Math.min(99, pct));
  }, [stemOrder.join("|"), stemLoad]);

  // Apply gains
  useEffect(() => {
    const master = Math.max(0, Math.min(1, masterVolume / 100));
    if (masterGainRef.current) masterGainRef.current.gain.value = master;

    // reset buses to unity
    if (instGainRef.current) instGainRef.current.gain.value = 1;
    if (vocalsBusGainRef.current) vocalsBusGainRef.current.gain.value = 1;
    if (focusBusGainRef.current) focusBusGainRef.current.gain.value = 1;
    if (otherBusGainRef.current) otherBusGainRef.current.gain.value = 1;

    if (mixMode.kind === "master") {
      // Instrumental bus gain is fixed at unity (instrumental slider removed).
      if (instGainRef.current) instGainRef.current.gain.value = 1.0;

      for (const stem of stemOrder) {
        const g = stemGainsRef.current[stem];
        if (!g) continue;
        const sv = Math.max(0, Math.min(1, Number(stemVolumes[stem] ?? 100) / 100));
        g.gain.value = sv;
      }
      return;
    }

    // non-master modes: use simple buses; keep per-stem at unity
    for (const stem of stemOrder) {
      const g = stemGainsRef.current[stem];
      if (g) g.gain.value = 1.0;
    }

    if (mixMode.kind === "vocal") {
      if (focusBusGainRef.current) focusBusGainRef.current.gain.value = Math.max(0, Math.min(1, vocalsVol / 100));
      if (otherBusGainRef.current) otherBusGainRef.current.gain.value = Math.max(0, Math.min(1, instrumentsVol / 100));
      return;
    }

    if (mixMode.kind === "instrument") {
      if (focusBusGainRef.current) focusBusGainRef.current.gain.value = Math.max(0, Math.min(1, focusVol / 100));
      if (otherBusGainRef.current) otherBusGainRef.current.gain.value = Math.max(0, Math.min(1, otherInstVol / 100));
      if (vocalsBusGainRef.current) vocalsBusGainRef.current.gain.value = Math.max(0, Math.min(1, vocalsVol / 100));
      return;
    }
  }, [masterVolume, stemVolumes, stemOrder, mixMode, focusVol, otherInstVol, vocalsVol, instrumentsVol]);

  // Keep video muted always
  useEffect(() => {
    try {
      if (videoRef.current) {
        videoRef.current.muted = true;
        videoRef.current.volume = 0;
      }
    } catch {}
  }, [videoUrl, status, isPlaying]);

  // Keyboard shortcuts
  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement | null;
      const tag = (target?.tagName || "").toLowerCase();
      if (tag === "input" || tag === "textarea" || (target as any)?.isContentEditable) return;

      if (e.code === "Space") {
        e.preventDefault();
        togglePlay();
      } else if (e.code === "ArrowLeft") {
        e.preventDefault();
        jumpSeconds(-5);
      } else if (e.code === "ArrowRight") {
        e.preventDefault();
        jumpSeconds(5);
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isPlaying, canPlay, videoUrl, audioDuration]);

  // Video -> Audio seeking / pausing (if user uses video controls)
  useEffect(() => {
    const p = videoRef.current;
    if (!p) return;

    const onSeeked = () => {
      // Ignore seeked events caused by our own programmatic nudges.
      if (Date.now() - videoProgrammaticSeekAtRef.current < 500) return;
      // If the user scrubs the video, treat that as the seek source-of-truth.
      seekTo(Number(p.currentTime || 0));
    };

    const onPause = () => {
      if (isPlayingRef.current) pauseAudio();
    };

    const onPlay = () => {
      if (!isPlayingRef.current && audioReady) playAudio();
    };

    p.addEventListener("seeked", onSeeked);
    p.addEventListener("pause", onPause);
    p.addEventListener("play", onPlay);

    return () => {
      p.removeEventListener("seeked", onSeeked);
      p.removeEventListener("pause", onPause);
      p.removeEventListener("play", onPlay);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [videoUrl, isPlaying, audioReady, audioDuration]);

  // Audio -> Video follow (audio is master clock)
  // We avoid constant hard seeking, but we *do*:
  // - nudge playbackRate slightly to correct drift
  // - occasionally snap if drift grows too large
  useEffect(() => {
    const p = videoRef.current;
    if (!p) return;

    // Reset rate when not playing
    if (!isPlaying) {
      try {
        p.playbackRate = 1.0;
      } catch {}
      return;
    }

    const clamp = (v: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, v));

    const t = setInterval(() => {
      try {
        if (!isPlayingRef.current) return;
        if ((p.readyState || 0) < 2) return;

        // Keep video playing (muted)
        if (p.paused) {
          p.play().catch(() => {});
        }

        const a = currentPosition();
        const v = Number(p.currentTime || 0);
        const drift = v - a; // + => video ahead, - => video behind

        // If drift is large, do a rare snap
        if (Math.abs(drift) > 0.35) {
          try {
            videoProgrammaticSeekAtRef.current = Date.now();
            p.currentTime = a;
            p.playbackRate = 1.0;
          } catch {}
          return;
        }

        // Small drift: correct via playbackRate
        // If video is behind (drift negative), speed up a bit.
        // If video is ahead, slow down a bit.
        const rate = clamp(1.0 - drift * 0.75, 0.94, 1.06);
        try {
          p.playbackRate = rate;
        } catch {}
      } catch {}
    }, 250);

    return () => {
      clearInterval(t);
      try {
        p.playbackRate = 1.0;
      } catch {}
    };
  }, [isPlaying, videoUrl, audioDuration]);

  const resetPlayer = () => {
    pauseAudio();
    offsetRef.current = 0;
    setAudioReady(false);
    setAudioDuration(0);
    buffersRef.current = {};
    setStemLoad({});

    // Cancel in-flight fetches
    try {
      loadAbortRef.current?.abort();
    } catch {}
    loadAbortRef.current = null;
  };

  // Cleanup: stop audio if user navigates away from this page.
  useEffect(() => {
    return () => {
      // best-effort (can't await in cleanup)
      unloadAudio();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Load & decode stems when job completes (or preset loads)
  useEffect(() => {
    if (!cacheKey) return;
    if (!stemOrder.length) return;

    const load = async () => {
      resetPlayer();

      const cached = decodedStemCache.get(cacheKey);
      if (cached) {
        buffersRef.current = cached;
        let dur = 0;
        for (const b of Object.values(cached)) dur = Math.max(dur, b.duration || 0);
        setAudioDuration(dur);
        setAudioReady(true);
        setStemLoad(() => {
          const m: Record<string, StemLoadState> = {};
          for (const stem of stemOrder) {
            const b = cached[stem];
            if (b) m[stem] = { status: "ready", duration: b.duration || 0 };
            else m[stem] = { status: "error", error: "missing buffer" };
          }
          return m;
        });
        return;
      }

      const ac = getAudioCtx();
      const abort = new AbortController();
      loadAbortRef.current = abort;

      const out: Record<string, AudioBuffer> = {};
      let dur = 0;

      for (const stem of stemOrder) {
        const stemUrl = stems[stem];
        if (!stemUrl) continue;

        setStemLoad((prev) => ({ ...prev, [stem]: { status: "loading", loaded: 0 } }));

        try {
          const ab = await fetchArrayBufferWithProgress(
            stemUrl,
            (loaded, total) => {
              setStemLoad((prev) => ({
                ...prev,
                [stem]: { status: "loading", loaded, total },
              }));
            },
            abort.signal
          );

          setStemLoad((prev) => ({ ...prev, [stem]: { status: "decoding" } }));

          const buf = await ac.decodeAudioData(ab.slice(0));
          out[stem] = buf;
          dur = Math.max(dur, buf.duration || 0);
          setStemLoad((prev) => ({ ...prev, [stem]: { status: "ready", duration: buf.duration || 0 } }));
        } catch (e: any) {
          const msg = String(e?.message || e);
          setStemLoad((prev) => ({ ...prev, [stem]: { status: "error", error: msg } }));
        }
      }

      buffersRef.current = out;
      decodedStemCache.set(cacheKey, out);
      setAudioDuration(dur);

      const loadedCount = Object.keys(out).length;
      setAudioReady(loadedCount > 0);

      if (loadedCount === stemOrder.length) {
        toast("Audio decoded — ready");
      } else if (loadedCount > 0) {
        toast("Some stems failed to load (playing what we have)");
      } else {
        toast("No stems could be decoded");
      }
    };

    load();

    return () => {
      try {
        loadAbortRef.current?.abort();
      } catch {}
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [cacheKey, stemOrder.join("|"), JSON.stringify(stems)]);

  // refreshJobs removed

  const refreshPresets = async () => {
    try {
      const r = await fetch(`${BACKEND}/presets`);
      const j = await r.json();
      setPresets(j.items || []);
    } catch {
      setPresets([]);
    }
  };

  // Jobs polling removed

  // If navigated from /library, auto-load url from query string.
  useEffect(() => {
    if (handledQueryRef.current) return;
    if (typeof window === "undefined") return;
    const sp = new URLSearchParams(window.location.search);
    const qUrl = sp.get("url") || "";
    if (!qUrl) return;
    handledQueryRef.current = true;
    const qBpm = sp.get("bpm") || "";
    if (qBpm) setBpm(qBpm);
    loadFromUrl(qUrl);
  }, []);

  const loadFromUrl = async (inputUrl: string) => {
    const id = extractYouTubeId(inputUrl);
    if (!/^[A-Za-z0-9_-]{11}$/.test(id)) {
      toast("Invalid YouTube URL");
      return;
    }

    // Reset transpose per requirement (library stores defaults)
    setSemitones(0);
    setSemitonesPending(0);

    // Load saved meta (bpm/key) from Library if present
    try {
      const rr = await fetch(`${BACKEND}/presets/${encodeURIComponent(id)}`);
      const jj = await rr.json();
      if (rr.ok && jj?.item) {
        if (jj.item.bpm != null) setBpm(String(jj.item.bpm));
        if (jj.item.key_tonic) setKeyTonic(String(jj.item.key_tonic));
        if (jj.item.key_mode) setKeyMode(String(jj.item.key_mode) === "minor" ? "minor" : "major");
      }
    } catch {}

    setUrl(inputUrl);
    setVideoId(id);
    setStems({});
    setStemOrder([]);
    setStemVolumes({});
    setVideoUrl("");
    setNowTitle("");
    setNowThumb("");
    setTaskId("");
    // NOTE: do not reset bpm/key here; they may be restored from Library preset meta

    resetPlayer();

    setProgress(0);
    setStage("queued");
    setStatus("processing");

    const canonical = `https://www.youtube.com/watch?v=${id}`;

    // oEmbed early
    try {
      const oe = await fetch(`https://www.youtube.com/oembed?url=${encodeURIComponent(canonical)}&format=json`);
      if (oe.ok) {
        const j = await oe.json();
        setNowTitle(j.title || "");
        setNowThumb(j.thumbnail_url || "");
      }
    } catch {}

    const resp = await fetch(`${BACKEND}/process?url=${encodeURIComponent(canonical)}&queued=true`, { method: "POST" });
    const jj = await resp.json();
    const tid = String(jj.task_id || "");
    setTaskId(tid);

    const applyCompletedStatus = async (sj: any) => {
      setStatus("completed");
      setStage("completed");
      setProgress(100);

      const stemsRel = (sj.stems || {}) as Record<string, string>;
      const full: StemsMap = {};
      for (const [k, v] of Object.entries(stemsRel)) {
        full[k] = `${BACKEND}${v}`;
      }

      // If transposed, swap stems to transposed variants.
      if (semitones !== 0 && tid) {
        try {
          const tr = await fetch(`${BACKEND}/transpose/${tid}?semitones=${semitones}`, { method: "POST" });
          const tj = await tr.json();
          if (tr.ok && tj?.stems) {
            for (const [k, v] of Object.entries(tj.stems as Record<string, string>)) {
              full[k] = `${BACKEND}${v}`;
            }
          }
        } catch {}
      }

      const names = Object.keys(full);
      names.sort((a, b) => {
        if (a === "vocals") return -1;
        if (b === "vocals") return 1;
        return a.localeCompare(b);
      });

      setStems(full);
      setStemOrder(names);

      setStemVolumes((prev) => {
        const out = { ...prev };
        for (const nm of names) if (out[nm] == null) out[nm] = 100;
        return out;
      });

      setVideoUrl(sj.video ? `${BACKEND}${sj.video}` : "");

      // Auto-save to Library (preset) using title/thumbnail when available.
      try {
        await fetch(`${BACKEND}/presets`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            id,
            name: nowTitle || id,
            source_url: canonical,
            video_id: id,
            title: nowTitle || null,
            thumbnail_url: nowThumb || null,
            // Do NOT overwrite saved BPM/Key on autosave; those are set via Detect/manual edits.
            bpm: null,
            key_tonic: null,
            key_mode: null,
            semitones: 0,
            master_volume: masterVolume / 100,
            vocal_volume: 1.0,
          }),
        });
      } catch {}

      // If user is still on the select screen, default to Master mode after first load.
      setMixMode((m) => (m.kind === "select" ? { kind: "master" } : m));
      toast("Stems ready — decoding audio…");
    };

    // If backend already has cached output, it may return completed immediately.
    if (jj.status === "completed") {
      try {
        const sr = await fetch(`${BACKEND}/status/${tid}`);
        const sj = await sr.json();
        if (sj.status === "completed") {
          await applyCompletedStatus(sj);
          return;
        }
      } catch {}
    }

    const poll = setInterval(async () => {
      try {
        const sr = await fetch(`${BACKEND}/status/${tid}`);
        const sj = await sr.json();
        if (sj.stage) setStage(String(sj.stage));
        if (typeof sj.progress === "number") setProgress(sj.progress);

        if (sj.status === "completed") {
          clearInterval(poll);
          await applyCompletedStatus(sj);
        }

        if (sj.status === "error") {
          clearInterval(poll);
          setStatus("idle");
          toast("Job failed");
        }
      } catch {
        // ignore
      }
    }, 1000);
  };

  const saveMetaToLibrary = async (patch: any) => {
    if (!videoId) return;
    try {
      await fetch(`${BACKEND}/presets/${encodeURIComponent(videoId)}/meta`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          title: nowTitle || null,
          thumbnail_url: nowThumb || null,
          // IMPORTANT: store default/original key regardless of transpose.
          ...patch,
        }),
      });
    } catch {
      // ignore
    }
  };

  const detectBpm = async () => {
    if (!taskId) return;
    setBpmDetecting(true);
    try {
      const resp = await fetch(`${BACKEND}/bpm/${taskId}/detect`, { method: "POST" });
      const j = await resp.json();
      if (!resp.ok) throw new Error(j?.detail || "detect failed");
      setBpm(String(j.bpm ?? ""));
      setBpmDetectedAt(Date.now());
      setBpmStartOffset(Number(j.start_offset ?? 0) || 0);
      await saveMetaToLibrary({ bpm: j.bpm ?? null });
      toast(`BPM detected (${j.source || "?"}): ${j.bpm}${j.start_offset ? ` (start ${Number(j.start_offset).toFixed(2)}s)` : ""}`);
    } catch (e: any) {
      toast(`BPM detect failed: ${String(e?.message || e)}`);
    } finally {
      setBpmDetecting(false);
    }
  };

  const detectKey = async () => {
    if (!taskId) return;
    setKeyDetecting(true);
    try {
      const resp = await fetch(`${BACKEND}/key/${taskId}/detect`, { method: "POST" });
      const j = await resp.json();
      if (!resp.ok) throw new Error(j?.detail || "detect failed");
      const t = j.tonic ? String(j.tonic) : "";
      const m = j.mode === "minor" ? "minor" : "major";
      if (t) setKeyTonic(t);
      setKeyMode(m);
      await saveMetaToLibrary({ key_tonic: t || null, key_mode: m });
      toast(`Key detected (${j.source || "?"}): ${t}${m === "minor" ? "m" : "M"}`);
    } catch (e: any) {
      toast(`Key detect failed: ${String(e?.message || e)}`);
    } finally {
      setKeyDetecting(false);
    }
  };

  const exportMp3 = async () => {
    if (!taskId) return;
    try {
      const resp = await fetch(`${BACKEND}/export/${taskId}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          mode: mixMode,
          masterVolume,
          // instrumentalMaster removed
          stemVolumes,
          focusVol,
          otherInstVol,
          vocalsVol,
          instrumentsVol,
        }),
      });
      const j = await resp.json();
      if (!resp.ok) throw new Error(j?.detail || "export failed");

      const url = `${BACKEND}${j.mp3}`;
      const fname = `mix_${videoId || taskId}.mp3`;

      // If running inside Electron (mac app), use native Save As dialog.
      // @ts-ignore
      const holy = (globalThis as any).holy;
      if (holy?.saveFile) {
        await holy.saveFile(url, fname);
        toast("Exported MP3");
      } else {
        const a = document.createElement("a");
        a.href = url;
        a.download = fname;
        document.body.appendChild(a);
        a.click();
        a.remove();
        toast("Exported MP3 (check Downloads)");
      }
    } catch (e: any) {
      toast(`Export failed: ${String(e?.message || e)}`);
    }
  };

  const savePreset = async () => {
    const vid = extractYouTubeId(url) || videoId;
    if (!vid) return;

    const name = presetName || nowTitle || vid;
    await fetch(`${BACKEND}/presets`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        id: vid,
        name,
        source_url: url || `https://www.youtube.com/watch?v=${vid}`,
        video_id: vid,
        title: nowTitle || null,
        thumbnail_url: nowThumb || null,
        // Save current should persist BPM/Key explicitly.
        bpm: bpm ? Number(bpm) : null,
        key_tonic: keyTonic || null,
        key_mode: keyMode || null,
        semitones: 0,
        master_volume: masterVolume / 100,
        vocal_volume: 1.0,
      }),
    });
    setPresetName("");
    await refreshPresets();
  };

  const loadPreset = async (p: Preset) => {
    setLibraryOpen(false);
    setBpm(p.bpm != null ? String(p.bpm) : "");
    if (p.key_tonic) setKeyTonic(String(p.key_tonic));
    if (p.key_mode) setKeyMode(String(p.key_mode) === "minor" ? "minor" : "major");
    await loadFromUrl(p.source_url || `https://www.youtube.com/watch?v=${p.video_id}`);
  };

  // refresh presets when library opened
  useEffect(() => {
    if (!libraryOpen) return;
    refreshPresets();
  }, [libraryOpen]);

  return (
    <main className="relative flex flex-col h-screen bg-black text-white overflow-hidden">
      <div className="p-4 border-b border-zinc-800 bg-zinc-950">
        <div className="flex gap-2 items-center">
          <input
            className="flex-1 bg-zinc-900 border border-zinc-700 rounded-lg px-3 py-2 text-sm"
            placeholder="Paste YouTube URL (lyric video)…"
            value={url}
            onChange={(e) => setUrl(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") loadFromUrl(url);
            }}
          />

          {/* Load moved next to URL input */}
          <button onClick={() => loadFromUrl(url)} className="px-3 py-2 rounded-lg border border-zinc-700 text-sm">
            Load
          </button>

          <button
            onClick={async () => {
              await unloadAudio();
              router.push("/library");
            }}
            className="px-3 py-2 rounded-lg border border-zinc-700 text-sm"
          >
            Library
          </button>

          <div className="flex-1" />

          <button
            onClick={exportMp3}
            disabled={status !== "completed"}
            className={`px-3 py-2 rounded-lg border text-sm ${status === "completed" ? "border-zinc-700 hover:bg-zinc-900" : "border-zinc-800 text-zinc-600 cursor-not-allowed"}`}
          >
            Export MP3
          </button>
        </div>
      </div>

      {/* Video / status area */}
      <div className="flex-1 relative">
        {videoUrl ? (
          <div className="absolute inset-0 bg-black">
            <video ref={videoRef} src={videoUrl} className="absolute inset-0 w-full h-full object-contain bg-black" playsInline muted />
            <div className="absolute top-3 left-3 max-w-[75%] rounded-xl border border-zinc-800 bg-black/60 backdrop-blur px-3 py-2 text-white text-sm sm:text-base font-semibold shadow">
              {nowTitle || videoId}
            </div>
          </div>
        ) : videoId ? (
          <div className="h-full w-full flex items-center justify-center bg-black">
            <div className="text-center space-y-3 max-w-md px-6">
              {nowThumb ? (
                // eslint-disable-next-line @next/next/no-img-element
                <img src={nowThumb} alt="thumb" className="mx-auto w-64 rounded-xl border border-zinc-800" />
              ) : null}
              <div className="text-sm text-zinc-200">{nowTitle || "Preparing…"}</div>
              {/* Saved name moved below transport */}
              <div className="text-[12px] text-zinc-500">{status === "completed" ? "Video not available (audio ready)." : "Working…"}</div>
              {status === "processing" ? (
                <div className="rounded-xl border border-zinc-800 bg-zinc-950/40 px-4 py-3">
                  <div className="text-[11px] text-zinc-400">
                    {stage} • {progress}%
                  </div>
                  <div className="h-2 bg-zinc-800/70 rounded-full overflow-hidden mt-2">
                    <div className="h-full bg-blue-500/70" style={{ width: `${Math.max(0, Math.min(100, progress))}%` }} />
                  </div>
                </div>
              ) : null}
            </div>
          </div>
        ) : (
          <div className="h-full flex items-center justify-center text-zinc-500 text-sm">Paste a YouTube link above.</div>
        )}

        {toastMsg ? (
          <div className="absolute left-1/2 -translate-x-1/2 top-4 rounded-full border border-zinc-800 bg-zinc-950/70 backdrop-blur px-4 py-2 shadow text-xs text-zinc-200">
            {toastMsg}
          </div>
        ) : null}
      </div>

      {/* Controls */}
      <div className="p-6 bg-zinc-900/80 border-t border-zinc-800">
        {/* Start screen: choose instrument */}
        {mixMode.kind === "select" ? (
          <div className="max-w-3xl mx-auto">
            <div className="text-center text-zinc-200 text-lg font-semibold">Choose your instrument</div>
            <div className="mt-4 grid grid-cols-2 sm:grid-cols-3 gap-3">
              <button
                onClick={() => setMixMode({ kind: "vocal" })}
                className="rounded-2xl border border-zinc-800 bg-zinc-950/60 px-4 py-4 text-left hover:bg-zinc-900"
              >
                <div className="text-2xl">🎤</div>
                <div className="mt-2 text-sm text-zinc-200">Vocal</div>
              </button>
              <button
                onClick={() => setMixMode({ kind: "instrument", instrument: "drums" })}
                className="rounded-2xl border border-zinc-800 bg-zinc-950/60 px-4 py-4 text-left hover:bg-zinc-900"
              >
                <div className="text-2xl">🥁</div>
                <div className="mt-2 text-sm text-zinc-200">Drum</div>
              </button>
              <button
                onClick={() => setMixMode({ kind: "instrument", instrument: "bass" })}
                className="rounded-2xl border border-zinc-800 bg-zinc-950/60 px-4 py-4 text-left hover:bg-zinc-900"
              >
                <div className="text-2xl">🎸</div>
                <div className="mt-2 text-sm text-zinc-200">Bass</div>
              </button>
              <button
                onClick={() => setMixMode({ kind: "instrument", instrument: "guitar" })}
                className="rounded-2xl border border-zinc-800 bg-zinc-950/60 px-4 py-4 text-left hover:bg-zinc-900"
              >
                <div className="text-2xl">🎸</div>
                <div className="mt-2 text-sm text-zinc-200">Guitar</div>
              </button>
              <button
                onClick={() => setMixMode({ kind: "instrument", instrument: "piano" })}
                className="rounded-2xl border border-zinc-800 bg-zinc-950/60 px-4 py-4 text-left hover:bg-zinc-900"
              >
                <div className="text-2xl">🎹</div>
                <div className="mt-2 text-sm text-zinc-200">Piano</div>
              </button>
              <button
                onClick={() => setMixMode({ kind: "master" })}
                className="rounded-2xl border border-blue-800 bg-blue-950/30 px-4 py-4 text-left hover:bg-blue-900/30"
              >
                <div className="text-2xl">🎛️</div>
                <div className="mt-2 text-sm text-zinc-100">Master</div>
              </button>
            </div>
            <div className="mt-4 text-center text-[11px] text-zinc-500">
              Load a song first (top). Then pick a mode.
            </div>
          </div>
        ) : null}
        <div className="max-w-3xl mx-auto space-y-4">
          <div className="flex items-center justify-center gap-4">
            <button
              onClick={() => jumpSeconds(-5)}
              disabled={!audioReady}
              className={`px-3 py-2 rounded-lg font-bold border ${audioReady ? "border-zinc-700 hover:bg-zinc-800" : "border-zinc-800 text-zinc-600 cursor-not-allowed"}`}
            >
              −5s
            </button>

            <button
              onClick={togglePlay}
              disabled={!canPlay}
              className={`relative overflow-hidden px-4 py-2 rounded-lg font-bold flex items-center gap-2 border ${canPlay ? "border-zinc-700 hover:bg-zinc-800" : "border-zinc-800 text-zinc-600 cursor-not-allowed"}`}
            >
              {isPlaying ? <Pause size={18} /> : <Play size={18} />}
              <span className="relative z-10">
                {isPlaying ? "Pause" : audioReady ? "Play" : audioLoadPercent != null ? `Loading ${audioLoadPercent}%` : "Loading"}
              </span>
              {!audioReady && audioLoadPercent != null ? (
                <span
                  className="absolute inset-y-0 left-0 bg-blue-600/25"
                  style={{ width: `${Math.max(0, Math.min(100, audioLoadPercent))}%` }}
                />
              ) : null}
              {!audioReady && audioLoadPercent == null ? <span className="absolute inset-0 bg-blue-600/10 animate-pulse" /> : null}
            </button>

            <button
              onClick={() => jumpSeconds(5)}
              disabled={!audioReady}
              className={`px-3 py-2 rounded-lg font-bold border ${audioReady ? "border-zinc-700 hover:bg-zinc-800" : "border-zinc-800 text-zinc-600 cursor-not-allowed"}`}
            >
              +5s
            </button>
          </div>

          {/* Timeline (seek) */}
          <div className="flex items-center gap-3">
            <div className="text-xs font-mono text-zinc-400 w-12 text-right">{fmtTime(isScrubbing ? scrubPos : uiPos)}</div>
            <input
              type="range"
              min={0}
              max={Math.max(0, audioDuration || 0)}
              step={0.01}
              value={Math.max(0, Math.min(audioDuration || 0, isScrubbing ? scrubPos : uiPos))}
              onChange={(e) => {
                const v = Number(e.target.value);
                setIsScrubbing(true);
                setScrubPos(v);
              }}
              onMouseUp={async () => {
                setIsScrubbing(false);
                await seekTo(scrubPos);
              }}
              onTouchEnd={async () => {
                setIsScrubbing(false);
                await seekTo(scrubPos);
              }}
              disabled={!audioReady}
              className={`flex-1 ${audioReady ? "" : "opacity-50"}`}
            />
            <div className="text-xs font-mono text-zinc-400 w-12">{fmtTime(audioDuration || 0)}</div>
          </div>

          <div className="flex items-center justify-center gap-3 flex-wrap">
            <div className="flex items-center gap-2">
              <div className="text-xs text-zinc-400">BPM</div>
              <input
                value={bpm}
                onChange={(e) => setBpm(e.target.value.replace(/[^0-9.]/g, ""))}
                onBlur={() => saveMetaToLibrary({ bpm: bpm ? Number(bpm) : null })}
                placeholder=""
                className="w-20 bg-zinc-900 border border-zinc-700 rounded-lg px-2 py-2 text-sm"
              />
              <button
                onClick={detectBpm}
                disabled={status !== "completed" || bpmDetecting}
                className={`relative overflow-hidden px-3 py-2 rounded-lg border text-sm ${status === "completed" && !bpmDetecting ? "border-zinc-700 hover:bg-zinc-900" : "border-zinc-800 text-zinc-600 cursor-not-allowed"}`}
              >
                <span className="relative z-10">{bpmDetecting ? "Detecting…" : "Detect"}</span>
                {bpmDetecting ? <span className="absolute inset-0 bg-blue-600/10 animate-pulse" /> : null}
              </button>
            </div>

            <div className="flex items-center gap-2">
              <div className="text-xs text-zinc-400">Key</div>
              <select
                value={displayedKeyTonicPending}
                onChange={(e) => {
                  const disp = e.target.value;
                  const base = shiftTonic(disp, -semitonesPending);
                  setKeyTonic(base);
                  saveMetaToLibrary({ key_tonic: base, key_mode: keyMode });
                }}
                className="bg-zinc-900 border border-zinc-700 rounded-lg px-2 py-2 text-sm"
              >
                {NOTE_OPTIONS.map((n) => (
                  <option key={n} value={n}>
                    {n}
                  </option>
                ))}
              </select>
              <button
                onClick={() => {
                  setKeyMode("major");
                  saveMetaToLibrary({ key_tonic: keyTonic, key_mode: "major" });
                }}
                className={`px-2 py-2 rounded-lg border text-sm ${keyMode === "major" ? "border-blue-700 text-blue-200" : "border-zinc-700 text-zinc-300"}`}
                title="Major"
              >
                M
              </button>
              <button
                onClick={() => {
                  setKeyMode("minor");
                  saveMetaToLibrary({ key_tonic: keyTonic, key_mode: "minor" });
                }}
                className={`px-2 py-2 rounded-lg border text-sm ${keyMode === "minor" ? "border-blue-700 text-blue-200" : "border-zinc-700 text-zinc-300"}`}
                title="Minor"
              >
                m
              </button>
              <button
                onClick={detectKey}
                disabled={status !== "completed" || keyDetecting}
                className={`relative overflow-hidden px-3 py-2 rounded-lg border text-sm ${status === "completed" && !keyDetecting ? "border-zinc-700 hover:bg-zinc-900" : "border-zinc-800 text-zinc-600 cursor-not-allowed"}`}
              >
                <span className="relative z-10">{keyDetecting ? "Detecting…" : "Detect"}</span>
                {keyDetecting ? <span className="absolute inset-0 bg-blue-600/10 animate-pulse" /> : null}
              </button>

              <div className="w-px h-6 bg-zinc-800 mx-1" />

              <button
                onClick={() => setSemitonesPending((v) => Math.max(-6, v - 1))}
                disabled={semitonesPending <= -6 || transposing}
                className={`px-2 py-2 rounded-lg border text-sm ${semitonesPending > -6 && !transposing ? "border-zinc-700 hover:bg-zinc-900" : "border-zinc-800 text-zinc-600 cursor-not-allowed"}`}
                title="Transpose down"
              >
                −
              </button>
              <div className="text-xs font-mono text-zinc-300 w-12 text-center">
                {semitonesPending >= 0 ? `+${semitonesPending}` : String(semitonesPending)}
              </div>
              <button
                onClick={() => setSemitonesPending((v) => Math.min(6, v + 1))}
                disabled={semitonesPending >= 6 || transposing}
                className={`px-2 py-2 rounded-lg border text-sm ${semitonesPending < 6 && !transposing ? "border-zinc-700 hover:bg-zinc-900" : "border-zinc-800 text-zinc-600 cursor-not-allowed"}`}
                title="Transpose up"
              >
                +
              </button>
              <button
                onClick={() => setSemitonesPending(0)}
                disabled={semitonesPending === 0 || transposing}
                className={`px-2 py-2 rounded-lg border text-[11px] ${semitonesPending !== 0 && !transposing ? "border-zinc-700 hover:bg-zinc-900 text-zinc-200" : "border-zinc-800 text-zinc-600 cursor-not-allowed"}`}
                title="Reset transpose (pending)"
              >
                0
              </button>
              <button
                onClick={() => applyTranspose(semitonesPending)}
                disabled={transposing || semitonesPending === semitones || status !== "completed"}
                className={`relative overflow-hidden px-3 py-2 rounded-lg border text-sm ${!transposing && semitonesPending !== semitones && status === "completed" ? "border-blue-700 text-blue-200 hover:bg-blue-950/30" : "border-zinc-800 text-zinc-600 cursor-not-allowed"}`}
                title="Apply transpose"
              >
                {transposing ? (
                  <span className="relative z-10">Transposing {transposePct}%</span>
                ) : (
                  <span className="relative z-10">Apply</span>
                )}
                {transposing ? (
                  <span
                    className="absolute inset-y-0 left-0 bg-blue-600/30"
                    style={{ width: `${Math.max(0, Math.min(100, transposePct))}%` }}
                  />
                ) : null}
              </button>
            </div>
          </div>

          {/* Saved name removed (title is shown as overlay on video) */}

          <div className="flex items-center gap-4">
            <div className="text-xs text-zinc-400 w-32">Master</div>
            <input type="range" min={0} max={100} value={masterVolume} onChange={(e) => setMasterVolume(Number(e.target.value))} className="flex-1" />
            <div className="text-xs font-mono text-zinc-300 w-12 text-right">{masterVolume}%</div>
          </div>

          {/* Instrumental slider removed */}

          <div className="flex items-center justify-between">
            <div className="text-[11px] text-zinc-500">Stems: {stemOrder.length ? stemOrder.join(", ") : "(none yet)"}</div>
            <div className="flex gap-2">
              <button
                onClick={() => setMixMode({ kind: "select" })}
                className="text-[11px] px-2 py-1 rounded-md border border-zinc-800 text-zinc-300"
              >
                Mode
              </button>
              <button
                onClick={() => setMixMode({ kind: "master" })}
                className={`text-[11px] px-2 py-1 rounded-md border ${mixMode.kind === "master" ? "border-blue-700 text-blue-200" : "border-zinc-800 text-zinc-300"}`}
              >
                Master
              </button>
            </div>
          </div>

          {mixMode.kind === "vocal" ? (
            <>
              <div className="flex flex-nowrap gap-3 overflow-x-auto pb-2">
                <div className="rounded-2xl border border-zinc-800 bg-zinc-950/40 p-2 flex flex-col items-center gap-2 min-w-[120px]">
                  <div className="text-xs text-zinc-300">{stemIcon("vocals")} Vocals</div>
                  <VerticalFader value={vocalsVol} disabled={!audioReady} height={180} onChange={(v) => setVocalsVol(v)} />
                </div>
                <div className="rounded-2xl border border-zinc-800 bg-zinc-950/40 p-2 flex flex-col items-center gap-2 min-w-[120px]">
                  <div className="text-xs text-zinc-300">{stemIcon("instruments")} Instruments</div>
                  <VerticalFader value={instrumentsVol} disabled={!audioReady} height={180} onChange={(v) => setInstrumentsVol(v)} />
                </div>
              </div>
            </>
          ) : null}

          {mixMode.kind === "instrument" ? (
            <>
              <div className="flex flex-nowrap gap-3 overflow-x-auto pb-2">
                <div className="rounded-2xl border border-zinc-800 bg-zinc-950/40 p-2 flex flex-col items-center gap-2 min-w-[120px]">
                  <div className="text-xs text-zinc-300">{stemIcon(mixMode.instrument)} My {mixMode.instrument}</div>
                  <VerticalFader value={focusVol} disabled={!audioReady} height={180} onChange={(v) => setFocusVol(v)} />
                </div>
                <div className="rounded-2xl border border-zinc-800 bg-zinc-950/40 p-2 flex flex-col items-center gap-2 min-w-[120px]">
                  <div className="text-xs text-zinc-300">{stemIcon("other")} Other</div>
                  <VerticalFader value={otherInstVol} disabled={!audioReady} height={180} onChange={(v) => setOtherInstVol(v)} />
                </div>
                <div className="rounded-2xl border border-zinc-800 bg-zinc-950/40 p-2 flex flex-col items-center gap-2 min-w-[120px]">
                  <div className="text-xs text-zinc-300">{stemIcon("vocals")} Vocals</div>
                  <VerticalFader value={vocalsVol} disabled={!audioReady} height={180} onChange={(v) => setVocalsVol(v)} />
                </div>
              </div>
            </>
          ) : null}

          {mixMode.kind === "master" ? (
            <>
              <div className="flex flex-nowrap gap-3 overflow-x-auto pb-2">
                {stemOrder.map((stem) => {
                  const st = stemLoad[stem] || { status: "idle" as const };
                  let sub = "";
                  if (st.status === "loading") {
                    const pct = st.total ? Math.round((st.loaded / Math.max(1, st.total)) * 100) : null;
                    sub = pct != null ? `loading ${pct}%` : `loading ${Math.round(st.loaded / 1024)}KB`;
                  } else if (st.status === "decoding") {
                    sub = "decoding…";
                  } else if (st.status === "ready") {
                    sub = "ready";
                  } else if (st.status === "error") {
                    sub = `error: ${st.error}`;
                  }

                  return (
                    <div key={stem} className="rounded-2xl border border-zinc-800 bg-zinc-950/40 p-2 flex flex-col items-center gap-2 min-w-[120px]">
                      <div className="text-xs text-zinc-300 capitalize">{stemIcon(stem)} {stem}</div>
                      <VerticalFader
                        value={Number(stemVolumes[stem] ?? 100)}
                        disabled={!audioReady}
                        height={180}
                        onChange={(v) => setStemVolumes((prev) => ({ ...prev, [stem]: v }))}
                      />
                      {sub ? <div className="text-[11px] text-zinc-600 text-center">{sub}</div> : <div className="text-[11px] text-zinc-800">&nbsp;</div>}
                    </div>
                  );
                })}
              </div>
            </>
          ) : null}

          {/* Library */}
          {libraryOpen ? (
            <div className="mt-4 rounded-2xl border border-zinc-800 bg-zinc-950/70 p-4">
              <div className="flex items-center gap-2">
                <input
                  className="flex-1 bg-zinc-900 border border-zinc-700 rounded-lg px-3 py-2 text-sm"
                  placeholder="Preset name (optional)"
                  value={presetName}
                  onChange={(e) => setPresetName(e.target.value)}
                />
                <button onClick={savePreset} className="px-3 py-2 rounded-lg border border-zinc-700 text-sm">
                  Save current
                </button>
              </div>

              <div className="mt-3 space-y-2 max-h-[300px] overflow-auto pr-1">
                {presets.map((p) => (
                  <div key={p.id} className="flex items-start gap-3 rounded-xl border border-zinc-800 p-3">
                    {p.thumbnail_url ? (
                      // eslint-disable-next-line @next/next/no-img-element
                      <img src={p.thumbnail_url} alt="thumb" className="w-12 h-12 rounded-lg object-cover border border-zinc-800" />
                    ) : (
                      <div className="w-12 h-12 rounded-lg bg-zinc-900 border border-zinc-800" />
                    )}
                    <button onClick={() => loadPreset(p)} className="flex-1 text-left">
                      <div className="text-sm text-zinc-200">{p.name || p.title || p.video_id}</div>
                      <div className="text-[11px] text-zinc-600 break-all">{p.source_url}</div>
                    </button>
                    <div className="flex gap-2">
                      <button
                        onClick={async () => {
                          const nn = prompt("Rename preset", p.name || p.title || p.video_id);
                          if (!nn) return;
                          await fetch(`${BACKEND}/presets`, {
                            method: "POST",
                            headers: { "Content-Type": "application/json" },
                            body: JSON.stringify({
                              id: p.id,
                              name: nn,
                              source_url: p.source_url,
                              video_id: p.video_id,
                              title: p.title || null,
                              thumbnail_url: p.thumbnail_url || null,
                              semitones: 0,
                              master_volume: 1,
                              vocal_volume: 1,
                            }),
                          });
                          await refreshPresets();
                        }}
                        className="text-[11px] px-2 py-1 rounded-md border border-zinc-800 text-zinc-300"
                      >
                        Rename
                      </button>
                      <button
                        onClick={async () => {
                          if (!confirm(`Delete: ${p.name || p.title || p.video_id}?`)) return;
                          await fetch(`${BACKEND}/presets/${p.id}`, { method: "DELETE" });
                          await refreshPresets();
                        }}
                        className="text-[11px] px-2 py-1 rounded-md border border-red-900/60 text-red-300"
                      >
                        Delete
                      </button>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          ) : null}
        </div>
      </div>

      {/* Jobs modal removed */}
    </main>
  );
}
