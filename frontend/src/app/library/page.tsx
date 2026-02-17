"use client";

import React, { useEffect, useMemo, useState } from "react";
import { useRouter } from "next/navigation";

const BACKEND = process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8011";

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
};

export default function LibraryPage() {
  const router = useRouter();
  const [presets, setPresets] = useState<Preset[]>([]);
  const [query, setQuery] = useState("");
  const [loading, setLoading] = useState(false);

  const [sort, setSort] = useState<"recent" | "bpm_asc" | "bpm_desc" | "key_asc" | "key_desc">("recent");

  const refresh = async () => {
    setLoading(true);
    try {
      const r = await fetch(`${BACKEND}/presets?sort=${encodeURIComponent(sort)}`);
      const j = await r.json();
      setPresets((j.items || []) as Preset[]);
    } catch {
      setPresets([]);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    refresh();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sort]);

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return presets;
    return presets.filter((p) => {
      const hay = `${p.name || ""} ${p.title || ""} ${p.video_id || ""} ${p.source_url || ""}`.toLowerCase();
      return hay.includes(q);
    });
  }, [presets, query]);

  const openPreset = (p: Preset) => {
    const url = p.source_url || `https://www.youtube.com/watch?v=${p.video_id}`;
    const bpm = p.bpm != null ? String(p.bpm) : "";
    const qp = new URLSearchParams();
    qp.set("url", url);
    if (bpm) qp.set("bpm", bpm);
    router.push(`/?${qp.toString()}`);
  };

  const renamePreset = async (p: Preset) => {
    const nn = prompt("Rename", p.name || p.title || p.video_id);
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
        bpm: p.bpm ?? null,
        semitones: 0,
        master_volume: 1.0,
        vocal_volume: 1.0,
      }),
    });
    await refresh();
  };

  const deletePreset = async (p: Preset) => {
    const ok = confirm(`Delete from Library?\n\n${p.name || p.title || p.video_id}`);
    if (!ok) return;
    await fetch(`${BACKEND}/presets/${encodeURIComponent(p.id)}`, { method: "DELETE" });
    await refresh();
  };

  return (
    <main className="min-h-screen bg-black text-white">
      <div className="sticky top-0 z-10 border-b border-zinc-800 bg-zinc-950/90 backdrop-blur">
        <div className="max-w-6xl mx-auto px-4 py-4 flex items-center gap-3">
          <button onClick={() => router.push("/")} className="px-3 py-2 rounded-lg border border-zinc-700 text-sm">
            Back
          </button>
          <div className="text-sm text-zinc-200 font-semibold">Library</div>
          <div className="flex-1" />

          <select
            value={sort}
            onChange={(e) => setSort(e.target.value as any)}
            className="px-3 py-2 rounded-lg border border-zinc-700 text-sm bg-zinc-950"
          >
            <option value="recent">Recent</option>
            <option value="bpm_asc">BPM (slow → fast)</option>
            <option value="bpm_desc">BPM (fast → slow)</option>
            <option value="key_asc">Key (C → B)</option>
            <option value="key_desc">Key (B → C)</option>
          </select>

          <button onClick={refresh} className="px-3 py-2 rounded-lg border border-zinc-700 text-sm">
            {loading ? "Refreshing…" : "Refresh"}
          </button>
        </div>
        <div className="max-w-6xl mx-auto px-4 pb-4">
          <input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search…"
            className="w-full bg-zinc-900 border border-zinc-700 rounded-xl px-4 py-3 text-sm"
          />
        </div>
      </div>

      <div className="max-w-6xl mx-auto px-4 py-6">
        {filtered.length === 0 ? (
          <div className="text-zinc-500 text-sm">No items.</div>
        ) : null}

        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-4">
          {filtered.map((p) => (
            <div key={p.id} className="relative rounded-2xl border border-zinc-800 bg-zinc-950/60 hover:bg-zinc-900/60 overflow-hidden">
              <button onClick={() => openPreset(p)} className="block w-full text-left">
                <div className="aspect-square bg-zinc-900">
                  {p.thumbnail_url ? (
                    // eslint-disable-next-line @next/next/no-img-element
                    <img src={p.thumbnail_url} alt="thumb" className="w-full h-full object-cover" />
                  ) : null}
                </div>
                <div className="p-3 space-y-1">
                  <div className="text-sm text-zinc-100 line-clamp-2">{p.title || p.name || p.video_id}</div>
                  <div className="text-[11px] text-zinc-500 line-clamp-1">{p.video_id}</div>
                  <div className="flex items-center gap-2 pt-1 flex-wrap">
                    {p.bpm != null ? <div className="text-[11px] px-2 py-0.5 rounded-full border border-zinc-800 text-zinc-300">{p.bpm} BPM</div> : null}
                    {p.key_tonic && p.key_mode ? (
                      <div className="text-[11px] px-2 py-0.5 rounded-full border border-zinc-800 text-zinc-300">
                        {p.key_tonic}{p.key_mode === "minor" ? "m" : "M"}
                      </div>
                    ) : null}
                  </div>
                </div>
              </button>

              <div className="absolute top-2 right-2 flex gap-2">
                <button
                  onClick={() => renamePreset(p)}
                  className="text-[11px] px-2 py-1 rounded-md border border-zinc-800 bg-zinc-950/70"
                >
                  Rename
                </button>
                <button
                  onClick={() => deletePreset(p)}
                  className="text-[11px] px-2 py-1 rounded-md border border-red-900 bg-red-950/40 text-red-200"
                >
                  Delete
                </button>
              </div>
            </div>
          ))}
        </div>
      </div>
    </main>
  );
}
