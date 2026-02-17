'use client';

import React from 'react';
import Link from 'next/link';
import { useSettings } from '../../lib/useSettings';
import { loadLastTrack } from '../../lib/lastTrack';

function Row({ label, value, min, max, step, onChange }: {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (v: number) => void;
}) {
  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <div className="text-sm font-semibold text-zinc-200">{label}</div>
        <div className="text-xs font-mono text-zinc-400">{value.toFixed(2)}</div>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className="w-full"
      />
    </div>
  );
}

export default function SettingsPage() {
  const { settings, patch, reset } = useSettings();
  const meta = loadLastTrack();

  return (
    <main className="min-h-screen bg-black text-white">
      <div className="max-w-3xl mx-auto p-8 space-y-6">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-black">Settings</h1>
            <p className="text-zinc-400 text-sm">UI tuning for album-art background + readability</p>
          </div>
          <div className="flex gap-2">
            <button
              onClick={reset}
              className="text-sm px-3 py-2 rounded-lg border border-zinc-800 hover:bg-zinc-900"
            >
              Reset
            </button>
            <Link
              href="/"
              className="text-sm px-3 py-2 rounded-lg border border-zinc-800 hover:bg-zinc-900"
            >
              Back
            </Link>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="rounded-xl border border-zinc-800 bg-zinc-950/60 p-5 space-y-5">
            <div className="text-sm font-semibold text-zinc-200">Album-art background</div>
            <Row label="Background opacity" value={settings.bgOpacity} min={0} max={1} step={0.01} onChange={(v) => patch({ bgOpacity: v })} />
            <Row label="Background blur (px)" value={settings.bgBlurPx} min={0} max={60} step={1} onChange={(v) => patch({ bgBlurPx: v })} />
            <Row label="Background saturation" value={settings.bgSaturate} min={0.5} max={2.5} step={0.05} onChange={(v) => patch({ bgSaturate: v })} />
          </div>

          <div className="rounded-xl border border-zinc-800 bg-zinc-950/60 p-5 space-y-5">
            <div className="text-sm font-semibold text-zinc-200">Readability overlays</div>
            <Row label="Overall dark overlay" value={settings.overlayOpacity} min={0} max={0.8} step={0.01} onChange={(v) => patch({ overlayOpacity: v })} />
            <Row label="Vignette top" value={settings.vignetteTop} min={0} max={1} step={0.01} onChange={(v) => patch({ vignetteTop: v })} />
            <Row label="Vignette bottom" value={settings.vignetteBottom} min={0} max={1} step={0.01} onChange={(v) => patch({ vignetteBottom: v })} />
            <Row label="Vignette sides" value={settings.vignetteSides} min={0} max={1} step={0.01} onChange={(v) => patch({ vignetteSides: v })} />
          </div>
        </div>

        <div className="rounded-xl border border-zinc-800 bg-zinc-950/60 p-5 space-y-3">
          <div className="text-sm font-semibold text-zinc-200">Preview</div>
          <p className="text-sm text-zinc-400">This preview uses the last loaded song's cover (saved locally). Tweak sliders until it looks right.</p>

          <div className="rounded-xl overflow-hidden border border-zinc-800 mt-3">
            <div className="relative h-44 bg-black">
              {meta.coverUrl && (
                <div
                  className="absolute inset-0 scale-110"
                  style={{
                    backgroundImage: `url(${meta.coverUrl})`,
                    backgroundSize: 'cover',
                    backgroundPosition: 'center',
                    opacity: settings.bgOpacity,
                    filter: `blur(${settings.bgBlurPx}px) saturate(${settings.bgSaturate})`,
                  }}
                />
              )}
              <div
                className="absolute inset-0"
                style={{
                  background: `linear-gradient(to bottom, rgba(0,0,0,${settings.vignetteTop}) 0%, rgba(0,0,0,0) 40%, rgba(0,0,0,${settings.vignetteBottom}) 100%)`,
                }}
              />
              <div
                className="absolute inset-0"
                style={{
                  background: `linear-gradient(to right, rgba(0,0,0,${settings.vignetteSides}) 0%, rgba(0,0,0,0) 50%, rgba(0,0,0,${settings.vignetteSides}) 100%)`,
                }}
              />
              <div className="absolute inset-0" style={{ backgroundColor: `rgba(0,0,0,${settings.overlayOpacity})` }} />

              <div className="absolute inset-0 flex flex-col justify-end p-4">
                <div className="text-sm font-semibold text-white/90">{meta.title || 'Load a song on the main page'}</div>
                <div className="text-xs text-white/60">{meta.artist || ''}</div>
              </div>
            </div>
          </div>

          <p className="text-xs text-zinc-500">Settings are saved in your browser (localStorage).</p>
        </div>
      </div>
    </main>
  );
}
