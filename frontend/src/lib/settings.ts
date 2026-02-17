export type UiSettings = {
  bgOpacity: number;        // 0..1
  bgBlurPx: number;         // px
  bgSaturate: number;       // 0.5..2.5
  overlayOpacity: number;   // 0..1
  vignetteTop: number;      // 0..1
  vignetteBottom: number;   // 0..1
  vignetteSides: number;    // 0..1
};

export const DEFAULT_SETTINGS: UiSettings = {
  bgOpacity: 0.55,
  bgBlurPx: 20,
  bgSaturate: 1.4,
  overlayOpacity: 0.25,
  vignetteTop: 0.55,
  vignetteBottom: 0.65,
  vignetteSides: 0.55,
};

const KEY = 'karaoke_ui_settings_v1';

export function loadSettings(): UiSettings {
  if (typeof window === 'undefined') return DEFAULT_SETTINGS;
  try {
    const raw = window.localStorage.getItem(KEY);
    if (!raw) return DEFAULT_SETTINGS;
    const parsed = JSON.parse(raw);
    return { ...DEFAULT_SETTINGS, ...parsed } as UiSettings;
  } catch {
    return DEFAULT_SETTINGS;
  }
}

export function saveSettings(s: UiSettings) {
  if (typeof window === 'undefined') return;
  window.localStorage.setItem(KEY, JSON.stringify(s));
}
