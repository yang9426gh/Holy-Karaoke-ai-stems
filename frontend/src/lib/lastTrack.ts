export type LastTrackMeta = {
  coverUrl?: string | null;
  title?: string | null;
  artist?: string | null;
};

const KEY = 'karaoke_last_track_meta_v1';

export function saveLastTrack(meta: LastTrackMeta) {
  if (typeof window === 'undefined') return;
  try {
    window.localStorage.setItem(KEY, JSON.stringify(meta));
  } catch {}
}

export function loadLastTrack(): LastTrackMeta {
  if (typeof window === 'undefined') return {};
  try {
    const raw = window.localStorage.getItem(KEY);
    if (!raw) return {};
    return JSON.parse(raw) as LastTrackMeta;
  } catch {
    return {};
  }
}
