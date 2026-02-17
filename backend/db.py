import os
import sqlite3
from typing import Optional


def db_path(data_dir: str) -> str:
    return os.path.join(data_dir, "index.db")


def init_db(data_dir: str) -> None:
    os.makedirs(data_dir, exist_ok=True)
    path = db_path(data_dir)
    con = sqlite3.connect(path)
    try:
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS tracks (
              video_id TEXT PRIMARY KEY,
              source_url TEXT,
              task_id TEXT,
              title TEXT,
              artist TEXT,
              cover_url TEXT,
              vocals_path TEXT,
              no_vocals_path TEXT,
              lyrics_json_path TEXT,
              created_at INTEGER DEFAULT (strftime('%s','now')),
              updated_at INTEGER DEFAULT (strftime('%s','now'))
            );
            """
        )

        # lightweight migrations (add columns if this DB was created before they existed)
        cols = {row[1] for row in con.execute("PRAGMA table_info(tracks)").fetchall()}
        for name, ddl in [
            ("title", "ALTER TABLE tracks ADD COLUMN title TEXT"),
            ("artist", "ALTER TABLE tracks ADD COLUMN artist TEXT"),
            ("cover_url", "ALTER TABLE tracks ADD COLUMN cover_url TEXT"),
        ]:
            if name not in cols:
                con.execute(ddl)

        con.execute(
            """
            CREATE TABLE IF NOT EXISTS presets (
              id TEXT PRIMARY KEY,
              name TEXT,
              source_url TEXT,
              video_id TEXT,
              title TEXT,
              thumbnail_url TEXT,
              semitones INTEGER DEFAULT 0,
              master_volume REAL DEFAULT 1.0,
              vocal_volume REAL DEFAULT 1.0,
              created_at INTEGER DEFAULT (strftime('%s','now')),
              updated_at INTEGER DEFAULT (strftime('%s','now'))
            );
            """
        )

        # migrations for presets
        pcols = {row[1] for row in con.execute("PRAGMA table_info(presets)").fetchall()}
        for name, ddl in [
            ("title", "ALTER TABLE presets ADD COLUMN title TEXT"),
            ("thumbnail_url", "ALTER TABLE presets ADD COLUMN thumbnail_url TEXT"),
            ("bpm", "ALTER TABLE presets ADD COLUMN bpm REAL"),
            ("key_tonic", "ALTER TABLE presets ADD COLUMN key_tonic TEXT"),
            ("key_mode", "ALTER TABLE presets ADD COLUMN key_mode TEXT"),
        ]:
            if name not in pcols:
                con.execute(ddl)

        con.commit()
    finally:
        con.close()


def upsert_track(
    data_dir: str,
    *,
    video_id: str,
    source_url: str,
    task_id: str,
    vocals_path: str,
    no_vocals_path: str,
    title: Optional[str] = None,
    artist: Optional[str] = None,
    cover_url: Optional[str] = None,
    lyrics_json_path: Optional[str] = None,
) -> None:
    con = sqlite3.connect(db_path(data_dir))
    try:
        con.execute(
            """
            INSERT INTO tracks(video_id, source_url, task_id, title, artist, cover_url, vocals_path, no_vocals_path, lyrics_json_path)
            VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(video_id) DO UPDATE SET
              source_url=excluded.source_url,
              task_id=excluded.task_id,
              title=COALESCE(excluded.title, tracks.title),
              artist=COALESCE(excluded.artist, tracks.artist),
              cover_url=COALESCE(excluded.cover_url, tracks.cover_url),
              vocals_path=excluded.vocals_path,
              no_vocals_path=excluded.no_vocals_path,
              lyrics_json_path=COALESCE(excluded.lyrics_json_path, tracks.lyrics_json_path),
              updated_at=strftime('%s','now');
            """
        , (video_id, source_url, task_id, title, artist, cover_url, vocals_path, no_vocals_path, lyrics_json_path))
        con.commit()
    finally:
        con.close()


def get_track(data_dir: str, video_id: str) -> Optional[dict]:
    con = sqlite3.connect(db_path(data_dir))
    con.row_factory = sqlite3.Row
    try:
        row = con.execute("SELECT * FROM tracks WHERE video_id=?", (video_id,)).fetchone()
        if not row:
            return None
        return dict(row)
    finally:
        con.close()


def get_preset(data_dir: str, id: str) -> Optional[dict]:
    con = sqlite3.connect(db_path(data_dir))
    con.row_factory = sqlite3.Row
    try:
        row = con.execute("SELECT * FROM presets WHERE id=?", (id,)).fetchone()
        if not row:
            return None
        return dict(row)
    finally:
        con.close()


def list_tracks(data_dir: str, limit: int = 50) -> list[dict]:
    con = sqlite3.connect(db_path(data_dir))
    con.row_factory = sqlite3.Row
    try:
        rows = con.execute(
            "SELECT * FROM tracks ORDER BY updated_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        con.close()


def upsert_preset(
    data_dir: str,
    *,
    id: str,
    name: str,
    source_url: str,
    video_id: str,
    title: Optional[str] = None,
    thumbnail_url: Optional[str] = None,
    bpm: Optional[float] = None,
    key_tonic: Optional[str] = None,
    key_mode: Optional[str] = None,
    semitones: int = 0,
    master_volume: float = 1.0,
    vocal_volume: float = 1.0,
) -> None:
    con = sqlite3.connect(db_path(data_dir))
    try:
        con.execute(
            """
            INSERT INTO presets(id, name, source_url, video_id, title, thumbnail_url, bpm, key_tonic, key_mode, semitones, master_volume, vocal_volume)
            VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
              name=excluded.name,
              source_url=excluded.source_url,
              video_id=excluded.video_id,
              title=COALESCE(excluded.title, presets.title),
              thumbnail_url=COALESCE(excluded.thumbnail_url, presets.thumbnail_url),
              bpm=COALESCE(excluded.bpm, presets.bpm),
              key_tonic=COALESCE(excluded.key_tonic, presets.key_tonic),
              key_mode=COALESCE(excluded.key_mode, presets.key_mode),
              semitones=excluded.semitones,
              master_volume=excluded.master_volume,
              vocal_volume=excluded.vocal_volume,
              updated_at=strftime('%s','now');
            """
        , (id, name, source_url, video_id, title, thumbnail_url, bpm, key_tonic, key_mode, int(semitones), float(master_volume), float(vocal_volume)))
        con.commit()
    finally:
        con.close()


def delete_preset(data_dir: str, id: str) -> Optional[dict]:
    """Delete preset and return the deleted row (so callers can cascade cleanup)."""
    con = sqlite3.connect(db_path(data_dir))
    con.row_factory = sqlite3.Row
    try:
        row = con.execute("SELECT * FROM presets WHERE id=?", (id,)).fetchone()
        con.execute("DELETE FROM presets WHERE id=?", (id,))
        con.commit()
        return dict(row) if row else None
    finally:
        con.close()


def delete_track(data_dir: str, video_id: str) -> None:
    con = sqlite3.connect(db_path(data_dir))
    try:
        con.execute("DELETE FROM tracks WHERE video_id=?", (video_id,))
        con.commit()
    finally:
        con.close()


def list_presets(data_dir: str, limit: int = 100) -> list[dict]:
    con = sqlite3.connect(db_path(data_dir))
    con.row_factory = sqlite3.Row
    try:
        rows = con.execute(
            "SELECT * FROM presets ORDER BY updated_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        con.close()
