import sqlite3
from typing import Optional

from db import db_path


def update_preset_meta(
    data_dir: str,
    *,
    id: str,
    bpm: Optional[float] = None,
    key_tonic: Optional[str] = None,
    key_mode: Optional[str] = None,
    title: Optional[str] = None,
    thumbnail_url: Optional[str] = None,
) -> None:
    con = sqlite3.connect(db_path(data_dir))
    try:
        con.execute(
            """
            UPDATE presets
            SET
              bpm = COALESCE(?, bpm),
              key_tonic = COALESCE(?, key_tonic),
              key_mode = COALESCE(?, key_mode),
              title = COALESCE(?, title),
              thumbnail_url = COALESCE(?, thumbnail_url),
              updated_at=strftime('%s','now')
            WHERE id=?;
            """
            , (bpm, key_tonic, key_mode, title, thumbnail_url, id)
        )
        con.commit()
    finally:
        con.close()
