from fastapi import HTTPException

from db import get_preset


def get_preset_or_404(data_dir: str, preset_id: str) -> dict:
    p = get_preset(data_dir, preset_id)
    if not p:
        raise HTTPException(status_code=404, detail="Preset not found")
    return p
