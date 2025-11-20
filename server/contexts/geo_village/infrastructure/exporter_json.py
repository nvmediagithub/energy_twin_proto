from __future__ import annotations
import json
from dataclasses import asdict
from ..domain.models import Village

def village_to_json(v: Village) -> str:
    return json.dumps(asdict(v), ensure_ascii=False, indent=2)
