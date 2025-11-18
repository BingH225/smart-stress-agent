from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional


@dataclass
class RagDocument:
    id: Optional[str]
    content: str
    source: Optional[str] = None
    section: Optional[str] = None
    created_at: str = field(
        default_factory=lambda: datetime.utcnow().isoformat() + "Z"
    )
    tags: List[str] = field(default_factory=list)



