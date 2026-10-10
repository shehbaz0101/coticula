"""Load and verify the frozen study protocol."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "research" / "protocols" / "protocol.json"
HASH_PATH = ROOT / "research" / "protocols" / "protocol.sha256"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_protocol() -> dict:
    return json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))


def verify_frozen() -> tuple[dict, str]:
    """Refuse to score if the protocol bytes do not match the frozen digest."""
    digest = sha256_file(PROTOCOL_PATH)
    frozen = HASH_PATH.read_text(encoding="utf-8").strip()
    if digest != frozen:
        raise SystemExit(
            "Refusing to score: protocol SHA-256 does not match the frozen digest. "
            f"file={digest} frozen={frozen}"
        )
    return load_protocol(), digest
