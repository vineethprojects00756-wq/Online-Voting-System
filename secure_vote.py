import base64
import hashlib
import json
import os
from datetime import UTC, datetime
from uuid import uuid4


VOTE_ENCRYPTION_KEY = os.getenv("VOTE_ENCRYPTION_KEY", "smart-voting-secret-key")


def _build_key_stream(length: int) -> bytes:
    seed = hashlib.sha256(VOTE_ENCRYPTION_KEY.encode("utf-8")).digest()
    stream = bytearray()
    counter = 0

    while len(stream) < length:
        block = hashlib.sha256(seed + counter.to_bytes(4, "big")).digest()
        stream.extend(block)
        counter += 1

    return bytes(stream[:length])


def encrypt_vote(candidate_id: str, constituency: str) -> dict:
    payload = json.dumps(
        {
            "candidate_id": candidate_id,
            "constituency": constituency,
            "timestamp": datetime.now(UTC).isoformat(),
        },
        separators=(",", ":"),
    ).encode("utf-8")
    key_stream = _build_key_stream(len(payload))
    cipher_bytes = bytes(value ^ key_stream[index] for index, value in enumerate(payload))
    return {
        "vote_id": f"VT-{uuid4().hex[:12].upper()}",
        "encrypted_candidate_id": base64.urlsafe_b64encode(cipher_bytes).decode("ascii"),
    }


def decrypt_vote(encrypted_candidate_id: str) -> dict:
    cipher_bytes = base64.urlsafe_b64decode(encrypted_candidate_id.encode("ascii"))
    key_stream = _build_key_stream(len(cipher_bytes))
    payload = bytes(value ^ key_stream[index] for index, value in enumerate(cipher_bytes))
    return json.loads(payload.decode("utf-8"))
