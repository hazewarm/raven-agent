from __future__ import annotations

import asyncio
from pathlib import Path

from raven_agent.channels.base import AttachmentStore, MessageDeduper, SessionIdentityIndex
from raven_agent.session import SessionManager
from raven_agent.session_store import SessionStore


def test_attachment_store_writes_under_configured_root(tmp_path: Path) -> None:
    store = AttachmentStore(tmp_path / "uploads")

    path = store.write_bytes(b"hello", prefix="img_", suffix=".png")

    assert path.is_relative_to(tmp_path / "uploads")
    assert path.suffix == ".png"
    assert path.read_bytes() == b"hello"


def test_message_deduper_evicts_oldest_keys() -> None:
    deduper = MessageDeduper(max_size=2)

    assert deduper.seen("a") is False
    assert deduper.seen("b") is False
    assert deduper.seen("a") is True
    assert deduper.seen("c") is False
    assert deduper.seen("a") is False


def test_session_identity_index_rebuilds_and_persists_metadata(tmp_path: Path) -> None:
    async def run() -> None:
        manager = SessionManager(SessionStore(tmp_path / "sessions.db"))
        try:
            existing = manager.get_or_create("telegram:123")
            existing.metadata["username"] = "alice"
            manager.save(existing)

            index = SessionIdentityIndex(
                manager,
                channel="telegram",
                metadata_key="username",
                normalizer=lambda value: value.lower().removeprefix("@"),
            )

            assert index.rebuild() == {"alice": "123"}
            assert index.resolve("@ALICE") == "123"

            await index.remember("@Bob", "456")
            assert index.resolve("bob") == "456"
            saved = manager.get_or_create("telegram:456")
            assert saved.metadata["username"] == "bob"
        finally:
            manager.close()

    asyncio.run(run())
