"""Read committed SQLite metadata with explicit offline or runtime coordination."""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import shutil
import sqlite3
import tempfile
from typing import Iterator


def _signature(path: Path) -> tuple[int, ...] | None:
    try:
        stat = path.stat()
    except FileNotFoundError:
        return None
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


@contextmanager
def open_metadata_snapshot(path: Path, *, runtime: bool = False) -> Iterator[sqlite3.Connection]:
    """Yield a transient read-only committed metadata view.

    Copy only metadata and its WAL into a private system temporary directory.
    SQLite reconstructs the WAL index there. Stable source signatures prevent
    accepting a view copied across a concurrent commit or checkpoint. No backend
    artifacts, recovery operations or durable secondary authority are involved.
    Explicit ``runtime=True`` uses ordinary SQLite read coordination on the
    source, including WAL/SHM, instead of requiring a stable offline copy. A
    read-only connection never initializes storage or recovers a hot journal.
    """
    if runtime:
        connection = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
        try:
            connection.execute("PRAGMA query_only=ON")
            connection.execute("BEGIN")
            # BEGIN alone does not acquire a snapshot. Acquire SQLite's shared
            # read lock now so every caller query sees the same committed view.
            # Recovery-required hot journals fail on this read-only connection.
            connection.execute("SELECT name FROM sqlite_master LIMIT 1").fetchall()
            yield connection
        finally:
            connection.close()
        return
    wal = path.with_name(path.name + "-wal")
    journal = path.with_name(path.name + "-journal")
    for _attempt in range(3):
        with tempfile.TemporaryDirectory(prefix="bagelquant-metadata-inspect-") as temporary:
            before = (_signature(path), _signature(wal), _signature(journal))
            if before[0] is None:
                raise FileNotFoundError(path)
            # An active/hot rollback journal may protect uncommitted main-DB
            # pages. Inspection must neither read them as committed nor recover
            # the journal. PERSIST's invalidated zero header is harmless.
            if before[2] is not None:
                try:
                    with journal.open("rb") as stream:
                        if any(stream.read(8)):
                            raise sqlite3.OperationalError("metadata has an active rollback journal")
                except FileNotFoundError:
                    continue
            target = Path(temporary) / "meta.sqlite"
            try:
                shutil.copyfile(path, target)
                if before[1] is not None:
                    shutil.copyfile(wal, target.with_name(target.name + "-wal"))
            except FileNotFoundError:
                continue
            if before != (_signature(path), _signature(wal), _signature(journal)):
                continue
            connection = sqlite3.connect(target.as_uri() + "?mode=ro", uri=True)
            try:
                yield connection
            finally:
                connection.close()
            return
    raise sqlite3.OperationalError("metadata changed during read-only inspection")
