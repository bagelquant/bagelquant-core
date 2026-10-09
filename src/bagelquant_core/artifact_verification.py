"""Finite, thread-safe checksum proofs for immutable local artifacts."""
from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable
from pathlib import Path
from threading import RLock


class ArtifactVerification:
    """Deduplicate successful checks only while this operation remains open.

    File identity is checked before every hit and after hashing. Closing or any
    verification failure discards proofs. This is never a persisted certificate.
    """

    def __init__(self, *, max_entries: int = 4096):
        if max_entries < 1:
            raise ValueError("verification capacity must be positive")
        self._proofs: OrderedDict[tuple, None] = OrderedDict()
        self._lock = RLock()
        self._open = True
        self._max_entries = max_entries

    @staticmethod
    def _signature(path: Path) -> tuple:
        stat = path.stat()
        return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)

    def verify(self, path: Path, expected: str, *, receipt: str,
               checksum: Callable[[Path], str]) -> None:
        """Hash unchanged bytes once per exact receipt and file identity."""
        with self._lock:
            try:
                before = self._signature(path)
                key = (path, expected, receipt, before)
                if self._open and key in self._proofs:
                    self._proofs.move_to_end(key)
                    return
                if checksum(path) != expected or self._signature(path) != before:
                    raise ValueError(f"corrupt or changing artifact: {path.name}")
                if self._open:
                    self._proofs[key] = None
                    while len(self._proofs) > self._max_entries:
                        self._proofs.popitem(last=False)
            except BaseException:
                self._proofs.clear()
                raise

    def close(self) -> None:
        """Expire every proof, including on a retained reader object."""
        with self._lock:
            self._open = False
            self._proofs.clear()
