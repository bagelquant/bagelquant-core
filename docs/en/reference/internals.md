# Implementation guide

See the implemented [architecture](../architecture.md).

`node.py` and `domain.py` hold typed values and coordinates; `operator/` is the
single operator registry and numerical module tree. `logical.py`, `graph.py` and
`dsl.py` share validation and normalized definitions. `updates.py` plans ready
nodes; `incremental.py` implements canonical finite blocks and verified
checkpoint replay. `store.py` owns publication, integrity, lifecycle and receipt
recovery. Private modules are not application integration APIs.

## Read-only storage inspection

`CoreStore(meta_path, artifact_path).inspect()` reports `uninitialized`, `ready`
or `incompatible`, with schema version and a reason where available. Inspection
checks the version, required tables and required columns. Matching a version
alone does not make an incomplete schema ready. It reads a transient metadata
snapshot through the public `bagelquant_core.inspection.open_metadata_snapshot`
primitive, including committed WAL contents. Source signatures must remain
stable during copying; a changing store is retried and then reported unreadable.
SQLite creates any WAL index only in the private system temporary directory,
which is removed after inspection. No source directory, database, sidecar or
recovery state is created or changed. Explicit `initialize()` remains the only
empty-store creation path. Active or hot rollback journals make inspection
unreadable rather than exposing uncommitted pages or triggering recovery;
invalidated zero-header PERSIST journals are safe to inspect.
