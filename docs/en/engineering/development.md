# Core ownership and engineering contracts

- Target ownership: Domain/Node conversion, numerical transforms/ML, graph
  execution and generic numerical artifact/materialization/cache/checkpoint
  persistence belong to Core. CoreStore implements caller-located SQLite/Parquet persistence; applications use public receipts.
- Accept neutral frames/schema/availability/immutable identity evidence through
  public APIs. Data owns dataset storage and frozen input evidence; Core does
  not query Data or provider state. BT owns account/evaluation artifact storage.
- Workbench owns China semantics, app metadata, research governance and task
  orchestration, referencing backend artifacts. Generic conversion, computation,
  numerical persistence and reuse proofs must not be reimplemented there.

Public filesystem accounting belongs to Core's `storage_usage` module and
`CoreStore.storage_usage(within_root=...)`; consumers compose stat inventories
without walking private artifacts. Preserve logical/allocated bytes, hidden and
unknown files, symlink-object semantics and deterministic inode deduplication.
This passive API never initializes storage or reads numerical bodies.
