# Implementation guide

See the implemented [architecture](../architecture.md).

`node.py` and `domain.py` hold typed values and coordinates; `operator/` is the
single operator registry and numerical module tree. `logical.py`, `graph.py` and
`dsl.py` share validation and normalized definitions. `updates.py` plans ready
nodes; `incremental.py` implements canonical finite blocks and verified
checkpoint replay. `store.py` owns publication, integrity, lifecycle and receipt
recovery. Private modules are not application integration APIs.
