# Node

See the [Core architecture](../../architecture.md) for the authoritative object,
type, dependency, state and execution contract.

`Node.align_frames(*frames, join="outer")` aligns sparse frames to the union of
their keys, including keys present only in later frames. Missing values remain
null. The default `join="inner"` keeps the shared keys. Key columns are coalesced across all
inputs, so aligning three or more frames does not introduce duplicate key
columns. Empty inputs retain their declared key and value types.
