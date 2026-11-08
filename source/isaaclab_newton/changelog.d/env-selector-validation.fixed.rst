* Reject out-of-range, duplicate, or malformed environment selectors before Newton articulation state writes and resets.
  Boolean-mask writes remain capture-safe; explicit CUDA index selectors are validated before writing and must be used outside capture.
