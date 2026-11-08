* Fixed Newton articulation state, control, and model-property bindings for identical robots in worlds with different
  surrounding topology, preserving strided aliases for regular layouts and using explicit native indices for irregular layouts.
* Preserved selected-world resets, actuator telemetry, gravity queries, and state-buffer isolation, and released
  model-specific callbacks before articulation reinitialization.
