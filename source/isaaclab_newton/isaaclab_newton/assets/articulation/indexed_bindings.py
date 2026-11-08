# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Stable Isaac Lab bindings for explicitly indexed Newton articulations.

Strided selections still alias Newton memory. Indexed selections keep stable dense
buffers for Lab kernels and publish writes through the original, complete model.
"""

from __future__ import annotations

from functools import wraps
from typing import Any

import warp as wp

from isaaclab.utils.warp.index_kernel import IndexKernelDispatcher

from isaaclab_newton.physics import NewtonManager as SimulationManager


@wp.kernel(enable_backward=False)
def _select_rows(ids: wp.array(dtype=Any), mask: wp.array(dtype=wp.bool)):
    mask[wp.int32(ids[wp.tid()])] = True


_SELECT_ROWS = IndexKernelDispatcher(_select_rows, ("ids",))


@wp.kernel(enable_backward=False)
def _gather_gravity(source: wp.array(dtype=wp.vec3), worlds: wp.array(dtype=int), target: wp.array(dtype=wp.vec3)):
    row = wp.tid()
    world = worlds[row]
    if world < 0:
        world = source.shape[0] - 1
    target[row] = source[world]


class IndexedBindings:
    """Track the full native selection arrays underlying Lab's sliced aliases."""

    def __init__(self, view):
        self.view = view
        self.model = SimulationManager.get_model()
        if view.model is not self.model:
            raise RuntimeError("Newton articulation model changed; rebuild the articulation view before rebinding.")
        if view.count_per_world != 1:
            raise ValueError("Isaac Lab articulations require one selected articulation per world.")
        self.enabled = (
            view.is_sparse
            or view.world_count != self.model.world_count
            or any(
                layout is not None and layout.uses_explicit_model_indices for layout in view.frequency_layouts.values()
            )
        )
        self.entries = {}
        self.writing = False
        self.row_mask = wp.zeros(view.count, dtype=wp.bool, device=view.device) if self.enabled else None

    def _source(self, kind):
        if SimulationManager.get_model() is not self.model:
            raise RuntimeError("Newton articulation model changed; rebuild the articulation view before rebinding.")
        return {
            "model": SimulationManager.get_model,
            "state": SimulationManager.get_state_0,
            "control": SimulationManager.get_control,
        }[kind]()

    def _bind(self, key, source, getter, setter):
        values = getter(source)
        if self.enabled and values is not None:
            # A sparse or mixed layout can still return native aliases for some
            # frequencies. Own every dense binding so state swaps never copy into
            # an inactive native State through one of those aliases.
            values = wp.clone(values)
            kind = (
                "model" if source is self.model else "control" if source is SimulationManager.get_control() else "state"
            )
            self.entries[key] = (values, kind, getter, setter)
            if kind == "state" and (other_state := SimulationManager.get_state_1()) is not None:
                getter(other_state)
        return values

    def get_attribute(self, name, source):
        return self._bind(
            name,
            source,
            lambda src: self.view.get_attribute(name, src),
            lambda dst, values, mask: self.view.set_attribute(name, dst, values, mask),
        )

    def get_root_transforms(self, source):
        if not self.view.is_floating_base:
            source = self.model
        return self._bind("root_pose", source, self.view.get_root_transforms, self.view.set_root_transforms)

    def get_root_velocities(self, source):
        return self._bind("root_velocity", source, self.view.get_root_velocities, self.view.set_root_velocities)

    def get_link_transforms(self, source):
        return self.get_attribute("body_q", source)

    def get_link_velocities(self, source):
        return self.get_attribute("body_qd", source)

    def get_dof_positions(self, source):
        return self.get_attribute("joint_q", source)

    def get_dof_velocities(self, source):
        return self.get_attribute("joint_qd", source)

    def refresh(self, *keys):
        if not self.enabled or self.writing:
            return
        for key in keys or self.entries:
            if key not in self.entries:
                continue
            bound, kind, getter, _ = self.entries[key]
            current = getter(self._source(kind))
            # Newton alternates State objects inside a decimation loop. Keep Lab's
            # aliases pinned even when the native getter uses a different cache entry.
            if current.ptr != bound.ptr:
                wp.copy(bound, current)

    def flush(self, *keys, mask=None):
        if not self.enabled:
            return
        for key in keys:
            if key in self.entries:
                bound, kind, _, setter = self.entries[key]
                setter(self._source(kind), bound, mask)

    def refresh_gravity(self, target):
        if self.enabled:
            wp.launch(
                _gather_gravity,
                dim=self.view.count,
                inputs=[self._source("model").gravity, self.view.world_ids],
                outputs=[target],
                device=self.view.device,
            )


def indexed_write(*keys):
    """Gather before a native Lab writer and scatter its selected rows on success."""

    def decorate(method):
        @wraps(method)
        def wrapped(self, *args, **kwargs):
            bindings = self.data._indexed_bindings
            if bindings.writing:
                return method(self, *args, **kwargs)
            # Reject unsafe selectors before the writer can touch either its
            # dense staging buffers or native arrays. Mask validation reads
            # metadata only and remains safe inside a captured step.
            env_ids, env_mask = self._validate_env_selection(
                env_ids=kwargs.get("env_ids"), env_mask=kwargs.get("env_mask")
            )
            if "env_ids" in kwargs:
                kwargs["env_ids"] = env_ids
            if "env_mask" in kwargs:
                kwargs["env_mask"] = env_mask
            if not bindings.enabled:
                return method(self, *args, **kwargs)
            # Writers may read related inertia, root frames or joint properties.
            bindings.refresh()
            self.data._gather_joint_coordinates()
            bindings.writing = True
            try:
                result = method(self, *args, **kwargs)
                mask = kwargs.get("env_mask")
                if mask is None and kwargs.get("env_ids") is not None:
                    ids = self._resolve_env_ids(kwargs["env_ids"])
                    mask = bindings.row_mask
                    mask.zero_()
                    wp.launch(
                        _SELECT_ROWS.select(ids), dim=ids.shape[0], inputs=[ids], outputs=[mask], device=self.device
                    )
                bindings.flush(*keys, mask=mask)
                # Fixed-root writes notify the solver synchronously in the native
                # implementation; repeat after scatter so it sees the new transform.
                if "root_pose" in keys and not self.root_view.is_floating_base:
                    from newton import ModelFlags

                    if SimulationManager._solver is not None:
                        SimulationManager._solver.notify_model_changed(ModelFlags.JOINT_PROPERTIES)
                return result
            finally:
                bindings.writing = False

        return wrapped

    return decorate
