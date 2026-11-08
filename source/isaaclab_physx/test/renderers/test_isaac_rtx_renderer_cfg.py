# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for Isaac RTX renderer configuration."""

import isaaclab_physx.renderers as renderers
import pytest
from isaaclab_physx.renderers.isaac_rtx_renderer_cfg import IsaacRtxRendererGlobalSettingsCfg


def test_global_settings_cfg_is_publicly_exported():
    """Test that the global settings config is exported from the renderer package."""
    assert renderers.IsaacRtxRendererGlobalSettingsCfg is IsaacRtxRendererGlobalSettingsCfg
    assert "IsaacRtxRendererGlobalSettingsCfg" in renderers.__all__


@pytest.mark.parametrize("columns", [0, -1, True, 2.5, "5"])
def test_invalid_tile_columns_are_rejected(columns):
    """Invalid native grid sizes must fail configuration validation before Kit starts."""
    with pytest.raises(ValueError, match="tile_columns.*positive integer"):
        renderers.IsaacRtxRendererCfg(tile_columns=columns).validate()
