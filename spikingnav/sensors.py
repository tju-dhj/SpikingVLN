"""RGB sensor wrapper that applies RobustNav visual corruptions."""

from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np
from allenact_plugins.ithor_plugin.ithor_sensors import RGBSensorThor

from spikingnav.corruptions import apply_corruption_sequence


def maybe_corrupt_frame(
    frame: np.ndarray,
    corruptions: Optional[Sequence[str]] = None,
    severities: Optional[Sequence[int]] = None,
) -> np.ndarray:
    if not corruptions:
        return frame
    return apply_corruption_sequence(frame, corruptions, severities)


class CorruptibleRGBSensorThor(RGBSensorThor):
    def __init__(
        self,
        corruptions: Optional[Sequence[str]] = None,
        severities: Optional[Sequence[int]] = None,
        **kwargs: Any,
    ) -> None:
        self._sn_corruptions = list(corruptions or [])
        self._sn_severities = list(severities or [5] * len(self._sn_corruptions))
        super().__init__(**kwargs)

    def frame_from_env(self, env, task):  # type: ignore[override]
        frame = super().frame_from_env(env, task)
        return maybe_corrupt_frame(frame, self._sn_corruptions, self._sn_severities)


def make_rgb_sensor(
    height: int,
    width: int,
    uuid: str = "rgb_lowres",
    corruptions: Optional[Sequence[str]] = None,
    severities: Optional[Sequence[int]] = None,
    use_resnet_normalization: bool = True,
):
    return CorruptibleRGBSensorThor(
        height=height,
        width=width,
        use_resnet_normalization=use_resnet_normalization,
        uuid=uuid,
        corruptions=corruptions,
        severities=severities,
    )
