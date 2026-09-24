"""Choose which evaluated checkpoints to keep."""

from __future__ import annotations

import os
import re
from typing import Dict, List, Sequence, Tuple

CKPT_NAME = re.compile(r"^ckpt_steps_(\d+)\.pt$")


def checkpoint_steps(path: str) -> int:
    match = CKPT_NAME.match(os.path.basename(path))
    if not match:
        raise ValueError(f"not a training checkpoint: {path}")
    return int(match.group(1))


def checkpoint_payload(model, optimizer, steps: int, **extra) -> dict:
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "steps": int(steps),
    }
    payload.update(extra)
    return payload


def restore_checkpoint(path: str, model, optimizer, map_location) -> tuple:
    """Load weights and, when present, optimizer state.

    Returns ``(steps, optimizer_restored)``. Checkpoints written before
    optimizer state was saved restore the network only.
    """
    import torch

    payload = torch.load(path, map_location=map_location)
    model.load_state_dict(payload["model"])
    steps = int(payload.get("steps", checkpoint_steps(path)))
    optimizer_state = payload.get("optimizer")
    if optimizer is not None and optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)
        return steps, True
    return steps, False


def rank_checkpoints(records: Sequence[Dict], keep: int = 2) -> Tuple[List[Dict], List[Dict]]:
    """Split evaluated records into the ones to keep and the ones to delete.

    Higher success rate wins. Equal success rates are broken by SPL, then by
    the training step, so a later checkpoint of the same quality stays.
    """
    ordered = sorted(
        records,
        key=lambda record: (
            float(record["sr"]),
            float(record["spl"]),
            int(record["steps"]),
        ),
        reverse=True,
    )
    limit = max(0, int(keep))
    return list(ordered[:limit]), list(ordered[limit:])
