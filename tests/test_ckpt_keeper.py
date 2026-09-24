import os

import torch
from torch import nn

from spikingnav.habitat_objectnav.ckpt_keeper import (
    checkpoint_payload,
    checkpoint_steps,
    rank_checkpoints,
    restore_checkpoint,
)


def test_checkpoint_steps_reads_the_filename():
    assert checkpoint_steps("/tmp/ckpt_steps_12800.pt") == 12800


def test_rank_keeps_the_two_highest_success_rates(tmp_path):
    records = [
        {"path": "ckpt_steps_100.pt", "sr": 0.10, "spl": 0.90, "steps": 100},
        {"path": "ckpt_steps_200.pt", "sr": 0.40, "spl": 0.10, "steps": 200},
        {"path": "ckpt_steps_300.pt", "sr": 0.40, "spl": 0.20, "steps": 300},
        {"path": "ckpt_steps_400.pt", "sr": 0.25, "spl": 0.80, "steps": 400},
    ]
    kept, dropped = rank_checkpoints(records, keep=2)
    assert [item["path"] for item in kept] == [
        "ckpt_steps_300.pt",
        "ckpt_steps_200.pt",
    ]
    assert [item["path"] for item in dropped] == [
        "ckpt_steps_400.pt",
        "ckpt_steps_100.pt",
    ]
    assert os.path.basename(kept[0]["path"]).startswith("ckpt_steps_")


def test_restore_keeps_steps_when_optimizer_is_absent(tmp_path):
    source = nn.Linear(2, 2)
    with torch.no_grad():
        source.weight.fill_(3.0)
    path = tmp_path / "ckpt_steps_358400.pt"
    torch.save({"model": source.state_dict(), "steps": 358400}, path)
    target = nn.Linear(2, 2)
    optimizer = torch.optim.Adam(target.parameters(), lr=0.1)
    steps, restored = restore_checkpoint(str(path), target, optimizer, "cpu")
    assert steps == 358400
    assert restored is False
    assert torch.equal(target.weight, source.weight)


def test_restore_loads_optimizer_when_present(tmp_path):
    source = nn.Linear(2, 2)
    optimizer = torch.optim.Adam(source.parameters(), lr=0.1)
    source.weight.sum().backward()
    optimizer.step()
    path = tmp_path / "ckpt_steps_10.pt"
    torch.save(checkpoint_payload(source, optimizer, 10, algo="ppo"), path)
    target = nn.Linear(2, 2)
    resumed = torch.optim.Adam(target.parameters(), lr=0.1)
    steps, restored = restore_checkpoint(str(path), target, resumed, "cpu")
    assert steps == 10
    assert restored is True
    assert resumed.state_dict()["state"]
