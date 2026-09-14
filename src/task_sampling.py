"""Frozen, seeded task sampling for both environments.

Stratified by task type/family so that a small pilot is not accidentally
concentrated in one difficulty regime.
"""

from __future__ import annotations

import glob
import json
import os
import random

SEED = 42

ALFWORLD_TASK_TYPES = [
    "pick_and_place_simple",
    "look_at_obj_in_light",
    "pick_clean_then_place_in_recep",
    "pick_heat_then_place_in_recep",
    "pick_cool_then_place_in_recep",
    "pick_two_obj_and_place",
]

# Families whose deep state replayed identically across k=4 replicas
# (tests/test_scienceworld_determinism.py). The two excluded families
# (find-non-living-thing, grow-plant) carry latent stochastic simulator state.
SCIENCEWORLD_NONDETERMINISTIC = {"find-non-living-thing", "grow-plant"}


def alfworld_tasks(n: int, split: str = "valid_seen", seed: int = SEED) -> list[dict]:
    data = os.environ.get("ALFWORLD_DATA", os.path.expanduser("~/.cache/alfworld"))
    by_type: dict[str, list[str]] = {t: [] for t in ALFWORLD_TASK_TYPES}
    pattern = os.path.join(data, f"json_2.1.1/{split}/*/*/game.tw-pddl")
    for gf in sorted(glob.glob(pattern)):
        root = os.path.dirname(gf)
        if "movable" in root or "Sliced" in root:
            continue
        try:
            with open(gf) as f:
                if not json.load(f).get("solvable", False):
                    continue
        except Exception:
            continue
        tj = os.path.join(root, "traj_data.json")
        try:
            with open(tj) as f:
                ttype = json.load(f)["task_type"]
        except Exception:
            continue
        if ttype in by_type:
            by_type[ttype].append(gf)

    rng = random.Random(seed)
    for t in by_type:
        rng.shuffle(by_type[t])
    out, i = [], 0
    while len(out) < n:
        added = False
        for t in ALFWORLD_TASK_TYPES:
            if len(out) >= n:
                break
            if i < len(by_type[t]):
                out.append({"game_file": by_type[t][i], "task_type": t})
                added = True
        if not added:
            break
        i += 1
    return out[:n]


def scienceworld_tasks(
    n: int, seed: int = SEED, exclude_nondeterministic: bool = True
) -> list[dict]:
    from scienceworld import ScienceWorldEnv

    env = ScienceWorldEnv("", envStepLimit=100)
    families = [
        t
        for t in env.get_task_names()
        if not (exclude_nondeterministic and t in SCIENCEWORLD_NONDETERMINISTIC)
    ]
    rng = random.Random(seed)
    chosen = list(families)
    rng.shuffle(chosen)
    out = []
    for i in range(n):
        fam = chosen[i % len(chosen)]
        env.load(fam, 0, "")
        variations = list(env.get_variations_test()) or [0]
        out.append(
            {
                "task_name": fam,
                "variation": int(rng.choice(variations)),
                "simplification": "",
                "task_type": fam,
            }
        )
    env.close()
    return out
