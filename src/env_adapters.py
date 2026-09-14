"""Uniform adapters over ALFWorld (TextWorld backend) and ScienceWorld.

The adapters expose a single API used by the counterfactual-replay design:

    reset()                -> observation
    step(action)           -> (observation, reward, done, info)
    state_hash()           -> stable hash of the *simulator* state
    checkpoint()           -> serialisable checkpoint (action-sequence replay)
    fork(checkpoint)       -> a NEW independent adapter restored to that state
    rollback_checkpoint()  -> checkpoint of the state just BEFORE the last action

Checkpoint/restore is implemented as *deterministic action-sequence replay*:
neither backend exposes a usable state save/load (TextWorld's `PddlEnv.copy()`
is inherited-but-unimplemented; ScienceWorld's py4j bridge has no snapshot API),
so we re-create the environment from the same game/task specification and
re-issue the identical stored action list. Fidelity of this mechanism is
verified empirically in tests/test_checkpoint_fidelity.py.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass, field, asdict
from typing import Any

# ---------------------------------------------------------------------------
# Official episode limits (frozen; see configs/env_contract.json)
# ---------------------------------------------------------------------------
ALFWORLD_MAX_STEPS = 50  # alfworld/configs/base_config.yaml: max_nb_steps_per_episode
SCIENCEWORLD_MAX_STEPS = 100  # ScienceWorldEnv.__init__ default envStepLimit


def _sha(payload: str) -> str:
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def normalize_text(text: str) -> str:
    """Whitespace-normalise observation text so hashing is not format-sensitive."""
    if text is None:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [re.sub(r"[ \t]+", " ", ln).strip() for ln in text.split("\n")]
    return "\n".join(ln for ln in lines if ln)


@dataclass
class Checkpoint:
    """Everything needed to reconstruct an environment state from scratch."""

    env_name: str
    task_spec: dict[str, Any]
    actions: list[str]
    step_index: int
    state_hash: str
    mechanism: str = "action_replay"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class BaseAdapter:
    env_name = "base"
    max_steps = 0

    def __init__(self, task_spec: dict[str, Any]):
        self.task_spec = dict(task_spec)
        self.actions: list[str] = []
        self.last_obs: str = ""
        self.done: bool = False
        self.score: float = 0.0
        self.won: bool = False
        self.steps: int = 0

    # -- required API -------------------------------------------------------
    def reset(self) -> str:  # pragma: no cover - abstract
        raise NotImplementedError

    def step(self, action: str):  # pragma: no cover - abstract
        raise NotImplementedError

    def state_hash(self) -> str:  # pragma: no cover - abstract
        raise NotImplementedError

    def close(self) -> None:
        pass

    # -- checkpointing ------------------------------------------------------
    def checkpoint(self) -> Checkpoint:
        return Checkpoint(
            env_name=self.env_name,
            task_spec=dict(self.task_spec),
            actions=list(self.actions),
            step_index=self.steps,
            state_hash=self.state_hash(),
        )

    def rollback_checkpoint(self) -> Checkpoint | None:
        """Checkpoint of the state immediately BEFORE the most recent action.

        Used by arm A3 (ROLLBACK+REPLAN). Returns None at step 0.
        """
        if not self.actions:
            return None
        probe = self.__class__(self.task_spec)
        probe.reset()
        for a in self.actions[:-1]:
            probe.step(a)
        ck = probe.checkpoint()
        probe.close()
        return ck

    @classmethod
    def from_checkpoint(cls, ck: Checkpoint | dict) -> "BaseAdapter":
        if isinstance(ck, dict):
            ck = Checkpoint(**ck)
        env = cls(ck.task_spec)
        env.reset()
        for a in ck.actions:
            env.step(a)
        return env


# ---------------------------------------------------------------------------
# ALFWorld (TextWorld / PDDL backend)
# ---------------------------------------------------------------------------
class AlfWorldAdapter(BaseAdapter):
    """task_spec = {"game_file": "<abs path to game.tw-pddl>"}"""

    env_name = "alfworld"
    max_steps = ALFWORLD_MAX_STEPS

    def __init__(self, task_spec: dict[str, Any]):
        super().__init__(task_spec)
        self._env = None
        self._state = None
        self.task_description = ""
        self.admissible: list[str] = []

    def _make(self):
        import textworld
        from alfworld.agents.environment.alfred_tw_env import (
            AlfredDemangler,
            AlfredInfos,
        )

        infos = textworld.EnvInfos(
            won=True,
            lost=True,
            admissible_commands=True,
            facts=True,
            moves=True,
            extras=["gamefile"],
        )
        return textworld.start(
            self.task_spec["game_file"],
            request_infos=infos,
            wrappers=[AlfredDemangler(shuffle=False), AlfredInfos],
        )

    def reset(self) -> str:
        if self._env is not None:
            self.close()
        self._env = self._make()
        self._state = self._env.reset()
        self.actions = []
        self.steps = 0
        self.done = False
        self.won = False
        self.score = 0.0
        obs = self._state["feedback"]
        self.last_obs = obs
        self.admissible = list(self._state.get("admissible_commands") or [])
        m = re.search(r"Your task is to:\s*(.+)", obs)
        self.task_description = m.group(1).strip() if m else ""
        return obs

    def step(self, action: str):
        self._state, reward, done = self._env.step(action)
        self.actions.append(action)
        self.steps += 1
        self.last_obs = self._state["feedback"]
        self.admissible = list(self._state.get("admissible_commands") or [])
        self.won = bool(self._state.get("won"))
        self.score = float(reward or 0.0)
        # Episode limit is enforced by us (textworld.start has no step cap).
        self.done = bool(done) or self.steps >= self.max_steps
        info = {
            "won": self.won,
            "lost": bool(self._state.get("lost")),
            "moves": self._state.get("moves"),
            "admissible_commands": self.admissible,
            "step_limit_reached": self.steps >= self.max_steps,
        }
        return self.last_obs, self.score, self.done, info

    def _facts_repr(self) -> str:
        facts = self._state.get("facts") or []
        out = []
        for p in facts:
            try:
                args = ",".join(f"{v.name}:{v.type}" for v in p.arguments)
                out.append(f"{p.name}({args})")
            except Exception:
                out.append(str(p))
        return "\n".join(sorted(out))

    def state_hash(self) -> str:
        """Hash of the full PDDL fact set + normalised observation.

        The PDDL fact set IS the simulator state for this backend, so an
        identical hash is strong evidence of identical state (not just
        identical rendering).
        """
        payload = json.dumps(
            {
                "facts": self._facts_repr(),
                "obs": normalize_text(self.last_obs),
                "won": self.won,
            },
            sort_keys=True,
        )
        return _sha(payload)

    def obs_hash(self) -> str:
        return _sha(normalize_text(self.last_obs))

    def close(self) -> None:
        try:
            if self._env is not None:
                self._env.close()
        except Exception:
            pass
        self._env = None


# ---------------------------------------------------------------------------
# ScienceWorld
# ---------------------------------------------------------------------------
class ScienceWorldAdapter(BaseAdapter):
    """task_spec = {"task_name": str, "variation": int, "simplification": str}"""

    env_name = "scienceworld"
    max_steps = SCIENCEWORLD_MAX_STEPS

    def __init__(self, task_spec: dict[str, Any]):
        super().__init__(task_spec)
        self._env = None
        self.task_description = ""
        self.last_info: dict[str, Any] = {}

    def _get_env(self):
        """Each adapter owns its own JVM bridge.

        Sharing a bridge would break fork isolation: `load()` mutates the single
        underlying simulator, so a forked arm would silently destroy the state
        of its sibling. One bridge per adapter costs ~2 s of start-up but makes
        the counterfactual arms provably independent.
        """
        if self._env is None:
            from scienceworld import ScienceWorldEnv

            self._env = ScienceWorldEnv("", envStepLimit=SCIENCEWORLD_MAX_STEPS)
        return self._env

    def reset(self) -> str:
        env = self._get_env()
        env.load(
            self.task_spec["task_name"],
            int(self.task_spec["variation"]),
            self.task_spec.get("simplification", ""),
        )
        obs, info = env.reset()
        self.actions = []
        self.steps = 0
        self.done = False
        self.won = False
        self.score = 0.0
        self.last_obs = obs
        self.last_info = dict(info)
        self.task_description = env.get_task_description()
        return obs

    def step(self, action: str):
        obs, reward, done, info = self._env.step(action)
        self.actions.append(action)
        self.steps += 1
        self.last_obs = obs
        self.last_info = dict(info)
        self.score = float(info.get("score", 0.0))
        # ScienceWorld encodes unrecoverable failure as score == -100.
        self.won = self.score >= 100.0
        self.done = bool(done) or self.steps >= self.max_steps
        out = {
            "score": self.score,
            "moves": info.get("moves"),
            "won": self.won,
            "inv": info.get("inv"),
            "step_limit_reached": self.steps >= self.max_steps,
        }
        return obs, self.score, self.done, out

    def state_hash(self) -> str:
        """Hash of the ScienceWorld object tree (full simulator state) + obs."""
        try:
            tree_s = json.dumps(self._env.getObjectTree(), sort_keys=True)
        except Exception:
            tree_s = ""
        payload = json.dumps(
            {
                "tree": tree_s,
                "obs": normalize_text(self.last_obs),
                "look": normalize_text(str(self.last_info.get("look", ""))),
                "inv": normalize_text(str(self.last_info.get("inv", ""))),
                "score": self.score,
            },
            sort_keys=True,
        )
        return _sha(payload)

    def obs_hash(self) -> str:
        return _sha(normalize_text(self.last_obs))

    def close(self) -> None:
        try:
            if self._env is not None:
                self._env.close()
        except Exception:
            pass
        self._env = None


ADAPTERS = {"alfworld": AlfWorldAdapter, "scienceworld": ScienceWorldAdapter}


def make_adapter(env_name: str, task_spec: dict[str, Any]) -> BaseAdapter:
    return ADAPTERS[env_name](task_spec)
