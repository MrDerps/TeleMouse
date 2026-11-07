"""The recording protocol of INTERFACE §6 as a list of timed steps."""
from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass
class Step:
    kind: str           # "rest" | "relax" | "cue" | "free"
    seconds: float
    label: str | None   # gesture id for cues, "none" for rest/free, None for relax gaps (unlabelled)
    text: str           # what the screen says
    gesture_kind: str = "instant"
    trial: int = 0      # 1-based index among cue steps


def build_plan(gestures: list[dict], reps: int | dict = 20, rest_s: float = 10.0, free_s: float = 150.0,
               gap=(1.5, 2.5), cue_s: float = 2.0, hold_cue_s: float = 3.0, seed=None) -> list[Step]:
    """gestures: [{"id","name","kind"}]. reps: int, or {gesture id: reps}. Random trial order, random relax gaps."""
    rng = random.Random(seed)
    steps = [Step("rest", rest_s, "none", "RELAX\nlet your arm and hand go limp")]
    order = [g for g in gestures for _ in range(reps.get(g["id"], 0) if isinstance(reps, dict) else reps)]
    rng.shuffle(order)
    for i, g in enumerate(order, 1):
        steps.append(Step("relax", rng.uniform(*gap), None, "relax", trial=i))
        if g["kind"] == "hold":
            steps.append(Step("cue", hold_cue_s, g["id"], f"{g['name'].upper()}\nhold it once, until the bar ends", "hold", i))
        else:
            steps.append(Step("cue", cue_s, g["id"], f"{g['name'].upper()}\ndo it ONCE, then relax", "instant", i))
    if free_s > 0:
        steps.append(Step("relax", 2.0, None, "relax"))
        steps.append(Step("free", free_s, "none",
                          "FREE ACTIVITY  (no gestures)\ntype, use the real mouse, move your hand around,\n"
                          "pick things up, gesture while talking"))
    steps.append(Step("relax", 1.0, None, "done"))
    return steps


def total_seconds(steps) -> float:
    return sum(s.seconds for s in steps)
