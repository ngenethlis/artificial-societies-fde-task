"""Ground-truth opinion distributions and the sampling-noise floor.

Each Question carries the real GB-adult proportions (topline and, where the
source publishes them, subgroup crosstabs), read from the poll named in its
`source` field: death penalty and sentencing from YouGov (n=1665, Feb 2022),
immigration from Ipsos / British Future, reunification from Ipsos MORI / KCL.
Figures were transcribed from the published tables, not approximated. The
minister-liability question has no poll (real=None) and is reported unscored.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class Question:
    """A single-select survey question.

    `real is None` marks an **application** question: a consequential decision
    with no published poll to validate against (the real product use case). The
    pipeline still reports the simulated distribution and subgroup splits for it,
    but skips fidelity scoring - we cannot grade against a truth we do not have.
    """

    key: str
    text: str
    # options in their natural (ordinal where applicable) order
    options: list[str]
    # real GB-adult proportions, same order as `options`, summing to ~1.0;
    # None for an ungrounded application question
    real: list[float] | None = None
    # True when the options lie on an ordinal scale (enables Wasserstein)
    ordinal: bool = True
    # option indices that count as "support" - used to report a single net-support
    # number (e.g. for the framing experiment's "how far did opinion move?")
    positive: list[int] = field(default_factory=list)
    # how the (weighted) sample splits across subgroups, for subgroup fidelity
    # mapping: subgroup_label -> proportions over `options`
    subgroups: dict[str, list[float]] = field(default_factory=dict)
    # short provenance string for the report / slides
    source: str = ""
    # optional framing variants: frame_label -> a short message the respondent
    # sees just before the question. Each is run as an extra pass to measure how
    # the answer distribution shifts under different framings.
    frames: dict[str, str] = field(default_factory=dict)

    @property
    def grounded(self) -> bool:
        return self.real is not None

    def real_array(self) -> np.ndarray | None:
        if self.real is None:
            return None
        a = np.asarray(self.real, dtype=float)
        return a / a.sum()


# ---------------------------------------------------------------------------
# Q1 - Death penalty (5 categories incl. Don't know). Genuine ~40/50 split.
# ---------------------------------------------------------------------------
DEATH_PENALTY = Question(
    key="death_penalty",
    text=(
        "In principle, do you support or oppose the death penalty for those "
        "convicted of the most serious crimes?"
    ),
    options=[
        "Strongly support",
        "Tend to support",
        "Tend to oppose",
        "Strongly oppose",
        "Don't know",
    ],
    real=[0.14, 0.26, 0.22, 0.28, 0.10],
    ordinal=True,
    # Verified weighted crosstabs read directly from the YouGov PDF tables
    # (page 1). Order: Strongly support / Tend support / Tend oppose /
    # Strongly oppose / Don't know.
    subgroups={
        # 2019 vote
        "Con 2019": [0.22, 0.36, 0.19, 0.15, 0.07],
        "Lab 2019": [0.06, 0.17, 0.23, 0.43, 0.10],
        # EU referendum 2016
        "Leave": [0.24, 0.36, 0.17, 0.16, 0.07],
        "Remain": [0.06, 0.18, 0.26, 0.42, 0.08],
        # region
        "London": [0.07, 0.18, 0.22, 0.41, 0.12],
    },
    source="YouGov, GB adults, n=1665, 8-9 Feb 2022",
)

# ---------------------------------------------------------------------------
# Q2 - Sentencing harshness (4 categories incl. Don't know).
# ---------------------------------------------------------------------------
SENTENCING = Question(
    key="sentencing",
    text=(
        "In general, do you think the sentences handed down by the courts in "
        "Britain are too harsh, not harsh enough, or about right?"
    ),
    options=[
        "Too harsh",
        "Not harsh enough",
        "About right",
        "Don't know",
    ],
    real=[0.02, 0.65, 0.12, 0.21],
    ordinal=False,
    # Verified weighted crosstabs from the YouGov PDF tables (page 1).
    # Order: Too harsh / Not harsh enough / About right / Don't know.
    subgroups={
        "Con 2019": [0.01, 0.83, 0.08, 0.08],
        "Lab 2019": [0.02, 0.51, 0.17, 0.30],
        "Leave": [0.01, 0.81, 0.08, 0.10],
        "Remain": [0.03, 0.54, 0.18, 0.25],
        "London": [0.04, 0.52, 0.13, 0.32],
    },
    source="YouGov, GB adults, n=1665, 8-9 Feb 2022",
)

# ---------------------------------------------------------------------------
# Q3 - Immigration level (ordinal: reduced -> increased). Ipsos / British
# Future tracker Wave 16, GB adults n=3000, weighted, 17-28 Feb 2024.
# Collapsed 4-option form; crosstabs are by current party support / EU-ref vote.
# ---------------------------------------------------------------------------
IMMIGRATION = Question(
    key="immigration",
    text=(
        "Do you think the number of immigrants coming to Britain nowadays "
        "should be reduced, remain the same as it is, or increased?"
    ),
    options=["Reduced", "Remain the same", "Increased", "Don't know"],
    real=[0.52, 0.23, 0.17, 0.08],
    ordinal=True,
    subgroups={
        "Con 2019": [0.72, 0.17, 0.09, 0.02],  # Ipsos "Conservative supporters"
        "Lab 2019": [0.40, 0.32, 0.20, 0.08],  # Ipsos "Labour supporters"
        "Leave": [0.74, 0.12, 0.10, 0.04],
        "Remain": [0.37, 0.32, 0.20, 0.10],
    },
    source="Ipsos / British Future, GB adults, n=3000, 17-28 Feb 2024",
)

# ---------------------------------------------------------------------------
# Q4 - Irish reunification (nominal). Ipsos MORI / King's College London, GB
# adults, 2019. Topline only (full subgroup crosstabs not published).
# ---------------------------------------------------------------------------
REUNIFICATION = Question(
    key="reunification",
    text=(
        "If there were a referendum in Northern Ireland on its future, would "
        "you personally prefer Northern Ireland to stay in the UK, or to leave "
        "the UK and join the Republic of Ireland?"
    ),
    options=[
        "Stay in the UK",
        "Join the Republic of Ireland",
        "No preference",
        "Don't know",
    ],
    real=[0.36, 0.19, 0.36, 0.09],
    ordinal=False,
    subgroups={},  # only the "stay" cell is published by party; not enough to score
    source="Ipsos MORI / King's College London, GB adults, 2019",
)

# ---------------------------------------------------------------------------
# Q5 - APPLICATION question (ungrounded). No published single-select poll - this
# is the kind of consequential decision you would use personas FOR, precisely
# because you cannot cheaply survey it. Reported, not scored.
# ---------------------------------------------------------------------------
MINISTER_LIABILITY = Question(
    key="minister_liability",
    text=(
        "Should government ministers be able to be criminally prosecuted if "
        "their economic policies are later judged to have seriously and "
        "avoidably damaged the country's economy?"
    ),
    options=[
        "Strongly support",
        "Tend to support",
        "Tend to oppose",
        "Strongly oppose",
        "Don't know",
    ],
    real=None,  # application question - no ground truth to validate against
    ordinal=True,
    positive=[0, 1],  # "Strongly support" + "Tend to support" = net support
    source="No published poll (application / product-demo question)",
    # Two opposing balanced frames + two controls. The experiment asks how far
    # support moves and WHICH factions move - but frame-sensitivity is also the
    # documented LLM instability artifact (Bisbee 2024), so we bound it: the
    # `placebo` frame is argumentative but IRRELEVANT to the question. If it moves
    # support as much as the real frames, we are measuring sycophancy, not
    # persuasion; the real deltas are only signal ABOVE the placebo's artifact floor.
    frames={
        "neutral": (
            "A think tank has proposed the following change to the law. "
            "Please give your own view."
        ),
        "placebo": (
            "Consider this argument: towns should plant far more street trees, "
            "because tree-lined streets are cooler in summer, soak up rainwater, "
            "and make neighbourhoods more pleasant to walk through. Now, on a "
            "separate matter:"
        ),
        "accountability": (
            "Consider this argument: when ordinary people make serious mistakes "
            "at work they can be sacked or even prosecuted, yet ministers whose "
            "decisions wreck the economy keep their pensions and move on, while "
            "families lose jobs and savings. Real accountability, supporters say, "
            "means ministers carry legal responsibility for the damage they cause."
        ),
        "chilling": (
            "Consider this argument: governing means taking hard decisions under "
            "deep uncertainty, and hindsight is easy. The threat of prosecution, "
            "critics say, would scare capable people away from public service and "
            "leave ministers too afraid to act boldly in a crisis - punishing bad "
            "luck as if it were a crime."
        ),
    },
)

QUESTIONS: dict[str, Question] = {
    q.key: q
    for q in (
        DEATH_PENALTY,
        SENTENCING,
        IMMIGRATION,
        REUNIFICATION,
        MINISTER_LIABILITY,
    )
}

GROUNDED = {k: q for k, q in QUESTIONS.items() if q.grounded}
APPLICATION = {k: q for k, q in QUESTIONS.items() if not q.grounded}


def sampling_noise_floor(
    real: np.ndarray,
    n: int,
    metric,
    draws: int = 2000,
    seed: int = 0,
) -> dict[str, float]:
    """Bootstrap the metric between two independent size-`n` samples of `real`.

    Answers "how different would two honest n-person surveys of the SAME
    population look by chance?" Every method is judged against this floor, not
    against zero. Returns the mean and the 95th percentile of the metric.
    """
    rng = np.random.default_rng(seed)
    real = np.asarray(real, dtype=float)
    real = real / real.sum()  # published crosstabs round to 99-101%
    k = len(real)
    vals = np.empty(draws)
    for i in range(draws):
        a = np.bincount(rng.choice(k, size=n, p=real), minlength=k) / n
        b = np.bincount(rng.choice(k, size=n, p=real), minlength=k) / n
        vals[i] = metric(a, b)
    return {
        "mean": float(vals.mean()),
        "p95": float(np.percentile(vals, 95)),
        "n": n,
    }
