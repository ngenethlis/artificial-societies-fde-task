"""Persona construction methods behind a single `Persona` interface.

  1. naive        - no persona; identical generic respondents (the control).
  2. demographic  - drawn from GB marginals, terse trait list.
  3. seeded       - drawn from an individual-level joint (BES rows) so trait
                    correlations survive; Argyle-style silicon sampling. Falls
                    back to marginal sampling, with a warning, if no rows given.
  4. backstory    - an LLM writes a first-person life narrative per seed; the
                    persona answers in character (Anthology-style).

The MARGINALS below only seed personas (a population fact); opinion ground truth
lives in ground_truth.py. Re-confirm the figures before any production use.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# ---------------------------------------------------------------------------
# GB adult marginal distributions (approximate; for seeding only)
# ---------------------------------------------------------------------------
# Derived from the YouGov poll's own weighted bases (n=1665), so the seeded
# population matches the one the validation crosstabs describe.
MARGINALS: dict[str, dict[str, float]] = {
    "age": {"18-24": 0.109, "25-49": 0.417, "50-64": 0.241, "65+": 0.233},
    "gender": {"man": 0.486, "woman": 0.514},
    "region": {
        "London": 0.120,
        "South": 0.336,
        "Midlands/Wales": 0.217,
        "North": 0.241,
        "Scotland": 0.086,
    },
    "social_grade": {"ABC1": 0.570, "C2DE": 0.430},
    "vote_2019": {
        "Conservative": 0.339,
        "Labour": 0.248,
        "Lib Dem": 0.090,
        "Other": 0.123,
        "Did not vote": 0.200,
    },
    "eu_ref": {"Leave": 0.396, "Remain": 0.371, "Did not vote": 0.233},
}


@dataclass
class Persona:
    """A simulated respondent. `system` is the prompt that establishes identity;
    `attrs` carries the demographic tags used for subgroup-fidelity analysis."""

    id: str
    method: str
    system: str
    attrs: dict[str, str] = field(default_factory=dict)


def _sample_marginal(rng: np.random.Generator, dist: dict[str, float]) -> str:
    labels = list(dist)
    probs = np.asarray(list(dist.values()), dtype=float)
    probs = probs / probs.sum()
    return labels[int(rng.choice(len(labels), p=probs))]


def sample_demographics(rng: np.random.Generator) -> dict[str, str]:
    """Draw one demographic profile from the GB marginals (independent draws)."""
    return {k: _sample_marginal(rng, dist) for k, dist in MARGINALS.items()}


# ---------------------------------------------------------------------------
# Prompt framing shared by all conditioned methods
# ---------------------------------------------------------------------------
_SURVEY_FRAME = (
    "You are taking part in an anonymous, private research survey. There are no "
    "right or wrong answers - answer honestly as the person described, the way "
    "they really would, even if their view is unfashionable."
)


def _traits_block(attrs: dict[str, str]) -> str:
    pretty = {
        "age": "Age",
        "gender": "Gender",
        "region": "Region",
        "social_grade": "Social grade",
        "vote_2019": "Voted in 2019",
        "eu_ref": "EU referendum",
    }
    return "\n".join(f"- {pretty.get(k, k)}: {v}" for k, v in attrs.items())


# ---- Method 1: naive control ---------------------------------------------
def naive_personas(n: int = 100) -> list[Persona]:
    system = (
        "You are a member of the British public taking part in an anonymous "
        "research survey. Answer honestly."
    )
    return [Persona(id=f"naive-{i:03d}", method="naive", system=system) for i in range(n)]


# ---- Method 2: demographic conditioning ----------------------------------
def demographic_personas(n: int = 100, seed: int = 1) -> list[Persona]:
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        attrs = sample_demographics(rng)
        system = (
            f"{_SURVEY_FRAME}\n\nYou are a British adult with these "
            f"characteristics:\n{_traits_block(attrs)}\n\n"
            "Answer as this person would."
        )
        out.append(Persona(id=f"demo-{i:03d}", method="demographic", system=system, attrs=attrs))
    return out


# ---- Method 3: per-row "silicon sampling" --------------------------------
def seeded_personas(
    n: int = 100,
    rows: list[dict[str, str]] | None = None,
    seed: int = 2,
) -> list[Persona]:
    """Seed each persona from a real respondent row (preferred) so demographic
    correlations survive. `rows` should be individual-level records using the
    same attr keys as MARGINALS. If absent, falls back to independent marginal
    sampling and tags the method so the report can disclose the limitation."""
    rng = np.random.default_rng(seed)
    out = []
    if rows:
        idx = rng.choice(len(rows), size=n, replace=len(rows) < n)
        for i, j in enumerate(idx):
            attrs = {k: rows[j][k] for k in MARGINALS if k in rows[j]}
            system = (
                f"{_SURVEY_FRAME}\n\nYou are a real British survey respondent "
                f"with these characteristics:\n{_traits_block(attrs)}\n\n"
                "Answer as this person would."
            )
            out.append(Persona(id=f"seed-{i:03d}", method="seeded", system=system, attrs=attrs))
        return out

    # fallback: marginal sampling (NOT distinct from method 2 - disclose it)
    import warnings

    warnings.warn(
        "seeded_personas: no individual-level rows supplied; falling back to "
        "marginal sampling, which is NOT distinct from the demographic method. "
        "Provide BES rows or use the verbalized-distribution contingency.",
        stacklevel=2,
    )
    base = demographic_personas(n, seed=seed)
    for p in base:
        p.id = p.id.replace("demo", "seed-fallback")
        p.method = "seeded_fallback"
    return base


# ---- Method 4: Anthology-style backstories -------------------------------
# Backstories are generated by the LLM at runtime (see survey.build_backstory_personas)
# because each one is itself a model call; we keep the seed generation here.
def backstory_seeds(n: int = 100, seed: int = 3) -> list[dict[str, str]]:
    rng = np.random.default_rng(seed)
    return [sample_demographics(rng) for _ in range(n)]


def backstory_prompt(attrs: dict[str, str]) -> str:
    """Instruction that asks the model to write a first-person life narrative for
    a person with these traits (used to build a richer persona system prompt)."""
    return (
        "Write a short first-person life story (4-6 sentences) for a British "
        "adult with the following characteristics. Make them a specific, "
        "ordinary individual - name, job, neighbourhood, what they care about, "
        "how they see the world. Do not mention surveys or opinions on policy.\n\n"
        f"{_traits_block(attrs)}"
    )


def backstory_system(attrs: dict[str, str], narrative: str) -> str:
    return (
        f"{_SURVEY_FRAME}\n\nYou are the following person. Stay fully in "
        f"character and answer as they genuinely would:\n\n{narrative}\n\n"
        f"(For reference, your background: {_traits_block(attrs)})"
    )


METHODS = ("naive", "demographic", "seeded", "backstory")
