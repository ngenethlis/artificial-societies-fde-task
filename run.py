"""End-to-end pipeline: build personas, run the survey, score fidelity.

Usage:
  python run.py --quick      # vertical slice: naive + backstory, 1 question
  python run.py              # full: 4 methods, validation + application questions,
                             #       subgroup fidelity, and the framing experiment
  python run.py --selftest   # offline: validate metrics + noise floor, no API

Requires GEMINI_API_KEY (or ANTHROPIC_API_KEY with PERSONA_PROVIDER=anthropic)
for everything except --selftest. Responses are cached under results/, so a
re-run continues from the cache rather than repeating calls.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from personas import metrics as M
from personas.ground_truth import QUESTIONS, sampling_noise_floor
from personas.personas import demographic_personas, naive_personas, seeded_personas

RESULTS = Path(__file__).resolve().parent / "results"


def _load_dotenv() -> None:
    """Load KEY=VALUE lines from a gitignored .env into the environment, so the
    API key need not be exported by hand. Existing env vars win; quotes stripped."""
    path = Path(__file__).resolve().parent / ".env"
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip().strip("'\"")
        os.environ.setdefault(key, val)

# subgroup label -> (attr key, attr value) that selects matching personas
SUBGROUPS = {
    "Con 2019": ("vote_2019", "Conservative"),
    "Lab 2019": ("vote_2019", "Labour"),
    "Leave": ("eu_ref", "Leave"),
    "Remain": ("eu_ref", "Remain"),
    "London": ("region", "London"),
}


def score(sim, real, ordinal, n) -> dict:
    floor = sampling_noise_floor(np.asarray(real), n=n, metric=M.total_variation)
    out = M.all_metrics(sim, real, ordinal)
    out["tvd_floor_mean"] = floor["mean"]
    out["tvd_floor_p95"] = floor["p95"]
    out["n"] = n
    out["beats_noise"] = out["tvd"] <= floor["p95"]
    return out


def selftest() -> None:
    """Offline validation of the scoring harness - no API calls."""
    import numpy as np

    a = np.array([0.14, 0.26, 0.22, 0.28, 0.10])
    assert M.total_variation(a, a) == 0
    assert M.modal_accuracy(a, a) == 1
    assert M.variance_ratio(np.array([0, 1, 0, 0, 0.0]), a) < 1e-6  # collapse
    q = QUESTIONS["death_penalty"]
    f = sampling_noise_floor(q.real_array(), n=100, metric=M.total_variation)
    assert 0.05 < f["mean"] < 0.2, f
    print("selftest OK - metrics, collapse detector, and noise floor all sane.")
    print(f"  death_penalty n=100 TVD noise floor: mean={f['mean']:.3f} p95={f['p95']:.3f}")


def smoke(model: str | None = None) -> None:
    """A few real API calls - validate single-select AND that sampling varies,
    before committing to the ~2,500-call full run. The cheapest place to discover
    a schema surprise, a wrong model id, or a variance-collapse artifact.
    """
    from personas.survey import Surveyor
    from personas.personas import naive_personas

    q = QUESTIONS["death_penalty"]
    sv = Surveyor(model=model)
    p = naive_personas(1)[0]
    # 8 draws of the SAME prompt (fixed option order) - pure temperature
    # variation. Constrained/enum decoding can collapse to argmax regardless of
    # temperature; if it does, the naive baseline's whole spread is a provider
    # artifact, not signal. This is the check to do before trusting any numbers.
    draws = [sv._ask_once(p.system, q, order_seed=0) for _ in range(8)]
    labels = [q.options[i] for i in draws]
    distinct = len(set(draws))
    print(f"smoke OK ({sv.provider}/{sv.model}) - 8 identical-prompt draws:")
    print(f"  {labels}")
    if distinct > 1:
        print(f"  {distinct}/8 distinct -> sampling varies at temperature. Good.")
    else:
        print(f"  WARNING: 1/8 distinct -> enum decoding may be ignoring "
              f"temperature. The naive baseline's variance would be a provider "
              f"artifact. See README (variance-collapse Plan B) before the full run.")


def run(quick: bool, model: str | None = None) -> None:
    from personas.survey import Surveyor, to_distribution

    sv = Surveyor(model=model)

    questions = [QUESTIONS["death_penalty"]] if quick else list(QUESTIONS.values())
    if quick:
        method_personas = {
            "naive": naive_personas(100),
            "backstory": sv.build_backstory_personas(100),
        }
    else:
        method_personas = {
            "naive": naive_personas(100),
            "demographic": demographic_personas(100),
            # seeded: pass BES rows here when available; falls back + warns otherwise
            "seeded": seeded_personas(100, rows=_load_bes_rows()),
            "backstory": sv.build_backstory_personas(100),
        }

    scorecard: dict = {}
    for q in questions:
        k = len(q.options)
        scorecard[q.key] = {
            "real": q.real,
            "options": q.options,
            "grounded": q.grounded,
            "source": q.source,
            "methods": {},
        }
        for method, personas in method_personas.items():
            ans = sv.survey(personas, q, repeats=1)
            sim = to_distribution(ans, k)
            entry: dict = {"sim": sim.tolist()}
            if q.grounded:  # only score against a real distribution we actually have
                entry.update(score(sim, q.real_array(), q.ordinal, n=len(personas)))

            # subgroup splits. For grounded questions this is the fidelity hero
            # result; for the application question it's the whole point (who
            # splits which way), reported without a fidelity score.
            subs = {}
            if personas[0].attrs:
                for label, (attr, val) in SUBGROUPS.items():
                    idx = [i for i, p in enumerate(personas) if p.attrs.get(attr) == val]
                    if len(idx) < 5:
                        continue
                    sub_sim = to_distribution(ans[idx], k)
                    sub_entry = {"sim": sub_sim.tolist(), "n": len(idx)}
                    if q.grounded and label in q.subgroups:
                        sub_entry.update(
                            {"real": q.subgroups[label],
                             **score(sub_sim, q.subgroups[label], q.ordinal, n=len(idx))}
                        )
                    subs[label] = sub_entry
            entry["subgroups"] = subs
            scorecard[q.key]["methods"][method] = entry

        # Verbalized mode: soft per-persona votes on the demographic personas, to
        # isolate the asking-mode effect. Marked as an estimate - a smooth average
        # has lower variance than 100 hard votes, so the n=100 floor does not apply
        # (no beats_noise star, and it is excluded from the "best method" charts).
        if not quick and "demographic" in method_personas:
            personas = method_personas["demographic"]
            vecs = sv.survey_distribution(personas, q)  # [n, k]
            sim = vecs.mean(axis=0)
            entry = {"sim": sim.tolist(), "estimate": True}
            if q.grounded:
                m = M.all_metrics(sim, q.real_array(), q.ordinal)
                floor = sampling_noise_floor(q.real_array(), n=len(personas), metric=M.total_variation)
                m["tvd_floor_p95"] = floor["p95"]
                m["beats_noise"] = None  # estimate, not a sample - floor is n/a
                entry.update(m)
            vsubs = {}
            for label, (attr, val) in SUBGROUPS.items():
                idx = [i for i, p in enumerate(personas) if p.attrs.get(attr) == val]
                if len(idx) < 5:
                    continue
                sub_sim = vecs[idx].mean(axis=0)
                sub_entry = {"sim": sub_sim.tolist(), "n": len(idx), "estimate": True}
                if q.grounded and label in q.subgroups:
                    sub_real = np.asarray(q.subgroups[label], float)
                    sub_entry.update(M.all_metrics(sub_sim, sub_real / sub_real.sum(), q.ordinal))
                vsubs[label] = sub_entry
            entry["subgroups"] = vsubs
            scorecard[q.key]["methods"]["verbalized"] = entry

    framing_experiment(sv, scorecard, method_personas, questions, to_distribution)

    RESULTS.mkdir(exist_ok=True)
    payload = json.dumps(scorecard, indent=2)
    # Per-model copy so cross-model comparisons survive, plus the canonical latest.
    (RESULTS / f"scorecard_{sv.tag}.json").write_text(payload)
    (RESULTS / "scorecard.json").write_text(payload)
    print(f"\n[model: {sv.model}]")
    _print_scorecard(scorecard)
    try:
        from personas.plots import render

        render(scorecard, RESULTS)
        print(f"\nFigures written to {RESULTS}/")
    except Exception as e:  # plotting is non-critical
        print(f"(plots skipped: {e})")


def framing_experiment(sv, scorecard, method_personas, questions, to_distribution) -> None:
    """Re-ask a question under each frame and record how the answer distribution
    moves, overall and by subgroup. Runs on the richest available method."""
    richest = next(
        (m for m in ("backstory", "seeded", "demographic", "naive") if m in method_personas),
        None,
    )
    if richest is None:
        return
    personas = method_personas[richest]
    for q in questions:
        if not q.frames:
            continue
        k = len(q.options)
        frames_out: dict = {}
        for label, text in q.frames.items():
            ans = sv.survey(personas, q, repeats=1, frame_label=label, frame_text=text)
            subs: dict = {}
            if personas[0].attrs:
                for slabel, (attr, val) in SUBGROUPS.items():
                    idx = [i for i, p in enumerate(personas) if p.attrs.get(attr) == val]
                    if len(idx) < 5:
                        continue
                    subs[slabel] = {"sim": to_distribution(ans[idx], k).tolist(), "n": len(idx)}
            frames_out[label] = {"sim": to_distribution(ans, k).tolist(), "subgroups": subs}
        scorecard[q.key]["framing"] = {
            "method": richest,
            "positive": q.positive,
            "frames": frames_out,
        }


def _load_bes_rows():
    """Return individual-level demographic rows for method-3 seeding, or None.

    Place a CSV at data/bes_rows.csv with columns matching MARGINALS keys
    (age, gender, region, social_grade, vote_2019, eu_ref). See README for the
    BES download; absent this, method 3 falls back to marginal sampling.
    """
    path = Path(__file__).resolve().parent / "data" / "bes_rows.csv"
    if not path.exists():
        return None
    import csv

    with path.open() as f:
        return list(csv.DictReader(f))


def _print_scorecard(sc: dict) -> None:
    for qkey, q in sc.items():
        if not q.get("grounded", True):
            print(f"\n=== {qkey} === [APPLICATION - no ground truth; reported, not scored]")
            print(f"  ({q.get('source','')})")
            opts = q["options"]
            for method, e in q["methods"].items():
                dist = ", ".join(f"{o} {p:.0%}" for o, p in zip(opts, e["sim"]))
                print(f"  {method:<12} {dist}")
            fr = q.get("framing")
            if fr:
                pos = fr["positive"]
                net = lambda sim: 100 * sum(sim[i] for i in pos)
                base = fr["frames"].get("neutral")
                print(f"\n  framing experiment ({fr['method']}; net support = "
                      f"{'+'.join(opts[i] for i in pos)}):")
                for label, fe in fr["frames"].items():
                    n = net(fe["sim"])
                    delta = "" if base is None or label == "neutral" else f"  (Δ {n-net(base['sim']):+.0f} pts vs neutral)"
                    print(f"    {label:<16} net support {n:>4.0f}%{delta}")
            continue
        print(f"\n=== {qkey} ===   (lower TVD = better; * = beats noise floor; † = verbalized estimate, floor n/a)")
        print(f"  ({q.get('source','')})")
        print(f"{'method':<14}{'TVD':>7}{'floor':>8}{'JSD':>7}{'var.ratio':>11}{'modal':>7}")
        for method, e in q["methods"].items():
            star = "†" if e.get("estimate") else ("*" if e.get("beats_noise") else " ")
            print(
                f"{method:<14}{e['tvd']:>6.3f}{star}{e['tvd_floor_p95']:>8.3f}"
                f"{e['jsd']:>7.3f}{e['variance_ratio']:>11.2f}{e['modal_match']:>7.0f}"
            )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="vertical slice (naive+backstory, 1 q)")
    ap.add_argument("--selftest", action="store_true", help="offline harness validation")
    ap.add_argument("--smoke", action="store_true", help="one real API call to validate single-select")
    ap.add_argument("--model", default=None, help="override the model id (e.g. gemini-2.5-flash)")
    args = ap.parse_args()
    _load_dotenv()
    if args.selftest:
        selftest()
    elif args.smoke:
        smoke(model=args.model)
    else:
        try:
            run(quick=args.quick, model=args.model)
        except Exception as e:
            # Free-tier daily cap (or any mid-run stop): every answer so far is
            # cached, so just re-run later to resume where it left off.
            from personas.survey import RateLimitExhausted

            if isinstance(e, RateLimitExhausted):
                print(f"\n{e}\n  -> Re-run `python run.py` later to continue from the cache.")
            else:
                raise
