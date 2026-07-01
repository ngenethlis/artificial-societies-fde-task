# Artificial Societies FDE Task — LLM Personas of a Human Group

Build 100 LLM personas that model **one** group of humans (GB adults) and use them
to do what a one-shot poll can't: **answer a question no poll exists for**, and
**test how opinion moves under different messaging**. We first prove the simulator
is faithful — against published polls, at topline *and* across factions — then
spend that earned trust on an un-pollable decision and a message-framing
experiment. Full reasoning and the stakeholder narrative are in **`REPORT.md`**.

## TL;DR

- This builds 100 GB-adult personas and runs them as a survey panel.
- It validates against real published polls first, with a noise-floor check.
- Then it answers one un-pollable question and tests which framing moves which groups.

## What's here

| File | Role |
|---|---|
| `personas/ground_truth.py` | Real YouGov distributions + the bootstrapped **sampling-noise floor** |
| `personas/metrics.py` | TVD (headline), Jensen-Shannon, Wasserstein, modal accuracy, **variance ratio** (flattening detector) |
| `personas/personas.py` | The four persona methods: naive → demographic → seeded → backstory |
| `personas/survey.py` | Forced single-select via strict tool use, option-order shuffling, response cache, backstory generation |
| `personas/plots.py` | Slide-ready scorecard, real-vs-simulated, and subgroup-fidelity figures |
| `run.py` | Orchestrator |
| `results/` | Cached model responses, `scorecard.json`, figures |

## Setup

The model backend is **provider-agnostic** — the modelling *method* is the
contribution, not the vendor. The default is **Google Gemini** (free tier, no
credit card); the Anthropic/Haiku path is kept and selectable.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export GEMINI_API_KEY=...                 # free key from aistudio.google.com

# To use Anthropic instead:
#   uncomment `anthropic` in requirements.txt (and re-run pip install)
#   export ANTHROPIC_API_KEY=sk-ant-...
#   export PERSONA_PROVIDER=anthropic
```

## Run

```bash
python run.py --selftest   # offline: validate metrics + noise floor (no API)
python run.py --smoke      # a few real calls: single-select + sampling-varies check
python run.py --quick      # vertical slice: naive + backstory, 1 question
python run.py              # full: 4 methods, validation + application qs, framing
```

Run order: `--selftest` (free) → `--smoke` → `--quick` → full.

**Watch `--smoke`'s variance check.** It draws the same prompt 8× and reports how
many distinct answers come back. Constrained/enum decoding can collapse to one
answer regardless of temperature — if that happens, the naive baseline's spread
(and the variance-ratio flattening metric) would be a *provider artifact*, not a
real finding. **Plan B if it collapses:** drop enum mode for the naive method and
parse a plain "respond with exactly one of these options" reply, which preserves
natural sampling.

**Free-tier daily limits.** The full run is ~3,000 calls (2,000 survey + 100
backstories + ~500 verbalized + ~400 framing). Free-tier *daily* request caps (RPD) may not clear
that in one day, so the deliverable can take **multiple days** to generate. This
is fine: every answer is cached after each call, a per-minute (RPM) 429 is
retried with short backoff, and a persistent (daily) 429 exits cleanly — just
re-run `python run.py` later (or with a fresh key) to resume from the cache.

Every model response is cached under `results/`, so re-runs are free and
reproducible (caching, not seeds, is the reproducibility lever — `temperature>0`
still varies between fresh calls).

## Method ladder

1. **naive** — no persona; 100 generic respondents. The control. Exposes the
   model's default skew and variance flattening.
2. **demographic** — 100 personas drawn from GB marginals; terse trait list.
3. **seeded** — Argyle-style per-row "silicon sampling". Seed each persona from a
   **real** individual-level row so demographic correlations survive. Drop a CSV
   at `data/bes_rows.csv` (British Election Study columns: `age, gender, region,
   social_grade, vote_2019, eu_ref`). Absent that, it falls back to marginal
   sampling and says so — at which point promote the verbalized-distribution
   method to keep the rungs distinct.
4. **backstory** — Anthology-style: the LLM writes a first-person life narrative
   per seed; the persona answers in character. Best non-interview fidelity;
   mitigates caricature and restores within-group variance.

## How fidelity is judged

A "survey response" is **one** answer per persona (temperature > 0) — one person,
one answer, like a real survey. We **sample** one answer per persona (rather than
reading an option-probability distribution off the model), which is what makes the
naive baseline's spread a real signal. The 100-response distribution is compared to
the real YouGov distribution by **TVD** (headline), with Jensen-Shannon, Wasserstein
(ordinal questions), modal-answer match, and the variance ratio. Every method is
judged against a **bootstrapped sampling-noise floor** — how far two honest
100-person surveys of the *same* population differ by chance — computed per
subgroup too, since subgroup cells are small (London ≈ n=9–13) and noisy.

## Questions & ground truth

**Validation set** (scored against real polls):
- **Death penalty** & **sentencing** — YouGov, GB adults, n=1665, Feb 2022
  ([PDF](https://d3nkl3psvxxpe9.cloudfront.net/documents/YouGov_-_Death_Penalty_and_Sentencing_Survey_Results.pdf)),
  toplines + 5 subgroup crosstabs, verified from the tables.
- **Immigration** (reduced/same/increased) — Ipsos / British Future tracker
  Wave 16, GB adults, n=3000, Feb 2024; topline 52/23/17/8 + crosstabs by EU-ref
  and party.

**Extra topline check** (kept out of the headline):
- **Irish reunification** — Ipsos MORI / King's College London, GB adults, 2019;
  topline 36/19/36/9 (no published subgroup crosstabs).

The three validation questions share one latent axis — *"the system's too soft;
get tough; hold elites accountable"* — so they're coherent evidence for, not a
grab-bag around, the application question below. (Irish reunification is also in
`ground_truth.py` as an extra topline check but kept out of the headline: off-axis,
no published crosstabs.)

**Application question** (no poll — reported, not scored): *should ministers be
criminally liable for economically damaging policy?* — the un-pollable decision
the personas exist to inform. A `Question` with `real=None` is handled
automatically: distribution + subgroup splits are reported, fidelity scoring is
skipped.

**Framing experiment** (the product demo): the application question is re-asked
under two opposing, balanced messages — an *accountability* frame and a
*chilling-effect* counter-frame — plus two controls (neutral + placebo). We
measure how far net support moves overall and **per faction**
(`fig_minister_liability_framing.png`).
A static poll gives one number; this shows which message wins which voters — the
thing the simulator exists to deliver. Defined as `Question.frames`; threaded
through `survey()` with a frame-tagged cache key so passes never collide.

All percentages are read directly from the primary sources and verified, not
approximated.
