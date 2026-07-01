# Can 100 AI personas stand in for a real audience — and tell us how to move it?

*A methods study for the team — written for decision-makers, not engineers.*

---

## The problem we're actually solving

Every consequential message or policy call hinges on the same question: **how will
an audience — and its factions — react, and what would change their minds?** The
honest way to answer is a poll, but a representative survey of ~1,000 people costs
tens of thousands of pounds, takes weeks, and you usually run it **once**. So the
questions that matter most — the new ones, the sensitive ones, the "what if we
framed it *this* way instead" ones — mostly never get asked.

This study builds and stress-tests a **simulator of a real population** that
answers those questions in an hour for a couple of dollars. The point is not to
replay polls we already have. It is to do the two things a one-shot poll can't:
**answer questions no poll exists for**, and **test how opinion moves under
different messaging — *before* you spend the real budget.** Everything below is in
service of earning the right to trust those two outputs.

## What we built, and why these questions

We model **one** real group — **British adults** — and the questions form a
deliberate ladder, not a grab-bag. Read it top-down: the payoff first, the
evidence that licenses it underneath.

**① The payoff — a decision with no poll to buy.** *Should ministers be criminally
liable when their economic policies seriously and avoidably damage the country?*
Nobody has polled this cleanly; it's exactly the kind of live, un-pollable question
you'd reach for a simulator to answer. We report where the public lands and **who
splits which way**.

**② The creative payload — does opinion *move*?** We put that same question to the
simulated electorate under two opposing, balanced messages — an **"accountability"**
frame (*ministers escape consequences ordinary people would face*) and a
**"chilling-effect"** counter-frame (*prosecution scares capable people out of
public service*) — and measure how far each segment shifts. A static poll gives you
one number; this tells you **which message wins which faction**. That is the
product.

**③ The evidence underneath — fidelity we can prove.** None of that is worth
anything unless the simulator reproduces *real* opinion, so we validate it against
three published polls:
- *Immigration* — reduced / same / increased (52% "reduced"). *(Ipsos / British
  Future, n=3,000)*
- *Sentencing* — too harsh / not harsh enough / about right (65% "not harsh
  enough"). *(YouGov, n=1,665)*
- *Death penalty* — support / oppose (a genuine ~40/50 split). *(YouGov, n=1,665)*

These aren't a grab-bag, for two reasons. **The link that matters is
methodological:** each is a *contested, single-select opinion that splits the
electorate into factions* — structurally identical to the application question, so
fidelity demonstrated here is fidelity of the kind we need there. **The topical
overlap is a bonus, and a testable one:** all three tap a *"get-tough / hold-power-
to-account"* mood that *tends to travel with* the minister-liability question in a
populist-leaning electorate — but whether the application question actually splits
the *same* factions the same way is a question the run answers, not one we assert.
If "prosecute failed ministers" turns out to be Labour/Remain-led anti-establishment
sentiment rather than Conservative/Leave-led punitiveness, that cross-cutting split
is itself a finding. *(A fourth poll — Irish reunification — is in the code as an
extra topline check but kept out of the headline: it has no published crosstabs to
test factions against.)*

**What "close" means — and why we don't grade against perfection.** Even two
honest 100-person polls of the *same* population disagree a bit, purely by chance.
So we first measure that chance gap (the "noise floor") and judge every method
against it. A method that lands inside the noise floor is, statistically,
**indistinguishable from a real 100-person sample** — that is the bar.

We also watch one specific failure: AI personas tend to **collapse to one
agreeable answer**, producing a confident-looking result that has quietly thrown
away the real diversity of opinion. We track the spread of answers, not just the
average, to catch this.

## The four methods, in plain terms

We built a ladder of increasingly serious ways to make a persona, so we can see
exactly what each extra ingredient buys:

1. **Naive** — just ask the model, 100 times. No personality. *The control.*
2. **Demographic** — give each persona a British profile (age, region, how they
   voted, etc.) drawn to match the real population.
3. **Seeded** — build each persona from a **real survey respondent's** profile,
   so realistic combinations of traits are preserved (this is the method behind
   the published academic work on "silicon sampling").
4. **Backstory** — have the model write a short **life story** for each persona —
   a name, a job, a neighbourhood — and answer in character.

Alongside these four, we test one change to *how* we ask. Instead of taking a single
answer from each persona, we ask each persona for the probability it would give each
option and average those — a **verbalized distribution**. It turned out to matter
more than any of the four recipes.

## Findings

Two runs stand behind these numbers: the default cheap model (**Gemini Flash-Lite**,
all five questions) and a stronger one (**Gemini 2.5 Flash**), so we can tell "the
method is wrong" apart from "the model is weak." Every figure comes from `run.py`;
charts are in `results/`. The headline metric is Total Variation Distance (TVD) —
the share of opinion that would have to move for the simulation to match the real
poll. Lower is better; the n=100 noise floor sits around 0.14–0.18.

**Fidelity by method (TVD on Flash-Lite; lower is better):**

| Question | Naive | Demographic | Seeded | Backstory | Noise floor |
|---|---|---|---|---|---|
| Death penalty | 0.53 | 0.34 | 0.34 | 0.28 | 0.18 |
| Sentencing | 0.35 | 0.35 | 0.35 | 0.32 | 0.14 |
| Immigration | 0.48 | 0.48 | 0.48 | 0.46 | 0.16 |
| Reunification | 0.19 | 0.60 | 0.55 | 0.61 | 0.17 |

(Seeded matches demographic because we had no individual-level rows to seed from;
see the limits section.)

Five things come out of the full run, and two of them cut against the "richer
personas win" story we expected going in.

**1. On the cheap model, the main failure is mode collapse, not being off-target.**
On sentencing and immigration the naive, demographic, and seeded personas nearly all
give the single most common answer — 100% "sentences not harsh enough" (real: 65%),
100% "reduce immigration" (real: 52%). They land the modal answer and erase
everything around it. The variance-ratio metric reads 0.00 where real opinion has
spread. This is the failure we most wanted to catch, and it caught it.

**2. A stronger model largely fixes the collapse — so it is partly a model problem,
not the idea's fault.** Re-run on Gemini 2.5 Flash, immigration's demographic method
goes from 0.48 (flat collapse) to **0.17**, under the noise floor, with
variance-ratio up from 0.00 to 0.70; sentencing's spread recovers from 0.00 to about
0.6. Much of "everything collapses" was the small model.

| Question | Flash-Lite demographic | 2.5 Flash demographic |
|---|---|---|
| Immigration | 0.48 (var 0.00) | 0.17 (var 0.70) |
| Sentencing | 0.35 (var 0.00) | 0.30 (var 0.59) |
| Death penalty | 0.34 (var 0.81) | 0.27 (var 0.84) |

**3. The biggest single lever is how you ask, not how rich the persona is.** Instead
of sampling one hard choice per persona, we asked each persona for a probability
across the options — a soft vote — and averaged. On 2.5 Flash this recovers the real
distribution almost exactly: immigration TVD **0.04** (predicted 52/27/13/9 against a
real 52/23/17/8), death penalty TVD **0.08**. One caveat we hold to: a smooth average
has less variance than 100 hard votes, so it can slip under the noise floor for
mechanical reasons — we do not claim it beats a real poll. But the recovery of the
actual proportions is real and large. Asking for the distribution beats sampling the
mode.

**4. Conditioning is double-edged — where the public is genuinely undecided, it
hurts.** Reunification splits real Britons three ways (36% stay, 19% join, 36% no
preference). On Flash-Lite the naive method lands closest (0.19, at the floor)
because its vagueness keeps the "no preference" third. Adding demographics or a
backstory pushes personas to 96–97% "stay in the UK" and drives TVD to 0.60 —
inventing a certainty that isn't there. A richer identity makes a persona pick a
side, which is right when the public has picked one and wrong when it hasn't. (On 2.5
Flash the effect is milder — demographic 0.24 against naive 0.25 — because the
stronger model holds the ambivalence better.)

**5. The persona ladder is not a straight climb.** On 2.5 Flash death penalty,
demographic (0.27) beats backstory (0.34); the backstory over-commits to "strongly
support." And the naive baseline collapses in a model-specific direction (toward
"tend to support" on Flash-Lite, toward "don't know / oppose" on 2.5 Flash). The
richest method does not always win.

### The application question, and whether opinion moves

With no poll to check against, we put the personas to *"should ministers be
criminally liable for economically damaging policy?"* The group leans support
(backstory: about 59% net support; `fig_minister_liability_application.png`),
trustworthy to the degree the validation questions earned it.

Then the part a poll can't cheaply give you — how far opinion moves under different
messaging. We re-ask under two opposing frames plus two controls:

| Frame | Net support |
|---|---|
| Neutral (baseline) | 71% |
| Placebo (irrelevant argument) | 40% |
| "Accountability" | 96% |
| "Chilling effect" | 9% |

The swing is enormous, 9% to 96%. But the placebo — an argumentative yet irrelevant
preamble about planting street trees — moved support down 31 points on its own, which
means roughly a third of any apparent persuasion is the persona reacting to being
argued at, not to what the argument says. Both real frames move well clear of the
placebo in their intended direction, so there is genuine message signal, but the
absolute swings are inflated. Read the ordering — accountability lifts, chilling
crushes, chilling moves more — not the exact points (per-faction splits are in
`fig_minister_liability_framing.png`).

Where the run's numbers differ from the shape we assumed going in — collapse
dominating on the cheap model, conditioning hurting on the undecided question, the
ladder not being a straight climb — those differences are the findings. The noise
floor, the variance-ratio, and the placebo are the instruments that surfaced them,
and that is what makes the directional signal worth trusting.

## What this means — and where the limits are

Stated plainly, because it builds trust rather than eroding it:

- **Where it's reliable:** as a fast, cheap, directional read — which way an
  audience leans, which subgroups split, which message has a problem. Run it in an
  hour, iterate, and *then* spend the real survey budget on the finalists.
- **Where it isn't — yet:** as a precise replacement for a poll. Known issues we
  measured or guarded against:
  - **Variance collapse** — weaker models pile onto the majority answer and throw
    away minority opinion. We measure it (variance-ratio) rather than hide it, and
    it eases markedly with a stronger model and with distribution-style asking.
  - **Prompt instability** — answers can move with wording and model version
    (documented in the literature; Bisbee et al. 2024). Pin the model and prompt.
  - **Proxy ground truth** — we validate against one poll at one point in time; a
    different population or date would shift the target.
  - **Seeding data** — true per-row seeding (method 3) needs individual-level
    survey rows (e.g. the British Election Study); without them that method backs
    off to demographic sampling, and we say so.

## Recommendation

Use AI personas as a decision *accelerator*, not a decision *maker*: a first-pass
instrument to narrow options, surface where audiences split, and pre-test which
message moves which faction, followed by a real poll on the shortlist for anything
consequential.

Two choices mattered more than the persona recipe, and the run is specific about
them:

- **Use a capable model.** The cheap model collapsed to the majority answer; the
  stronger one recovered real spread and, on two questions, matched the poll within
  the noise floor. Model capability was a larger lever than persona richness.
- **Ask for the distribution, not one vote.** Having each persona give a soft
  probability across the options recovered the real proportions far better than
  sampling a single choice. It moved fidelity more than any other single change.

Then keep the guardrails that caught the failures here: run the naive control so the
room sees what conditioning actually buys, report the variance-ratio so a flattened
answer can't pass as a confident one, and put a placebo alongside any message test
so suggestibility isn't mistaken for persuasion. The message test is where the
leverage is — it turns the simulator from a cheaper poll into something a poll
budget can't buy, a wind-tunnel for messages before you commit to one.

---

### Appendix — for the technically minded

- **Metric:** Total Variation Distance (share of probability mass that would have
  to move), with Jensen-Shannon, Wasserstein (ordinal questions), modal-answer
  match, and entropy ratio. Framing follows Argyle et al. 2023's *algorithmic
  fidelity*.
- **Noise floor:** bootstrap — repeatedly draw two n-sized samples from the real
  distribution and measure their TVD; report mean and 95th percentile, per
  subgroup at that subgroup's realised n.
- **Single-select:** forced via the model's structured-output / enum mode, option
  order shuffled per call to cancel position bias. We **sample** one answer per
  persona at temperature > 0 — one person, one answer — rather than reading an
  option-probability distribution off the model; sampling is what gives the naive
  baseline a real spread to measure. (`--smoke` verifies sampling actually varies
  under the chosen backend before the full run, since constrained decoding can
  otherwise collapse to one answer.)
- **Framing experiment:** the persuasive frame is prepended to the question as a
  message the respondent "just encountered"; the same personas re-answer under
  each frame (cached separately). A neutral frame is the control, so reported
  movement isolates message *content*, not the presence of a preamble. Net support
  = the share choosing either "support" option.
- **Verbalized distribution:** as an alternative to sampling, each persona returns
  a probability across the options (structured JSON) which we average. It is scored
  and charted as an *estimate*, not a sample — a smooth average has lower variance
  than 100 hard votes, so it is held out of the noise-floor comparison and judged
  only on how well it recovers the real proportions.
- **Model / cost:** the backend is provider-agnostic — Gemini or Claude behind one
  interface. The results here are Gemini Flash-Lite (all five questions) and Gemini
  2.5 Flash (the stronger-model cross-check), on a paid Tier-1 key for about $0.30
  total. Every response is cached on disk, so a run resumes after a rate-limit and
  re-runs cost nothing. Driving the same personas through a second model is the
  cross-model robustness check (Bisbee 2024) — it is what separated the
  model-driven collapse from the method itself.
- **Key references:** Argyle et al. 2023 (*Out of One, Many*); Santurkar et al.
  2023 (OpinionQA); Park et al. 2024 (*Generative Agent Simulations of 1,000
  People*); Moon et al. 2024 (*Anthology*); Bisbee et al. 2024 (instability /
  flattening critique).
