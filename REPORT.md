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

## Findings

> Generated from `results/scorecard.json` after running `python run.py`. The
> headline chart is `results/fig_scorecard.png`; the subgroup chart is
> `results/fig_<question>_subgroups.png`. Numbers below are filled from that run.

**Step 1 — is the simulator faithful? Distance to real opinion across the
validation set (Total Variation Distance, lower is better):**

| Method | Immigration | Sentencing | Death penalty |
|---|---|---|---|
| Naive | _«run»_ | _«run»_ | _«run»_ |
| Demographic | _«run»_ | _«run»_ | _«run»_ |
| Seeded | _«run»_ | _«run»_ | _«run»_ |
| Backstory | _«run»_ | _«run»_ | _«run»_ |

*Noise floor (n=100): a method at or below it is statistically indistinguishable
from a real 100-person poll. A `*` in the console scorecard marks methods that
clear it.*

**Step 1b — does it capture *who* disagrees?** The topline is the easy test. The
real proof is whether the personas reproduce the *gaps between factions* — do
simulated Leave and Remain voters differ the way real ones do?
`fig_<question>_subgroups.png` shows simulated-vs-real distance for each subgroup
against its own (larger, because the cells are smaller) noise floor. If the
simulator gets the factions right on questions we *can* check, we have earned the
right to trust it on the one we can't.

**Step 2 — the answer no poll exists for.** We apply the best method to *"should
ministers be criminally liable for economically damaging policy?"* and report the
predicted split overall and by faction (`fig_minister_liability_application.png`)
— trustworthy *to the degree Step 1 earned it*.

**Step 3 — the product: does opinion *move*?** This is the part a poll can't cheaply
give you. We re-ask the same question under the two opposing frames and measure the
shift in net support, overall and per faction
(`fig_minister_liability_framing.png`):

| Frame | Net support (all) | Leave | Remain | Con 2019 | Lab 2019 |
|---|---|---|---|---|---|
| Neutral (baseline) | _«run»_ | _«run»_ | _«run»_ | _«run»_ | _«run»_ |
| Placebo *(artifact floor)* | _«run»_ | _«run»_ | _«run»_ | _«run»_ | _«run»_ |
| "Accountability" | _«run»_ | _«run»_ | _«run»_ | _«run»_ | _«run»_ |
| "Chilling effect" | _«run»_ | _«run»_ | _«run»_ | _«run»_ | _«run»_ |

The gap between a faction's rows is its **persuadability**: which message wins which
voters, and where the room is already made up.

**Read this layer with the most caution — and here's how we keep it honest.** A
known weakness of LLM personas is that their answers wobble with *wording itself*,
not just with meaning (Bisbee et al. 2024). That instability is the very mechanism
this experiment exploits, so we can't assume the swings are real persuasion. Two
guards:
- **A neutral control**, so movement reflects a message's *content*, not the mere
  presence of a preamble.
- **A placebo frame** — an argumentative but *irrelevant* preamble (about planting
  street trees). It measures how much the model moves when "persuaded" by something
  that should change nothing. The real frames only count as signal to the extent
  they move opinion **more than the placebo does**. If the placebo swings as hard
  as the accountability frame, we are watching the model agree with whoever spoke
  last — and we report *that*, not a fake mandate.

And a scope line we hold to: **matching real opinion *levels* (Steps 1–2) does not
prove the simulator gets the *response to a message* right** — that's a different
and harder thing to be right about. So we present the framing magnitudes as
**directional, not calibrated**: trust the *ordering* (which message helps, which
faction is most movable) well before any single percentage-point figure.

**Did the personas keep the diversity of opinion?** The variance-ratio column in
the scorecard: ~1.0 means the spread of opinion was preserved; well under 1.0
means the personas flattened toward one answer and the result is less trustworthy
than its topline suggests.

*(Expected shape, to be confirmed by the run: naive should be the worst and the
most flattened; conditioning should help; the richest methods should approach the
noise floor. Where reality differs from this, that difference is itself a finding.)*

## What this means — and where the limits are

Stated plainly, because it builds trust rather than eroding it:

- **Where it's reliable:** as a fast, cheap, directional read — which way an
  audience leans, which subgroups split, which message has a problem. Run it in an
  hour, iterate, and *then* spend the real survey budget on the finalists.
- **Where it isn't — yet:** as a precise replacement for a poll. Known issues we
  measured or guarded against:
  - **Variance collapse** — personas can agree too much; we report it rather than
    hide it.
  - **Prompt instability** — answers can move with wording and model version
    (documented in the literature; Bisbee et al. 2024). Pin the model and prompt.
  - **Proxy ground truth** — we validate against one poll at one point in time; a
    different population or date would shift the target.
  - **Seeding data** — true per-row seeding (method 3) needs individual-level
    survey rows (e.g. the British Election Study); without them that method backs
    off to demographic sampling, and we say so.

## Recommendation

Use AI personas as a **decision *accelerator*, not a decision *maker***: a
first-pass instrument to narrow options, surface audience splits, and **pre-test
which message moves which faction** cheaply — followed by a real poll on the
shortlist for anything consequential. Adopt the best method the study identifies,
always run the naive control alongside to show what conditioning is buying, and
always report the variance check so the room can see when a clean-looking answer
is actually a flattened one. The framing experiment is where the leverage is: it
turns the simulator from a cheaper poll into something a poll budget can't buy —
a wind-tunnel for messages before you commit to one.

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
- **Model / cost:** the backend is **provider-agnostic** — the modelling method is
  the contribution, not the vendor. Default is **Google Gemini Flash-Lite on
  the free tier** (the whole study runs for **£0**); the harness also supports
  Claude Haiku 4.5 (~$1–1.5). Responses are cached on disk, so the run resumes
  across free-tier daily limits and re-runs cost nothing. Because the same
  personas can be driven through either backend, **running both is the natural
  cross-model robustness check** the literature calls for (Bisbee 2024): if a
  method lands near the noise floor on *two* model families, the result isn't an
  artifact of one vendor's model — that is the next fidelity test on the list.
- **Key references:** Argyle et al. 2023 (*Out of One, Many*); Santurkar et al.
  2023 (OpinionQA); Park et al. 2024 (*Generative Agent Simulations of 1,000
  People*); Moon et al. 2024 (*Anthology*); Bisbee et al. 2024 (instability /
  flattening critique).
