"""Ask personas single-select questions and aggregate the answers.

  * One sampled answer per persona per question (temperature > 0), like one
    survey respondent. Answers are sampled, not read off a probability head.
  * Single-select is forced via the provider's structured-output/enum mode;
    option order is shuffled per call to cancel position bias.
  * Every call is cached on disk (keyed by provider+model, persona, question,
    repeat, frame) and flushed after each call, so a run resumes after a
    rate-limit or crash without repeating work.

`provider` selects Gemini (default) or Anthropic behind one interface, which
also allows running identical personas across models.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np

from .ground_truth import Question
from .personas import Persona, backstory_prompt, backstory_seeds, backstory_system

# One constant per backend, so a --smoke call validates the exact model id.
# Free-tier quota is PER-MODEL: gemini-2.5-flash gives better persona nuance but a
# low daily cap (exhausts in tens of calls); gemini-flash-lite-latest has a much
# larger free bucket, which is what makes the ~2,500-call study actually completable
# for free. Default to lite; pass model="gemini-2.5-flash" for higher fidelity if
# its quota suffices. (`-latest` is an alias; caching pins the realised outputs.)
GEMINI_MODEL = "gemini-flash-lite-latest"
ANTHROPIC_MODEL = "claude-haiku-4-5"
DEFAULT_PROVIDER = os.environ.get("PERSONA_PROVIDER", "gemini").lower()

CACHE_DIR = Path(__file__).resolve().parent.parent / "results"
CACHE_PATH = CACHE_DIR / "cache.json"


class RateLimitExhausted(RuntimeError):
    """Raised when short retries can't clear a transient API failure - either a
    persistent 429 (free-tier rate/daily limit) or a 5xx overload (the model is
    busy). Either way the cure is the same: progress is already cached, so resume
    later, switch model, or use a fresh key. The message says which cause it was."""


class Surveyor:
    def __init__(
        self,
        provider: str | None = None,
        model: str | None = None,
        temperature: float = 1.0,
        cache_path: Path = CACHE_PATH,
    ):
        self.provider = (provider or DEFAULT_PROVIDER).lower()
        self.temperature = temperature
        if self.provider == "gemini":
            from google import genai  # lazy: keep no-API modules importable
            from google.genai import types

            # reads GEMINI_API_KEY or GOOGLE_API_KEY; the per-request timeout (ms)
            # is essential - without it a stalled connection hangs the whole run
            # indefinitely (one call froze a full run for 7h). On timeout the call
            # errors and the backoff retries it.
            self.client = genai.Client(http_options=types.HttpOptions(timeout=120_000))
            self.model = model or GEMINI_MODEL
        elif self.provider == "anthropic":
            import anthropic

            self.client = anthropic.Anthropic()
            self.model = model or ANTHROPIC_MODEL
        else:
            raise ValueError(f"unknown provider {self.provider!r} (gemini|anthropic)")

        # Cache is namespaced by the EXACT model: mixing answers from two models
        # into one distribution is the prompt/model instability the study warns
        # about (Bisbee 2024), so switching models starts a clean cache.
        self.tag = f"{self.provider}-{self.model}".replace("/", "_")
        self.cache_path = Path(cache_path)
        self._cache: dict[str, int] = {}
        if self.cache_path.exists():
            self._cache = json.loads(self.cache_path.read_text())

    # -- cache -------------------------------------------------------------
    def _save(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(json.dumps(self._cache, indent=0))

    # -- provider plumbing -------------------------------------------------
    def _gemini_generate(self, contents: str, config, max_retries: int = 9):
        """Call Gemini, retrying transient failures with backoff:
          * 429 / RESOURCE_EXHAUSTED  -> rate or daily quota
          * 5xx (e.g. 503 UNAVAILABLE) -> model temporarily overloaded
        Both are transient and retried the same way; if they persist we bail
        cleanly (cache is saved) with a message naming the actual cause, rather
        than sleeping for hours."""
        from google.genai import errors

        try:
            import httpx
            net_errors = (httpx.TimeoutException, httpx.TransportError)
        except Exception:
            net_errors = ()

        delay, last = 2.0, None
        for _ in range(max_retries):
            try:
                return self.client.models.generate_content(
                    model=self.model, contents=contents, config=config
                )
            except errors.APIError as e:
                code = getattr(e, "code", None)
                transient = (
                    code == 429
                    or (code is not None and code >= 500)
                    or "RESOURCE_EXHAUSTED" in str(e)
                )
                if not transient:
                    raise
                last = e
                time.sleep(delay)
                delay = min(delay * 2, 60)
            except net_errors as e:  # timeout / dropped connection - retry
                last = e
                time.sleep(delay)
                delay = min(delay * 2, 60)
        code = getattr(last, "code", None)
        if code == 429 or (last and "RESOURCE_EXHAUSTED" in str(last)):
            cause = f"repeated 429 (free-tier rate/daily limit) on {self.model}"
        elif code:
            cause = f"repeated {code} ({getattr(last, 'status', 'error')}) - {self.model} overloaded"
        else:
            cause = f"repeated timeout/connection error on {self.model}"
        raise RateLimitExhausted(
            f"{cause}. Progress is cached; re-run later, or pass a different model "
            f"(e.g. gemini-flash-lite-latest), or a fresh GEMINI_API_KEY."
        )

    # -- one forced single-select call ------------------------------------
    def _ask_once(self, system: str, question: Question, order_seed: int, frame: str = "") -> int:
        """Return the index (into question.options) of the chosen answer.

        `frame`, if given, is a short message the respondent encounters just
        before the question - the persuasion-experiment lever.
        """
        rng = np.random.default_rng(order_seed)
        perm = rng.permutation(len(question.options))
        shown = [question.options[i] for i in perm]
        prompt = f"{question.text}\n\nChoose exactly one option."
        if frame:
            prompt = f"{frame}\n\n{prompt}"

        if self.provider == "gemini":
            chosen = self._ask_gemini(system, prompt, shown)
        else:
            chosen = self._ask_anthropic(system, prompt, shown)

        # Strict mapping back to the canonical option order - no fuzzy matching.
        # Data integrity over silent robustness: a mismatch surfaces, not slips.
        if chosen not in question.options:
            chosen2 = next((o for o in question.options if o.strip() == chosen.strip()), None)
            if chosen2 is None:
                raise ValueError(
                    f"{self.provider} returned {chosen!r}, not one of {question.options}"
                )
            chosen = chosen2
        return question.options.index(chosen)

    def _ask_gemini(self, system: str, prompt: str, shown: list[str]) -> str:
        """Single-select via Gemini enum mode (the structured-output analog of
        strict tool use). Returns exactly one of `shown`."""
        from google.genai import types

        config = types.GenerateContentConfig(
            system_instruction=system or None,
            temperature=self.temperature,
            response_mime_type="text/x.enum",
            response_schema={"type": "STRING", "enum": shown},
        )
        resp = self._gemini_generate(prompt, config)
        return (resp.text or "").strip()

    def _ask_distribution(self, system: str, question: Question) -> np.ndarray:
        """Ask the persona for the probability they would choose each option (a
        soft vote) and return it normalised over question.options. Averaging soft
        votes keeps each persona's residual uncertainty that a single hard pick
        discards. The framing is deliberately individual-uncertainty: asking 'how
        would people like you split' drifts into recalling published polls rather
        than simulating a view."""
        opts = question.options
        keys = [f"p{i}" for i in range(len(opts))]  # index keys dodge schema issues
        listing = "\n".join(f"  {k} = {o}" for k, o in zip(keys, opts))
        prompt = (
            f"{question.text}\n\nOptions:\n{listing}\n\nThinking about your own "
            "views and how certain or torn you personally feel, give the percentage "
            "chance you would pick each option. The numbers must sum to 100."
        )
        if self.provider == "gemini":
            from google.genai import types

            schema = {
                "type": "OBJECT",
                "properties": {k: {"type": "NUMBER", "description": o} for k, o in zip(keys, opts)},
                "required": keys,
            }
            cfg = types.GenerateContentConfig(
                system_instruction=system or None,
                temperature=self.temperature,
                response_mime_type="application/json",
                response_schema=schema,
            )
            resp = self._gemini_generate(prompt, cfg)
            d = json.loads(resp.text or "{}")
        else:
            tool = {
                "name": "record_distribution",
                "description": "Record your personal probability for each option (summing to 100).",
                "strict": True,
                "input_schema": {
                    "type": "object",
                    "properties": {k: {"type": "number"} for k in keys},
                    "required": keys,
                    "additionalProperties": False,
                },
            }
            msg = self.client.messages.create(
                model=self.model, max_tokens=256, temperature=self.temperature,
                system=system, tools=[tool],
                tool_choice={"type": "tool", "name": "record_distribution"},
                messages=[{"role": "user", "content": prompt}],
            )
            d = next(b.input for b in msg.content if b.type == "tool_use")
        vec = np.array([float(d.get(k, 0.0)) for k in keys])
        s = vec.sum()
        return vec / s if s > 0 else np.ones(len(opts)) / len(opts)

    def survey_distribution(self, personas: list[Persona], question: Question, progress: bool = True) -> np.ndarray:
        """Verbalized mode: per-persona probability vectors, shape [n_personas, k].
        Cached under a distinct '|verbalize' key so it never collides with the
        sampling answers."""
        vecs = []
        for i, p in enumerate(personas):
            key = f"{self.tag}|{p.method}|{p.id}|{question.key}|verbalize"
            if key not in self._cache:
                self._cache[key] = self._ask_distribution(p.system, question).tolist()
                self._save()
            vecs.append(self._cache[key])
            if progress and (i + 1) % 25 == 0:
                print(f"  {question.key} (verbalize): {i+1}/{len(personas)}", flush=True)
        self._save()
        return np.asarray(vecs, dtype=float)

    def _ask_anthropic(self, system: str, prompt: str, shown: list[str]) -> str:
        """Single-select via strict tool use with a forced enum."""
        tool = {
            "name": "record_answer",
            "description": "Record the single answer that best reflects your view.",
            "strict": True,
            "input_schema": {
                "type": "object",
                "properties": {"answer": {"type": "string", "enum": shown}},
                "required": ["answer"],
                "additionalProperties": False,
            },
        }
        msg = self.client.messages.create(
            model=self.model,
            max_tokens=128,
            temperature=self.temperature,
            system=system,
            tools=[tool],
            tool_choice={"type": "tool", "name": "record_answer"},
            messages=[{"role": "user", "content": prompt}],
        )
        return next(b.input["answer"] for b in msg.content if b.type == "tool_use")

    # -- persona x question, cached, with repeats -------------------------
    def survey(
        self,
        personas: list[Persona],
        question: Question,
        repeats: int = 1,
        progress: bool = True,
        frame_label: str = "",
        frame_text: str = "",
    ) -> np.ndarray:
        """Return the integer answer indices for every (persona, repeat) call.

        With `frame_label`/`frame_text` set, the same personas re-answer under a
        persuasive frame; results are cached under a distinct key so the baseline
        (no frame) and each framed pass never collide.
        """
        answers: list[int] = []
        total = len(personas) * repeats
        done = 0
        for p in personas:
            for r in range(repeats):
                key = f"{self.tag}|{p.method}|{p.id}|{question.key}|{r}"
                if frame_label:
                    key += f"|frame={frame_label}"
                if key not in self._cache:
                    seed = abs(hash(key)) % (2**31)
                    self._cache[key] = self._ask_once(p.system, question, seed, frame=frame_text)
                    self._save()  # save after every call - resume-safe on free tier
                answers.append(self._cache[key])
                done += 1
                if progress and done % 25 == 0:
                    print(f"  {question.key}: {done}/{total}", flush=True)
        self._save()
        return np.asarray(answers, dtype=int)

    # -- build backstory personas (method 4) ------------------------------
    def _generate_text(self, prompt: str, max_tokens: int = 400) -> str:
        """Free-form text generation, provider-dispatched (for backstories)."""
        if self.provider == "gemini":
            from google.genai import types

            config = types.GenerateContentConfig(
                temperature=1.0, max_output_tokens=max_tokens
            )
            resp = self._gemini_generate(prompt, config)
            return resp.text or ""
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=1.0,
            messages=[{"role": "user", "content": prompt}],
        )
        return next(b.text for b in resp.content if b.type == "text")

    def build_backstory_personas(self, n: int = 100, seed: int = 3) -> list[Persona]:
        """Generate a first-person narrative per demographic seed, cached on disk.
        Stories are namespaced by provider so the two backends never mix."""
        story_path = CACHE_DIR / f"backstories_{self.tag}.json"
        stories = json.loads(story_path.read_text()) if story_path.exists() else {}
        seeds = backstory_seeds(n, seed=seed)
        out = []
        for i, attrs in enumerate(seeds):
            pid = f"story-{i:03d}"
            if pid not in stories:
                stories[pid] = self._generate_text(backstory_prompt(attrs), max_tokens=400)
                story_path.parent.mkdir(parents=True, exist_ok=True)
                story_path.write_text(json.dumps(stories, indent=0))
                if i % 10 == 0:
                    print(f"  backstory {i+1}/{n}", flush=True)
            out.append(
                Persona(
                    id=pid,
                    method="backstory",
                    system=backstory_system(attrs, stories[pid]),
                    attrs=attrs,
                )
            )
        return out


def to_distribution(answers: np.ndarray, k: int) -> np.ndarray:
    """Convert raw answer indices into a probability vector over k options."""
    counts = np.bincount(answers, minlength=k).astype(float)
    return counts / counts.sum() if counts.sum() else counts
