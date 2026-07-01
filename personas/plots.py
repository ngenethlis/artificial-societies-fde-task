"""matplotlib figures for the report.

Per question: a fidelity scorecard (methods x TVD with the noise floor drawn in),
a real-vs-simulated topline for the best method, and a subgroup-fidelity panel.
The ungrounded application question gets an opinion-split chart and, if a framing
experiment ran, a net-support-by-frame chart.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"figure.dpi": 130, "font.size": 10})


def render(scorecard: dict, out: Path) -> None:
    grounded = {k: q for k, q in scorecard.items() if q.get("real") is not None}
    if grounded:
        _scorecard_chart(grounded, out / "fig_scorecard.png")
    for qkey, q in scorecard.items():
        if q.get("real") is not None:
            _real_vs_sim(qkey, q, out / f"fig_{qkey}_topline.png")
            _subgroup_chart(qkey, q, out / f"fig_{qkey}_subgroups.png")
        else:  # application question - no ground truth; show who splits which way
            _application_chart(qkey, q, out / f"fig_{qkey}_application.png")
        if q.get("framing"):  # persuasion experiment: does opinion MOVE?
            _framing_chart(qkey, q, out / f"fig_{qkey}_framing.png")


def _scorecard_chart(sc: dict, path: Path) -> None:
    qkeys = list(sc)
    methods = list(next(iter(sc.values()))["methods"])
    fig, axes = plt.subplots(1, len(qkeys), figsize=(5 * len(qkeys), 4), squeeze=False)
    for ax, qkey in zip(axes[0], qkeys):
        e = sc[qkey]["methods"]
        tvds = [e[m]["tvd"] for m in methods]
        floor = e[methods[0]]["tvd_floor_p95"]
        # green = beats floor (a real sample); blue = verbalized ESTIMATE (floor
        # n/a, don't read as pass/fail); red = real sample that misses the floor.
        colors = [
            "#5b8fb0" if e[m].get("estimate") else ("#4c9f70" if e[m].get("beats_noise") else "#c0504d")
            for m in methods
        ]
        ax.bar(methods, tvds, color=colors)
        ax.axhline(floor, ls="--", color="#333", lw=1)
        ax.text(len(methods) - 0.5, floor, " noise floor (n=100)", va="bottom", ha="right", fontsize=8)
        ax.set_title(f"{qkey}: distance to real opinion (TVD)")
        ax.set_ylabel("Total Variation Distance (lower = better)")
        ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _best_method(q: dict) -> str:
    # Exclude verbalized ESTIMATES: a smooth estimate can have the lowest TVD
    # mechanically, so letting it win "best method" would be a false comparison.
    cands = {m: v for m, v in q["methods"].items() if not v.get("estimate")} or q["methods"]
    if q.get("real") is None:  # ungrounded: prefer the richest method
        return "backstory" if "backstory" in cands else next(iter(cands))
    return min(cands, key=lambda m: cands[m]["tvd"])


def _application_chart(qkey: str, q: dict, path: Path) -> None:
    """Stacked opinion split for the whole group and each subgroup (ungrounded
    question, so nothing to score it against)."""
    method = _best_method(q)
    e = q["methods"][method]
    options = q["options"]
    rows = [("All GB adults", e["sim"])]
    for label, sub in e.get("subgroups", {}).items():
        rows.append((label, sub["sim"]))
    labels = [r[0] for r in rows]
    data = np.array([r[1] for r in rows])  # rows x options
    y = np.arange(len(labels))
    cmap = plt.get_cmap("RdYlBu")
    colors = [cmap(i / max(1, len(options) - 1)) for i in range(len(options))]
    fig, ax = plt.subplots(figsize=(8, 0.6 * len(labels) + 1.5))
    left = np.zeros(len(labels))
    for j, opt in enumerate(options):
        ax.barh(y, data[:, j], left=left, color=colors[j], label=opt)
        left += data[:, j]
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Share of respondents")
    ax.set_title(f"{qkey}: predicted opinion split ({method})\n[no poll exists - unvalidated]")
    ax.legend(ncol=len(options), fontsize=7, loc="lower center", bbox_to_anchor=(0.5, -0.25))
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _framing_chart(qkey: str, q: dict, path: Path) -> None:
    """Net support by segment under each frame (grouped bars). The gap between a
    segment's bars is how far that segment moves between messages."""
    fr = q["framing"]
    pos = fr["positive"]
    frames = list(fr["frames"])
    net = lambda sim: 100 * sum(sim[i] for i in pos)
    segments = ["All GB adults"]
    for label in fr["frames"][frames[0]].get("subgroups", {}):
        segments.append(label)

    def seg_value(seg, frame):
        fe = fr["frames"][frame]
        if seg == "All GB adults":
            return net(fe["sim"])
        return net(fe["subgroups"][seg]["sim"])

    x = np.arange(len(segments))
    w = 0.8 / len(frames)
    cmap = plt.get_cmap("coolwarm")
    fig, ax = plt.subplots(figsize=(1.4 * len(segments) + 2, 4.5))
    for j, frame in enumerate(frames):
        vals = [seg_value(s, frame) for s in segments]
        ax.bar(x + (j - (len(frames) - 1) / 2) * w, vals, width=w,
               label=frame, color=cmap(j / max(1, len(frames) - 1)))
    ax.set_xticks(x)
    ax.set_xticklabels(segments, rotation=20, ha="right")
    ax.set_ylabel("Net support (%)")
    ax.set_ylim(0, 100)
    ax.set_title(f"{qkey}: does opinion MOVE under different framing?\n"
                 f"(net support by segment x message frame; {fr['method']})")
    ax.legend(title="message frame", fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _real_vs_sim(qkey: str, q: dict, path: Path) -> None:
    options = q["options"]
    best = _best_method(q)
    real = np.asarray(q["real"])
    sim = np.asarray(q["methods"][best]["sim"])
    x = np.arange(len(options))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - 0.2, real, width=0.4, label="Real (YouGov)", color="#33548c")
    ax.bar(x + 0.2, sim, width=0.4, label=f"Simulated ({best})", color="#e08214")
    ax.set_xticks(x)
    ax.set_xticklabels(options, rotation=20, ha="right")
    ax.set_ylabel("Share of respondents")
    ax.set_title(f"{qkey}: real vs simulated (best method = {best})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _subgroup_chart(qkey: str, q: dict, path: Path) -> None:
    best = _best_method(q)
    subs = q["methods"][best].get("subgroups", {})
    labels = [s for s in subs if "tvd" in subs[s]]  # only scored subgroups
    if not labels:
        return
    sim_tvd = [subs[s]["tvd"] for s in labels]
    floor = [subs[s]["tvd_floor_p95"] for s in labels]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x, sim_tvd, width=0.5, color="#e08214", label="simulated vs real (TVD)")
    ax.plot(x, floor, "k--", marker="o", lw=1, label="n-matched noise floor")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Total Variation Distance")
    ax.set_title(f"{qkey}: subgroup fidelity ({best}) - does it capture WHO disagrees?")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
