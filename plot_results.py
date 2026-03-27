"""
plot_results.py — jedini owner grafičkih funkcionalnosti

Javne funkcije:
    plot_summary(json_path)
        2x2 graf po-kreatura (fitness, reward, distance, upright)
        poziva se on_generation iz GA.py

    plot_progress(out_path, best_fitness_each_gen, last_gen_fitness, title)
        1x2 graf napretka (best/gen + histogram)
        poziva se na kraju GA.py

    generate_total_summary(experiment_dir)
        čita sve run_N/run_N.json unutar experiment_dir,
        računa prosjek metrika svih runova i izdvaja globalno najbolje biće,
        sprema totalSummary.json u experiment_dir

    plot_total_summary(experiment_dir)
        čita totalSummary.json i generira totalSummary.png

Standalone pokretanje:
    python plot_results.py results/eksperiment_1/run_1/run_1.json
    python plot_results.py results/eksperiment_1/
    python plot_results.py --total results/eksperiment_1/
"""

import os
import sys
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive — radi bez displaya i u subprocessima
import matplotlib.pyplot as plt
from pathlib import Path


# ══════════════════════════════════════════════════════════════════════════════
#  Summary graf (2x2) — čita iz JSON-a, crta po-kreatura
# ══════════════════════════════════════════════════════════════════════════════

def plot_summary(json_path: str | Path) -> str | None:
    """
    Čita run JSON i sprema 2x2 summary PNG pored njega.
    Vraća putanju PNG-a ili None ako nema podataka.

    Poziva se iz GA.py on_generation i može se pozvati standalone.
    """
    json_path = Path(json_path)
    if not json_path.exists():
        print(f"[plot] File not found: {json_path}")
        return None

    with open(json_path) as f:
        doc = json.load(f)

    creatures = doc.get("creatures", [])
    if not creatures:
        print(f"[plot] No creatures yet in {json_path.name}, skipping.")
        return None

    # Sort by generation then creature index
    def _sort_key(c):
        rid = c.get("run_id", "Gen0Creature0")
        try:
            parts = str(rid).replace("Gen", "").split("Creature")
            return (int(parts[0]), int(parts[1]))
        except (ValueError, IndexError):
            return (0, 0)
    creatures.sort(key=_sort_key)

    fitness   = [c["fitness_score"] for c in creatures]
    rewards   = [c["mean_reward"]   for c in creatures]
    distances = [c["mean_distance"] for c in creatures]
    uprights  = [c["mean_upright"]  for c in creatures]
    x         = list(range(len(creatures)))

    best_so_far = list(np.maximum.accumulate(fitness))

    # Generation boundary lines
    exp_info = doc.get("experiment", {})
    ga_cfg   = exp_info.get("ga_settings", {})

    gen_boundaries = []
    gen_labels     = {}
    prev_gen = None
    for i, c in enumerate(creatures):
        rid = c.get("run_id", "")
        try:
            g = int(str(rid).replace("Gen", "").split("Creature")[0])
        except (ValueError, IndexError):
            g = None
        if g is not None and g != prev_gen:
            gen_boundaries.append(i)
            gen_labels[i] = f"G{g}"
            prev_gen = g

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    title = (f"{json_path.stem}  |  "
             f"pop={ga_cfg.get('population_size', '?')}  "
             f"gen={ga_cfg.get('num_generations', '?')}  "
             f"legs={ga_cfg.get('num_legs', '?')}")
    fig.suptitle(title, fontsize=12, fontweight="bold")

    def _plot(ax, y, label, color, extra=None):
        ax.plot(x, y, marker="o", markersize=2,
                color=color, linewidth=1.0, alpha=0.75, label=label)
        if extra is not None:
            ax.plot(x, extra, color=color, linewidth=2,
                    linestyle="--", alpha=0.6, label="best so far")

        if len(x) >= 2:
            m, b = np.polyfit(x, y, 1)
            ax.plot(x, [m * xi + b for xi in x],
                    color="red", linewidth=1.8, linestyle="-", alpha=0.9,
                    label=f"trend (m={m:.4f})")

        ax.legend(fontsize=8)

        for bx in gen_boundaries:
            ax.axvline(x=bx, color="gray", linewidth=0.6, linestyle=":", alpha=0.7)

        if gen_labels:
            tick_positions = list(gen_labels.keys())
            tick_names     = list(gen_labels.values())
            max_ticks = 20
            if len(tick_positions) > max_ticks:
                step = len(tick_positions) // max_ticks
                tick_positions = tick_positions[::step]
                tick_names     = tick_names[::step]
            ax.set_xticks(tick_positions)
            ax.set_xticklabels(tick_names, fontsize=7, rotation=45, ha="right")
        else:
            ax.set_xlabel("Creature index")

        ax.set_ylabel(label)
        ax.set_title(label)
        ax.grid(True, alpha=0.3)

    _plot(axes[0, 0], fitness,   "Composite Fitness",    "#2196F3", extra=best_so_far)
    _plot(axes[0, 1], rewards,   "Mean Reward",          "#4CAF50")
    _plot(axes[1, 0], distances, "Mean Forward Distance", "#FF9800")
    _plot(axes[1, 1], uprights,  "Mean Upright Fraction", "#9C27B0")

    plt.tight_layout()
    out_png = json_path.with_name(json_path.stem + "_summary.png")
    os.makedirs(out_png.parent, exist_ok=True)
    plt.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"[plot] Summary saved → {out_png}")
    return str(out_png)


# ══════════════════════════════════════════════════════════════════════════════
#  Progress graf (1x2) — crta se na kraju GA.py
# ══════════════════════════════════════════════════════════════════════════════

def plot_progress(out_path: str | Path,
                  best_fitness_each_gen: list,
                  last_gen_fitness: list,
                  title: str = "GA Evolucija") -> str | None:
    """
    Sprema 1x2 progress PNG:
      lijevo  — best fitness po generaciji + best-so-far krivulja
      desno   — histogram fitnessa zadnje generacije

    Parametri:
        out_path              — putanja PNG izlaza (npr. RUN_PREFIX + "_progress.png")
        best_fitness_each_gen — lista best fitnessa po generaciji (iz on_generation)
        last_gen_fitness      — lista fitnessa svih jedinki zadnje generacije
        title                 — naslov grafa
    """
    if not best_fitness_each_gen:
        print("[plot] No generation data for progress plot, skipping.")
        return None

    out_path = Path(out_path)
    os.makedirs(out_path.parent, exist_ok=True)

    arr         = np.array(best_fitness_each_gen, dtype=np.float32)
    best_so_far = np.maximum.accumulate(arr)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(title, fontsize=13, fontweight="bold")

    axes[0].plot(arr,         label="Best generacije", color="#2196F3", linewidth=2)
    axes[0].plot(best_so_far, label="Best do sada",    color="#FF5722", linewidth=2, linestyle="--")
    axes[0].set_xlabel("Generacija")
    axes[0].set_ylabel("Fitness")
    axes[0].set_title("Napredak fitnessa")
    axes[0].legend()
    axes[0].grid(True, alpha=0.4)

    axes[1].hist(last_gen_fitness, bins=10, color="#4CAF50", edgecolor="white")
    axes[1].set_xlabel("Fitness")
    axes[1].set_ylabel("Broj jedinki")
    axes[1].set_title("Distribucija fitnessa (zadnja generacija)")
    axes[1].grid(True, alpha=0.4)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[plot] Progress saved → {out_path}")
    return str(out_path)



# ══════════════════════════════════════════════════════════════════════════════
#  Total summary JSON — agregira sve runove eksperimenta
# ══════════════════════════════════════════════════════════════════════════════

def generate_total_summary(experiment_dir: str | Path) -> str | None:
    """
    Čita sve run_N/run_N.json unutar experiment_dir, računa prosjek metrika
    svih runova i izdvaja globalno najbolje biće.
    Sprema totalSummary.json u experiment_dir i vraća putanju.

    Struktura totalSummary.json:
    {
      "experiment":    { naziv, ga_settings, fitness_weights, ... },
      "runs_included": ["run_1", "run_2", ...],
      "averaged_metrics": {
          "fitness_score":   { mean, std, min, max },
          "mean_reward":     { mean, std, min, max },
          "mean_distance":   { mean, std, min, max },
          "mean_upright":    { mean, std, min, max },
      },
      "best_creature": {   <- globalno najfit biće iz svih runova
          "from_run":    "run_3",
          "run_id":      "Gen18Creature5",
          "generation":  18,
          "fitness_score": ...,
          "mean_reward":   ...,
          "mean_distance": ...,
          "mean_upright":  ...,
      }
    }
    """
    from datetime import datetime

    experiment_dir = Path(experiment_dir)
    if not experiment_dir.is_dir():
        print(f"[total] Not a directory: {experiment_dir}")
        return None

    # Pronađi sve run_N/run_N.json unutar experiment_dir
    run_jsons = sorted(
        f for f in experiment_dir.glob("*/")
        if f.is_dir() and (f / f"{f.name}.json").exists()
    )
    run_jsons = [r / f"{r.name}.json" for r in run_jsons]

    if not run_jsons:
        print(f"[total] No run JSON files found in {experiment_dir}")
        return None

    print(f"[total] Found {len(run_jsons)} run(s): {[r.parent.name for r in run_jsons]}")

    # Učitaj sve runove
    runs = []
    experiment_meta = None
    for rj in run_jsons:
        with open(rj) as f:
            d = json.load(f)
        summary = d.get("summary", {})
        if not summary:
            print(f"[total] WARNING: {rj} has no summary, skipping.")
            continue
        runs.append({"run_name": rj.parent.name, "summary": summary, "doc": d})
        if experiment_meta is None:
            experiment_meta = d.get("experiment", {})

    if not runs:
        print("[total] No runs with valid summaries found.")
        return None

    # Prosjek metrika best_creature po runu
    def _avg(key):
        vals = [r["summary"]["best_creature"].get(key, 0.0) for r in runs]
        return {
            "mean": round(float(np.mean(vals)), 6),
            "std":  round(float(np.std(vals)),  6),
            "min":  round(float(np.min(vals)),  6),
            "max":  round(float(np.max(vals)),  6),
            "values_per_run": {r["run_name"]: round(float(v), 6)
                               for r, v in zip(runs, vals)},
        }

    # Globalno najbolje biće
    best_run  = max(runs, key=lambda r: r["summary"]["best_creature"].get("fitness_score", 0.0))
    best_bc   = best_run["summary"]["best_creature"].copy()
    best_bc["from_run"] = best_run["run_name"]

    # Per-generation aggregates — averaged across all runs
    # For each generation: mean, std, best fitness across all creatures in that gen
    def _parse_gen(rid):
        try:
            return int(str(rid).replace("Gen", "").split("Creature")[0])
        except (ValueError, IndexError):
            return None

    # Collect per-generation data from all runs
    gen_data = {}   # gen_number -> {"fitness": [], "reward": [], "distance": [], "upright": []}
    for run in runs:
        for c in run["doc"].get("creatures", []):
            g = _parse_gen(c.get("run_id", ""))
            if g is None:
                continue
            if g not in gen_data:
                gen_data[g] = {"fitness": [], "reward": [], "distance": [], "upright": []}
            gen_data[g]["fitness"].append(c.get("fitness_score", 0.0))
            gen_data[g]["reward"].append(c.get("mean_reward",   0.0))
            gen_data[g]["distance"].append(c.get("mean_distance", 0.0))
            gen_data[g]["upright"].append(c.get("mean_upright",  0.0))

    per_generation = []
    for g in sorted(gen_data.keys()):
        gd = gen_data[g]
        per_generation.append({
            "generation":       g,
            "n_creatures":      len(gd["fitness"]),
            "fitness_mean":     round(float(np.mean(gd["fitness"])),   6),
            "fitness_std":      round(float(np.std(gd["fitness"])),    6),
            "fitness_best":     round(float(np.max(gd["fitness"])),    6),
            "reward_mean":      round(float(np.mean(gd["reward"])),    6),
            "reward_std":       round(float(np.std(gd["reward"])),     6),
            "distance_mean":    round(float(np.mean(gd["distance"])),  6),
            "distance_std":     round(float(np.std(gd["distance"])),   6),
            "upright_mean":     round(float(np.mean(gd["upright"])),   6),
            "upright_std":      round(float(np.std(gd["upright"])),    6),
        })

    total = {
        "generated_at":   datetime.now().isoformat(timespec="seconds"),
        "experiment_dir": str(experiment_dir),
        "experiment":     experiment_meta,
        "runs_included":  [r["run_name"] for r in runs],
        "n_runs":         len(runs),
        "averaged_metrics": {
            "fitness_score": _avg("fitness_score"),
            "mean_reward":   _avg("mean_reward"),
            "mean_distance": _avg("mean_distance"),
            "mean_upright":  _avg("mean_upright"),
        },
        "best_creature":  best_bc,
        "per_generation": per_generation,
    }

    out_path = experiment_dir / "totalSummary.json"
    with open(out_path, "w") as f:
        json.dump(total, f, indent=2)
    print(f"[total] Saved → {out_path}")
    return str(out_path)


# ══════════════════════════════════════════════════════════════════════════════
#  Total summary graf — vizualizacija totalSummary.json
# ══════════════════════════════════════════════════════════════════════════════

def plot_total_summary(experiment_dir: str | Path) -> str | None:
    """
    Čita totalSummary.json iz experiment_dir i sprema totalSummary.png.

    Graf prikazuje 4 metrike (fitness, reward, distance, upright) kao
    bar chart s mean ± std po runu + horizontalna linija za prosjek.
    """
    experiment_dir = Path(experiment_dir)
    json_path = experiment_dir / "totalSummary.json"

    if not json_path.exists():
        print(f"[total] totalSummary.json not found in {experiment_dir}. "
              f"Run generate_total_summary() first.")
        return None

    with open(json_path) as f:
        doc = json.load(f)

    runs     = doc["runs_included"]
    metrics  = doc["averaged_metrics"]
    best_bc  = doc["best_creature"]
    exp_meta = doc.get("experiment", {})
    ga_cfg   = exp_meta.get("ga_settings", {})

    # Per-run values for each metric
    def _vals(key):
        return [metrics[key]["values_per_run"].get(r, 0.0) for r in runs]

    fit_vals  = _vals("fitness_score")
    rew_vals  = _vals("mean_reward")
    dist_vals = _vals("mean_distance")
    upr_vals  = _vals("mean_upright")

    x      = np.arange(len(runs))
    width  = 0.6
    labels = runs

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    exp_name = experiment_dir.name
    title = (f"{exp_name}  |  Total Summary  |  "
             f"pop={ga_cfg.get('population_size', '?')}  "
             f"gen={ga_cfg.get('num_generations', '?')}  "
             f"n_runs={doc['n_runs']}")
    fig.suptitle(title, fontsize=12, fontweight="bold")

    def _bar(ax, vals, label, color, mean_val):
        bars = ax.bar(x, vals, width, color=color, alpha=0.75, label=label)

        # Value labels on top of bars
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{v:.3f}", ha="center", va="bottom", fontsize=8)

        # Mean line
        ax.axhline(mean_val, color="red", linewidth=1.8, linestyle="--",
                   label=f"mean = {mean_val:.4f}")

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9, rotation=30, ha="right")
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")

    _bar(axes[0, 0], fit_vals,  "Composite Fitness",    "#2196F3", metrics["fitness_score"]["mean"])
    _bar(axes[0, 1], rew_vals,  "Mean Reward",           "#4CAF50", metrics["mean_reward"]["mean"])
    _bar(axes[1, 0], dist_vals, "Mean Forward Distance",  "#FF9800", metrics["mean_distance"]["mean"])
    _bar(axes[1, 1], upr_vals,  "Mean Upright Fraction",  "#9C27B0", metrics["mean_upright"]["mean"])

    # Annotation: best creature
    best_text = (f"Best creature: {best_bc['from_run']}  |  "
                 f"fitness={best_bc['fitness_score']:.4f}  |  "
                 f"reward={best_bc['mean_reward']:.1f}  |  "
                 f"dist={best_bc['mean_distance']:.3f}")
    fig.text(0.5, 0.01, best_text, ha="center", fontsize=9,
             style="italic", color="#333333")

    plt.tight_layout(rect=[0, 0.03, 1, 1])
    out_png = experiment_dir / "totalSummary.png"
    plt.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"[total] Plot saved → {out_png}")
    return str(out_png)




# ══════════════════════════════════════════════════════════════════════════════
#  Experiment comparison graf — linijski grafovi za 2+ eksperimenta
# ══════════════════════════════════════════════════════════════════════════════

# Colours cycling for up to 8 experiments
_EXP_COLOURS = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63",
                 "#9C27B0", "#00BCD4", "#FF5722", "#607D8B"]


def plot_experiment_comparison(experiment_dirs: list[str | Path],
                                out_path: str | Path | None = None,
                                metric: str = "all") -> str | None:
    """
    Čita totalSummary.json iz svakog experiment_dir i crta linijske grafove
    koji uspoređuju evoluciju po generacijama za 2 ili više eksperimenta.

    Parametri:
        experiment_dirs  — lista putanja do experiment foldera (svaki mora imati
                           totalSummary.json, generiraj s generate_total_summary())
        out_path         — putanja PNG izlaza; ako None, sprema se pored prvog
                           experiment_dir kao "comparison.png"
        metric           — "all" (2x2 grid) ili jedna od:
                           "fitness", "reward", "distance", "upright"

    Linijski grafovi prikazuju:
        — mean po generaciji (deblja linija)
        — best po generaciji (tanka isprekidana, samo za fitness)
        — ± std sjena oko linije
    """
    dirs = [Path(d) for d in experiment_dirs]

    # Učitaj totalSummary.json za svaki eksperiment
    summaries = []
    for d in dirs:
        jp = d / "totalSummary.json"
        if not jp.exists():
            print(f"[compare] totalSummary.json not found in {d} — "
                  f"run generate_total_summary() first.")
            return None
        with open(jp) as f:
            doc = json.load(f)
        if "per_generation" not in doc or not doc["per_generation"]:
            print(f"[compare] No per_generation data in {d}/totalSummary.json — "
                  f"regenerate with updated generate_total_summary().")
            return None
        summaries.append({"name": d.name, "doc": doc})

    if len(summaries) < 2:
        print("[compare] Need at least 2 experiment directories to compare.")
        return None

    # Helper: extract generation series from per_generation list
    def _series(pg, mean_key, std_key, best_key=None):
        gens  = [e["generation"]   for e in pg]
        means = [e[mean_key]       for e in pg]
        stds  = [e[std_key]        for e in pg]
        bests = [e[best_key]       for e in pg] if best_key else None
        return np.array(gens), np.array(means), np.array(stds), \
               (np.array(bests) if bests is not None else None)

    metrics_cfg = {
        "fitness":  ("fitness_mean",   "fitness_std",   "fitness_best",  "Composite Fitness"),
        "reward":   ("reward_mean",    "reward_std",    None,            "Mean Reward"),
        "distance": ("distance_mean",  "distance_std",  None,            "Mean Forward Distance"),
        "upright":  ("upright_mean",   "upright_std",   None,            "Mean Upright Fraction"),
    }

    selected = list(metrics_cfg.keys()) if metric == "all" else [metric]
    if metric != "all" and metric not in metrics_cfg:
        print(f"[compare] Unknown metric '{metric}'. Choose from: all, fitness, reward, distance, upright")
        return None

    if metric == "all":
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes_flat = [axes[0,0], axes[0,1], axes[1,0], axes[1,1]]
    else:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        axes_flat = [ax]

    exp_names = " vs ".join(s["name"] for s in summaries)
    fig.suptitle(f"Experiment Comparison  |  {exp_names}", fontsize=12, fontweight="bold")

    for ax, met_key in zip(axes_flat, selected):
        mean_k, std_k, best_k, label = metrics_cfg[met_key]

        for i, s in enumerate(summaries):
            colour = _EXP_COLOURS[i % len(_EXP_COLOURS)]
            pg     = s["doc"]["per_generation"]
            gens, means, stds, bests = _series(pg, mean_k, std_k, best_k)

            # Mean line
            ax.plot(gens, means, color=colour, linewidth=2.2,
                    label=f"{s['name']} mean")

            # Std shading
            ax.fill_between(gens, means - stds, means + stds,
                            color=colour, alpha=0.15)

            # Best line (fitness only)
            if bests is not None:
                ax.plot(gens, bests, color=colour, linewidth=1.2,
                        linestyle="--", alpha=0.7, label=f"{s['name']} best")

        ax.set_xlabel("Generacija")
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # Hide unused axes if single metric
    if metric == "all":
        for ax in axes_flat[len(selected):]:
            ax.set_visible(False)

    plt.tight_layout()

    if out_path is None:
        out_path = dirs[0].parent / "comparison.png"
    out_path = Path(out_path)
    os.makedirs(out_path.parent, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[compare] Saved → {out_path}")
    return str(out_path)



# ══════════════════════════════════════════════════════════════════════════════
#  Runs progress graf — fitness kretanje svih runova unutar eksperimenta
# ══════════════════════════════════════════════════════════════════════════════

def plot_runs_progress(experiment_dir: str | Path) -> str | None:
    """
    Za svaki run unutar experiment_dir crta best fitness po generaciji
    kao zasebnu liniju na istom grafu — omogućuje usporedbu varijabilnosti
    između ponavljanja istog eksperimenta.
    Sprema progress.png u experiment_dir.
    """
    experiment_dir = Path(experiment_dir)

    # Pronađi sve run_N/run_N.json
    run_dirs = sorted(
        d for d in experiment_dir.glob("*/")
        if d.is_dir() and (d / f"{d.name}.json").exists()
    )

    if not run_dirs:
        print(f"[progress] No run JSON files found in {experiment_dir}")
        return None

    fig, ax = plt.subplots(figsize=(12, 6))
    exp_name = experiment_dir.name
    ax.set_title(f"{exp_name}  |  Best fitness po generaciji — svi runovi",
                 fontsize=12, fontweight="bold")

    all_best_series = []

    for i, run_dir in enumerate(run_dirs):
        colour   = _EXP_COLOURS[i % len(_EXP_COLOURS)]
        run_name = run_dir.name
        json_path = run_dir / f"{run_name}.json"

        with open(json_path) as f:
            doc = json.load(f)

        creatures = doc.get("creatures", [])
        if not creatures:
            continue

        # Best fitness per generation for this run
        gen_best = {}
        for c in creatures:
            rid = c.get("run_id", "")
            try:
                g = int(str(rid).replace("Gen", "").split("Creature")[0])
            except (ValueError, IndexError):
                continue
            fit = c.get("fitness_score", 0.0)
            if g not in gen_best or fit > gen_best[g]:
                gen_best[g] = fit

        if not gen_best:
            continue

        gens = sorted(gen_best.keys())
        fits = [gen_best[g] for g in gens]
        all_best_series.append(fits)

        ax.plot(gens, fits, color=colour, linewidth=1.8, alpha=0.85,
                marker="o", markersize=3, label=run_name)

    # Mean across runs per generation
    if len(all_best_series) >= 2:
        min_len  = min(len(s) for s in all_best_series)
        arr      = np.array([s[:min_len] for s in all_best_series])
        mean_fit = arr.mean(axis=0)
        ax.plot(list(range(min_len)), mean_fit, color="black", linewidth=2.5,
                linestyle="--", alpha=0.9, label="mean svih runova")

    ax.set_xlabel("Generacija")
    ax.set_ylabel("Best Fitness")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_png = experiment_dir / "progress.png"
    plt.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"[progress] Saved → {out_png}")
    return str(out_png)


# ══════════════════════════════════════════════════════════════════════════════
#  Standalone CLI
# ══════════════════════════════════════════════════════════════════════════════

def _collect_json_files(paths: list[str]) -> list[Path]:
    result = []
    for p in paths:
        p = Path(p)
        if p.is_file() and p.suffix == ".json":
            result.append(p)
        elif p.is_dir():
            found = sorted([
                f for f in p.rglob("*.json")
                if "best_creature" not in f.name
                and "totalSummary"  not in f.name
            ])
            if not found:
                print(f"[plot] No JSON files found in {p}")
            result.extend(found)
        else:
            print(f"[plot] Skipping: {p}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate plots from GA run JSON files."
    )
    parser.add_argument("paths", nargs="+",
                        help=".json files or experiment/run directories")
    parser.add_argument("--total", action="store_true",
                        help="Generate totalSummary.json + totalSummary.png "
                             "for the given experiment directory")
    parser.add_argument("--compare", action="store_true",
                        help="Compare 2+ experiment directories using totalSummary.json")
    parser.add_argument("--metric", type=str, default="all",
                        help="Metric to compare: all | fitness | reward | distance | upright")
    parser.add_argument("--out", type=str, default=None,
                        help="Output PNG path for --compare (default: comparison.png next to first experiment)")
    args = parser.parse_args()

    if args.compare:
        plot_experiment_comparison(args.paths, out_path=args.out, metric=args.metric)
    elif args.total:
        for p in args.paths:
            generate_total_summary(p)
            plot_total_summary(p)
            plot_runs_progress(p)
    else:
        json_files = _collect_json_files(args.paths)
        if not json_files:
            print("[plot] Nothing to plot.")
            sys.exit(1)
        print(f"[plot] Plotting {len(json_files)} file(s)...")
        for jf in json_files:
            plot_summary(jf)