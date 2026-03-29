"""
scripts/05_generate_all_plots.py — Generate All 8 Publication-Quality Plots
============================================================================
Reads ALL data exclusively from results/ and dataset/.
ZERO imports from torch, gymnasium, transformers, or src/algorithms.
Safe to run on a CPU instance with no GPU.

Run: python scripts/05_generate_all_plots.py
Requires: results/*_metrics.json (at minimum: 01_seed_published_results.py)
Outputs:  plots/plot1_*.png through plots/plot8_*.png
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.patches import FancyBboxPatch
import matplotlib.colors as mcolors
import seaborn as sns
from pathlib import Path

# ── Global style ───────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11,
    "axes.titlesize": 13, "axes.titleweight": "bold",
    "figure.dpi": 150, "savefig.dpi": 150, "savefig.bbox": "tight",
})

# Consistent colour mapping — use C[key] everywhere, no exceptions
C = {
    "nasdaq_100":           "#000000",
    "cppo_deepseek":        "#adb5bd",
    "dapo_sr":              "#4895ef",
    "grpo_vanilla":         "#fd7e14",
    "llm_call":             "#dee2e6",
    "movr_acc_only":        "#b5e48c",
    "movr_sharpe_only":     "#76c893",
    "movr_mdd_only":        "#52b69a",
    "movr_balanced":        "#1a6b3c",
    "movr_sentiment_heavy": "#34a0a4",
}

RESULTS = Path("results")
PLOTS   = Path("plots")
PLOTS.mkdir(exist_ok=True)


def load_metrics(key: str) -> dict:
    path = RESULTS / f"{key}_metrics.json"
    if not path.exists():
        print(f"  WARNING: {path} not found — skipping in plots")
        return {}
    with open(path) as f:
        d = json.load(f)
    return {k: (np.nan if v is None else v) for k, v in d.items()}


def load_backtest(key: str):
    path = RESULTS / f"{key}_backtest.csv"
    if not path.exists():
        return None
    return pd.read_csv(path, parse_dates=["date"])


# ── Plot 1 — Prior work comparison ────────────────────────────────────────
def plot_1_prior_work_comparison():
    nasdaq = load_metrics("nasdaq_100")
    cppo   = load_metrics("cppo_deepseek")
    dapo   = load_metrics("dapo_sr_2019")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Prior Work Baseline — arXiv:2505.06408", fontsize=14, fontweight="bold")

    # Left: cumulative return bars
    methods = ["NASDAQ-100\n(Buy & Hold)", "CPPO-DeepSeek\n(2020-2023)", "DAPO-SR\n(2019-2023)"]
    returns = [
        (nasdaq.get("cumulative_return", np.nan) or np.nan) * 100,
        (cppo.get("cumulative_return", np.nan) or np.nan) * 100,
        (dapo.get("cumulative_return", np.nan) or np.nan) * 100,
    ]
    colors = [C["nasdaq_100"], C["cppo_deepseek"], C["dapo_sr"]]
    bars = ax1.bar(methods, returns, color=colors, width=0.55, edgecolor="white", linewidth=1.5)
    for bar, val in zip(bars, returns):
        if not np.isnan(val):
            ax1.annotate(f"{val:.1f}%", xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                         xytext=(0, 5), textcoords="offset points", ha="center", fontsize=10, fontweight="bold")
    ax1.set_ylabel("Cumulative Return (%)")
    ax1.set_title("Cumulative Return (2019–2023)")
    ax1.set_ylim(0, max(v for v in returns if not np.isnan(v)) * 1.25)
    ax1.text(0.5, -0.12, "† Values from arXiv:2505.06408, Tables 1 & 2",
             transform=ax1.transAxes, ha="center", fontsize=8, color="gray")

    # Right: training time & RAM comparison
    method_labels = ["CPPO-DeepSeek", "DAPO-SR"]
    time_hrs = [cppo.get("training_time_hrs", np.nan), dapo.get("training_time_hrs", np.nan)]
    ram_gb   = [cppo.get("ram_usage_gb", np.nan), dapo.get("ram_usage_gb", np.nan)]

    x = np.arange(len(method_labels))
    width = 0.35
    ax2b = ax2.twinx()
    ax2.bar(x - width/2, time_hrs, width, color=[C["cppo_deepseek"], C["dapo_sr"]],
            label="Training Time (hrs)", alpha=0.85)
    ax2b.bar(x + width/2, ram_gb, width, color=[C["cppo_deepseek"], C["dapo_sr"]],
             label="RAM (GB)", alpha=0.5, hatch="//")
    ax2.set_xticks(x); ax2.set_xticklabels(method_labels)
    ax2.set_ylabel("Training Time (hours)"); ax2b.set_ylabel("RAM Usage (GB)")
    ax2.set_title("Computational Cost Comparison")
    ax2.annotate("3× faster, 8× less RAM\nvs CPPO-DeepSeek", xy=(1, time_hrs[1]),
                 xytext=(0, 15), textcoords="offset points", ha="center", fontsize=9,
                 color=C["dapo_sr"], fontweight="bold")
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2b.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=9)

    plt.tight_layout()
    fig.savefig(PLOTS / "plot1_prior_work_comparison.png")
    plt.close(fig)


# ── Plot 2 — MOVR Reward Ablation ─────────────────────────────────────────
def plot_2_movr_ablation():
    movr_configs = ["acc_only", "sharpe_only", "mdd_only", "balanced", "sentiment_heavy"]
    config_labels = {
        "acc_only": "Acc Only", "sharpe_only": "Sharpe Only",
        "mdd_only": "MDD Only", "balanced": "Balanced ★",
        "sentiment_heavy": "Senti-Heavy",
    }
    dapo_ref = load_metrics("dapo_sr_2019")

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("MOVR Reward Ablation Study", fontsize=14, fontweight="bold")
    ax_curve, ax_sharpe, ax_mdd = axes

    # Left: cumulative return curves
    dapo_ref_cr = (dapo_ref.get("annualised_return") or 0.337)
    days = pd.date_range("2019-01-01", "2023-12-31", freq="B")
    dapo_curve = [(1 + dapo_ref_cr) ** (i / 252) - 1 for i in range(len(days))]
    ax_curve.plot(days, [v * 100 for v in dapo_curve], color=C["dapo_sr"],
                  linestyle="--", lw=1.5, label="DAPO-SR (paper, est.)")
    ax_curve.plot([days[0], days[-1]], [15.45, 154.5], color=C["nasdaq_100"],
                  linestyle=":", lw=1.2, label="NASDAQ-100")

    for cfg_name in movr_configs:
        bt = load_backtest(cfg_name)
        color = C.get(f"movr_{cfg_name}", "#888888")
        lw = 2.5 if cfg_name == "balanced" else 1.5
        if bt is not None and "cumulative_return" in bt.columns:
            ax_curve.plot(bt["date"], bt["cumulative_return"] * 100, color=color,
                          lw=lw, label=config_labels[cfg_name])
            last_val = bt["cumulative_return"].iloc[-1] * 100
            ax_curve.annotate(f"{last_val:.0f}%", xy=(bt["date"].iloc[-1], last_val),
                              xytext=(5, 0), textcoords="offset points", fontsize=8, color=color)

    ax_curve.set_ylabel("Cumulative Return (%)")
    ax_curve.set_title("Cumulative Return by MOVR Config (2019–2023)")
    ax_curve.legend(fontsize=8)
    ax_curve.xaxis.set_major_locator(plt.YearLocator())

    # Middle: Sharpe ratio
    sharpes = []
    config_names_plot = []
    for cfg_name in movr_configs:
        m = load_metrics(cfg_name)
        sr = m.get("sharpe_ratio", np.nan)
        sharpes.append(sr if not np.isnan(sr) else 0.0)
        config_names_plot.append(config_labels[cfg_name])
    bar_colors = [C.get(f"movr_{n}", "#888") for n in movr_configs]
    bars = ax_sharpe.bar(config_names_plot, sharpes, color=bar_colors, edgecolor="white")
    for i, (bar, val) in enumerate(zip(bars, sharpes)):
        if val > 0:
            ax_sharpe.annotate(f"{val:.3f}", xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                               xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9)
    dapo_sr_val = dapo_ref.get("sharpe_ratio")
    if dapo_sr_val and not np.isnan(dapo_sr_val):
        ax_sharpe.axhline(dapo_sr_val, color=C["dapo_sr"], linestyle="--", lw=1.5, label="DAPO-SR paper")
        ax_sharpe.legend(fontsize=9)
    ax_sharpe.set_title("Sharpe Ratio"); ax_sharpe.set_ylabel("Sharpe Ratio")
    ax_sharpe.tick_params(axis='x', rotation=15)

    # Right: Max Drawdown
    mdds = []
    for cfg_name in movr_configs:
        m = load_metrics(cfg_name)
        mdd = m.get("max_drawdown", np.nan)
        mdds.append((mdd or 0) * 100)
    bars_mdd = ax_mdd.bar(config_names_plot, mdds, color=bar_colors, edgecolor="white")
    for bar, val in zip(bars_mdd, mdds):
        ax_mdd.annotate(f"{val:.1f}%", xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                        xytext=(0, -15), textcoords="offset points", ha="center", fontsize=9)
    dapo_mdd = (dapo_ref.get("max_drawdown") or -0.5024) * 100
    ax_mdd.axhline(dapo_mdd, color=C["dapo_sr"], linestyle="--", lw=1.5, label="DAPO-SR paper")
    ax_mdd.annotate("Lower absolute = better", xy=(0.02, 0.05), xycoords="axes fraction",
                    fontsize=8, color="gray")
    ax_mdd.set_title("Max Drawdown (%)"); ax_mdd.set_ylabel("Max Drawdown (%)")
    ax_mdd.legend(fontsize=9); ax_mdd.tick_params(axis='x', rotation=15)

    plt.tight_layout()
    fig.savefig(PLOTS / "plot2_movr_ablation.png")
    plt.close(fig)


# ── Plot 3 — vLLM Benchmark ───────────────────────────────────────────────
def plot_3_vllm_benchmark():
    bench_path = RESULTS / "vllm_benchmark.json"
    if not bench_path.exists():
        print("  WARNING: vllm_benchmark.json not found — skipping Plot 3")
        return
    with open(bench_path) as f:
        data = json.load(f)

    batch_sizes = data["batch_sizes"]
    hf_tps  = data["hf"]["throughput_tok_per_sec"]
    vl_tps  = data["vllm"]["throughput_tok_per_sec"]
    speedup = data["speedup_ratio"]
    peak    = data["peak_speedup"]
    mean_sp = data["mean_speedup"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("vLLM vs HuggingFace Throughput Benchmark", fontsize=14, fontweight="bold")

    ax1.plot(batch_sizes, hf_tps, color=C["cppo_deepseek"], marker="o", lw=2,
             label="HuggingFace generate()")
    ax1.plot(batch_sizes, vl_tps, color=C["movr_balanced"], marker="s", lw=2,
             label="vLLM (PagedAttention)")
    ax1.set_yscale("log"); ax1.set_xlabel("Batch Size"); ax1.set_ylabel("Tokens / second")
    ax1.set_title("Throughput: HuggingFace vs vLLM")
    ax1.legend(); ax1.set_xticks(batch_sizes)
    peak_idx = np.argmax(vl_tps)
    ax1.annotate(f"Peak: {vl_tps[peak_idx]:.0f} tok/s",
                 xy=(batch_sizes[peak_idx], vl_tps[peak_idx]),
                 xytext=(0, 12), textcoords="offset points", ha="center",
                 color=C["movr_balanced"], fontsize=9, fontweight="bold")

    norm = mcolors.Normalize(vmin=min(speedup), vmax=max(speedup))
    cmap = plt.cm.Greens
    bar_colors = [cmap(norm(s)) for s in speedup]
    bars = ax2.bar([str(b) for b in batch_sizes], speedup, color=bar_colors, edgecolor="white")
    for bar, sp in zip(bars, speedup):
        ax2.annotate(f"{sp:.1f}×", xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                     xytext=(0, 4), textcoords="offset points", ha="center", fontsize=10, fontweight="bold")
    ax2.axhline(1.0, color="gray", linestyle="--", lw=1.2, label="No speedup (1×)")
    ax2.set_xlabel("Batch Size"); ax2.set_ylabel("Speedup ratio (vLLM / HF)")
    ax2.set_title(f"vLLM Speedup — Peak: {peak:.1f}×  Mean: {mean_sp:.1f}×")
    ax2.legend()

    plt.tight_layout()
    fig.savefig(PLOTS / "plot3_vllm_benchmark.png")
    plt.close(fig)


# ── Plot 4 — Full comparison table ────────────────────────────────────────
def plot_4_comparison_table():
    movr_configs = ["acc_only", "sharpe_only", "mdd_only", "balanced", "sentiment_heavy"]
    all_keys = [
        ("nasdaq_100",   "NASDAQ-100 (Buy & Hold) †"),
        ("cppo_deepseek","CPPO-DeepSeek †"),
        ("dapo_sr_2019", "DAPO-SR (paper, 2019-2023) †"),
        ("grpo_vanilla",  "GRPO Vanilla"),
        ("llm_call",      "LLM Call (zero-shot)"),
    ] + [(k, f"DAPO+MOVR {k}" + (" ★" if k == "balanced" else "")) for k in movr_configs]

    cols = ["Cumul. Return (%)", "Sharpe", "Max DD (%)", "Calmar", "Rachev", "Info. Ratio", "Train (hrs)"]
    rows = []
    row_labels = []
    for key, label in all_keys:
        m = load_metrics(key)
        cr  = round((m.get("cumulative_return") or np.nan) * 100, 1)
        sr  = round(m.get("sharpe_ratio") or np.nan, 3)
        mdd = round((m.get("max_drawdown") or np.nan) * 100, 1)
        cal = round(m.get("calmar_ratio") or np.nan, 3)
        rac = round(m.get("rachev_ratio") or np.nan, 3)
        ir  = round(m.get("information_ratio") or np.nan, 3)
        t   = round(m.get("training_time_hrs") or np.nan, 2)
        rows.append([cr, sr, mdd, cal, rac, ir, t])
        row_labels.append(label)

    df_table = pd.DataFrame(rows, index=row_labels, columns=cols)

    fig, ax = plt.subplots(figsize=(18, 7))
    ax.axis("off")
    tbl = ax.table(
        cellText=df_table.applymap(lambda v: "—" if (isinstance(v, float) and np.isnan(v)) else str(v)).values,
        rowLabels=row_labels,
        colLabels=cols,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.2, 1.6)

    # Colour cells: RdYlGn per column
    invert_cols = {2}  # Max DD: lower absolute = better
    for col_idx, col in enumerate(cols):
        col_vals = [rows[r][col_idx] for r in range(len(rows))]
        valid = [v for v in col_vals if not np.isnan(v)]
        if not valid:
            continue
        vmin, vmax = min(valid), max(valid)
        for row_idx in range(len(rows)):
            val = col_vals[row_idx]
            cell = tbl[row_idx + 1, col_idx]
            if np.isnan(val):
                cell.set_facecolor("#f0f0f0")
            else:
                norm_val = (val - vmin) / (vmax - vmin + 1e-8)
                norm_val = 1 - norm_val if col_idx in invert_cols else norm_val
                cell.set_facecolor(plt.cm.RdYlGn(norm_val * 0.7 + 0.15))

    # Bold the balanced row
    balanced_idx = next((i for i, (k, _) in enumerate(all_keys) if k == "balanced"), None)
    if balanced_idx is not None:
        for c in range(len(cols)):
            tbl[balanced_idx + 1, c].set_text_props(fontweight="bold")

    ax.set_title("Full Method Comparison — NASDAQ-100, 2019–2023", fontsize=13, fontweight="bold", pad=20)
    ax.text(0.5, -0.02, "† Results from Zha & Liu, arXiv:2505.06408, Tables 1 & 2",
            transform=ax.transAxes, ha="center", fontsize=8, color="gray")

    fig.savefig(PLOTS / "plot4_comparison_table.png", bbox_inches="tight")
    plt.close(fig)


# ── Plot 5 — All methods cumulative return ─────────────────────────────────
def plot_5_all_returns():
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.set_title("Cumulative Return — All Methods (2019–2023)")

    def estimated_curve(ann_return, label, color, lw=1.5, ls="-"):
        days = pd.date_range("2019-01-01", "2023-12-31", freq="B")
        curve = [(1 + ann_return) ** (i / 252) - 1 for i in range(len(days))]
        ax.plot(days, [v * 100 for v in curve], color=color, lw=lw, ls=ls, label=label)
        ax.annotate(f"{curve[-1]*100:.0f}%", xy=(days[-1], curve[-1]*100),
                    xytext=(5, 0), textcoords="offset points", fontsize=8, color=color)

    # Published: estimated from annualised return
    dapo_m = load_metrics("dapo_sr_2019")
    cppo_m = load_metrics("cppo_deepseek")
    nasdaq_m = load_metrics("nasdaq_100")
    if dapo_m.get("annualised_return") and not np.isnan(dapo_m["annualised_return"]):
        estimated_curve(dapo_m["annualised_return"], "DAPO-SR (est.)", C["dapo_sr"], lw=1.8, ls="--")
    if cppo_m.get("annualised_return") and not np.isnan(cppo_m["annualised_return"]):
        estimated_curve(cppo_m["annualised_return"], "CPPO-DeepSeek (est.)", C["cppo_deepseek"], lw=1.2, ls="--")
    if nasdaq_m.get("annualised_return") and not np.isnan(nasdaq_m["annualised_return"]):
        estimated_curve(nasdaq_m["annualised_return"], "NASDAQ-100", C["nasdaq_100"], lw=1.2, ls=":")

    # Our trained methods: from backtest CSV
    all_trained = [("grpo_vanilla", "GRPO Vanilla", C["grpo_vanilla"]),
                   ("llm_call", "LLM Call", C["llm_call"]),
                   ("acc_only", "MOVR acc_only", C["movr_acc_only"]),
                   ("sharpe_only", "MOVR sharpe_only", C["movr_sharpe_only"]),
                   ("mdd_only", "MOVR mdd_only", C["movr_mdd_only"]),
                   ("balanced", "MOVR balanced ★", C["movr_balanced"]),
                   ("sentiment_heavy", "MOVR senti-heavy", C["movr_sentiment_heavy"])]
    for key, label, color in all_trained:
        bt = load_backtest(key)
        lw = 2.5 if key == "balanced" else 1.5
        if bt is not None and "cumulative_return" in bt.columns:
            ax.plot(bt["date"], bt["cumulative_return"] * 100, color=color, lw=lw, label=label)
            last_val = bt["cumulative_return"].iloc[-1] * 100
            ax.annotate(f"{last_val:.0f}%", xy=(bt["date"].iloc[-1], last_val),
                        xytext=(5, 0), textcoords="offset points", fontsize=8, color=color)

    ax.set_ylabel("Cumulative Return (%)")
    ax.set_xlabel("Date")
    ax.legend(loc="upper left", ncol=2, fontsize=9)
    ax.xaxis.set_major_locator(plt.YearLocator())
    plt.tight_layout()
    fig.savefig(PLOTS / "plot5_all_returns.png")
    plt.close(fig)


# ── Plot 6 — Radar chart ───────────────────────────────────────────────────
def plot_6_radar():
    categories = ["Cumul. Return", "Sharpe Ratio", "Inverted MDD", "Calmar Ratio", "Win Rate"]
    all_methods = {
        "NASDAQ-100":        ("nasdaq_100",   C["nasdaq_100"]),
        "DAPO-SR (paper)":   ("dapo_sr_2019", C["dapo_sr"]),
        "GRPO Vanilla":      ("grpo_vanilla",  C["grpo_vanilla"]),
        "MOVR balanced ★":   ("balanced",      C["movr_balanced"]),
        "MOVR sharpe_only":  ("sharpe_only",   C["movr_sharpe_only"]),
    }

    data = {}
    for label, (key, _) in all_methods.items():
        m = load_metrics(key)
        cr  = m.get("cumulative_return", np.nan)
        sr  = m.get("sharpe_ratio", np.nan)
        mdd = m.get("max_drawdown", np.nan)
        cal = m.get("calmar_ratio", np.nan)
        wr  = m.get("win_rate", np.nan)
        vals = [cr, sr, 1 + mdd if not np.isnan(mdd) else np.nan, cal, wr]
        if any(np.isnan(v) for v in vals):
            continue
        data[label] = vals

    if len(data) < 2:
        print("  WARNING: Not enough data for radar chart — skipping Plot 6")
        return

    # Normalise per category
    for cat_idx in range(len(categories)):
        col_vals = [data[k][cat_idx] for k in data]
        vmin, vmax = min(col_vals), max(col_vals)
        for k in data:
            data[k][cat_idx] = (data[k][cat_idx] - vmin) / (vmax - vmin + 1e-8)

    num_vars = len(categories)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"polar": True})
    for label, (_, color) in all_methods.items():
        if label not in data:
            continue
        values = data[label] + data[label][:1]
        lw = 2.5 if "balanced" in label else 1.5
        alpha = 0.2 if "balanced" in label else 0.0
        ax.plot(angles, values, color=color, lw=lw, label=label)
        if alpha > 0:
            ax.fill(angles, values, color=color, alpha=alpha)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=10)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=9)
    ax.set_title("Risk-Return Radar (normalised)", fontsize=13, fontweight="bold", pad=20)
    plt.tight_layout()
    fig.savefig(PLOTS / "plot6_radar.png", bbox_inches="tight")
    plt.close(fig)


# ── Plot 7 — Training time vs performance scatter ──────────────────────────
def plot_7_time_vs_perf():
    all_keys = [
        ("nasdaq_100", "NASDAQ-100", C["nasdaq_100"]),
        ("cppo_deepseek", "CPPO-DeepSeek", C["cppo_deepseek"]),
        ("dapo_sr_2019", "DAPO-SR", C["dapo_sr"]),
        ("grpo_vanilla", "GRPO Vanilla", C["grpo_vanilla"]),
        ("acc_only", "MOVR acc_only", C["movr_acc_only"]),
        ("sharpe_only", "MOVR sharpe_only", C["movr_sharpe_only"]),
        ("mdd_only", "MOVR mdd_only", C["movr_mdd_only"]),
        ("balanced", "MOVR balanced ★", C["movr_balanced"]),
        ("sentiment_heavy", "MOVR senti-hvy", C["movr_sentiment_heavy"]),
    ]

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_title("Training Time vs Cumulative Return (marker size ∝ Sharpe)")

    points = []
    for key, label, color in all_keys:
        m = load_metrics(key)
        t = m.get("training_time_hrs", np.nan)
        cr = (m.get("cumulative_return") or np.nan) * 100
        sr = m.get("sharpe_ratio") or 0.5
        if np.isnan(t) or np.isnan(cr):
            continue
        size = max(40, (sr or 0.5) * 80)
        lw = 2.5 if "balanced" in label else 1.0
        ax.scatter(t, cr, color=color, s=size, zorder=5, edgecolors="white", linewidths=lw)
        ax.annotate(label, xy=(t, cr), xytext=(8, 4), textcoords="offset points", fontsize=8, color=color)
        points.append((t, cr))

    # Pareto frontier
    if len(points) > 1:
        points_sorted = sorted(points, key=lambda p: p[0])
        pareto, best_cr = [], -np.inf
        for p in points_sorted:
            if p[1] > best_cr:
                pareto.append(p)
                best_cr = p[1]
        if len(pareto) > 1:
            px, py = zip(*pareto)
            ax.plot(px, py, "--", color="gray", lw=1.5, alpha=0.6, label="Pareto frontier")

    ax.axvspan(0, min(t for t, _ in points) if points else 0, alpha=0.06, color="green",
               label="Ideal region (fast + high return)")
    ax.set_xlabel("Training Time (hours)")
    ax.set_ylabel("Cumulative Return (%)")
    ax.legend(fontsize=9)
    plt.tight_layout()
    fig.savefig(PLOTS / "plot7_time_vs_perf.png")
    plt.close(fig)


# ── Plot 8 — Training time bar chart ──────────────────────────────────────
def plot_8_training_time():
    method_data = [
        ("CPPO-DeepSeek †",       7.5,  C["cppo_deepseek"], True),
        ("GRPO Vanilla",          None, C["grpo_vanilla"],  False),
        ("MOVR acc_only",         None, C["movr_acc_only"], False),
        ("MOVR sharpe_only",      None, C["movr_sharpe_only"], False),
        ("MOVR mdd_only",         None, C["movr_mdd_only"], False),
        ("MOVR sentiment_heavy",  None, C["movr_sentiment_heavy"], False),
        ("MOVR balanced ★",       None, C["movr_balanced"], False),
        ("DAPO-SR (paper) †",     2.5,  C["dapo_sr"],      True),
        ("LLM Call (zero-shot)",  0.0,  C["llm_call"],     False),
    ]

    # Try to fill None from saved metrics
    keys_map = {
        "GRPO Vanilla":       "grpo_vanilla",
        "MOVR acc_only":      "acc_only",
        "MOVR sharpe_only":   "sharpe_only",
        "MOVR mdd_only":      "mdd_only",
        "MOVR sentiment_heavy": "sentiment_heavy",
        "MOVR balanced ★":    "balanced",
        "LLM Call (zero-shot)": "llm_call",
    }
    filled = []
    for label, t, color, published in method_data:
        if t is None:
            m = load_metrics(keys_map.get(label, ""))
            t = m.get("training_time_hrs", None)
        filled.append((label, t, color, published))

    labels = [x[0] for x in filled]
    times  = [x[1] for x in filled]
    colors = [x[2] for x in filled]
    hatches = ["//" if x[3] else "" for x in filled]

    fig, ax = plt.subplots(figsize=(12, 7))
    y_pos = np.arange(len(labels))

    for i, (y, t, c, h) in enumerate(zip(y_pos, times, colors, hatches)):
        if t is None or np.isnan(t):
            ax.barh(y, 0.1, color="#e0e0e0", edgecolor="white", hatch=h)
            ax.text(0.2, y, "not yet run", va="center", fontsize=8, color="gray")
        else:
            ax.barh(y, t, color=c, edgecolor="white", hatch=h, alpha=0.85)
            ax.text(t + 0.1, y, f"{t:.2f}h", va="center", fontsize=9, fontweight="bold")

    ax.axvline(7.5, color="#999", linestyle="--", lw=1.5, label="CPPO baseline (7.5 hrs)")
    ax.axvline(2.5, color=C["dapo_sr"], linestyle="--", lw=1.5, label="DAPO-SR paper (2.5 hrs)")
    ax.set_yticks(y_pos); ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("Training Time (hours)")
    ax.set_title("Training Time per Method (100 epochs, A10G GPU)", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.text(0.5, -0.06, "† Published values from arXiv:2505.06408, Table 3",
            transform=ax.transAxes, ha="center", fontsize=8, color="gray")
    plt.tight_layout()
    fig.savefig(PLOTS / "plot8_training_time.png")
    plt.close(fig)


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    print("Generating plots from results/ ...\n")
    fns = [
        ("Plot 1: Prior work comparison",   plot_1_prior_work_comparison),
        ("Plot 2: MOVR ablation",           plot_2_movr_ablation),
        ("Plot 3: vLLM benchmark",          plot_3_vllm_benchmark),
        ("Plot 4: Comparison table",        plot_4_comparison_table),
        ("Plot 5: All returns",             plot_5_all_returns),
        ("Plot 6: Radar chart",             plot_6_radar),
        ("Plot 7: Time vs performance",     plot_7_time_vs_perf),
        ("Plot 8: Training time",           plot_8_training_time),
    ]
    for name, fn in fns:
        try:
            fn()
            print(f"  ✓ {name}")
        except Exception as e:
            print(f"  ✗ {name} — {e}")

    print(f"\nDone. Plots saved to ./plots/")
    print("Note: plots with missing data (nan/None) render '—' and skip curves.")
    print("After script 01, plots 1, 4, 7, 8 render with published data immediately.")


if __name__ == "__main__":
    main()
