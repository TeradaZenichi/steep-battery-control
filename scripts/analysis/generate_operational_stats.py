from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import font_manager


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_TEST_ROOT = PROJECT_ROOT / "Results" / "test"
ANALYSIS_ROOT = PROJECT_ROOT / "Results" / "analysis" / "new"
FIGURES_ROOT = PROJECT_ROOT / "Results" / "figures" / "analysis" / "operational_stats"

MODELS = ("ATT", "ATT_MEM", "GRU", "MLP", "TCN")
APPROACH_MAP = {
    "1-IL": "IL",
    "2-RL": "RL",
}
PREFERRED_TARIFF_ORDER = ["tar_flat", "tar_s", "tar_sw", "tar_tou", "tar_w"]

EPS_POWER = 1e-6
EPS_COST = 1e-9
PLOT_EXT = "pdf"
DPI = 220


def setup_gulliver_font() -> None:
    sns.set_theme(style="whitegrid")

    font_path = PROJECT_ROOT / "data" / "Gulliver.otf"
    if font_path.exists():
        font_manager.fontManager.addfont(str(font_path))
        prop = font_manager.FontProperties(fname=str(font_path))
        plt.rcParams["font.family"] = "Gulliver"
        plt.rcParams["font.sans-serif"] = prop.get_name()

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
        }
    )
    plt.rcParams["axes.unicode_minus"] = False


def build_approach_arch_label(approach: str, model: str) -> str:
    return f"{approach} {model}"


def discover_tariffs() -> list[str]:
    discovered: set[str] = set()
    for model in MODELS:
        for approach_dir in APPROACH_MAP:
            base = RESULTS_TEST_ROOT / model / approach_dir
            if not base.exists():
                continue
            for tariff_dir in base.iterdir():
                if tariff_dir.is_dir():
                    discovered.add(tariff_dir.name)

    ordered = [tariff for tariff in PREFERRED_TARIFF_ORDER if tariff in discovered]
    ordered.extend(sorted(discovered.difference(ordered)))
    return ordered


def discover_actor_files(tariffs: list[str]) -> list[dict[str, Path | str]]:
    items: list[dict[str, Path | str]] = []
    for model in MODELS:
        for approach_dir, approach_name in APPROACH_MAP.items():
            for tariff in tariffs:
                folder = RESULTS_TEST_ROOT / model / approach_dir / tariff
                if not folder.exists():
                    continue

                for op_path in sorted(folder.glob("*_actor_env_operation.csv")):
                    run_name = op_path.name.replace("_actor_env_operation.csv", "")
                    breakdown_path = folder / f"{run_name}_actor_env_operation_breakdown.csv"
                    items.append(
                        {
                            "model": model,
                            "approach_dir": approach_dir,
                            "approach": approach_name,
                            "tariff": tariff,
                            "run_name": run_name,
                            "operation_path": op_path,
                            "breakdown_path": breakdown_path,
                        }
                    )
    return items


def read_columns(csv_path: Path, columns: list[str]) -> pd.DataFrame:
    header = pd.read_csv(csv_path, nrows=0)
    available = set(header.columns)

    usecols = [c for c in columns if c in available]
    if "timestamp" not in usecols:
        usecols = ["timestamp"] + usecols

    df = pd.read_csv(csv_path, usecols=usecols, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").set_index("timestamp")
    return df


def timestep_hours(index: pd.Index) -> float:
    if len(index) < 2:
        return 5.0 / 60.0

    diffs = pd.Series(index).diff().dropna().dt.total_seconds().to_numpy(dtype=float)
    diffs = diffs[diffs > 0.0]
    if diffs.size == 0:
        return 5.0 / 60.0

    return float(np.median(diffs) / 3600.0)


def compute_run_metrics(op_df: pd.DataFrame, breakdown_df: pd.DataFrame | None) -> dict[str, float | int]:
    n_steps = int(len(op_df))
    dt_h = timestep_hours(op_df.index)
    duration_h = n_steps * dt_h
    duration_days = duration_h / 24.0

    pbess = op_df.get("PBESS", pd.Series(0.0, index=op_df.index)).astype(float)
    pev = op_df.get("PEV", pd.Series(0.0, index=op_df.index)).astype(float)
    pgrid = op_df.get("PGrid", pd.Series(0.0, index=op_df.index)).astype(float)
    grid_penalty = op_df.get("grid_penalty", pd.Series(0.0, index=op_df.index)).astype(float)

    bess_active = pbess.abs() > EPS_POWER
    v2g_active = pev < -EPS_POWER
    grid_violation = grid_penalty > EPS_COST

    fast_events = np.nan
    fast_cost_total = np.nan
    fast_data_available = 0
    if breakdown_df is not None and "ev_arrival_fast_cost" in breakdown_df.columns:
        fast_data_available = 1
        aligned = breakdown_df.reindex(op_df.index)
        fast_series = aligned["ev_arrival_fast_cost"].fillna(0.0).astype(float)
        fast_events = float((fast_series > EPS_COST).sum())
        fast_cost_total = float(fast_series.clip(lower=0.0).sum())

    metrics = {
        "n_steps": n_steps,
        "dt_hours": dt_h,
        "duration_hours": duration_h,
        "duration_days": duration_days,
        "bess_usage_rate_pct": float(bess_active.mean() * 100.0),
        "bess_throughput_kwh": float((pbess.abs() * dt_h).sum()),
        "v2g_usage_rate_pct": float(v2g_active.mean() * 100.0),
        "v2g_energy_kwh": float(((-pev).clip(lower=0.0) * dt_h).sum()),
        "fast_charging_events": float(fast_events),
        "fast_charging_events_per_day": float(fast_events / max(duration_days, 1e-9)) if fast_data_available else np.nan,
        "fast_charging_cost_total": float(fast_cost_total),
        "fast_data_available": int(fast_data_available),
        "grid_violation_steps": int(grid_violation.sum()),
        "grid_violation_rate_pct": float(grid_violation.mean() * 100.0),
        "grid_penalty_total": float(grid_penalty.sum()),
        "pgrid_mean_kw": float(pgrid.mean()),
        "pgrid_std_kw": float(pgrid.std(ddof=0)),
        "pgrid_min_kw": float(pgrid.min()),
        "pgrid_p05_kw": float(pgrid.quantile(0.05)),
        "pgrid_p25_kw": float(pgrid.quantile(0.25)),
        "pgrid_p50_kw": float(pgrid.quantile(0.50)),
        "pgrid_p75_kw": float(pgrid.quantile(0.75)),
        "pgrid_p95_kw": float(pgrid.quantile(0.95)),
        "pgrid_max_kw": float(pgrid.max()),
    }
    return metrics


def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [
        col if isinstance(col, str) else (col[0] if col[1] == "" else f"{col[0]}_{col[1]}")
        for col in df.columns.to_flat_index()
    ]
    return df


def make_grouped_bar_plot(
    summary_df: pd.DataFrame,
    metric_base: str,
    ylabel: str,
    out_path: Path,
    tariff_order: list[str],
    approach_arch_order: list[str],
) -> None:
    metric_col = f"{metric_base}_mean"
    pivot = summary_df.pivot(index="tariff", columns="approach_arch", values=metric_col)
    pivot = pivot.reindex(tariff_order)

    ordered_cols = [c for c in approach_arch_order if c in pivot.columns]
    pivot = pivot[ordered_cols]

    fig, ax = plt.subplots(figsize=(14, 6))
    pivot.plot(kind="bar", ax=ax, width=0.85)
    ax.set_xlabel("Tariff")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(title="Approach Architecture", ncol=2, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def make_grid_power_boxplots(
    grid_samples: dict[tuple[str, str], list[np.ndarray]],
    tariff_order: list[str],
    approach_arch_order: list[str],
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    for tariff in tariff_order:
        labels = [label for label in approach_arch_order if (tariff, label) in grid_samples]
        if not labels:
            continue

        plot_rows: list[pd.DataFrame] = []
        for label in labels:
            values = np.concatenate(grid_samples[(tariff, label)]).astype(float)
            if values.size > 50000:
                idx = np.linspace(0, values.size - 1, num=50000, dtype=int)
                values = values[idx]
            plot_rows.append(
                pd.DataFrame(
                    {
                        "approach_arch": label,
                        "PGrid": values,
                    }
                )
            )

        plot_df = pd.concat(plot_rows, ignore_index=True)
        palette = sns.color_palette("tab10", n_colors=len(labels))

        fig, ax = plt.subplots(figsize=(14, 6))
        sns.boxplot(
            data=plot_df,
            x="approach_arch",
            y="PGrid",
            order=labels,
            hue="approach_arch",
            hue_order=labels,
            palette=palette,
            dodge=False,
            legend=False,
            showfliers=False,
            linewidth=1.1,
            ax=ax,
        )
        if ax.legend_ is not None:
            ax.legend_.remove()
        sns.despine(ax=ax)
        ax.axhline(0.0, color="black", linewidth=1.0)
        ax.set_title(f"Grid power distribution ({tariff})")
        ax.set_ylabel("PGrid (kW)")
        ax.set_xlabel("Approach Architecture")
        ax.grid(axis="y", alpha=0.3)
        ax.tick_params(axis="x", rotation=25)
        fig.tight_layout()
        fig.savefig(out_dir / f"grid_power_boxplot_{tariff}.{PLOT_EXT}", dpi=DPI, bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    ANALYSIS_ROOT.mkdir(parents=True, exist_ok=True)
    FIGURES_ROOT.mkdir(parents=True, exist_ok=True)

    tariffs = discover_tariffs()
    actor_items = discover_actor_files(tariffs)
    if not actor_items:
        raise RuntimeError("No actor operation files were found in Results/test.")

    records: list[dict[str, float | int | str]] = []
    grid_samples: dict[tuple[str, str], list[np.ndarray]] = defaultdict(list)

    for item in actor_items:
        op_path = Path(item["operation_path"])
        breakdown_path = Path(item["breakdown_path"])

        op_df = read_columns(op_path, ["timestamp", "PBESS", "PEV", "PGrid", "grid_penalty"])
        breakdown_df = None
        if breakdown_path.exists():
            breakdown_df = read_columns(breakdown_path, ["timestamp", "ev_arrival_fast_cost"])

        run_metrics = compute_run_metrics(op_df, breakdown_df)
        approach_arch = build_approach_arch_label(str(item["approach"]), str(item["model"]))

        row = {
            "model": item["model"],
            "approach": item["approach"],
            "approach_dir": item["approach_dir"],
            "approach_arch": approach_arch,
            "tariff": item["tariff"],
            "run_name": item["run_name"],
        }
        row.update(run_metrics)
        records.append(row)

        pgrid_values = op_df.get("PGrid", pd.Series(0.0, index=op_df.index)).astype(np.float32).to_numpy(copy=True)
        grid_samples[(str(item["tariff"]), approach_arch)].append(pgrid_values)

    per_run_df = pd.DataFrame(records)
    per_run_df["tariff"] = pd.Categorical(per_run_df["tariff"], categories=tariffs, ordered=True)
    per_run_df = per_run_df.sort_values(["tariff", "approach", "model", "run_name"]).reset_index(drop=True)

    per_run_csv = ANALYSIS_ROOT / "operational_metrics_per_run.csv"
    per_run_df.to_csv(per_run_csv, index=False, encoding="utf-8")

    metric_cols = [
        "bess_usage_rate_pct",
        "bess_throughput_kwh",
        "fast_charging_events",
        "fast_charging_events_per_day",
        "fast_charging_cost_total",
        "v2g_usage_rate_pct",
        "v2g_energy_kwh",
        "grid_violation_steps",
        "grid_violation_rate_pct",
        "grid_penalty_total",
        "pgrid_mean_kw",
        "pgrid_std_kw",
        "pgrid_min_kw",
        "pgrid_p05_kw",
        "pgrid_p25_kw",
        "pgrid_p50_kw",
        "pgrid_p75_kw",
        "pgrid_p95_kw",
        "pgrid_max_kw",
    ]

    group_keys = ["tariff", "model", "approach", "approach_arch"]
    grouped = per_run_df.groupby(group_keys, observed=True)

    summary_df = grouped[metric_cols].agg(["mean", "std", "median"]).reset_index()
    summary_df = flatten_columns(summary_df)
    n_runs_df = grouped.size().reset_index(name="n_runs")
    summary_df = summary_df.merge(n_runs_df, on=group_keys, how="left")
    summary_df = summary_df.sort_values(["tariff", "approach", "model"]).reset_index(drop=True)

    summary_csv = ANALYSIS_ROOT / "operational_metrics_summary.csv"
    summary_df.to_csv(summary_csv, index=False, encoding="utf-8")

    plot_table_cols = ["tariff", "approach_arch", "n_runs"] + [f"{col}_mean" for col in metric_cols]
    plot_table_df = summary_df[plot_table_cols].copy()
    plot_table_df = plot_table_df.sort_values(["tariff", "approach_arch"]).reset_index(drop=True)
    plot_table_csv = ANALYSIS_ROOT / "operational_metrics_plot_table.csv"
    plot_table_df.to_csv(plot_table_csv, index=False, encoding="utf-8")

    setup_gulliver_font()

    approach_arch_order = [
        *[
            build_approach_arch_label("IL", m)
            for m in MODELS
            if build_approach_arch_label("IL", m) in set(summary_df["approach_arch"])
        ],
        *[
            build_approach_arch_label("RL", m)
            for m in MODELS
            if build_approach_arch_label("RL", m) in set(summary_df["approach_arch"])
        ],
    ]

    make_grouped_bar_plot(
        summary_df,
        metric_base="bess_usage_rate_pct",
        ylabel="BESS usage rate (%)",
        out_path=FIGURES_ROOT / f"bess_usage_rate_by_tariff.{PLOT_EXT}",
        tariff_order=tariffs,
        approach_arch_order=approach_arch_order,
    )
    make_grouped_bar_plot(
        summary_df,
        metric_base="fast_charging_events",
        ylabel="Fast charging events (count)",
        out_path=FIGURES_ROOT / f"fast_charging_events_by_tariff.{PLOT_EXT}",
        tariff_order=tariffs,
        approach_arch_order=approach_arch_order,
    )
    make_grouped_bar_plot(
        summary_df,
        metric_base="v2g_usage_rate_pct",
        ylabel="V2G usage rate (%)",
        out_path=FIGURES_ROOT / f"v2g_usage_rate_by_tariff.{PLOT_EXT}",
        tariff_order=tariffs,
        approach_arch_order=approach_arch_order,
    )
    make_grouped_bar_plot(
        summary_df,
        metric_base="grid_violation_rate_pct",
        ylabel="Grid violation rate (%)",
        out_path=FIGURES_ROOT / f"grid_violation_rate_by_tariff.{PLOT_EXT}",
        tariff_order=tariffs,
        approach_arch_order=approach_arch_order,
    )

    make_grid_power_boxplots(
        grid_samples=grid_samples,
        tariff_order=tariffs,
        approach_arch_order=approach_arch_order,
        out_dir=FIGURES_ROOT,
    )

    print(f"Saved: {per_run_csv}")
    print(f"Saved: {summary_csv}")
    print(f"Saved: {plot_table_csv}")
    print(f"Plots dir: {FIGURES_ROOT}")


if __name__ == "__main__":
    main()
