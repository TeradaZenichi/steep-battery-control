from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_TEST_ROOT = PROJECT_ROOT / "Results" / "test"
OUTPUT_ROOT = PROJECT_ROOT / "Results" / "statistical_tests"

MODELS = ("ATT", "ATT_MEM", "GRU", "MLP", "TCN")
APPROACHES = {
    "IL": "1-IL",
    "RL": "2-RL",
}
PREFERRED_TARIFF_ORDER = ["tar_flat", "tar_s", "tar_sw", "tar_tou", "tar_w"]
BOOTSTRAP_SAMPLES = 10000
RNG_SEED = 42


def discover_tariffs() -> list[str]:
    discovered: set[str] = set()
    for model in MODELS:
        for approach_dir in APPROACHES.values():
            base = RESULTS_TEST_ROOT / model / approach_dir
            if not base.exists():
                continue
            for tariff_dir in base.iterdir():
                if tariff_dir.is_dir():
                    discovered.add(tariff_dir.name)

    ordered = [t for t in PREFERRED_TARIFF_ORDER if t in discovered]
    ordered.extend(sorted(discovered.difference(ordered)))
    return ordered


def load_summary(path: Path) -> dict[str, float]:
    data = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, float] = {}
    for run_name, rec in data.items():
        if isinstance(rec, dict) and "actor_reward" in rec:
            out[run_name] = float(rec["actor_reward"])
    return out


def bootstrap_ci_mean(values: np.ndarray, n_boot: int, rng: np.random.Generator) -> tuple[float, float]:
    n = len(values)
    if n == 0:
        return np.nan, np.nan
    idx = rng.integers(0, n, size=(n_boot, n))
    means = values[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def wilcoxon_pvalue(values: np.ndarray) -> float:
    nz = values[np.abs(values) > 1e-12]
    if len(nz) == 0:
        return 1.0
    stat, p = wilcoxon(nz, alternative="two-sided", zero_method="wilcox", mode="auto")
    return float(p)


def summarize_pair(diffs: np.ndarray, il_values: np.ndarray, rl_values: np.ndarray, n_boot: int, rng: np.random.Generator) -> dict[str, float | int | str]:
    ci_low, ci_high = bootstrap_ci_mean(diffs, n_boot=n_boot, rng=rng)
    p_value = wilcoxon_pvalue(diffs)

    return {
        "n_pairs": int(len(diffs)),
        "il_mean_reward": float(il_values.mean()),
        "rl_mean_reward": float(rl_values.mean()),
        "diff_mean_rl_minus_il": float(diffs.mean()),
        "diff_median_rl_minus_il": float(np.median(diffs)),
        "diff_std_rl_minus_il": float(diffs.std(ddof=0)),
        "ci95_low_diff_mean": ci_low,
        "ci95_high_diff_mean": ci_high,
        "wilcoxon_pvalue_two_sided": p_value,
        "significant_0_05": int(p_value < 0.05),
        "wins_rl": int((diffs > 0).sum()),
        "wins_il": int((diffs < 0).sum()),
        "ties": int((np.abs(diffs) <= 1e-12).sum()),
    }


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    tariffs = discover_tariffs()
    rng = np.random.default_rng(RNG_SEED)

    rows_model_tariff: list[dict[str, float | int | str]] = []
    all_pairs_rows: list[dict[str, float | int | str]] = []

    for model in MODELS:
        for tariff in tariffs:
            il_path = RESULTS_TEST_ROOT / model / APPROACHES["IL"] / tariff / "summary.json"
            rl_path = RESULTS_TEST_ROOT / model / APPROACHES["RL"] / tariff / "summary.json"

            if not il_path.exists() or not rl_path.exists():
                continue

            il_map = load_summary(il_path)
            rl_map = load_summary(rl_path)
            common_runs = sorted(set(il_map).intersection(rl_map))
            if not common_runs:
                continue

            il_values = np.array([il_map[r] for r in common_runs], dtype=float)
            rl_values = np.array([rl_map[r] for r in common_runs], dtype=float)
            diffs = rl_values - il_values

            summary = summarize_pair(diffs, il_values, rl_values, n_boot=BOOTSTRAP_SAMPLES, rng=rng)
            row = {
                "model": model,
                "tariff": tariff,
                **summary,
            }
            rows_model_tariff.append(row)

            for run_name, il_v, rl_v, diff_v in zip(common_runs, il_values, rl_values, diffs):
                all_pairs_rows.append(
                    {
                        "model": model,
                        "tariff": tariff,
                        "run_name": run_name,
                        "il_reward": float(il_v),
                        "rl_reward": float(rl_v),
                        "diff_rl_minus_il": float(diff_v),
                    }
                )

    model_tariff_df = pd.DataFrame(rows_model_tariff)
    if not model_tariff_df.empty:
        model_tariff_df["tariff"] = pd.Categorical(model_tariff_df["tariff"], categories=tariffs, ordered=True)
        model_tariff_df = model_tariff_df.sort_values(["tariff", "model"]).reset_index(drop=True)

    pairs_df = pd.DataFrame(all_pairs_rows)
    if not pairs_df.empty:
        pairs_df["tariff"] = pd.Categorical(pairs_df["tariff"], categories=tariffs, ordered=True)
        pairs_df = pairs_df.sort_values(["tariff", "model", "run_name"]).reset_index(drop=True)

    by_tariff_rows: list[dict[str, float | int | str]] = []
    if not pairs_df.empty:
        for tariff, g in pairs_df.groupby("tariff", observed=True):
            diffs = g["diff_rl_minus_il"].to_numpy(dtype=float)
            il_values = g["il_reward"].to_numpy(dtype=float)
            rl_values = g["rl_reward"].to_numpy(dtype=float)
            summary = summarize_pair(diffs, il_values, rl_values, n_boot=BOOTSTRAP_SAMPLES, rng=rng)
            by_tariff_rows.append({"tariff": str(tariff), **summary})

    by_tariff_df = pd.DataFrame(by_tariff_rows)
    if not by_tariff_df.empty:
        by_tariff_df["tariff"] = pd.Categorical(by_tariff_df["tariff"], categories=tariffs, ordered=True)
        by_tariff_df = by_tariff_df.sort_values("tariff").reset_index(drop=True)

    overall_df = pd.DataFrame()
    if not pairs_df.empty:
        diffs = pairs_df["diff_rl_minus_il"].to_numpy(dtype=float)
        il_values = pairs_df["il_reward"].to_numpy(dtype=float)
        rl_values = pairs_df["rl_reward"].to_numpy(dtype=float)
        summary = summarize_pair(diffs, il_values, rl_values, n_boot=BOOTSTRAP_SAMPLES, rng=rng)
        overall_df = pd.DataFrame([{"scope": "all_models_all_tariffs", **summary}])

    out_pairs = OUTPUT_ROOT / "paired_rewards_rl_vs_il.csv"
    out_model_tariff = OUTPUT_ROOT / "bootstrap_wilcoxon_rl_vs_il_by_model_tariff.csv"
    out_tariff = OUTPUT_ROOT / "bootstrap_wilcoxon_rl_vs_il_by_tariff.csv"
    out_overall = OUTPUT_ROOT / "bootstrap_wilcoxon_rl_vs_il_overall.csv"

    pairs_df.to_csv(out_pairs, index=False, encoding="utf-8")
    model_tariff_df.to_csv(out_model_tariff, index=False, encoding="utf-8")
    by_tariff_df.to_csv(out_tariff, index=False, encoding="utf-8")
    overall_df.to_csv(out_overall, index=False, encoding="utf-8")

    print(f"Saved: {out_pairs}")
    print(f"Saved: {out_model_tariff}")
    print(f"Saved: {out_tariff}")
    print(f"Saved: {out_overall}")


if __name__ == "__main__":
    main()
