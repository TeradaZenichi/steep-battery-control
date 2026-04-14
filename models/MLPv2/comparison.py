from pathlib import Path
import json
import pandas as pd


# ===== User configuration =====
ALGORITHMS = ["1-IL", "2-RL"]
BASELINE_ALGO = "1-IL"
CANDIDATE_ALGO = "2-RL"
TARIFFS = ["tar_flat", "tar_s", "tar_sw", "tar_tou", "tar_w"]
SAVE_OUTPUTS = True
# =============================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = PROJECT_ROOT / "Results" / "test" / "MLP"
OUTPUT_DIR = RESULTS_ROOT / "comparison"


def load_summary(algorithm: str, tariff: str) -> dict:
    summary_path = RESULTS_ROOT / algorithm / tariff / "summary.json"
    if not summary_path.exists():
        return {}
    with open(summary_path, "r", encoding="utf-8") as file:
        return json.load(file)


def collect_rows() -> pd.DataFrame:
    rows = []

    for algorithm in ALGORITHMS:
        for tariff in TARIFFS:
            summary = load_summary(algorithm, tariff)
            if not summary:
                print(f"[WARN] Missing summary: {algorithm}/{tariff}")
                continue

            tests = sorted(key for key in summary.keys() if key.startswith("test_"))
            for test_name in tests:
                entry = summary[test_name]
                actor_breakdown = entry.get("actor_breakdown", {})

                rows.append(
                    {
                        "algorithm": algorithm,
                        "tariff": tariff,
                        "scenario": test_name,
                        "actor_reward": float(entry.get("actor_reward", 0.0)),
                        "teacher_reward": float(entry.get("teacher_reward", 0.0)),
                        "reward_gap": float(entry.get("reward_diff", entry.get("actor_reward", 0.0) - entry.get("teacher_reward", 0.0))),
                        "energy_cost": float(actor_breakdown.get("energy_cost", 0.0)),
                        "grid_penalty": float(actor_breakdown.get("grid_penalty", 0.0)),
                        "bess_cost": float(actor_breakdown.get("bess_cost", 0.0)),
                        "ev_cost": float(actor_breakdown.get("ev_cost", 0.0)),
                        "ev_arrival_fast_cost": float(actor_breakdown.get("ev_arrival_fast_cost", 0.0)),
                        "ev_soc_min_cost": float(actor_breakdown.get("ev_soc_min_cost", 0.0)),
                    }
                )

    return pd.DataFrame(rows)


def build_tariff_table(df: pd.DataFrame) -> pd.DataFrame:
    agg = (
        df.groupby(["algorithm", "tariff"], as_index=False)
        .agg(
            n_scenarios=("scenario", "count"),
            actor_reward_mean=("actor_reward", "mean"),
            teacher_reward_mean=("teacher_reward", "mean"),
            reward_gap_mean=("reward_gap", "mean"),
            ev_cost_mean=("ev_cost", "mean"),
            grid_penalty_mean=("grid_penalty", "mean"),
            ev_arrival_fast_cost_mean=("ev_arrival_fast_cost", "mean"),
            ev_soc_min_cost_mean=("ev_soc_min_cost", "mean"),
        )
    )

    base = agg[agg["algorithm"] == BASELINE_ALGO].copy()
    cand = agg[agg["algorithm"] == CANDIDATE_ALGO].copy()

    merged = base.merge(cand, on="tariff", suffixes=("_base", "_cand"))
    merged["actor_reward_delta_cand_minus_base"] = merged["actor_reward_mean_cand"] - merged["actor_reward_mean_base"]
    merged["reward_gap_delta_cand_minus_base"] = merged["reward_gap_mean_cand"] - merged["reward_gap_mean_base"]
    merged["ev_cost_delta_cand_minus_base"] = merged["ev_cost_mean_cand"] - merged["ev_cost_mean_base"]
    merged["grid_penalty_delta_cand_minus_base"] = merged["grid_penalty_mean_cand"] - merged["grid_penalty_mean_base"]

    merged = merged.sort_values("actor_reward_delta_cand_minus_base", ascending=False)
    return merged


def build_scenario_table(df: pd.DataFrame) -> pd.DataFrame:
    base = df[df["algorithm"] == BASELINE_ALGO].copy()
    cand = df[df["algorithm"] == CANDIDATE_ALGO].copy()

    merged = base.merge(
        cand,
        on=["tariff", "scenario"],
        suffixes=("_base", "_cand"),
    )

    merged["actor_reward_delta_cand_minus_base"] = merged["actor_reward_cand"] - merged["actor_reward_base"]
    merged["reward_gap_delta_cand_minus_base"] = merged["reward_gap_cand"] - merged["reward_gap_base"]
    merged["ev_cost_delta_cand_minus_base"] = merged["ev_cost_cand"] - merged["ev_cost_base"]
    merged["grid_penalty_delta_cand_minus_base"] = merged["grid_penalty_cand"] - merged["grid_penalty_base"]
    merged["candidate_better"] = (merged["actor_reward_delta_cand_minus_base"] > 0.0).astype(int)

    merged = merged.sort_values(["tariff", "scenario"])
    return merged


def build_global_table(df: pd.DataFrame, scenario_table: pd.DataFrame) -> pd.DataFrame:
    global_by_algo = (
        df.groupby("algorithm", as_index=False)
        .agg(
            n_rows=("scenario", "count"),
            actor_reward_mean=("actor_reward", "mean"),
            teacher_reward_mean=("teacher_reward", "mean"),
            reward_gap_mean=("reward_gap", "mean"),
            ev_cost_mean=("ev_cost", "mean"),
            grid_penalty_mean=("grid_penalty", "mean"),
            ev_arrival_fast_cost_mean=("ev_arrival_fast_cost", "mean"),
            ev_soc_min_cost_mean=("ev_soc_min_cost", "mean"),
        )
    )

    baseline_row = global_by_algo.loc[global_by_algo["algorithm"] == BASELINE_ALGO].iloc[0]
    candidate_row = global_by_algo.loc[global_by_algo["algorithm"] == CANDIDATE_ALGO].iloc[0]

    total = len(scenario_table)
    wins = int(scenario_table["candidate_better"].sum())
    win_rate = wins / total if total > 0 else 0.0

    summary = pd.DataFrame(
        [
            {
                "baseline_algorithm": BASELINE_ALGO,
                "candidate_algorithm": CANDIDATE_ALGO,
                "n_scenarios_compared": total,
                "candidate_wins": wins,
                "candidate_win_rate": win_rate,
                "baseline_actor_reward_mean": float(baseline_row["actor_reward_mean"]),
                "candidate_actor_reward_mean": float(candidate_row["actor_reward_mean"]),
                "actor_reward_delta_cand_minus_base": float(candidate_row["actor_reward_mean"] - baseline_row["actor_reward_mean"]),
                "baseline_reward_gap_mean": float(baseline_row["reward_gap_mean"]),
                "candidate_reward_gap_mean": float(candidate_row["reward_gap_mean"]),
                "reward_gap_delta_cand_minus_base": float(candidate_row["reward_gap_mean"] - baseline_row["reward_gap_mean"]),
                "ev_cost_delta_cand_minus_base": float(candidate_row["ev_cost_mean"] - baseline_row["ev_cost_mean"]),
                "grid_penalty_delta_cand_minus_base": float(candidate_row["grid_penalty_mean"] - baseline_row["grid_penalty_mean"]),
                "ev_arrival_fast_delta_cand_minus_base": float(candidate_row["ev_arrival_fast_cost_mean"] - baseline_row["ev_arrival_fast_cost_mean"]),
            }
        ]
    )
    return summary


def print_console_summary(tariff_table: pd.DataFrame, global_table: pd.DataFrame) -> None:
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 180)

    cols_tariff = [
        "tariff",
        "actor_reward_mean_base",
        "actor_reward_mean_cand",
        "actor_reward_delta_cand_minus_base",
        "reward_gap_mean_base",
        "reward_gap_mean_cand",
        "ev_cost_mean_base",
        "ev_cost_mean_cand",
        "grid_penalty_mean_base",
        "grid_penalty_mean_cand",
    ]

    print("\n=== Comparison by tariff ===")
    print(tariff_table[cols_tariff].to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    print("\n=== Global summary ===")
    print(global_table.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


def save_outputs(scenario_table: pd.DataFrame, tariff_table: pd.DataFrame, global_table: pd.DataFrame) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    scenario_table.to_csv(OUTPUT_DIR / "comparison_by_scenario.csv", index=False)
    tariff_table.to_csv(OUTPUT_DIR / "comparison_by_tariff.csv", index=False)
    global_table.to_csv(OUTPUT_DIR / "comparison_global.csv", index=False)

    with pd.ExcelWriter(OUTPUT_DIR / "comparison_tables.xlsx", engine="openpyxl") as writer:
        scenario_table.to_excel(writer, sheet_name="by_scenario", index=False)
        tariff_table.to_excel(writer, sheet_name="by_tariff", index=False)
        global_table.to_excel(writer, sheet_name="global", index=False)


def main() -> None:
    df = collect_rows()
    if df.empty:
        print("No data found to compare.")
        return

    scenario_table = build_scenario_table(df)
    tariff_table = build_tariff_table(df)
    global_table = build_global_table(df, scenario_table)

    print_console_summary(tariff_table, global_table)

    if SAVE_OUTPUTS:
        save_outputs(scenario_table, tariff_table, global_table)
        print(f"\nSaved outputs to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
