from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


RESULTS_ROOT = Path("Results/test")
OUTPUT_ROOT = Path("Results/analysis")

MODELS = ("ATT", "GRU", "MLP")
APPROACH_MAP = {
    "1-IL": "IL",
    "2-RL": "RL",
}
PREFERRED_TARIFF_ORDER = ["tar_flat", "tar_s", "tar_sw", "tar_tou", "tar_w"]


def to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def pick_numeric(*values: Any) -> float | None:
    for value in values:
        parsed = to_float(value)
        if parsed is not None:
            return parsed
    return None


def safe_mean(values: list[float]) -> float | None:
    return mean(values) if values else None


def safe_std(values: list[float]) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return 0.0
    return pstdev(values)


def fmt(value: float | None) -> str:
    return "" if value is None else f"{value:.6f}"


def discover_tariffs(results_root: Path) -> list[str]:
    discovered: set[str] = set()
    for model in MODELS:
        for approach_dir in APPROACH_MAP:
            base = results_root / model / approach_dir
            if not base.exists():
                continue
            for tariff_dir in base.iterdir():
                if tariff_dir.is_dir():
                    discovered.add(tariff_dir.name)

    ordered: list[str] = [tariff for tariff in PREFERRED_TARIFF_ORDER if tariff in discovered]
    ordered.extend(sorted(discovered.difference(ordered)))
    return ordered


def extract_case_metrics(case_data: dict[str, Any]) -> dict[str, float | None]:
    actor_breakdown = case_data.get("actor_breakdown") or {}
    teacher_breakdown = case_data.get("teacher_breakdown") or {}

    actor_reward = pick_numeric(actor_breakdown.get("total_reward"), case_data.get("actor_reward"))
    teacher_reward = pick_numeric(teacher_breakdown.get("total_reward"), case_data.get("teacher_reward"))

    actor_cost = pick_numeric(
        actor_breakdown.get("total_cost"),
        -actor_reward if actor_reward is not None else None,
    )
    teacher_cost = pick_numeric(
        teacher_breakdown.get("total_cost"),
        -teacher_reward if teacher_reward is not None else None,
    )

    reward_gap = actor_reward - teacher_reward if actor_reward is not None and teacher_reward is not None else None
    cost_gap = actor_cost - teacher_cost if actor_cost is not None and teacher_cost is not None else None

    return {
        "actor_reward": actor_reward,
        "teacher_reward": teacher_reward,
        "actor_cost": actor_cost,
        "teacher_cost": teacher_cost,
        "reward_gap": reward_gap,
        "cost_gap": cost_gap,
        "energy_cost": pick_numeric(actor_breakdown.get("energy_cost")),
        "grid_penalty": pick_numeric(actor_breakdown.get("grid_penalty")),
        "bess_cost": pick_numeric(actor_breakdown.get("bess_cost")),
        "ev_cost": pick_numeric(actor_breakdown.get("ev_cost")),
        "total_penalties": pick_numeric(actor_breakdown.get("total_penalties")),
    }


def aggregate_summary(summary_path: Path) -> dict[str, str]:
    content = json.loads(summary_path.read_text(encoding="utf-8"))
    metrics_by_case = [extract_case_metrics(case_data) for case_data in content.values()]

    def collect(field: str) -> list[float]:
        values: list[float] = []
        for entry in metrics_by_case:
            field_value = entry[field]
            if field_value is not None:
                values.append(field_value)
        return values

    actor_reward_values = collect("actor_reward")
    teacher_reward_values = collect("teacher_reward")
    actor_cost_values = collect("actor_cost")
    teacher_cost_values = collect("teacher_cost")
    reward_gap_values = collect("reward_gap")
    cost_gap_values = collect("cost_gap")

    teacher_cost_mean = safe_mean(teacher_cost_values)
    cost_gap_mean = safe_mean(cost_gap_values)
    cost_gap_pct = None
    if teacher_cost_mean is not None and teacher_cost_mean != 0 and cost_gap_mean is not None:
        cost_gap_pct = cost_gap_mean / teacher_cost_mean * 100.0

    return {
        "status": "ok",
        "n_cases": str(len(metrics_by_case)),
        "actor_reward_mean": fmt(safe_mean(actor_reward_values)),
        "actor_reward_std": fmt(safe_std(actor_reward_values)),
        "teacher_reward_mean": fmt(safe_mean(teacher_reward_values)),
        "actor_cost_mean": fmt(safe_mean(actor_cost_values)),
        "actor_cost_std": fmt(safe_std(actor_cost_values)),
        "teacher_cost_mean": fmt(teacher_cost_mean),
        "actor_minus_teacher_reward_mean": fmt(safe_mean(reward_gap_values)),
        "actor_minus_teacher_cost_mean": fmt(cost_gap_mean),
        "actor_minus_teacher_cost_pct": fmt(cost_gap_pct),
        "energy_cost_mean": fmt(safe_mean(collect("energy_cost"))),
        "grid_penalty_mean": fmt(safe_mean(collect("grid_penalty"))),
        "bess_cost_mean": fmt(safe_mean(collect("bess_cost"))),
        "ev_cost_mean": fmt(safe_mean(collect("ev_cost"))),
        "total_penalties_mean": fmt(safe_mean(collect("total_penalties"))),
    }


def empty_row() -> dict[str, str]:
    return {
        "status": "missing",
        "n_cases": "0",
        "actor_reward_mean": "",
        "actor_reward_std": "",
        "teacher_reward_mean": "",
        "actor_cost_mean": "",
        "actor_cost_std": "",
        "teacher_cost_mean": "",
        "actor_minus_teacher_reward_mean": "",
        "actor_minus_teacher_cost_mean": "",
        "actor_minus_teacher_cost_pct": "",
        "energy_cost_mean": "",
        "grid_penalty_mean": "",
        "bess_cost_mean": "",
        "ev_cost_mean": "",
        "total_penalties_mean": "",
    }


def parse_field(row: dict[str, str], field: str) -> float | None:
    value = row.get(field, "")
    if value == "":
        return None
    return to_float(value)


def build_il_vs_rl_rows(comparison_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    by_model_approach: dict[tuple[str, str], dict[str, str]] = {}
    for row in comparison_rows:
        by_model_approach[(row["model"], row["approach"])] = row

    output_rows: list[dict[str, str]] = []
    for model in MODELS:
        il_row = by_model_approach.get((model, "IL"))
        rl_row = by_model_approach.get((model, "RL"))

        il_reward = parse_field(il_row, "actor_reward_mean") if il_row else None
        rl_reward = parse_field(rl_row, "actor_reward_mean") if rl_row else None
        il_cost = parse_field(il_row, "actor_cost_mean") if il_row else None
        rl_cost = parse_field(rl_row, "actor_cost_mean") if rl_row else None

        reward_delta = None
        reward_delta_pct = None
        if il_reward is not None and rl_reward is not None:
            reward_delta = rl_reward - il_reward
            if il_reward != 0:
                reward_delta_pct = reward_delta / abs(il_reward) * 100.0

        cost_delta = None
        cost_delta_pct = None
        if il_cost is not None and rl_cost is not None:
            cost_delta = rl_cost - il_cost
            if il_cost != 0:
                cost_delta_pct = cost_delta / il_cost * 100.0

        better_reward = ""
        if il_reward is not None and rl_reward is not None:
            better_reward = "RL" if rl_reward > il_reward else "IL"

        better_cost = ""
        if il_cost is not None and rl_cost is not None:
            better_cost = "RL" if rl_cost < il_cost else "IL"

        output_rows.append(
            {
                "model": model,
                "il_reward_mean": fmt(il_reward),
                "rl_reward_mean": fmt(rl_reward),
                "rl_minus_il_reward": fmt(reward_delta),
                "rl_minus_il_reward_pct_absbase": fmt(reward_delta_pct),
                "il_cost_mean": fmt(il_cost),
                "rl_cost_mean": fmt(rl_cost),
                "rl_minus_il_cost": fmt(cost_delta),
                "rl_minus_il_cost_pct": fmt(cost_delta_pct),
                "better_reward": better_reward,
                "better_cost": better_cost,
            }
        )

    return output_rows


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    tariffs = discover_tariffs(RESULTS_ROOT)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    overview_rows: list[dict[str, str]] = []

    for tariff in tariffs:
        comparison_rows: list[dict[str, str]] = []

        for model in MODELS:
            for approach_dir, approach_label in APPROACH_MAP.items():
                summary_path = RESULTS_ROOT / model / approach_dir / tariff / "summary.json"
                row = {
                    "tariff": tariff,
                    "model": model,
                    "approach": approach_label,
                }
                if summary_path.exists():
                    row.update(aggregate_summary(summary_path))
                else:
                    row.update(empty_row())
                comparison_rows.append(row)

        comparison_fieldnames = [
            "tariff",
            "model",
            "approach",
            "status",
            "n_cases",
            "actor_reward_mean",
            "actor_reward_std",
            "teacher_reward_mean",
            "actor_cost_mean",
            "actor_cost_std",
            "teacher_cost_mean",
            "actor_minus_teacher_reward_mean",
            "actor_minus_teacher_cost_mean",
            "actor_minus_teacher_cost_pct",
            "energy_cost_mean",
            "grid_penalty_mean",
            "bess_cost_mean",
            "ev_cost_mean",
            "total_penalties_mean",
        ]

        comparison_csv = OUTPUT_ROOT / f"{tariff}_comparison.csv"
        write_csv(comparison_csv, comparison_fieldnames, comparison_rows)

        il_vs_rl_rows = build_il_vs_rl_rows(comparison_rows)
        il_vs_rl_fieldnames = [
            "model",
            "il_reward_mean",
            "rl_reward_mean",
            "rl_minus_il_reward",
            "rl_minus_il_reward_pct_absbase",
            "il_cost_mean",
            "rl_cost_mean",
            "rl_minus_il_cost",
            "rl_minus_il_cost_pct",
            "better_reward",
            "better_cost",
        ]
        il_vs_rl_csv = OUTPUT_ROOT / f"{tariff}_il_vs_rl.csv"
        write_csv(il_vs_rl_csv, il_vs_rl_fieldnames, il_vs_rl_rows)

        valid_rows = [
            row for row in comparison_rows if row["status"] == "ok" and row["actor_reward_mean"] and row["actor_cost_mean"]
        ]
        best_reward_row = max(valid_rows, key=lambda r: float(r["actor_reward_mean"])) if valid_rows else None
        best_cost_row = min(valid_rows, key=lambda r: float(r["actor_cost_mean"])) if valid_rows else None

        overview_rows.append(
            {
                "tariff": tariff,
                "best_reward_model": best_reward_row["model"] if best_reward_row else "",
                "best_reward_approach": best_reward_row["approach"] if best_reward_row else "",
                "best_reward_mean": best_reward_row["actor_reward_mean"] if best_reward_row else "",
                "best_cost_model": best_cost_row["model"] if best_cost_row else "",
                "best_cost_approach": best_cost_row["approach"] if best_cost_row else "",
                "best_cost_mean": best_cost_row["actor_cost_mean"] if best_cost_row else "",
            }
        )

        print(f"Generated: {comparison_csv}")
        print(f"Generated: {il_vs_rl_csv}")

    overview_csv = OUTPUT_ROOT / "overview_by_tariff.csv"
    overview_fieldnames = [
        "tariff",
        "best_reward_model",
        "best_reward_approach",
        "best_reward_mean",
        "best_cost_model",
        "best_cost_approach",
        "best_cost_mean",
    ]
    write_csv(overview_csv, overview_fieldnames, overview_rows)
    print(f"Generated: {overview_csv}")


if __name__ == "__main__":
    main()