from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

ACTOR_VARIANTS = ("combo", "det", "stoch")


def actor_operation_filename(run_name: str, actor_variant: str) -> str:
    return f"{run_name}_actor_env_operation_{actor_variant}.csv"


def actor_operation_breakdown_filename(run_name: str, actor_variant: str) -> str:
    return f"{run_name}_actor_env_operation_breakdown_{actor_variant}.csv"


def actor_summary_filename(actor_variant: str) -> str:
    return f"summary_{actor_variant}.json"


def write_variant_summary(
    folder: Path,
    summary: dict[str, Any],
    actor_variant: str,
    write_legacy_summary_for_combo: bool = True,
) -> Path:
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)

    summary_path = folder / actor_summary_filename(actor_variant)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)

    if write_legacy_summary_for_combo and actor_variant == "combo":
        with open(folder / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=4)

    return summary_path


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _load_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def update_variant_comparison_csv(model_stage_root: Path) -> Path:
    model_stage_root = Path(model_stage_root)
    model_stage_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []

    tariff_dirs = [p for p in sorted(model_stage_root.iterdir()) if p.is_dir()]
    for tariff_dir in tariff_dirs:
        run_rows: dict[str, dict[str, Any]] = {}

        for variant in ACTOR_VARIANTS:
            summary = _load_summary(tariff_dir / actor_summary_filename(variant))
            for run_name, info in summary.items():
                row = run_rows.setdefault(
                    run_name,
                    {
                        "tariff": tariff_dir.name,
                        "run_name": run_name,
                        "dataset": info.get("dataset"),
                        "date": info.get("date"),
                        "days": info.get("days"),
                        "soc": info.get("soc"),
                        "teacher_reward": _safe_float(info.get("teacher_reward")),
                        "actor_reward_combo": math.nan,
                        "actor_reward_det": math.nan,
                        "actor_reward_stoch": math.nan,
                        "reward_diff_combo": math.nan,
                        "reward_diff_det": math.nan,
                        "reward_diff_stoch": math.nan,
                    },
                )

                # Keep metadata filled if one variant has it and others do not.
                for key in ("dataset", "date", "days", "soc"):
                    if row.get(key) is None and info.get(key) is not None:
                        row[key] = info.get(key)

                teacher_reward = _safe_float(info.get("teacher_reward"))
                if math.isnan(row["teacher_reward"]) and not math.isnan(teacher_reward):
                    row["teacher_reward"] = teacher_reward

                row[f"actor_reward_{variant}"] = _safe_float(info.get("actor_reward"))
                row[f"reward_diff_{variant}"] = _safe_float(info.get("reward_diff"))

        rows.extend(run_rows.values())

    comparison_path = model_stage_root / "actor_variant_comparison.csv"
    if rows:
        df = pd.DataFrame(rows)
        df = df.sort_values(["tariff", "run_name"]).reset_index(drop=True)
        df.to_csv(comparison_path, index=False)
    else:
        pd.DataFrame(
            columns=[
                "tariff",
                "run_name",
                "dataset",
                "date",
                "days",
                "soc",
                "teacher_reward",
                "actor_reward_combo",
                "actor_reward_det",
                "actor_reward_stoch",
                "reward_diff_combo",
                "reward_diff_det",
                "reward_diff_stoch",
            ]
        ).to_csv(comparison_path, index=False)

    return comparison_path
