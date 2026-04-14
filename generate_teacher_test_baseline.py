import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
from typing import Any

from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_ROOT = PROJECT_ROOT / "models"
CACHE_ROOT = PROJECT_ROOT / "Results" / "test" / "_teacher_shared_cache"
CACHE_SUMMARY_FILENAME = "cache_summary.json"

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(MODEL_ROOT))

from models.test_utils.teacher_eval import run_teacher_runs, save_teacher_summary

DEFAULT_TARIFFS = ["tar_s", "tar_w", "tar_sw", "tar_tou", "tar_flat"]
DEFAULT_FAMILIES = ["ATT", "ATT_MEM", "GRU", "MLP", "TCN", "ATTv2", "ATT_MEMv2", "GRUv2", "MLPv2", "TCNv2"]
DEFAULT_STAGES = ["1-IL", "2-RL"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run teacher-only test baselines for all tariffs using shared cache "
            "(teacher is architecture-independent) and write teacher_summary.json files."
        ),
    )
    parser.add_argument(
        "--family",
        nargs="+",
        default=["all"],
        help="Family list (e.g. ATT MLPv2) or 'all'.",
    )
    parser.add_argument(
        "--stage",
        choices=["1-IL", "2-RL", "all"],
        default="all",
        help="Test stage to execute.",
    )
    parser.add_argument(
        "--skip-operation-csv",
        action="store_true",
        help="Do not write teacher operation CSV files.",
    )
    parser.add_argument(
        "--skip-breakdown-csv",
        action="store_true",
        help="Do not write teacher breakdown CSV files.",
    )
    parser.add_argument(
        "--skip-breakdown-summary",
        action="store_true",
        help="Do not include detailed teacher_breakdown in summary JSON.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce progress bar output.",
    )
    return parser.parse_args()


def resolve_families(requested: list[str]) -> list[str]:
    if any(name.lower() == "all" for name in requested):
        candidates = DEFAULT_FAMILIES
    else:
        candidates = requested

    families = []
    for family in candidates:
        family_dir = MODEL_ROOT / family
        if family_dir.exists() and family_dir.is_dir():
            families.append(family)
        else:
            print(f"[teacher] skipping unknown family: {family}")
    return families


def resolve_stages(requested: str) -> list[str]:
    if requested == "all":
        return list(DEFAULT_STAGES)
    return [requested]


def run_signature(run: dict[str, Any], tariff: str) -> str:
    dataset = str(run.get("dataset", "")).strip()
    date = str(run.get("date", "")).strip()
    days = str(run.get("days", "")).strip()

    try:
        soc = f"{float(run.get('soc', 0.0)):.10f}"
    except (TypeError, ValueError):
        soc = str(run.get("soc", "")).strip()

    raw = f"{tariff}|{dataset}|{date}|{days}|{soc}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def sanitize_token(text: str, max_len: int = 40) -> str:
    ascii_text = text.encode("ascii", "ignore").decode("ascii").lower()
    token = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in ascii_text)
    while "__" in token:
        token = token.replace("__", "_")
    token = token.strip("_")
    if not token:
        token = "na"
    return token[:max_len]


def run_cache_name(run: dict[str, Any]) -> str:
    run_name = sanitize_token(str(run.get("name", "run")), max_len=24)
    dataset_stem = sanitize_token(Path(str(run.get("dataset", "dataset"))).stem, max_len=20)
    date = sanitize_token(str(run.get("date", "")), max_len=19)
    days = sanitize_token(str(run.get("days", "")), max_len=8)

    try:
        soc = f"{float(run.get('soc', 0.0)):.3f}".replace(".", "p")
    except (TypeError, ValueError):
        soc = sanitize_token(str(run.get("soc", "")), max_len=10)

    return f"{run_name}__{dataset_stem}__{date}__d{days}__soc{soc}"


def has_cache_entry(
    cache_summary: dict[str, Any],
    key: str,
    include_breakdown_summary: bool,
    save_operation_csv: bool,
    save_breakdown_csv: bool,
    cache_folder: Path,
) -> bool:
    entry = cache_summary.get(key)
    if not isinstance(entry, dict):
        return False
    if "teacher_reward" not in entry:
        return False
    if include_breakdown_summary and entry.get("teacher_breakdown", None) is None:
        return False

    cache_name = str(entry.get("cache_name", key))

    if save_operation_csv:
        if not (cache_folder / f"{cache_name}_teacher_operation.csv").exists():
            return False
        if not (cache_folder / f"{cache_name}_env_operation.csv").exists():
            return False

    if save_breakdown_csv and not (cache_folder / f"{cache_name}_env_operation_breakdown.csv").exists():
        return False

    return True


def copy_required_file(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(f"Required cached file not found: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def main() -> None:
    args = parse_args()

    families = resolve_families(args.family)
    stages = resolve_stages(args.stage)
    tariffs = list(DEFAULT_TARIFFS)

    if not families:
        raise SystemExit("No valid family selected.")

    with open(PROJECT_ROOT / "data" / "parameters.json", encoding="utf-8") as f:
        par = json.load(f)

    combo_list: list[tuple[str, str]] = []
    for family in families:
        for stage in stages:
            cfg_path = MODEL_ROOT / family / stage / "config.json"
            if cfg_path.exists():
                combo_list.append((family, stage))
            else:
                print(f"[teacher] skipping missing config: {cfg_path}")

    if not combo_list:
        raise SystemExit("No valid family/stage combinations found.")

    combo_runs: dict[tuple[str, str], list[dict[str, Any]]] = {}
    combo_iter = combo_list
    if not args.quiet:
        combo_iter = tqdm(combo_list, desc="Load test suites", position=0, dynamic_ncols=True)

    for family, stage in combo_iter:
        cfg_path = MODEL_ROOT / family / stage / "config.json"
        with open(cfg_path, encoding="utf-8") as f:
            cfg = json.load(f)

        runs = list(cfg.get("test", []))
        if not runs:
            print(f"[teacher] no test runs in {cfg_path}")
            continue

        combo_runs[(family, stage)] = runs

    if not combo_runs:
        raise SystemExit("No test runs found in selected configurations.")

    required_by_tariff: dict[str, dict[str, dict[str, Any]]] = {tariff: {} for tariff in tariffs}
    used_cache_names_by_tariff: dict[str, set[str]] = {tariff: set() for tariff in tariffs}
    for runs in combo_runs.values():
        for run in runs:
            for tariff in tariffs:
                key = run_signature(run, tariff)
                if key in required_by_tariff[tariff]:
                    continue

                cache_name = run_cache_name(run)
                if cache_name in used_cache_names_by_tariff[tariff]:
                    cache_name = f"{cache_name}__{key[:6]}"
                used_cache_names_by_tariff[tariff].add(cache_name)

                run_for_cache = dict(run)
                run_for_cache["name"] = cache_name
                required_by_tariff[tariff][key] = {
                    "run": run_for_cache,
                    "cache_name": cache_name,
                }

    total_unique = sum(len(v) for v in required_by_tariff.values())
    print(f"[teacher] unique teacher solves required (tariff+run): {total_unique}")

    cache_by_tariff: dict[str, dict[str, Any]] = {}
    tariff_iter = tariffs
    if not args.quiet:
        tariff_iter = tqdm(tariffs, desc="Build shared teacher cache", position=0, dynamic_ncols=True)

    for tariff in tariff_iter:
        cache_folder = CACHE_ROOT / tariff
        cache_folder.mkdir(parents=True, exist_ok=True)
        cache_summary_path = cache_folder / CACHE_SUMMARY_FILENAME

        if cache_summary_path.exists():
            with open(cache_summary_path, "r", encoding="utf-8") as f:
                cache_summary = json.load(f)
        else:
            cache_summary = {}

        required = required_by_tariff[tariff]
        missing_keys = [
            key
            for key in required.keys()
            if not has_cache_entry(
                cache_summary=cache_summary,
                key=key,
                include_breakdown_summary=not args.skip_breakdown_summary,
                save_operation_csv=not args.skip_operation_csv,
                save_breakdown_csv=not args.skip_breakdown_csv,
                cache_folder=cache_folder,
            )
        ]

        miss_iter = missing_keys
        if not args.quiet and missing_keys:
            miss_iter = tqdm(
                missing_keys,
                desc=f"{tariff} cache misses",
                position=1,
                dynamic_ncols=True,
                leave=False,
            )

        for key in miss_iter:
            cache_item = required[key]
            cache_name = str(cache_item["cache_name"])
            run_result = run_teacher_runs(
                runs=[cache_item["run"]],
                tariff=tariff,
                par=par,
                folder=cache_folder,
                save_operation_csv=not args.skip_operation_csv,
                save_breakdown_csv=not args.skip_breakdown_csv,
                include_breakdown_summary=not args.skip_breakdown_summary,
                show_progress=not args.quiet,
                pbar_position=2,
            )
            cache_entry = dict(run_result[cache_name])
            cache_entry["cache_name"] = cache_name
            cache_summary[key] = cache_entry

        with open(cache_summary_path, "w", encoding="utf-8") as f:
            json.dump(cache_summary, f, indent=4)

        cache_by_tariff[tariff] = cache_summary

    materialize_iter = list(combo_runs.items())
    if not args.quiet:
        materialize_iter = tqdm(materialize_iter, desc="Write suite outputs", position=0, dynamic_ncols=True)

    for (family, stage), runs in materialize_iter:
        for tariff in tariffs:
            folder = PROJECT_ROOT / "Results" / "test" / family / stage / tariff
            folder.mkdir(parents=True, exist_ok=True)

            cache_folder = CACHE_ROOT / tariff
            cache_summary = cache_by_tariff[tariff]
            summary: dict[str, dict[str, Any]] = {}

            for run in runs:
                key = run_signature(run, tariff)
                if key not in cache_summary:
                    raise RuntimeError(
                        f"Missing cached teacher result for key={key} tariff={tariff} run={run.get('name', '<unnamed>')}"
                    )

                cache_entry = cache_summary[key]
                cache_name = str(cache_entry.get("cache_name", key))
                summary[run["name"]] = {
                    "teacher_reward": float(cache_entry["teacher_reward"]),
                    "teacher_breakdown": cache_entry.get("teacher_breakdown", None),
                    "dataset": run["dataset"],
                    "date": run["date"],
                    "days": run["days"],
                    "soc": run["soc"],
                }

                if not args.skip_operation_csv:
                    copy_required_file(
                        cache_folder / f"{cache_name}_teacher_operation.csv",
                        folder / f"{run['name']}_teacher_operation.csv",
                    )
                    copy_required_file(
                        cache_folder / f"{cache_name}_env_operation.csv",
                        folder / f"{run['name']}_env_operation.csv",
                    )

                if not args.skip_breakdown_csv:
                    copy_required_file(
                        cache_folder / f"{cache_name}_env_operation_breakdown.csv",
                        folder / f"{run['name']}_env_operation_breakdown.csv",
                    )

            save_teacher_summary(folder, summary)
            print(f"[teacher] saved {folder / 'teacher_summary.json'}")


if __name__ == "__main__":
    main()
