"""One-shot script to update test + plot entries in all 9 config.json files."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent

CY = "data/Simulation_CY_Fut_HP__PV5000-HB5000.csv"
WY = "data/Simulation_WY_Fut_HP__PV5000-HB5000.csv"

# Month -> days available in dataset (Dec ends on 30th)
MONTH_DAYS = {
    1: 31, 2: 29, 3: 31, 4: 30, 5: 31, 6: 30,
    7: 31, 8: 31, 9: 30, 10: 31, 11: 30, 12: 30,
}

# Build 24 test entries
tests = []
for m in range(1, 13):
    tests.append({
        "name": f"test_cy_{m:02d}",
        "dataset": CY,
        "date": f"2000-{m:02d}-01 00:00:00",
        "days": MONTH_DAYS[m],
        "soc": 0.5,
    })
for m in range(1, 13):
    tests.append({
        "name": f"test_wy_{m:02d}",
        "dataset": WY,
        "date": f"2000-{m:02d}-01 00:00:00",
        "days": MONTH_DAYS[m],
        "soc": 0.5,
    })

# Build 24 plot entries
plots = []
for entry in tests:
    plots.append({
        "name": entry["name"],
        "date": entry["date"],
        "days": 3,
    })

# All 9 config.json files
configs = [
    ROOT / "models" / model / stage / "config.json"
    for model in ("MLP", "GRU", "ATT")
    for stage in ("1-IL", "2-RL", "3-FT")
]

for path in configs:
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    cfg["test"] = tests
    cfg["plot"] = plots

    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=4, ensure_ascii=False)
        f.write("\n")

    print(f"[OK] {path.relative_to(ROOT)}")

print(f"\nDone — {len(tests)} test entries + {len(plots)} plot entries in {len(configs)} files.")
