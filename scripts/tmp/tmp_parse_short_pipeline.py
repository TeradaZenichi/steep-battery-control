import json
from pathlib import Path

p = Path("Results/analysis/short_pipeline_v2_rl_updates_summary.json")
data = json.loads(p.read_text(encoding="utf-8"))
data = [d for d in data if d.get("status") == "ok"]
data.sort(key=lambda x: x["checkpoint_score"], reverse=True)

print("model|checkpoint_score|eval_reward_det|frac_violation|lambda|n_updates")
for d in data:
    print(
        "{}|{:.6f}|{:.6f}|{:.6f}|{:.6f}|{}".format(
            d["model"],
            d["checkpoint_score"],
            d["eval_reward_det"],
            d["frac_violation"],
            d["lambda"],
            int(d["n_updates"]),
        )
    )
