import json
from pathlib import Path

root = Path("Results/train")
models = ["ATTv2", "ATT_MEMv2", "GRUv2", "MLPv2", "TCNv2"]
for m in models:
    p = root / m / "2-RL" / "tar_tou" / "best_eval_meta.json"
    if not p.exists():
        print(f"{m}|MISSING")
        continue
    d = json.loads(p.read_text(encoding="utf-8"))
    print(
        f"{m}|best_ckpt={d.get('best_checkpoint_score')}|best_eval_det={d.get('best_eval_reward')}|best_eval_stoch={d.get('best_eval_reward_stoch')}|episode={d.get('best_checkpoint_episode')}"
    )
