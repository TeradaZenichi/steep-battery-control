"""Apply config alignment changes across all 2-RL and 3-FT configs."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent

configs = {
    "MLP_2RL": ROOT / "models/MLP/2-RL/config.json",
    "MLP_3FT": ROOT / "models/MLP/3-FT/config.json",
    "GRU_2RL": ROOT / "models/GRU/2-RL/config.json",
    "GRU_3FT": ROOT / "models/GRU/3-FT/config.json",
    "ATT_2RL": ROOT / "models/ATT/2-RL/config.json",
    "ATT_3FT": ROOT / "models/ATT/3-FT/config.json",
}

for name, path in configs.items():
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    t = cfg["train"]
    changes = []

    # --- buffer_size: 400K → 200K (all) ---
    if t.get("buffer_size") != 200000:
        changes.append(f"buffer_size {t.get('buffer_size')} → 200000")
        t["buffer_size"] = 200000

    # --- Remove checkpoint_min_delta and policy_delay (all) ---
    if "checkpoint_min_delta" in t:
        changes.append(f"remove checkpoint_min_delta={t.pop('checkpoint_min_delta')}")
    if "policy_delay" in t:
        changes.append(f"remove policy_delay={t.pop('policy_delay')}")

    # --- 2-RL specific ---
    if "2RL" in name:
        # Align target_entropy to -1.5
        if t.get("target_entropy") != -1.5:
            changes.append(f"target_entropy {t.get('target_entropy')} → -1.5")
            t["target_entropy"] = -1.5

        # Align lambda_lr to 5e-4 (MLP value)
        if t.get("lambda_lr") != 5e-4:
            changes.append(f"lambda_lr {t.get('lambda_lr')} → 5e-4")
            t["lambda_lr"] = 5e-4

        # Align n_step to 16 (MLP value)
        if t.get("n_step") != 16:
            changes.append(f"n_step {t.get('n_step')} → 16")
            t["n_step"] = 16

    # --- 3-FT specific ---
    if "3FT" in name:
        # MLP 3-FT: evaluate_every 5 → 2
        if t.get("evaluate_every") != 2:
            changes.append(f"evaluate_every {t.get('evaluate_every')} → 2")
            t["evaluate_every"] = 2

    if changes:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=4, ensure_ascii=False)
            f.write("\n")
        print(f"[OK] {name}: {', '.join(changes)}")
    else:
        print(f"[--] {name}: no changes needed")
