from __future__ import annotations

import csv
import json
import random
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import optuna
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.append(str(Path(__file__).resolve().parent))

from hp import HyperParameters  # type: ignore
from model import Actor  # type: ignore
from opt import Teacher  # type: ignore

RUN_CONFIG_PATH = Path(__file__).with_name("run_config.json")


def rpath(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else (PROJECT_ROOT / p)


def load_df(p: Path) -> pd.DataFrame:
    df = pd.read_csv(p, sep=";")
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], dayfirst=True, errors="coerce")
    return df


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _write_epoch_log_row(log_path: Path, row: dict) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not log_path.exists()
    with open(log_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if new_file:
            w.writeheader()
        w.writerow(row)


def fit_actor(
    X: np.ndarray,
    Y: np.ndarray,
    hp_dict: dict,
    il: dict,
    seed: int,
    device: torch.device,
    log_path: Optional[Path] = None,
) -> Tuple[dict, float, dict]:
    """
    Treina um Actor e retorna:
      - state_dict
      - best_val_loss
      - hp.to_dict()
    Se log_path for fornecido, salva loss por época (train/val).
    """
    hp_dict = dict(hp_dict)
    hp_dict["input_dim"] = int(X.shape[1])
    hp_dict["output_dim"] = int(Y.shape[1])
    hp = HyperParameters.from_dict(hp_dict)

    rs = np.random.RandomState(seed)
    idx = rs.permutation(X.shape[0])
    split = int(X.shape[0] * (1.0 - float(il.get("val_frac", 0.2))))
    tr, va = idx[:split], idx[split:]

    Xt = torch.as_tensor(X, dtype=torch.float32, device=device)
    Yt = torch.as_tensor(Y, dtype=torch.float32, device=device)

    actor = Actor(hp).to(device)
    opt = torch.optim.Adam(
        actor.parameters(),
        lr=float(il.get("lr", 1e-3)),
        weight_decay=float(il.get("weight_decay", 0.0)),
    )
    crit = nn.MSELoss()

    best, best_sd, bad = 1e30, None, 0
    pat = int(il.get("patience", 10))
    tol = float(il.get("improvement_tol", 1e-6))
    clip = float(il.get("grad_clip", 1.0))
    bsz = int(il.get("batch_size", 256))
    epochs = int(il.get("epochs", 50))

    ntr = int(tr.size)
    nva = int(va.size)

    for ep in range(1, epochs + 1):
        actor.train()
        tr_loss_sum = 0.0
        tr_count = 0

        for i in range(0, tr.size, bsz):
            b = tr[i : i + bsz]
            pred = actor(Xt[b])
            loss = crit(pred, Yt[b])

            opt.zero_grad(set_to_none=True)
            loss.backward()
            if clip > 0:
                nn.utils.clip_grad_norm_(actor.parameters(), clip)
            opt.step()

            bs = int(b.size)
            tr_loss_sum += float(loss.item()) * bs
            tr_count += bs

        actor.eval()
        with torch.no_grad():
            if va.size:
                v = float(crit(actor(Xt[va]), Yt[va]).item())
            else:
                v = float(crit(actor(Xt[tr]), Yt[tr]).item())

        tr_loss = (tr_loss_sum / max(1, tr_count))

        if log_path is not None:
            _write_epoch_log_row(
                log_path,
                {
                    "epoch": ep,
                    "train_loss": tr_loss,
                    "val_loss": v,
                    "n_train": ntr,
                    "n_val": nva,
                    "batch_size": bsz,
                    "lr": float(il.get("lr", 1e-3)),
                    "weight_decay": float(il.get("weight_decay", 0.0)),
                },
            )

        if best - v > tol:
            best, best_sd, bad = v, {k: t.detach().cpu().clone() for k, t in actor.state_dict().items()}, 0
        else:
            bad += 1
            if pat > 0 and bad >= pat:
                break

    actor.load_state_dict(best_sd if best_sd is not None else actor.state_dict(), strict=False)
    return actor.state_dict(), float(best), hp.to_dict()


def build_full_xy_for_tariff(run_cfg: dict, tariff: str, actions: list[str], start_soc: float, solver_name: str):
    cfg = json.load(open(rpath(run_cfg["config"]), "r", encoding="utf-8"))
    cfg.setdefault("Grid", {})
    cfg["Grid"]["tariff_column"] = tariff

    out_root = rpath(run_cfg.get("results_root", "Results")) / tariff / "cache"
    out_root.mkdir(parents=True, exist_ok=True)

    Xs, Ys, labels = [], [], None

    for r in run_cfg["runs"]:
        df = load_df(rpath(r["data"]))
        sd = str(r["start_date"])
        days = int(r["days"])
        run_label = str(r.get("run_label", Path(r["data"]).stem))

        tag = f"ILFULL_{run_label}_{days}_{sd.replace('/','-').replace(':','-').replace(' ','_')}_soc{start_soc:.3f}.npz"
        cpath = out_root / tag

        if cpath.exists():
            z = np.load(cpath, allow_pickle=True)
            Xs.append(z["X"].astype(np.float32, copy=False))
            Ys.append(z["Y"].astype(np.float32, copy=False))
            if labels is None:
                labels = list(z["labels"].tolist())
            print(f"[IL/cache] {tariff} {run_label}: loaded {z['X'].shape[0]} samples")
            continue

        print(f"[IL/teacher] {tariff} {run_label}: solving MILP for {days} days from {sd}")
        teacher = Teacher(cfg, df, start_date=sd, days=days, state_mask=None)
        teacher.build(start_soc=start_soc)
        teacher.solve(solver_name=solver_name)

        X, lbl = teacher.get_full_states()
        rdf = teacher.results_df()

        missing = [a for a in actions if a not in rdf.columns]
        if missing:
            raise KeyError(
                f"Teacher.results_df() não possui colunas de ação {missing}. "
                f"Disponíveis: {list(rdf.columns)}"
            )

        Y = rdf.loc[:, actions].to_numpy(dtype=np.float32)

        np.savez_compressed(cpath, X=X.astype(np.float32, copy=False), Y=Y, labels=np.array(lbl, dtype=object))
        Xs.append(X.astype(np.float32, copy=False))
        Ys.append(Y)
        if labels is None:
            labels = list(lbl)
        print(f"[IL/cache] {tariff} {run_label}: saved {X.shape[0]} samples")

    return np.concatenate(Xs, axis=0), np.concatenate(Ys, axis=0), (labels or [])


def main() -> None:
    run_cfg = json.load(open(RUN_CONFIG_PATH, "r", encoding="utf-8"))
    seed_all(int(run_cfg.get("seed", 0)))

    cfg = json.load(open(rpath(run_cfg.get("config", "data/parameters.json")), "r", encoding="utf-8"))
    tariffs = run_cfg.get("tariffs") or [cfg.get("Grid", {}).get("tariff_column", "tar_tou")]

    il_base = dict(run_cfg.get("il", {}))
    hp_base = dict(il_base.get("hyperparameters", {}))

    hpo = dict(run_cfg.get("il_hpo", {}))
    enabled = bool(hpo.get("enabled", True))

    # IMPORTANTE: 3 ações por default (compatível com env.step: pb, pe, x)
    actions = list(il_base.get("actions", ["PBESS", "Pev", "chi_pv"]))
    start_soc = float(il_base.get("start_soc", 0.5))
    solver_name = str(il_base.get("solver_name", "gurobi"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results_root = rpath(run_cfg.get("results_root", "Results"))

    print(f"[IL] device={device} actions={actions} start_soc={start_soc} solver={solver_name}")

    

    for tariff in [str(t) for t in tariffs]:
        print(f"\n[IL] ===== tariff={tariff} =====")
        Xfull, Yfull, labels = build_full_xy_for_tariff(run_cfg, tariff, actions, start_soc, solver_name)
        print(f"[IL] full dataset: X={Xfull.shape} Y={Yfull.shape} features={len(labels)}")

        out_dir = results_root / tariff / "1_MLP_IL"
        out_dir.mkdir(parents=True, exist_ok=True)

        if not enabled:
            mv = np.ones(len(labels), dtype=bool)
            sd, best, hp_save = fit_actor(
                Xfull[:, mv], Yfull, hp_base, il_base, int(run_cfg.get("seed", 0)), device,
                log_path=out_dir / "train_log.csv",
            )
            torch.save(
                {
                    "model_state_dict": sd,
                    "hyperparameters": hp_save,
                    "state_mask": {"spec": "all", "vector": mv.tolist(), "labels": labels},
                    "actions": actions,
                    "run_cfg": run_cfg,
                    "best_val_loss": best,
                },
                out_dir / "best.pt",
            )
            print(f"[IL] saved: {out_dir / 'best.pt'} val={best:.6g}")
            continue

        n_trials = int(hpo.get("n_trials", 30))
        trial_epochs = int(hpo.get("trial_epochs", 20))
        trial_patience = int(hpo.get("trial_patience", 3))
        sample_frac = float(hpo.get("sample_frac", 0.10))
        max_samples = int(hpo.get("max_samples", 0))
        seed = int(hpo.get("seed", int(run_cfg.get("seed", 0))))
        mask_mode = str(hpo.get("mask_mode", "per_feature"))
        mask_min = int(hpo.get("mask_min_features", 1))

        hidden_sizes = tuple(tuple(h) for h in hpo.get("hidden_sizes", [[256, 128], [256, 256], [512, 256]]))
        batch_sizes = hpo.get("batch_sizes", [128, 256, 512])
        dropout_rng = hpo.get("dropout_range", [0.0, 0.3])
        lr_log = hpo.get("lr_log10_range", [-4.5, -2.5])
        wd_log = hpo.get("wd_log10_range", [-6.0, -3.0])

        n = Xfull.shape[0]
        k = int(n * sample_frac)
        if max_samples > 0:
            k = min(k, max_samples)
        k = max(256, min(k, n))

        hpo_logs_dir = out_dir / "hpo_logs"
        hpo_logs_dir.mkdir(parents=True, exist_ok=True)

        def objective(trial: optuna.Trial) -> float:
            rs = np.random.RandomState(seed + trial.number)

            if mask_mode == "p_drop":
                p_drop = float(trial.suggest_float("p_drop", 0.0, 0.8))
                mv = (rs.rand(len(labels)) > p_drop)
            else:
                mv = np.array([trial.suggest_int(f"m{i}", 0, 1) for i in range(len(labels))], dtype=bool)

            if mv.sum() < mask_min:
                on = rs.choice(len(labels), size=min(mask_min, len(labels)), replace=False)
                mv[:] = False
                mv[on] = True

            idx = rs.choice(n, size=k, replace=False)

            il = dict(il_base)
            il["epochs"] = trial_epochs
            il["patience"] = trial_patience
            il["batch_size"] = int(trial.suggest_categorical("batch_size", batch_sizes))
            il["lr"] = 10 ** float(trial.suggest_float("lr_log10", float(lr_log[0]), float(lr_log[1])))
            il["weight_decay"] = 10 ** float(trial.suggest_float("wd_log10", float(wd_log[0]), float(wd_log[1])))

            hp = dict(hp_base)
            hp["hidden_sizes"] = trial.suggest_categorical("hidden_sizes", hidden_sizes)
            hp["dropout"] = float(trial.suggest_float("dropout", float(dropout_rng[0]), float(dropout_rng[1])))

            trial_log = hpo_logs_dir / f"trial_{trial.number:04d}.csv"
            _, v, _ = fit_actor(
                Xfull[idx][:, mv],
                Yfull[idx],
                hp,
                il,
                seed,
                device,
                log_path=trial_log,
            )
            return v

        print(f"[IL/HPO] tariff={tariff} trials={n_trials} sample={k}/{n} mask_mode={mask_mode} mask_min={mask_min}")
        study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed, multivariate=True))
        pbar = tqdm(total=n_trials, desc=f"HPO {tariff}")
        study.optimize(objective, n_trials=n_trials, callbacks=[lambda *_: pbar.update(1)], show_progress_bar=False)
        pbar.close()

        p = study.best_trial.params
        if mask_mode == "p_drop":
            rs = np.random.RandomState(seed + study.best_trial.number)
            mv = (rs.rand(len(labels)) > float(p.get("p_drop", 0.0)))
        else:
            mv = np.array([bool(p.get(f"m{i}", 1)) for i in range(len(labels))], dtype=bool)

        if mv.sum() < mask_min:
            rs = np.random.RandomState(seed + 9999)
            on = rs.choice(len(labels), size=min(mask_min, len(labels)), replace=False)
            mv[:] = False
            mv[on] = True

        il_final = dict(il_base)
        hp_final = dict(hp_base)

        # IMPORTANTE: o treino completo usa os hiperparâmetros do MELHOR trial (HPO)
        il_final["batch_size"] = int(p["batch_size"])
        il_final["lr"] = 10 ** float(p["lr_log10"])
        il_final["weight_decay"] = 10 ** float(p["wd_log10"])
        hp_final["hidden_sizes"] = p["hidden_sizes"]
        hp_final["dropout"] = float(p["dropout"])

        sd, best, hp_save = fit_actor(
            Xfull[:, mv],
            Yfull,
            hp_final,
            il_final,
            seed,
            device,
            log_path=out_dir / "train_log.csv",
        )

        print(f"[IL] best_trial={float(study.best_value):.6g} final_val={best:.6g} kept={int(mv.sum())}/{len(mv)}")

        torch.save(
            {
                "model_state_dict": sd,
                "hyperparameters": hp_save,
                "state_mask": {"spec": "binary_per_feature", "vector": mv.tolist(), "labels": labels},
                "actions": actions,
                "run_cfg": run_cfg,
                "best_val_loss": best,
            },
            out_dir / "best.pt",
        )

        study.trials_dataframe().to_csv(out_dir / "hpo_trials.csv", index=False)
        json.dump(
            {"best_value": float(study.best_value), "best_params": study.best_trial.params},
            open(out_dir / "hpo_best.json", "w", encoding="utf-8"),
            indent=2,
            ensure_ascii=False,
        )

        print(f"[IL] saved: {out_dir / 'best.pt'}")
        print(f"[IL] saved: {out_dir / 'hpo_trials.csv'}")
        print(f"[IL] saved: {out_dir / 'hpo_best.json'}")
        print(f"[IL] saved: {out_dir / 'train_log.csv'}")
        print(f"[IL] saved: {hpo_logs_dir}/trial_XXXX.csv")


if __name__ == "__main__":
    main()
