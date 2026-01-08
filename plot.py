# plot.py
#
# Ajuste as variáveis na seção CONFIG e rode:
#   python plot.py
#
# Para cada entrada, o script gera dois PDFs (actor e teacher) a partir dos
# CSVs de avaliação com colunas: timestamp, PPV, Pload, PGRID, EBESS, SoCBESS,
# EEV, SoCEV, PBESS, PEV, XPV.

from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================
# CONFIG (EDIT HERE)
# =========================
RESULTS_ROOT = Path("Results")  # relativo ao projeto
PARAMS_FILE = Path("data/parameters.json")  # usado para normalizar SoC se necessário

# Listas com o mesmo tamanho: uma entrada por par (actor/teacher)
MODELS = [
    "1_MLP_IL","2_MLP_FT_SAC"
]

TARIFFS = [
    "tar_s",
]

RUN_LABELS = [
    "EVAL_CY_year",
]

START_DATES = [
    "2000-01-10 00:00:00",
]

# Pode ser inteiro (igual para todas as entradas) ou lista com mesmo tamanho
N_DAYS = 5

# Se seu CSV usa nomes diferentes, ajuste aqui.
CAND_PGRID = ["PGRID", "pgrid", "Pgrid"]
CAND_PBESS = ["PBESS", "PBESS_env", "act_PBESS"]
CAND_PEV = ["PEV", "Pev"]
CAND_PLOAD = ["Pload", "Load", "load_kw", "load"]
CAND_PV_USED = ["PPV", "Ppv", "ppv_used", "act_ppv_used"]
CAND_XPV = ["XPV", "xpv", "chi_pv", "act_XPV"]
CAND_SOC_BESS = ["SoCBESS", "soc_bess", "SoC_BESS"]
CAND_SOC_EV = ["SoCEV", "soc_ev", "SoC_EV"]
CAND_EBESS = ["EBESS", "E_bess"]
CAND_EEV = ["EEV", "E_ev"]

# Figure style
FIGSIZE = (14, 8)
DPI = 200
# =========================


def _pick_first_existing(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _load_params(params_file: Path) -> dict:
    """Load normalization parameters from JSON config."""
    if not params_file.exists():
        raise FileNotFoundError(f"Parameters file not found: {params_file}")
    with open(params_file, "r") as f:
        return json.load(f)


def _require(col: str | None, what: str, df: pd.DataFrame, candidates: list[str]) -> str:
    if col is None:
        raise KeyError(
            f"Missing column for '{what}'.\n"
            f"Candidates tried: {candidates}\n"
            f"Available columns ({len(df.columns)}): {list(df.columns)}"
        )
    return col


def _expand_list(val, n):
    """Retorna lista de tamanho n.

    - Se val não é lista, repete n vezes.
    - Se val é lista de tamanho 1, repete n vezes.
    - Se é lista de tamanho n, usa como está.
    - Caso contrário, lança erro.
    """
    if not isinstance(val, list):
        return [val] * n
    if len(val) == 1:
        return val * n
    if len(val) == n:
        return val
    raise ValueError(f"Lista deve ter tamanho 1 ou {n}, mas veio {len(val)}: {val}")


def validate_config():
    n = len(MODELS)
    if n == 0:
        raise ValueError("MODELS está vazio.")
    # Toleramos listas de tamanho 1 ou do mesmo tamanho de MODELS.
    for name, seq in [("TARIFFS", TARIFFS), ("RUN_LABELS", RUN_LABELS), ("START_DATES", START_DATES)]:
        if not isinstance(seq, list):
            raise ValueError(f"{name} deve ser lista.")
        if len(seq) not in (1, n):
            raise ValueError(
                f"{name} deve ter tamanho 1 ou {n}. len({name})={len(seq)}"
            )


def build_csv_path(results_root: Path, model: str, tariff: str, run_label: str, role: str) -> Path:
    base = results_root / model
    eval_dir = base / "evaluation"
    if eval_dir.exists():
        base = eval_dir
    return base / f"{role}_{tariff}_{run_label}.csv"


def load_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, sep=None, engine="python")  # infere separador

    # Caso 1: já existe coluna timestamp
    if "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.drop(columns=["timestamp"])
        df.index = ts

    else:
        # Caso 2: timestamp foi salvo como índice -> vira "Unnamed: 0" (mais comum)
        if "Unnamed: 0" in df.columns:
            ts = pd.to_datetime(df["Unnamed: 0"], errors="coerce")
            df = df.drop(columns=["Unnamed: 0"])
            df.index = ts
        else:
            # Caso 3: primeira coluna contém strings de data/hora (sem nome específico)
            first_col = df.columns[0]
            ts_try = pd.to_datetime(df[first_col], errors="coerce")
            if ts_try.notna().mean() > 0.9:  # maioria parseável => assume timestamp
                df = df.drop(columns=[first_col])
                df.index = ts_try
            else:
                # Último fallback (não recomendado): tenta usar o índice atual
                df.index = pd.to_datetime(df.index, errors="coerce")

    df = df[~df.index.isna()].sort_index()
    df.index.name = "timestamp"
    return df



def slice_window(df: pd.DataFrame, start_date: str, n_days: int) -> pd.DataFrame:
    start = pd.to_datetime(start_date)
    end = start + pd.Timedelta(days=int(n_days))
    w = df.loc[(df.index >= start) & (df.index < end)].copy()
    if w.empty:
        raise ValueError(
            f"No data in selected window.\n"
            f"start={start} end(exclusive)={end}\n"
            f"CSV time range: {df.index.min()} .. {df.index.max()}"
        )
    return w


def build_series(df: pd.DataFrame) -> dict[str, pd.Series]:
    col_pgrid = _require(_pick_first_existing(df, CAND_PGRID), "PGRID", df, CAND_PGRID)
    col_pbess = _require(_pick_first_existing(df, CAND_PBESS), "PBESS", df, CAND_PBESS)
    col_pev = _require(_pick_first_existing(df, CAND_PEV), "PEV", df, CAND_PEV)
    col_pload = _require(_pick_first_existing(df, CAND_PLOAD), "PLOAD", df, CAND_PLOAD)
    col_pv_used = _require(_pick_first_existing(df, CAND_PV_USED), "PPV", df, CAND_PV_USED)

    col_xpv = _pick_first_existing(df, CAND_XPV)

    pv_used = df[col_pv_used].astype(float)
    if col_xpv is not None:
        xpv = df[col_xpv].astype(float).clip(0.0, 1.0)
        denom = (1.0 - xpv).replace(0.0, np.nan)
        pv_available = (pv_used / denom).fillna(pv_used)
    else:
        xpv = pd.Series(0.0, index=pv_used.index)
        pv_available = pv_used

    col_soc_bess = _pick_first_existing(df, CAND_SOC_BESS)
    col_soc_ev = _pick_first_existing(df, CAND_SOC_EV)
    col_ebess = _pick_first_existing(df, CAND_EBESS)
    col_eev = _pick_first_existing(df, CAND_EEV)

    if col_soc_bess is not None:
        soc_bess = df[col_soc_bess].astype(float)
    elif col_ebess is not None:
        soc_bess = df[col_ebess].astype(float)
    else:
        raise KeyError("Coluna de SoC/energia do BESS não encontrada.")

    if col_soc_ev is not None:
        soc_ev = df[col_soc_ev].astype(float)
    elif col_eev is not None:
        soc_ev = df[col_eev].astype(float)
    else:
        raise KeyError("Coluna de SoC/energia do EV não encontrada.")

    return {
        "PV_available": pv_available.astype(float),
        "PV_used_env": pv_used,
        "PV_times_(1-XPV)": (pv_available * (1.0 - xpv)).astype(float),
        "PGRID": df[col_pgrid].astype(float),
        "PBESS_env": df[col_pbess].astype(float),
        "PEV_env": df[col_pev].astype(float),
        "PLOAD": df[col_pload].astype(float),
        "XPV": xpv.astype(float),
        "SoC_BESS": soc_bess.astype(float),
        "SoC_EV": soc_ev.astype(float),
    }


def build_output_path(tariff: str, model: str, role: str, run_label: str, start: pd.Timestamp, n_days: int) -> Path:
    fig_root = Path("Figures") / str(tariff)
    fig_root.mkdir(parents=True, exist_ok=True)

    day_start = start.strftime("%Y-%m-%d")
    day_end = (start + pd.Timedelta(days=int(n_days)) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    fname = f"{model}-{role}-{tariff}-{run_label}-{day_start}-{day_end}.pdf"
    return fig_root / fname


def plot_and_save(series: dict[str, pd.Series], out_pdf: Path, title: str):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=FIGSIZE, sharex=True)

    ax1.plot(series["PV_available"].index, series["PV_available"].values, label="PV (available)")
    ax1.plot(series["PV_used_env"].index, series["PV_used_env"].values, label="PV used (env)")
    ax1.plot(series["PV_times_(1-XPV)"].index, series["PV_times_(1-XPV)"].values, label="PV * (1 - XPV)")
    ax1.plot(series["PGRID"].index, series["PGRID"].values, label="PGRID")
    ax1.plot(series["PBESS_env"].index, series["PBESS_env"].values, label="PBESS")
    ax1.plot(series["PEV_env"].index, series["PEV_env"].values, label="PEV")
    ax1.plot(series["PLOAD"].index, series["PLOAD"].values, label="PLOAD")
    ax1.set_ylabel("Power (kW)")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="best")

    ax2.plot(series["SoC_BESS"].index, series["SoC_BESS"].values, label="SoC BESS")
    ax2.plot(series["SoC_EV"].index, series["SoC_EV"].values, label="SoC EV")
    ax2.set_ylabel("State of Charge")
    ax2.set_xlabel("Time")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="best")

    fig.suptitle(title)
    fig.tight_layout()

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved: {out_pdf}")


def main():
    validate_config()

    n = len(MODELS)
    tariffs = _expand_list(TARIFFS, n)
    run_labels = _expand_list(RUN_LABELS, n)
    start_dates = _expand_list(START_DATES, n)
    n_days_list = _expand_list(N_DAYS if isinstance(N_DAYS, list) else [N_DAYS], n)

    for model, tariff, run_label, start_date, nd in zip(MODELS, tariffs, run_labels, start_dates, n_days_list):
        for role in ("actor", "teacher"):
            csv_path = build_csv_path(RESULTS_ROOT, model, tariff, run_label, role)
            if not csv_path.exists():
                print(f"[WARN] CSV não encontrado para {role}: {csv_path}")
                continue

            df = load_csv(csv_path)
            w = slice_window(df, start_date, nd)
            series = build_series(w)

            start = pd.to_datetime(start_date)
            out_pdf = build_output_path(tariff, model, role, run_label, start, nd)
            title = (
                f"{model} | {role.upper()} | tariff={tariff} | data={run_label} "
                f"| {start.strftime('%Y-%m-%d')} +{nd}d"
            )

            plot_and_save(series, out_pdf=out_pdf, title=title)


if __name__ == "__main__":
    main()
