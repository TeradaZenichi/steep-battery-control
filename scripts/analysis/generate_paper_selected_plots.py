from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import dates as mdates
from matplotlib import font_manager
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = PROJECT_ROOT / "Results"
TEST_ROOT = RESULTS_ROOT / "test"
OUT_DIR = RESULTS_ROOT / "figures" / "paper_selected"

DPI = 300
FIGSIZE = (7.0, 3.0)

TARIFF_LABELS = {
    "tar_flat": "Flat",
    "tar_tou": "ToU",
    "tar_s": "Solar",
    "tar_w": "Wind",
    "tar_sw": "Solar + wind",
}

MODEL_LABELS = {
    "MLP": "MLP-S",
    "MLPv2": "MLP-H",
    "GRU": "GRU-H",
    "GRUv2": "GRU-AC",
    "ATT": "Trf-H",
    "ATTv2": "Trf-AC",
    "TCN": "TCN-H",
    "TCNv2": "TCN-AC",
    "ATT_MEM": "AM-H",
    "ATT_MEMv2": "AM-AC",
}


@dataclass(frozen=True)
class OperationFigure:
    number: int
    slug: str
    model: str
    phase: str
    tariff: str
    case: str
    day: str
    variant: str | None
    ylim: tuple[float, float]
    reason: str

    @property
    def csv_path(self) -> Path:
        if self.phase == "1-IL":
            name = f"{self.case}_actor_env_operation.csv"
        else:
            name = f"{self.case}_actor_env_operation_{self.variant}.csv"
        return TEST_ROOT / self.model / self.phase / self.tariff / name

    @property
    def policy_label(self) -> str:
        method = "IL" if self.phase == "1-IL" else "RL"
        arch = MODEL_LABELS[self.model]
        if self.variant is None:
            return f"{method} {arch}"
        return f"{method} {arch} ({self.variant})"


@dataclass(frozen=True)
class VariantFigure:
    number: int
    slug: str
    model: str
    tariff: str
    case: str
    day: str
    ylim: tuple[float, float]
    reason: str

    def csv_path(self, variant: str) -> Path:
        name = f"{self.case}_actor_env_operation_{variant}.csv"
        return TEST_ROOT / self.model / "2-RL" / self.tariff / name


OPERATION_FIGURES = [
    OperationFigure(1, "flat_best_gru_ac_det", "GRUv2", "2-RL", "tar_flat", "test_wy_07", "2000-07-12", "det", (-2.2, 4.4), "Melhor politica para Flat; o mesmo caso/dia tambem e rank 1 nas demais tarifas."),
    OperationFigure(2, "tou_best_gru_h_det", "GRU", "2-RL", "tar_tou", "test_wy_07", "2000-07-12", "det", (-2.2, 4.4), "Excecao principal ao melhor modelo global: GRU-H det vence em ToU."),
    OperationFigure(3, "solar_best_gru_ac_det", "GRUv2", "2-RL", "tar_s", "test_wy_07", "2000-07-12", "det", (-2.2, 4.4), "Melhor politica para Solar, mantendo o mesmo dia para comparacao direta."),
    OperationFigure(4, "wind_best_gru_ac_det", "GRUv2", "2-RL", "tar_w", "test_wy_07", "2000-07-12", "det", (-2.2, 4.4), "Melhor politica para Wind no mesmo caso operacional das figuras principais."),
    OperationFigure(5, "solar_wind_best_gru_ac_det", "GRUv2", "2-RL", "tar_sw", "test_wy_07", "2000-07-12", "det", (-2.2, 4.4), "Melhor politica para Solar + wind, fechando o bloco comparavel entre tarifas."),
    OperationFigure(6, "flat_il_mlp_s_competitive", "MLP", "1-IL", "tar_flat", "test_wy_07", "2000-07-12", None, (-3.5, 5.8), "Caso em que IL MLP-S e competitivo em Flat e pode ser comparado diretamente com a Figura 1."),
    OperationFigure(8, "wind_tcn_h_combo_strong", "TCN", "2-RL", "tar_w", "test_wy_07", "2000-07-12", "combo", (-2.0, 6.0), "TCN-H combo aparece como alternativa forte em Wind, evitando uma narrativa centrada apenas em GRU."),
    OperationFigure(9, "tou_trf_ac_det_strong", "ATTv2", "2-RL", "tar_tou", "test_wy_05", "2000-05-31", "det", (-2.2, 4.2), "Caso favoravel ao Transformer Actor-Critic em ToU, apesar de nao ser o melhor medio."),
    OperationFigure(10, "tou_il_am_h_failure", "ATT_MEM", "1-IL", "tar_tou", "test_cy_03", "2000-03-25", None, (-6.0, 11.0), "Falha clara de IL AM-H em ToU, com grandes variacoes de potencia e penalidades."),
]

VARIANT_FIGURE = VariantFigure(
    7,
    "solar_gru_ac_variant_comparison",
    "GRUv2",
    "tar_s",
    "test_cy_08",
    "2000-08-09",
    (-3.0, 6.5),
    "Comparacao das variantes det, combo e stoch para GRU-AC em Solar; det tem reward muito superior neste caso.",
)


def configure_matplotlib() -> None:
    font_path = PROJECT_ROOT / "data" / "Gulliver.otf"
    if font_path.exists():
        font_manager.fontManager.addfont(str(font_path))
        prop = font_manager.FontProperties(fname=str(font_path))
        plt.rcParams["font.family"] = "Gulliver"
        plt.rcParams["font.sans-serif"] = prop.get_name()
    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 7,
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def read_operation(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path, parse_dates=["timestamp"], index_col="timestamp").sort_index()
    return df


def slice_day(df: pd.DataFrame, day: str) -> pd.DataFrame:
    start = pd.Timestamp(day)
    end = start + pd.Timedelta(days=1)
    out = df.loc[(df.index >= start) & (df.index < end)].copy()
    if out.empty:
        raise ValueError(f"No data for {day}; range is {df.index.min()} to {df.index.max()}")
    return out


def add_opaque_legend(ax, *args, **kwargs):
    leg = ax.legend(*args, frameon=True, fancybox=False, framealpha=1.0, **kwargs)
    frame = leg.get_frame()
    frame.set_facecolor("white")
    frame.set_edgecolor("0.7")
    frame.set_alpha(1.0)
    leg.set_zorder(20)
    return leg


def align_tariff_axis(ax_power, ax_tariff, tariff: pd.Series) -> None:
    left_min, left_max = ax_power.get_ylim()
    zero_frac = (0.0 - left_min) / (left_max - left_min)
    t_max = max(float(tariff.max()) * 1.08, 0.01)
    if 0.0 < zero_frac < 1.0:
        t_min = -zero_frac * t_max / (1.0 - zero_frac)
    else:
        t_min = min(0.0, float(tariff.min()) * 0.95)
    ax_tariff.set_ylim(t_min, t_max)


def plot_tariff_axis(ax_power, df: pd.DataFrame):
    ax_tariff = ax_power.twinx()
    ax_tariff.step(df.index, df["tariff"].astype(float), where="post", color="0.15", linewidth=1.0, label="Tariff")
    ax_tariff.set_ylabel("Tariff ($/kWh)")
    align_tariff_axis(ax_power, ax_tariff, df["tariff"].astype(float))
    ax_power.patch.set_visible(False)
    ax_tariff.set_zorder(ax_power.get_zorder() - 1)
    ax_tariff.patch.set_visible(False)
    return ax_tariff


def add_top_legends(ax_power, ax_tariff, power_kwargs: dict) -> None:
    power_handles, power_labels = ax_power.get_legend_handles_labels()
    tariff_handles, tariff_labels = ax_tariff.get_legend_handles_labels()
    power_leg = add_opaque_legend(ax_tariff, power_handles, power_labels, **power_kwargs)
    ax_tariff.add_artist(power_leg)
    add_opaque_legend(ax_tariff, tariff_handles, tariff_labels, loc="upper right")


def plot_operation(figspec: OperationFigure) -> None:
    df = slice_day(read_operation(figspec.csv_path), figspec.day)
    fig, (ax0, ax1) = plt.subplots(
        2,
        1,
        figsize=FIGSIZE,
        sharex=True,
        gridspec_kw={"height_ratios": [3.0, 0.9], "hspace": 0.08},
    )

    colors = {
        "PPV": "#2ca02c",
        "PV avail.": "#2ca02c",
        "PLoad": "#d62728",
        "PGrid": "#6f4aa2",
        "PBESS": "#1f77b4",
        "PEV": "#ff7f0e",
        "SoCBESS": "#1f77b4",
        "SoCEV": "#ff7f0e",
    }

    curtail_col = "χPV" if "χPV" in df.columns else "Ï‡PV"
    if curtail_col in df.columns:
        denom = (1.0 - df[curtail_col].astype(float)).replace(0.0, np.nan)
        pv_available = (df["PPV"].astype(float) / denom).fillna(df["PPV"].astype(float))
        ax0.step(df.index, pv_available, where="post", label="PV avail.", color=colors["PV avail."], linestyle="--", linewidth=1.0)

    for col in ["PPV", "PLoad", "PGrid"]:
        ax0.step(df.index, df[col].astype(float), where="post", label=col, color=colors[col], linewidth=1.0)

    if len(df.index) >= 2:
        bar_width = (df.index[1] - df.index[0]).total_seconds() / 86400.0 * 0.8
    else:
        bar_width = 0.003
    for col in ["PBESS", "PEV"]:
        ax0.bar(
            df.index,
            df[col].astype(float),
            width=bar_width,
            label=col,
            color=colors[col],
            alpha=0.65,
            align="edge",
            linewidth=0.0,
        )

    ax0.axhline(0.0, color="0.25", linewidth=0.8)
    ax0.set_ylabel("Power (kW)")
    ax0.set_ylim(*figspec.ylim)
    ax0.grid(True, alpha=0.25, linewidth=0.6)
    ax0.set_axisbelow(True)
    ax_tariff = plot_tariff_axis(ax0, df)
    add_top_legends(ax0, ax_tariff, {"loc": "upper left", "ncol": 3, "columnspacing": 0.8, "handlelength": 1.4})

    for col in ["SoCBESS", "SoCEV"]:
        ax1.step(df.index, df[col].astype(float), where="post", label=col, color=colors[col], linewidth=1.0)
    ax1.set_ylabel("SoC")
    ax1.set_ylim(-0.05, 1.05)
    ax1.grid(True, alpha=0.25, linewidth=0.6)
    ax1.set_xlabel("Time")
    add_opaque_legend(ax1, loc="upper right", ncol=2, columnspacing=0.8, handlelength=1.4)

    ax1.xaxis.set_major_locator(mdates.HourLocator(byhour=[0, 6, 12, 18, 24]))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    fig.align_ylabels([ax0, ax1])
    save_figure(fig, figspec.number, figspec.slug)


def plot_variant_comparison(figspec: VariantFigure) -> None:
    frames = {variant: slice_day(read_operation(figspec.csv_path(variant)), figspec.day) for variant in ["det", "combo", "stoch"]}
    fig, (ax0, ax1) = plt.subplots(
        2,
        1,
        figsize=FIGSIZE,
        sharex=True,
        gridspec_kw={"height_ratios": [3.0, 0.9], "hspace": 0.08},
    )
    colors = {"det": "#1f77b4", "combo": "#2ca02c", "stoch": "#d62728"}

    ref = frames["det"]
    ax0.step(ref.index, ref["PLoad"].astype(float), where="post", color="0.55", linewidth=0.9, linestyle="--", label="PLoad")
    ax0.step(ref.index, ref["PPV"].astype(float), where="post", color="0.25", linewidth=0.9, linestyle=":", label="PPV")
    for variant, df in frames.items():
        ax0.step(df.index, df["PGrid"].astype(float), where="post", color=colors[variant], linewidth=1.1, label=f"PGrid {variant}")

    ax0.axhline(0.0, color="0.25", linewidth=0.8)
    ax0.set_ylabel("Power (kW)")
    ax0.set_ylim(*figspec.ylim)
    ax0.grid(True, alpha=0.25, linewidth=0.6)
    ax0.set_axisbelow(True)
    ax_tariff = plot_tariff_axis(ax0, ref)
    add_top_legends(ax0, ax_tariff, {"loc": "upper left", "ncol": 3, "columnspacing": 0.8, "handlelength": 1.4})

    for variant, df in frames.items():
        ax1.step(df.index, df["SoCBESS"].astype(float), where="post", color=colors[variant], linewidth=1.0, label=f"BESS {variant}")
    ax1.set_ylabel("SoC")
    ax1.set_ylim(-0.05, 1.05)
    ax1.grid(True, alpha=0.25, linewidth=0.6)
    ax1.set_xlabel("Time")
    add_opaque_legend(ax1, loc="upper right", ncol=3, columnspacing=0.8, handlelength=1.4)

    ax1.xaxis.set_major_locator(mdates.HourLocator(byhour=[0, 6, 12, 18, 24]))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    fig.align_ylabels([ax0, ax1])
    save_figure(fig, figspec.number, figspec.slug)


def save_figure(fig, number: int, slug: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"{number:02d}_{slug}"
    for ext in ["pdf", "png"]:
        fig.savefig(OUT_DIR / f"{stem}.{ext}", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def clean_old_figures() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for pattern in ["[0-9][0-9]_*.pdf", "[0-9][0-9]_*.png"]:
        for path in OUT_DIR.glob(pattern):
            path.unlink()


def write_summary() -> None:
    rows = []
    for spec in OPERATION_FIGURES:
        rows.append(
            (
                spec.number,
                TARIFF_LABELS[spec.tariff],
                spec.policy_label,
                spec.case,
                spec.day,
                spec.reason,
            )
        )
    rows.append(
        (
            VARIANT_FIGURE.number,
            TARIFF_LABELS[VARIANT_FIGURE.tariff],
            "RL GRU-AC (det/combo/stoch)",
            VARIANT_FIGURE.case,
            VARIANT_FIGURE.day,
            VARIANT_FIGURE.reason,
        )
    )
    rows = sorted(rows, key=lambda x: x[0])

    lines = [
        "# Figuras selecionadas para o artigo",
        "",
        "As figuras foram refeitas para sustentar uma narrativa em tres partes: primeiro, o melhor modelo por tarifa em um mesmo dia; depois, casos complementares que mostram IL competitivo, variacao entre politicas e falhas informativas.",
        "",
        "## Criterio geral",
        "",
        "- As Figuras 1--5 usam o mesmo caso (`test_wy_07`) e o mesmo dia (`2000-07-12`) para tornar a comparacao entre tarifas visualmente justa.",
        "- Para cada tarifa, foi usada a politica com maior reward medio no conjunto de teste.",
        "- As Figuras 6--10 foram escolhidas por insight: comparacao com IL, diferenca entre variantes, modelos alternativos fortes e um caso de falha clara.",
        "",
        "## Lista das figuras",
        "",
    ]
    for number, tariff, policy, case, day, reason in rows:
        lines.extend(
            [
                f"### Figura {number:02d}",
                "",
                f"- Tarifa: {tariff}",
                f"- Politica: {policy}",
                f"- Caso e dia: `{case}`, `{day}`",
                f"- Criterio: {reason}",
                "",
            ]
        )
    (OUT_DIR / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    configure_matplotlib()
    clean_old_figures()
    for spec in sorted(OPERATION_FIGURES, key=lambda x: x.number):
        plot_operation(spec)
    plot_variant_comparison(VARIANT_FIGURE)
    write_summary()


if __name__ == "__main__":
    main()
