from pathlib import Path

import pandas as pd


def _format_pvalue(p: float) -> str:
    if p < 1e-4:
        return f"{p:.2e}"
    return f"{p:.4f}"


def _add_report_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["RL-IL mean"] = out["diff_mean_rl_minus_il"].map(lambda x: f"{x:.1f}")
    out["RL-IL median"] = out["diff_median_rl_minus_il"].map(lambda x: f"{x:.1f}")
    out["CI95% (mean diff)"] = out.apply(
        lambda row: f"[{row['ci95_low_diff_mean']:.1f}, {row['ci95_high_diff_mean']:.1f}]", axis=1
    )
    out["p-value (Wilcoxon)"] = out["wilcoxon_pvalue_two_sided"].map(_format_pvalue)
    out["sig@5%"] = out["significant_0_05"].map(lambda x: "yes" if int(x) == 1 else "no")
    return out


def _save_latex_table(df: pd.DataFrame, path: Path, caption: str, label: str) -> None:
    latex = df.to_latex(
        index=False,
        escape=True,
        caption=caption,
        label=label,
        column_format="l" + "r" * (len(df.columns) - 1),
    )
    path.write_text(latex, encoding="utf-8")


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    out_dir = project_root / "Results" / "statistical_tests"

    by_tariff = pd.read_csv(out_dir / "bootstrap_wilcoxon_rl_vs_il_by_tariff.csv")
    by_model_tariff = pd.read_csv(out_dir / "bootstrap_wilcoxon_rl_vs_il_by_model_tariff.csv")
    overall = pd.read_csv(out_dir / "bootstrap_wilcoxon_rl_vs_il_overall.csv")

    by_tariff = _add_report_columns(by_tariff)
    by_model_tariff = _add_report_columns(by_model_tariff)
    overall = _add_report_columns(overall)

    model_summary = (
        by_model_tariff.groupby("model", as_index=False)
        .agg(
            n_pairs_total=("n_pairs", "sum"),
            mean_diff_rl_minus_il=("diff_mean_rl_minus_il", "mean"),
            median_diff_rl_minus_il=("diff_median_rl_minus_il", "mean"),
            significant_pairs=("significant_0_05", "sum"),
        )
        .sort_values("mean_diff_rl_minus_il", ascending=False)
        .reset_index(drop=True)
    )

    tariff_consistency = (
        by_model_tariff.assign(positive_mean=by_model_tariff["diff_mean_rl_minus_il"] > 0)
        .groupby("tariff", as_index=False)
        .agg(
            models=("model", "count"),
            models_with_positive_mean=("positive_mean", "sum"),
            models_significant=("significant_0_05", "sum"),
        )
    )

    model_summary_report = model_summary.copy()
    model_summary_report["Mean RL-IL"] = model_summary_report["mean_diff_rl_minus_il"].map(lambda x: f"{x:.1f}")
    model_summary_report["Median RL-IL"] = model_summary_report["median_diff_rl_minus_il"].map(lambda x: f"{x:.1f}")
    model_summary_report = model_summary_report[
        ["model", "n_pairs_total", "Mean RL-IL", "Median RL-IL", "significant_pairs"]
    ].rename(
        columns={
            "model": "Model",
            "n_pairs_total": "N",
            "significant_pairs": "Significant pairs (5%)",
        }
    )

    tariff_consistency_report = tariff_consistency.rename(
        columns={
            "tariff": "Tariff",
            "models": "Models",
            "models_with_positive_mean": "Models with RL-IL > 0",
            "models_significant": "Significant models (5%)",
        }
    )

    by_tariff_report = by_tariff[
        [
            "tariff",
            "n_pairs",
            "RL-IL mean",
            "RL-IL median",
            "CI95% (mean diff)",
            "p-value (Wilcoxon)",
            "sig@5%",
            "wins_rl",
            "wins_il",
        ]
    ].rename(columns={"tariff": "Tariff", "n_pairs": "N", "wins_rl": "RL wins", "wins_il": "IL wins"})

    by_model_tariff_report = by_model_tariff[
        [
            "model",
            "tariff",
            "n_pairs",
            "RL-IL mean",
            "RL-IL median",
            "CI95% (mean diff)",
            "p-value (Wilcoxon)",
            "sig@5%",
        ]
    ].rename(columns={"model": "Model", "tariff": "Tariff", "n_pairs": "N"})

    by_model_tariff_report = by_model_tariff_report.sort_values(["Tariff", "Model"]).reset_index(drop=True)

    overall_report = overall[
        [
            "scope",
            "n_pairs",
            "RL-IL mean",
            "RL-IL median",
            "CI95% (mean diff)",
            "p-value (Wilcoxon)",
            "sig@5%",
            "wins_rl",
            "wins_il",
        ]
    ].rename(columns={"scope": "Scope", "n_pairs": "N", "wins_rl": "RL wins", "wins_il": "IL wins"})

    by_tariff_report.to_csv(out_dir / "bootstrap_wilcoxon_report_by_tariff.csv", index=False)
    by_model_tariff_report.to_csv(out_dir / "bootstrap_wilcoxon_report_by_model_tariff.csv", index=False)
    overall_report.to_csv(out_dir / "bootstrap_wilcoxon_report_overall.csv", index=False)
    model_summary_report.to_csv(out_dir / "bootstrap_wilcoxon_report_by_model_summary.csv", index=False)
    tariff_consistency_report.to_csv(out_dir / "bootstrap_wilcoxon_report_tariff_consistency.csv", index=False)

    _save_latex_table(
        by_tariff_report,
        out_dir / "bootstrap_wilcoxon_report_by_tariff.tex",
        "Bootstrap CI and paired Wilcoxon test for RL vs IL by tariff.",
        "tab:bootstrap_wilcoxon_by_tariff",
    )

    _save_latex_table(
        by_model_tariff_report,
        out_dir / "bootstrap_wilcoxon_report_by_model_tariff.tex",
        "Bootstrap CI and paired Wilcoxon test for RL vs IL by model and tariff.",
        "tab:bootstrap_wilcoxon_by_model_tariff",
    )

    _save_latex_table(
        overall_report,
        out_dir / "bootstrap_wilcoxon_report_overall.tex",
        "Bootstrap CI and paired Wilcoxon test for RL vs IL over all models and tariffs.",
        "tab:bootstrap_wilcoxon_overall",
    )

    _save_latex_table(
        model_summary_report,
        out_dir / "bootstrap_wilcoxon_report_by_model_summary.tex",
        "Aggregated RL vs IL summary by model over all tariffs.",
        "tab:bootstrap_wilcoxon_by_model_summary",
    )

    _save_latex_table(
        tariff_consistency_report,
        out_dir / "bootstrap_wilcoxon_report_tariff_consistency.tex",
        "Consistency summary by tariff across models.",
        "tab:bootstrap_wilcoxon_tariff_consistency",
    )

    print("Saved report tables:")
    print(out_dir / "bootstrap_wilcoxon_report_by_tariff.csv")
    print(out_dir / "bootstrap_wilcoxon_report_by_model_tariff.csv")
    print(out_dir / "bootstrap_wilcoxon_report_overall.csv")
    print(out_dir / "bootstrap_wilcoxon_report_by_tariff.tex")
    print(out_dir / "bootstrap_wilcoxon_report_by_model_tariff.tex")
    print(out_dir / "bootstrap_wilcoxon_report_overall.tex")
    print(out_dir / "bootstrap_wilcoxon_report_by_model_summary.csv")
    print(out_dir / "bootstrap_wilcoxon_report_tariff_consistency.csv")
    print(out_dir / "bootstrap_wilcoxon_report_by_model_summary.tex")
    print(out_dir / "bootstrap_wilcoxon_report_tariff_consistency.tex")


if __name__ == "__main__":
    main()
