# -*- coding: utf-8 -*-
"""
Forest plot generation script for lung nodule classification model comparison.
"""

import os
import re
import sys
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

# Add local path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Fix Windows terminal encoding
import encoding_fix

encoding_fix.fix_windows_encoding()

import config


MODALITY_ORDER = ["Tongue", "Pulse", "Fusion"]
REQUIRED_COLUMNS = {"Modality", "Model", "AUC (95% CI)"}
MODEL_SHORT_NAME = {
    "Logistic Regression": "LR",
    "Random Forest": "RF",
    "XGBoost": "XGBoost",
    "ANN": "ANN",
    "SVM": "SVM",
}
MODALITY_LABEL = {
    "Tongue": "Tongue-only",
    "Pulse": "Pulse-only",
    "Fusion": "Fusion (Tongue+Pulse)",
}


def get_plot_style() -> Dict[str, object]:
    """Publication-style plotting configuration."""
    return {
        "palette": {
            "Tongue": "#0072B2",  # Blue (Okabe-Ito)
            "Pulse": "#009E73",  # Green (Okabe-Ito)
            "Fusion": "#D55E00",  # Vermillion (Okabe-Ito)
        },
        "font_family": "Times New Roman",
        "title_size": 16,
        "label_size": 14,
        "tick_size": 12,
        "y_tick_size": 12,
        "ci_text_size": 12,
        "dot_size": 7.5,
        "line_width": 2.0,
        "cap_size": 4.0,
        "cap_thick": 1.4,
        "tick_step": 0.02,
        "x_padding_ratio": 0.12,
        "x_min_span": 0.12,
        "x_bounds": (0.5, 1.0),
        "text_col_pad": 0.012,
        "text_col_width": 0.075,
        "separator_gap": 0.75,
        "separator_alpha": 0.35,
        "grid_alpha": 0.18,
        "grid_style": "--",
        "figsize_best": (10.8, 5.2),
        "figsize_all": (13.2, 11.0),
        "reference_lines": [0.70, 0.80],
        "save_facecolor": "white",
        "legend_ncol": 3,
        "x_label_pad": 14,
        "modality_group_x": -0.17,
        "modality_group_size": 14,
        "best_row_step": 0.52,
        "best_top_offset": 0.48,
        "all_top_offset": 0.58,
        "best_header_y_offset": 0.18,
        "all_header_y_offset": 0.45,
    }


def format_model_name(model_name: str) -> str:
    """Convert long model names to compact labels."""
    return MODEL_SHORT_NAME.get(str(model_name), str(model_name))


def format_modality_label(modality: str, short: bool = False) -> str:
    """Format modality labels for axis text and group headers."""
    if short:
        return str(modality)
    return MODALITY_LABEL.get(str(modality), str(modality))


def build_ci_text(row: pd.Series) -> str:
    """Build right-side value text."""
    return f"{row['AUC']:.3f} ({row['CI_lower']:.3f}-{row['CI_upper']:.3f})"


def parse_auc_ci(auc_str: str) -> Tuple[float, float, float]:
    """
    Parse AUC (95% CI) strings.
    Accepted formats include:
    - 0.754 (0.719-0.790)
    - 0.754 (0.719–0.790)
    """
    pattern = r"([\d.]+)\s*\(([\d.]+)[^\d]+([\d.]+)\)"
    match = re.search(pattern, str(auc_str))
    if match:
        return float(match.group(1)), float(match.group(2)), float(match.group(3))
    return np.nan, np.nan, np.nan


def validate_input_columns(df: pd.DataFrame) -> None:
    """Validate mandatory columns in input data."""
    missing = sorted(REQUIRED_COLUMNS - set(df.columns))
    if missing:
        raise ValueError(
            "Missing required columns in Excel: "
            + ", ".join(missing)
            + ". Required columns are: Modality, Model, AUC (95% CI)."
        )


def load_and_prepare_data(excel_path: str) -> pd.DataFrame:
    """Load Excel data and parse AUC/CI columns."""
    print(f"Loading data: {excel_path}")
    df = pd.read_excel(excel_path)
    validate_input_columns(df)

    df[["AUC", "CI_lower", "CI_upper"]] = df["AUC (95% CI)"].apply(
        lambda x: pd.Series(parse_auc_ci(x))
    )
    df["Full_Name"] = df["Modality"].astype(str) + "-" + df["Model"].astype(str)

    if df[["AUC", "CI_lower", "CI_upper"]].isnull().any().any():
        failed_rows = df[df["AUC"].isnull()][["Modality", "Model", "AUC (95% CI)"]]
        raise ValueError(
            "AUC (95% CI) parsing failed for some rows. Please check format like "
            "'0.754 (0.719-0.790)'.\nFailed rows:\n"
            + failed_rows.to_string(index=False)
        )

    print(f"Loaded rows: {len(df)}")
    print(f"Modality distribution: {df['Modality'].value_counts().to_dict()}")
    return df


def select_best_models(df: pd.DataFrame) -> pd.DataFrame:
    """Pick the best model (highest AUC) from each modality."""
    best_rows = []
    for modality in MODALITY_ORDER:
        modality_df = df[df["Modality"] == modality]
        if modality_df.empty:
            raise ValueError(f"No rows found for modality '{modality}'.")
        best_idx = modality_df["AUC"].idxmax()
        best_rows.append(df.loc[best_idx])

    result = pd.DataFrame(best_rows).sort_values("AUC", ascending=False).reset_index(
        drop=True
    )

    print("\nBest model per modality:")
    for _, row in result.iterrows():
        print(
            f"  {row['Modality']}: {row['Model']} - "
            f"AUC={row['AUC']:.3f} ({row['CI_lower']:.3f}-{row['CI_upper']:.3f})"
        )
    return result


def compute_axis_limits(
    df: pd.DataFrame,
    padding_ratio: float = 0.12,
    min_span: float = 0.12,
    bounds: Tuple[float, float] = (0.5, 1.0),
) -> Tuple[float, float]:
    """Compute dynamic x-axis limits from CI range."""
    x_min_raw = float(df["CI_lower"].min())
    x_max_raw = float(df["CI_upper"].max())
    span = max(x_max_raw - x_min_raw, min_span)
    pad = span * padding_ratio

    x_min = max(bounds[0], x_min_raw - pad)
    x_max = min(bounds[1], x_max_raw + pad)

    if x_max - x_min < min_span:
        center = (x_min + x_max) / 2
        half = min_span / 2
        x_min = max(bounds[0], center - half)
        x_max = min(bounds[1], center + half)

    return x_min, x_max


def compute_xticks(x_min: float, x_max: float, step: float) -> np.ndarray:
    """Compute evenly spaced x ticks in data area."""
    start = np.floor(x_min / step) * step
    end = np.ceil(x_max / step) * step
    ticks = np.arange(start, end + step / 2, step)
    return np.round(ticks, 2)


def prepare_all_models_for_plot(
    df: pd.DataFrame, sort_mode: str, modality_order: Sequence[str], separator_gap: float
) -> Tuple[pd.DataFrame, List[Dict[str, float]]]:
    """
    Prepare sorted rows and y positions for the all-models plot.
    Returns:
    - plotting dataframe with `plot_y`
    - group metadata (start/end/center positions)
    """
    if sort_mode not in {"grouped", "global"}:
        raise ValueError("sort_mode must be 'grouped' or 'global'.")

    if sort_mode == "global":
        df_plot = df.sort_values("AUC", ascending=False).reset_index(drop=True).copy()
        df_plot["plot_y"] = np.arange(len(df_plot), dtype=float)
        return df_plot, []

    rows = []
    group_meta: List[Dict[str, float]] = []
    y_cursor = 0.0

    known = [m for m in modality_order if m in set(df["Modality"])]
    others = [m for m in sorted(df["Modality"].unique()) if m not in known]
    modality_sequence = known + others

    for modality in modality_sequence:
        part = df[df["Modality"] == modality].sort_values("AUC", ascending=False).copy()
        if part.empty:
            continue
        start = y_cursor
        part["plot_y"] = np.arange(start, start + len(part), dtype=float)
        y_cursor += len(part)
        end = y_cursor - 1.0
        center = (start + end) / 2.0

        group_meta.append({"modality": modality, "start": start, "end": end, "center": center})
        rows.append(part)
        y_cursor += separator_gap

    df_plot = pd.concat(rows, axis=0).reset_index(drop=True)
    return df_plot, group_meta


def save_figure(
    fig: plt.Figure, output_path: str, save_formats: Iterable[str], dpi: int, facecolor: str
) -> None:
    """Save figure in requested formats."""
    for fmt in save_formats:
        fmt = fmt.lower().strip()
        filepath = f"{output_path}.{fmt}"
        save_kwargs = {"bbox_inches": "tight", "facecolor": facecolor}
        if fmt == "png":
            save_kwargs["dpi"] = dpi
        fig.savefig(filepath, **save_kwargs)
        print(f"  [OK] Saved: {os.path.basename(filepath)}")


def apply_axes_base_style(ax: plt.Axes, style: Dict[str, object]) -> None:
    """Shared axis styling."""
    ax.grid(
        True,
        axis="x",
        alpha=float(style["grid_alpha"]),
        linestyle=str(style["grid_style"]),
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", labelsize=int(style["tick_size"]))
    ax.tick_params(axis="y", labelsize=int(style["y_tick_size"]))


def plot_forest_best_models(
    df: pd.DataFrame,
    output_path: str,
    style: Dict[str, object],
    save_formats: Iterable[str],
    dpi: int,
    show_ci_text: bool,
) -> None:
    """Plot forest chart for best model of each modality."""
    print("\n" + "=" * 70)
    print("Plotting best-model forest plot (3 rows)")
    print("=" * 70)

    df_plot = df.sort_values("AUC", ascending=False).reset_index(drop=True).copy()
    df_plot["plot_y"] = (
        np.arange(len(df_plot), dtype=float) * float(style["best_row_step"])
        + float(style["best_top_offset"])
    )

    x_min, x_data_max = compute_axis_limits(
        df_plot,
        padding_ratio=float(style["x_padding_ratio"]),
        min_span=float(style["x_min_span"]),
        bounds=tuple(style["x_bounds"]),
    )
    text_x = x_data_max + float(style["text_col_pad"])
    x_max = (
        text_x + float(style["text_col_width"])
        if show_ci_text
        else x_data_max + float(style["text_col_pad"])
    )

    with plt.rc_context({"font.family": style["font_family"], "axes.unicode_minus": False}):
        fig, ax = plt.subplots(figsize=tuple(style["figsize_best"]))

        y_labels = []
        for _, row in df_plot.iterrows():
            color = style["palette"].get(row["Modality"], "#4C4C4C")
            xerr = [[row["AUC"] - row["CI_lower"]], [row["CI_upper"] - row["AUC"]]]
            ax.errorbar(
                [row["AUC"]],
                [row["plot_y"]],
                xerr=xerr,
                fmt="o",
                color=color,
                ecolor=color,
                capsize=float(style["cap_size"]),
                capthick=float(style["cap_thick"]),
                markersize=float(style["dot_size"]),
                markeredgewidth=1.2,
                linewidth=float(style["line_width"]),
                alpha=0.96,
                zorder=3,
            )

            if show_ci_text:
                ax.text(
                    text_x,
                    row["plot_y"],
                    build_ci_text(row),
                    va="center",
                    ha="left",
                    fontsize=int(style["ci_text_size"]),
                    color="#202020",
                )

            y_labels.append(
                f"{format_modality_label(row['Modality'])}\n({format_model_name(row['Model'])})"
            )

        ax.set_yticks(df_plot["plot_y"].to_numpy())
        ax.set_yticklabels(y_labels)
        ax.invert_yaxis()
        ax.set_ylim(df_plot["plot_y"].max() + 0.16, df_plot["plot_y"].min() - 0.30)

        ax.set_xlim(x_min, x_max)
        ax.set_xticks(compute_xticks(x_min, x_data_max, step=float(style["tick_step"])))
        ax.set_xlabel(
            "ROC-AUC (10-fold CV)",
            fontsize=int(style["label_size"]),
            fontweight="bold",
            labelpad=float(style["x_label_pad"]),
        )
        ax.set_title(
            "Model Performance Comparison - Best Models per Modality",
            fontsize=int(style["title_size"]),
            fontweight="bold",
            pad=10,
        )

        for x_ref in style["reference_lines"]:
            if x_min <= x_ref <= x_data_max:
                ax.axvline(
                    x=x_ref,
                    color="#6E6E6E",
                    linestyle=":",
                    linewidth=1.1,
                    alpha=0.6,
                    zorder=1,
                )

        if show_ci_text:
            first_row_y = df_plot["plot_y"].min()
            header_y = first_row_y - float(style["best_header_y_offset"])
            ax.text(
                text_x,
                header_y,
                "AUC (95% CI)",
                ha="left",
                va="center",
                fontsize=int(style["ci_text_size"]),
                fontweight="bold",
                color="#202020",
            )

        apply_axes_base_style(ax, style)
        fig.subplots_adjust(left=0.31, right=0.97, top=0.89, bottom=0.17)
        save_figure(fig, output_path, save_formats, dpi=int(dpi), facecolor=style["save_facecolor"])
        plt.close(fig)

    print("[DONE] Best-model forest plot generated.\n")


def plot_forest_all_models(
    df: pd.DataFrame,
    output_path: str,
    style: Dict[str, object],
    save_formats: Iterable[str],
    dpi: int,
    sort_mode: str,
    show_ci_text: bool,
) -> None:
    """Plot forest chart for all models with grouped modality layout."""
    print("\n" + "=" * 70)
    print("Plotting all-model forest plot")
    print("=" * 70)

    df_plot, group_meta = prepare_all_models_for_plot(
        df=df,
        sort_mode=sort_mode,
        modality_order=MODALITY_ORDER,
        separator_gap=float(style["separator_gap"]),
    )
    all_top_offset = float(style["all_top_offset"])
    df_plot["plot_y"] = df_plot["plot_y"] + all_top_offset
    for meta in group_meta:
        meta["start"] += all_top_offset
        meta["end"] += all_top_offset
        meta["center"] += all_top_offset

    x_min, x_data_max = compute_axis_limits(
        df_plot,
        padding_ratio=float(style["x_padding_ratio"]),
        min_span=float(style["x_min_span"]),
        bounds=tuple(style["x_bounds"]),
    )
    text_x = x_data_max + float(style["text_col_pad"])
    x_max = (
        text_x + float(style["text_col_width"])
        if show_ci_text
        else x_data_max + float(style["text_col_pad"])
    )

    with plt.rc_context({"font.family": style["font_family"], "axes.unicode_minus": False}):
        fig, ax = plt.subplots(figsize=tuple(style["figsize_all"]))

        y_labels: List[str] = []
        for _, row in df_plot.iterrows():
            modality = row["Modality"]
            color = style["palette"].get(modality, "#4C4C4C")
            xerr = [[row["AUC"] - row["CI_lower"]], [row["CI_upper"] - row["AUC"]]]
            ax.errorbar(
                [row["AUC"]],
                [row["plot_y"]],
                xerr=xerr,
                fmt="o",
                color=color,
                ecolor=color,
                capsize=float(style["cap_size"]),
                capthick=float(style["cap_thick"]),
                markersize=float(style["dot_size"]) - 0.2,
                markeredgewidth=1.1,
                linewidth=float(style["line_width"]) - 0.1,
                alpha=0.95,
                zorder=3,
            )

            if show_ci_text:
                ax.text(
                    text_x,
                    row["plot_y"],
                    build_ci_text(row),
                    va="center",
                    ha="left",
                    fontsize=int(style["ci_text_size"]) - 1,
                    color="#202020",
                )

            if sort_mode == "grouped":
                y_labels.append(format_model_name(row["Model"]))
            else:
                y_labels.append(
                    f"{format_modality_label(modality, short=True)}-{format_model_name(row['Model'])}"
                )

        ax.set_yticks(df_plot["plot_y"].to_numpy())
        ax.set_yticklabels(y_labels)
        ax.invert_yaxis()
        ax.set_ylim(df_plot["plot_y"].max() + 0.65, df_plot["plot_y"].min() - 0.35)

        ax.set_xlim(x_min, x_max)
        ax.set_xticks(compute_xticks(x_min, x_data_max, step=float(style["tick_step"])))
        ax.set_xlabel(
            "ROC-AUC (10-fold CV)",
            fontsize=int(style["label_size"]),
            fontweight="bold",
            labelpad=float(style["x_label_pad"]),
        )
        ax.set_title(
            "Model Performance Comparison - All Models (15 models)",
            fontsize=int(style["title_size"]),
            fontweight="bold",
            pad=18,
        )

        for x_ref in style["reference_lines"]:
            if x_min <= x_ref <= x_data_max:
                ax.axvline(
                    x=x_ref,
                    color="#6E6E6E",
                    linestyle=":",
                    linewidth=1.1,
                    alpha=0.6,
                    zorder=1,
                )

        if sort_mode == "grouped" and group_meta:
            for idx, meta in enumerate(group_meta):
                if idx < len(group_meta) - 1:
                    y_sep = meta["end"] + float(style["separator_gap"]) / 2.0
                    ax.axhline(
                        y=y_sep,
                        color="#8A8A8A",
                        linewidth=1.0,
                        alpha=float(style["separator_alpha"]),
                    )

                ax.text(
                    float(style["modality_group_x"]),
                    meta["center"],
                    format_modality_label(meta["modality"], short=True),
                    transform=ax.get_yaxis_transform(),
                    ha="left",
                    va="center",
                    fontsize=int(style["modality_group_size"]),
                    fontweight="bold",
                    color=style["palette"].get(meta["modality"], "#4C4C4C"),
                    clip_on=False,
                )

        if show_ci_text:
            first_row_y = df_plot["plot_y"].min()
            header_y = first_row_y - float(style["all_header_y_offset"])
            ax.text(
                text_x,
                header_y,
                "AUC (95% CI)",
                ha="left",
                va="center",
                fontsize=int(style["ci_text_size"]),
                fontweight="bold",
                color="#202020",
            )

        legend_handles = [
            Line2D(
                [0],
                [0],
                color=style["palette"][modality],
                marker="o",
                linewidth=float(style["line_width"]),
                markersize=float(style["dot_size"]),
                label=format_modality_label(modality, short=True),
            )
            for modality in MODALITY_ORDER
            if modality in set(df_plot["Modality"])
        ]
        ax.legend(
            handles=legend_handles,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.075),
            frameon=False,
            ncol=int(style["legend_ncol"]),
            fontsize=int(style["tick_size"]),
        )

        apply_axes_base_style(ax, style)
        fig.subplots_adjust(left=0.30, right=0.97, top=0.91, bottom=0.18)
        save_figure(fig, output_path, save_formats, dpi=int(dpi), facecolor=style["save_facecolor"])
        plt.close(fig)

    print("[DONE] All-model forest plot generated.\n")


def generate_forest_plots(
    excel_path: str,
    output_dir: str,
    save_formats: Sequence[str] = ("png", "svg"),
    dpi: int = 600,
    sort_mode: str = "grouped",
    show_ci_text: bool = True,
) -> None:
    """
    Generate forest plots with publication-style defaults.

    Parameters:
    - excel_path: source Excel path
    - output_dir: directory for saved plots
    - save_formats: output image formats (e.g. ('png', 'svg'))
    - dpi: PNG resolution
    - sort_mode: 'grouped' or 'global' ordering for all-models plot
    - show_ci_text: whether to show right-side AUC (95% CI) text column
    """
    print("\n" + "=" * 70)
    print("Forest plot generation")
    print("=" * 70)

    os.makedirs(output_dir, exist_ok=True)
    style = get_plot_style()

    df = load_and_prepare_data(excel_path)
    df_best = select_best_models(df)

    output_best = os.path.join(output_dir, "Figure_ForestPlot_BestModels")
    plot_forest_best_models(
        df_best,
        output_best,
        style=style,
        save_formats=save_formats,
        dpi=dpi,
        show_ci_text=show_ci_text,
    )

    output_all = os.path.join(output_dir, "Figure_ForestPlot_AllModels")
    plot_forest_all_models(
        df,
        output_all,
        style=style,
        save_formats=save_formats,
        dpi=dpi,
        sort_mode=sort_mode,
        show_ci_text=show_ci_text,
    )

    print("\n" + "=" * 70)
    print("All forest plots generated.")
    print("=" * 70)
    print(f"\nOutput directory: {output_dir}")
    print("\nGenerated files:")
    print("  - Figure_ForestPlot_BestModels.png/svg")
    print("  - Figure_ForestPlot_AllModels.png/svg")


if __name__ == "__main__":
    EXCEL_PATH = sys.argv[1] if len(sys.argv) > 1 else config.RAW_DATA_PATH
    OUTPUT_DIR = os.path.join(config.OUTPUT_BASE_DIR, "forest_plots")

    try:
        generate_forest_plots(
            EXCEL_PATH,
            OUTPUT_DIR,
            save_formats=("png", "svg"),
            dpi=600,
            sort_mode="grouped",
            show_ci_text=True,
        )
        print("\n[SUCCESS] Completed.\n")
    except Exception as exc:
        print(f"\n[ERROR] {exc}\n")
        import traceback

        traceback.print_exc()
