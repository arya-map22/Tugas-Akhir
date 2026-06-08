from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pandas import DataFrame, Timestamp


def plot_usia_vs_tahun(
    mortalitas_df: DataFrame,
    age_col: str,
    year_col: str,
    gender_col: str,
    mortality_col: str,
    start_age: int,
    end_age: int,
    factor: int,
    plots_dir: Path,
):
    assert 0 <= start_age <= end_age <= 100
    assert factor >= 1

    assert age_col in mortalitas_df.columns, "age_col harus ada di dalam mortalitas_df"

    plot_name = f"usia vs tahun ({start_age}-{end_age})"
    file_path = plots_dir / f"{plot_name}.png"

    mask = (
        (mortalitas_df[age_col] >= start_age)
        & (mortalitas_df[age_col] <= end_age)
        & (mortalitas_df[age_col] % factor == 0)
    )

    g = sns.FacetGrid(
        mortalitas_df[mask],
        col=age_col,
        hue=gender_col,
        height=6,
        col_wrap=4,
        sharex=False,
        sharey=False,
    )

    g.map_dataframe(sns.lineplot, x=year_col, y=mortality_col)

    g.set_titles("Age {col_name}", fontsize=12)
    g.set_axis_labels(year_col, "Mortality Rate")
    g.add_legend()

    g.tight_layout()
    g.savefig(file_path)
    print(f"Plot {plot_name} saved to {file_path}")


def plot_tahun_vs_usia(
    mortalitas_df: DataFrame,
    age_col: str,
    year_col: str,
    gender_col: str,
    mortality_col: str,
    start_year: str | datetime,
    end_year: str | datetime,
    factor: int,
    plots_dir: Path,
):
    assert factor >= 1

    plot_name = f"tahun vs usia ({start_year}-{end_year})"
    file_path = plots_dir / f"{plot_name}.png"

    start_year_dt = (
        Timestamp(
            year=int(start_year),
            month=1,
            day=1,
        )
        if not isinstance(start_year, datetime)
        else Timestamp(
            year=start_year.year,
            month=1,
            day=1,
        )
    )

    end_year_dt = (
        Timestamp(
            year=int(end_year),
            month=1,
            day=1,
        )
        if not isinstance(end_year, datetime)
        else Timestamp(
            year=end_year.year,
            month=1,
            day=1,
        )
    )

    df = mortalitas_df.copy()
    df["Year Only"] = df[year_col].dt.year

    assert not pd.isna(start_year_dt), "year_start harus bisa dikonversi ke datetime"
    assert not pd.isna(end_year_dt), "year_end harus bisa dikonversi ke datetime"

    mask = (
        (mortalitas_df[year_col] >= start_year_dt)
        & (mortalitas_df[year_col] <= end_year_dt)
        & (mortalitas_df[year_col].dt.year % factor == 0)
    )
    g = sns.FacetGrid(
        df[mask],
        col="Year Only",
        hue=gender_col,
        height=6,
        col_wrap=4,
        sharex=False,
        sharey=False,
    )

    g.map_dataframe(sns.lineplot, x=age_col, y=mortality_col)

    g.set_titles("Year {col_name}", fontsize=12)
    g.set_axis_labels("Age", "Mortality Rate")
    g.add_legend()

    g.tight_layout()
    g.savefig(file_path)
    print(f"Plot {plot_name} saved to {file_path}")


def plot_mean_std(
    df: DataFrame,
    plots_dir: Path,
    palette: dict | None = None,
) -> None:
    filepath = plots_dir / "line_plot_mortalitas_mean_std.png"
    if palette is None:
        palette = {
            "Female": {"line": "#c0394b", "band": "#e07b8a"},
            "Male": {"line": "#1a5fa8", "band": "#5a8fcb"},
        }

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    stat_cfg = [("mean", "Mean"), ("std", "Std Dev")]
    for ax, (col, title) in zip(axes, stat_cfg):
        for gender, colors in palette.items():
            d = df[df["gender"] == gender]
            sns.lineplot(
                data=d,
                x="age",
                y=col,
                color=colors["line"],
                linewidth=2.5,
                label=gender,
                ax=ax,
            )
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xlabel("Umur (tahun)")
        ax.set_ylabel("Nilai Mortalitas")
        ax.legend(title="Jenis Kelamin")
        ax.grid(True, linestyle="--", alpha=0.4)

    fig.suptitle(
        "Mean dan Std Dev Mortalitas per Umur dan Jenis Kelamin",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    print("Berhasil tersimpan!")


def plot_min_med_max(
    df: DataFrame,
    plots_dir: Path,
    palette: dict | None = None,
) -> None:
    filepath = plots_dir / "line_plot_mortalitas_min_med_max.png"
    if palette is None:
        palette = {
            "Female": {"line": "#c0394b", "band": "#e07b8a"},
            "Male": {"line": "#1a5fa8", "band": "#5a8fcb"},
        }

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    for ax, (gender, colors) in zip(axes, palette.items()):
        d = df[df["gender"] == gender]
        ax.fill_between(
            d["age"],
            d["min"],
            d["max"],
            color=colors["band"],
            alpha=0.25,
            label="Min–Max",
        )
        for col, lw, ls in [("min", 1, ":"), ("max", 1, ":"), ("median", 2.5, "-")]:
            sns.lineplot(
                data=d,
                x="age",
                y=col,
                color=colors["line"],
                linewidth=lw,
                linestyle=ls,
                label=("Median" if col == "median" else "_nolegend_"),
                ax=ax,
            )
        ax.set_title(f"Min – Median – Max ({gender})", fontsize=13, fontweight="bold")
        ax.set_xlabel("Umur (tahun)")
        ax.set_ylabel("Nilai Mortalitas")
        ax.legend(fontsize=9)
        ax.grid(True, linestyle="--", alpha=0.4)

    fig.suptitle(
        "Min – Median – Max Mortalitas per Umur dan Jenis Kelamin",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    print("Berhasil tersimpan!")
