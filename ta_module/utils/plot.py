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
    sex_col: str,
    mortality_col: str,
    age_start: int,
    age_end: int,
    plots_dir: Path,
):
    assert age_col in mortalitas_df.columns, "age_col harus ada di dalam mortalitas_df"

    plot_name = f"usia vs tahun ({age_start}-{age_end})"
    file_path = plots_dir / f"{plot_name}.png"

    mask = (
        (mortalitas_df[age_col] >= age_start)
        & (mortalitas_df[age_col] <= age_end)
        & (mortalitas_df[age_col] % 5 == 0)
    )
    g = sns.FacetGrid(
        mortalitas_df[mask],
        col=age_col,
        hue=sex_col,
        height=6,
        col_wrap=4,
        sharex=False,
        sharey=False,
    )

    g.map_dataframe(sns.lineplot, x=year_col, y=mortality_col)

    g.figure.suptitle(
        "Mortalitas per Tahun untuk Setiap Kelompok Usia",
        fontsize=16,
        fontweight="bold",
    )
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
    sex_col: str,
    mortality_col: str,
    year_start: str | datetime,
    year_end: str | datetime,
    plots_dir: Path,
):
    plot_name = f"tahun vs usia ({year_start}-{year_end})"
    file_path = plots_dir / f"{plot_name}.png"

    year_start_dt = (
        Timestamp(
            year=int(year_start),
            month=1,
            day=1,
        )
        if not isinstance(year_start, datetime)
        else Timestamp(
            year=year_start.year,
            month=1,
            day=1,
        )
    )

    year_end_dt = (
        Timestamp(
            year=int(year_end),
            month=1,
            day=1,
        )
        if not isinstance(year_end, datetime)
        else Timestamp(
            year=year_end.year,
            month=1,
            day=1,
        )
    )

    df = mortalitas_df.copy()
    df["Year Only"] = df[year_col].dt.year

    assert not pd.isna(year_start_dt), "year_start harus bisa dikonversi ke datetime"
    assert not pd.isna(year_end_dt), "year_end harus bisa dikonversi ke datetime"

    mask = (
        (mortalitas_df[year_col] >= year_start_dt)
        & (mortalitas_df[year_col] <= year_end_dt)
        & (mortalitas_df[year_col].dt.year % 5 == 0)
    )
    g = sns.FacetGrid(
        df[mask],
        col="Year Only",
        hue=sex_col,
        height=6,
        col_wrap=4,
        sharex=False,
        sharey=False,
    )

    g.map_dataframe(sns.lineplot, x=age_col, y=mortality_col)

    g.figure.suptitle(
        "Mortalitas per Usia untuk Setiap Tahun", fontsize=16, fontweight="bold"
    )
    g.set_titles("Year {col_name}", fontsize=12)
    g.set_axis_labels("Age", "Mortality Rate")
    g.add_legend()

    g.tight_layout()
    g.savefig(file_path)
    print(f"Plot {plot_name} saved to {file_path}")


def plot_mortalitas_statdesc(df: DataFrame, plots_dir: Path):
    filepath = plots_dir / "line_plot_mortalitas_statdesc.png"
    palette = {
        "Female": {"line": "#c0394b", "band": "#e07b8a"},
        "Male": {"line": "#1a5fa8", "band": "#5a8fcb"},
    }

    fig, axes = plt.subplots(1, 4, figsize=(22, 5))

    # ── Plot 1: Mean ───────────────────────────────────────────────────────────────
    ax = axes[0]
    for sex, colors in palette.items():
        d = df[df["sex"] == sex]
        sns.lineplot(
            data=d,
            x="age",
            y="mean",
            color=colors["line"],
            linewidth=2.5,
            label=sex,
            ax=ax,
        )
    ax.set_title("Mean", fontsize=13, fontweight="bold")
    ax.set_xlabel("Umur (tahun)")
    ax.set_ylabel("Nilai Mortalitas")
    ax.legend(title="Jenis Kelamin")
    ax.grid(True, linestyle="--", alpha=0.4)

    # ── Plot 2: Std ────────────────────────────────────────────────────────────────
    ax = axes[1]
    for sex, colors in palette.items():
        d = df[df["sex"] == sex]
        sns.lineplot(
            data=d,
            x="age",
            y="std",
            color=colors["line"],
            linewidth=2.5,
            label=sex,
            ax=ax,
        )
    ax.set_title("Std Dev", fontsize=13, fontweight="bold")
    ax.set_xlabel("Umur (tahun)")
    ax.set_ylabel("Nilai Mortalitas")
    ax.legend(title="Jenis Kelamin")
    ax.grid(True, linestyle="--", alpha=0.4)

    # ── Plot 3 & 4: Min–Median–Max, terpisah per jenis kelamin ────────────────────
    for i, (sex, colors) in enumerate(palette.items()):
        ax = axes[2 + i]
        d = df[df["sex"] == sex]
        ax.fill_between(
            d["age"],
            d["min"],
            d["max"],
            color=colors["band"],
            alpha=0.25,
            label="Min–Max",
        )
        sns.lineplot(
            data=d,
            x="age",
            y="min",
            color=colors["line"],
            linewidth=1,
            linestyle=":",
            ax=ax,
        )
        sns.lineplot(
            data=d,
            x="age",
            y="max",
            color=colors["line"],
            linewidth=1,
            linestyle=":",
            ax=ax,
        )
        sns.lineplot(
            data=d,
            x="age",
            y="median",
            color=colors["line"],
            linewidth=2.5,
            label="Median",
            ax=ax,
        )
        ax.set_title(f"Min – Median – Max ({sex})", fontsize=13, fontweight="bold")
        ax.set_xlabel("Umur (tahun)")
        ax.set_ylabel("Nilai Mortalitas")
        ax.legend(fontsize=9)
        ax.grid(True, linestyle="--", alpha=0.4)

    fig.suptitle(
        "Statistika Deskriptif Mortalitas per Umur dan Jenis Kelamin",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    print("Berhasil tersimpan!")
