from datetime import datetime
from pathlib import Path

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
    plot_dir: Path,
):
    assert age_col in mortalitas_df.columns, "age_col harus ada di dalam mortalitas_df"

    plot_name = f"usia vs tahun ({age_start}-{age_end})"
    file_path = plot_dir / f"{plot_name}.png"
    if not file_path.exists():
        mask = (mortalitas_df[age_col] >= age_start) & (
            mortalitas_df[age_col] <= age_end
        )
        g = sns.FacetGrid(
            mortalitas_df[mask],
            col=age_col,
            hue=sex_col,
            height=5,
            col_wrap=5,
            sharex=False,
            sharey=False,
        )

        g.map_dataframe(sns.lineplot, x=year_col, y=mortality_col)

        g.set_titles("Age {col_name}")
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
    year_end: str,
    plot_dir: Path,
):
    plot_name = f"tahun vs usia ({year_start}-{year_end})"
    file_path = plot_dir / f"{plot_name}.png"
    if not file_path.exists():
        year_start_dt = (
            Timestamp(
                year=int(year_start),
                month=1,
                day=1,
            )
            if not isinstance(year_start, datetime)
            else year_start.year
        )

        year_end_dt = (
            Timestamp(
                year=int(year_end),
                month=1,
                day=1,
            )
            if not isinstance(year_end, datetime)
            else year_end.year
        )

        df = mortalitas_df.copy()
        df["Year Only"] = df[year_col].dt.year

        mask = (mortalitas_df[year_col] >= year_start_dt) & (
            mortalitas_df[year_col] <= year_end_dt
        )
        g = sns.FacetGrid(
            df[mask],
            col="Year Only",
            hue=sex_col,
            height=5,
            col_wrap=5,
            sharex=False,
            sharey=False,
        )

        g.map_dataframe(sns.lineplot, x=age_col, y=mortality_col)

        g.set_titles("Year {col_name}")
        g.set_axis_labels("Age", "Mortality Rate")
        g.add_legend()

        g.tight_layout()
        g.savefig(file_path)
        print(f"Plot {plot_name} saved to {file_path}")
