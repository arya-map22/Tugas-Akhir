# %%
from enum import StrEnum


class RuntimeEnvironment(StrEnum):
    colab = "colab"
    local = "local"
# %%
RUNTIME_ENVIRONMENT = RuntimeEnvironment.colab
# %%
if RUNTIME_ENVIRONMENT == RuntimeEnvironment.colab:
    !pip install lightning pydantic pydantic-settings optuna optuna-integration openpyxl

    from google.colab import drive

    # Mount google drive
    drive.mount('/content/drive', force_remount=True)

    import os

    # Change working directory to coding environment
    from google.colab import userdata

    os.chdir(userdata.get("colab_cwd"))
# %%
from pathlib import Path

cwd = Path.cwd()
print(cwd)
# %%
from ta_module.config import DotEnv, Config, load_config, load_dot_env

DOT_ENV: DotEnv = load_dot_env()
CONFIG: Config = load_config(DOT_ENV.config_file)

print(f"DOT_ENV:\n{DOT_ENV}")
print(f"\nCONFIG:\n{CONFIG}")
# %%
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Available device: {DEVICE}")
# %%
from lightning import seed_everything

seed_everything(CONFIG.seed, workers=True)
# %% [markdown]
# # Analisis karakteristik data
# %% [markdown]
# ## Load data dari penyimpanan
# %%
MORTALITAS_CONFIG = CONFIG.data.mortalitas

YEAR_COL = MORTALITAS_CONFIG.year_col
AGE_COL = MORTALITAS_CONFIG.age_col
GENDER_COL = MORTALITAS_CONFIG.gender_col
MORTALITY_COL = MORTALITAS_CONFIG.mortality_col

BI_RATE_CONFIG = CONFIG.data.bi_rate
# %%
import pandas as pd

mortalitas_df = pd.read_csv(
    DOT_ENV.mortalitas_file, parse_dates=[YEAR_COL], date_format=MORTALITAS_CONFIG.date_format
)

populasi_df = pd.read_csv(DOT_ENV.populasi_file)

bi_rate_df = pd.read_csv(
    DOT_ENV.bi_rate_file, parse_dates=[BI_RATE_CONFIG.date_col], date_format=BI_RATE_CONFIG.date_format
)
# %%
AGE_MIN: int = min(mortalitas_df[AGE_COL])
AGE_MAX: int = max(mortalitas_df[AGE_COL])

YEAR_MIN: int = min(mortalitas_df[YEAR_COL].dt.year)
YEAR_MAX: int = max(mortalitas_df[YEAR_COL].dt.year)
# %% [markdown]
# ## Analisa statistika deskriptif
# %%
mortalitas_df.head()
# %%
mortalitas_df.info()
# %%
statdesc = ["mean", "std", "min", "median", "max"]
# %%
mortalitas_df.groupby([GENDER_COL])[MORTALITY_COL].aggregate(
    statdesc
)
# %%
mortalitas_df[MORTALITY_COL].aggregate(
    statdesc
)
# %%
mortalitas_statdesc_df = mortalitas_df.groupby([GENDER_COL, AGE_COL])[MORTALITY_COL].aggregate(
    statdesc
)
mortalitas_statdesc_df.to_csv(DOT_ENV.results_dir / "mortalitas_statdesc.csv")
# %%
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme(style="whitegrid")

def plot_mean_std(
    df: pd.DataFrame,
    plots_dir: Path,
    palette: dict | None = None,
) -> None:
    filepath = plots_dir / "line_plot_mortalitas_mean_std.png"
    if palette is None:
        palette = {
            "Female": {"line": "#c0394b", "band": "#e07b8a"},
            "Male":   {"line": "#1a5fa8", "band": "#5a8fcb"},
        }

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    stat_cfg = [("mean", "Mean"), ("std", "Std Dev")]
    for ax, (col, title) in zip(axes, stat_cfg):
        for gender, colors in palette.items():
            d = df[df["gender"] == gender]
            sns.lineplot(
                data=d, x="age", y=col,
                color=colors["line"], linewidth=2.5,
                label=gender, ax=ax,
            )
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xlabel("Umur (tahun)")
        ax.set_ylabel("Nilai Mortalitas")
        ax.legend(title="Jenis Kelamin")
        ax.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    print("Berhasil tersimpan!")
# %%
mortalitas_statdesc_df = pd.read_csv(DOT_ENV.results_dir / "mortalitas_statdesc.csv")
plot_mean_std(df=mortalitas_statdesc_df, plots_dir=DOT_ENV.plots_dir)
# %%
def plot_min_med_max(
    df: pd.DataFrame,
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

    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    print("Berhasil tersimpan!")
# %%
plot_min_med_max(df=mortalitas_statdesc_df, plots_dir=DOT_ENV.plots_dir)
# %%
populasi_df.head()
# %%
populasi_df.info()
# %%
bi_rate_df.head()
# %%
bi_rate_df.info()
# %% [markdown]
# ## Histogram mortalitas
# %%
import torch
import numpy as np

# Male mortalitas
mortalitas_df_male = mortalitas_df[mortalitas_df[GENDER_COL] == "Male"]
M_male = mortalitas_df_male.pivot(
    index=YEAR_COL,
    columns=AGE_COL,
    values=MORTALITY_COL
)
M_male = torch.from_numpy(M_male.to_numpy(copy=True, dtype=np.float32)).to(DEVICE)

# Female mortalitas
mortalitas_df_female = mortalitas_df[mortalitas_df[GENDER_COL] == "Female"]
M_female = mortalitas_df_female.pivot(
    index=YEAR_COL,
    columns=AGE_COL,
    values=MORTALITY_COL
)
M_female = torch.from_numpy(M_female.to_numpy(copy=True, dtype=np.float32)).to(DEVICE)
# %%
import matplotlib.pyplot as plt
import seaborn as sns

fig, ax = plt.subplots(1, 2, figsize=(15, 7.5), dpi=300)

ax[0].hist(M_male.cpu().numpy().reshape(-1), density=True, color="blue", bins="auto")
sns.kdeplot(M_male.cpu().numpy().reshape(-1), ax=ax[0], color="black", alpha=0.7)
ax[0].set_title("Laki-laki")
ax[0].set_xlabel("Mortality Rate")
ax[0].set_ylabel("Density")

ax[1].hist(M_female.cpu().numpy().reshape(-1), density=True, color="red", bins="auto")
sns.kdeplot(M_female.cpu().numpy().reshape(-1), ax=ax[1], color="black", alpha=0.7)
ax[1].set_title("Perempuan")
ax[1].set_xlabel("Mortality Rate")
ax[1].set_ylabel("Density")

fig.savefig(DOT_ENV.plots_dir / "distribusi_data_mortalitas.png")
plt.show()
# %% [markdown]
# ## Plot mortalitas satu usia untuk semua tahun
# %%
plots_dir = DOT_ENV.plots_dir
# %%
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd
import seaborn as sns
from pandas import DataFrame


# ─────────────────────────────────────────────────────────────────────────────
# Core function
# ─────────────────────────────────────────────────────────────────────────────

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
    # ── parameter estetika & kualitas ──────────────────────────────────────
    palette: list[str] | None = None,
    col_wrap: int = 4,
    panel_height: float = 3.2,
    panel_aspect: float = 1.4,
    linewidth: float = 1.8,
    dpi: int = 300,
    fig_title: str | None = None,
    y_label: str = "Mortality Rate",
    x_label: str | None = None,
    log_y: bool = False,
) -> None:
    """
    Plot mortality rate (y) vs year (x) per panel usia.

    Parameters
    ----------
    mortalitas_df : DataFrame dengan kolom usia, tahun, gender, dan mortality.
    age_col       : nama kolom usia.
    year_col      : nama kolom tahun.
    gender_col    : nama kolom gender/jenis kelamin.
    mortality_col : nama kolom mortality rate.
    start_age     : usia awal (inklusif), 0–100.
    end_age       : usia akhir (inklusif), 0–100.
    factor        : tampilkan hanya usia yang habis dibagi factor
                    (misal factor=5 → 30, 35, 40, ...).
    plots_dir     : direktori output untuk menyimpan file PNG.
    palette       : list 2 warna hex/named untuk gender [val1, val2].
                    Default: biru-coral ["#2E6FA3", "#D4704F"].
    col_wrap      : jumlah panel per baris.
    panel_height  : tinggi tiap panel dalam inci.
    panel_aspect  : rasio lebar/tinggi tiap panel.
    linewidth     : ketebalan garis di setiap panel.
    dpi           : resolusi gambar (300 = HD cetak, 150 = layar cukup).
    fig_title     : judul besar di atas semua panel (opsional).
    y_label       : label sumbu-y (default "Mortality Rate").
    x_label       : label sumbu-x; jika None, pakai nama year_col.
    log_y         : True = pakai skala logaritmik pada sumbu-y.

    Returns
    -------
    Path ke file PNG yang tersimpan.
    """
    # ── validasi ────────────────────────────────────────────────────────────
    assert 0 <= start_age <= end_age <= 100, "start_age dan end_age harus 0–100"
    assert factor >= 1, "factor harus >= 1"
    assert age_col in mortalitas_df.columns, f"'{age_col}' tidak ada di DataFrame"
    assert year_col in mortalitas_df.columns, f"'{year_col}' tidak ada di DataFrame"
    assert gender_col in mortalitas_df.columns, f"'{gender_col}' tidak ada di DataFrame"
    assert mortality_col in mortalitas_df.columns, f"'{mortality_col}' tidak ada di DataFrame"

    # ── tema global ──────────────────────────────────────────────────────────
    sns.set_theme(
        style="white",
        rc={
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": True,
            "axes.spines.bottom": True,
            "axes.linewidth": 0.8,
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "axes.grid": True,
            "grid.linestyle": "--",
            "grid.alpha": 0.4,
            "grid.linewidth": 0.6,
            "font.family": "sans-serif",
        },
    )
    mpl.rcParams["figure.dpi"] = dpi

    _palette = palette or ["#2E6FA3", "#D4704F"]  # blue-coral default

    # ── filter data ──────────────────────────────────────────────────────────
    mask = (
        mortalitas_df[age_col].between(start_age, end_age)
        & (mortalitas_df[age_col] % factor == 0)
    )
    df_plot = mortalitas_df[mask].copy()

    if df_plot.empty:
        raise ValueError(
            f"Tidak ada data untuk usia {start_age}–{end_age} dengan factor={factor}."
        )

    # ── FacetGrid ────────────────────────────────────────────────────────────
    g = sns.FacetGrid(
        df_plot,
        col=age_col,
        hue=gender_col,
        palette=_palette,
        height=panel_height,
        aspect=panel_aspect,
        col_wrap=col_wrap,
        sharex=False,
        sharey=False,
        despine=True,
    )

    g.map_dataframe(
        sns.lineplot,
        x=year_col,
        y=mortality_col,
        linewidth=linewidth,
        errorbar=None,
    )

    # ── per-axes styling ─────────────────────────────────────────────────────
    _x_label = x_label or year_col
    for ax in g.axes.flat:
        if ax is None:
            continue
        ax.set_xlabel(_x_label, fontsize=10, color="#555555")
        ax.set_ylabel(y_label, fontsize=10, color="#555555")
        ax.tick_params(labelsize=9, colors="#444444")
        for spine in ax.spines.values():
            spine.set_color("#CCCCCC")
        if log_y:
            ax.set_yscale("log")
            ax.yaxis.set_major_formatter(
                mpl.ticker.FuncFormatter(lambda v, _: f"{v:.3g}")
            )

    # ── titles & legend ──────────────────────────────────────────────────────
    g.set_titles(
        "Age {col_name}",
        fontsize=11,
        fontweight="semibold",
        pad=6,
    )
    g.set_axis_labels(_x_label, y_label)

    if fig_title:
        g.figure.suptitle(
            fig_title,
            fontsize=14,
            fontweight="semibold",
            y=1.01,
            color="#222222",
        )

    # ── layout & save ────────────────────────────────────────────────────────
    g.figure.tight_layout(pad=1.5, h_pad=2.0, w_pad=1.5)

    # legend di bawah semua panel, tengah
    g.add_legend(
        title=gender_col.title(),
        title_fontsize=10,
        fontsize=9,
        frameon=True,
        framealpha=0.85,
        edgecolor="#DDDDDD",
        loc="upper center",
        bbox_to_anchor=(0.5, 0.0),
        ncol=len(mortalitas_df[gender_col].unique()),
    )
    g.figure.subplots_adjust(bottom=0.08)

    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    plot_name = f"usia_vs_tahun_{start_age}_{end_age}"
    file_path = plots_dir / f"{plot_name}.png"

    g.savefig(file_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.show()

    print(f"✓  Saved → {file_path}  ({dpi} dpi)")
# %%
def wrapper_plot_usia_vs_tahun(age_start: int, age_end: int, factor: int):
    plot_usia_vs_tahun(
        mortalitas_df=mortalitas_df,
        age_col=AGE_COL,
        year_col=YEAR_COL,
        gender_col=GENDER_COL,
        mortality_col=MORTALITY_COL,
        start_age=age_start,
        end_age=age_end,
        factor=factor,
        plots_dir=plots_dir
    )
# %%
wrapper_plot_usia_vs_tahun(AGE_MIN, AGE_MAX, 10)
# %% [markdown]
# ## Plot mortalitas semua usia untuk satu tahun
# %%
from datetime import datetime
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd
import seaborn as sns
from pandas import DataFrame, Timestamp


# ─────────────────────────────────────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────────────────────────────────────

def _to_timestamp(value: str | int | datetime) -> Timestamp:
    """Konversi string/int tahun atau datetime ke Timestamp awal tahun."""
    if isinstance(value, datetime):
        return Timestamp(year=value.year, month=1, day=1)
    return Timestamp(year=int(value), month=1, day=1)


# ─────────────────────────────────────────────────────────────────────────────
# Core function
# ─────────────────────────────────────────────────────────────────────────────

def plot_tahun_vs_usia(
    mortalitas_df: DataFrame,
    age_col: str,
    year_col: str,
    gender_col: str,
    mortality_col: str,
    start_year: str | int | datetime,
    end_year: str | int | datetime,
    factor: int,
    plots_dir: Path,
    # ── parameter estetika & kualitas ──────────────────────────────────────
    palette: list[str] | None = None,
    col_wrap: int = 4,
    panel_height: float = 3.2,
    panel_aspect: float = 1.4,
    linewidth: float = 1.8,
    dpi: int = 300,
    fig_title: str | None = None,
    y_label: str = "Mortality Rate",
    x_label: str = "Age",
    log_y: bool = False,
) -> None:
    """
    Plot mortality rate (y) vs usia (x) per panel tahun.

    Parameters
    ----------
    mortalitas_df : DataFrame dengan kolom usia, tahun, gender, dan mortality.
    age_col       : nama kolom usia.
    year_col      : nama kolom tahun (tipe datetime/Timestamp).
    gender_col    : nama kolom gender/jenis kelamin.
    mortality_col : nama kolom mortality rate.
    start_year    : tahun awal (inklusif); bisa int, str, atau datetime.
    end_year      : tahun akhir (inklusif); bisa int, str, atau datetime.
    factor        : tampilkan hanya tahun yang habis dibagi factor
                    (misal factor=5 → 1990, 1995, 2000, ...).
    plots_dir     : direktori output untuk menyimpan file PNG.
    palette       : list 2 warna hex/named untuk gender [val1, val2].
                    Default: biru-coral ["#2E6FA3", "#D4704F"].
    col_wrap      : jumlah panel per baris.
    panel_height  : tinggi tiap panel dalam inci.
    panel_aspect  : rasio lebar/tinggi tiap panel.
    linewidth     : ketebalan garis di setiap panel.
    dpi           : resolusi gambar (300 = HD cetak, 150 = layar cukup).
    fig_title     : judul besar di atas semua panel (opsional).
    y_label       : label sumbu-y (default "Mortality Rate").
    x_label       : label sumbu-x (default "Age").
    log_y         : True = pakai skala logaritmik pada sumbu-y.

    Returns
    -------
    Path ke file PNG yang tersimpan.
    """
    # ── validasi ────────────────────────────────────────────────────────────
    assert factor >= 1, "factor harus >= 1"
    for col in [age_col, year_col, gender_col, mortality_col]:
        assert col in mortalitas_df.columns, f"'{col}' tidak ada di DataFrame"

    start_ts = _to_timestamp(start_year)
    end_ts   = _to_timestamp(end_year)

    assert not pd.isna(start_ts), "start_year tidak bisa dikonversi ke Timestamp"
    assert not pd.isna(end_ts),   "end_year tidak bisa dikonversi ke Timestamp"
    assert start_ts <= end_ts,    "start_year harus <= end_year"

    # ── tema global ──────────────────────────────────────────────────────────
    sns.set_theme(
        style="white",
        rc={
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": True,
            "axes.spines.bottom": True,
            "axes.linewidth": 0.8,
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "axes.grid": True,
            "grid.linestyle": "--",
            "grid.alpha": 0.4,
            "grid.linewidth": 0.6,
            "font.family": "sans-serif",
        },
    )
    mpl.rcParams["figure.dpi"] = dpi

    _palette = palette or ["#2E6FA3", "#D4704F"]  # blue-coral default

    # ── filter & siapkan data ────────────────────────────────────────────────
    df = mortalitas_df.copy()
    df["_year_only"] = df[year_col].dt.year

    mask = (
        (df[year_col] >= start_ts)
        & (df[year_col] <= end_ts)
        & (df["_year_only"] % factor == 0)
    )
    df_plot = df[mask].copy()

    if df_plot.empty:
        raise ValueError(
            f"Tidak ada data untuk tahun {int(start_year)}–{int(end_year)} "
            f"dengan factor={factor}."
        )

    # ── FacetGrid ────────────────────────────────────────────────────────────
    g = sns.FacetGrid(
        df_plot,
        col="_year_only",
        hue=gender_col,
        palette=_palette,
        height=panel_height,
        aspect=panel_aspect,
        col_wrap=col_wrap,
        sharex=False,
        sharey=False,
        despine=True,
    )

    g.map_dataframe(
        sns.lineplot,
        x=age_col,
        y=mortality_col,
        linewidth=linewidth,
        errorbar=None,
    )

    # ── per-axes styling ─────────────────────────────────────────────────────
    for ax in g.axes.flat:
        if ax is None:
            continue
        ax.set_xlabel(x_label, fontsize=10, color="#555555")
        ax.set_ylabel(y_label, fontsize=10, color="#555555")
        ax.tick_params(labelsize=9, colors="#444444")
        for spine in ax.spines.values():
            spine.set_color("#CCCCCC")
        if log_y:
            ax.set_yscale("log")
            ax.yaxis.set_major_formatter(
                mpl.ticker.FuncFormatter(lambda v, _: f"{v:.3g}")
            )

    # ── titles & legend ──────────────────────────────────────────────────────
    g.set_titles(
        "Year {col_name}",
        fontsize=11,
        fontweight="semibold",
        pad=6,
    )
    g.set_axis_labels(x_label, y_label)

    if fig_title:
        g.figure.suptitle(
            fig_title,
            fontsize=14,
            fontweight="semibold",
            y=1.01,
            color="#222222",
        )

    # ── layout & save ────────────────────────────────────────────────────────
    g.figure.tight_layout(pad=1.5, h_pad=2.0, w_pad=1.5)

    # legend di bawah semua panel, tengah
    g.add_legend(
        title=gender_col.title(),
        title_fontsize=10,
        fontsize=9,
        frameon=True,
        framealpha=0.85,
        edgecolor="#DDDDDD",
        loc="upper center",
        bbox_to_anchor=(0.5, 0.0),
        ncol=len(mortalitas_df[gender_col].unique()),
    )
    g.figure.subplots_adjust(bottom=0.08)

    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    plot_name = f"tahun_vs_usia_{int(start_year)}_{int(end_year)}"
    file_path = plots_dir / f"{plot_name}.png"

    g.savefig(file_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.show()

    print(f"✓  Saved → {file_path}  ({dpi} dpi)")
# %%
def wrapper_plot_tahun_vs_usia(year_start: str, year_end: str, factor: int):
    plot_tahun_vs_usia(
        mortalitas_df=mortalitas_df,
        age_col=AGE_COL,
        year_col=YEAR_COL,
        gender_col=GENDER_COL,
        mortality_col=MORTALITY_COL,
        start_year=year_start,
        end_year=year_end,
        factor=factor,
        plots_dir=plots_dir
    )
# %%
wrapper_plot_tahun_vs_usia(str(YEAR_MIN), str(YEAR_MAX), 10)
# %% [markdown]
# # Persiapan data mortalitas untuk pelatihan model
# %%
TRAINING_CONFIG = CONFIG.training
# %% [markdown]
# ## Transformasi data
# %%
from ta_module.utils import ScaledLogitTransform

# Dari persamaan asumsi UDD qx = 2mx / (2 + mx)
# 0 <= 1x <= 1
# Jadi constraint untuk mx adalah:
lower_bound = 0.0
upper_bound = 2.0

transform_fn = ScaledLogitTransform(lb=lower_bound, ub=upper_bound)

transformed_M_male = transform_fn(M_male)
transformed_M_female = transform_fn(M_female)
# %%
pd.DataFrame(transformed_M_male.cpu().numpy(),
             index=range(YEAR_MIN, YEAR_MAX + 1),
             columns=range(AGE_MIN, AGE_MAX + 1)
).to_csv(DOT_ENV.results_dir / "transformed_male_mortality.csv", sep=";", decimal=",")
pd.DataFrame(transformed_M_female.cpu().numpy(),
             index=range(YEAR_MIN, YEAR_MAX + 1),
             columns=range(AGE_MIN, AGE_MAX + 1)
).to_csv(DOT_ENV.results_dir / "transformed_female_mortality.csv", sep=';', decimal=",")
# %% [markdown]
# ## Pembagian data
# %%
from ta_module.data import get_train_val_test_split

train_split = CONFIG.split.train
val_split = CONFIG.split.validation
test_split = CONFIG.split.test

M_male_train, M_male_val, M_male_test = get_train_val_test_split(
    mortality_matrix=transformed_M_male,
    train_split=train_split,
    val_split=val_split,
    test_split=test_split
)

M_female_train, M_female_val, M_female_test = get_train_val_test_split(
    mortality_matrix=transformed_M_female,
    train_split=train_split,
    val_split=val_split,
    test_split=test_split
)
# %%
lookback = CONFIG.dataset.lookback
horizon = CONFIG.dataset.horizon

# Tambahkan konteks dari data train ke data val dan test agar sesuai dengan kebutuhan lookback
M_male_val_extend = torch.cat([M_male_train[-lookback:, :], M_male_val], dim=0)
M_male_test_extend = torch.cat([M_male_val_extend[-lookback:, :], M_male_test], dim=0)
M_female_val_extend = torch.cat([M_female_train[-lookback:, :], M_female_val], dim=0)
M_female_test_extend = torch.cat([M_female_val_extend[-lookback:, :], M_female_test], dim=0)
# %% [markdown]
# ## Normalisasi
# %% [markdown]
# ### Laki-laki
# %%
import torch

train_male_mean = M_male_train.mean(dim=0)
train_male_std = M_male_train.std(dim=0)

pd.DataFrame({"mean": train_male_mean.cpu().numpy(), "std": train_male_std.cpu().numpy()}).to_csv(
    DOT_ENV.results_dir / "train_male_mean_std.csv", sep=";", decimal=","
)
# %%
pd.DataFrame(((transformed_M_male - train_male_mean) / train_male_std).cpu().numpy(),
             index=range(YEAR_MIN, YEAR_MAX + 1),
             columns=range(AGE_MIN, AGE_MAX + 1)
).to_csv(DOT_ENV.results_dir / "male_normalized.csv", sep=";", decimal=",")
# %%
from ta_module.data import NormalizedMortalityDataset

create_male_dataset_split = NormalizedMortalityDataset.factory(
    lookback=lookback,
    horizon=horizon,
    mean=train_male_mean,
    std=train_male_std,
)

train_male_dataset = create_male_dataset_split(M_male_train)
val_male_dataset = create_male_dataset_split(M_male_val_extend)
test_male_dataset = create_male_dataset_split(M_male_test_extend)
# %%
import matplotlib.pyplot as plt
from ta_module.utils import normalize

male_train_normalized = normalize(M_male_train, train_male_mean, train_male_std).cpu().numpy()
male_val_normalized = normalize(M_male_val, train_male_mean, train_male_std).cpu().numpy()
male_test_normalized = normalize(M_male_test, train_male_mean, train_male_std).cpu().numpy()

fig, ax = plt.subplots(1, 2, figsize=(15, 7.5), dpi=300)
ax[0].plot(male_train_normalized.mean(axis=0), color="blue", linestyle="-", label="train")
ax[0].plot(male_val_normalized.mean(axis=0), color="red", linestyle="--", label="val")
ax[0].plot(male_test_normalized.mean(axis=0), color="orange", linestyle="-.", label="test")
ax[0].set_title("Mean")
ax[0].legend()
ax[0].set_xlabel("Age")
ax[0].set_ylabel("Mean value")

ax[1].plot(male_train_normalized.std(axis=0), color="blue", linestyle="-", label="train")
ax[1].plot(male_val_normalized.std(axis=0), color="red", linestyle="--", label="val")
ax[1].plot(male_test_normalized.std(axis=0), color="orange", linestyle="-.", label="test")
ax[1].set_title("Std")
ax[1].legend()
ax[1].set_xlabel("Age")
ax[1].set_ylabel("Std value")

fig.suptitle("Laki-laki", fontsize=18, fontweight="bold")
fig.savefig(DOT_ENV.plots_dir / "mean_std_mortalitas_laki-laki_setelah_normalisasi.png")
plt.show()
# %% [markdown]
# ### Perempuan
# %%
train_female_mean = M_female_train.mean(dim=0)
train_female_std = M_female_train.std(dim=0)

pd.DataFrame({"mean": train_female_mean.cpu().numpy(), "std": train_female_std.cpu().numpy()}).to_csv(
    DOT_ENV.results_dir / "train_female_mean_std.csv", sep=";", decimal=","
)
# %%
pd.DataFrame(((transformed_M_female - train_female_mean) / train_female_std).cpu().numpy(),
             index=range(YEAR_MIN, YEAR_MAX + 1),
             columns=range(AGE_MIN, AGE_MAX + 1)
).to_csv(DOT_ENV.results_dir / "female_normalized.csv", sep=";", decimal=",")
# %%
create_female_dataset_split = NormalizedMortalityDataset.factory(
    lookback=lookback,
    horizon=horizon,
    mean=train_female_mean,
    std=train_female_std,
)

train_female_dataset = create_female_dataset_split(M_female_train)
val_female_dataset = create_female_dataset_split(M_female_val_extend)
test_female_dataset = create_female_dataset_split(M_female_test_extend)
# %%
female_train_normalized = normalize(M_female_train, train_female_mean, train_female_std).cpu().numpy()
female_val_normalized = normalize(M_female_val, train_female_mean, train_female_std).cpu().numpy()
female_test_normalized = normalize(M_female_test, train_female_mean, train_female_std).cpu().numpy()

fig, ax = plt.subplots(1, 2, figsize=(15, 7.5), dpi=300)
ax[0].plot(female_train_normalized.mean(axis=0), color="blue", linestyle="-", label="train")
ax[0].plot(female_val_normalized.mean(axis=0), color="red", linestyle="--", label="val")
ax[0].plot(female_test_normalized.mean(axis=0), color="orange", linestyle="-.", label="test")
ax[0].set_title("Mean")
ax[0].legend()
ax[0].set_xlabel("Age")
ax[0].set_ylabel("Mean value")

ax[1].plot(female_train_normalized.std(axis=0), color="blue", linestyle="-", label="train")
ax[1].plot(female_val_normalized.std(axis=0), color="red", linestyle="--", label="val")
ax[1].plot(female_test_normalized.std(axis=0), color="orange", linestyle="-.", label="test")
ax[1].set_title("Std")
ax[1].legend()
ax[1].set_xlabel("Age")
ax[1].set_ylabel("Std value")

fig.suptitle("Perempuan", fontsize=18, fontweight="bold")
fig.savefig(DOT_ENV.plots_dir / "mean_std_mortalitas_perempuan_setelah_normalisasi.png")
plt.show()
# %%
fig, ax = plt.subplots(1, 3, figsize=(15, 7.5), dpi=300)
ax[0].hist(male_train_normalized.reshape(-1), color="blue", bins="auto", density=True)
sns.kdeplot(male_train_normalized.reshape(-1), color="black", alpha=0.7, ax=ax[0], bw_adjust=3)
ax[0].set_title("Train")
ax[0].set_xlabel("Transformed and Normalized Mortality Rate")
ax[0].set_ylabel("Density")

ax[1].hist(male_val_normalized.reshape(-1), color="red", bins="auto", density=True)
sns.kdeplot(male_val_normalized.reshape(-1), color="black", alpha=0.7, ax=ax[1], bw_adjust=3)
ax[1].set_title("Val")
ax[1].set_xlabel("Transformed and Normalized Mortality Rate")
ax[1].set_ylabel("Density")

ax[2].hist(male_test_normalized.reshape(-1), color="orange", bins="auto", density=True)
sns.kdeplot(male_test_normalized.reshape(-1), color="black", alpha=0.7, ax=ax[2], bw_adjust=3)
ax[2].set_title("Test")
ax[2].set_xlabel("Transformed and Normalized Mortality Rate")
ax[2].set_ylabel("Density")

fig.suptitle("Laki-laki", fontsize=16, fontweight="bold")
fig.savefig(DOT_ENV.plots_dir / "distribusi_data_mortalitas_setelah_normalisasi_laki-laki.png")
plt.show()
# %%
fig, ax = plt.subplots(1, 3, figsize=(15, 7.5), dpi=300)
ax[0].hist(female_train_normalized.reshape(-1), color="blue", bins="auto", density=True)
sns.kdeplot(female_train_normalized.reshape(-1), color="black", alpha=0.7, ax=ax[0], bw_adjust=3)
ax[0].set_title("Train")
ax[0].set_xlabel("Transformed and Normalized Mortality Rate")
ax[0].set_ylabel("Density")

ax[1].hist(female_val_normalized.reshape(-1), color="red", bins="auto", density=True)
sns.kdeplot(female_val_normalized.reshape(-1), color="black", alpha=0.7, ax=ax[1], bw_adjust=3)
ax[1].set_title("Val")
ax[1].set_xlabel("Transformed and Normalized Mortality Rate")
ax[1].set_ylabel("Density")

ax[2].hist(female_test_normalized.reshape(-1), color="orange", bins="auto", density=True)
sns.kdeplot(female_test_normalized.reshape(-1), color="black", alpha=0.7, ax=ax[2], bw_adjust=3)
ax[2].set_title("Test")
ax[2].set_xlabel("Transformed and Normalized Mortality Rate")
ax[2].set_ylabel("Density")

fig.suptitle("Perempuan", fontsize=16, fontweight="bold")
fig.savefig(DOT_ENV.plots_dir / "distribusi_data_mortalitas_setelah_normalisasi_perempuan.png")
plt.show()
# %% [markdown]
# ## Penggabungan data mortalitas laki-laki dan perempuan untuk pelatihan
# %%
from torch.utils.data import ConcatDataset

train_dataset = ConcatDataset([train_male_dataset, train_female_dataset])
val_dataset = ConcatDataset([val_male_dataset, val_female_dataset])
# %%
from torch.utils.data import DataLoader

batch_size = TRAINING_CONFIG.batch_size

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=len(val_male_dataset), shuffle=False)
# %%
for batch in val_dataloader:
    x, y = batch
# %% [markdown]
# # Pelatihan model
# %% [markdown]
# ### Plot learning rate
# %%
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import seaborn as sns

# Load data
df = pd.read_excel(DOT_ENV.results_dir / "learning_rate_schedule_500_epochs.xlsx", decimal=",")
df.columns = ["epoch", "lr", "phase", ""]

warmup = df[df["phase"] == "Warm-up"]
cosine = df[df["phase"] == "Cosine Annealing WR"]

# Deteksi restart: puncak lokal di cosine annealing dengan LR > 4.98e-3
cosine_vals = cosine["lr"].values
cosine_eps = cosine["epoch"].values
restart_epochs = [
    cosine_eps[i]
    for i in range(1, len(cosine_vals) - 1)
    if cosine_vals[i] >= cosine_vals[i - 1]
       and cosine_vals[i] >= cosine_vals[i + 1]
       and cosine_vals[i] > 4.98e-3
]

warmup_end = warmup["epoch"].max()  # epoch 20

# Styling
sns.set_theme(style="whitegrid", font="serif")
plt.rcParams.update(
    {
        "axes.facecolor"  : "#FAFAFA",
        "figure.facecolor": "#FFFFFF",
        "grid.color"      : "#E8E8E8",
        "grid.linewidth"  : 0.7,
        "axes.edgecolor"  : "#CCCCCC",
        "axes.labelcolor" : "#333333",
        "xtick.color"     : "#555555",
        "ytick.color"     : "#555555",
        "xtick.labelsize" : 10,
        "ytick.labelsize" : 10,
        "axes.labelsize"  : 12,
        "text.color"      : "#222222",
        "font.family"     : "serif",
    }
)

WARMUP_COLOR = "#E8834A"  # oranye hangat
COSINE_COLOR = "#4A90D9"  # biru
RESTART_COLOR = "#CC3333"  # merah tua

fig, ax = plt.subplots(figsize=(14, 5.5))

# Shading background per fase
ax.axvspan(
    warmup["epoch"].min(), warmup_end,
    color=WARMUP_COLOR, alpha=0.06, zorder=0
)
ax.axvspan(
    warmup_end, cosine["epoch"].max(),
    color=COSINE_COLOR, alpha=0.04, zorder=0
)

# Plot garis LR
ax.plot(
    warmup["epoch"], warmup["lr"],
    color=WARMUP_COLOR, linewidth=2.2, zorder=3,
    solid_capstyle="round"
)

ax.plot(
    cosine["epoch"], cosine["lr"],
    color=COSINE_COLOR, linewidth=1.8, zorder=3,
    solid_capstyle="round"
)

# Titik sambung warm-up cosine
ax.plot(
    warmup_end, warmup[warmup["epoch"] == warmup_end]["lr"].values[0],
    "o", color=WARMUP_COLOR, markersize=6, zorder=5
)

# Garis vertikal restart
for i, ep in enumerate(restart_epochs):
    lr_at_restart = cosine[cosine["epoch"] == ep]["lr"].values[0]
    ax.axvline(
        ep, color=RESTART_COLOR, linewidth=1.1,
        linestyle="--", alpha=0.75, zorder=2
    )
    # Label "R1", "R2", ... di atas garis
    ax.text(
        ep + 1.5, lr_at_restart * 1.01,
        f"R{i + 1}", fontsize=8.5, color=RESTART_COLOR,
        fontweight="bold", va="bottom"
        )

# Annotation warm-up end
ax.axvline(
    warmup_end + 0.5, color="#AAAAAA", linewidth=1.0,
    linestyle=":", zorder=2
    )
ax.text(
    warmup_end - 1, 5e-3 * 1.015, "warm-up\nend",
    ha="right", fontsize=8, color="#888888", style="italic", va="bottom"
    )

# Estetika sumbu
ax.set_title(
    "Learning Rate Schedule  -  500 Epochs  -  Warm-up + Cosine Annealing with Restarts",
    fontsize=14, fontweight="bold", color="#111111",
    pad=14, loc="center",
)
ax.set_xlabel("Epoch", labelpad=10)
ax.set_ylabel("Learning Rate", labelpad=10)

ax.set_xlim(0, df["epoch"].max() + 5)
y_max = df["lr"].max()
ax.set_ylim(-0.0001, y_max * 1.08)

ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=8, prune="both"))
ax.yaxis.set_major_formatter(
    mticker.FuncFormatter(
        lambda x, _: f"{x:.3f}" if x >= 0.001 else f"{x:.2e}"
    )
)
ax.xaxis.set_major_locator(mticker.MultipleLocator(50))
ax.xaxis.set_minor_locator(mticker.MultipleLocator(10))

# Legenda
handles = [
    mpatches.Patch(color=WARMUP_COLOR, alpha=0.85, label="Warm-up (epoch 1 - 20)"),
    mpatches.Patch(color=COSINE_COLOR, alpha=0.85, label="Cosine Annealing WR (epoch 21 - 500)"),
    Line2D(
        [0], [0], color=RESTART_COLOR, linewidth=1.5,
        linestyle="--", label=f"Cosine restart  (epoch {', '.join(map(str, restart_epochs))})"
    ),
]
leg = ax.legend(
    handles=handles,
    loc="upper right",
    fontsize=9.5,
    framealpha=0.8,
    facecolor="#FFFFFF",
    edgecolor="#CCCCCC",
    handlelength=2.0,
)
for text in leg.get_texts():
    text.set_color("#222222")

plt.tight_layout(pad=1.5)
fig.savefig(
    DOT_ENV.plots_dir / "lr_schedule.png", dpi=150, bbox_inches="tight",
    facecolor=fig.get_facecolor()
    )

print("Plot disimpan ke: lr_schedule.png")
plt.show()
# %% [markdown]
# ## Definisi arsitektur model
# %%
MODEL_CONFIG = CONFIG.model
# %% [markdown]
# ### LCN
# %%
from torch import nn
from ta_module.models import LocalGLMnet, LocallyConnected2D

ukuran_matriks_input = (lookback, AGE_MAX - AGE_MIN + 1)
lcn_activation_function = nn.Sigmoid()
LCN_CONFIG = MODEL_CONFIG.lcn

create_lcn_layer = LocallyConnected2D.factory(
    input_size=ukuran_matriks_input,
    activation_fn=lcn_activation_function,
    kernel_size=LCN_CONFIG.kernel_size,
    zero_padding=LCN_CONFIG.zero_padding,
    bias=LCN_CONFIG.bias,
)
# %% [markdown]
# ### LocalGLMnet
# %%
from ta_module.utils import IdentityTransform

link_fn = IdentityTransform()

LOCALGLMNET_CONFIG = MODEL_CONFIG.localglmnet
create_localglmnet_model = LocalGLMnet.factory(
    input_size=ukuran_matriks_input,
    link_fn=link_fn,
    bias=LOCALGLMNET_CONFIG.bias,
)
# %% [markdown]
# ## Definisi metode pelatihan model
# %%
TRAINING_CONFIG = CONFIG.training
# %% [markdown]
# ### Loss function dan metrik evaluasi
# %%
import torch.nn.functional as F

loss_metric_fn = F.mse_loss
eval_metric_fn = F.l1_loss
# %% [markdown]
# ### Optimizer dan lr scheduler
# %%
from typing import Iterator
from torch.optim import NAdam

OPTIMIZER_CONFIG = TRAINING_CONFIG.optimizer


def create_optimizer(params: Iterator[nn.Parameter]):
    return NAdam(
        params=params,
        lr=OPTIMIZER_CONFIG.lr,
    )
# %%
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingWarmRestarts, SequentialLR

LR_SCHEDULER_CONFIG = TRAINING_CONFIG.lr_scheduler


def create_lr_scheduler(optimizer: Optimizer):
    warm_up_epochs = LR_SCHEDULER_CONFIG.total_iters
    warm_up_scheduler = LinearLR(
        optimizer=optimizer,
        start_factor=LR_SCHEDULER_CONFIG.start_factor,
        end_factor=LR_SCHEDULER_CONFIG.end_factor,
        total_iters=warm_up_epochs
    )

    sgdr_scheduler = CosineAnnealingWarmRestarts(
        optimizer=optimizer,
        T_0=LR_SCHEDULER_CONFIG.T_0,
        T_mult=LR_SCHEDULER_CONFIG.T_mult,
        eta_min=LR_SCHEDULER_CONFIG.eta_min
    )

    return SequentialLR(
        optimizer=optimizer,
        schedulers=[warm_up_scheduler, sgdr_scheduler],
        milestones=[warm_up_epochs]
    )
# %% [markdown]
# ## Grid search untuk hyperparameter koefisien regularisasi LASSO
# %%
TUNING_CONFIG = CONFIG.tuning
# %% [markdown]
# ### Proses grid search
# %%
from optuna.trial import Trial
from typing import Callable, Iterator
from torch.nn import Parameter

from ta_module.tuning import reg_coef_grid_search, reg_coef_objective
from ta_module.config import TuneMetadata, ModeEnum
from ta_module.utils import get_current_run_datetime, ElasticNetRegularizationTerm
from ta_module.models import LocalGLMnetLightning

if CONFIG.mode == ModeEnum.TUNE:
    max_epochs = TRAINING_CONFIG.max_epochs
    min_epochs = TRAINING_CONFIG.min_epochs
    reg_coef_candidates = TUNING_CONFIG.reg_coef_candidates


    def create_mymodel_with_reg_coef(
        reg_coef: float
    ):
        def _create_regularization_term(model_weights_getter: Callable[[], Iterator[Parameter]]):
            return ElasticNetRegularizationTerm(
                reg_coef=reg_coef,
                alpha=TRAINING_CONFIG.regularization.alpha,
                model_weights_getter=model_weights_getter,
            )

        localglmnet = create_localglmnet_model(create_lcn_layer())
        localglmnet_attention_weights_getter = lambda: (
            params for name, params in
            localglmnet.regression_attention_model.named_parameters()
            if "bias" not in name
        )

        return LocalGLMnetLightning(
            model=localglmnet,
            loss_metric=loss_metric_fn,
            eval_metric=eval_metric_fn,
            create_optimizer=create_optimizer,
            create_lr_scheduler=create_lr_scheduler,
            regularization_term=_create_regularization_term(model_weights_getter=localglmnet_attention_weights_getter),
        )

    objective: Callable[[Trial], float] = lambda trial: (
        reg_coef_objective(
            trial=trial,
            create_my_model_with_reg_coef=create_mymodel_with_reg_coef,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            max_epochs=max_epochs,
            min_epochs=min_epochs,
            log_dir=DOT_ENV.tuning_logs_dir,
            checkpoint_dir=DOT_ENV.tuning_checkpoints_dir,
            reg_coef_candidates=reg_coef_candidates,
            gradient_clip_val=TRAINING_CONFIG.regularization.gradient_clip_val,
            seed=CONFIG.seed
        )
    )

    run_datetime = get_current_run_datetime()
    print(f"==================================================================")
    print("Mulai grid search untuk hyperparameter reg_coef dalam lasso loss")
    print(f"Run datetime: {run_datetime}")
    print(f"reg_coef candidates: {reg_coef_candidates}")
    print(f"==================================================================\n")
    grid_search_result = reg_coef_grid_search(
        objective_fn=objective,
        reg_coef_candidates=reg_coef_candidates,
        storage=DOT_ENV.optuna_db_url,
        seed=CONFIG.seed,
    )
    print(f"\n==================================================================\n")
    print("Grid_search selesai!")
    print(f"Result:\n{grid_search_result}\n")

    last_tune_metadata_filepath = DOT_ENV.last_tune_metadata_file
    print(f"Update file {last_tune_metadata_filepath}:")
    tune_metadata: TuneMetadata = TuneMetadata.model_validate(
        {
            "datetime": run_datetime,
            "result"  : grid_search_result,
        }
    )
    print(f"Tune metadata:\n{tune_metadata}\n")
    with open(last_tune_metadata_filepath, "w") as f:
        f.write(tune_metadata.model_dump_json(indent=4))
    print(f"Update berhasil!")
else:
    print(f"Mode = {CONFIG.mode}")
    print("Skip proses tuning")
# %% [markdown]
# ### Plot perkembangan metrik dalam tuning
# %%
"""
plot_metrics.py
===============
Fungsi untuk memplot training/validation metrics dari multiple CSV files
yang dihasilkan PyTorch Lightning.

Usage:
    from plot_metrics import plot_metrics

    plot_metrics(
        csv_paths=[
            "metrics_fold1.csv",
            "metrics_fold2.csv",
            "metrics_fold3.csv",
        ],
        titles=[
            "Fold 1 – Male",
            "Fold 2 – Female",
            "Fold 3 – Combined",
        ],
        save_path="training_curves.png",
    )
"""

import math
import warnings
from pathlib import Path
from typing import Optional

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# Design tokens
# ─────────────────────────────────────────────────────────────────────────────

# Refined palette — cool-warm contrast, all readable on white
PALETTE = {
    "train_loss":            "#5B8DB8",   # steel blue
    "train_regularized_loss": "#E07B4F",  # terracotta
    "val_loss":              "#7B6FBF",   # soft violet
    "val_score":             "#3DAA6E",   # emerald green
}

LINE_STYLES = {
    "train_loss":            (0, ()),          # solid
    "train_regularized_loss": (0, (5, 2)),     # dashed
    "val_loss":              (0, (1, 1)),      # dense dotted
    "val_score":             (0, (4, 2, 1, 2)),# dash-dot
}

LABELS = {
    "train_loss":            "Train Loss",
    "train_regularized_loss": "Train Reg. Loss",
    "val_loss":              "Val Loss",
    "val_score":             "Val Score",
}

# column mapping: key → CSV column name
COL_MAP = {
    "train_loss":            "train_loss_epoch",
    "train_regularized_loss": "train_loss_regularized_epoch",
    "val_loss":              "val_loss",
    "val_score":             "val_score",
}

LOSS_KEYS  = ["train_loss", "train_regularized_loss", "val_loss"]
SCORE_KEYS = ["val_score"]


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────
def _load_epoch_df(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=",", decimal=".")
    required = list(COL_MAP.values())
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")
    epoch_df = (
        df.groupby("epoch")[required]
        .first()
        .reset_index()
    )
    epoch_df.rename(columns={v: k for k, v in COL_MAP.items()}, inplace=True)
    return epoch_df


# ─────────────────────────────────────────────────────────────────────────────
# Subplot renderer
# ─────────────────────────────────────────────────────────────────────────────
def _render_subplot(
    ax_loss,
    epoch_df: pd.DataFrame,
    title: str,
    mark_best_val: bool,
) -> list:
    """
    Draw one subplot. Returns (lines, labels) for the shared legend.
    Uses twin-axis: losses on ax_loss (left), score on ax_score (right).
    Both axes use log scale.
    """
    ax_score = ax_loss.twinx()

    collected_lines = []

    # ── Loss curves (left axis) ─────────────────────────────────────────────
    for key in LOSS_KEYS:
        s = epoch_df[key].dropna()
        if s.empty:
            continue
        x = epoch_df.loc[s.index, "epoch"].values
        y = s.values

        ln, = ax_loss.plot(
            x, y,
            color=PALETTE[key],
            linestyle=LINE_STYLES[key],
            linewidth=2.0,
            marker="none",
            label=LABELS[key],
            zorder=3,
            solid_capstyle="round",
        )
        collected_lines.append(ln)

    # ── Score curve (right axis) ────────────────────────────────────────────
    for key in SCORE_KEYS:
        s = epoch_df[key].dropna()
        if s.empty:
            continue
        x = epoch_df.loc[s.index, "epoch"].values
        y = s.values

        ln, = ax_score.plot(
            x, y,
            color=PALETTE[key],
            linestyle=LINE_STYLES[key],
            linewidth=2.0,
            marker="none",
            label=LABELS[key],
            zorder=3,
            solid_capstyle="round",
        )
        collected_lines.append(ln)

    # ── Log scale both axes ─────────────────────────────────────────────────
    ax_loss.set_yscale("log")
    ax_score.set_yscale("log")

    # ── Spine / tick cosmetics ──────────────────────────────────────────────
    ax_score.spines["right"].set_visible(True)
    ax_score.spines["right"].set_color(PALETTE["val_score"])
    ax_score.spines["right"].set_linewidth(1.2)
    ax_score.tick_params(
        axis="y",
        labelcolor=PALETTE["val_score"],
        labelsize=8,
        length=3,
    )
    ax_score.set_ylabel(
        "Val Score (log)",
        color=PALETTE["val_score"],
        fontsize=8.5,
        labelpad=6,
    )

    ax_loss.set_ylabel("Loss (log)", fontsize=8.5, labelpad=6, color="#444444")
    ax_loss.set_xlabel("Epoch", fontsize=8.5, labelpad=4, color="#444444")
    ax_loss.tick_params(axis="both", labelsize=8, length=3)
    ax_loss.xaxis.set_major_locator(mticker.MaxNLocator(integer=True, nbins=7))

    # Minor grid only on loss axis (log minor ticks)
    ax_loss.yaxis.set_minor_locator(mticker.LogLocator(subs="all"))
    ax_loss.grid(True, which="major", axis="both",
                 color="#EBEBEB", linewidth=0.8, zorder=0)
    ax_loss.grid(True, which="minor", axis="y",
                 color="#F4F4F4", linewidth=0.5, zorder=0)

    # ── Best val_loss marker ────────────────────────────────────────────────
    if mark_best_val:
        val_s = epoch_df["val_loss"].dropna()
        if not val_s.empty:
            best_idx = val_s.idxmin()
            best_ep  = epoch_df.loc[best_idx, "epoch"]
            best_val = val_s.min()
            ax_loss.axvline(
                best_ep,
                color="#AAAAAA",
                linestyle=(0, (3, 4)),
                linewidth=1.1,
                zorder=2,
            )
            ax_loss.annotate(
                f"best\nep {best_ep}",
                xy=(best_ep, best_val),
                xytext=(6, 6),
                textcoords="offset points",
                fontsize=7,
                color="#888888",
                va="bottom",
            )

    # ── Subplot title ───────────────────────────────────────────────────────
    ax_loss.set_title(
        title,
        fontsize=11,
        fontweight="semibold",
        color="#222222",
        pad=10,
        loc="left",
    )

    return collected_lines


# ─────────────────────────────────────────────────────────────────────────────
# Public function
# ─────────────────────────────────────────────────────────────────────────────
def plot_metrics(
    csv_paths: list,
    titles: Optional[list] = None,
    *,
    figsize_per_cell: tuple = (7.2, 4.6),
    n_cols: int = 3,
    mark_best_val: bool = True,
    save_path=None,
    dpi: int = 300,
) -> None:
    """
    Plot train_loss, train_regularized_loss, val_loss, val_score per epoch
    for n metrics CSV files in a 2-column grid. Legend placed outside plots.

    Parameters
    ----------
    csv_paths       : list of CSV file paths (one per subplot)
    titles          : custom subtitle per subplot (defaults to filename stem)
    fig_title       : master figure title
    figsize_per_cell: (width, height) inches per subplot cell
    n_cols          : number of columns (default 2)
    marker_every    : marker interval; auto if None
    mark_best_val   : draw vertical line at epoch with lowest val_loss
    save_path       : save figure to this path if given
    dpi             : resolution for saved file
    show            : call plt.show()
    """
    n = len(csv_paths)
    if n == 0:
        raise ValueError("csv_paths must not be empty.")
    if titles is None:
        titles = [Path(p).stem for p in csv_paths]
    if len(titles) != n:
        raise ValueError(f"len(titles) must equal len(csv_paths).")

    # ── Seaborn theme ────────────────────────────────────────────────────────
    sns.set_theme(
        style="white",
        context="notebook",
        rc={
            "font.family":       "DejaVu Sans",
            "axes.spines.top":   False,
            "axes.spines.right": False,
            "axes.edgecolor":    "#D0D0D0",
            "axes.linewidth":    0.9,
            "figure.facecolor":  "white",
            "axes.facecolor":    "white",
        },
    )

    # ── Layout ───────────────────────────────────────────────────────────────
    n_rows    = math.ceil(n / n_cols)
    fig_w     = figsize_per_cell[0] * n_cols
    # extra vertical space: legend band at top + suptitle
    legend_h  = 0.55
    fig_h     = figsize_per_cell[1] * n_rows + legend_h + 0.7

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="white")

    # Reserve top strip for legend; subplots fill the rest
    legend_frac = legend_h / fig_h
    gs = fig.add_gridspec(
        n_rows, n_cols,
        top=1.0 - legend_frac - 0.04,
        bottom=0.07,
        hspace=0.52,
        wspace=0.38,
        left=0.07,
        right=0.93,
    )

    axes_list = [fig.add_subplot(gs[r, c])
                 for r in range(n_rows) for c in range(n_cols)]

    # ── Per-subplot rendering ────────────────────────────────────────────────
    all_lines = None
    for idx, (path, title) in enumerate(zip(csv_paths, titles)):
        ax = axes_list[idx]
        epoch_df   = _load_epoch_df(path)
        lines      = _render_subplot(ax, epoch_df, title, mark_best_val)
        if all_lines is None:
            all_lines = lines   # grab handles from first subplot

    # ── Hide unused axes ─────────────────────────────────────────────────────
    for ax in axes_list[n:]:
        ax.set_visible(False)

    # ── Shared legend — horizontal strip above subplots ───────────────────
    if all_lines:
        leg = fig.legend(
            handles=all_lines,
            labels=[ln.get_label() for ln in all_lines],
            loc="upper center",
            bbox_to_anchor=(0.5, 1.0 - legend_frac * 0.18),
            ncol=len(all_lines),
            fontsize=9.5,
            frameon=True,
            framealpha=0.95,
            edgecolor="#DDDDDD",
            fancybox=False,
            handlelength=2.6,
            handleheight=1.0,
            columnspacing=2.0,
            handletextpad=0.6,
        )
        leg.get_frame().set_linewidth(0.8)

    # ── Thin separator line under legend ────────────────────────────────────
    line_y = 1.0 - legend_frac
    fig.add_artist(
        mpl.lines.Line2D(
            [0.05, 0.95], [line_y, line_y],
            transform=fig.transFigure,
            color="#E0E0E0",
            linewidth=0.8,
        )
    )

    # ── Save / show ───────────────────────────────────────────────────────────
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight", facecolor="white")
        print(f"Saved → {save_path}")

    plt.show()
# %%
tuning_log_path = DOT_ENV.tuning_logs_dir / "tune_reg_coef_elasticnet_regularization"
metric_files = [
    tuning_log_path / "Trial_0_reg_coef_1e_02/2026-05-17_22-08-24/metrics.csv",
    tuning_log_path / "Trial_1_reg_coef_5e_06/2026-05-17_22-11-40/metrics.csv",
    tuning_log_path / "Trial_2_reg_coef_5e_04/2026-05-17_22-14-19/metrics.csv",
    tuning_log_path / "Trial_3_reg_coef_1e_06/2026-05-17_22-16-49/metrics.csv",
    tuning_log_path / "Trial_4_reg_coef_5e_03/2026-05-17_22-19-20/metrics.csv",
    tuning_log_path / "Trial_5_reg_coef_1e_05/2026-05-17_22-21-48/metrics.csv",
    tuning_log_path / "Trial_6_reg_coef_0e00/2026-05-17_22-24-14/metrics.csv",
    tuning_log_path / "Trial_7_reg_coef_1e_04/2026-05-17_22-26-39/metrics.csv",
    tuning_log_path / "Trial_8_reg_coef_5e_05/2026-05-17_22-29-02/metrics.csv",
    tuning_log_path / "Trial_9_reg_coef_1e_03/2026-05-17_22-31-25/metrics.csv"
]
titles = [
    "\u03BB = 1E-2",
    "\u03BB = 5E-6",
    "\u03BB = 5E-4",
    "\u03BB = 1E-6",
    "\u03BB = 5E-3",
    "\u03BB = 1E-5",
    "\u03BB = 0",
    "\u03BB = 1E-4",
    "\u03BB = 5E-5",
    "\u03BB = 1E-3"
]
plot_metrics(
    csv_paths=metric_files,
    titles=titles,
    save_path=DOT_ENV.plots_dir / "perkembangan_metrik_tuning.png"
)
# %% [markdown]
# ### Plot metrik dan learning rate tuning
# %%
"""
------------------------
2-panel plot:
  (atas) train_loss, train_loss_regularized, val_loss (kiri)
         val_score (kanan, dual y-axis)
  (bawah) learning rate schedule
Garis vertikal restart cosine annealing tembus ke kedua panel.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import seaborn as sns

# Load data
metrics = pd.read_csv(
    DOT_ENV.tuning_logs_dir / "tune_reg_coef_elasticnet_regularization/Trial_2_reg_coef_5e_04/2026-05-17_22-14-19" / "metrics.csv"
)

train_df = (metrics.dropna(subset=["train_loss_epoch"])
            [["epoch", "train_loss_epoch"]].reset_index(drop=True))
train_reg = (metrics.dropna(subset=["train_loss_regularized_epoch"])
             [["epoch", "train_loss_regularized_epoch"]].reset_index(drop=True))
val_df = (metrics.dropna(subset=["val_loss"])
          [["epoch", "val_loss"]].reset_index(drop=True))
score_df = (metrics.dropna(subset=["val_score"])
            [["epoch", "val_score"]].reset_index(drop=True))

lr_df = pd.read_excel(
    DOT_ENV.results_dir / "learning_rate_schedule_500_epochs.xlsx",
    sheet_name="LR_Schedule_500",
    usecols=["Epoch", "Learning Rate", "Phase"],
)

lr_df.columns = ["epoch", "lr", "phase"]
lr_df["epoch"] = lr_df["epoch"] - 1

warmup = lr_df[lr_df["phase"] == "Warm-up"]
cosine = lr_df[lr_df["phase"] == "Cosine Annealing WR"]

# Deteksi restart
cosine_vals = cosine["lr"].values
cosine_eps = cosine["epoch"].values
restart_epochs = [
    cosine_eps[i]
    for i in range(1, len(cosine_vals) - 1)
    if cosine_vals[i] >= cosine_vals[i - 1]
       and cosine_vals[i] >= cosine_vals[i + 1]
       and cosine_vals[i] > 4.98e-3
]
warmup_end = warmup["epoch"].max()

# Styling
sns.set_theme(style="whitegrid", font="serif")
plt.rcParams.update(
    {
        "axes.facecolor"  : "#FAFAFA",
        "figure.facecolor": "#FFFFFF",
        "grid.color"      : "#E8E8E8",
        "grid.linewidth"  : 0.7,
        "axes.edgecolor"  : "#CCCCCC",
        "axes.labelcolor" : "#333333",
        "xtick.color"     : "#555555",
        "ytick.color"     : "#555555",
        "xtick.labelsize" : 10,
        "ytick.labelsize" : 10,
        "axes.labelsize"  : 12,
        "text.color"      : "#222222",
        "font.family"     : "serif",
    }
)

TRAIN_COLOR = "#4A90D9"  # biru
TRAIN_REG_COLOR = "#1A5FA8"  # biru tua
VAL_COLOR = "#E8834A"  # oranye
SCORE_COLOR = "#27AE60"  # hijau (sumbu kanan)
WARMUP_COLOR = "#F5A623"
COSINE_COLOR = "#7B68EE"
RESTART_COLOR = "#CC3333"

# Figure: 2 panel, shared x
fig, (ax_loss, ax_lr) = plt.subplots(
    2, 1,
    figsize=(14, 8),
    sharex=True,
    gridspec_kw={"height_ratios": [2.2, 1], "hspace": 0.06},
)

# Dual y-axis untuk panel atas
ax_score = ax_loss.twinx()

# PANEL ATAS Loss (kiri) + Val Score (kanan)

# Shading fase
ax_loss.axvspan(
    warmup["epoch"].min(), warmup_end,
    color=WARMUP_COLOR, alpha=0.07, zorder=0
)
ax_loss.axvspan(
    warmup_end, cosine["epoch"].max(),
    color=COSINE_COLOR, alpha=0.04, zorder=0
)

# Garis restart & warm-up end
for ep in restart_epochs:
    ax_loss.axvline(
        ep, color=RESTART_COLOR, linewidth=1.0,
        linestyle="--", alpha=0.65, zorder=2
    )
ax_loss.axvline(
    warmup_end, color="#AAAAAA", linewidth=1.0,
    linestyle=":", zorder=2
)

# Sumbu kiri: 3 garis loss
ax_loss.plot(
    train_df["epoch"], train_df["train_loss_epoch"],
    color=TRAIN_COLOR, linewidth=1.8, zorder=3, alpha=0.9,
    label="Train Loss"
)
ax_loss.plot(
    train_reg["epoch"], train_reg["train_loss_regularized_epoch"],
    color=TRAIN_REG_COLOR, linewidth=1.6, zorder=3, alpha=0.85,
    linestyle="--", label="Train Loss (Regularized)"
)
ax_loss.plot(
    val_df["epoch"], val_df["val_loss"],
    color=VAL_COLOR, linewidth=1.8, zorder=3, alpha=0.9,
    label="Validation Loss"
)

# Log scale sumbu kiri
all_loss = np.concatenate(
    [
        train_df["train_loss_epoch"].values,
        train_reg["train_loss_regularized_epoch"].values,
        val_df["val_loss"].values,
    ]
)
ax_loss.set_yscale("log")
ax_loss.set_ylim(all_loss.min() * 0.92, all_loss.max() * 1.08)
ax_loss.yaxis.set_major_locator(mticker.LogLocator(base=10, numticks=8))
ax_loss.yaxis.set_major_formatter(mticker.ScalarFormatter())
ax_loss.yaxis.set_minor_formatter(mticker.NullFormatter())
ax_loss.set_ylabel("Loss (log scale)", labelpad=10, color="#333333")

# Sumbu kanan: val_score
ax_score.plot(
    score_df["epoch"], score_df["val_score"],
    color=SCORE_COLOR, linewidth=1.6, zorder=3, alpha=0.85,
    linestyle="-.", label="Validation Score"
)

s_min, s_max = score_df["val_score"].min(), score_df["val_score"].max()
ax_score.set_yscale("log")
ax_score.set_ylim(s_min * 0.92, s_max * 1.08)
ax_score.yaxis.set_major_locator(mticker.LogLocator(base=10, numticks=6))
ax_score.yaxis.set_major_formatter(mticker.ScalarFormatter())
ax_score.yaxis.set_minor_formatter(mticker.NullFormatter())

ax_score.set_ylabel("Validation Score (log scale)", labelpad=10, color=SCORE_COLOR)
ax_score.tick_params(axis="y", colors=SCORE_COLOR, labelsize=10)
ax_score.spines["right"].set_edgecolor(SCORE_COLOR)
ax_score.spines["right"].set_alpha(0.6)

# Label nilai akhir
for ax_, arr, ep_col, col, fmt in [
    (ax_loss, train_df["train_loss_epoch"].values, train_df["epoch"], TRAIN_COLOR, ".2f"),
    (ax_loss, train_reg["train_loss_regularized_epoch"].values, train_reg["epoch"], TRAIN_REG_COLOR, ".2f"),
    (ax_loss, val_df["val_loss"].values, val_df["epoch"], VAL_COLOR, ".2f"),
    (ax_score, score_df["val_score"].values, score_df["epoch"], SCORE_COLOR, ".3f"),
]:
    last_ep = ep_col.iloc[-1]
    last_val = arr[-1]
    ax_.plot(last_ep, last_val, "o", color=col, markersize=5, zorder=6)
    ax_.annotate(
        f"  {last_val:{fmt}}", xy=(last_ep, last_val),
        fontsize=8, color=col, va="center", fontweight="bold"
    )

# Annotasi restart
for i, ep in enumerate(restart_epochs):
    ax_loss.text(
        ep + 1.5, all_loss.max() * 0.97,
        f"R{i + 1}", fontsize=8, color=RESTART_COLOR,
        fontweight="bold", va="top", zorder=6
        )

# Legenda gabungan (kiri + kanan)
loss_handles = [
    Line2D(
        [0], [0], color=TRAIN_COLOR, linewidth=2.0,
        label="Train Loss"
    ),
    Line2D(
        [0], [0], color=TRAIN_REG_COLOR, linewidth=1.8,
        linestyle="--", label="Train Loss (Regularized)"
    ),
    Line2D(
        [0], [0], color=VAL_COLOR, linewidth=2.0,
        label="Validation Loss"
    ),
    Line2D(
        [0], [0], color=SCORE_COLOR, linewidth=1.8,
        linestyle="-.", label="Validation Score  (kanan)"
    ),
    mpatches.Patch(color=WARMUP_COLOR, alpha=0.5, label="Warm-up phase"),
    mpatches.Patch(color=COSINE_COLOR, alpha=0.4, label="Cosine Annealing WR"),
    Line2D(
        [0], [0], color=RESTART_COLOR, linewidth=1.5,
        linestyle="--", label=f"Cosine restart (R1-R{len(restart_epochs)})"
    ),
]

leg = ax_loss.legend(
    handles=loss_handles, loc="upper right",
    fontsize=9.0, framealpha=0.88,
    facecolor="#FFFFFF", edgecolor="#CCCCCC",
    handlelength=2.2
)
for t in leg.get_texts():
    t.set_color("#222222")

# PANEL BAWAH Learning Rate

ax_lr.axvspan(
    warmup["epoch"].min(), warmup_end,
    color=WARMUP_COLOR, alpha=0.07, zorder=0
)
ax_lr.axvspan(
    warmup_end, cosine["epoch"].max(),
    color=COSINE_COLOR, alpha=0.04, zorder=0
)

for i, ep in enumerate(restart_epochs):
    lr_val = cosine[cosine["epoch"] == ep]["lr"].values[0]
    ax_lr.axvline(
        ep, color=RESTART_COLOR, linewidth=1.0,
        linestyle="--", alpha=0.65, zorder=2
    )
    ax_lr.text(
        ep + 1.5, lr_val * 1.01, f"R{i + 1}",
        fontsize=8, color=RESTART_COLOR, fontweight="bold",
        va="bottom", zorder=6
        )

ax_lr.axvline(
    warmup_end, color="#AAAAAA", linewidth=1.0,
    linestyle=":", zorder=2
)

ax_lr.plot(
    warmup["epoch"], warmup["lr"],
    color=WARMUP_COLOR, linewidth=2.2, zorder=3, solid_capstyle="round"
)
ax_lr.plot(
    cosine["epoch"], cosine["lr"],
    color=COSINE_COLOR, linewidth=1.8, zorder=3, solid_capstyle="round"
)

ax_lr.set_xlabel("Epoch", labelpad=10)
ax_lr.set_ylabel("Learning Rate", labelpad=10)

ax_lr.set_xlim(-2, lr_df["epoch"].max() + 5)
lr_y_max = lr_df["lr"].max()
ax_lr.set_ylim(-0.0001, lr_y_max * 1.12)
ax_lr.yaxis.set_major_locator(mticker.MaxNLocator(nbins=5, prune="both"))
ax_lr.yaxis.set_major_formatter(
    mticker.FuncFormatter(
        lambda x, _: f"{x:.3f}" if x >= 0.001 else f"{x:.2e}"
    )
)
ax_lr.xaxis.set_major_locator(mticker.MultipleLocator(50))
ax_lr.xaxis.set_minor_locator(mticker.MultipleLocator(10))

lr_handles = [
    Line2D([0], [0], color=WARMUP_COLOR, linewidth=2.2, label="Warm-up"),
    Line2D([0], [0], color=COSINE_COLOR, linewidth=1.8, label="Cosine Annealing WR"),
]
leg2 = ax_lr.legend(
    handles=lr_handles, loc="upper right",
    fontsize=9.5, framealpha=0.85,
    facecolor="#FFFFFF", edgecolor="#CCCCCC",
    handlelength=2.0
)

for t in leg2.get_texts():
    t.set_color("#222222")

plt.savefig(
    DOT_ENV.plots_dir / "tuning_metrics_vs_learning_rate.png", dpi=150, bbox_inches="tight",
    facecolor=fig.get_facecolor()
)

print("Plot disimpan!")
plt.show()
# %% [markdown]
# ## Pelatihan model dengan hyperparameter terbaik
# %% [markdown]
# ### Load hyperparameter terbaik dari tune metadata
# %%
from ta_module.config import load_last_tune_metadata

last_tune_metadata = load_last_tune_metadata(DOT_ENV.last_tune_metadata_file)
print(f"Last tune metadata:\n{last_tune_metadata}")
# %%
tune_trials_result = pd.DataFrame(last_tune_metadata.result.trials)
tune_trials_result.to_csv(DOT_ENV.results_dir / "tune_trials_result.csv", sep=";", decimal=",")
# %%
best_reg_coef = last_tune_metadata.result.best_params.get("reg_coef", None)
create_elasticnet_regularization = ElasticNetRegularizationTerm.factory(
    reg_coef=best_reg_coef,
    alpha=TRAINING_CONFIG.regularization.alpha,
)
# %% [markdown]
# ### Proses pelatihan
# %%
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger, CSVLogger
from lightning import Trainer

from ta_module.config import TrainMetadata
from ta_module.models import LocalGLMnetLightning
from ta_module.utils import get_current_run_datetime_str

if CONFIG.mode in [ModeEnum.TRAIN, ModeEnum.TUNE]:
    num_ensembles = MODEL_CONFIG.num_ensembles
    run_datetime = get_current_run_datetime()
    run_datetime_str = get_current_run_datetime_str()
    checkpoint_dirs = []

    # Train ensembles pada data mortalitas laki-laki dan perempuan
    print(f"Melatih {num_ensembles} model LocalGLMnet secara independen")
    print(f"Run datetime: {run_datetime_str}")
    for i in range(num_ensembles):
        # Random seed direset tiap model dengan nilai berbeda-beda agar inisiasi acak dan independen
        seed_everything(seed=CONFIG.seed * i + i, workers=True)

        localglmnet_model = create_localglmnet_model(create_lcn_layer())

        attention_weight_getter = lambda: (
            params for name, params in localglmnet_model.regression_attention_model.named_parameters() if
                 "bias" not in name
        )

        lightning_module = LocalGLMnetLightning(
            model=localglmnet_model,
            loss_metric=loss_metric_fn,
            eval_metric=eval_metric_fn,
            regularization_term=create_elasticnet_regularization(attention_weight_getter),
            create_optimizer=create_optimizer,
            create_lr_scheduler=create_lr_scheduler
        )

        max_epochs = TRAINING_CONFIG.max_epochs
        min_epochs = TRAINING_CONFIG.min_epochs
        model_name = f"LocalGLMnet_{i + 1}"
        checkpoint_dir = DOT_ENV.training_checkpoints_dir / model_name
        checkpoint_dirs.append(checkpoint_dir)

        trainer = Trainer(
            max_epochs=max_epochs,
            min_epochs=min_epochs,
            log_every_n_steps=1,
            deterministic=True,
            gradient_clip_val=TRAINING_CONFIG.regularization.gradient_clip_val,
            logger=[
                TensorBoardLogger(
                    save_dir=DOT_ENV.training_logs_dir,
                    name=model_name,
                    version=run_datetime_str
                ),
                CSVLogger(
                    save_dir=DOT_ENV.training_logs_dir,
                    name=model_name,
                    version=run_datetime_str,
                )
            ],
            callbacks=[
                ModelCheckpoint(
                    dirpath=checkpoint_dir,
                    filename=run_datetime_str,
                    monitor="val_loss",
                    mode="min",
                    save_top_k=1
                )
            ]
        )

        print("\n================================================================")
        print(f"Training model {i + 1} pada mortalitas laki-laki dan perempuan:")
        print("================================================================\n")
        trainer.fit(
            model=lightning_module,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader,
        )

    print("Pelatihan selesai!")

    # Save metadadata
    last_train_metadata_filepath = DOT_ENV.last_train_metadata_file
    checkpoint_file_paths = [checkpoint_dirs[i] / f"{run_datetime_str}.ckpt" for i in range(num_ensembles)]
    print(f"Update file {last_train_metadata_filepath}:")
    # metadata digunakan untuk load checkpoint model terakhir secara otomatis
    # jika tidak run proses pelatihan
    train_metadata = TrainMetadata.model_validate(
        {
            "datetime"             : run_datetime,
            "checkpoint_file_paths": checkpoint_file_paths,
        }, extra="forbid"
    )
    print(f"Train metadata:\n{train_metadata}\n")
    with open(last_train_metadata_filepath, "w") as f:
        f.write(train_metadata.model_dump_json(indent=4))
    print("Update berhasil!")
else:
    print(f"Mode = {CONFIG.mode}")
    print("Skip proses pelatihan, langsung inference")
# %% [markdown]
# ### Plot perkembangan metrik dalam training
# %%
training_log_path = DOT_ENV.training_logs_dir
metric_files = [
    training_log_path / "LocalGLMnet_1/2026-05-17_23-07-20/metrics.csv",
    training_log_path / "LocalGLMnet_2/2026-05-17_23-07-20/metrics.csv",
    training_log_path / "LocalGLMnet_3/2026-05-17_23-07-20/metrics.csv",
    training_log_path / "LocalGLMnet_4/2026-05-17_23-07-20/metrics.csv",
    training_log_path / "LocalGLMnet_5/2026-05-17_23-07-20/metrics.csv",
    training_log_path / "LocalGLMnet_6/2026-05-17_23-07-20/metrics.csv",
    training_log_path / "LocalGLMnet_7/2026-05-17_23-07-20/metrics.csv",
    training_log_path / "LocalGLMnet_8/2026-05-17_23-07-20/metrics.csv",
    training_log_path / "LocalGLMnet_9/2026-05-17_23-07-20/metrics.csv",
    training_log_path / "LocalGLMnet_10/2026-05-17_23-07-20/metrics.csv",
]
titles = [
    "LocalGLMnet1",
    "LocalGLMnet2",
    "LocalGLMnet3",
    "LocalGLMnet4",
    "LocalGLMnet5",
    "LocalGLMnet6",
    "LocalGLMnet7",
    "LocalGLMnet8",
    "LocalGLMnet9",
    "LocalGLMnet10",
]
plot_metrics(
    csv_paths=metric_files,
    titles=titles,
    save_path=DOT_ENV.plots_dir / "perkembangan_metrik_training.png"
)
# %% [markdown]
# ### Plot metrik dan learning rate training
# %%
"""
------------------------
2-panel plot:
  (atas) train_loss, train_loss_regularized, val_loss (kiri)
         val_score (kanan, dual y-axis)
  (bawah) learning rate schedule
Garis vertikal restart cosine annealing tembus ke kedua panel.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import seaborn as sns

# Load data
metrics = pd.read_csv(DOT_ENV.training_logs_dir / "LocalGLMnet_10/2026-05-17_23-07-20" / "metrics.csv")

train_df = (metrics.dropna(subset=["train_loss_epoch"])
            [["epoch", "train_loss_epoch"]].reset_index(drop=True))
train_reg = (metrics.dropna(subset=["train_loss_regularized_epoch"])
             [["epoch", "train_loss_regularized_epoch"]].reset_index(drop=True))
val_df = (metrics.dropna(subset=["val_loss"])
          [["epoch", "val_loss"]].reset_index(drop=True))
score_df = (metrics.dropna(subset=["val_score"])
            [["epoch", "val_score"]].reset_index(drop=True))

lr_df = pd.read_excel(
    DOT_ENV.results_dir / "learning_rate_schedule_500_epochs.xlsx",
    sheet_name="LR_Schedule_500",
    usecols=["Epoch", "Learning Rate", "Phase"],
)
lr_df.columns = ["epoch", "lr", "phase"]
lr_df["epoch"] = lr_df["epoch"] - 1

warmup = lr_df[lr_df["phase"] == "Warm-up"]
cosine = lr_df[lr_df["phase"] == "Cosine Annealing WR"]

# Deteksi restart
cosine_vals = cosine["lr"].values
cosine_eps = cosine["epoch"].values
restart_epochs = [
    cosine_eps[i]
    for i in range(1, len(cosine_vals) - 1)
    if cosine_vals[i] >= cosine_vals[i - 1]
       and cosine_vals[i] >= cosine_vals[i + 1]
       and cosine_vals[i] > 4.98e-3
]
warmup_end = warmup["epoch"].max()

# Styling
sns.set_theme(style="whitegrid", font="serif")
plt.rcParams.update(
    {
        "axes.facecolor"  : "#FAFAFA",
        "figure.facecolor": "#FFFFFF",
        "grid.color"      : "#E8E8E8",
        "grid.linewidth"  : 0.7,
        "axes.edgecolor"  : "#CCCCCC",
        "axes.labelcolor" : "#333333",
        "xtick.color"     : "#555555",
        "ytick.color"     : "#555555",
        "xtick.labelsize" : 10,
        "ytick.labelsize" : 10,
        "axes.labelsize"  : 12,
        "text.color"      : "#222222",
        "font.family"     : "serif",
    }
)

TRAIN_COLOR = "#4A90D9"  # biru
TRAIN_REG_COLOR = "#1A5FA8"  # biru tua
VAL_COLOR = "#E8834A"  # oranye
SCORE_COLOR = "#27AE60"  # hijau (sumbu kanan)
WARMUP_COLOR = "#F5A623"
COSINE_COLOR = "#7B68EE"
RESTART_COLOR = "#CC3333"

# Figure: 2 panel, shared x
fig, (ax_loss, ax_lr) = plt.subplots(
    2, 1,
    figsize=(14, 8),
    sharex=True,
    gridspec_kw={"height_ratios": [2.2, 1], "hspace": 0.06},
)

# Dual y-axis untuk panel atas
ax_score = ax_loss.twinx()

# PANEL ATAS Loss (kiri) + Val Score (kanan)

# Shading fase
ax_loss.axvspan(
    warmup["epoch"].min(), warmup_end,
    color=WARMUP_COLOR, alpha=0.07, zorder=0
)
ax_loss.axvspan(
    warmup_end, cosine["epoch"].max(),
    color=COSINE_COLOR, alpha=0.04, zorder=0
)

# Garis restart & warm-up end
for ep in restart_epochs:
    ax_loss.axvline(
        ep, color=RESTART_COLOR, linewidth=1.0,
        linestyle="--", alpha=0.65, zorder=2
    )
ax_loss.axvline(
    warmup_end, color="#AAAAAA", linewidth=1.0,
    linestyle=":", zorder=2
)

# Sumbu kiri: 3 garis loss
ax_loss.plot(
    train_df["epoch"], train_df["train_loss_epoch"],
    color=TRAIN_COLOR, linewidth=1.8, zorder=3, alpha=0.9,
    label="Train Loss"
)
ax_loss.plot(
    train_reg["epoch"], train_reg["train_loss_regularized_epoch"],
    color=TRAIN_REG_COLOR, linewidth=1.6, zorder=3, alpha=0.85,
    linestyle="--", label="Train Loss (Regularized)"
)
ax_loss.plot(
    val_df["epoch"], val_df["val_loss"],
    color=VAL_COLOR, linewidth=1.8, zorder=3, alpha=0.9,
    label="Validation Loss"
)

# Log scale sumbu kiri
all_loss = np.concatenate(
    [
        train_df["train_loss_epoch"].values,
        train_reg["train_loss_regularized_epoch"].values,
        val_df["val_loss"].values,
    ]
)
ax_loss.set_yscale("log")
ax_loss.set_ylim(all_loss.min() * 0.92, all_loss.max() * 1.08)
ax_loss.yaxis.set_major_locator(mticker.LogLocator(base=10, numticks=8))
ax_loss.yaxis.set_major_formatter(mticker.ScalarFormatter())
ax_loss.yaxis.set_minor_formatter(mticker.NullFormatter())
ax_loss.set_ylabel("Loss (log scale)", labelpad=10, color="#333333")

# Sumbu kanan: val_score
ax_score.plot(
    score_df["epoch"], score_df["val_score"],
    color=SCORE_COLOR, linewidth=1.6, zorder=3, alpha=0.85,
    linestyle="-.", label="Validation Score"
)

s_min, s_max = score_df["val_score"].min(), score_df["val_score"].max()
ax_score.set_yscale("log")
ax_score.set_ylim(s_min * 0.92, s_max * 1.08)
ax_score.yaxis.set_major_locator(mticker.LogLocator(base=10, numticks=6))
ax_score.yaxis.set_major_formatter(mticker.ScalarFormatter())
ax_score.yaxis.set_minor_formatter(mticker.NullFormatter())

ax_score.set_ylabel("Validation Score (log scale)", labelpad=10, color=SCORE_COLOR)
ax_score.tick_params(axis="y", colors=SCORE_COLOR, labelsize=10)
ax_score.spines["right"].set_edgecolor(SCORE_COLOR)
ax_score.spines["right"].set_alpha(0.6)

# Label nilai akhir
for ax_, arr, ep_col, col, fmt in [
    (ax_loss, train_df["train_loss_epoch"].values, train_df["epoch"], TRAIN_COLOR, ".2f"),
    (ax_loss, train_reg["train_loss_regularized_epoch"].values, train_reg["epoch"], TRAIN_REG_COLOR, ".2f"),
    (ax_loss, val_df["val_loss"].values, val_df["epoch"], VAL_COLOR, ".2f"),
    (ax_score, score_df["val_score"].values, score_df["epoch"], SCORE_COLOR, ".3f"),
]:
    last_ep = ep_col.iloc[-1]
    last_val = arr[-1]
    ax_.plot(last_ep, last_val, "o", color=col, markersize=5, zorder=6)
    ax_.annotate(
        f"  {last_val:{fmt}}", xy=(last_ep, last_val),
        fontsize=8, color=col, va="center", fontweight="bold"
    )

# Annotasi restart
for i, ep in enumerate(restart_epochs):
    ax_loss.text(
        ep + 1.5, all_loss.max() * 0.97,
        f"R{i + 1}", fontsize=8, color=RESTART_COLOR,
        fontweight="bold", va="top", zorder=6
        )

# Legenda gabungan (kiri + kanan)
loss_handles = [
    Line2D(
        [0], [0], color=TRAIN_COLOR, linewidth=2.0,
        label="Train Loss"
    ),
    Line2D(
        [0], [0], color=TRAIN_REG_COLOR, linewidth=1.8,
        linestyle="--", label="Train Loss (Regularized)"
    ),
    Line2D(
        [0], [0], color=VAL_COLOR, linewidth=2.0,
        label="Validation Loss"
    ),
    Line2D(
        [0], [0], color=SCORE_COLOR, linewidth=1.8,
        linestyle="-.", label="Validation Score  (kanan)"
    ),
    mpatches.Patch(color=WARMUP_COLOR, alpha=0.5, label="Warm-up phase"),
    mpatches.Patch(color=COSINE_COLOR, alpha=0.4, label="Cosine Annealing WR"),
    Line2D(
        [0], [0], color=RESTART_COLOR, linewidth=1.5,
        linestyle="--", label=f"Cosine restart (R1 - R{len(restart_epochs)})"
    ),
]
leg = ax_loss.legend(
    handles=loss_handles, loc="upper right",
    fontsize=9.0, framealpha=0.88,
    facecolor="#FFFFFF", edgecolor="#CCCCCC",
    handlelength=2.2
)
for t in leg.get_texts():
    t.set_color("#222222")

# PANEL BAWAH Learning Rate

ax_lr.axvspan(
    warmup["epoch"].min(), warmup_end,
    color=WARMUP_COLOR, alpha=0.07, zorder=0
)
ax_lr.axvspan(
    warmup_end, cosine["epoch"].max(),
    color=COSINE_COLOR, alpha=0.04, zorder=0
)

for i, ep in enumerate(restart_epochs):
    lr_val = cosine[cosine["epoch"] == ep]["lr"].values[0]
    ax_lr.axvline(
        ep, color=RESTART_COLOR, linewidth=1.0,
        linestyle="--", alpha=0.65, zorder=2
    )
    ax_lr.text(
        ep + 1.5, lr_val * 1.01, f"R{i + 1}",
        fontsize=8, color=RESTART_COLOR, fontweight="bold",
        va="bottom", zorder=6
        )

ax_lr.axvline(
    warmup_end, color="#AAAAAA", linewidth=1.0,
    linestyle=":", zorder=2
)

ax_lr.plot(
    warmup["epoch"], warmup["lr"],
    color=WARMUP_COLOR, linewidth=2.2, zorder=3, solid_capstyle="round"
)
ax_lr.plot(
    cosine["epoch"], cosine["lr"],
    color=COSINE_COLOR, linewidth=1.8, zorder=3, solid_capstyle="round"
)

ax_lr.set_xlabel("Epoch", labelpad=10)
ax_lr.set_ylabel("Learning Rate", labelpad=10)

ax_lr.set_xlim(-2, lr_df["epoch"].max() + 5)
lr_y_max = lr_df["lr"].max()
ax_lr.set_ylim(-0.0001, lr_y_max * 1.12)
ax_lr.yaxis.set_major_locator(mticker.MaxNLocator(nbins=5, prune="both"))
ax_lr.yaxis.set_major_formatter(
    mticker.FuncFormatter(
        lambda x, _: f"{x:.3f}" if x >= 0.001 else f"{x:.2e}"
    )
)
ax_lr.xaxis.set_major_locator(mticker.MultipleLocator(50))
ax_lr.xaxis.set_minor_locator(mticker.MultipleLocator(10))

lr_handles = [
    Line2D([0], [0], color=WARMUP_COLOR, linewidth=2.2, label="Warm-up"),
    Line2D([0], [0], color=COSINE_COLOR, linewidth=1.8, label="Cosine Annealing WR"),
]
leg2 = ax_lr.legend(
    handles=lr_handles, loc="upper right",
    fontsize=9.5, framealpha=0.85,
    facecolor="#FFFFFF", edgecolor="#CCCCCC",
    handlelength=2.0
)
for t in leg2.get_texts():
    t.set_color("#222222")

plt.savefig(
    DOT_ENV.plots_dir / "training_metrics_vs_learning_rate.png", dpi=150, bbox_inches="tight",
    facecolor=fig.get_facecolor()
    )

print("Plot disimpan!")
plt.show()
# %% [markdown]
# # Evaluasi model LocalGLMnet
# %% [markdown]
# ## Load model yang sudah dilatih
# %%
from ta_module.config import load_last_train_metadata

last_train_metadata = load_last_train_metadata(DOT_ENV.last_train_metadata_file)
# %%
last_train_metadata
# %%
localglmnet_models = []
for checkpoint_filepath in last_train_metadata.checkpoint_file_paths:
    localglmnet_model = create_localglmnet_model(create_lcn_layer())
    ckpt = torch.load(checkpoint_filepath, map_location=DEVICE)
    state_dict = ckpt["state_dict"]

    # sesuaikan dengan nama atribut di LightningModule-mu
    prefix = "model."
    state_dict = {
        k[len(prefix):]: v
        for k, v in state_dict.items()
        if k.startswith(prefix)
    }

    localglmnet_model.load_state_dict(state_dict)
    localglmnet_model.eval()
    localglmnet_models.append(localglmnet_model)

print(localglmnet_models)
# %%
from ta_module.models import EnsembleLocalGLMNet

localglmnet_ensemble = EnsembleLocalGLMNet(
    models=localglmnet_models,
).to(DEVICE)

print(localglmnet_ensemble)
# %% [markdown]
# ## Evaluasi performa
# %%
from torch import Tensor
from ta_module.utils import denormalize


def inverse_transform_mortality(x: Tensor, mean: Tensor, std: Tensor) -> Tensor:
    denormalized = denormalize(x, mean, std)
    detransformed = transform_fn.inv(denormalized)

    return detransformed


def inverse_transform_male_mortality(x: Tensor) -> Tensor:
    return inverse_transform_mortality(x, train_male_mean, train_male_std)


def inverse_transform_female_mortality(x: Tensor) -> Tensor:
    return inverse_transform_mortality(x, train_female_mean, train_female_std)
# %%
from ta_module.utils import recursive_forecast

test_male_dataloader = DataLoader(test_male_dataset, batch_size=len(test_male_dataset), shuffle=False)
male_test_loss = 0.0
male_test_score = 0.0

for batch in test_male_dataloader:
    x, y = batch
    x = x[:1, :, :]
    h = y.shape[0]

    with torch.no_grad():
        y_pred = recursive_forecast(
            model=localglmnet_ensemble,
            x=x,
            forecast_horizon=h,
            n_sim=1
        )

    y_pred = y_pred.permute(1, 0, 2)
    y_detransformed = inverse_transform_male_mortality(y)
    y_pred_detransformed = inverse_transform_male_mortality(y_pred)

    male_test_loss = F.mse_loss(y_detransformed, y_pred_detransformed).detach()
    male_test_score = F.l1_loss(y_detransformed, y_pred_detransformed).detach()

print(f"Male test MSE = {male_test_loss:.6f}")
print(f"Male test MAE = {male_test_score:.6f}")
# %%
male_individual_test_loss = []
male_individual_test_score = []
for batch in test_male_dataloader:
    x, y = batch
    x = x[:1, :, :]
    h = y.shape[0]

    y_detransformed = inverse_transform_male_mortality(y)
    for i in range(len(localglmnet_ensemble.models)):
        model = localglmnet_ensemble.models[i]
        with torch.no_grad():
            y_pred = recursive_forecast(
                model=model,
                x=x,
                forecast_horizon=h,
                n_sim=1
            )

        y_pred = y_pred.permute(1, 0, 2)
        y_pred_detransformed = inverse_transform_male_mortality(y_pred)

        test_loss = F.mse_loss(y_detransformed, y_pred_detransformed).detach()
        test_score = F.l1_loss(y_detransformed, y_pred_detransformed).detach()

        male_individual_test_loss.append(test_loss)
        male_individual_test_score.append(test_score)

print(55 * "=")
print("Male test individu model LocalGLMnet:")
print(55 * "=")
for i in range(len(male_individual_test_loss)):
    print(f"LocalGLMnet{i+1} : MSE = {male_individual_test_loss[i]:.6f}; MAE = {male_individual_test_score[i]:.6f}")
# %%
test_female_dataloader = DataLoader(test_female_dataset, batch_size=len(test_male_dataset), shuffle=False)
female_test_loss = 0.0
female_test_score = 0.0

for batch in test_female_dataloader:
    x, y = batch
    x = x[:1, :, :]
    h = y.shape[0]
    with torch.no_grad():
        y_pred = recursive_forecast(
            model=localglmnet_ensemble,
            x=x,
            forecast_horizon=h,
            n_sim=1
        )

    y_pred = y_pred.permute(1, 0, 2)
    y_detransformed = inverse_transform_female_mortality(y)
    y_pred_detransformed = inverse_transform_female_mortality(y_pred)

    female_test_loss = F.mse_loss(y_detransformed, y_pred_detransformed).detach()
    female_test_score = F.l1_loss(y_detransformed, y_pred_detransformed).detach()

print(f"Female test MSE = {female_test_loss:.6f}")
print(f"Female test MAE = {female_test_score:.6f}")
# %%
female_individual_test_loss = []
female_individual_test_score = []
for batch in test_female_dataloader:
    x, y = batch
    x = x[:1, :, :]
    h = y.shape[0]
    y_detransformed = inverse_transform_male_mortality(y)
    for i in range(len(localglmnet_ensemble.models)):
        model = localglmnet_ensemble.models[i]
        with torch.no_grad():
            y_pred = recursive_forecast(
                model=model,
                x=x,
                forecast_horizon=h,
                n_sim=1
            )

        y_pred = y_pred.permute(1, 0, 2)
        y_pred_detransformed = inverse_transform_male_mortality(y_pred)

        test_loss = F.mse_loss(y_detransformed, y_pred_detransformed).detach()
        test_score = F.l1_loss(y_detransformed, y_pred_detransformed).detach()

        female_individual_test_loss.append(test_loss)
        female_individual_test_score.append(test_score)

print(55 * "=")
print("Female test individu model LocalGLMnet:")
print(55 * "=")
for i in range(len(female_individual_test_score)):
    print(f"LocalGLMnet{i+1} : MSE = {female_individual_test_loss[i]:.6f}; MAE = {female_individual_test_score[i]:.6f}")
# %%
print(f"Average MSE = {(male_test_loss + female_test_loss) / 2:.6f}")
print(f"Average MAE = {(male_test_score + female_test_score) / 2:.6f}")
# %%
print(55 * "=")
print("Average test individu model LocalGLMnet:")
print(55 * "=")
for i in range(len(female_individual_test_score)):
    print(f"LocalGLMnet{i+1} : MSE = {((female_individual_test_loss[i] + male_individual_test_loss[i]) / 2):.6f}; MAE = {((female_individual_test_score[i] + male_individual_test_score[i]) / 2):.6f}")
# %% [markdown]
# ## Plot peramalan data tes
# %%
y_male_pred, y_male_test = None, None
y_female_pred, y_female_test = None, None

for batch in test_male_dataloader:
    x, y = batch
    with torch.no_grad():
        y_pred = localglmnet_ensemble(x)

    y_pred = inverse_transform_male_mortality(y_pred)
    y_test = inverse_transform_male_mortality(y)

    y_male_pred = y_pred
    y_male_test = y_test

for batch in test_female_dataloader:
    x, y = batch
    with torch.no_grad():
        y_pred = localglmnet_ensemble(x)

    y_pred = inverse_transform_female_mortality(y_pred)
    y_test = inverse_transform_female_mortality(y)

    y_female_pred = y_pred
    y_female_test = y_test
# %%
y_male_pred = y_male_pred.squeeze(1)
y_male_test = y_male_test.squeeze(1)

y_female_pred = y_female_pred.squeeze(1)
y_female_test = y_female_test.squeeze(1)
# %%
from matplotlib.gridspec import GridSpec
import matplotlib.dates as mdates


def plot_tes_vs_peramalan(
    start_age: int,
    end_age: int,
    factor: int,
    gender: int
):
    assert 0 <= start_age <= end_age <= 100
    assert factor >= 1
    assert gender == 0 or gender == 1

    if gender == 1:
        y_pred = y_male_pred.cpu()
        y_test = y_male_test.cpu()
    else:
        y_pred = y_female_pred.cpu()
        y_test = y_female_test.cpu()

    residual = y_test - y_pred
    x = pd.date_range(start="2014", end="2024", freq="YS")

    sns.set_theme(style="whitegrid", context="paper")

    PALETTE = {
        "pred" : "#378ADD",  # blue
        "real" : "#1D9E75",  # teal
        "resid": "#D85A30",  # coral
    }

    ages = range(start_age, end_age + 1, factor)
    n_plots = len(ages)  # 11 plots

    ncols = 3
    nrows = (n_plots + 1) // ncols  # 6 rows

    fig = plt.figure(figsize=(18, nrows * 3.2))
    fig.patch.set_facecolor("#F8F9FA")

    gs = GridSpec(nrows, ncols, figure=fig, hspace=0.30, wspace=0.15)

    for idx, usia in enumerate(ages):
        row, col = divmod(idx, ncols)
        ax = fig.add_subplot(gs[row, col])
        ax.set_facecolor("white")

        ax.plot(
            x, y_pred[:, usia],
            color=PALETTE["pred"], lw=1.8, ls="--",
            label="Prediksi", zorder=3
        )
        ax.plot(
            x, y_test[:, usia],
            color=PALETTE["real"], lw=2, ls="-",
            label="Aktual", zorder=4
        )
        ax.plot(
            x, residual[:, usia],
            color=PALETTE["resid"], lw=1.4, ls="-.",
            label="Residual", alpha=0.75, zorder=2
        )

        ax.fill_between(
            x, y_pred[:, usia], y_test[:, usia],
            alpha=0.08, color=PALETTE["pred"]
        )

        ax.axhline(0, color="#888780", lw=0.7, ls=":", zorder=1)

        ax.set_title(
            f"Usia {usia}", fontsize=10, fontweight="semibold",
            color="#2C2C2A", pad=6
        )
        ax.set_xlabel("Tahun", fontsize=8.5, color="#5F5E5A")
        ax.set_ylabel("Mortality Rate", fontsize=8.5, color="#5F5E5A")

        ax.tick_params(axis="both", labelsize=7.5, colors="#5F5E5A")
        ax.xaxis.set_major_formatter(
            mticker.FuncFormatter(
                lambda v, _: str(int(mdates.num2date(v).year)) if hasattr(mdates, 'num2date') else ""
            )
        )

        for spine in ax.spines.values():
            spine.set_edgecolor("#D3D1C7")
            spine.set_linewidth(0.6)

        ax.grid(True, color="#EEEDE8", linewidth=0.5, zorder=0)

    # Sembunyikan subplot kosong jika jumlah plot ganjil
    if n_plots % ncols != 0:
        fig.add_subplot(gs[nrows - 1, ncols - 1]).set_visible(False)

    fig.suptitle("Laki-laki" if gender == 1 else "Perempuan", fontsize=16, fontweight="bold")
    # Legend bersama di luar plot
    handles, labels = fig.axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="lower center", ncol=3,
        fontsize=9, frameon=True,
        framealpha=0.9, edgecolor="#D3D1C7",
        bbox_to_anchor=(0.5, -0.015)
    )

    plt.savefig(
        DOT_ENV.plots_dir / f"peramalan_mortalitas_data_tes_{"laki-laki" if gender == 1 else "perempuan"}_usia_{start_age}_{end_age}.png",
        dpi=300, bbox_inches="tight",
        facecolor=fig.get_facecolor()
        )

    plt.show()
# %%
plot_tes_vs_peramalan(
    start_age=0,
    end_age=100,
    factor=10,
    gender=0
)
# %%
plot_tes_vs_peramalan(
    start_age=0,
    end_age=100,
    factor=10,
    gender=1
)
# %% [markdown]
# # Interpretasi pemodelan
# %% [markdown]
# ## Perhitungan regression attention
# %%
male_dataset = create_male_dataset_split(M_male_train)
male_dataloader = DataLoader(male_dataset, batch_size=len(male_dataset))
male_mortality = None

male_reg_att = []
for batch in male_dataloader:
    x, _ = batch
    male_mortality = x

    window = []
    for model in localglmnet_ensemble.models:
        assert isinstance(model, LocalGLMnet)
        reg_att = model.get_regression_attention(x)
        window.append(reg_att)

    window = torch.cat(window).to(DEVICE)
    male_reg_att.append(window)

male_reg_att = torch.cat(male_reg_att).to(DEVICE)

print(male_reg_att.shape)
print(male_mortality.shape)
# %%
female_dataset = create_female_dataset_split(M_female_train)
female_dataloader = DataLoader(female_dataset, batch_size=len(female_dataset))
female_mortality = None

female_reg_att = []
for batch in female_dataloader:
    x, _ = batch
    female_mortality = x
    window = []
    for model in localglmnet_ensemble.models:
        assert isinstance(model, LocalGLMnet)
        reg_att = model.get_regression_attention(x)
        window.append(reg_att)

    window = torch.cat(window).to(DEVICE)
    female_reg_att.append(window)

female_reg_att = torch.cat(female_reg_att).to(DEVICE)

print(female_reg_att.shape)
print(female_mortality.shape)
# %%
n_models = 10
n_data = 43  # 430 / 10

male_reg_att_dfs = {}

for i in range(n_models):
    start = i * n_data
    end   = start + n_data

    # Slice: (43, 10, 101)
    chunk = male_reg_att[start:end]          # (43, 10, 101)

    # Buat MultiIndex: level-0 = data_idx (0-42), level-1 = row_idx (0-9)
    idx = pd.MultiIndex.from_product(
        [range(n_data), range(10, 0, -1)],    # (43, 10)
        names=["data_idx", "lag_idx"]
    )

    # Reshape (43, 10, 101) → (43*10, 101)
    df = pd.DataFrame(
        chunk.cpu().numpy().reshape(n_data * 10, 101),
        index=idx
    )

    male_reg_att_dfs[f"model_{i+1}"] = df
# %%
n_models = 10
n_data = 43  # 430 / 10

female_reg_att_dfs = {}

for i in range(n_models):
    start = i * n_data
    end   = start + n_data

    # Slice: (43, 10, 101)
    chunk = female_reg_att[start:end]          # (43, 10, 101)

    # Buat MultiIndex: level-0 = data_idx (0-42), level-1 = row_idx (0-9)
    idx = pd.MultiIndex.from_product(
        [range(n_data), range(10, 0, -1)],    # (43, 10)
        names=["data_idx", "lag_idx"]
    )

    # Reshape (43, 10, 101) → (43*10, 101)
    df = pd.DataFrame(
        chunk.cpu().numpy().reshape(n_data * 10, 101),
        index=idx
    )

    female_reg_att_dfs[f"model_{i+1}"] = df
# %%
male_reg_att_dfs["model_1"].to_csv(DOT_ENV.results_dir / "male_reg_att_model_1.csv", sep=";", decimal=",")
female_reg_att_dfs["model_1"].to_csv(DOT_ENV.results_dir / "female_reg_att_model_1.csv", sep=";", decimal=",")
# %% [markdown]
# ## Plot mortalitas vs regression attention
# %%
import numpy as np

# warna per lag
_BASE_COLORS = [
    "#E63946",   # lag 0  ﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿ vivid red
    "#F4A261",   # lag 1  ﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿ warm orange
    "#2A9D8F",   # lag 2  ﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿ teal
    "#457B9D",   # lag 3  ﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿ steel blue
    "#A8DADC",   # lag 4  ﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿ pale cyan
    "#8338EC",   # lag 5  ﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿ violet
    "#FB5607",   # lag 6  ﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿ burnt orange
    "#3A86FF",   # lag 7  ﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿ bright blue
    "#06D6A0",   # lag 8  ﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿ mint
    "#FFBE0B",   # lag 9  ﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿ golden yellow
]


def _to_numpy(t) -> np.ndarray:
    """Konversi torch.Tensor / np.ndarray ﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿ float64 numpy."""
    if hasattr(t, "detach"):
        return t.detach().cpu().numpy().astype(float)
    return np.asarray(t, dtype=float)


def _flatten3d(arr: np.ndarray) -> np.ndarray:
    """
    Flatten semua dimensi kecuali dua terakhir (n_lags, n_ages).
    Input  : (..., n_lags, n_ages)
    Output : (N, n_lags, n_ages)
    """
    *leading, n_lags, n_ages = arr.shape
    N = int(np.prod(leading)) if leading else 1
    return arr.reshape(N, n_lags, n_ages)


def _smart_fmt(v: float, _) -> str:
    if v == 0:
        return "0"
    av = abs(v)
    if av < 0.001:
        return f"{v:.4f}"
    if av < 0.01:
        return f"{v:.3f}"
    if av < 0.1:
        return f"{v:.2f}"
    return f"{v:.2f}"


# FUNGSI UTAMA
def plot_attention_regression(
    mortality_tensor: Tensor,
    attention_tensor: Tensor,
    save_path: Path,
    start_age:   int  = 0,
    end_age:     int  = 100,
    factor:      int  = 10,
    suptitle:    str  = "LocalGLMnet - Attention Regression Coefficients",
    panel_cols:  int  = 4,
    dpi:         int  = 300,
) -> None:
    """
    Plot scatter (m_{t,x}, beta_{t-s,x}) untuk setiap usia x dan lag s.

    Parameters
    ----------
    mortality_tensor : (..., n_lags, n_ages)
        Data mortalitas. Dua dimensi terakhir wajib (n_lags, n_ages).
        Semua dimensi sebelumnya di-flatten menjadi sumbu sampel.

    attention_tensor : (..., n_lags, n_ages)
        Attention coefficient. Dua dimensi terakhir wajib sama dengan
        mortality_tensor.

    start_age, end_age, factor : int
        Panel usia: start_age, start_age+factor, ..., end_age.

    suptitle : str
        Judul besar di atas semua panel.

    panel_cols : int
        Jumlah kolom panel.

    save_path : str | None
        Path simpan PNG; None = tidak disimpan.

    dpi : int
        Resolusi gambar.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    # 0. konversi & flatten
    mx  = _to_numpy(mortality_tensor)
    att = _to_numpy(attention_tensor)

    if mx.ndim < 2 or att.ndim < 2:
        raise ValueError("Kedua tensor harus minimal 2 dimensi (..., n_lags, n_ages).")

    # pastikan minimal 3D sebelum flatten
    if mx.ndim == 2:
        mx = mx[np.newaxis]
    if att.ndim == 2:
        att = att[np.newaxis]

    mx_flat  = _flatten3d(mx)    # (N_mort, n_lags, n_ages)
    att_flat = _flatten3d(att)   # (N_att,  n_lags, n_ages)

    N_mort, n_lags_mx, n_ages_mx   = mx_flat.shape
    N_att,  n_lags_att, n_ages_att = att_flat.shape

    # validasi dimensi terakhir
    if n_lags_mx != n_lags_att:
        raise ValueError(
            f"Dimensi lag tidak cocok: mortality={n_lags_mx}, attention={n_lags_att}."
        )
    if n_ages_mx != n_ages_att:
        raise ValueError(
            f"Dimensi usia tidak cocok: mortality={n_ages_mx}, attention={n_ages_att}."
        )

    n_lags = n_lags_mx
    n_ages = n_ages_mx

    # 1. sinkronisasi N_mort vs N_att via repeat
    if N_att != N_mort:
        if N_att % N_mort == 0:
            k = N_att // N_mort
            mx_flat = np.repeat(mx_flat, k, axis=0)   # (N_att, n_lags, n_ages)
        elif N_mort % N_att == 0:
            k = N_mort // N_att
            att_flat = np.repeat(att_flat, k, axis=0) # (N_mort, n_lags, n_ages)
        else:
            # Tidak habis dibagi: potong ke ukuran terkecil
            N_min = min(N_mort, N_att)
            mx_flat  = mx_flat[:N_min]
            att_flat = att_flat[:N_min]
            import warnings
            warnings.warn(
                f"N_mort={N_mort} dan N_att={N_att} tidak saling habis dibagi. "
                f"Dipotong ke {N_min} sampel pertama.",
                UserWarning,
                stacklevel=2,
            )

    N = mx_flat.shape[0]   # ukuran sampel final

    # 2. daftar usia panel
    ages_step = list(range(start_age, end_age, factor))
    if end_age not in ages_step:
        ages_step.append(end_age)
    ages_step = [a for a in ages_step if 0 <= a < n_ages]
    n_panels  = len(ages_step)
    n_rows    = (n_panels + panel_cols - 1) // panel_cols

    # 3. warna lag
    if n_lags <= len(_BASE_COLORS):
        lag_colors = _BASE_COLORS[:n_lags]
    else:
        cmap = plt.get_cmap("tab20", n_lags)
        lag_colors = [cmap(i) for i in range(n_lags)]

    # 4. figure layout
    sns.set_theme(style="white", font_scale=0.85)

    panel_w, panel_h = 3.0, 2.8
    fig_w      = panel_w * panel_cols
    top_pad    = 0.75
    bottom_pad = 1.50
    fig_h      = top_pad + panel_h * n_rows + bottom_pad

    top_frac    = 1.0 - top_pad    / fig_h
    bottom_frac = bottom_pad / fig_h

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="white")

    gs = fig.add_gridspec(
        n_rows, panel_cols,
        left=0.06, right=0.99,
        top=top_frac, bottom=bottom_frac,
        hspace=0.60, wspace=0.50,
    )

    # 5. suptitle & subtitle
    fig.suptitle(
        suptitle,
        fontsize=12, fontweight="bold", color="#1a1a2e",
        y=top_frac + (top_pad * 0.60) / fig_h,
        va="bottom",
    )

    # 6. plot panel
    for idx, age in enumerate(ages_step):
        row, col = divmod(idx, panel_cols)
        ax = fig.add_subplot(gs[row, col])

        for s in range(n_lags):   # lag kecil di depan
            x_vals = mx_flat[:,  s, age]   # (N,) mortalitas
            y_vals = att_flat[:, s, age]   # (N,) attention

            ax.scatter(
                x_vals, y_vals,
                s=4, color=lag_colors[s],
                alpha=0.45, linewidths=0,
                zorder=n_lags - s,
            )

        # styling
        ax.set_title(str(age), fontsize=9, fontweight="semibold",
                     pad=3, color="#1a1a2e")
        ax.tick_params(axis="both", labelsize=6.5, length=2.5, color="#666")

        ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=3, prune="both"))
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(_smart_fmt))
        ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=4, prune="both"))
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(_smart_fmt))

        sns.despine(ax=ax, top=True, right=True)
        for sp in ("bottom", "left"):
            ax.spines[sp].set_linewidth(0.6)
            ax.spines[sp].set_color("#bbb")
        ax.set_facecolor("white")

        ax.set_ylabel("Regression Attention", fontsize=7.5,
                          color="#333", labelpad=4)
        ax.set_xlabel("mx (transformed)", fontsize=7.5, color="#333", labelpad=2)

    # 7. legend bersama
    legend_handles = [
        plt.scatter([], [], s=26, color=lag_colors[n_lags - s - 1],
                    label=str(s + 1), alpha=0.9, linewidths=0)
        for s in range(n_lags)
    ]
    fig.legend(
        handles=legend_handles,
        title="lag", title_fontsize=8, fontsize=7.5,
        ncol=min(n_lags, 10),
        loc="lower center",
        bbox_to_anchor=(0.5, bottom_frac * 0.25),
        frameon=True, framealpha=0.92, edgecolor="#ddd",
        fancybox=False, columnspacing=0.9,
        handletextpad=0.3, borderpad=0.5,
    )

    # 9. simpan
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight", facecolor="white")
        print(f"Saved -> {save_path}")

    plt.show()
# %% [markdown]
# ### Laki-laki
# %%
n = 1
assert 0 < n <= 10

reg_att = male_reg_att[(n-1) * 43:n * 43]

plot_attention_regression(
    mortality_tensor=male_mortality,
    attention_tensor=reg_att,
    save_path=DOT_ENV.plots_dir / f"mortalitas_vs_reg_attention_laki-laki_localglmnet{n}.png",
    suptitle=f"Laki-laki",
    panel_cols=3,
    factor=20,
    dpi=300
)
# %% [markdown]
# ### Perempuan
# %%
n = 1
assert 0 < n <= 10

reg_att = female_reg_att[(n-1) * 43:n * 43]

plot_attention_regression(
    mortality_tensor=female_mortality,
    attention_tensor=reg_att,
    save_path=DOT_ENV.plots_dir / f"mortalitas_vs_reg_attention_perempuan_localglmnet{n}.png",
    suptitle=f"Perempuan",
    panel_cols=3,
    dpi=300,
    factor=20
)
# %% [markdown]
# ## Plot attention contribution
# %%
import numpy as np

# warna per lag
_BASE_COLORS = [
    "#E63946",   # lag 0  ﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿ vivid red
    "#F4A261",   # lag 1  ﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿ warm orange
    "#2A9D8F",   # lag 2  ﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿ teal
    "#457B9D",   # lag 3  ﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿ steel blue
    "#A8DADC",   # lag 4  ﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿ pale cyan
    "#8338EC",   # lag 5  ﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿ violet
    "#FB5607",   # lag 6  ﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿ burnt orange
    "#3A86FF",   # lag 7  ﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿ bright blue
    "#06D6A0",   # lag 8  ﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿ mint
    "#FFBE0B",   # lag 9  ﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿ golden yellow
]


def _to_numpy(t) -> np.ndarray:
    """Konversi torch.Tensor / np.ndarray ﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿ float64 numpy."""
    if hasattr(t, "detach"):
        return t.detach().cpu().numpy().astype(float)
    return np.asarray(t, dtype=float)


def _flatten3d(arr: np.ndarray) -> np.ndarray:
    """
    Flatten semua dimensi kecuali dua terakhir (n_lags, n_ages).
    Input  : (..., n_lags, n_ages)
    Output : (N, n_lags, n_ages)
    """
    *leading, n_lags, n_ages = arr.shape
    N = int(np.prod(leading)) if leading else 1
    return arr.reshape(N, n_lags, n_ages)


def _smart_fmt(v: float, _) -> str:
    if v == 0:
        return "0"
    av = abs(v)
    if av < 0.001:
        return f"{v:.4f}"
    if av < 0.01:
        return f"{v:.3f}"
    if av < 0.1:
        return f"{v:.2f}"
    return f"{v:.2f}"

# FUNGSI UTAMA

def plot_attention_contribution(
    mortality_tensor: Tensor,
    contribution_tensor: Tensor,
    save_path: Path,
    start_age:   int  = 0,
    end_age:     int  = 100,
    factor:      int  = 10,
    suptitle:    str  = "LocalGLMnet ﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿ Attention Regression Contribution",
    panel_cols:  int  = 3,
    dpi:         int  = 200,
) -> None:
    """
    Plot scatter (m_{t,x}, beta_{t-s,x}) untuk setiap usia x dan lag s.

    Parameters
    ----------
    mortality_tensor : (..., n_lags, n_ages)
        Data mortalitas. Dua dimensi terakhir wajib (n_lags, n_ages).
        Semua dimensi sebelumnya di-flatten menjadi sumbu sampel.

    contribution_tensor : (..., n_lags, n_ages)
        Attention coefficient. Dua dimensi terakhir wajib sama dengan
        mortality_tensor.

    start_age, end_age, factor : int
        Panel usia: start_age, start_age+factor, ..., end_age.

    suptitle : str
        Judul besar di atas semua panel.

    panel_cols : int
        Jumlah kolom panel.

    save_path : str | None
        Path simpan PNG; None = tidak disimpan.

    dpi : int
        Resolusi gambar.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    # 0. konversi & flatten
    mortality_tensor = mortality_tensor.cpu()
    contribution_tensor = contribution_tensor.cpu()
    att_contribution = _to_numpy(contribution_tensor)
    mx = mortality_tensor
    if mx.ndim < 2 or att_contribution.ndim < 2:
        raise ValueError("Kedua tensor harus minimal 2 dimensi (..., n_lags, n_ages).")

    # pastikan minimal 3D sebelum flatten
    if mx.ndim == 2:
        mx = mx[np.newaxis]
    if att_contribution.ndim == 2:
        att_contribution = att_contribution[np.newaxis]

    mx_flat  = _flatten3d(mx)    # (N_mort, n_lags, n_ages)
    att_flat = _flatten3d(att_contribution)   # (N_att,  n_lags, n_ages)

    N_mort, n_lags_mx, n_ages_mx   = mx_flat.shape
    N_att,  n_lags_att, n_ages_att = att_flat.shape

    # validasi dimensi terakhir
    if n_lags_mx != n_lags_att:
        raise ValueError(
            f"Dimensi lag tidak cocok: mortality={n_lags_mx}, attention={n_lags_att}."
        )
    if n_ages_mx != n_ages_att:
        raise ValueError(
            f"Dimensi usia tidak cocok: mortality={n_ages_mx}, attention={n_ages_att}."
        )

    n_lags = n_lags_mx
    n_ages = n_ages_mx

    # 1. sinkronisasi N_mort vs N_att via repeat
    if N_att != N_mort:
        if N_att % N_mort == 0:
            k = N_att // N_mort
            mx_flat = np.repeat(mx_flat, k, axis=0)   # (N_att, n_lags, n_ages)
        elif N_mort % N_att == 0:
            k = N_mort // N_att
            att_flat = np.repeat(att_flat, k, axis=0) # (N_mort, n_lags, n_ages)
        else:
            # Tidak habis dibagi: potong ke ukuran terkecil
            N_min = min(N_mort, N_att)
            mx_flat  = mx_flat[:N_min]
            att_flat = att_flat[:N_min]
            import warnings
            warnings.warn(
                f"N_mort={N_mort} dan N_att={N_att} tidak saling habis dibagi. "
                f"Dipotong ke {N_min} sampel pertama.",
                UserWarning,
                stacklevel=2,
            )

    N = mx_flat.shape[0]   # ukuran sampel final

    # 2. daftar usia panel
    ages_step = list(range(start_age, end_age, factor))
    if end_age not in ages_step:
        ages_step.append(end_age)
    ages_step = [a for a in ages_step if 0 <= a < n_ages]
    n_panels  = len(ages_step)
    n_rows    = (n_panels + panel_cols - 1) // panel_cols

    # 3. warna lag
    if n_lags <= len(_BASE_COLORS):
        lag_colors = _BASE_COLORS[:n_lags]
    else:
        cmap = plt.get_cmap("tab20", n_lags)
        lag_colors = [cmap(i) for i in range(n_lags)]

    # 4. figure layout
    panel_w, panel_h = 3.0, 2.8
    fig_w      = panel_w * panel_cols
    top_pad    = 0.75
    bottom_pad = 1.50
    fig_h      = top_pad + panel_h * n_rows + bottom_pad

    top_frac    = 1.0 - top_pad    / fig_h
    bottom_frac = bottom_pad / fig_h

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="white")

    gs = fig.add_gridspec(
        n_rows, panel_cols,
        left=0.06, right=0.99,
        top=top_frac, bottom=bottom_frac,
        hspace=0.60, wspace=0.50,
    )

    # 5. suptitle & subtitle
    fig.suptitle(
        suptitle,
        fontsize=12, fontweight="bold", color="#1a1a2e",
        y=top_frac + (top_pad * 0.60) / fig_h,
        va="bottom",
    )

    # 6. plot panel
    for idx, age in enumerate(ages_step):
        row, col = divmod(idx, panel_cols)
        ax = fig.add_subplot(gs[row, col])

        for s in range(n_lags):   # lag kecil di depan
            x_vals = mx_flat[:,  s, age]   # (N,) mortalitas
            y_vals = att_flat[:, s, age]   # (N,) attention

            ax.scatter(
                x_vals, y_vals,
                s=4, color=lag_colors[s],
                alpha=0.45, linewidths=0,
                zorder=n_lags - s,
            )

        # styling
        ax.set_title(str(age), fontsize=9, fontweight="semibold",
                     pad=3, color="#1a1a2e")
        ax.tick_params(axis="both", labelsize=6.5, length=2.5, color="#666")

        ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=3, prune="both"))
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(_smart_fmt))
        ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=4, prune="both"))
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(_smart_fmt))

        sns.despine(ax=ax, top=True, right=True)
        for sp in ("bottom", "left"):
            ax.spines[sp].set_linewidth(0.6)
            ax.spines[sp].set_color("#bbb")
        ax.set_facecolor("white")

        ax.set_ylabel("Contribution Value", fontsize=7.5,
                          color="#333", labelpad=4)
        ax.set_xlabel("mx (transformed)", fontsize=7.5, color="#333", labelpad=2)

    # 7. legend bersama
    legend_handles = [
        plt.scatter([], [], s=26, color=lag_colors[n_lags - s - 1],
                    label=str(s + 1), alpha=0.9, linewidths=0)
        for s in range(n_lags)
    ]
    fig.legend(
        handles=legend_handles,
        title="lag", title_fontsize=8, fontsize=7.5,
        ncol=min(n_lags, 10),
        loc="lower center",
        bbox_to_anchor=(0.5, bottom_frac * 0.25),
        frameon=True, framealpha=0.92, edgecolor="#ddd",
        fancybox=False, columnspacing=0.9,
        handletextpad=0.3, borderpad=0.5,
    )

    # 9. simpan
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight", facecolor="white")
        print(f"Saved -> {save_path}")

    plt.show()
# %% [markdown]
# ### Laki-laki
# %%
n = 1
assert 0 < n <= 10

reg_att = male_reg_att[(n-1) * 43:n * 43]
reg_att_cont = reg_att * male_mortality

plot_attention_contribution(
    mortality_tensor=male_mortality,
    contribution_tensor=reg_att_cont,
    save_path=DOT_ENV.plots_dir / f"mortalitas_vs_attention_contribution_laki-laki_localglmnet{n}.png",
    suptitle=f"Laki-laki",
    panel_cols=3,
    dpi=300,
    factor=20
)
# %% [markdown]
# ### Perempuan
# %%
n = 1
assert 0 < n <= 10

reg_att = female_reg_att[(n-1) * 43:n * 43]
reg_att_cont = reg_att * female_mortality

plot_attention_contribution(
    mortality_tensor=female_mortality,
    contribution_tensor=reg_att_cont,
    save_path=DOT_ENV.plots_dir / f"mortalitas_vs_attention_contribution_perempuan_localglmnet{n}.png",
    suptitle=f"Perempuan",
    panel_cols=3,
    dpi=300,
    factor=20
)
# %% [markdown]
# # Simulasi peramalan mortalitas
# %%
# Reset seed state
seed_everything(seed=CONFIG.seed, workers=True)
# %%
n_simulations = 10_000
forecast_horizon = 56
# %% [markdown]
# ## Laki-laki
# %%
male_residuals = None

male_dataloader = DataLoader(train_male_dataset, batch_size=len(train_male_dataset), shuffle=False)
for batch in male_dataloader:
    x, y = batch
    with torch.no_grad():
        y_pred = localglmnet_ensemble(x)
    male_residuals = y - y_pred

print(male_residuals[0])
print(male_residuals.shape)
# %%
from ta_module.utils import recursive_forecast_with_residual_bootstrap

male_simulations_file_path = DOT_ENV.results_dir / "male_mortality_simulations.pt"
if not male_simulations_file_path.exists():
    x, y = test_male_dataset[-1]
    # Dimensi = (10, W)
    x_in = torch.cat([x[1:, :], y])

    # Ubah menjadi dimensi = (1, 10, W)
    x_in = x_in.unsqueeze(0)

    print("=" * 100)
    print(f"Start recursive forecast with residual bootstrap for male mortality:")
    print("=" * 100)
    male_mortality_simulations = recursive_forecast_with_residual_bootstrap(
        model=localglmnet_ensemble,
        x=x_in,
        residuals=male_residuals,
        forecast_horizon=forecast_horizon,
        n_sim=n_simulations,
    )

    print("=" * 100)
    print(f"Simulation done!")
    print("=" * 100)

    # Return data scale to (0, 2)
    male_mortality_simulations = inverse_transform_male_mortality(male_mortality_simulations)

    print(f"Save simulations to {male_simulations_file_path}")
    torch.save(obj=male_mortality_simulations, f=male_simulations_file_path)
    print("All done!")
else:
    male_mortality_simulations = torch.load(male_simulations_file_path, map_location=DEVICE)

print(male_mortality_simulations[0])
print(male_mortality_simulations.shape)
# %% [markdown]
# ## Perempuan
# %%
female_residuals = None

female_dataloader = DataLoader(train_female_dataset, batch_size=len(train_female_dataset), shuffle=False)
for batch in female_dataloader:
    x, y = batch
    with torch.no_grad():
        y_pred = localglmnet_ensemble(x)
    female_residuals = y - y_pred

print(female_residuals[0])
print(female_residuals.shape)
# %%
female_simulations_file_path = DOT_ENV.results_dir / "female_mortality_simulations.pt"
if not female_simulations_file_path.exists():
    x, y = test_female_dataset[-1]
    # Dimensi = (10, W)
    x_in = torch.cat([x[1:, :], y])

    # Ubah menjadi dimensi = (1, 10, W)
    x_in = x_in.unsqueeze(0)

    print("=" * 100)
    print(f"Start recursive forecast with residual bootstrap for female mortality:")
    print("=" * 100)
    female_mortality_simulations = recursive_forecast_with_residual_bootstrap(
        model=localglmnet_ensemble,
        x=x_in,
        residuals=female_residuals,
        forecast_horizon=forecast_horizon,
        n_sim=n_simulations,
    )
    print("=" * 100)
    print(f"Simulation done!")
    print("=" * 100)

    # Return data scale to (0, 2)
    female_mortality_simulations = inverse_transform_female_mortality(female_mortality_simulations)

    print(f"Save simulations to {female_simulations_file_path}")
    torch.save(obj=female_mortality_simulations, f=female_simulations_file_path)
    print("All done!")
else:
    female_mortality_simulations = torch.load(female_simulations_file_path, map_location=DEVICE)

print(female_mortality_simulations[0])
print(female_mortality_simulations.shape)
# %% [markdown]
# ## Plot residual
# %%
male_residual_mean = male_residuals.mean(dim=0).squeeze(0).cpu().numpy()
male_residual_mean_abs = male_residuals.abs().mean(dim=0).squeeze(0).cpu().numpy()
male_residual_std = male_residuals.std(dim=0).squeeze(0).cpu().numpy()

female_residual_mean = female_residuals.mean(dim=0).squeeze(0).cpu().numpy()
female_residual_mean_abs = female_residuals.abs().mean(dim=0).squeeze(0).cpu().numpy()
female_residual_std = female_residuals.std(dim=0).squeeze(0).cpu().numpy()

ages = range(AGE_MIN, AGE_MAX + 1)

fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(21, 7), dpi=500)

ax[0].plot(ages, male_residual_mean, label="laki-laki")
ax[0].plot(ages, female_residual_mean, label="perempuan")
ax[0].set_xlabel("Usia")
ax[0].set_ylabel("Residual")
ax[0].set_title("Mean", fontsize=14, fontweight="semibold")
ax[0].legend()

ax[1].plot(ages, male_residual_mean_abs, label="laki-laki")
ax[1].plot(ages, female_residual_mean_abs, label="perempuan")
ax[1].set_xlabel("Usia")
ax[1].set_ylabel("Residual")
ax[1].set_title("Mean (Absolute Residual)", fontsize=14, fontweight="semibold")
ax[1].legend()

ax[2].plot(ages, male_residual_std, label="laki-laki")
ax[2].plot(ages, female_residual_std, label="perempuan")
ax[2].set_xlabel("Usia")
ax[2].set_ylabel("Residual")
ax[2].set_title("Std", fontsize=14, fontweight="semibold")
ax[2].legend()

fig.savefig(DOT_ENV.plots_dir / "mean_std_residual_peramalan_mortalitas.png")

plt.show()
# %% [markdown]
# # Pembuatan life table
# %% [markdown]
# ## Stokastik (simulasi peramalan)
# %% [markdown]
# ### Laki-laki
# %%
male_1qx = (2.0 * male_mortality_simulations) / (2.0 + male_mortality_simulations)
male_1px = 1.0 - male_1qx
# %%
from ta_module.actuarial import compute_ex, compute_kpx_table_from_xstart_dynamic

male_e45 = []
male_kpx_45_100 = []
for i in range(n_simulations):
    kpx_45_100 = compute_kpx_table_from_xstart_dynamic(
        p=male_1px[i],
        x_start=45,
        max_k=56,
        t0=0
    )

    male_kpx_45_100.append(kpx_45_100)
    kp45 = kpx_45_100[0]

    e45 = compute_ex(kp45)
    male_e45.append(e45)

male_e45 = torch.stack(male_e45)
male_kpx_45_100 = torch.stack(male_kpx_45_100)
# %%
_, male_e45_sorted_indices = torch.sort(male_e45, descending=False)

male_kpx_45_100_min = male_kpx_45_100[male_e45_sorted_indices[int(0.025 * n_simulations)]]
male_kpx_45_100_med = male_kpx_45_100[male_e45_sorted_indices[int(0.5 * n_simulations)]]
male_kpx_45_100_max = male_kpx_45_100[male_e45_sorted_indices[int(0.975 * n_simulations)]]

male_kqx_45_100_min = 1.0 - male_kpx_45_100_min
male_kqx_45_100_med = 1.0 - male_kpx_45_100_med
male_kqx_45_100_max = 1.0 - male_kpx_45_100_max
# %% [markdown]
# ### Perempuan
# %%
female_1qx = (2.0 * female_mortality_simulations) / (2.0 + female_mortality_simulations)
female_1px = 1.0 - female_1qx
# %%
from ta_module.actuarial import compute_ex, compute_kpx_table_from_xstart_dynamic

female_e45 = []
female_kpx_45_100 = []
for i in range(n_simulations):
    kpx_45_100 = compute_kpx_table_from_xstart_dynamic(
        p=female_1px[i],
        x_start=45,
        max_k=56,
        t0=0
    )
    female_kpx_45_100.append(kpx_45_100)
    kp45 = kpx_45_100[0]

    e45 = compute_ex(kp45)
    female_e45.append(e45)

female_e45 = torch.stack(female_e45)
female_kpx_45_100 = torch.stack(female_kpx_45_100)
# %%
_, female_e45_sorted_indices = torch.sort(female_e45, descending=False)

female_kpx_45_100_min = male_kpx_45_100[female_e45_sorted_indices[int(0.025 * n_simulations)]]
female_kpx_45_100_med = male_kpx_45_100[female_e45_sorted_indices[int(0.5 * n_simulations)]]
female_kpx_45_100_max = male_kpx_45_100[female_e45_sorted_indices[int(0.975 * n_simulations)]]

female_kqx_45_100_min = 1.0 - female_kpx_45_100_min
female_kqx_45_100_med = 1.0 - female_kpx_45_100_med
female_kqx_45_100_max = 1.0 - female_kpx_45_100_max
# %% [markdown]
# ### Plot e45
# %%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Union
import torch


def plot_e45_distribution(
    tensors: Union[list, dict],
    titles: Optional[list[str]] = None,
    percentiles: tuple[float, ...] = (2.5, 50, 97.5),
    figsize_per_plot: tuple[int, int] = (9, 4),
    ncols: int = 1,
    suptitle: Optional[str] = None,
    line_alpha: float = 0.8,
    fill_ci: bool = True,
    save_path: Optional[str] = None,
) -> None:
    """
    Plot KDE distribusi nilai tensor dengan garis persentil vertikal.

    Garis persentil diletakkan di nilai persentil ke-p dari data
    (sumbu x KDE adalah nilai data, bukan index).

    Parameters
    ----------
    tensors : list of array-like, atau dict {label: tensor}
        Satu atau beberapa tensor/array (shape 1-D atau akan di-flatten).
        Gunakan dict untuk memberi label otomatis per plot.
    titles : list of str, optional
        Judul per subplot. Diabaikan jika tensors berupa dict.
    percentiles : tuple of float
        Nilai persentil yang akan digambar sebagai garis vertikal.
        Default: (2.5, 50, 97.5).
    figsize_per_plot : tuple (w, h)
        Ukuran figure per subplot.
    ncols : int
        Jumlah kolom subplot. Baris dihitung otomatis.
    suptitle : str, optional
        Judul utama figure.
    line_alpha : float
        Opacity garis persentil (0-1).
    fill_ci : bool
        Jika True, isi area KDE antara persentil terkecil dan terbesar.
    save_path : str, optional
        Path untuk menyimpan figure (misal: 'plot.png').

    Returns
    -------
    fig : matplotlib.Figure
    """
    # --- normalise input ---
    if isinstance(tensors, dict):
        labels = list(tensors.keys())
        arrays = [
            v.detach().cpu().numpy() if isinstance(v, torch.Tensor)
            else np.asarray(v)
            for v in tensors.values()
        ]
    else:
        arrays = [
            v.detach().cpu().numpy() if isinstance(v, torch.Tensor)
            else np.asarray(v)
            for v in (tensors if isinstance(tensors, list) else [tensors])
        ]
        labels = titles if titles else [f"Tensor {i+1}" for i in range(len(arrays))]

    n_plots = len(arrays)
    ncols   = min(ncols, n_plots)
    nrows   = -(-n_plots // ncols)

    # --- palette & style ---
    sns.set_theme(style="white", font="DejaVu Sans")
    PCTL_COLORS = ["#378ADD", "#D85A30", "#1D9E75", "#7F77DD", "#BA7517"]
    KDE_COLOR   = "#5F5E5A"
    FILL_COLOR  = "#B5D4F4"

    W = figsize_per_plot[0] * ncols
    H = figsize_per_plot[1] * nrows
    fig, axes = plt.subplots(nrows, ncols, figsize=(W, H), squeeze=False)

    for idx, (arr, lbl) in enumerate(zip(arrays, labels)):
        ax   = axes[idx // ncols][idx % ncols]
        data = np.sort(arr.flatten())

        # KDE plot
        sns.kdeplot(
            data, ax=ax,
            color=KDE_COLOR, linewidth=1.5,
            fill=False,
        )

        # percentile values (pada sumbu x = nilai data)
        pctl_vals = np.percentile(data, percentiles)

        # optional fill antara persentil terkecil & terbesar
        if fill_ci and len(percentiles) >= 2:
            ax.axvspan(pctl_vals[0], pctl_vals[-1],
                       color=FILL_COLOR, alpha=0.2, zorder=1)

        # vertical percentile lines
        for i, (p, v) in enumerate(zip(percentiles, pctl_vals)):
            color = PCTL_COLORS[i % len(PCTL_COLORS)]
            ax.axvline(v, color=color, linewidth=1.6,
                       linestyle="--", alpha=line_alpha,
                       label=f"P{p:g} = {v:.3f}", zorder=3)

        # axes styling
        ax.set_title(lbl, fontsize=13, fontweight="bold",
                    color="#2C2C2A", pad=10)
        ax.set_xlabel("Value", fontsize=10, color="#888780")
        ax.set_ylabel("Density", fontsize=10, color="#888780")
        ax.tick_params(labelsize=9, color="#B4B2A9")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#D3D1C7")
        ax.spines["bottom"].set_color("#D3D1C7")
        ax.set_facecolor("white")

        ax.legend(
            fontsize=8.5, frameon=True,
            framealpha=0.85, edgecolor="#D3D1C7",
            loc="upper left",
            bbox_to_anchor=(0.0, -0.3),
            ncol=len(percentiles),
            borderaxespad=0,
        ).get_frame().set_linewidth(0.5)

    # hide empty axes
    for idx in range(n_plots, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    if suptitle:
        fig.suptitle(suptitle, fontsize=15, fontweight="bold",
                    color="#2C2C2A", y=1.01)

    fig.patch.set_facecolor("white")
    plt.tight_layout(pad=2.0, h_pad=4.5)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight",
                   facecolor="white")

    plt.show()
# %%
plot_e45_distribution(
    tensors=[male_e45, female_e45],
    titles = ["Laki-laki", "Perempuan"],
    figsize_per_plot=(9, 4),
    save_path=DOT_ENV.plots_dir / "e45.png"
)
# %% [markdown]
# ### Plot simulasi
# %%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from matplotlib.lines import Line2D

mortality_simulations = [
    female_mortality_simulations,
    male_mortality_simulations
]

mortality_data = [
    M_female,
    M_male
]

GENDER_LABEL = {0: "Perempuan", 1: "Laki-laki"}


def plot_mortality_fan(
    age: int,
    save_path: str,
    start_year: int = 2025,
    sim_alpha: float = 0.018,
    sim_color: str = "#4A90D9",
    band_colors: tuple = ("#E84545", "#F5A623", "#27AE60"),
    figsize: tuple = (13, 12),
    log_scale: bool = False,
    dpi: int = 300,
) -> None:

    # Styling
    sns.set_theme(style="whitegrid", font="serif")
    plt.rcParams.update(
        {
            "axes.facecolor"  : "#FAFAFA",
            "figure.facecolor": "#FFFFFF",
            "grid.color"      : "#E0E0E0",
            "grid.linewidth"  : 0.6,
            "axes.edgecolor"  : "#CCCCCC",
            "axes.labelcolor" : "#333333",
            "xtick.color"     : "#555555",
            "ytick.color"     : "#555555",
            "xtick.labelsize" : 10,
            "ytick.labelsize" : 10,
            "axes.labelsize"  : 12,
            "text.color"      : "#222222",
            "font.family"     : "serif",
        }
    )

    fig, axes = plt.subplots(2, 1, figsize=figsize)

    for gender, ax in enumerate(axes):
        # Data
        sims_data = mortality_simulations[gender].cpu().numpy()  # (n_sim, n_steps, n_ages)
        before_sims_data = mortality_data[gender].cpu().numpy()  # (n_years_hist, n_ages)

        n_sim, n_steps, n_ages = sims_data.shape
        assert 0 <= age < n_ages, f"age harus di rentang [0, {n_ages - 1}]"

        before_sim_years = np.arange(1950, 2025)
        fore_years = np.arange(start_year, start_year + n_steps)
        data = sims_data[:, :, age]   # (n_sim, n_steps)
        hist = before_sims_data[:, age]  # (75,)

        # Persentil
        pct_idx = female_e45_sorted_indices if gender == 0 else male_e45_sorted_indices
        pct_lo  = data[pct_idx[int(0.025 * n_sim)]]
        pct_med = data[pct_idx[int(0.5  * n_sim)]]
        pct_hi  = data[pct_idx[int(0.975 * n_sim)]]

        # Garis vertikal pemisah
        ax.axvline(2024.5, color="#AAAAAA", linewidth=1.2, linestyle=":", zorder=2)
        ax.text(
            2024.5 - 0.8, 0, "historis", ha="right", va="bottom",
            fontsize=8, color="#888888", style="italic",
            transform=ax.get_xaxis_transform(),
        )
        ax.text(
            2024.5 + 0.8, 0, "proyeksi", ha="left", va="bottom",
            fontsize=8, color="#888888", style="italic",
            transform=ax.get_xaxis_transform(),
        )

        # Data historis
        ax.plot(
            before_sim_years, hist,
            color="#333333", linewidth=1.8, zorder=4,
            label="Historis (1950 - 2024)",
        )

        # Fan simulasi
        for i in range(n_sim):
            ax.plot(
                fore_years, data[i],
                color=sim_color, alpha=sim_alpha, linewidth=0.4,
                rasterized=True,
            )

        ax.fill_between(
            fore_years, pct_lo, pct_hi,
            color=sim_color, alpha=0.10, linewidth=0,
        )

        # Garis persentil
        band_styles = [
            dict(linestyle="--", linewidth=1.8, alpha=0.90),
            dict(linestyle="-",  linewidth=2.6, alpha=1.00),
            dict(linestyle="--", linewidth=1.8, alpha=0.90),
        ]
        for arr, color, style, label in zip(
            [pct_lo, pct_med, pct_hi],
            band_colors, band_styles,
            ["min (P2.5)", "med (P50)", "max (P97.5)"],
        ):
            ax.plot(fore_years, arr, color=color, label=label, zorder=5, **style)
            ax.plot(fore_years[-1], arr[-1], "o", color=color, markersize=5, zorder=6)
            ax.annotate(
                f"  {arr[-1]:.4f}",
                xy=(fore_years[-1], arr[-1]),
                fontsize=8.5, color=color, va="center", fontweight="bold",
            )

        # Judul per subplot
        ax.set_title(
            f"{GENDER_LABEL[gender]}",
            fontsize=13, fontweight="bold", color="#111111", pad=10,
        )
        ax.set_xlabel("Tahun", labelpad=8)
        ax.set_ylabel("Mortality Rate", labelpad=8)

        # Skala Y
        all_vals = np.concatenate([hist, pct_lo, pct_med, pct_hi])
        y_min, y_max = all_vals.min(), all_vals.max()
        pad_y = (y_max - y_min) * 0.05
        ax.set_ylim(max(0, y_min - pad_y), y_max + pad_y)

        if log_scale:
            ax.set_yscale("log")
            ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
            ax.yaxis.set_minor_formatter(mticker.NullFormatter())
            ax.set_ylim(max(1e-6, y_min * 0.9), y_max * 1.1)

        ax.set_xlim(before_sim_years[0] - 0.5, fore_years[-1] + 3)

        # Highlight dekade
        for yr in range(1950, start_year + n_steps + 1, 10):
            ax.axvline(yr, color="#000000", linewidth=0.3, alpha=0.12, zorder=1)

        # Legenda
        custom_handles = [
            Line2D([0], [0], color="#333333", linewidth=1.8,
                   label="Historis (1950 - 2024)"),
            Line2D([0], [0], color=sim_color, alpha=0.5, linewidth=1.2,
                   label=f"Simulasi individual (n={n_sim:,})"),
            Line2D([0], [0], color=band_colors[0], linewidth=1.8,
                   linestyle="--", label="min (P2.5)"),
            Line2D([0], [0], color=band_colors[1], linewidth=2.6,
                   label="med (P50)"),
            Line2D([0], [0], color=band_colors[2], linewidth=1.8,
                   linestyle="--", label="max (P97.5)"),
        ]
        leg = ax.legend(
            handles=custom_handles,
            loc="upper left", fontsize=9.5,
            framealpha=0.7, facecolor="#FFFFFF", edgecolor="#CCCCCC",
            handlelength=2.2,
        )
        for text in leg.get_texts():
            text.set_color("#222222")

    # Judul figure keseluruhan
    fig.suptitle(
        f"Usia {age}",
        fontsize=15, fontweight="bold", color="#111111", y=1.01,
    )

    fig.text(
        0.99, 0.005,
        f"Proyeksi {start_year} - {start_year + n_steps - 1}  |  {n_sim:,} simulasi",
        ha="right", va="bottom", fontsize=7.5, color="#AAAAAA", style="italic",
    )

    plt.tight_layout(pad=2.0)
    fig.savefig(
        save_path, dpi=dpi, bbox_inches="tight",
        facecolor=fig.get_facecolor(),
    )
    print(f"Plot disimpan ke: {save_path}")
    plt.show()
# %%
plot_mortality_fan(
    age=0,
    save_path=DOT_ENV.plots_dir / "simulasi_mortalitas_usia_0.png"
)
# %%
plot_mortality_fan(
    age=50,
    save_path=DOT_ENV.plots_dir / "simulasi_mortalitas_usia_50.png"
)
# %%
plot_mortality_fan(
    age=100,
    save_path=DOT_ENV.plots_dir / "simulasi_mortalitas_usia_100.png"
)
# %% [markdown]
# ### Plot rekursif tanpa residual bootstrap
# %%
from ta_module.utils import recursive_forecast

x, y = test_male_dataset[-1]
x_in = torch.cat([x[1:, :], y])
x_in = x_in.unsqueeze(0)

male_recursive_forecast = recursive_forecast(
    model=localglmnet_ensemble,
    x=x_in,
    forecast_horizon=forecast_horizon,
    n_sim=1
)
male_recursive_forecast = inverse_transform_male_mortality(male_recursive_forecast)
# %%
x, y = test_female_dataset[-1]
x_in = torch.cat([x[1:, :], y])
x_in = x_in.unsqueeze(0)

female_recursive_forecast = recursive_forecast(
    model=localglmnet_ensemble,
    x=x_in,
    forecast_horizon=forecast_horizon,
    n_sim=1
)
female_recursive_forecast = inverse_transform_female_mortality(female_recursive_forecast)
# %%
sims_data = [
    female_recursive_forecast,
    male_recursive_forecast,
]


def plot_mortality_single(
    age: int,
    gender: int,
    save_path: str,
    sim_index: int = 0,
    start_year: int = 2025,
    figsize: tuple = (13, 6),
    log_scale: bool = False,
    dpi: int = 150,
) -> None:
    assert gender in (0, 1)

    sims_data = mortality_simulations[gender].cpu().numpy()
    before_sims_data = mortality_data[gender].cpu().numpy()

    n_sim, n_steps, n_ages = sims_data.shape
    assert 0 <= age < n_ages
    assert 0 <= sim_index < n_sim, f"sim_index harus di rentang [0, {n_sim - 1}]"

    before_sim_years = np.arange(1950, 2025)
    fore_years = np.arange(start_year, start_year + n_steps)
    hist = before_sims_data[:, age]
    sim = sims_data[sim_index, :, age]

    # Styling
    sns.set_theme(style="whitegrid", font="serif")
    plt.rcParams.update(
        {
            "axes.facecolor"  : "#FAFAFA",
            "figure.facecolor": "#FFFFFF",
            "grid.color"      : "#E0E0E0",
            "grid.linewidth"  : 0.6,
            "axes.edgecolor"  : "#CCCCCC",
            "axes.labelcolor" : "#333333",
            "xtick.color"     : "#555555",
            "ytick.color"     : "#555555",
            "xtick.labelsize" : 10,
            "ytick.labelsize" : 10,
            "axes.labelsize"  : 12,
            "text.color"      : "#222222",
            "font.family"     : "serif",
        }
    )

    fig, ax = plt.subplots(figsize=figsize)

    # Pemisah historis / proyeksi
    ax.axvline(2024.5, color="#AAAAAA", linewidth=1.2, linestyle=":", zorder=2)
    ax.text(
        2024.5 - 0.8, 0, "historis", ha="right", va="bottom",
        fontsize=8, color="#888888", style="italic",
        transform=ax.get_xaxis_transform()
        )
    ax.text(
        2024.5 + 0.8, 0, "proyeksi", ha="left", va="bottom",
        fontsize=8, color="#888888", style="italic",
        transform=ax.get_xaxis_transform()
        )

    # Garis historis
    ax.plot(
        before_sim_years, hist,
        color="#333333", linewidth=1.8, zorder=4,
        label="Historis (1950 - 2024)"
    )

    # Satu garis simulasi
    ax.plot(
        fore_years, sim,
        color="#4A90D9", linewidth=1.8, zorder=4,
        label=f"Peramalan Rekursif"
    )

    # Titik sambung historis - simulasi
    ax.plot(2024, hist[-1], "o", color="#333333", markersize=5, zorder=5)
    ax.plot(start_year, sim[0], "o", color="#4A90D9", markersize=5, zorder=5)

    # Estetika
    ax.set_title(
        f"Peramalan Mortalitas  -  Usia {age}  -  {GENDER_LABEL[gender]}",
        fontsize=14, fontweight="bold", color="#111111", pad=14, loc="center",
    )
    ax.set_xlabel("Tahun", labelpad=10)
    ax.set_ylabel("Mortality Rate", labelpad=10)

    all_vals = np.concatenate([hist, sim])
    y_min, y_max = all_vals.min(), all_vals.max()
    pad_y = (y_max - y_min) * 0.05
    ax.set_ylim(max(0, y_min - pad_y), y_max + pad_y)

    if log_scale:
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
        ax.yaxis.set_minor_formatter(mticker.NullFormatter())
        ax.set_ylim(max(1e-6, y_min * 0.9), y_max * 1.1)
        ax.yaxis.set_major_locator(mticker.LogLocator(numticks=8))
    else:
        ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=8, prune="both"))

    ax.set_xlim(before_sim_years[0] - 0.5, fore_years[-1] + 1)

    for yr in range(1950, start_year + n_steps + 1, 10):
        ax.axvline(yr, color="#000000", linewidth=0.3, alpha=0.12, zorder=1)

    leg = ax.legend(
        fontsize=9.5, framealpha=0.7, facecolor="#FFFFFF",
        edgecolor="#CCCCCC", handlelength=2.2
    )
    for text in leg.get_texts():
        text.set_color("#222222")

    fig.text(
        0.99, 0.01,
        f"Proyeksi {start_year} - {start_year + n_steps - 1}",
        ha="right", va="bottom", fontsize=7.5, color="#AAAAAA", style="italic"
    )

    plt.tight_layout(pad=1.5)
    fig.savefig(
        save_path, dpi=dpi, bbox_inches="tight",
        facecolor=fig.get_facecolor()
    )
    print(f"Plot disimpan ke: {save_path}")
    plt.show()
# %%
plot_mortality_single(
    age=0,
    gender=0,
    save_path=DOT_ENV.plots_dir / "peramalan_rekursif_perempuan_usia_0.png"
)
# %%
plot_mortality_single(
    age=50,
    gender=0,
    save_path=DOT_ENV.plots_dir / "peramalan_rekursif_perempuan_usia_50.png"
)
# %%
plot_mortality_single(
    age=100,
    gender=0,
    save_path=DOT_ENV.plots_dir / "peramalan_rekursif_perempuan_usia_100.png"
)
# %%
plot_mortality_single(
    age=0,
    gender=1,
    save_path=DOT_ENV.plots_dir / "peramalan_rekursif_laki-laki_usia_0.png"
)
# %%
plot_mortality_single(
    age=50,
    gender=1,
    save_path=DOT_ENV.plots_dir / "peramalan_rekursif_laki-laki_usia_50.png"
)
# %%
plot_mortality_single(
    age=100,
    gender=1,
    save_path=DOT_ENV.plots_dir / "peramalan_rekursif_laki-laki_usia_100.png"
)
# %% [markdown]
# ## Deterministik (TMI IV)
# %%
tmi4_df = pd.read_csv("./data/tabel_mortalitas_indonesia_iv.csv", sep=";", decimal=",")
tmi4_df.tail()
# %% [markdown]
# ### Laki-laki
# %%
male_1qx_tmi4 = torch.from_numpy(tmi4_df.loc[:, "laki-laki"].to_numpy(copy=True, dtype=np.float32).reshape(1, -1)).to(
    DEVICE
)
male_1px_tmi4 = 1.0 - male_1qx_tmi4
# %%
male_1px_tmi4
# %%
from ta_module.actuarial import compute_kpx_table_from_xstart_static

male_kpx_45_100_tmi4 = compute_kpx_table_from_xstart_static(
    p=male_1px_tmi4,
    x_start=45,
    max_k=56,
)
# %% [markdown]
# ### Perempuan
# %%
female_1qx_tmi4 = torch.from_numpy(tmi4_df.loc[:, "perempuan"].to_numpy(copy=True, dtype=np.float32).reshape(1, -1)).to(
    DEVICE
)
female_1px_tmi4 = 1.0 - female_1qx_tmi4
# %%
female_kpx_45_100_tmi4 = compute_kpx_table_from_xstart_static(
    p=female_1px_tmi4,
    x_start=45,
    max_k=56,
)
# %%
male_kpx_45_100_tmi4.shape
# %%
pd.DataFrame(male_kpx_45_100_tmi4.cpu().numpy(),
             index=range(45, 101),
             columns=range(0, 57)
).to_csv(DOT_ENV.results_dir / "male_tmi4_life_table.csv", sep=";", decimal=",")
pd.DataFrame(female_kpx_45_100_tmi4.cpu().numpy(),
             index=range(45, 101),
             columns=range(0, 57)
).to_csv(DOT_ENV.results_dir / "female_tmi4_life_table.csv", sep=";", decimal=",")
# %% [markdown]
# # Perhitungan longevity risk
# %% [markdown]
# ## Persiapan
# %%
populasi_df.head()
# %%
populasi_joint_dist_df = (
    populasi_df
    .groupby(['gender', 'age'])["value"]
    .sum()
    .div(populasi_df["value"].sum())
    .rename('prob')
    .reset_index()
)

populasi_joint_dist_df['cdf'] = (
    populasi_joint_dist_df
    .groupby('gender')['prob']
    .cumsum()
)
# %%
populasi_joint_dist_df.head()
# %%
male_age_populasi_prob = populasi_joint_dist_df[populasi_joint_dist_df["gender"] == "Male"]["prob"]
male_age_populasi_prob = torch.from_numpy(male_age_populasi_prob.to_numpy(copy=True, dtype=np.float32)).to(DEVICE)

female_age_populasi_prob = populasi_joint_dist_df[populasi_joint_dist_df["gender"] == "Female"]["prob"]
female_age_populasi_prob = torch.from_numpy(female_age_populasi_prob.to_numpy(copy=True, dtype=np.float32)).to(DEVICE)
# %%
male_age_populasi_cdf = populasi_joint_dist_df[populasi_joint_dist_df["gender"] == "Male"]["cdf"]
male_age_populasi_cdf = torch.from_numpy(male_age_populasi_cdf.to_numpy(copy=True, dtype=np.float32)).to(DEVICE)

female_age_populasi_cdf = populasi_joint_dist_df[populasi_joint_dist_df["gender"] == "Female"]["cdf"]
female_age_populasi_cdf = torch.from_numpy(female_age_populasi_cdf.to_numpy(copy=True, dtype=np.float32)).to(DEVICE)
# %%
gender_populasi_dist_df = (
    populasi_df.groupby('gender')['value']
    .sum()
    .div(populasi_df['value'].sum())
    .rename('prob')
    .reset_index()
)

gender_populasi_dist_df['cdf'] = gender_populasi_dist_df['prob'].cumsum()
# %%
male_populasi_prob = torch.tensor(
    gender_populasi_dist_df[gender_populasi_dist_df["gender"] == "Male"]["prob"].iloc[0], device=DEVICE,
    dtype=torch.float32
)

female_populasi_prob = torch.tensor(
    gender_populasi_dist_df[gender_populasi_dist_df["gender"] == "Female"]["prob"].iloc[0], device=DEVICE,
    dtype=torch.float32
)

gender_populasi_cdf = torch.from_numpy(gender_populasi_dist_df["cdf"].to_numpy(copy=True, dtype=np.float32)).to(DEVICE)
# %%
age_populasi_given_male_cdf = male_age_populasi_cdf / male_populasi_prob
age_populasi_given_female_cdf = female_age_populasi_cdf / female_populasi_prob
# %%
bi_rate_df.head()
# %%
suku_bunga_efektif_tahunan = torch.tensor(bi_rate_df["bi_rate"].mean(), device=DEVICE, dtype=torch.float32)
suku_bunga_efektif_tahunan
# %%
suku_bunga_nominal_bulanan = 12.0 * ((1.0 + suku_bunga_efektif_tahunan).pow(1.0 / 12.0) - 1.0)
suku_bunga_nominal_bulanan
# %%
from torch.distributions import LogNormal

premi_lognormal_rv = LogNormal(loc=5.9254, scale=0.6064)
# %%
start_age = 45
end_age = 60
m = 12

from ta_module.actuarial import compute_m_annuity_epv, compute_m_annuity_var, compute_m_annuity_epv2
from torch import Tensor


def wrapper_compute_m_annuity_epv(
    kpx: Tensor,
    gender_age_prob: Tensor
) -> Tensor:
    return compute_m_annuity_epv(
        start_age=start_age,
        end_age=end_age,
        m=m,
        i=suku_bunga_efektif_tahunan,
        kpx=kpx,
        gender_age_prob=gender_age_prob,
    )


def wrapper_compute_m_annuity_var(
    kpx: Tensor,
    gender_age_prob: Tensor
) -> Tensor:
    return compute_m_annuity_var(
        start_age=start_age,
        end_age=end_age,
        m=m,
        i=suku_bunga_efektif_tahunan,
        kpx=kpx,
        gender_age_prob=gender_age_prob,
    )


def wrapper_compute_m_annuity_epv2(
    kpx: Tensor,
    gender_age_prob: Tensor
) -> Tensor:
    return compute_m_annuity_epv2(
        start_age=start_age,
        end_age=end_age,
        m=m,
        i=suku_bunga_efektif_tahunan,
        kpx=kpx,
        gender_age_prob=gender_age_prob,
    )
# %%
N = [1, 10, 100, 1_000, 10_000, 100_000, 1_000_000]
# %% [markdown]
# ## Kerangka deterministik
# %% [markdown]
# ### EPV
# %%
male_tmi4_epv = wrapper_compute_m_annuity_epv(
    kpx=male_kpx_45_100_tmi4,
    gender_age_prob=male_age_populasi_prob
)

male_tmi4_epv
# %%
female_tmi4_epv = wrapper_compute_m_annuity_epv(
    kpx=female_kpx_45_100_tmi4,
    gender_age_prob=female_age_populasi_prob
)

female_tmi4_epv
# %%
tmi4_epv = male_tmi4_epv + female_tmi4_epv
tmi4_epv
# %%
expected_benefit_tmi4 = (premi_lognormal_rv.mean + 50) / tmi4_epv
print(f"Ekspektasi manfaat per bulan anuitas jiwa deterministik (dalam juta rupiah) = {expected_benefit_tmi4:.6f}")
# %%
portfolio_tmi4_epv = {}
for n in N:
    portfolio_tmi4_epv[n] = n * expected_benefit_tmi4 * tmi4_epv
    print(f"N = {n}\nEPV portfolio deterministik = {portfolio_tmi4_epv[n]:.6f}\n")
# %% [markdown]
# ### Varians
# %%
male_tmi4_var = wrapper_compute_m_annuity_var(
    kpx=male_kpx_45_100_tmi4,
    gender_age_prob=male_age_populasi_prob
)

male_tmi4_var
# %%
male_tmi4_epv2 = wrapper_compute_m_annuity_epv2(
    kpx=male_kpx_45_100_tmi4,
    gender_age_prob=male_age_populasi_prob
)

male_tmi4_epv2
# %%
female_tmi4_var = wrapper_compute_m_annuity_var(
    kpx=female_kpx_45_100_tmi4,
    gender_age_prob=female_age_populasi_prob
)

female_tmi4_var
# %%

female_tmi4_epv2 = wrapper_compute_m_annuity_epv2(
    kpx=female_kpx_45_100_tmi4,
    gender_age_prob=female_age_populasi_prob
)

female_tmi4_epv2
# %%
tmi4_epv2 = male_tmi4_epv2 + female_tmi4_epv2
tmi4_var = male_tmi4_var + female_tmi4_var
tmi4_total_var = tmi4_var + tmi4_epv2 - tmi4_epv.pow(2)
# %%
portfolio_tmi4_var = {}

E_B = expected_benefit_tmi4
# VAR(B) = VAR(P / E(Y)) = VAR([P* + 50] / E(Y)) = 1 / E(Y)^2 * VAR(P*)
VAR_B = 1 / tmi4_epv.pow(2) * premi_lognormal_rv.variance
# VAR(B) = E(B^2) - E(B)^2 => E(B^2) = E(B)^2 + VAR(B)
E_B2 = E_B.pow(2) + VAR_B

E_Y = tmi4_epv
VAR_Y = tmi4_total_var
E_Y2 = E_Y.pow(2) + VAR_Y

for n in N:
    portfolio_tmi4_var[n] = n * (E_B2 * E_Y2 - (E_B * E_Y).pow(2))
    print(f"N = {n}\nVAR portfolio deterministik = {portfolio_tmi4_var[n]:.6f}\n")
# %%
E_B = expected_benefit_tmi4
# VAR(B) = VAR(P / E(Y)) = VAR([P* + 50] / E(Y)) = 1 / E(Y)^2 * VAR(P*)
VAR_B = 1 / tmi4_epv.pow(2) * premi_lognormal_rv.variance
# VAR(B) = E(B^2) - E(B)^2 => E(B^2) = E(B)^2 + VAR(B)
E_B2 = E_B.pow(2) + VAR_B

E_Y = tmi4_epv
VAR_Y = tmi4_total_var
E_Y2 = E_Y.pow(2) + VAR_Y

print(premi_lognormal_rv.variance)
print(E_B)
print(VAR_B)
print(E_B2)
print(E_Y)
print(VAR_Y)
print(E_Y2)
# %% [markdown]
# ### Koefisien variasi
# %%
portfolio_tmi4_koefisien_variasi = {}
for n in N:
    portfolio_tmi4_koefisien_variasi[n] = portfolio_tmi4_var[n].sqrt() / portfolio_tmi4_epv[n]
    print(f"N = {n}\nKoefisien variasi portfolio deterministik = {portfolio_tmi4_koefisien_variasi[n]:.6f}\n")
# %%
portfolio_tmi4_epv.values()
# %%
tmi4_longevity_risk = {
    "epv"              : [v.item() for v in portfolio_tmi4_epv.values()],
    "var"              : [v.item() for v in portfolio_tmi4_var.values()],
    "koefisien_variasi": [v.item() for v in portfolio_tmi4_koefisien_variasi.values()]
}

pd.DataFrame(tmi4_longevity_risk, index=N).to_csv(DOT_ENV.results_dir / "tmi4_longevity_risk.csv", sep=";", decimal=",")
# %% [markdown]
# ## Kerangka stokastik
# %% [markdown]
# ### EPV
# %%
male_min_epv = wrapper_compute_m_annuity_epv(
    kpx=male_kpx_45_100_min,
    gender_age_prob=male_age_populasi_prob
)

male_min_epv
# %%
male_med_epv = wrapper_compute_m_annuity_epv(
    kpx=male_kpx_45_100_med,
    gender_age_prob=male_age_populasi_prob
)

male_med_epv
# %%
male_max_epv = wrapper_compute_m_annuity_epv(
    kpx=male_kpx_45_100_max,
    gender_age_prob=male_age_populasi_prob
)

male_max_epv
# %%
female_min_epv = wrapper_compute_m_annuity_epv(
    kpx=female_kpx_45_100_min,
    gender_age_prob=female_age_populasi_prob
)

female_min_epv
# %%
female_med_epv = wrapper_compute_m_annuity_epv(
    kpx=female_kpx_45_100_med,
    gender_age_prob=female_age_populasi_prob
)

female_med_epv
# %%
female_max_epv = wrapper_compute_m_annuity_epv(
    kpx=female_kpx_45_100_max,
    gender_age_prob=female_age_populasi_prob
)

female_max_epv
# %%
min_epv = male_min_epv + female_min_epv
med_epv = male_med_epv + female_med_epv
max_epv = male_max_epv + female_max_epv
# %%
r_min = 0.15
r_med = 0.70
r_max = 0.15

stokastik_epv = r_min * min_epv + r_med * med_epv + r_max * max_epv
print(f"EPV anuitas jiwa stokastik = {stokastik_epv:.6f}")
# %%
expected_benefit_min = (premi_lognormal_rv.mean + 50) / min_epv
expected_beneit_med = (premi_lognormal_rv.mean + 50) / med_epv
expected_beneit_max = (premi_lognormal_rv.mean + 50) / max_epv

expected_benefit_stokastik = (
        r_min * expected_benefit_min +
        r_med * expected_beneit_med +
        r_max * expected_beneit_max
)
print(f"Ekspektasi manfaat bulanan anuitas jiwa stokastik (dalam juta rupiah) = {expected_benefit_stokastik:.6f}")
# %%
portfolio_stokastik_epv = {}
for n in N:
    portfolio_stokastik_epv[n] = n * expected_benefit_stokastik * stokastik_epv
    print(f"\nN = {n}\nEPV portfolio stokastik = {portfolio_stokastik_epv[n]:.6f}")
# %% [markdown]
# ### Varians
# %%
male_min_var = wrapper_compute_m_annuity_var(
    kpx=male_kpx_45_100_min,
    gender_age_prob=male_age_populasi_prob
)

male_med_var = wrapper_compute_m_annuity_var(
    kpx=male_kpx_45_100_med,
    gender_age_prob=male_age_populasi_prob
)

male_max_var = wrapper_compute_m_annuity_var(
    kpx=male_kpx_45_100_max,
    gender_age_prob=male_age_populasi_prob
)
# %%
female_min_var = wrapper_compute_m_annuity_var(
    kpx=female_kpx_45_100_min,
    gender_age_prob=female_age_populasi_prob
)

female_med_var = wrapper_compute_m_annuity_var(
    kpx=female_kpx_45_100_med,
    gender_age_prob=female_age_populasi_prob
)

female_max_var = wrapper_compute_m_annuity_var(
    kpx=female_kpx_45_100_max,
    gender_age_prob=female_age_populasi_prob
)
# %%
min_var = male_min_var + female_min_var
med_var = male_med_var + female_med_var
max_var = male_max_var + female_max_var
# %%
male_min_epv2 = wrapper_compute_m_annuity_epv2(
    kpx=male_kpx_45_100_min,
    gender_age_prob=male_age_populasi_prob
)

male_med_epv2 = wrapper_compute_m_annuity_epv2(
    kpx=male_kpx_45_100_med,
    gender_age_prob=male_age_populasi_prob
)

male_max_epv2 = wrapper_compute_m_annuity_epv2(
    kpx=male_kpx_45_100_max,
    gender_age_prob=male_age_populasi_prob
)
# %%
female_min_epv2 = wrapper_compute_m_annuity_epv2(
    kpx=male_kpx_45_100_min,
    gender_age_prob=female_age_populasi_prob
)

female_med_epv2 = wrapper_compute_m_annuity_epv2(
    kpx=female_kpx_45_100_med,
    gender_age_prob=female_age_populasi_prob
)

female_max_epv2 = wrapper_compute_m_annuity_epv2(
    kpx=female_kpx_45_100_max,
    gender_age_prob=female_age_populasi_prob
)
# %%
min_epv2 = male_min_epv2 + female_min_epv2
med_epv2 = male_med_epv2 + female_med_epv2
max_epv2 = male_max_epv2 + female_max_epv2
# %%
min_total_var = min_var + min_epv2 - min_epv.pow(2)
med_total_var = med_var + med_epv2 - med_epv.pow(2)
max_total_var = max_var + max_epv2 - max_epv.pow(2)
# %%
portfolio_stokastik_var = {}
portfolio_stokastik_E_VAR = {}
portfolio_stokastik_VAR_E = {}

N_bar = 0

E_B = expected_benefit_stokastik
VAR_B = 1 / stokastik_epv.pow(2) * premi_lognormal_rv.variance
E_B2 = VAR_B + E_B.pow(2)

E_Y_min = min_epv
E_Y_med = med_epv
E_Y_max = max_epv

E_Y2_min = min_total_var + min_epv.pow(2)
E_Y2_med = med_total_var + med_epv.pow(2)
E_Y2_max = max_total_var + max_epv.pow(2)

E_EB_EY_2 = (
        r_min * (E_B * E_Y_min).pow(2) +
        r_med * (E_B * E_Y_med).pow(2) +
        r_max * (E_B * E_Y_max).pow(2)
)

E_EB_EY = (
        r_min * (E_B * E_Y_min) +
        r_med * (E_B * E_Y_med) +
        r_max * (E_B * E_Y_max)
)

for n in N:
    E_VAR = (
            r_min * (E_B2 * E_Y2_min - (E_B * E_Y_min).pow(2)) +
            r_med * (E_B2 * E_Y2_med - (E_B * E_Y_med).pow(2)) +
            r_max * (E_B2 * E_Y2_max - (E_B * E_Y_max).pow(2))
    )
    VAR_E = (E_EB_EY_2 - E_EB_EY.pow(2))
    portfolio_stokastik_var[n] = n * E_VAR + (n ** 2) * VAR_E
    portfolio_stokastik_E_VAR[n] = n * E_VAR
    portfolio_stokastik_VAR_E[n] = (n ** 2) * VAR_E

    N_bar = E_VAR / VAR_E

    print(f"N = {n}")
    print(f"N_bar = {N_bar:.4f}")
    print(f"VAR portfolio stokastik = {portfolio_stokastik_var[n]:.6f}\n")
# %%
E_B = expected_benefit_stokastik
VAR_B = 1 / stokastik_epv.pow(2) * premi_lognormal_rv.variance
E_B2 = VAR_B + E_B.pow(2)

E_Y_min = min_epv
E_Y_med = med_epv
E_Y_max = max_epv

E_Y2_min = min_total_var + min_epv.pow(2)
E_Y2_med = med_total_var + med_epv.pow(2)
E_Y2_max = max_total_var + max_epv.pow(2)

E_EB_EY_2 = (
        r_min * (E_B * E_Y_min).pow(2) +
        r_med * (E_B * E_Y_med).pow(2) +
        r_max * (E_B * E_Y_max).pow(2)
)

E_EB_EY = (
        r_min * (E_B * E_Y_min) +
        r_med * (E_B * E_Y_med) +
        r_max * (E_B * E_Y_max)
)

E_VAR = (
            r_min * (E_B2 * E_Y2_min - (E_B * E_Y_min).pow(2)) +
            r_med * (E_B2 * E_Y2_med - (E_B * E_Y_med).pow(2)) +
            r_max * (E_B2 * E_Y2_max - (E_B * E_Y_max).pow(2))
    )
VAR_E = (E_EB_EY_2 - E_EB_EY.pow(2))
print(E_VAR)
print(VAR_E)
# %% [markdown]
# ### Koefisien variasi
# %%
portfolio_stokastik_koefisien_variasi = {}
for n in N:
    portfolio_stokastik_koefisien_variasi[n] = portfolio_stokastik_var[n].sqrt() / portfolio_stokastik_epv[n]
    print(f"N = {n}")
    print(f"N_bar = {N_bar:.4f}")
    print(f"Koefisien variasi portfolio stokastik = {portfolio_stokastik_koefisien_variasi[n]:.6f}\n")
# %%
stokastik_longevity_risk = {
    "epv"              : [v.item() for v in portfolio_stokastik_epv.values()],
    "E_VAR"            : [v.item() for v in portfolio_stokastik_E_VAR.values()],
    "VAR_E"            : [v.item() for v in portfolio_stokastik_VAR_E.values()],
    "var"              : [v.item() for v in portfolio_stokastik_var.values()],
    "koefisien_variasi": [v.item() for v in portfolio_stokastik_koefisien_variasi.values()],
    "N_bar"            : N_bar.item()
}

pd.DataFrame(stokastik_longevity_risk, index=N).to_csv(
    DOT_ENV.results_dir / "stokastik_longevity_risk.csv", sep=";", decimal=","
)
# %% [markdown]
# # Perhitungan value at risk
# %%
# Reset seed state
seed_everything(seed=CONFIG.seed, workers=True)
# %%
from torch import Tensor

from ta_module.actuarial import create_fractional_m_cdf
from ta_module.utils import inverse_transform_sampling


def sampling_gender(
    n_sims: int = 10_000,
    n_samples: int = 1,
    device: str = "cpu"
) -> Tensor:
    x = torch.arange(0, 2, dtype=torch.uint8).expand(n_sims, n_samples, -1).contiguous().to(device)
    cdf = gender_populasi_cdf.expand(n_sims, n_samples, -1).contiguous()
    samples = inverse_transform_sampling(
        x=x,
        cdf=cdf,
    )

    return samples


def sampling_age(
    start_age: int,
    end_age: int,
    gender: Tensor,
    n_sims: int = 10_000,
    n_samples: int = 1,
    device: str = "cpu"
) -> Tensor:
    x = torch.arange(start_age, end_age + 1, dtype=torch.uint8).expand(n_sims, n_samples, -1).contiguous().to(device)
    cdf = torch.where(gender == 0, age_populasi_given_female_cdf, age_populasi_given_male_cdf).to(device)
    samples = inverse_transform_sampling(
        x=x,
        cdf=cdf,
    )

    return samples


def sampling_curtate_m_future_lifetime(
    fractional_m_cdf: Tensor,
    n_sims: int = 10_000,
    n_samples: int = 1,
    device: str = "cpu"
) -> Tensor:
    x = torch.arange(0, fractional_m_cdf.shape[-1], dtype=torch.int32).expand(n_sims, n_samples, -1).contiguous().to(device)
    cdf = fractional_m_cdf
    samples = inverse_transform_sampling(
        x=x,
        cdf=cdf,
    )

    return samples
# %% [markdown]
# ## Kerangka deterministik
# %%
m = 12
def sampling_portfolio_value_deterministic(
    n_sims: int = 10_000,
    n_samples: int = 1,
    device: str = "cpu"
) -> Tensor:
    gender_samples = sampling_gender(
        n_sims=n_sims,
        n_samples=n_samples,
        device=device
    )

    age_samples = sampling_age(
        start_age=45,
        end_age=60,
        gender=gender_samples,
        n_sims=n_sims,
        n_samples=n_samples,
        device=device
    )

    kpx = torch.stack([female_kpx_45_100_tmi4, male_kpx_45_100_tmi4])
    gender_idx = gender_samples.squeeze().long()
    age_idx = (age_samples - start_age).squeeze().long()
    fractional_m_cdf = create_fractional_m_cdf(kpx=kpx[gender_idx, age_idx], m=m).view(n_sims, n_samples, -1)

    premi_samples = 50 + premi_lognormal_rv.sample(sample_shape=(n_sims, n_samples, 1)).to(device)
    benefit_samples = premi_samples / tmi4_epv

    curtate_m_future_lifetime_samples = sampling_curtate_m_future_lifetime(
        fractional_m_cdf=fractional_m_cdf,
        n_sims=n_sims,
        n_samples=n_samples,
        device=device
    )

    i_m = suku_bunga_nominal_bulanan
    v_m = 1 / (1 + i_m / m)
    d_m = (i_m / m) * v_m

    annuity_value = ((1 - v_m.pow(curtate_m_future_lifetime_samples + 1)) / d_m - 1).clamp(min=0)
    portfolio_value = (benefit_samples * annuity_value).sum(dim=1)

    return portfolio_value.to(torch.float32)
# %%
n_simulations = 10_000
chunk_threshold = 100

portfolio_simulations_deterministik_file_path = DOT_ENV.results_dir / f"portfolio_simulations_deterministik.pt"
if not portfolio_simulations_deterministik_file_path.exists():
    print(55 * "=")
    print("Simulasi portolio value untuk kerangka deterministik:")
    print(55 * "=")
    portfolio_simulations_deterministik = torch.zeros(size=(len(N), n_simulations, 1), device=DEVICE)
    for i in range(len(N)):
        n = N[i]
        simulation_file_path = DOT_ENV.results_dir / f"portfolio_simulations_deterministik_n_{n}.pt"
        if not simulation_file_path.exists():
            print(f"Mulai simulasi portfolio N = {n}\n")
            chunk_size = n
            if n > chunk_threshold:
                chunk_size = chunk_threshold

            quotient, remainder = divmod(n, chunk_size)
            portfolio_simulation_value = torch.zeros(size=(n_simulations, 1), device=DEVICE)
            for j in range(quotient):
                print(f"Mulai simulasi untuk chunk ke {j + 1}/{quotient}...")
                portfolio_simulation_value += sampling_portfolio_value_deterministic(
                    n_sims=n_simulations,
                    n_samples=chunk_size,
                    device=DEVICE
                )

            if remainder > 0:
                print(f"Mulai simulasi untuk {remainder} sample sisanya...")
                portfolio_simulation_value += sampling_portfolio_value_deterministic(
                    n_sims=n_simulations,
                    n_samples=remainder,
                    device=DEVICE
                )

            print(f"\nMenyimpan ke {simulation_file_path}...")
            torch.save(obj=portfolio_simulation_value, f=simulation_file_path)
            portfolio_simulations_deterministik[i] = portfolio_simulation_value
        else:
            print(f"\nLoad {simulation_file_path}...")
            portfolio_simulations_deterministik[i] = torch.load(f=simulation_file_path, map_location=DEVICE)

        print(f"\nSimulasi portfolio N = {n} selesai")
        print(f"Hasil: {portfolio_simulations_deterministik[i][:5]}")
        print(55 * "=")

    print("\nSimulasi selesai!")
    print(f"Simpan hasil simulasi ke {portfolio_simulations_deterministik_file_path}")
    torch.save(obj=portfolio_simulations_deterministik, f=portfolio_simulations_deterministik_file_path)
    print("Selesai!")
else:
    print(f"Load {portfolio_simulations_deterministik_file_path}...")
    portfolio_simulations_deterministik = torch.load(
        f=portfolio_simulations_deterministik_file_path, map_location=DEVICE
    )

    print("Load selesai!")
# %%
portfolio_simulations_deterministik[(portfolio_simulations_deterministik < 0)]
# %%
alphas = [0.5, 0.9, 0.95, 0.99]
var_deterministik = {}

for a in alphas:
    var_deterministik[a] = []
    for i in range(len(N)):
        n = N[i]
        portfolio_n = portfolio_simulations_deterministik[i].squeeze(-1)
        _, indices = torch.sort(portfolio_n, descending=False)
        var_deterministik[a].append(portfolio_n[indices[int(a * n_simulations)]].item())
print(var_deterministik)
# %%
pd.DataFrame(var_deterministik, index=N).to_csv(DOT_ENV.results_dir / "var_deterministik.csv", decimal=",", sep=";")
# %%
mean = []
var = []
cv = []
for i in range(7):
    m = portfolio_simulations_deterministik[i].mean().item()
    v = portfolio_simulations_deterministik[i].var().item()
    s = portfolio_simulations_deterministik[i].std().item()
    mean.append(m)
    var.append(v)
    cv.append(s / m)

pd.DataFrame({
    "mean": mean,
    "var": var,
    "cv": cv
}, index=N).to_csv(DOT_ENV.results_dir / "portfolio_deterministik_statdesc.csv", sep=";", decimal=",")
# %% [markdown]
# ## Kerangka stokastik
# %%
m = 12
def sampling_mortality_trend(
    n_sims: int = 10_000,
    device: str = "cpu"
) -> Tensor:
    x = torch.arange(0, 3, dtype=torch.uint8).expand(n_sims, -1).contiguous().to(device)
    cdf = torch.tensor([0.15, 0.85, 1.00], device=device).expand(n_sims, -1).contiguous()
    samples = inverse_transform_sampling(
        x=x,
        cdf=cdf
    )

    return samples
# %%
def sampling_portfolio_value_stochastic(
    mortality_trend_idx: Tensor,
    n_sims: int = 10_000,
    n_samples: int = 1,
    device: str = "cpu"
) -> Tensor:
    gender_samples = sampling_gender(
        n_sims=n_sims,
        n_samples=n_samples,
        device=device
    )

    age_samples = sampling_age(
        start_age=45,
        end_age=60,
        gender=gender_samples,
        n_sims=n_sims,
        n_samples=n_samples,
        device=device
    )


    min_kpx = torch.stack([female_kpx_45_100_min, male_kpx_45_100_min]).to(DEVICE)
    med_kpx = torch.stack([female_kpx_45_100_med, male_kpx_45_100_med]).to(DEVICE)
    max_kpx = torch.stack([female_kpx_45_100_max, male_kpx_45_100_max]).to(DEVICE)
    kpx = torch.stack([min_kpx, med_kpx, max_kpx]).to(DEVICE)

    gender_idx = gender_samples.squeeze().long()
    age_idx = (age_samples - start_age).squeeze().long()
    mortality_trend_idx = mortality_trend_idx.repeat(1, n_samples).squeeze().long()
    fractional_m_cdf = create_fractional_m_cdf(kpx=kpx[mortality_trend_idx, gender_idx, age_idx], m=m).view(
        n_sims, n_samples, -1
    )

    premi_samples = 50 + premi_lognormal_rv.sample(sample_shape=(n_sims, n_samples, 1)).to(device)
    benefit_samples = premi_samples / stokastik_epv

    curtate_m_future_lifetime_samples = sampling_curtate_m_future_lifetime(
        fractional_m_cdf=fractional_m_cdf,
        n_sims=n_sims,
        n_samples=n_samples,
        device=device
    )

    i_m = suku_bunga_nominal_bulanan
    v_m = 1 / (1 + i_m / m)
    d_m = (i_m / m) * v_m

    annuity_value = ((1 - v_m.pow(curtate_m_future_lifetime_samples + 1)) / d_m - 1).clamp(min=0)
    portfolio_value = (benefit_samples * annuity_value).sum(dim=1)

    return portfolio_value.to(torch.float32)
# %%
n_simulations = 10_000
chunk_threshold = 100

portfolio_simulations_stokastik_file_path = DOT_ENV.results_dir / f"portfolio_simulations_stokastik.pt"
if not portfolio_simulations_stokastik_file_path.exists():
    print(55 * "=")
    print("Simulasi portolio value untuk kerangka stokastik:")
    print(55 * "=")
    portfolio_simulations_stokastik = torch.zeros(size=(len(N), n_simulations, 1), device=DEVICE, dtype=torch.float32)
    for i in range(len(N)):
        mortality_trend_idx = sampling_mortality_trend(n_sims=n_simulations, device=DEVICE)
        n = N[i]
        simulation_file_path = DOT_ENV.results_dir / f"portfolio_simulations_stokastik_n_{n}.pt"
        if not simulation_file_path.exists():
            print(f"Mulai simulasi portfolio N = {n}\n")
            chunk_size = n
            if n > chunk_threshold:
                chunk_size = chunk_threshold

            quotient, remainder = divmod(n, chunk_size)
            portfolio_simulation_value = torch.zeros(size=(n_simulations, 1), device=DEVICE)
            for j in range(quotient):
                print(f"Mulai simulasi untuk chunk ke {j + 1}/{quotient}...")
                portfolio_simulation_value += sampling_portfolio_value_stochastic(
                    mortality_trend_idx=mortality_trend_idx,
                    n_sims=n_simulations,
                    n_samples=chunk_size,
                    device=DEVICE
                )

            if remainder > 0:
                print(f"Mulai simulasi untuk {remainder} sample sisanya...")
                portfolio_simulation_value += sampling_portfolio_value_stochastic(
                    mortality_trend_idx=mortality_trend_idx,
                    n_sims=n_simulations,
                    n_samples=remainder,
                    device=DEVICE
                )

            print(f"\nMenyimpan ke {simulation_file_path}...")
            torch.save(obj=portfolio_simulation_value, f=simulation_file_path)
            portfolio_simulations_stokastik[i] = portfolio_simulation_value
        else:
            print(f"\nLoad {simulation_file_path}...")
            portfolio_simulations_stokastik[i] = torch.load(f=simulation_file_path, map_location=DEVICE)

        print(f"\nSimulasi portfolio N = {n} selesai")
        print(f"Hasil: {portfolio_simulations_stokastik[i][:5]}")
        print(55 * "=")

    print("\nSimulasi selesai!")
    print(f"Simpan hasil simulasi ke {portfolio_simulations_stokastik_file_path}")
    torch.save(obj=portfolio_simulations_stokastik, f=portfolio_simulations_stokastik_file_path)
    print("Selesai!")
else:
    print(f"Load {portfolio_simulations_stokastik_file_path}...")
    portfolio_simulations_stokastik = torch.load(f=portfolio_simulations_stokastik_file_path, map_location=DEVICE)
    print("Load selesai!")
# %%
portfolio_simulations_stokastik[(portfolio_simulations_stokastik < 0)]
# %%
alphas = [0.5, 0.9, 0.95, 0.99]
var_stokastik = {}

for a in alphas:
    var_stokastik[a] = []
    for i in range(len(N)):
        n = N[i]
        portfolio_n = portfolio_simulations_stokastik[i].squeeze(-1)
        _, indices = torch.sort(portfolio_n, descending=False)
        var_stokastik[a].append(portfolio_n[indices[int(a * n_simulations)]].item())
print(var_stokastik)
# %%
pd.DataFrame(var_stokastik, index=N).to_csv(DOT_ENV.results_dir / "var_stokastik.csv", decimal=",", sep=";")
# %%
mean = []
var = []
cv = []
for i in range(7):
    m = portfolio_simulations_stokastik[i].mean().item()
    v = portfolio_simulations_stokastik[i].var().item()
    s = portfolio_simulations_stokastik[i].std().item()
    mean.append(m)
    var.append(v)
    cv.append(s / m)

pd.DataFrame({
    "mean": mean,
    "var": var,
    "cv": cv
}, index=N).to_csv(DOT_ENV.results_dir / "portfolio_stokastik_statdesc.csv", sep=";", decimal=",")
# %% [markdown]
# ## Plot hasil simulasi
# %%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Union
import torch


def plot_portfolio_distribution(
    tensors: Union[list, dict],
    titles: Optional[list[str]] = None,
    percentiles: tuple[float, ...] = (2.5, 50, 97.5),
    figsize_per_plot: tuple[int, int] = (9, 4),
    ncols: int = 1,
    suptitle: Optional[str] = None,
    line_alpha: float = 0.8,
    fill_ci: bool = True,
    save_path: Optional[str] = None,
) -> None:
    """
    Plot KDE distribusi nilai tensor dengan garis persentil vertikal.

    Garis persentil diletakkan di nilai persentil ke-p dari data
    (sumbu x KDE adalah nilai data, bukan index).

    Parameters
    ----------
    tensors : list of array-like, atau dict {label: tensor}
        Satu atau beberapa tensor/array (shape 1-D atau akan di-flatten).
        Gunakan dict untuk memberi label otomatis per plot.
    titles : list of str, optional
        Judul per subplot. Diabaikan jika tensors berupa dict.
    percentiles : tuple of float
        Nilai persentil yang akan digambar sebagai garis vertikal.
        Default: (2.5, 50, 97.5).
    figsize_per_plot : tuple (w, h)
        Ukuran figure per subplot.
    ncols : int
        Jumlah kolom subplot. Baris dihitung otomatis.
    suptitle : str, optional
        Judul utama figure.
    line_alpha : float
        Opacity garis persentil (0-1).
    fill_ci : bool
        Jika True, isi area KDE antara persentil terkecil dan terbesar.
    save_path : str, optional
        Path untuk menyimpan figure (misal: 'plot.png').

    Returns
    -------
    fig : matplotlib.Figure
    """
    # --- normalise input ---
    if isinstance(tensors, dict):
        labels = list(tensors.keys())
        arrays = [
            v.detach().cpu().numpy() if isinstance(v, torch.Tensor)
            else np.asarray(v)
            for v in tensors.values()
        ]
    else:
        arrays = [
            v.detach().cpu().numpy() if isinstance(v, torch.Tensor)
            else np.asarray(v)
            for v in (tensors if isinstance(tensors, list) else [tensors])
        ]
        labels = titles if titles else [f"Tensor {i+1}" for i in range(len(arrays))]

    n_plots = len(arrays)
    ncols   = min(ncols, n_plots)
    nrows   = -(-n_plots // ncols)

    # --- palette & style ---
    sns.set_theme(style="white", font="DejaVu Sans")
    PCTL_COLORS = ["#378ADD", "#D85A30", "#1D9E75", "#7F77DD", "#BA7517"]
    KDE_COLOR   = "#5F5E5A"
    FILL_COLOR  = "#B5D4F4"

    W = figsize_per_plot[0] * ncols
    H = figsize_per_plot[1] * nrows
    fig, axes = plt.subplots(nrows, ncols, figsize=(W, H), squeeze=False, dpi=300)

    for idx, (arr, lbl) in enumerate(zip(arrays, labels)):
        ax   = axes[idx // ncols][idx % ncols]
        data = np.sort(arr.flatten())

        # KDE plot
        sns.kdeplot(
            data, ax=ax,
            color=KDE_COLOR, linewidth=1.5,
            fill=False,
        )

        # percentile values (pada sumbu x = nilai data)
        pctl_vals = np.percentile(data, percentiles)

        # optional fill antara persentil terkecil & terbesar
        if fill_ci and len(percentiles) >= 2:
            ax.axvspan(pctl_vals[0], pctl_vals[-1],
                       color=FILL_COLOR, alpha=0.2, zorder=1)

        # vertical percentile lines
        for i, (p, v) in enumerate(zip(percentiles, pctl_vals)):
            color = PCTL_COLORS[i % len(PCTL_COLORS)]
            ax.axvline(v, color=color, linewidth=1.6,
                       linestyle="--", alpha=line_alpha,
                       label=f"P{p:g} = {v:,.2f}", zorder=3)

        # axes styling
        ax.set_title(lbl, fontsize=13, fontweight="bold",
                    color="#2C2C2A", pad=10)
        ax.set_xlabel("Value", fontsize=10, color="#888780")
        ax.set_ylabel("Density", fontsize=10, color="#888780")
        ax.tick_params(labelsize=9, color="#B4B2A9")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#D3D1C7")
        ax.spines["bottom"].set_color("#D3D1C7")
        ax.set_facecolor("white")

        ax.legend(
            fontsize=8.5, frameon=True,
            framealpha=0.85, edgecolor="#D3D1C7",
            loc="upper left",
            bbox_to_anchor=(0.0, -0.30),
            ncol=len(percentiles),
            borderaxespad=0,
        ).get_frame().set_linewidth(0.5)

    # hide empty axes
    for idx in range(n_plots, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    if suptitle:
        fig.suptitle(suptitle, fontsize=15, fontweight="bold",
                    color="#2C2C2A", y=1.01)

    fig.patch.set_facecolor("white")
    plt.tight_layout(pad=2.0, h_pad=4.5)

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight",
                   facecolor="white")

    plt.show()
# %%
plot_portfolio_distribution(
    tensors=[portfolio_simulations_deterministik[i].reshape(-1) for i in range(7)],
    titles=["N = 1", "N = 10", "N = 100", "N = 1.000", "N = 10.000", "N = 100.000", "N = 1.000.000"],
    percentiles=(50, 90, 95, 99),
    figsize_per_plot=(9, 4),
    ncols=2,
    save_path=DOT_ENV.plots_dir / "portfolio_simulations_deterministik.png"
)
# %%
plot_portfolio_distribution(
    tensors=[portfolio_simulations_stokastik[i].reshape(-1) for i in range(7)],
    titles=["N = 1", "N = 10", "N = 100", "N = 1.000", "N = 10.000", "N = 100.000", "N = 1.000.000"],
    percentiles=(50, 90, 95, 99),
    figsize_per_plot=(9, 4),
    ncols=2,
    save_path=DOT_ENV.plots_dir / "portfolio_simulations_stokastik.png"
)