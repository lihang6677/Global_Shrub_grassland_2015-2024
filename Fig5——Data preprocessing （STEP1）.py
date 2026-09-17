

import os
import warnings
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import rasterio

warnings.filterwarnings("ignore")

# =========================================================================
# 0. Logging
# =========================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


class Config:
    # Paths
    BASE_PATH        = Path(r"E:/000000000000--GCAS/Month")
    LUCC_PATH        = Path(r"D:/00---全球NBP/Modis_LUCC_10_2_2023.tif")
    CLASS_PATH       = Path(r"E:/000000------文章1/数据库/GPP/annual"
                            r"/GPP_Trend_Combined_Classification_2015_2024.tif")
    OUTPUT_CSV       = Path(r"E:/SEM_ZScore_Monthly_2015_2024_STRICT_NEW0707.csv")

    # Time range
    START_YEAR = 2015
    END_YEAR   = 2024

    # LUCC classes for shrub/grass vegetation
    LUCC_CLASSES: List[int] = [6, 7, 9, 10]

    # Region definitions: name -> set of classification pixel values
    REGIONS: Dict[str, List[int]] = {
        "Mid_Latitude": [21, 22],
        "Equator":      [11, 12],
    }

    # Variable subdirectory names and filename patterns
    VAR_DIRS: Dict[str, str] = {
        "Temperature":   "temp",
        "Soil_Moisture": "smrt",
        #"PRE":           "PRE",
        "GPP":           "gpp",
        "TER":           "ter",
        "VPD":           "vpd",
        "LAI":           "lai",
        #"SSRD":          "SSRD",
        "CO2":           "co2",
    }

    FILE_PATTERNS: Dict[str, str] = {
        "Temperature":   "temp_{year}_{month:02d}.tif",
        "Soil_Moisture": "smrt_month_{year}_{month:02d}.tif",
        #"PRE":           "PRE_{year}_{month:02d}.tif",
        "GPP":           "GPP_{year}_{month:02d}.tif",
        "TER":           "TER_{year}_{month:02d}.tif",
        "VPD":           "VPD_{year}_{month:02d}.tif",
        "LAI":           "LAI_{year}_{month:02d}.tif",
        #"SSRD":          "SSRD_{year}_{month:02d}.tif",
        "CO2":           "CO2_{year}{month:02d}.tif",
    }

    # Physical plausibility ranges — variables absent here skip range check
    PHYSICAL_RANGES: Dict[str, Tuple[float, float]] = {
        "Temperature":   (-60,   60),
        "Soil_Moisture": (  0, 1000),
        "GPP":           (-1000, 2000),
        "TER":           (-1000, 2000),
        "VPD":           (  0,   10),
        "LAI":           (  0,   10),
        "CO2":           (300,  500),
    }

    OUTLIER_ZSCORE_THRESHOLD: float = 3.5
    MIN_VALID_RATIO:          float = 0.75   # 至少75%月份有数据才计算Z-score
    MIN_VARIABILITY_STD:      float = 0.05   # 标准差低于此值视为低变异性

    SEM_REQUIRED: List[str] = [
        "Temperature", "Soil_Moisture",
        "CO2",
        "GPP", "TER", "VPD", "LAI",
    ]

    OUTPUT_COLUMNS: List[str] = [
        "Region", "Year", "Month",
        "Temperature", "Soil_Moisture", "T_SM_Interaction",
        "CO2",
        "GPP", "TER", "VPD", "LAI",
    ]


def _read_raster(path: Path) -> Tuple[np.ndarray, dict]:
    """Read a single-band raster; return (array float32, profile)."""
    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float32)
        profile = src.profile
        if src.nodata is not None:
            arr[arr == src.nodata] = np.nan
    arr[~np.isfinite(arr)] = np.nan
    return arr, profile


def load_masks(cfg: Config) -> Tuple[Dict[str, np.ndarray], dict]:
    """
    Load LUCC + classification rasters and return per-region boolean masks.

    Returns
    -------
    masks   : {region_name: bool ndarray}
    profile : rasterio profile of the LUCC raster (used as spatial reference)
    """
    lucc, profile = _read_raster(cfg.LUCC_PATH)
    cls_arr, cls_profile = _read_raster(cfg.CLASS_PATH)

    if lucc.shape != cls_arr.shape:
        raise ValueError(
            f"Shape mismatch: LUCC {lucc.shape} vs Classification {cls_arr.shape}. "
            "Reproject to a common grid before running."
        )

    lucc_mask = np.isin(lucc, cfg.LUCC_CLASSES)

    masks: Dict[str, np.ndarray] = {}
    for region, cls_values in cfg.REGIONS.items():
        region_cls_mask = np.isin(cls_arr, cls_values)
        masks[region] = region_cls_mask & lucc_mask
        log.info("  %-15s: %d pixels", region, masks[region].sum())

    return masks, profile


def _apply_physical_range(arr: np.ndarray, var: str, cfg: Config) -> np.ndarray:
    """Set values outside physical range to NaN."""
    if var not in cfg.PHYSICAL_RANGES:
        return arr
    lo, hi = cfg.PHYSICAL_RANGES[var]
    arr[(arr < lo) | (arr > hi)] = np.nan
    return arr


def _remove_statistical_outliers(arr: np.ndarray, threshold: float) -> np.ndarray:
    """
    Modified Z-score outlier removal (Iglewicz & Hoaglin 1993).
    Threshold = 3.5 is the standard recommendation.
    Pixels with |MZS| > threshold are set to NaN.
    """
    median = np.nanmedian(arr)
    mad = np.nanmedian(np.abs(arr - median))
    if mad == 0:          # 常数场，无法判断异常值，跳过
        return arr
    mzs = 0.6745 * (arr - median) / mad
    arr[np.abs(mzs) > threshold] = np.nan
    return arr


def load_monthly_tile(
    var: str, year: int, month: int, cfg: Config
) -> Optional[np.ndarray]:
    """
    Load, nodata-mask, physical-filter, and outlier-filter one tile.

    Returns None if file is missing or unreadable (logged as warning).
    """
    fp = (cfg.BASE_PATH / cfg.VAR_DIRS[var]
          / cfg.FILE_PATTERNS[var].format(year=year, month=month))

    if not fp.exists():
        log.debug("Missing: %s", fp)
        return None

    try:
        arr, _ = _read_raster(fp)
    except rasterio.errors.RasterioIOError as exc:
        log.warning("Cannot read %s: %s", fp, exc)
        return None

    arr = _apply_physical_range(arr, var, cfg)
    arr = _remove_statistical_outliers(arr, cfg.OUTLIER_ZSCORE_THRESHOLD)
    return arr


def compute_robust_zscore(stack: np.ndarray) -> np.ndarray:
    """
    Pixel-wise climatological anomaly standardisation.

        Z(t) = [X(t) - median_clim(month)] / MAD_clim(month)

    Parameters
    ----------
    stack : float32 ndarray, shape (T, H, W).  T must be divisible by 12.

    Returns
    -------
    zscore_stack : same shape as stack, NaN where clim MAD == 0.
    """
    T, H, W = stack.shape
    if T % 12 != 0:
        raise ValueError(f"Stack length T={T} must be a multiple of 12.")

    n_years = T // 12
    # shape: (n_years, 12, H, W)
    cube = stack.reshape(n_years, 12, H, W)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        clim_med = np.nanmedian(cube, axis=0)          # (12, H, W)
        clim_mad = np.nanmedian(
            np.abs(cube - clim_med[None]), axis=0
        )                                               # (12, H, W)

    clim_mad[clim_mad == 0] = np.nan

    month_indices = np.arange(T) % 12              # (T,)
    med_t = clim_med[month_indices]                # (T, H, W)
    mad_t = clim_mad[month_indices]                # (T, H, W)

    return (stack - med_t) / mad_t                 # (T, H, W)

def aggregate_to_regions(
    zscore_stack: np.ndarray,
    masks: Dict[str, np.ndarray],
) -> Dict[str, List[Optional[float]]]:
    """
    Spatially average Z-score stack over each region mask.

    Returns {region: [float or NaN, ...]} with length T.
    """
    T = zscore_stack.shape[0]
    result: Dict[str, List[Optional[float]]] = {r: [] for r in masks}

    for t in range(T):
        frame = zscore_stack[t]
        for region, mask in masks.items():
            vals = frame[mask]
            valid = vals[~np.isnan(vals)]
            result[region].append(float(np.mean(valid)) if valid.size > 0 else np.nan)

    return result

def build_time_index(cfg: Config) -> List[Tuple[int, int]]:
    return [
        (y, m)
        for y in range(cfg.START_YEAR, cfg.END_YEAR + 1)
        for m in range(1, 13)
    ]


def process_variable(
    var: str,
    time_index: List[Tuple[int, int]],
    masks: Dict[str, np.ndarray],
    profile: dict,
    cfg: Config,
) -> Tuple[Dict[str, List[Optional[float]]], Optional[np.ndarray]]:
    """
    Full pipeline for one variable:
      load tiles → build stack → Z-score → aggregate.

    Returns
    -------
    region_ts  : {region: time_series}
    zscore_stk : raw Z-score stack (H,W,T) — returned only for interaction
                 computation; None if variable is not needed for interaction.
    """
    nT = len(time_index)
    H, W = profile["height"], profile["width"]
    stack = np.full((nT, H, W), np.nan, dtype=np.float32)

    valid_count = 0
    for t, (y, m) in enumerate(time_index):
        tile = load_monthly_tile(var, y, m, cfg)
        if tile is not None:
            stack[t] = tile
            valid_count += 1

    nan_ts = {r: [np.nan] * nT for r in masks}

    if valid_count < nT * cfg.MIN_VALID_RATIO:
        log.warning(
            "%-20s: only %d/%d months valid — skipping Z-score", var, valid_count, nT
        )
        return nan_ts, None

    zscore_stk = compute_robust_zscore(stack)
    del stack

    region_ts = aggregate_to_regions(zscore_stk, masks)


    for region, ts in region_ts.items():
        if np.nanstd(ts) < cfg.MIN_VARIABILITY_STD:
            log.warning("  Low variability — %s / %s", var, region)

    return region_ts, zscore_stk


def compute_interaction(
    t_stack: np.ndarray,
    sm_stack: np.ndarray,
    masks: Dict[str, np.ndarray],
) -> Dict[str, List[Optional[float]]]:
    """
    Pixel-wise interaction term: Z(T) × Z(SM), then spatially averaged.

    Parameters
    ----------
    t_stack  : Z-score stack for Temperature, shape (T, H, W)
    sm_stack : Z-score stack for Soil_Moisture, shape (T, H, W)
    masks    : {region: bool ndarray}

    Returns
    -------
    {region: [float or NaN, ...]} with length T
    """
    interaction_stack = t_stack * sm_stack          # element-wise, NaN-safe
    return aggregate_to_regions(interaction_stack, masks)

def orthogonalize_vpd(
    vpd_stack: np.ndarray,
    t_stack: np.ndarray
) -> np.ndarray:

    Tn, H, W = vpd_stack.shape
    vpd_ortho = np.full_like(vpd_stack, np.nan, dtype=np.float32)

    for i in range(H):
        for j in range(W):
            v = vpd_stack[:, i, j]
            t = t_stack[:, i, j]

            mask = (~np.isnan(v)) & (~np.isnan(t))
            if np.sum(mask) < 10:
                continue

            v_valid = v[mask]
            t_valid = t[mask]

            # OLS: v = a + b*t
            A = np.vstack([t_valid, np.ones_like(t_valid)]).T
            b, a = np.linalg.lstsq(A, v_valid, rcond=None)[0]

            # residual
            vpd_ortho[mask, i, j] = v_valid - (a + b * t_valid)

    return vpd_ortho

def _section(title: str) -> None:
    log.info("")
    log.info("=" * 70)
    log.info("  %s", title)
    log.info("=" * 70)


def print_quality_report(
    df: pd.DataFrame,
    region_masks: Dict[str, np.ndarray],
    cfg: Config,
    n_total_months: int,
) -> None:
    """Consolidated quality report written via logging."""

    _section("QUALITY REPORT")

    # --- 1. Completeness ---
    log.info("\n[1] Data Completeness")
    for var in cfg.SEM_REQUIRED:
        n_valid = df[var].notna().sum()
        pct = n_valid / len(df) * 100
        log.info("  %-25s : %4d / %d  (%.1f%%)", var, n_valid, len(df), pct)

    # --- 2. Summary statistics per region ---
    log.info("\n[2] Summary Statistics (Z-scores)")
    for region in region_masks:
        log.info("\n  %s", region)
        sub = df[df["Region"] == region][cfg.SEM_REQUIRED]
        stats = sub.describe().loc[["mean", "std", "min", "max"]].round(3)
        log.info("\n%s", stats.to_string())

    # --- 3. Temporal coverage ---
    log.info("\n[3] Temporal Coverage by Region")
    coverage = df.groupby("Region").size()
    for region, count in coverage.items():
        pct = count / n_total_months * 100
        log.info("  %-15s : %d / %d months  (%.1f%%)", region, count,
                 n_total_months, pct)

    # --- 4. Variability check ---
    log.info("\n[4] Variability Check  (threshold std > %.2f)", cfg.MIN_VARIABILITY_STD)
    check_vars = cfg.SEM_REQUIRED + ["T_SM_Interaction"]
    for region in region_masks:
        log.info("\n  %s", region)
        sub = df[df["Region"] == region]
        for var in check_vars:
            if var not in sub.columns:
                continue
            std  = sub[var].std()
            mean = sub[var].mean()
            flag = "✓" if std > cfg.MIN_VARIABILITY_STD else "⚠"
            log.info("    %s %-22s : mean=%7.3f  std=%.3f", flag, var, mean, std)

    # --- 5. Interaction term multicollinearity check ---
    log.info("\n[5] Interaction Term Correlation (VIF proxy)")
    for region in region_masks:
        sub = df[df["Region"] == region]
        corr_t_sm   = sub[["Temperature",    "Soil_Moisture"]].corr().iloc[0, 1]
        corr_t_int  = sub[["Temperature",    "T_SM_Interaction"]].corr().iloc[0, 1]
        corr_sm_int = sub[["Soil_Moisture",  "T_SM_Interaction"]].corr().iloc[0, 1]
        log.info("\n  %s", region)
        log.info("    Corr(T,  SM)     = %+.3f", corr_t_sm)
        log.info("    Corr(T,  T×SM)   = %+.3f", corr_t_int)
        log.info("    Corr(SM, T×SM)   = %+.3f", corr_sm_int)
        if abs(corr_t_int) > 0.8 or abs(corr_sm_int) > 0.8:
            log.warning("    ⚠  High correlation — potential multicollinearity")
        else:
            log.info("    ✓  Correlation levels acceptable")

    # --- 6. Regional mean differences ---
    log.info("\n[6] Regional Comparison  (Mid_Latitude − Equator)")
    mid = df[df["Region"] == "Mid_Latitude"]
    eq  = df[df["Region"] == "Equator"]
    for var in ["Temperature", "Soil_Moisture", "GPP", "TER", "VPD", "LAI"]:
        diff = mid[var].mean() - eq[var].mean()
        log.info("  %-20s : %+.3f", var, diff)


def save_regional_comparison(
    df: pd.DataFrame,
    cfg: Config,
) -> Path:
    """Save per-variable mean/std for each region to a companion CSV."""
    records = []
    for var in cfg.SEM_REQUIRED:
        row = {"Variable": var}
        for region in cfg.REGIONS:
            sub = df[df["Region"] == region][var]
            row[f"{region}_Mean"] = sub.mean()
            row[f"{region}_Std"]  = sub.std()
        # Difference: first region minus second region
        regions = list(cfg.REGIONS.keys())
        if len(regions) >= 2:
            row["Difference"] = (
                df[df["Region"] == regions[0]][var].mean()
                - df[df["Region"] == regions[1]][var].mean()
            )
        records.append(row)

    comp_df  = pd.DataFrame(records)
    comp_csv = cfg.OUTPUT_CSV.with_name(
        cfg.OUTPUT_CSV.stem + "_regional_comparison.csv"
    )
    comp_df.to_csv(comp_csv, index=False, float_format="%.6f")
    log.info("✓ Regional comparison saved → %s", comp_csv)
    return comp_csv


def main() -> None:
    cfg = Config()

    _section("SEM MONTHLY DATA PREPROCESSING")
    log.info("Period  : %d – %d", cfg.START_YEAR, cfg.END_YEAR)
    log.info("Regions : %s", list(cfg.REGIONS.keys()))

    # ------------------------------------------------------------------
    # Step 1 — Spatial masks
    # ------------------------------------------------------------------
    log.info("\n[1/4] Loading spatial masks …")
    region_masks, profile = load_masks(cfg)

    # ------------------------------------------------------------------
    # Step 2 — Time index
    # ------------------------------------------------------------------
    log.info("\n[2/4] Building time index …")
    time_index = build_time_index(cfg)
    nT = len(time_index)
    log.info("  %d time steps  (%d-%02d → %d-%02d)",
             nT,
             time_index[0][0],  time_index[0][1],
             time_index[-1][0], time_index[-1][1])

    # ------------------------------------------------------------------
    # Step 3 — Variable processing
    # ------------------------------------------------------------------
    log.info("\n[3/4] Processing variables …")
    
    regional_ts: Dict[str, Dict[str, List[Optional[float]]]] = {
        r: {} for r in region_masks
    }
    
    interaction_stacks: Dict[str, np.ndarray] = {}
    INTERACTION_VARS = {"Temperature", "Soil_Moisture", "VPD"}
    
    for var in cfg.VAR_DIRS:
        log.info("  %-22s …", var)
        region_ts, zscore_stk = process_variable(
            var, time_index, region_masks, profile, cfg
        )
    
        for region in region_masks:
            regional_ts[region][var] = region_ts[region]
    
        if var in INTERACTION_VARS and zscore_stk is not None:
            interaction_stacks[var] = zscore_stk
    
    # ------------------------------------------------------------------
    # Step 3a — Orthogonalize VPD (关键步骤)
    # ------------------------------------------------------------------
    log.info("  %-22s …", "Orthogonalizing VPD")
    
    if "VPD" in interaction_stacks and "Temperature" in interaction_stacks:
    
        vpd_ortho_stack = orthogonalize_vpd(
            interaction_stacks["VPD"],
            interaction_stacks["Temperature"]
        )
    
        # 替换 stack
        interaction_stacks["VPD"] = vpd_ortho_stack
    
        # 更新区域时间序列（非常关键，否则SEM用的还是旧VPD）
        vpd_ts = aggregate_to_regions(vpd_ortho_stack, region_masks)
        for region in region_masks:
            regional_ts[region]["VPD"] = vpd_ts[region]
    
    else:
        log.warning("VPD or Temperature missing — cannot orthogonalize")
    
    # ------------------------------------------------------------------
    # Step 3b — Interaction term
    # ------------------------------------------------------------------
    log.info("  %-22s …", "T_SM_Interaction")
    
    if {"Temperature", "Soil_Moisture"}.issubset(interaction_stacks):
        interaction_ts = compute_interaction(
            interaction_stacks["Temperature"],
            interaction_stacks["Soil_Moisture"],
            region_masks,
        )
        for region in region_masks:
            regional_ts[region]["T_SM_Interaction"] = interaction_ts[region]
    else:
        log.warning("Missing variables for interaction")
        for region in region_masks:
            regional_ts[region]["T_SM_Interaction"] = [np.nan] * nT
    
    # 最后再释放
    del interaction_stacks
    # ------------------------------------------------------------------
    # Step 4 — Assemble DataFrame
    # ------------------------------------------------------------------
    log.info("\n[4/4] Assembling final DataFrame …")

    frames = []
    years  = [y for y, _ in time_index]
    months = [m for _, m in time_index]

    for region in region_masks:
        frame = pd.DataFrame(regional_ts[region])
        frame["Region"] = region
        frame["Year"]   = years
        frame["Month"]  = months
        frames.append(frame)

    df = pd.concat(frames, ignore_index=True)

    # Enforce column order (only keep columns that actually exist)
    existing_cols = [c for c in cfg.OUTPUT_COLUMNS if c in df.columns]
    df = df[existing_cols]

    # Drop rows missing any SEM-required variable
    n_before = len(df)
    df = df.dropna(subset=cfg.SEM_REQUIRED)
    n_dropped = n_before - len(df)
    log.info("  Dropped %d incomplete rows → %d rows remaining", n_dropped, len(df))

    # ------------------------------------------------------------------
    # Step 5 — Save outputs
    # ------------------------------------------------------------------
    df.to_csv(cfg.OUTPUT_CSV, index=False, float_format="%.6f")
    log.info("✓ Main dataset saved → %s", cfg.OUTPUT_CSV)

    save_regional_comparison(df, cfg)

    # ------------------------------------------------------------------
    # Step 6 — Quality report
    # ------------------------------------------------------------------
    print_quality_report(df, region_masks, cfg, nT)

    _section("PIPELINE COMPLETE")


if __name__ == "__main__":
    main()