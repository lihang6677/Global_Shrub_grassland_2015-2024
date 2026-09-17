
import numpy as np
import rasterio
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
import glob
import os
from scipy import stats
import pandas as pd

plt.rcParams.update({
    'font.family':          'Arial',
    'font.size':            8,
    'axes.linewidth':       0.8,
    'axes.spines.top':      False,
    'axes.spines.right':    False,
    'xtick.major.width':    0.8,
    'ytick.major.width':    0.8,
    'xtick.major.size':     3,
    'ytick.major.size':     3,
    'xtick.direction':      'out',
    'ytick.direction':      'out',
    'pdf.fonttype':         42,
    'ps.fonttype':          42,
})


DATA_FOLDER = r"E:\000000------文章1\数据库\figure4"

AXIS_LIMITS = {
    # Carbon flux
    'carbon_mid': None,    
    'carbon_eq':  None,    

    # Soil moisture
    'sm_mid': (190, 225),
    'sm_eq':  None,

    # Temperature
    'temp_mid':(13.8, 16.0),
    'temp_eq':  (24, 28),

    'x_pad': 0.5,
}

FIG_WIDTH  = 10.80  
FIG_HEIGHT =  7.20

C_GPP      = '#2ca25f'
C_TER_GCAS = '#e6550d'
C_TER_GCB  = '#9e3a8c'
C_SM       = '#2166ac'
C_TEMP     = '#d73027'

SCATTER_S  = 26       
TREND_LW   = 1.6      
BAND_ALPHA = 0.12     

GS_HSPACE = 0.52
GS_WSPACE = 0.48
GS_LEFT   = 0.07
GS_RIGHT  = 0.95
GS_TOP    = 0.96
GS_BOTTOM = 0.08

def mann_kendall_test(x: np.ndarray):
    n = len(x)
    s = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            diff = x[j] - x[i]
            if diff > 0:   s += 1
            elif diff < 0: s -= 1
    _, counts = np.unique(x, return_counts=True)
    tie_sum = np.sum(counts * (counts - 1) * (2 * counts + 5))
    var_s   = (n * (n - 1) * (2 * n + 5) - tie_sum) / 18
    if   s > 0: z = (s - 1) / np.sqrt(var_s)
    elif s < 0: z = (s + 1) / np.sqrt(var_s)
    else:       z = 0.0
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))
    tau     = s / (n * (n - 1) / 2)
    return s, z, p_value, tau


import numpy as np
from scipy.stats import theilslopes
import pymannkendall as mk

def sens_slope(years: np.ndarray, data: np.ndarray):

    valid = ~np.isnan(data)
    if np.sum(valid) < 3:
        return np.nan, np.nan, np.nan, np.nan, np.nan
    
    x = years[valid].astype(float)
    y = data[valid]

    res = theilslopes(y, x)
    slope = res[0]
    intercept = res[1]

    mk_res = mk.original_test(y)
    p = mk_res.p
    z = mk_res.z
    tau = mk_res.Tau 

    return slope, intercept, p, tau, z

def sig_stars(p: float) -> str:

    if p < 0.001: return '***'
    elif p < 0.01: return '**'
    elif p < 0.05: return '*'
    else: return 'ns'

def _year_from_basename(fname: str) -> int | None:
    parts = os.path.basename(fname).replace('.tif', '').split('_')
    for part in reversed(parts):
        if part.isdigit() and len(part) == 4:
            return int(part)
    return None


def get_year_files(pattern: str, folder: str = DATA_FOLDER) -> list[str]:
    files = [f for f in glob.glob(os.path.join(folder, pattern))
             if _year_from_basename(f) is not None]
    return sorted(files, key=_year_from_basename)


def read_carbon_flux(
    file_list: list[str],
    masks: dict[str, np.ndarray],
    pixel_area: np.ndarray,
) -> dict[str, np.ndarray]:

    results = {k: [] for k in masks}
    for fpath in file_list:
        with rasterio.open(fpath) as src:
            arr = src.read(1).astype(float)
            arr[arr < -9000] = np.nan
        for name, mask in masks.items():
            val = np.nansum(arr[mask] * pixel_area[mask] * 1e6 / 1e15)
            results[name].append(val)
    return {k: np.array(v) for k, v in results.items()}


def read_env_variable(
    file_list: list[str],
    masks: dict[str, np.ndarray],
    pixel_area: np.ndarray,
) -> dict[str, np.ndarray]:

    results = {k: [] for k in masks}
    for fpath in file_list:
        with rasterio.open(fpath) as src:
            arr = src.read(1).astype(float)
            arr[arr < -9000] = np.nan
        for name, mask in masks.items():
            valid = mask & ~np.isnan(arr)
            val   = (np.nansum(arr[valid] * pixel_area[valid])
                     / np.sum(pixel_area[valid])) if valid.any() else np.nan
            results[name].append(val)
    return {k: np.array(v) for k, v in results.items()}

with rasterio.open(
    os.path.join("E:/000000------文章1/数据库/GPP/annual/GPP_Trend_Combined_Classification_2015_2024.tif")
) as src:
    classification = src.read(1)
    transform      = src.transform
    height, width  = src.shape

masks = {
    'mid_inc': classification == 21,
    'mid_dec': classification == 22,
    'equator': (classification == 11) | (classification == 12),
}

left  = transform[2];  top    = transform[5]
right = left + transform[0] * width
bottom = top + transform[4] * height
lons   = np.linspace(left,  right,  width)
lats   = np.linspace(top,   bottom, height)
lon_grid, lat_grid = np.meshgrid(lons, lats)

R           = 6371.0
lat_res_rad = np.deg2rad(abs(lats[1] - lats[0]))
lon_res_rad = np.deg2rad(abs(lons[1] - lons[0]))
lat_rad     = np.deg2rad(lat_grid)
pixel_area  = (R**2 * lon_res_rad
               * np.abs(np.sin(lat_rad + lat_res_rad / 2)
                        - np.sin(lat_rad - lat_res_rad / 2)))

gpp_files      = get_year_files("GPP_*.tif")
ter_gcas_files = get_year_files("TER_*.tif")
ter_gcb_files  = get_year_files("GCB_TER_*.tif")
smrt_files     = get_year_files("smrt_*.tif")
temp_files     = get_year_files("temp_*.tif")

years = np.array([_year_from_basename(f) for f in gpp_files])

gcb_years = np.array([_year_from_basename(f) for f in ter_gcb_files])
if not np.array_equal(years, gcb_years):
    print(f" error {gcb_years} ")

gpp_data      = read_carbon_flux(gpp_files,      masks, pixel_area)
ter_gcas_data = read_carbon_flux(ter_gcas_files, masks, pixel_area)
ter_gcb_data  = read_carbon_flux(ter_gcb_files,  masks, pixel_area)
smrt_data     = read_env_variable(smrt_files,    masks, pixel_area)
temp_data     = read_env_variable(temp_files,    masks, pixel_area)



regions   = ['mid_inc', 'mid_dec', 'equator']
variables = {
    'gpp':      gpp_data,
    'ter_gcas': ter_gcas_data,
    'ter_gcb':  ter_gcb_data,
    'smrt':     smrt_data,
    'temp':     temp_data,
}

trend_stats: dict[str, dict] = {}
for vname, vdict in variables.items():
    trend_stats[vname] = {}
    for region in regions:
        sl, ic, p, tau, z = sens_slope(years, vdict[region])
        trend_stats[vname][region] = {
            'slope': sl, 'intercept': ic, 'p': p, 'tau': tau, 'z': z,
        }

COL_TITLE = {
    'mid_inc': 'Mid-latitude increasing region',
    'mid_dec': 'Mid-latitude decreasing region',
    'equator': 'Equatorial region (±15°)',
}


def _add_trend_line(
    ax, years: np.ndarray, data: np.ndarray,
    color: str, marker: str, label: str,
    scatter_s: float = SCATTER_S,
    lw: float = TREND_LW,
    band_alpha: float = BAND_ALPHA,
) -> dict:
    sl, ic, p, tau, z = sens_slope(years, data)
    trend    = sl * years + ic
    resid_se = np.nanstd(data - trend)

    ax.fill_between(years,
                    trend - 1.96 * resid_se,
                    trend + 1.96 * resid_se,
                    color=color, alpha=band_alpha, zorder=1, linewidth=0)
    ax.plot(years, trend, color=color, linewidth=lw, alpha=0.9, zorder=3)
    ax.scatter(years, data, color=color, s=scatter_s, marker=marker,
               edgecolors='white', linewidths=0.6, label=label, zorder=4)
    return {'slope': sl, 'intercept': ic, 'p': p, 'tau': tau, 'z': z}


def _apply_ylim(ax, var_type: str, region: str):
    """
    var_type: carbon / sm / temp
    region: mid_inc / mid_dec / equator
    """
    if region == 'equator':
        lim = AXIS_LIMITS.get(f"{var_type}_eq")
    else:
        lim = AXIS_LIMITS.get(f"{var_type}_mid")

    if lim is not None:
        ax.set_ylim(lim)


def _apply_xlim(ax, years: np.ndarray) -> None:
    pad = AXIS_LIMITS.get('x_pad', 0.5)
    ax.set_xlim(years[0] - pad, years[-1] + pad)


def _stat_line(prefix: str, st: dict, unit: str) -> str:
    sign = '+' if st['slope'] > 0 else ''
    return (f"{prefix}  β={sign}{st['slope']:.4f} {unit}"
            f"   τ={st['tau']:+.2f}  {sig_stars(st['p'])}")

def plot_carbon_panel(
    ax,
    years: np.ndarray,
    gpp:      np.ndarray,
    ter_gcas: np.ndarray,
    ter_gcb:  np.ndarray,
    region:   str,
    letter:   str,
) -> None:

    st_g  = _add_trend_line(ax, years, gpp,      C_GPP,      'o', 'GPP (GCAS)')
    st_tc = _add_trend_line(ax, years, ter_gcas, C_TER_GCAS, 's', 'TER (GCAS)')
    st_tb = _add_trend_line(ax, years, ter_gcb,  C_TER_GCB,  '^', 'TER (GCB2025)')

    info = '\n'.join([
        _stat_line('GPP ',     st_g,  'Pg C yr⁻²'),
        _stat_line('TER-GCAS', st_tc, 'Pg C yr⁻²'),
        _stat_line('TER-GCB',  st_tb, 'Pg C yr⁻²'),
    ])
    ax.text(
        0.03, 0.97, info,
        transform=ax.transAxes,
        fontsize=6.0, va='top', ha='left', family='Arial',
        linespacing=1.55,
        bbox=dict(boxstyle='round,pad=0.32', facecolor='white',
                  edgecolor='#bbbbbb', linewidth=0.6, alpha=0.93),
    )

    ax.set_title(f'({letter}) {COL_TITLE[region]}',
                 fontsize=8.5, fontweight='bold', loc='left', pad=5)
    ax.set_ylabel('Carbon flux (Pg C yr$^{-1}$)', fontsize=8, labelpad=3)
    ax.set_xlabel('Year', fontsize=8, labelpad=3)
    _apply_xlim(ax, years)
    _apply_ylim(ax, 'carbon', region)
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.tick_params(labelsize=7.5)
    ax.legend(
        loc='lower right', fontsize=6.5, frameon=True,
        fancybox=False, edgecolor='#cccccc', framealpha=0.92,
        handlelength=1.4, handletextpad=0.4, borderpad=0.4,
        ncol=1,
    )
    ax.grid(True, alpha=0.15, linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)


def plot_env_panel(
    ax,
    ax2,
    years:  np.ndarray,
    smrt:   np.ndarray,
    temp:   np.ndarray,
    region: str,
    letter: str,
) -> None:

    st_s = _add_trend_line(ax,  years, smrt, C_SM,   'o', 'Soil moisture')
    st_t = _add_trend_line(ax2, years, temp, C_TEMP, 's', 'Temperature')

    ax.set_ylabel('Soil moisture (mm)', fontsize=8, color=C_SM, labelpad=3)
    ax.tick_params(axis='y', labelcolor=C_SM,   labelsize=7.5)
    ax.tick_params(axis='x',                    labelsize=7.5)
    _apply_ylim(ax, 'sm', region)

    ax2.set_ylabel('Temperature (°C)', fontsize=8, color=C_TEMP, labelpad=3)
    ax2.tick_params(axis='y', labelcolor=C_TEMP, labelsize=7.5)
    ax2.spines['top'].set_visible(False)
    _apply_ylim(ax2, 'temp', region)

    info = '\n'.join([
        _stat_line('SM  ', st_s, 'mm yr⁻¹'),
        _stat_line('Temp', st_t, '°C yr⁻¹'),
    ])
    ax.text(
        0.03, 0.03, info,
        transform=ax.transAxes,
        fontsize=6.0, va='bottom', ha='left', family='Arial',
        linespacing=1.55,
        bbox=dict(boxstyle='round,pad=0.32', facecolor='white',
                  edgecolor='#bbbbbb', linewidth=0.6, alpha=0.93),
    )

    handles = [
        Line2D([0], [0], marker='o', color=C_SM,   markersize=5,
               linewidth=1.5, markeredgecolor='white', label='Soil moisture'),
        Line2D([0], [0], marker='s', color=C_TEMP, markersize=5,
               linewidth=1.5, markeredgecolor='white', label='Temperature'),
    ]
    ax.legend(
        handles=handles, loc='upper left', fontsize=6.5,
        frameon=True, fancybox=False, edgecolor='#cccccc',
        framealpha=0.92, handlelength=1.4,
        handletextpad=0.4, borderpad=0.4,
    )

    ax.set_title(f'({letter}) {COL_TITLE[region]} — climate drivers',
                 fontsize=8.5, fontweight='bold', loc='left', pad=5)
    ax.set_xlabel('Year', fontsize=8, labelpad=3)
    _apply_xlim(ax,  years)
    _apply_xlim(ax2, years)
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.grid(True, alpha=0.15, linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)


fig = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=300)
gs  = gridspec.GridSpec(
    2, 3, figure=fig,
    hspace=GS_HSPACE, wspace=GS_WSPACE,
    left=GS_LEFT, right=GS_RIGHT,
    top=GS_TOP,   bottom=GS_BOTTOM,
)

ax_a = fig.add_subplot(gs[0, 0])
ax_b = fig.add_subplot(gs[0, 1])
ax_c = fig.add_subplot(gs[0, 2])

ax_d = fig.add_subplot(gs[1, 0])
ax_e = fig.add_subplot(gs[1, 1])
ax_f = fig.add_subplot(gs[1, 2])

ax_d2 = ax_d.twinx()
ax_e2 = ax_e.twinx()
ax_f2 = ax_f.twinx()

for ax in [ax_d, ax_e, ax_f]:
    ax.spines['right'].set_visible(False)

for ax, region, letter in [
    (ax_a, 'mid_inc', 'a'),
    (ax_b, 'mid_dec', 'b'),
    (ax_c, 'equator', 'c'),
]:
    plot_carbon_panel(
        ax, years,
        gpp_data[region],
        ter_gcas_data[region],
        ter_gcb_data[region],
        region, letter,
    )

for ax, ax2, region, letter in [
    (ax_d, ax_d2, 'mid_inc', 'd'),
    (ax_e, ax_e2, 'mid_dec', 'e'),
    (ax_f, ax_f2, 'equator', 'f'),
]:
    plot_env_panel(ax, ax2, years,
                   smrt_data[region], temp_data[region],
                   region, letter)

output_base = os.path.join(DATA_FOLDER, 'Nature_2x3_SensSlope_MK_GCB')
for fmt in ('png', 'pdf', 'svg'):
    fig.savefig(f"{output_base}.{fmt}",
                dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ saved: {output_base}.{fmt}")

plt.show()

region_labels = {
    'mid_inc': 'Mid-lat. increasing',
    'mid_dec': 'Mid-lat. decreasing',
    'equator': 'Equatorial (±15°)',
}
var_meta = {
    'gpp':      ('GPP',       'Pg C yr⁻²'),
    'ter_gcas': ('TER-GCAS',  'Pg C yr⁻²'),
    'ter_gcb':  ('TER-GCB',   'Pg C yr⁻²'),
    'smrt':     ('SM',        'mm yr⁻¹'),
    'temp':     ('Temp',      '°C yr⁻¹'),
}

rows = []
for region, rlabel in region_labels.items():
    for vkey, (vname, unit) in var_meta.items():
        s = trend_stats[vkey][region]
        rows.append({
            'Region':    rlabel,
            'Variable':  vname,
            'Sen slope': f"{s['slope']:+.5f}",
            'Unit':      unit,
            'Kendall τ': f"{s['tau']:+.3f}",
            'Z':         f"{s['z']:+.3f}",
            'p-value':   f"{s['p']:.4f}",
            'Sig.':      sig_stars(s['p']),
        })

df_summary = pd.DataFrame(rows)
print(df_summary.to_string(index=False))

csv_path = os.path.join(DATA_FOLDER, 'SenSlope_MK_Summary_GCB.csv')
df_summary.to_csv(csv_path, index=False, encoding='utf-8-sig')


df_ts = pd.DataFrame({
    'Year':                years,
    'GPP_MidLat_Inc':      gpp_data['mid_inc'],
    'GPP_MidLat_Dec':      gpp_data['mid_dec'],
    'GPP_Equator':         gpp_data['equator'],
    'TER_GCAS_MidLat_Inc': ter_gcas_data['mid_inc'],
    'TER_GCAS_MidLat_Dec': ter_gcas_data['mid_dec'],
    'TER_GCAS_Equator':    ter_gcas_data['equator'],
    'TER_GCB_MidLat_Inc':  ter_gcb_data['mid_inc'],
    'TER_GCB_MidLat_Dec':  ter_gcb_data['mid_dec'],
    'TER_GCB_Equator':     ter_gcb_data['equator'],
    'NEP_GCAS_MidLat_Inc': gpp_data['mid_inc'] - ter_gcas_data['mid_inc'],
    'NEP_GCAS_MidLat_Dec': gpp_data['mid_dec'] - ter_gcas_data['mid_dec'],
    'NEP_GCAS_Equator':    gpp_data['equator'] - ter_gcas_data['equator'],
    'NEP_GCB_MidLat_Inc':  gpp_data['mid_inc'] - ter_gcb_data['mid_inc'],
    'NEP_GCB_MidLat_Dec':  gpp_data['mid_dec'] - ter_gcb_data['mid_dec'],
    'NEP_GCB_Equator':     gpp_data['equator'] - ter_gcb_data['equator'],
    'SM_MidLat_Inc_mm':    smrt_data['mid_inc'],
    'SM_MidLat_Dec_mm':    smrt_data['mid_dec'],
    'SM_Equator_mm':       smrt_data['equator'],
    'Temp_MidLat_Inc_C':   temp_data['mid_inc'],
    'Temp_MidLat_Dec_C':   temp_data['mid_dec'],
    'Temp_Equator_C':      temp_data['equator'],
})
ts_path = os.path.join(DATA_FOLDER, 'TimeSeries_All_Regions_GCB.csv')
df_ts.to_csv(ts_path, index=False, float_format='%.6f', encoding='utf-8-sig')