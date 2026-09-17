
import os
import numpy as np
import rasterio
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.stats import pearsonr
import warnings
from matplotlib.lines import Line2D

# 忽略计算中的Runtime警告
warnings.filterwarnings('ignore')

GPP_DIR  = r"E:\000000------文章1\数据库\GPP\annual"
NDVI_DIR = r"E:\000000------文章1\数据库\ndvi_and_NIRV\drive-download-20260420T023333Z-3-001"
NIRV_DIR = NDVI_DIR
LUCC_PATH = r"D:\00---全球NBP\Modis_LUCC_10_2_2023.tif"

FLUXSAT_GPP_DIR = r"E:\000000000000--GOSIFandFLUXSAT\fluxsat\year"
GOSIF_GPP_DIR   = r"E:\000000000000--GOSIFandFLUXSAT\gosif\year"
GOSIF_SIF_DIR   = r"E:\000000------文章1\数据库\SIF\1degree"

YEARS = list(range(2015, 2025))
TARGET_LUCC = [6, 7, 9, 10]

def calculate_correlation(gpp_stack, target_stack, mask):

    H, W = gpp_stack.shape[1], gpp_stack.shape[2]
    r_map = np.full((H, W), np.nan)
    p_map = np.full((H, W), np.nan)
    
    for i in range(H):
        for j in range(W):
            if mask[i, j]:
                g_ts = gpp_stack[:, i, j]
                t_ts = target_stack[:, i, j]

                if np.all(np.isnan(g_ts)) or np.all(np.isnan(t_ts)) or np.std(g_ts) == 0 or np.std(t_ts) == 0:
                    continue

                valid = ~np.isnan(g_ts) & ~np.isnan(t_ts)
                if np.sum(valid) < 3: continue
                
                r, p = pearsonr(g_ts[valid], t_ts[valid])
                r_map[i, j] = r
                p_map[i, j] = p
    return r_map, p_map

with rasterio.open(LUCC_PATH) as src:
    lucc = src.read(1)
    transform = src.transform
   
    extent = [transform[2], transform[2] + src.width * transform[0], 
              transform[5] + src.height * transform[4], transform[5]]
    mask_grass = np.isin(lucc, TARGET_LUCC)

def load_yearly_stack(folder, prefix, mask):
    stack = []
    for y in YEARS:
        if prefix == "GPP" or prefix == "GOSIF_GPP": 
            fname = f"GPP_{y}.tif"
        elif prefix == "NDVI": 
            fname = f"NDVI_Global_1deg_{y}.tif"
        elif prefix == "NIRv": 
            fname = f"NIRv_Global_1deg_{y}.tif"
        elif prefix == "FLUXSAT_GPP": 
            fname = f"FluxSat_GPP_{y}.tif"
        elif prefix == "GOSIF_SIF": 
            fname = f"GOSIF_SIF_{y}.tif"
        
        file_path = os.path.join(folder, fname)
        with rasterio.open(file_path) as src:
            data = src.read(1).astype(np.float32)
            data[~mask] = np.nan
            stack.append(data)
    return np.stack(stack)

gpp_stack        = load_yearly_stack(GPP_DIR, "GPP", mask_grass)
ndvi_stack       = load_yearly_stack(NDVI_DIR, "NDVI", mask_grass)
nirv_stack       = load_yearly_stack(NIRV_DIR, "NIRv", mask_grass)

fluxsat_gpp_stack = load_yearly_stack(FLUXSAT_GPP_DIR, "FLUXSAT_GPP", mask_grass)
gosif_gpp_stack   = load_yearly_stack(GOSIF_GPP_DIR, "GOSIF_GPP", mask_grass)
gosif_sif_stack   = load_yearly_stack(GOSIF_SIF_DIR, "GOSIF_SIF", mask_grass)

r_ndvi, p_ndvi = calculate_correlation(gpp_stack, ndvi_stack, mask_grass)

r_nirv, p_nirv = calculate_correlation(gpp_stack, nirv_stack, mask_grass)

r_flux_gosif, p_flux_gosif = calculate_correlation(fluxsat_gpp_stack, gosif_gpp_stack, mask_grass)

r_gpp_sif, p_gpp_sif = calculate_correlation(gpp_stack, gosif_sif_stack, mask_grass)

fig, axes = plt.subplots(2, 2, figsize=(18, 10), 
                         subplot_kw={'projection': ccrs.PlateCarree()}, 
                         dpi=300)
axes = axes.flatten()

data_list = [
    (r_ndvi, p_ndvi, "A. Correlation: GPP vs NDVI (2015-2024)"),
    (r_nirv, p_nirv, "B. Correlation: GPP vs NIRv (2015-2024)"),
    (r_flux_gosif, p_flux_gosif, "C. Correlation: FLUXSAT GPP vs GOSIF GPP (2015-2024)"),
    (r_gpp_sif, p_gpp_sif, "D. Correlation: GPP vs GOSIF SIF (2015-2024)")
]

lons = np.linspace(extent[0], extent[1], r_ndvi.shape[1])
lats = np.linspace(extent[3], extent[2], r_ndvi.shape[0])
lon_grid, lat_grid = np.meshgrid(lons, lats)

for i, (r_map, p_map, title) in enumerate(data_list):
    ax = axes[i]
    ax.set_global()
    ax.add_feature(cfeature.COASTLINE, linewidth=0.4, edgecolor='black')
    ax.add_feature(cfeature.OCEAN, facecolor='#f0f0f0')
    
    im = ax.imshow(r_map, extent=extent, transform=ccrs.PlateCarree(), 
                   cmap='RdBu_r', vmin=-1, vmax=1, origin='upper', zorder=1)
    
    sig_mask = (p_map < 0.05) & (~np.isnan(r_map))
    sig_lons = lon_grid[sig_mask]
    sig_lats = lat_grid[sig_mask]
    
    ax.scatter(sig_lons, sig_lats, s=0.05, c='black', marker='o',  
               alpha=0.7, transform=ccrs.PlateCarree(), zorder=5)
    
    ax.set_title(title, fontsize=14, fontweight='bold', loc='left')
    gl = ax.gridlines(draw_labels=True, linestyle='--', alpha=0.2, linewidth=0.5)
    gl.top_labels = gl.right_labels = False

cbar_ax = fig.add_axes([0.92, 0.25, 0.015, 0.5])
cbar = fig.colorbar(im, cax=cbar_ax, orientation='vertical')
cbar.set_label("Pearson's $r$", fontsize=12, fontweight='bold')

legend_elements = [Line2D([0], [0], marker='.', color='w', label='Significant ($P < 0.05$)', 
                          markerfacecolor='black', markersize=8)]
fig.legend(handles=legend_elements, loc='lower center', ncol=1, bbox_to_anchor=(0.5, 0.02), fontsize=12)

out_dir = r"D:\GPP_NDVI_comparison"
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

out_name = os.path.join(out_dir, "GPP_Validation_Final_ScatterSig_4Panels.pdf")

plt.subplots_adjust(left=0.05, right=0.9, top=0.95, bottom=0.1, wspace=0.1, hspace=0.2)
plt.savefig(out_name, dpi=600, bbox_inches='tight')
plt.show()