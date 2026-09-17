
import numpy as np
import rasterio
from scipy import stats
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.patches as mpatches
import matplotlib
import glob
import os

matplotlib.use('Agg')

plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 10,
    'axes.linewidth': 0.8,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'xtick.major.size': 4,
    'ytick.major.size': 4,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})


gpp_folder = r"E:\000000------文章1\数据库\GPP\annual"
import re

gpp_files = []
pattern = re.compile(r'GPP_(\d{4})\.tif$')

for file in sorted(glob.glob(os.path.join(gpp_folder, "GPP_*.tif"))):
    basename = os.path.basename(file)
    match = pattern.match(basename)
    if match:
        gpp_files.append(file)


with rasterio.open(gpp_files[0]) as src:
    meta = src.meta
    transform = src.transform
    height, width = src.shape
    crs = src.crs
    
    left = transform[2]
    top = transform[5]
    right = left + transform[0] * width
    bottom = top + transform[4] * height
    

lons = np.linspace(left, right, width)
lats = np.linspace(top, bottom, height)
lon_grid, lat_grid = np.meshgrid(lons, lats)

gpp_stack = []
years = []

for gpp_file in gpp_files:
    year = int(os.path.basename(gpp_file).split('_')[1].split('.')[0])
    years.append(year)
    
    with rasterio.open(gpp_file) as src:
        gpp_data = src.read(1).astype(float)
        gpp_data[gpp_data < -9000] = np.nan
        gpp_stack.append(gpp_data)

gpp_stack = np.array(gpp_stack)
years = np.array(years)


lucc_path = r"D:\00---全球NBP\Modis_LUCC_10_2_2023.tif"
with rasterio.open(lucc_path) as src:
    lucc = src.read(1)
    
    if lucc.shape != (height, width):

        from rasterio.warp import reproject, Resampling
        from rasterio.transform import from_bounds
        
        dst_transform = from_bounds(left, bottom, right, top, width, height)
        lucc_resampled = np.empty((height, width), dtype=lucc.dtype)
        reproject(
            source=lucc,
            destination=lucc_resampled,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=dst_transform,
            dst_crs=crs,
            resampling=Resampling.nearest
        )
        lucc = lucc_resampled

shrub_grass_mask = np.isin(lucc, [6,7,9,10])


def calculate_sen_slope(data_stack):
    n_years, height, width = data_stack.shape
    slope = np.full((height, width), np.nan)
    time = np.arange(n_years)
    
    total_pixels = height * width
    processed = 0
    
    for i in range(height):
        for j in range(width):
            processed += 1
            if processed % 10000 == 0:
                print(f"  stage: {processed}/{total_pixels} ({processed/total_pixels*100:.1f}%)")
            
            pixel_data = data_stack[:, i, j]
            
            if np.all(np.isnan(pixel_data)):
                continue
            
            valid_mask = ~np.isnan(pixel_data)
            if np.sum(valid_mask) < 5:
                continue
            
            valid_time = time[valid_mask]
            valid_data = pixel_data[valid_mask]
            
            try:
                result = stats.theilslopes(valid_data, valid_time)
                slope[i, j] = result[0]
            except:
                continue
    
    return slope

gpp_slope = calculate_sen_slope(gpp_stack)


gpp_increasing = (gpp_slope > 0) & shrub_grass_mask & (~np.isnan(gpp_slope))
gpp_decreasing = (gpp_slope < 0) & shrub_grass_mask & (~np.isnan(gpp_slope))

mid_latitude_mask = (
    ((lat_grid >= 15) & (lat_grid <= 56)) | 
    ((lat_grid >= -60) & (lat_grid <= -15))
)

equatorial_mask = (lat_grid > -15) & (lat_grid < 15)


# 0: no SGL
# 11: EQ-GPP UP
# 12: EQ-GPP down
# 21: MID-GPP up
# 22: MID-GPP down
combined_classification = np.zeros((height, width), dtype=np.uint8)
combined_classification[(gpp_increasing) & (equatorial_mask)] = 11
combined_classification[(gpp_decreasing) & (equatorial_mask)] = 12
combined_classification[(gpp_increasing) & (mid_latitude_mask)] = 21
combined_classification[(gpp_decreasing) & (mid_latitude_mask)] = 22


output_combined = os.path.join(gpp_folder, f'GPP_Trend_Combined_Classification_{years.min()}_{years.max()}.tif')

meta_combined = meta.copy()
meta_combined.update({
    'dtype': 'uint8',
    'count': 1,
    'nodata': 0
})

with rasterio.open(output_combined, 'w', **meta_combined) as dst:
    dst.write(combined_classification, 1)

gpp_trend_classification = np.zeros((height, width), dtype=np.int16)
gpp_trend_classification[gpp_increasing] = 1
gpp_trend_classification[gpp_decreasing] = -1

output_simple = os.path.join(gpp_folder, f'GPP_Trend_Classification_{years.min()}_{years.max()}.tif')

meta_simple = meta.copy()
meta_simple.update({
    'dtype': 'int16',
    'count': 1,
    'nodata': 0
})

with rasterio.open(output_simple, 'w', **meta_simple) as dst:
    dst.write(gpp_trend_classification, 1)


output_slope = os.path.join(gpp_folder, f'GPP_Sen_Slope_{years.min()}_{years.max()}.tif')

meta_slope = meta.copy()
meta_slope.update({
    'dtype': 'float32',
    'count': 1,
    'nodata': -9999
})

gpp_slope_save = gpp_slope.copy()
gpp_slope_save[np.isnan(gpp_slope_save)] = -9999

with rasterio.open(output_slope, 'w', **meta_slope) as dst:
    dst.write(gpp_slope_save.astype(np.float32), 1)

print(f"✓ Sen-Slope: {output_slope}")

output_mask = os.path.join(gpp_folder, 'Shrub_Grass_Mask.tif')

meta_mask = meta.copy()
meta_mask.update({
    'dtype': 'uint8',
    'count': 1,
    'nodata': 0
})

with rasterio.open(output_mask, 'w', **meta_mask) as dst:
    dst.write(shrub_grass_mask.astype(np.uint8), 1)


output_lat_zones = os.path.join(gpp_folder, 'Latitude_Zones.tif')

lat_zones = np.zeros((height, width), dtype=np.uint8)
lat_zones[equatorial_mask] = 1
lat_zones[mid_latitude_mask] = 2

meta_zones = meta.copy()
meta_zones.update({
    'dtype': 'uint8',
    'count': 1,
    'nodata': 0
})

with rasterio.open(output_lat_zones, 'w', **meta_zones) as dst:
    dst.write(lat_zones, 1)


# ==================== 5. 统计 ====================
mid_shrub_total = np.sum(shrub_grass_mask & mid_latitude_mask & (~np.isnan(gpp_slope)))
mid_gpp_inc = np.sum(gpp_increasing & mid_latitude_mask)
mid_gpp_dec = np.sum(gpp_decreasing & mid_latitude_mask)

eq_shrub_total = np.sum(shrub_grass_mask & equatorial_mask & (~np.isnan(gpp_slope)))
eq_gpp_inc = np.sum(gpp_increasing & equatorial_mask)
eq_gpp_dec = np.sum(gpp_decreasing & equatorial_mask)


color_increasing = '#0173B2'  
color_decreasing = '#DE8F05'  

gpp_trend_map = np.full((height, width), np.nan)
gpp_trend_map[gpp_increasing] = 1
gpp_trend_map[gpp_decreasing] = -1

valid_data_count = np.sum(~np.isnan(gpp_trend_map))

if valid_data_count == 0:
    print("error：gpp_trend_map NaN")


fig1 = plt.figure(figsize=(16, 10), dpi=100, facecolor='white')


colors = [color_decreasing, color_increasing]
cmap = ListedColormap(colors)

cmap.set_bad(color='none', alpha=0)

bounds = [-1.5, -0.5, 1.5]
norm = BoundaryNorm(bounds, cmap.N)

ax1 = plt.axes(projection=ccrs.PlateCarree())
ax1.set_extent([-180, 180, -60, 85])

extent = [left, right, bottom, top]

coastline_width = 1.0
border_width = 0.8

ax1.add_feature(cfeature.COASTLINE, linewidth=coastline_width, color='#2F2F2F', zorder=11)

im1 = ax1.imshow(gpp_trend_map, extent=extent,
                 transform=ccrs.PlateCarree(), cmap=cmap, norm=norm,
                 interpolation='nearest', alpha=1.0, zorder=10)

ax1.plot([-180, 180], [15, 15], color='#7570B3', linestyle='--', linewidth=1.2, 
         transform=ccrs.PlateCarree(), alpha=0.7, zorder=12)
ax1.plot([-180, 180], [-15, -15], color='#7570B3', linestyle='--', linewidth=1.2, 
         transform=ccrs.PlateCarree(), alpha=0.7, zorder=12)
ax1.plot([-180, 180], [56, 56], color='#E7298A', linestyle='--', linewidth=1.2, 
         transform=ccrs.PlateCarree(), alpha=0.7, zorder=12)
ax1.plot([-180, 180], [-60, -60], color='#E7298A', linestyle='--', linewidth=1.2, 
         transform=ccrs.PlateCarree(), alpha=0.7, zorder=12)

legend_elements = [
    plt.Line2D([0], [0], color='#7570B3', linestyle='--', linewidth=1.2, label='±15° (Tropics boundary)'),
    plt.Line2D([0], [0], color='#E7298A', linestyle='--', linewidth=1.2, label='±56° (Mid-latitude boundary)')
]
ax1.legend(handles=legend_elements, loc='lower left', fontsize=11, 
          frameon=True, fancybox=False, edgecolor='black', framealpha=0.95)

ax1.set_title(f'GPP trends in global shrubland and grassland ecosystems ({years.min()}–{years.max()})', 
              fontsize=16, fontweight='bold', pad=25)

gl = ax1.gridlines(draw_labels=True, linewidth=0.8, color='#666666', 
                  alpha=0.6, linestyle=':', zorder=13)
gl.top_labels = False
gl.right_labels = False
gl.xlabel_style = {'size': 12, 'color': 'black', 'weight': 'normal'}
gl.ylabel_style = {'size': 12, 'color': 'black', 'weight': 'normal'}

cbar = plt.colorbar(im1, ax=ax1, orientation='horizontal', pad=0.08, 
                   fraction=0.06, shrink=0.8, aspect=30)
cbar.set_ticks([-1, 1])
cbar.set_ticklabels(['Decreasing trend', 'Increasing trend'])
cbar.ax.tick_params(labelsize=12)
cbar.outline.set_linewidth(1.5)
cbar.outline.set_edgecolor('black')

plt.tight_layout()

output_path = r'E:\000000000000--GCAS\GPP_Trend_Map_Nature'

dpi_high = 600
base_params = {
    'bbox_inches': 'tight',
    'facecolor': 'white',
    'edgecolor': 'none',
    'pad_inches': 0.2
}

try:
    pdf_params = base_params.copy()
    pdf_params.update({'format': 'pdf', 'dpi': dpi_high})
    fig1.savefig(f"{output_path}_high_quality.pdf", **pdf_params)
    print(f"✓ PDF (DPI: {dpi_high})")
except Exception as e:
    print(f"✗ PDF: {e}")


if mid_shrub_total > 0 and eq_shrub_total > 0:
    fig2, ax = plt.subplots(figsize=(10, 7), dpi=100, facecolor='white')

    categories = ['Mid-latitudes\n(15°–56°)', 'Tropics\n(±15°)']
    gpp_inc_pct = [mid_gpp_inc/mid_shrub_total*100, eq_gpp_inc/eq_shrub_total*100]
    gpp_dec_pct = [mid_gpp_dec/mid_shrub_total*100, eq_gpp_dec/eq_shrub_total*100]

    x = np.arange(len(categories))
    width = 0.35

    bars1 = ax.bar(x - width/2, gpp_inc_pct, width, label='Increasing trend', 
                   color=color_increasing, edgecolor='black', linewidth=1.5)
    bars2 = ax.bar(x + width/2, gpp_dec_pct, width, label='Decreasing trend', 
                   color=color_decreasing, edgecolor='black', linewidth=1.5)

    ax.set_ylabel('Percentage (%)', fontsize=13, fontweight='bold')
    ax.set_title(f'GPP trend statistics ({years.min()}–{years.max()})', 
                 fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=12)
    
    ax.legend(fontsize=12, framealpha=0.95, loc='upper right', 
             frameon=True, fancybox=False, edgecolor='black')
    
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(0, 100)

    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%', ha='center', va='bottom', 
                   fontsize=11, fontweight='bold')

    plt.tight_layout()
    
    output_path2 = r'E:\000000000000--GCAS\GPP_Trend_Statistics_Nature'
    
    try:
        pdf_params = base_params.copy()
        pdf_params.update({'format': 'pdf', 'dpi': dpi_high})
        fig2.savefig(f"{output_path2}_high_quality.pdf", **pdf_params)
        print(f"✓ PDF")
    except Exception as e:
        print(f"✗ PDF: {e}")

plt.close('all')

