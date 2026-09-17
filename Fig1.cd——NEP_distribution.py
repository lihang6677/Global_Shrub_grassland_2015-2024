
import os
import numpy as np
import pandas as pd
import rasterio
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm, Normalize
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['xtick.major.width'] = 0.8
plt.rcParams['ytick.major.width'] = 0.8
plt.rcParams['xtick.major.size'] = 4
plt.rcParams['ytick.major.size'] = 4
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'


EARTH_RADIUS = 6371000 
GC_TO_PGC = 1e-15  

REGION_DEFINITIONS = {
    'High_North': {'bounds': (60, 90), 'name': 'High North (≥60°N)', 'color': '#2166ac'},
    'Tropics': {'bounds': (-15, 15), 'name': 'Tropics (15°S-15°N)', 'color': '#d7191c'},
    'Mid_Latitudes': {'bounds': [(-60, -15), (15, 60)], 'name': 'Mid-latitudes (Others)', 'color': '#31a354'}
}

def create_nep_colormap():

    nep_colors = [
        '#8b0000',  # 深红 (强碳源)
        '#d73027',  # 红色
        '#f46d43',  # 橙红
        '#fdae61',  # 琥珀
        '#fee08b',  # 浅黄 (弱碳源)
        '#ffffff',  # 纯白 (零值)
        '#d9f0a3',  # 淡绿 (弱碳汇)
        '#abd9e9',  # 浅蓝
        '#74add1',  # 天蓝
        '#4575b4',  # 深蓝
        '#313695'   # 非常深蓝 (强碳汇)
    ]
    
    nep_cmap = LinearSegmentedColormap.from_list('nep_custom', nep_colors, N=256)
    nep_cmap.set_bad(color='none', alpha=0)
    
    return nep_cmap

class NEPVisualizerShrubland:
    
    def __init__(self, output_dir):

        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # 数据存储
        self.nep_mean = None
        self.mask = None
        self.pixel_areas = None
        self.transform = None
        self.crs = None
        self.bounds = None
        self.lat_grid = None
        self.lon_grid = None

    
    def load_nep_data(self, nep_file_paths):
        
        nep_data_list = []
        valid_files = []
        
        for file_path in nep_file_paths:
            if os.path.exists(file_path):
                try:
                    with rasterio.open(file_path) as src:
                        nep_data = src.read(1).astype(np.float32)
                        nep_data[nep_data == src.nodata] = np.nan
                        
                        if self.transform is None:
                            self.transform = src.transform
                            self.crs = src.crs
                            self.bounds = src.bounds
                        
                        nep_data_list.append(nep_data)
                        valid_files.append(os.path.basename(file_path))
                        
                        year = os.path.basename(file_path).split('_')[1]
                        
                except Exception as e:
                    print(f"  ✗  {file_path}: {e}")
            else:
                print(f"  ✗  {file_path}")
        
        if len(nep_data_list) == 0:
            raise ValueError("no NEP file")
        
        nep_stack = np.stack(nep_data_list, axis=0)
        self.nep_mean = np.nanmean(nep_stack, axis=0)

        print(f" {self.nep_mean.shape}")
        print(f"{np.nanmin(self.nep_mean):.2f} ~ {np.nanmax(self.nep_mean):.2f} gC/m²/year")

        self.create_coordinate_grids()
        
        self.create_pixel_area_grid()
    
    def create_coordinate_grids(self):

        height, width = self.nep_mean.shape
        
        lon_min, lat_max = self.transform * (0, 0)
        lon_max, lat_min = self.transform * (width, height)
        
        self.lon_grid = np.linspace(lon_min + 0.5, lon_max - 0.5, width)
        self.lat_grid = np.linspace(lat_max - 0.5, lat_min + 0.5, height)
        
    
    def create_pixel_area_grid(self):

        deg_to_m = np.pi * EARTH_RADIUS / 180.0
        
        lat_2d, lon_2d = np.meshgrid(self.lat_grid, self.lon_grid, indexing='ij')
        
        lat_rad = np.radians(lat_2d)
        pixel_height = deg_to_m
        pixel_width = deg_to_m * np.cos(lat_rad)
        
        self.pixel_areas = pixel_height * pixel_width

    
    def load_land_cover_mask(self, lucc_path, target_value=[6, 7, 9, 10]):
        
        with rasterio.open(lucc_path) as src:
            lucc_data = src.read(1)
        
        self.mask = np.isin(lucc_data, target_value)
        
        total_pixels = lucc_data.size
        shrubland_pixels = np.sum(self.mask)
        percentage = (shrubland_pixels / total_pixels) * 100

        if self.nep_mean is not None:
            nep_masked = self.nep_mean.copy()
            nep_masked[~self.mask] = np.nan
            
            valid_nep_pixels = np.sum(~np.isnan(nep_masked))

            self.nep_mean = nep_masked
    
    def get_region_mask(self, region_name):

        lat_2d, _ = np.meshgrid(self.lat_grid, self.lon_grid, indexing='ij')
        
        if region_name == 'High_North':
            bounds = REGION_DEFINITIONS[region_name]['bounds']
            return (lat_2d >= bounds[0]) & (lat_2d <= bounds[1])
        elif region_name == 'Tropics':
            bounds = REGION_DEFINITIONS[region_name]['bounds']
            return (lat_2d >= bounds[0]) & (lat_2d <= bounds[1])
        elif region_name == 'Mid_Latitudes':
            bounds = REGION_DEFINITIONS[region_name]['bounds']
            mask1 = (lat_2d >= bounds[0][0]) & (lat_2d <= bounds[0][1])
            mask2 = (lat_2d >= bounds[1][0]) & (lat_2d <= bounds[1][1])
            return mask1 | mask2
        else:
            raise ValueError(f"unknown area: {region_name}")
    
    def calculate_regional_statistics(self):

        if self.nep_mean is None:
            raise ValueError("load NEPdata")
        
        results = {}

        global_mask = ~np.isnan(self.nep_mean)
        if np.sum(global_mask) > 0:
            nep_values_global = self.nep_mean[global_mask]
            pixel_areas_global = self.pixel_areas[global_mask]
            
            total_nep_pgc = np.sum(nep_values_global * pixel_areas_global) * GC_TO_PGC
            mean_nep = np.mean(nep_values_global)
            median_nep = np.median(nep_values_global)
            std_nep = np.std(nep_values_global)
            
            results['Global_Shrubland'] = {
                'name': 'Global_SGL',
                'total_nep_PgC': total_nep_pgc,
                'mean_nep_gC_m2': mean_nep,
                'median_nep_gC_m2': median_nep,
                'std_nep_gC_m2': std_nep,
                'pixel_count': np.sum(global_mask),
                'total_area_km2': np.sum(pixel_areas_global) * 1e-6
            }
            
            print(f"  totalNEP: {total_nep_pgc:.3f} PgC/year")
            print(f"  meanNEP: {mean_nep:.2f} gC/m²/year")

        for region_key in REGION_DEFINITIONS.keys():
            region_mask = self.get_region_mask(region_key)
            combined_mask = region_mask & ~np.isnan(self.nep_mean)
            
            if np.sum(combined_mask) > 0:
                nep_values_regional = self.nep_mean[combined_mask]
                pixel_areas_regional = self.pixel_areas[combined_mask]
                
                total_nep_pgc = np.sum(nep_values_regional * pixel_areas_regional) * GC_TO_PGC
                mean_nep = np.mean(nep_values_regional)
                median_nep = np.median(nep_values_regional)
                std_nep = np.std(nep_values_regional)
                
                results[region_key] = {
                    'name': REGION_DEFINITIONS[region_key]['name'],
                    'total_nep_PgC': total_nep_pgc,
                    'mean_nep_gC_m2': mean_nep,
                    'median_nep_gC_m2': median_nep,
                    'std_nep_gC_m2': std_nep,
                    'pixel_count': np.sum(combined_mask),
                    'total_area_km2': np.sum(pixel_areas_regional) * 1e-6,
                    'color': REGION_DEFINITIONS[region_key]['color']
                }
                
                print(f"{REGION_DEFINITIONS[region_key]['name']}:")
                print(f"  totalNEP: {total_nep_pgc:.3f} PgC/year")
                print(f"  meanNEP: {mean_nep:.2f} gC/m²/year")
            else:
                results[region_key] = {
                    'name': REGION_DEFINITIONS[region_key]['name'],
                    'total_nep_PgC': 0.0,
                    'mean_nep_gC_m2': np.nan,
                    'median_nep_gC_m2': np.nan,
                    'std_nep_gC_m2': np.nan,
                    'pixel_count': 0,
                    'total_area_km2': 0.0,
                    'color': REGION_DEFINITIONS[region_key]['color']
                }
        
        return results
    
    def plot_nep_distribution(self, projection='robinson', vmin=None, vmax=None, 
                             symmetric=True, output_name=None):
        
        if self.nep_mean is None:
            raise ValueError("load NEPdata")
        
        valid_nep = self.nep_mean[~np.isnan(self.nep_mean)]
        
        if vmin is None or vmax is None:
            if len(valid_nep) > 0:

                p5 = np.percentile(valid_nep, 5)
                p95 = np.percentile(valid_nep, 95)
                
                if vmin is None:
                    vmin = p5
                if vmax is None:
                    vmax = p95
                
                if symmetric:
                    abs_max = max(abs(vmin), abs(vmax))
                    vmin = -abs_max
                    vmax = abs_max
            else:
                vmin, vmax = -200, 200

        projection_dict = {
            'mollweide': ccrs.Mollweide(),
            'robinson': ccrs.Robinson(), 
            'eckert4': ccrs.EckertIV(),
            'orthographic': ccrs.Orthographic(central_longitude=0, central_latitude=15),
            'plate_carree': ccrs.PlateCarree()
        }

        if projection in ['mollweide', 'robinson', 'eckert4']:
            figsize = (18, 12)
        elif projection == 'orthographic':
            figsize = (14, 14)
        else:
            figsize = (16, 10)

        fig = plt.figure(figsize=figsize, dpi=100, facecolor='white')
        
        if projection in projection_dict:
            ax = plt.axes(projection=projection_dict[projection])
        else:
            ax = plt.axes(projection=ccrs.Robinson())

        ax.set_extent([-180, 180, -60,85], crs=ccrs.PlateCarree())
        

        ax.add_feature(cfeature.LAND, facecolor='#F5F5F5', alpha=1.0, zorder=2)
        ax.add_feature(cfeature.COASTLINE, linewidth=1.2, color='#2F2F2F', zorder=6)

        nep_cmap = create_nep_colormap()
        norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
        
        extent = [-180, 180, self.lat_grid[-1], self.lat_grid[0]]
        im = ax.imshow(self.nep_mean, extent=extent, transform=ccrs.PlateCarree(),
                       cmap=nep_cmap, norm=norm, alpha=0.9, zorder=4)
        
        lon_range = np.linspace(-180, 180, 100)
        
        ax.plot(lon_range, np.full_like(lon_range, -15), color='white', 
                linestyle='--', linewidth=2.5, alpha=0.8, transform=ccrs.PlateCarree(), zorder=7)
        ax.plot(lon_range, np.full_like(lon_range, 15), color='white', 
                linestyle='--', linewidth=2.5, alpha=0.8, transform=ccrs.PlateCarree(), zorder=7)
        ax.plot(lon_range, np.full_like(lon_range, 60), color='white', 
                linestyle='--', linewidth=2.5, alpha=0.8, transform=ccrs.PlateCarree(), zorder=7)

        if projection not in ['orthographic']:
            gl = ax.gridlines(draw_labels=True, linewidth=1.0, color='#666666', 
                              alpha=0.6, linestyle=':', zorder=7)
            gl.top_labels = False
            gl.right_labels = False
            gl.xlabel_style = {'size': 12, 'color': 'black', 'weight': 'normal'}
            gl.ylabel_style = {'size': 12, 'color': 'black', 'weight': 'normal'}
        else:
            ax.gridlines(linewidth=1.0, color='#666666', alpha=0.5, linestyle=':', zorder=7)

        if projection == 'orthographic':
            cbar = plt.colorbar(im, ax=ax, orientation='vertical', pad=0.1, 
                               fraction=0.05, shrink=0.7, aspect=25)
        else:
            cbar = plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.08, 
                               fraction=0.06, shrink=0.8, aspect=35)

        label_text = ('Net Ecosystem Productivity (NEP) in Shrublands (gC/m²/year)\n'
                     'Red: Carbon source (land to atmosphere), Blue: Carbon sink (atmosphere to land)\n'
                     'White dashed lines: Latitudinal zone boundaries')
        
        cbar.set_label(label_text, fontsize=14, fontweight='normal')
        cbar.ax.tick_params(labelsize=12)
        cbar.outline.set_linewidth(1.5)
        cbar.outline.set_edgecolor('black')
        
        if projection == 'orthographic':
            cbar.ax.axhline(y=0, color='black', linewidth=2.5, alpha=0.9)
        else:
            cbar.ax.axvline(x=0, color='black', linewidth=2.5, alpha=0.9)
        
        regional_stats = self.calculate_regional_statistics()
        
        stats_text = (f"Global Shrubland NEP: {regional_stats['Global_Shrubland']['total_nep_PgC']:.2f} PgC/year\n"
                     f"Mean: {regional_stats['Global_Shrubland']['mean_nep_gC_m2']:.1f} gC/m²/year")
        
        if projection == 'orthographic':
            cbar.ax.text(3.0, 0.5, stats_text, rotation=90,
                        transform=cbar.ax.transAxes, ha='left', va='center', fontsize=11,
                        bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.9))
        else:
            cbar.ax.text(0.5, -3.5, stats_text, 
                        transform=cbar.ax.transAxes, ha='center', va='top', fontsize=11,
                        bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.9))
        
        title_text = 'Global Shrubland Net Ecosystem Productivity (NEP)\nMean Annual Distribution (2015-2024)'
        ax.set_title(title_text, fontsize=18, fontweight='bold', pad=30)
        
        legend_text = (f'Data: Mean annual NEP (2015-2024)\n'
                      f'Focus: Global shrubland ecosystems\n'
                      f'Latitudinal zones:\n'
                      f'• High North (≥60°N): {regional_stats["High_North"]["total_nep_PgC"]:.2f} PgC/year\n'
                      f'• Tropics (15°S-15°N): {regional_stats["Tropics"]["total_nep_PgC"]:.2f} PgC/year\n'
                      f'• Mid-latitudes (Others): {regional_stats["Mid_Latitudes"]["total_nep_PgC"]:.2f} PgC/year\n'
                      f'Total area: {regional_stats["Global_Shrubland"]["total_area_km2"]:.0f} thousand km²')
        
        legend_pos = (0.02, 0.02) if projection != 'orthographic' else (1.15, 0.02)
        
        ax.text(legend_pos[0], legend_pos[1], legend_text, transform=ax.transAxes, 
                fontsize=12, verticalalignment='bottom', horizontalalignment='left',
                bbox=dict(boxstyle='round,pad=0.6', facecolor='white', alpha=0.95, 
                         edgecolor='black', linewidth=1.5),
                zorder=10)
        
        plt.tight_layout()
        
        if output_name is None:
            output_name = f"Global_Shrubland_NEP_Distribution_2015_2024_{projection}"
        
        output_path = os.path.join(self.output_dir, output_name)
        
        save_params = {
            'bbox_inches': 'tight',
            'facecolor': 'white',
            'edgecolor': 'none',
            'pad_inches': 0.3
        }
        
        try:
            fig.savefig(f"{output_path}_nature_quality.png", dpi=600, format='png', **save_params)
            
            fig.savefig(f"{output_path}_high_quality.jpg", dpi=600, format='jpg', **save_params)
            
            fig.savefig(f"{output_path}_vector.pdf", format='pdf', **save_params)
            
            fig.savefig(f"{output_path}_vector.svg", format='svg', **save_params)
            
        except Exception as e:
            print(f"fail: {e}")
        
        plt.close(fig)
        
        self.save_regional_statistics(regional_stats, output_name)
        
        return regional_stats
    
    def save_regional_statistics(self, regional_stats, output_name):
        
        data_rows = []
        for region_key, stats in regional_stats.items():
            data_rows.append({
                'Region': stats['name'],
                'Region_Code': region_key,
                'Total_NEP_PgC_year': stats['total_nep_PgC'],
                'Mean_NEP_gC_m2_year': stats['mean_nep_gC_m2'],
                'Median_NEP_gC_m2_year': stats['median_nep_gC_m2'],
                'Std_NEP_gC_m2_year': stats['std_nep_gC_m2'],
                'Pixel_Count': stats['pixel_count'],
                'Total_Area_thousand_km2': stats['total_area_km2'] / 1000
            })
        
        df = pd.DataFrame(data_rows)
        csv_path = os.path.join(self.output_dir, f"{output_name}_regional_statistics.csv")
        df.to_csv(csv_path, index=False, float_format='%.4f')
        
    
    def plot_latitudinal_profile(self, output_name=None):

        if self.nep_mean is None:
            raise ValueError("loadNEP")

        lat_means = []
        lat_stds = []
        lat_counts = []
        valid_lats = []
        
        for i, lat in enumerate(self.lat_grid):
            row_nep = self.nep_mean[i, :]
            valid_nep = row_nep[~np.isnan(row_nep)]
            
            if len(valid_nep) > 0:
                lat_means.append(np.mean(valid_nep))
                lat_stds.append(np.std(valid_nep))
                lat_counts.append(len(valid_nep))
                valid_lats.append(lat)
        
        valid_lats = np.array(valid_lats)
        lat_means = np.array(lat_means)
        lat_stds = np.array(lat_stds)
        lat_counts = np.array(lat_counts)

        fig, ax = plt.subplots(figsize=(10, 12))

        ax.plot(lat_means, valid_lats, '-', color='#2c7fb8', linewidth=3, zorder=5, label='Mean NEP')

        ax.fill_betweenx(valid_lats, lat_means - lat_stds, lat_means + lat_stds, 
                        alpha=0.3, color='#2c7fb8', zorder=2, label='±1 Std Dev')

        regional_means = {}
        for region_key, region_def in REGION_DEFINITIONS.items():
            if region_key == 'Mid_Latitudes':
                bounds_list = region_def['bounds']
                mask = np.zeros_like(valid_lats, dtype=bool)
                for bounds in bounds_list:
                    mask_part = (valid_lats >= bounds[0]) & (valid_lats <= bounds[1])
                    mask = mask | mask_part
            else:
                bounds = region_def['bounds']
                mask = (valid_lats >= bounds[0]) & (valid_lats <= bounds[1])
            
            if np.sum(mask) > 0:
                regional_mean = np.average(lat_means[mask], weights=lat_counts[mask])
                regional_means[region_key] = regional_mean
            else:
                regional_means[region_key] = np.nan

        for region_key, mean_value in regional_means.items():
            if not np.isnan(mean_value):
                region_def = REGION_DEFINITIONS[region_key]
                color = region_def['color']
                
                if region_key == 'High_North':
                    y_region = np.array([60, 80])
                    ax.plot([mean_value, mean_value], y_region, color=color, 
                           linestyle='-', linewidth=4, alpha=0.8, zorder=4)
                    ax.text(mean_value, 82, f'{mean_value:.1f}', ha='center', va='bottom', 
                           fontsize=12, color=color, fontweight='bold')
                elif region_key == 'Tropics':
                    y_region = np.array([-15, 15])
                    ax.plot([mean_value, mean_value], y_region, color=color, 
                           linestyle='-', linewidth=4, alpha=0.8, zorder=4)
                    ax.text(mean_value, 17, f'{mean_value:.1f}', ha='center', va='bottom', 
                           fontsize=12, color=color, fontweight='bold')
                elif region_key == 'Mid_Latitudes':
                
                    y_mid_s = np.array([-60, -15])
                    ax.plot([mean_value, mean_value], y_mid_s, color=color, 
                           linestyle='-', linewidth=4, alpha=0.8, zorder=4)
             
                    y_mid_n = np.array([15, 60])
                    ax.plot([mean_value, mean_value], y_mid_n, color=color, 
                           linestyle='-', linewidth=4, alpha=0.8, zorder=4)
                    ax.text(mean_value, -62, f'{mean_value:.1f}', ha='center', va='top', 
                           fontsize=12, color=color, fontweight='bold')

        ax.set_ylabel('Latitude (°N)', fontsize=14, fontweight='normal')
        ax.set_xlabel('NEP (gC/m²/year)', fontsize=14, fontweight='normal')
        ax.set_ylim(-60, 80)

        ax.axvline(x=0, color='black', linestyle='-', linewidth=2, alpha=0.8, zorder=3)

        ax.axhline(y=-15, color='gray', linestyle='--', linewidth=1.5, alpha=0.7, zorder=2)
        ax.axhline(y=15, color='gray', linestyle='--', linewidth=1.5, alpha=0.7, zorder=2)
        ax.axhline(y=60, color='gray', linestyle='--', linewidth=1.5, alpha=0.7, zorder=2)
        
        ax.set_yticks([-60, -30, -15, 0, 15, 30, 60])
        ax.set_yticklabels(['60°S', '30°S', '15°S', '0°', '15°N', '30°N', '60°N'])
        
        ax.grid(True, axis='x', alpha=0.3, linestyle='-', linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        
        ax.legend(loc='upper right', fontsize=12, framealpha=0.9)
        
        ax.set_title('Latitudinal Distribution of NEP in Global Shrublands\n(2015-2024 Mean)', 
                    fontsize=16, fontweight='bold', pad=20)
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(1.5)
        ax.spines['bottom'].set_linewidth(1.5)
        
        plt.tight_layout()
        
        if output_name is None:
            output_name = "Global_Shrubland_NEP_Latitudinal_Profile_2015_2024"
        
        profile_path = os.path.join(self.output_dir, output_name)
        
        save_params = {
            'bbox_inches': 'tight',
            'facecolor': 'white',
            'edgecolor': 'none',
            'pad_inches': 0.2
        }
        
        fig.savefig(f"{profile_path}.png", dpi=600, format='png', **save_params)
        fig.savefig(f"{profile_path}.pdf", format='pdf', **save_params)
        fig.savefig(f"{profile_path}.jpg", dpi=600, format='jpg', **save_params)
        
        plt.close()
        
        profile_data = pd.DataFrame({
            'Latitude': valid_lats,
            'Mean_NEP_gC_m2_year': lat_means,
            'Std_NEP_gC_m2_year': lat_stds,
            'Pixel_Count': lat_counts.astype(int)
        })
        
        profile_csv = os.path.join(self.output_dir, f"{output_name}_data.csv")
        profile_data.to_csv(profile_csv, index=False, float_format='%.4f')


def main():

    nep_data_dir = r"E:/000000------文章1/数据库/GCB2025/models—indivi"
    lucc_path = r"D:\00---全球NBP\Modis_LUCC_10_2_2023.tif"
    output_dir = r"E:\000000------文章1\figuresOUTPUT"
    
    nep_files = [
        os.path.join(nep_data_dir, "GCB_NEP_2015.tif"),
        os.path.join(nep_data_dir, "GCB_NEP_2016.tif"),
        os.path.join(nep_data_dir, "GCB_NEP_2017.tif"),
        os.path.join(nep_data_dir, "GCB_NEP_2018.tif"),
        os.path.join(nep_data_dir, "GCB_NEP_2019.tif"),
        os.path.join(nep_data_dir, "GCB_NEP_2020.tif"),
        os.path.join(nep_data_dir, "GCB_NEP_2021.tif"),
        os.path.join(nep_data_dir, "GCB_NEP_2022.tif"),
        os.path.join(nep_data_dir, "GCB_NEP_2023.tif"),
        os.path.join(nep_data_dir, "GCB_NEP_2024.tif")
    ]
    
    try:
        visualizer = NEPVisualizerShrubland(output_dir)
        
        visualizer.load_nep_data(nep_files)

        visualizer.load_land_cover_mask(lucc_path, target_value=[6, 7, 9, 10])

        
        regional_stats = visualizer.plot_nep_distribution(
            projection='plate_carree',
            vmin=-150,  
            vmax=150,
            symmetric=True,
            output_name="Global_Shrubland_NEP_Distribution_2015_2024"
        )
        
        
        visualizer.plot_latitudinal_profile(
            output_name="Global_Shrubland_NEP_Latitudinal_Profile_2015_2024"
        )
        
        
    except Exception as e:
        print(f"\error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()