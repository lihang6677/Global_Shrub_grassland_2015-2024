
import os
import numpy as np
import pandas as pd
import rasterio
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm, Normalize, ListedColormap
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
M2_TO_KM2 = 1e-6  
GC_TO_PGC = 1e-15 
DAYS_PER_YEAR = 365.25 

REGION_DEFINITIONS = {
    'High_North': {'bounds': (56, 90), 'name': 'High North (≥56°N)', 'color': '#2166ac'},
    'Tropics': {'bounds': (-15, 15), 'name': 'Tropics (15°S-15°N)', 'color': '#d7191c'},
    'Mid_Latitudes': {'bounds': [(-60, -15), (15, 56)], 'name': 'Mid-latitudes (Others)', 'color': '#31a354'}
}

DATA_TYPE_CONFIG = {
    'NEP': {
        'unit_per_area': 'gC/m²/yr',
        'unit_total': 'PgC/yr',
        'conversion_factor': 1e-15,
        'color_scheme': 'RdBu',
        'vmin_default': -40,
        'vmax_default': 40,
        'positive_meaning': 'Carbon source (land to atmosphere)',
        'center_zero': True
    },
    'TER': {
        'unit_per_area': 'gC/m²/yr',
        'unit_total': 'PgC/yr',
        'conversion_factor': 1e-15,
        'color_scheme': 'custom_ter',
        'vmin_default': -100,
        'vmax_default': 100,
        'positive_meaning': 'Higher ecosystem respiration',
        'center_zero': True
    },
    'GPP': {
        'unit_per_area': 'gC/m²/yr', 
        'unit_total': 'PgC/yr',
        'conversion_factor': 1e-15,
        'color_scheme': 'custom_gpp',
        'vmin_default': -100,
        'vmax_default': 100,
        'positive_meaning': 'Higher gross primary productivity',
        'center_zero': True
    },
}

def create_custom_colormaps():

    from matplotlib.colors import LinearSegmentedColormap
    
    custom_cmaps = {}
    
    nep_colors = [
        '#d73027',  # 深红 (强碳源)
        '#f46d43',  # 橙红
        '#fdae61',  # 琥珀
        '#fee090',  # 浅黄 (负值近零)
        '#ffffff',  # 纯白 (零值)
        '#e0f3f8',  # 淡青 (正值近零)
        '#abd9e9',  # 浅蓝
        '#74add1',  # 天蓝
        '#4565b4'   # 深蓝 (强碳汇)
    ]
    custom_cmaps['custom_nep'] = LinearSegmentedColormap.from_list(
        'custom_nep', nep_colors, N=256)
    
    gpp_colors = [
        '#a50026',  # 非常深红 (显著减少)
        '#d73027',  # 深红
        '#f46d43',  # 橙红
        '#fdae61',  # 琥珀
        '#ffffff',  # 白色 (接近0)
        '#a6bddb',  # 浅钢蓝 (正值接近0)
        '#74add1',  # 天蓝
        '#4565b4',  # 深蓝
        '#313695'   # 非常深蓝 (显著增加)
    ]
    custom_cmaps['custom_gpp'] = LinearSegmentedColormap.from_list(
        'custom_gpp', gpp_colors, N=256)
    
    ter_colors = [
        '#313695',  # 非常深蓝 (呼吸显著减少)
        '#4565b4',  # 深蓝
        '#74add1',  # 天蓝
        '#a6bddb',  # 浅蓝
        '#ffffff',  # 白色（接近0变化）
        '#fdae61',  # 琥珀
        '#f46d43',  # 橙红
        '#d73027',  # 深红
        '#a50026'   # 非常深红 (显著增加)
    ]
    custom_cmaps['custom_ter'] = LinearSegmentedColormap.from_list(
        'custom_ter', ter_colors, N=256)
        
    return custom_cmaps

def get_colormap_and_norm(data_type, vmin, vmax, data_config):
    
    from matplotlib.colors import TwoSlopeNorm, Normalize
    
    custom_cmaps = create_custom_colormaps()
    
    cmap_name = data_config['color_scheme']
    
    if cmap_name in custom_cmaps:
        cmap = custom_cmaps[cmap_name]
    else:
        cmap = plt.get_cmap(cmap_name)
    
    if data_config.get('center_zero', False) and vmin < 0 < vmax:
        
        norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    else:
        
        norm = Normalize(vmin=vmin, vmax=vmax)
    
   
    cmap_with_alpha = cmap.copy()
    cmap_with_alpha.set_bad(color='none', alpha=0)
    
    return cmap_with_alpha, norm

class TrendVisualizer:

    
    def __init__(self, data_type, data_name, resolution, output_dir, mk_results_dir):

        if data_type not in DATA_TYPE_CONFIG:
            raise ValueError(f"wrong: {data_type}. need: {list(DATA_TYPE_CONFIG.keys())}")
        
        self.data_type = data_type
        self.data_config = DATA_TYPE_CONFIG[data_type]
        self.data_name = data_name
        self.resolution = resolution
        self.output_dir = output_dir
        self.mk_results_dir = mk_results_dir

        os.makedirs(output_dir, exist_ok=True)
        
        self.sen_slope = None
        self.mk_p_value = None
        self.mk_z_value = None
        self.tau_value = None
        self.s_value = None
        self.mask = None
        self.pixel_areas = None

        
    def load_mk_results(self):

        slope_file = os.path.join(self.mk_results_dir, self.data_type, f"{self.data_type}_Sen_Slope.tif")
        p_value_file = os.path.join(self.mk_results_dir, self.data_type, f"{self.data_type}_P_value.tif")
        tau_file = os.path.join(self.mk_results_dir, self.data_type, f"{self.data_type}_Tau.tif")
        z_file = os.path.join(self.mk_results_dir, self.data_type, f"{self.data_type}_Z_value.tif")
        s_file = os.path.join(self.mk_results_dir, self.data_type, f"{self.data_type}_S_statistic.tif")
        
        required_files = [slope_file, p_value_file]
        missing_files = [f for f in required_files if not os.path.exists(f)]
        
        if missing_files:
            raise FileNotFoundError(f"file_need: {missing_files}")

        with rasterio.open(slope_file) as src:
            self.sen_slope = src.read(1).astype(np.float32)
            self.sen_slope[self.sen_slope == src.nodata] = np.nan
            self.transform = src.transform
            self.crs = src.crs
            self.bounds = src.bounds
            
        with rasterio.open(p_value_file) as src:
            self.mk_p_value = src.read(1).astype(np.float32)
            self.mk_p_value[self.mk_p_value == src.nodata] = np.nan
        
        optional_files = [tau_file, z_file, s_file]
        for file_path, attr_name in zip(optional_files, ['tau_value', 'mk_z_value', 's_value']):
            if os.path.exists(file_path):
                with rasterio.open(file_path) as src:
                    data = src.read(1).astype(np.float32)
                    data[data == src.nodata] = np.nan
                    setattr(self, attr_name, data)
            else:
                print(f"警告: 未找到文件 {file_path}")
                setattr(self, attr_name, None)

    def calculate_pixel_area(self, lat):
        lat_rad = np.radians(lat)

        lon_distance = 2 * np.pi * EARTH_RADIUS * np.cos(lat_rad) * self.resolution / 360
        
        lat_distance = 2 * np.pi * EARTH_RADIUS * self.resolution / 360
        
        return lon_distance * lat_distance
    
    def create_pixel_area_grid(self):
        if self.sen_slope is None:
            raise ValueError("load_mk_results")
            
        n_lats, n_lons = self.sen_slope.shape
        latitudes = np.linspace(90 - self.resolution/2, -90 + self.resolution/2, n_lats)
        
        # 创建面积网格
        self.pixel_areas = np.zeros((n_lats, n_lons))
        for i, lat in enumerate(latitudes):
            area = self.calculate_pixel_area(lat)
            self.pixel_areas[i, :] = area
            
    
    def load_land_cover_mask(self, lucc_path, target_value=[6, 7, 9, 10]):
        with rasterio.open(lucc_path) as src:
            lucc_data = src.read(1)
            
            if isinstance(target_value, (list, tuple, np.ndarray)):
                self.mask = np.isin(lucc_data, target_value)
            else:
                self.mask = (lucc_data == target_value)
            
            total_pixels = lucc_data.size
            target_pixels = np.sum(self.mask)
            percentage = (target_pixels / total_pixels) * 100
        
        self.create_pixel_area_grid()
    
    def create_regional_mask(self, region_key):
        n_lats = self.mask.shape[0] if self.mask is not None else self.sen_slope.shape[0]
        latitudes = np.linspace(90 - self.resolution/2, -90 + self.resolution/2, n_lats)

        lat_grid = np.repeat(latitudes[:, np.newaxis], 
                           self.mask.shape[1] if self.mask is not None else self.sen_slope.shape[1], 
                           axis=1)
        
        region_def = REGION_DEFINITIONS[region_key]
        
        if region_key == 'Mid_Latitudes':
            bounds_list = region_def['bounds']
            regional_mask = np.zeros_like(lat_grid, dtype=bool)
            for bounds in bounds_list:
                mask_part = (lat_grid >= bounds[0]) & (lat_grid <= bounds[1])
                regional_mask = regional_mask | mask_part
        else:
            bounds = region_def['bounds']
            regional_mask = (lat_grid >= bounds[0]) & (lat_grid <= bounds[1])

        if self.mask is not None:
            regional_mask = regional_mask & self.mask
            
        return regional_mask
    
    def plot_trend_with_significance(self, output_name=None, vmin=None, vmax=None, 
                                   percentile_range=None, symmetric=True, 
                                   projection='robinson', lat_limit=-60, 
                                   significance_threshold=0.1, marker_type='dots',
                                   corel_friendly=True):
        
        if self.sen_slope is None or self.mk_p_value is None:
            raise ValueError("请先加载MK检验结果")
        
        n_lats, n_lons = self.sen_slope.shape
        latitudes = np.linspace(90 - self.resolution/2, -90 + self.resolution/2, n_lats)
        
        lat_mask = latitudes >= lat_limit
        if np.any(lat_mask):
            lat_end_idx = np.where(lat_mask)[0][-1]
            trend_clipped = self.sen_slope[:lat_end_idx+1, :]
            p_value_clipped = self.mk_p_value[:lat_end_idx+1, :]
            clipped_latitudes = latitudes[:lat_end_idx+1]
            
            if self.mask is not None:
                mask_clipped = self.mask[:lat_end_idx+1, :]
                trend_clipped[~mask_clipped] = np.nan
                p_value_clipped[~mask_clipped] = np.nan
        else:
            trend_clipped = self.sen_slope.copy()
            p_value_clipped = self.mk_p_value.copy()
            clipped_latitudes = latitudes
            if self.mask is not None:
                trend_clipped[~self.mask] = np.nan
                p_value_clipped[~self.mask] = np.nan
        
        projection_dict = {
            'mollweide': ccrs.Mollweide(),
            'robinson': ccrs.Robinson(), 
            'eckert4': ccrs.EckertIV(),
            'orthographic': ccrs.Orthographic(central_longitude=0, central_latitude=15),
            'plate_carree': ccrs.PlateCarree(),
            'miller': ccrs.Miller(),
            'mercator': ccrs.Mercator()
        }
        
        if projection in ['mollweide', 'robinson', 'eckert4', 'Mollweide']:
            figsize = (16, 10)
        elif projection == 'orthographic':
            figsize = (12, 12)
        else:
            figsize = (14, 8)
        
        fig = plt.figure(figsize=figsize, dpi=100, facecolor='white')
        
        if projection in projection_dict:
            ax = plt.axes(projection=projection_dict[projection])
        else:
            ax = plt.axes(projection=ccrs.Robinson())
        
        if projection == 'orthographic':
            ax.set_global()
        else:
            ax.set_extent([-180, 180, lat_limit, 85], crs=ccrs.PlateCarree())
        
        coastline_width = 1.0
        border_width = 0.8
        
        ax.add_feature(cfeature.LAND, facecolor='#E6F3FF', alpha=1.0, zorder=2)
        ax.add_feature(cfeature.COASTLINE, linewidth=coastline_width, color='#2F2F2F', zorder=6)

        valid_trend = trend_clipped[~np.isnan(trend_clipped)]

        if vmin is None or vmax is None:
            if len(valid_trend) > 0:
                if percentile_range is not None:
                    auto_vmin = np.percentile(valid_trend, percentile_range[0])
                    auto_vmax = np.percentile(valid_trend, percentile_range[1])
                else:
                    if self.data_type == 'NEP':
                        default_vmin, default_vmax = -10, 10 
                    elif self.data_type in ['TER', 'GPP']:
                        default_vmin, default_vmax = -50, 50 
                    else:
                        default_vmin = np.nanmin(valid_trend)
                        default_vmax = np.nanmax(valid_trend)
                    
                    data_min = np.nanmin(valid_trend)
                    data_max = np.nanmax(valid_trend)
                    
                    auto_vmin = min(default_vmin, data_min)
                    auto_vmax = max(default_vmax, data_max)
                
                if symmetric and self.data_config.get('center_zero', False):
                    abs_max = max(abs(auto_vmin), abs(auto_vmax))
                    auto_vmin = -abs_max
                    auto_vmax = abs_max
                
                if vmin is None:
                    vmin = auto_vmin
                if vmax is None:
                    vmax = auto_vmax
            else:
                vmin = self.data_config['vmin_default'] if vmin is None else vmin
                vmax = self.data_config['vmax_default'] if vmax is None else vmax
        
        cmap, norm = get_colormap_and_norm(self.data_type, vmin, vmax, self.data_config)
        
        extent = [-180, 180, clipped_latitudes[-1], clipped_latitudes[0]]
        im = ax.imshow(trend_clipped, extent=extent, transform=ccrs.PlateCarree(),
                       cmap=cmap, norm=norm, alpha=0.9, zorder=4)

        significant_mask = (p_value_clipped < significance_threshold) & ~np.isnan(p_value_clipped)

        total_valid = np.sum(~np.isnan(trend_clipped) & ~np.isnan(p_value_clipped))
        significant_count = np.sum(significant_mask)
        significant_percentage = (significant_count / total_valid) * 100 if total_valid > 0 else 0


        if np.sum(significant_mask) > 0:
            sig_indices = np.where(significant_mask)
            if len(sig_indices[0]) > 0:
                lat_centers = clipped_latitudes[:-1] + np.diff(clipped_latitudes) / 2
                lon_centers = np.linspace(-179.5, 179.5, trend_clipped.shape[1])

                sig_lats = lat_centers[sig_indices[0]]
                sig_lons = lon_centers[sig_indices[1]]

                if len(sig_lats) > 15000:
                    sample_indices = np.random.choice(len(sig_lats), 15000, replace=False)
                    sig_lats = sig_lats[sample_indices]
                    sig_lons = sig_lons[sample_indices]

                if marker_type == 'dots':
                    ax.scatter(sig_lons, sig_lats, s=0.1, c='black', marker='.',
                              alpha=0.7, transform=ccrs.PlateCarree(), zorder=5)
                elif marker_type == 'x':
                    if len(sig_lats) > 5000:
                        sample_indices = np.random.choice(len(sig_lats), 5000, replace=False)
                        sig_lats = sig_lats[sample_indices]
                        sig_lons = sig_lons[sample_indices]
                    
                    ax.scatter(sig_lons, sig_lats, s=1.0, c='black', marker='x',
                              alpha=0.8, transform=ccrs.PlateCarree(), zorder=5)

        if projection not in ['orthographic']:
            gl = ax.gridlines(draw_labels=True, linewidth=0.8, color='#666666', 
                              alpha=0.6, linestyle=':', zorder=7)
            gl.top_labels = False
            gl.right_labels = False
            gl.xlabel_style = {'size': 12, 'color': 'black', 'weight': 'normal'}
            gl.ylabel_style = {'size': 12, 'color': 'black', 'weight': 'normal'}
            
            if projection in ['mollweide', 'eckert4']:
                gl.left_labels = False
                gl.bottom_labels = False
        else:
            ax.gridlines(linewidth=0.8, color='#666666', alpha=0.5, linestyle=':', zorder=7)

        if projection == 'orthographic':
            cbar = plt.colorbar(im, ax=ax, orientation='vertical', pad=0.1, 
                               fraction=0.05, shrink=0.6, aspect=20)
        else:
            cbar = plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.08, 
                               fraction=0.06, shrink=0.8, aspect=30)
        
        meaning_text = self.data_config['positive_meaning']
        unit_text = self.data_config['unit_per_area']
        
        if self.data_config.get('center_zero', False):
            label_text = f'{self.data_type} Sen-Slope Trend ({unit_text})\n'
            if self.data_type == 'Temperature':
                label_text += 'Blue: Cooling, Red: Warming'
            elif self.data_type == 'TER':
                label_text += 'Blue: Respiration decrease, Red: Respiration increase'
            elif self.data_type == 'GPP':
                label_text += 'Red: Productivity decrease, Blue: Productivity increase'
            elif self.data_type == 'NEP':
                label_text += 'Red: Carbon source, Blue: Carbon sink'
            elif self.data_type in ['Ra', 'Rh']:
                label_text += f'Blue: {self.data_type} decrease, Red: {self.data_type} increase'
            else:
                label_text += f'White: No change, Colors: {meaning_text}'

            label_text += f'\nBlack markers: p < {significance_threshold} ({significant_percentage:.1f}%)'
        else:
            label_text = f'{self.data_type} Sen-Slope Trend ({unit_text})\n'
            label_text += f'Black markers: p < {significance_threshold} ({significant_percentage:.1f}%)'
        
        cbar.set_label(label_text, fontsize=14, fontweight='normal')
        cbar.ax.tick_params(labelsize=12)
        cbar.outline.set_linewidth(1.5)
        cbar.outline.set_edgecolor('black')

        if self.data_config.get('center_zero', False) and vmin < 0 < vmax:
            if projection == 'orthographic':
                cbar.ax.axhline(y=0, color='black', linewidth=2.0, alpha=0.8)
            else:
                cbar.ax.axvline(x=0, color='black', linewidth=2.0, alpha=0.8)

        mean_trend = np.nanmean(valid_trend)
        median_trend = np.nanmedian(valid_trend)
        std_trend = np.nanstd(valid_trend)
        
        stats_text = f'Mean: {mean_trend:.4f}, Median: {median_trend:.4f}, Std: {std_trend:.4f}'
        
        if projection == 'orthographic':
            cbar.ax.text(2.5, 0.5, stats_text, rotation=90,
                        transform=cbar.ax.transAxes, ha='left', va='center', fontsize=11,
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        else:
            cbar.ax.text(0.5, -2.8, stats_text, 
                        transform=cbar.ax.transAxes, ha='center', va='top', fontsize=11,
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

        title_text = f'{self.data_name} {self.data_type} Sen-Slope Trend with Significance (2015-2024)'
        ax.set_title(title_text, fontsize=16, fontweight='bold', pad=25)

        projection_names = {
            'mollweide': 'Mollweide',
            'robinson': 'Robinson', 
            'eckert4': 'Eckert IV',
            'orthographic': 'Orthographic',
            'plate_carree': 'Plate Carrée',
            'miller': 'Miller',
            'mercator': 'Mercator'
        }
        proj_name = projection_names.get(projection, projection.title())
        
        legend_pos = (0.02, 0.02) if projection != 'orthographic' else (1.15, 0.02)
        
        color_meaning = {
            'Temperature': 'Blue tones: Temperature decrease\nRed tones: Temperature increase',
            'TER': 'Blue tones: Respiration decrease\nRed tones: Respiration increase',
            'GPP': 'Red tones: Productivity decrease\nBlue tones: Productivity increase',
            'NEP': 'Red tones: Carbon source\nBlue tones: Carbon sink',
            'Ra': 'Blue tones: Ra decrease\nRed tones: Ra increase',
            'Rh': 'Blue tones: Rh decrease\nRed tones: Rh increase'
        }
        
        legend_text = (f'Projection: {proj_name}\n'
                      f'Latitude range: {lat_limit}°S to 85°N\n'
                      f'Color interpretation:\n'
                      f'{color_meaning.get(self.data_type, "Standard color scale")}\n'
                      f'White: Near-zero trend\n'
                      f'Black markers: Significant trend (p<{significance_threshold})')
        
        ax.text(legend_pos[0], legend_pos[1], legend_text, transform=ax.transAxes, 
                fontsize=11, verticalalignment='bottom', horizontalalignment='left',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.95, 
                         edgecolor='black', linewidth=1.5),
                zorder=10)
        
        plt.tight_layout()
        
        if output_name is None:
            mask_suffix = "_grassland" if self.mask is not None else "_global"
            output_name = f"{self.data_name}_{self.data_type}_trend_with_significance{mask_suffix}_{projection}"
        
        output_path = os.path.join(self.output_dir, output_name)
        os.makedirs(self.output_dir, exist_ok=True)

        dpi_high = 600
        base_params = {
            'bbox_inches': 'tight',
            'facecolor': 'white',
            'edgecolor': 'none',
            'pad_inches': 0.2
        }
        
        try:
            jpg_params = base_params.copy()
            jpg_params.update({'dpi': dpi_high, 'format': 'jpg'})
            fig.savefig(f"{output_path}_high_quality.jpg", **jpg_params)
        except Exception as e:
            print(f"✗ JPGfail: {e}")
        
        
        try:
            pdf_params = base_params.copy()
            pdf_params.update({'format': 'pdf', 'dpi': dpi_high})
            fig.savefig(f"{output_path}_high_quality.pdf", **pdf_params)
        except Exception as e:
            print(f"✗ PDFfail: {e}")
        
        plt.close(fig)

        stats_df = pd.DataFrame([{
            'Variable': self.data_type,
            'Total_Valid_Pixels': total_valid,
            'Significant_Pixels': significant_count,
            'Significant_Percentage': significant_percentage,
            'Significance_Threshold': significance_threshold,
            'Min_Trend': np.nanmin(valid_trend),
            'Max_Trend': np.nanmax(valid_trend),
            'Mean_Trend': mean_trend,
            'Median_Trend': median_trend,
            'Std_Trend': std_trend
        }])
        
        stats_csv = os.path.join(self.output_dir, f"{output_name}_statistics.csv")
        stats_df.to_csv(stats_csv, index=False, float_format='%.6f')
        
        return {
            'significant_percentage': significant_percentage,
            'significant_count': significant_count,
            'total_pixels': total_valid,
            'mean_trend': mean_trend,
            'median_trend': median_trend,
            'std_trend': std_trend
        }
    
    def calculate_regional_statistics(self, significance_threshold=0.1):

        if self.sen_slope is None or self.mk_p_value is None:
            raise ValueError("loadDATA")
        
        results = {}
        
        global_mask = ~np.isnan(self.sen_slope) & ~np.isnan(self.mk_p_value)
        if self.mask is not None:
            global_mask = global_mask & self.mask
        
        if np.sum(global_mask) > 0:
            global_trends = self.sen_slope[global_mask]
            global_p_values = self.mk_p_value[global_mask]
            global_significant = global_p_values < significance_threshold
            
            results['Global'] = {
                'name': 'Global',
                'total_pixels': np.sum(global_mask),
                'significant_pixels': np.sum(global_significant),
                'significant_percentage': np.sum(global_significant) / np.sum(global_mask) * 100,
                'mean_trend': np.mean(global_trends),
                'median_trend': np.median(global_trends),
                'std_trend': np.std(global_trends)
            }
        
        for region_key in REGION_DEFINITIONS.keys():
            regional_mask = self.create_regional_mask(region_key)
            valid_mask = regional_mask & ~np.isnan(self.sen_slope) & ~np.isnan(self.mk_p_value)
            
            if np.sum(valid_mask) > 0:
                regional_trends = self.sen_slope[valid_mask]
                regional_p_values = self.mk_p_value[valid_mask]
                regional_significant = regional_p_values < significance_threshold
                
                results[region_key] = {
                    'name': REGION_DEFINITIONS[region_key]['name'],
                    'total_pixels': np.sum(valid_mask),
                    'significant_pixels': np.sum(regional_significant),
                    'significant_percentage': np.sum(regional_significant) / np.sum(valid_mask) * 100,
                    'mean_trend': np.mean(regional_trends),
                    'median_trend': np.median(regional_trends),
                    'std_trend': np.std(regional_trends)
                }
        
        return results
    
    def plot_latitudinal_trend(self, output_name=None):
        if self.sen_slope is None:
            raise ValueError("loadDATA")
        
        n_lats = self.sen_slope.shape[0]
        latitudes = np.linspace(90 - self.resolution/2, -90 + self.resolution/2, n_lats)
        
        lat_trends = []
        lat_values = []
        lat_stds = []
        lat_counts = []
        
        for i, lat in enumerate(latitudes):
            row_trend = self.sen_slope[i, :]
            if self.mask is not None:
                row_trend = row_trend[self.mask[i, :]]
            
            valid_trend = row_trend[~np.isnan(row_trend)]
            if len(valid_trend) > 0:
                lat_trends.append(np.mean(valid_trend))
                lat_stds.append(np.std(valid_trend))
                lat_counts.append(len(valid_trend))
                lat_values.append(lat)
        
        lat_values = np.array(lat_values)
        lat_trends = np.array(lat_trends)
        lat_stds = np.array(lat_stds)
        lat_counts = np.array(lat_counts)
        
        fig, ax = plt.subplots(figsize=(8, 10))
        
        ax.plot(lat_trends, lat_values, '-', color='#2c7fb8', linewidth=2.5, zorder=5)
        
        regional_means = {}
        for region_key, region_def in REGION_DEFINITIONS.items():
            if region_key == 'Mid_Latitudes':
                bounds_list = region_def['bounds']
                mask = np.zeros_like(lat_values, dtype=bool)
                for bounds in bounds_list:
                    mask_part = (lat_values >= bounds[0]) & (lat_values <= bounds[1])
                    mask = mask | mask_part
            else:
                bounds = region_def['bounds']
                mask = (lat_values >= bounds[0]) & (lat_values <= bounds[1])
            
            if np.sum(mask) > 0:
                regional_mean = np.average(lat_trends[mask], weights=lat_counts[mask])
                regional_means[region_key] = regional_mean
            else:
                regional_means[region_key] = np.nan

        for i in range(len(lat_values)-1):
            if lat_trends[i] > 0 and lat_trends[i+1] > 0:
                ax.fill_betweenx([lat_values[i], lat_values[i+1]], 0, 
                               [lat_trends[i], lat_trends[i+1]], 
                               alpha=0.25, color='#31a354', edgecolor='none', zorder=1)
            elif lat_trends[i] < 0 and lat_trends[i+1] < 0:
                ax.fill_betweenx([lat_values[i], lat_values[i+1]], 
                               [lat_trends[i], lat_trends[i+1]], 0,
                               alpha=0.25, color='#e34a33', edgecolor='none', zorder=1)
        
        ax.set_ylabel('Latitude (°N)', fontsize=14, fontweight='normal')
        ax.set_xlabel(f'{self.data_type} Sen-Slope Trend ({self.data_config["unit_per_area"]})', 
                     fontsize=14, fontweight='normal')
        ax.set_ylim(-60, 80)
        
        ax.axvline(x=0, color='#636363', linestyle='-', linewidth=1.2, zorder=2)
        
        ax.axhline(y=-15, color='#636363', linestyle='--', linewidth=1.2, alpha=0.7, zorder=2)
        ax.axhline(y=15, color='#636363', linestyle='--', linewidth=1.2, alpha=0.7, zorder=2)
        ax.axhline(y=56, color='#636363', linestyle='--', linewidth=1.2, alpha=0.7, zorder=2)
        
        for region_key, mean_value in regional_means.items():
            if not np.isnan(mean_value):
                region_def = REGION_DEFINITIONS[region_key]
                color = region_def['color']
                
                if region_key == 'High_North':
                    y_region = np.array([56, 80])
                    x_region = np.array([mean_value, mean_value])
                    ax.plot(x_region, y_region, color=color, linestyle='-', 
                            linewidth=3, alpha=0.8, zorder=4)
                    ax.text(mean_value, 82, f'{mean_value:.4f}', 
                            ha='center', va='bottom', fontsize=11, color=color, fontweight='bold')
                elif region_key == 'Tropics':
                    y_region = np.array([-15, 15])
                    x_region = np.array([mean_value, mean_value])
                    ax.plot(x_region, y_region, color=color, linestyle='-', 
                            linewidth=3, alpha=0.8, zorder=4)
                    ax.text(mean_value, 17, f'{mean_value:.4f}', 
                            ha='center', va='bottom', fontsize=11, color=color, fontweight='bold')
                elif region_key == 'Mid_Latitudes':
                    y_mid_s = np.array([-60, -15])
                    x_mid_s = np.array([mean_value, mean_value])
                    ax.plot(x_mid_s, y_mid_s, color=color, linestyle='-', 
                            linewidth=3, alpha=0.8, zorder=4)
                    
                    y_mid_n = np.array([15, 56])
                    x_mid_n = np.array([mean_value, mean_value])
                    ax.plot(x_mid_n, y_mid_n, color=color, linestyle='-', 
                            linewidth=3, alpha=0.8, zorder=4)
                    
                    ax.text(mean_value, -62, f'{mean_value:.4f}', 
                            ha='center', va='top', fontsize=11, color=color, fontweight='bold')
        
        ax.set_yticks([-60, -30, -15, 0, 15, 30, 56])
        ax.set_yticklabels(['60°S', '30°S', '15°S', '0°', '15°N', '30°N', '56°N'])
        ax.grid(True, axis='x', alpha=0.3, linestyle='-', linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(1.5)
        ax.spines['bottom'].set_linewidth(1.5)
        
        plt.tight_layout()
        
        if output_name is None:
            output_name = f"{self.data_name}_{self.data_type}_latitudinal_sen_slope_trend"
        
        output_path = os.path.join(self.output_dir, output_name)
        fig.savefig(f"{output_path}.jpg", dpi=600, bbox_inches='tight')
        fig.savefig(f"{output_path}.pdf", format='pdf', bbox_inches='tight')
        fig.savefig(f"{output_path}.png", dpi=600, bbox_inches='tight')
        plt.close()
        
        print(f"{self.data_type} saved: {output_path}")

        trend_data = pd.DataFrame({
            'Latitude': lat_values,
            'Mean_Sen_Slope_Trend': lat_trends,
            'Std_Dev': lat_stds,
            'Pixel_Count': lat_counts.astype(int)
        })
        
        summary_stats = pd.DataFrame({
            'Region': [REGION_DEFINITIONS[key]['name'] for key in regional_means.keys()],
            'Mean_Sen_Slope_Trend': list(regional_means.values())
        })
        
        csv_path = os.path.join(self.output_dir, f"{output_name}_data.csv")
        trend_data.to_csv(csv_path, index=False, float_format='%.6f')
        
        summary_csv_path = os.path.join(self.output_dir, f"{output_name}_regional_summary.csv")
        summary_stats.to_csv(summary_csv_path, index=False, float_format='%.6f')
        


def batch_visualize_trends(mk_results_dir, output_base_dir, lucc_path=None, 
                          significance_threshold=0.1, projection='robinson'):


    variables = ['NEP', 'TER', 'GPP']

    summary_results = []
    
    
    for variable in variables:
        
        try:
            var_mk_dir = os.path.join(mk_results_dir, variable)
            if not os.path.exists(var_mk_dir):
                print(f"nodata {var_mk_dir}")
                continue

            var_output_dir = os.path.join(output_base_dir, variable)
            

            visualizer = TrendVisualizer(
                data_type=variable,
                data_name=variable,
                resolution=1.0,
                output_dir=var_output_dir,
                mk_results_dir=mk_results_dir
            )

            visualizer.load_mk_results()

            if lucc_path and os.path.exists(lucc_path):
                visualizer.load_land_cover_mask(lucc_path, target_value=[6,7,9,10])

            if variable == 'NEP':
                default_vmin, default_vmax = -10, 10
            elif variable in ['TER', 'GPP']:
                default_vmin, default_vmax = -50, 50
            elif variable in ['FLUC']:
                default_vmin, default_vmax = -1, 1
            elif variable == 'Temperature':
                default_vmin, default_vmax = -0.05, 0.05
            elif variable == 'SM_gleam':
                default_vmin, default_vmax = -0.003, 0.003
            else:
                default_vmin, default_vmax = None, None
            
            result = visualizer.plot_trend_with_significance(
                vmin=default_vmin,
                vmax=default_vmax,
                significance_threshold=significance_threshold,
                projection=projection,
                marker_type='dots'
            )
            
            visualizer.plot_latitudinal_trend()
            
            regional_stats = visualizer.calculate_regional_statistics(significance_threshold)
            
            summary_results.append({
                'Variable': variable,
                'Significant_Percentage': result['significant_percentage'],
                'Significant_Count': result['significant_count'],
                'Total_Pixels': result['total_pixels'],
                'Mean_Trend': result['mean_trend'],
                'Median_Trend': result['median_trend'],
                'Std_Trend': result['std_trend']
            })
            
            regional_df = pd.DataFrame([stats for stats in regional_stats.values()])
            regional_csv = os.path.join(var_output_dir, f"{variable}_regional_statistics.csv")
            regional_df.to_csv(regional_csv, index=False, float_format='%.6f')
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            continue
    
    if summary_results:
        summary_df = pd.DataFrame(summary_results)
        summary_csv = os.path.join(output_base_dir, "all_variables_significance_summary.csv")
        summary_df.to_csv(summary_csv, index=False, float_format='%.6f')

        
        for result in summary_results:
            print(f"{result['Variable']:12s}: "
                  f"{result['Significant_Percentage']:6.2f}% "
                  f"({result['Significant_Count']:,}/{result['Total_Pixels']:,}) "
                  f"Mean: {result['Mean_Trend']:8.4f}")

def example_usage():
    mk_results_dir = r"D:\00_GLB_NEE\0000000000000000---MK趋势检验\MK_results"
    output_base_dir = r"D:\00_GLB_NEE\0000000---全球草原图_三区域_含显著性"
    lucc_path = r"D:\00---全球NBP\Modis_LUCC_10_2_2023.tif"
    batch_visualize_trends(
        mk_results_dir=mk_results_dir,
        output_base_dir=output_base_dir,
        lucc_path=lucc_path,  
        significance_threshold=0.1, 
        projection='plate_carree' 
    )


if __name__ == "__main__":


    missing_libs = []
    
    try:
        example_usage()
        
    except KeyboardInterrupt:
        print("\n interupt")
    except Exception as e:
        print(f"\n error : {e}")
        import traceback
        traceback.print_exc()