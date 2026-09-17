

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

try:
    from semopy import Model
    from semopy.inspector import inspect
    from semopy import calc_stats
except ImportError:
    raise ImportError("请安装 semopy: pip install semopy")

# =========================================================================
# Configuration
# =========================================================================

INPUT_CSV = r"E:/SEM_ZScore_Monthly_2015_2024_STRICT_NEW0707.csv"
OUTPUT_DIR = r"E:/SEM_Results_Two_Regions"
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

COLORS = {
    'positive': '#00A087',
    'negative': '#F39B7F',
    'nonsig': '#B0B0B0',
    'Temperature': '#FF6347',
    'Soil_Moisture': '#4682B4',
    'VPD': '#FF8C00',
    'LAI': '#32CD32',
    'GPP': '#228B22',
    'TER': '#CD853F',
    'CO2': '#3C5488',
}

plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 11,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})

# =========================================================================
# Data Loading
# =========================================================================

def load_and_prepare_data(csv_path: str) -> pd.DataFrame:
    print("="*60)
    print(f"Loading data: {csv_path}")
    df = pd.read_csv(csv_path)
    
    col_mapping = {}
    for col in df.columns:
        col_lower = col.lower()
        if col_lower in ['soil_moisture', 'soilmoisture']:
            col_mapping[col] = 'Soil_Moisture'
        elif col_lower == 'temperature':
            col_mapping[col] = 'Temperature'
        elif col_lower == 'vpd':
            col_mapping[col] = 'VPD'
        elif col_lower == 'lai':
            col_mapping[col] = 'LAI'
        elif col_lower == 'gpp':
            col_mapping[col] = 'GPP'
        elif col_lower == 'ter':
            col_mapping[col] = 'TER'
        elif col_lower == 'co2':
            col_mapping[col] = 'CO2'
    
    if col_mapping:
        df = df.rename(columns=col_mapping)
    
    df = df.sort_values(['Region', 'Year', 'Month']).reset_index(drop=True)
    
    vars_needed = ['Temperature', 'Soil_Moisture', 'VPD', 'LAI', 'GPP', 'TER', 'CO2']
    missing_vars = [v for v in vars_needed if v not in df.columns]
    if missing_vars:
        raise ValueError(f"Missing variables: {missing_vars}")
    
    df_clean = df.dropna(subset=vars_needed)
    
    print(f"Total rows: {len(df_clean)}")
    for region in df_clean['Region'].unique():
        count = len(df_clean[df_clean['Region'] == region])
        print(f"  {region}: {count} months")
    print("="*60)
    
    return df_clean

# =========================================================================
# SEM Model Definition
# =========================================================================

def define_sem_model() -> str:
    model_spec = """
    # Structural paths
   #VPD ~ Temperature
    LAI ~ Soil_Moisture + Temperature
    GPP ~ LAI + Soil_Moisture + CO2 + VPD + Temperature
    TER ~ GPP + Soil_Moisture + Temperature
    
    # Covariances (only exogenous climate variables)
    Temperature ~~ Soil_Moisture
    # Temperature ~~ VPD
    # Soil_Moisture ~~ VPD
    """
    return model_spec

# =========================================================================
# Model Fitting
# =========================================================================

def calculate_fit_indices(model, df_region):
    try:
        stats = calc_stats(model)
        if isinstance(stats, pd.DataFrame):
            stats_dict = stats.iloc[0].to_dict() if len(stats) > 0 else {}
        elif isinstance(stats, pd.Series):
            stats_dict = stats.to_dict()
        else:
            stats_dict = stats if isinstance(stats, dict) else {}
        
        def safe_get(key, default=np.nan):
            val = stats_dict.get(key, default)
            if val is None or (isinstance(val, float) and np.isnan(val)):
                return default
            try:
                return float(val)
            except:
                return default
        
        chi2 = safe_get('chi2')
        dof = safe_get('DoF')
        
        return {
            'chi2': chi2,
            'dof': dof,
            'chi2_dof': chi2 / dof if (dof > 0 and not np.isnan(chi2)) else np.nan,
            'pvalue': safe_get('chi2 p-value'),
            'CFI': safe_get('CFI'),
            'TLI': safe_get('TLI'),
            'RMSEA': safe_get('RMSEA'),
            'SRMR': safe_get('SRMR', np.nan),
            'AIC': safe_get('AIC'),
            'BIC': safe_get('BIC')
        }
    except:
        return {k: np.nan for k in ['chi2', 'dof', 'chi2_dof', 'pvalue', 'CFI', 'TLI', 'RMSEA', 'SRMR', 'AIC', 'BIC']}

def calculate_total_effects(model_results: dict) -> pd.DataFrame:
    std_results = model_results['standardized']
    region = model_results['region']
    
    paths = {}
    for _, row in std_results.iterrows():
        if row['op'] == '~':
            key = f"{row['rval']}_{row['lval']}"
            paths[key] = row['Est. Std']
    
    direct_effects = {
        'T_VPD': paths.get('Temperature_VPD', 0),
        'T_LAI': paths.get('Temperature_LAI', 0),
        'SM_LAI': paths.get('Soil_Moisture_LAI', 0),
        'T_GPP': paths.get('Temperature_GPP', 0),
        'SM_GPP': paths.get('Soil_Moisture_GPP', 0),
        'VPD_GPP': paths.get('VPD_GPP', 0),
        'LAI_GPP': paths.get('LAI_GPP', 0),
        'CO2_GPP': paths.get('CO2_GPP', 0),
        'T_TER': paths.get('Temperature_TER', 0),
        'SM_TER': paths.get('Soil_Moisture_TER', 0),
        'GPP_TER': paths.get('GPP_TER', 0),
    }
    
    # Temperature indirect effects on GPP
    # T -> VPD -> GPP
    T_GPP_via_VPD = direct_effects['T_VPD'] * direct_effects['VPD_GPP']
    # T -> LAI -> GPP
    T_GPP_via_LAI = direct_effects['T_LAI'] * direct_effects['LAI_GPP']
    # Total indirect: T -> GPP
    T_GPP_indirect = T_GPP_via_VPD + T_GPP_via_LAI
    
    # Temperature indirect effects on TER
    # T -> VPD -> GPP -> TER
    T_TER_via_VPD_GPP = direct_effects['T_VPD'] * direct_effects['VPD_GPP'] * direct_effects['GPP_TER']
    # T -> LAI -> GPP -> TER
    T_TER_via_LAI_GPP = direct_effects['T_LAI'] * direct_effects['LAI_GPP'] * direct_effects['GPP_TER']
    # T -> GPP -> TER
    T_TER_via_GPP = direct_effects['T_GPP'] * direct_effects['GPP_TER']
    # Total indirect: T -> TER
    T_TER_indirect = T_TER_via_VPD_GPP + T_TER_via_LAI_GPP + T_TER_via_GPP
    
    # Soil Moisture indirect effects on GPP
    # SM -> LAI -> GPP
    SM_GPP_indirect = direct_effects['SM_LAI'] * direct_effects['LAI_GPP']
    
    # Soil Moisture indirect effects on TER
    # SM -> LAI -> GPP -> TER
    SM_TER_via_LAI_GPP = direct_effects['SM_LAI'] * direct_effects['LAI_GPP'] * direct_effects['GPP_TER']
    # SM -> GPP -> TER
    SM_TER_via_GPP = direct_effects['SM_GPP'] * direct_effects['GPP_TER']
    # Total indirect: SM -> TER
    SM_TER_indirect = SM_TER_via_LAI_GPP + SM_TER_via_GPP
    
    # VPD indirect effect on TER
    VPD_TER_indirect = direct_effects['VPD_GPP'] * direct_effects['GPP_TER']
    
    # CO2 indirect effect on TER
    CO2_TER_indirect = direct_effects['CO2_GPP'] * direct_effects['GPP_TER']
    
    effects_data = [
        {'Region': region, 'Driver': 'Temperature', 'Target': 'GPP',
         'Direct': direct_effects['T_GPP'], 'Indirect': T_GPP_indirect,
         'Total': direct_effects['T_GPP'] + T_GPP_indirect,
         'Via_VPD': T_GPP_via_VPD, 'Via_LAI': T_GPP_via_LAI},
        
        {'Region': region, 'Driver': 'Soil_Moisture', 'Target': 'GPP',
         'Direct': direct_effects['SM_GPP'], 'Indirect': SM_GPP_indirect,
         'Total': direct_effects['SM_GPP'] + SM_GPP_indirect,
         'Via_VPD': 0, 'Via_LAI': SM_GPP_indirect},
        
        {'Region': region, 'Driver': 'VPD', 'Target': 'GPP',
         'Direct': direct_effects['VPD_GPP'], 'Indirect': 0,
         'Total': direct_effects['VPD_GPP'],
         'Via_VPD': 0, 'Via_LAI': 0},
        
        {'Region': region, 'Driver': 'CO2', 'Target': 'GPP',
         'Direct': direct_effects['CO2_GPP'], 'Indirect': 0,
         'Total': direct_effects['CO2_GPP'],
         'Via_VPD': 0, 'Via_LAI': 0},
        
        {'Region': region, 'Driver': 'Temperature', 'Target': 'TER',
         'Direct': direct_effects['T_TER'], 'Indirect': T_TER_indirect,
         'Total': direct_effects['T_TER'] + T_TER_indirect,
         'Via_VPD': T_TER_via_VPD_GPP, 'Via_LAI': T_TER_via_LAI_GPP},
        
        {'Region': region, 'Driver': 'Soil_Moisture', 'Target': 'TER',
         'Direct': direct_effects['SM_TER'], 'Indirect': SM_TER_indirect,
         'Total': direct_effects['SM_TER'] + SM_TER_indirect,
         'Via_VPD': 0, 'Via_LAI': SM_TER_via_LAI_GPP},
        
        {'Region': region, 'Driver': 'VPD', 'Target': 'TER',
         'Direct': 0, 'Indirect': VPD_TER_indirect,
         'Total': VPD_TER_indirect,
         'Via_VPD': 0, 'Via_LAI': 0},
        
        {'Region': region, 'Driver': 'CO2', 'Target': 'TER',
         'Direct': 0, 'Indirect': CO2_TER_indirect,
         'Total': CO2_TER_indirect,
         'Via_VPD': 0, 'Via_LAI': 0},
    ]
    
    return pd.DataFrame(effects_data)

def fit_sem_model(data: pd.DataFrame, region: str, model_spec: str) -> dict:
    df_region = data[data['Region'] == region].copy()
    
    print(f"\nFitting: {region} (n={len(df_region)})")
    model = Model(model_spec)
    try:
        model.fit(df_region)
    except Exception as e:
        print(f"  Error: {e}")
        return None
    
    results = inspect(model)
    standardized = model.inspect(std_est=True)
    
    params = model.inspect()
    r2_scores = {}
    for var in ['LAI', 'GPP', 'TER']:
        try:
            total_var = df_region[var].var()
            residual_row = params[
                (params['lval'] == var) & 
                (params['rval'] == var) & 
                (params['op'] == '~~')
            ]
            
            if not residual_row.empty:
                residual_var = residual_row['Estimate'].values[0]
                r2 = 1 - (residual_var / total_var)
                r2_scores[var] = max(0, float(r2))
            else:
                r2_scores[var] = np.nan
        except:
            r2_scores[var] = np.nan
    
    fit_indices = calculate_fit_indices(model, df_region)
    
    print(
          f"R²:LAI={r2_scores.get('LAI', np.nan):.2f}, "
          f"GPP={r2_scores.get('GPP', np.nan):.2f}, "
          f"TER={r2_scores.get('TER', np.nan):.2f}")
    print(f"  Fit: χ²/df={fit_indices['chi2_dof']:.2f}, "
          f"CFI={fit_indices['CFI']:.3f}, "
          f"RMSEA={fit_indices['RMSEA']:.3f}")

    return {
        'model': model,
        'results': results,
        'standardized': standardized,
        'region': region,
        'n': len(df_region),
        'r2': r2_scores,
        'fit_indices': fit_indices
    }

# =========================================================================
# Visualization
# =========================================================================

def create_path_diagram(model_results: dict, save_path: str):
    region = model_results['region']
    std_results = model_results['standardized']
    r2_scores = model_results['r2']
    fit_indices = model_results['fit_indices']
    
    fig, ax = plt.subplots(figsize=(16, 11))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 11)
    ax.axis('off')
    
    nodes = {
        'Temperature': (2, 8),
        'Soil_Moisture': (2, 4),
        'CO2': (2, 1),
        'VPD': (7, 9),
        'LAI': (7, 5),
        'GPP': (13, 7),
        'TER': (13, 3.5)
    }
    
    node_styles = {
        'Temperature': {'c': COLORS['Temperature'], 's': 'circle', 'sz': 1.3},
        'Soil_Moisture': {'c': COLORS['Soil_Moisture'], 's': 'circle', 'sz': 1.3},
        'CO2': {'c': COLORS['CO2'], 's': 'circle', 'sz': 1.3},
        'VPD': {'c': COLORS['VPD'], 's': 'box', 'sz': 1.4},
        'LAI': {'c': COLORS['LAI'], 's': 'box', 'sz': 1.4},
        'GPP': {'c': COLORS['GPP'], 's': 'box', 'sz': 1.6},
        'TER': {'c': COLORS['TER'], 's': 'box', 'sz': 1.6},
    }
    
    for var, (x, y) in nodes.items():
        style = node_styles[var]
        if style['s'] == 'circle':
            patch = Circle((x, y), style['sz']/2, 
                         facecolor=style['c'], edgecolor='white', 
                         linewidth=3, alpha=0.9, zorder=10)
        else:
            patch = FancyBboxPatch((x-style['sz']/2, y-style['sz']/2), style['sz'], style['sz'],
                                 boxstyle="round,pad=0.2",
                                 facecolor=style['c'], edgecolor='white',
                                 linewidth=3, alpha=0.9, zorder=10)
        ax.add_patch(patch)
        
        label = var.replace('_', '\n')
        ax.text(x, y, label, ha='center', va='center', 
                color='white', fontweight='bold', fontsize=10, zorder=11)
        
        if var in r2_scores and not np.isnan(r2_scores[var]):
            ax.text(x, y + style['sz']/2 + 0.5, f"$R^2={r2_scores[var]:.2f}$",
                   ha='center', fontsize=10, color='#333333', fontweight='bold')

    def draw_path(u, v, cur=0, y_off=0):
        path_row = std_results[
            (std_results['lval'] == v) & 
            (std_results['rval'] == u) & 
            (std_results['op'] == '~')
        ]
        
        if path_row.empty: 
            return
        
        coef = path_row['Est. Std'].values[0]
        pval = path_row['p-value'].values[0]
        
        width = abs(coef) * 5 + 1.0
        if pval < 0.05:
            color = COLORS['positive'] if coef > 0 else COLORS['negative']
            style = '-'
            alpha = 0.9
        else:
            color = COLORS['nonsig']
            style = '--'
            alpha = 0.6
        
        sig = '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else 'ns'
        
        x1, y1 = nodes[u]
        x2, y2 = nodes[v]
        
        dist = np.sqrt((x2-x1)**2 + (y2-y1)**2)
        shrink = 0.7
        dx, dy = (x2-x1)/dist, (y2-y1)/dist
        x1 += dx*shrink; y1 += dy*shrink
        x2 -= dx*shrink; y2 -= dy*shrink
        
        arrow = FancyArrowPatch((x1, y1), (x2, y2),
                               arrowstyle='->,head_width=0.6,head_length=0.6',
                               connectionstyle=f"arc3,rad={cur}",
                               color=color, linewidth=width, linestyle=style,
                               alpha=alpha, zorder=5)
        ax.add_patch(arrow)
        
        mid_x = (x1+x2)/2
        mid_y = (y1+y2)/2 + y_off
        if cur != 0: mid_y += cur * 2
            
        txt = f"{coef:.2f}{sig}"
        ax.text(mid_x, mid_y, txt, ha='center', va='center', fontsize=9, fontweight='bold', color=color,bbox=dict(facecolor='white', edgecolor='none', alpha=0.9, pad=0.5),
               zorder=6)

    connections = [
        ('Temperature', 'VPD', 0, 0),
        ('Temperature', 'LAI', 0.15, 0.3),
        ('Soil_Moisture', 'LAI', 0, 0),
        ('Temperature', 'GPP', 0.25, 0.5),
        ('Soil_Moisture', 'GPP', 0.2, -0.3),
        ('VPD', 'GPP', 0, 0),
        ('LAI', 'GPP', 0, 0),
        ('CO2', 'GPP', 0.15, -0.6),
        
        ('Temperature', 'TER', -0.25, 0.5),
        ('Soil_Moisture', 'TER', -0.15, -0.4),
        ('GPP', 'TER', 0, 0)
    ]
    
    for u, v, c, off in connections:
        draw_path(u, v, c, off)

    residual_corrs = [
        ('Temperature', 'Soil_Moisture', 0.3),
        ('Temperature', 'VPD', 0.3),
    ]
    
    for u, v, rad in residual_corrs:
        x1, y1 = nodes[u]
        x2, y2 = nodes[v]
        
        arrow = FancyArrowPatch(
            (x1, y1), (x2, y2),
            arrowstyle='<->,head_width=0.4,head_length=0.4',
            connectionstyle=f"arc3,rad={rad}",
            color='#808080',
            linewidth=2.0,
            linestyle=':',
            alpha=0.7,
            zorder=4
        )
        ax.add_patch(arrow)

    fit_text = f"χ²/df={fit_indices['chi2_dof']:.2f}, p={fit_indices['pvalue']:.3f}\n"
    fit_text += f"CFI={fit_indices['CFI']:.3f}, RMSEA={fit_indices['RMSEA']:.3f}"
    ax.text(8, 10.3, f"{region}: Climate-Carbon Coupling",
           ha='center', fontsize=16, fontweight='bold')
    ax.text(8, 0.3, fit_text, ha='center', fontsize=11, color='#333333',
           bbox=dict(facecolor='white', edgecolor='#333333', linewidth=2, alpha=0.95, pad=10))
    
    legend_elements = [
        mpatches.Patch(color=COLORS['positive'], label='Positive (p<0.05)'),
        mpatches.Patch(color=COLORS['negative'], label='Negative (p<0.05)'),
        mpatches.Patch(color=COLORS['nonsig'], label='Non-significant'),
        mpatches.Patch(color='#808080', label='Covariance')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10, framealpha=0.9)
    
    plt.savefig(save_path, dpi=300, facecolor='white', bbox_inches='tight')
    plt.close()

def create_total_effects_comparison(effects_df: pd.DataFrame, output_dir: str):
    plt.rcParams.update({
        'font.family': 'Arial',
        'font.size': 10,
        'axes.linewidth': 1.2,
    })
    
    regions = effects_df['Region'].unique()
    n_regions = len(regions)
    
    fig, axes = plt.subplots(1, n_regions, figsize=(14, 5.5), gridspec_kw={'wspace': 0.35})
    
    if n_regions == 1:
        axes = [axes]
    
    colors = {
        'Temperature_direct': '#E64B35',
        'Temperature_indirect': '#F39B7F',
        'Moisture_direct': '#4DBBD5',
        'Moisture_indirect': '#91D1C2',
    }
    
    patterns = {'direct': None, 'indirect': '///'}
    
    for idx, region in enumerate(regions):
        ax = axes[idx]
        
        region_data = effects_df[
            (effects_df['Region'] == region) & 
            (effects_df['Driver'].isin(['Temperature', 'Soil_Moisture']))
        ]
        
        targets = ['GPP', 'TER']
        drivers = ['Temperature', 'Soil_Moisture']
        
        x_positions = np.arange(len(targets))
        bar_width = 0.35
        
        plot_data = {target: {driver: {'direct': 0, 'indirect': 0} 
                              for driver in drivers} for target in targets}
        
        for _, row in region_data.iterrows():
            target = row['Target']
            driver = row['Driver']
            if driver in drivers:
                plot_data[target][driver]['direct'] = row['Direct']
                plot_data[target][driver]['indirect'] = row['Indirect']
        
        for i, target in enumerate(targets):
            x_base = x_positions[i]
            
            temp_direct = plot_data[target]['Temperature']['direct']
            temp_indirect = plot_data[target]['Temperature']['indirect']
            
            moist_direct = plot_data[target]['Soil_Moisture']['direct']
            moist_indirect = plot_data[target]['Soil_Moisture']['indirect']

            x_temp = x_base - bar_width/2
            
            ax.bar(x_temp, temp_direct, bar_width, 
                  color=colors['Temperature_direct'],
                  edgecolor='white', linewidth=1.5,
                  label='Temperature (Direct)' if i == 0 else '',
                  zorder=3)
            
            if temp_direct >= 0:
                ax.bar(x_temp, temp_indirect, bar_width, bottom=temp_direct,
                      color=colors['Temperature_indirect'],
                      edgecolor='white', linewidth=1.5,
                      hatch=patterns['indirect'],
                      label='Temperature (Indirect)' if i == 0 else '',
                      zorder=3)
                temp_total = temp_direct + temp_indirect
                y_text = temp_total + 0.02
                va = 'bottom'
            else:
                ax.bar(x_temp, temp_indirect, bar_width, bottom=temp_direct,
                      color=colors['Temperature_indirect'],
                      edgecolor='white', linewidth=1.5,
                      hatch=patterns['indirect'],
                      label='Temperature (Indirect)' if i == 0 else '',
                      zorder=3)
                temp_total = temp_direct + temp_indirect
                y_text = temp_total - 0.02
                va = 'top'
            
            ax.text(x_temp, y_text, f'{temp_total:.2f}',
                   ha='center', va=va, fontsize=9, fontweight='bold',
                   color='#2C3E50', zorder=4)
            
            x_moist = x_base + bar_width/2

            ax.bar(x_moist, moist_direct, bar_width,
                  color=colors['Moisture_direct'],
                  edgecolor='white', linewidth=1.5,
                  label='Soil Moisture (Direct)' if i == 0 else '',
                  zorder=3)
            
            if moist_direct >= 0:
                ax.bar(x_moist, moist_indirect, bar_width, bottom=moist_direct,
                      color=colors['Moisture_indirect'],
                      edgecolor='white', linewidth=1.5,
                      hatch=patterns['indirect'],
                      label='Soil Moisture (Indirect)' if i == 0 else '',
                      zorder=3)
                moist_total = moist_direct + moist_indirect
                y_text = moist_total + 0.02
                va = 'bottom'
            else:
                ax.bar(x_moist, moist_indirect, bar_width, bottom=moist_direct,
                      color=colors['Moisture_indirect'],
                      edgecolor='white', linewidth=1.5,
                      hatch=patterns['indirect'],
                      label='Soil Moisture (Indirect)' if i == 0 else '',
                      zorder=3)
                moist_total = moist_direct + moist_indirect
                y_text = moist_total - 0.02
                va = 'top'
            
            ax.text(x_moist, y_text, f'{moist_total:.2f}',
                   ha='center', va=va, fontsize=9, fontweight='bold',
                   color='#2C3E50', zorder=4)
        
        ax.set_xticks(x_positions)
        ax.set_xticklabels(targets, fontsize=11, fontweight='bold')
        
        if idx == 0:
            ax.set_ylabel('Standardized Effect', fontsize=11, fontweight='bold')
        
        ax.axhline(y=0, color='#34495E', linestyle='-', linewidth=1.2, alpha=0.8, zorder=1)
        ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        
        all_values = []
        for target in targets:
            for driver in drivers:
                direct = plot_data[target][driver]['direct']
                indirect = plot_data[target][driver]['indirect']
                all_values.extend([direct, indirect, direct + indirect])
        
        if len(all_values) > 0:
            y_max = max(abs(min(all_values)), abs(max(all_values))) * 1.25
            ax.set_ylim(-y_max, y_max)
            ax.set_title(f'{region}', fontsize=12, fontweight='bold', pad=10, color='#2C3E50')
        for spine in ax.spines.values():
            spine.set_linewidth(1.2)
            spine.set_color('#34495E')
        
        if idx == 0:
            handles, labels = ax.get_legend_handles_labels()
            order = [0, 1, 2, 3]
            ax.legend([handles[i] for i in order], [labels[i] for i in order],
                     loc='upper left', frameon=True, framealpha=0.95,
                     edgecolor='#34495E', fontsize=9,
                     ncol=1, columnspacing=1, handlelength=2)
    fig.suptitle('Temperature and Soil Moisture Effects on Carbon Fluxes\n(Mid-Latitude vs Equator)', 
                fontsize=14, fontweight='bold', y=0.98, color='#2C3E50')
    
    save_path = Path(output_dir) / "total_effects_comparison_two_regions.pdf"
    plt.savefig(save_path, dpi=300, facecolor='white', bbox_inches='tight')
    
    save_path_png = Path(output_dir) / "total_effects_comparison_two_regions.png"
    plt.savefig(save_path_png, dpi=300, facecolor='white', bbox_inches='tight')
    
    plt.close()
    print(f"  Effects comparison saved: {save_path}")

# =========================================================================
# Main
# =========================================================================

def main():
    print("\n" + "="*60)
    print("COMPREHENSIVE CLIMATE-CARBON COUPLING ANALYSIS")
    print("Two Regions: Mid-Latitude vs Equator")
    print("="*60)
    
    df = load_and_prepare_data(INPUT_CSV)
    model_spec = define_sem_model()
    
    regions = df['Region'].unique()
    print(f"\nRegions to analyze: {list(regions)}")
    
    all_results = []
    all_effects = []
    
    for region in regions:
        print(f"\n{'='*40}")
        print(f"Region: {region}")
        print(f"{'='*40}")
        
        res = fit_sem_model(df, region, model_spec)
        if res is None: 
            continue
        all_results.append(res)
        effects = calculate_total_effects(res)
        all_effects.append(effects)
        
        out_path = Path(OUTPUT_DIR) / f"path_diagram_{region.replace(' ', '_')}.pdf"
        create_path_diagram(res, str(out_path))
        print(f"  Path diagram saved: {out_path}")
    
    if not all_results:
        print("Error: No successful model fits!")
        return
    
    effects_df = pd.concat(all_effects, ignore_index=True)
    
    print(f"\n{'='*40}")
    print("Exporting Results")
    print(f"{'='*40}")
    
    effects_csv = Path(OUTPUT_DIR) / "comprehensive_effects_two_regions.csv"
    effects_df.to_csv(effects_csv, index=False, float_format="%.4f")
    print(f"  Effects table: {effects_csv}")
    
    fit_summary = []
    for res in all_results:
        fit_summary.append({
            'Region': res['region'],
            'n': res['n'],
            'chi2_dof': res['fit_indices']['chi2_dof'],
            'p_value': res['fit_indices']['pvalue'],
            'CFI': res['fit_indices']['CFI'],
            'RMSEA': res['fit_indices']['RMSEA'],
            'R2_VPD': res['r2'].get('VPD', np.nan),
            'R2_LAI': res['r2'].get('LAI', np.nan),
            'R2_GPP': res['r2'].get('GPP', np.nan),
            'R2_TER': res['r2'].get('TER', np.nan)
        })
    
    fit_df = pd.DataFrame(fit_summary)
    fit_csv = Path(OUTPUT_DIR) / "model_fit_indices_two_regions.csv"
    fit_df.to_csv(fit_csv, index=False, float_format="%.3f")
    print(f"  Fit indices: {fit_csv}")
    
    create_total_effects_comparison(effects_df, OUTPUT_DIR)
    # Generate comprehensive report
    report_path = Path(OUTPUT_DIR) / "comprehensive_analysis_report_two_regions.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("COMPREHENSIVE CLIMATE-CARBON COUPLING ANALYSIS\n")
        f.write("Two Regions: Mid-Latitude vs Equator\n")
        f.write("="*80 + "\n\n")
        
        f.write("Model Structure:\n")
        f.write("  VPD ~ Temperature\n")
        f.write("  LAI ~ Soil_Moisture + Temperature\n")
        f.write("  GPP ~ LAI + Soil_Moisture + CO2 + VPD + Temperature\n")
        f.write("  TER ~ GPP + Soil_Moisture + Temperature\n\n")
        
        f.write("Covariances:\n")
        f.write("  Temperature ~~ Soil_Moisture (exogenous climate variables)\n\n")
        
        f.write("="*80 + "\n\n")
        f.write("1. Total Effects Decomposition:\n\n")
        f.write(effects_df.to_string(index=False, float_format="%.3f"))
        
        f.write("\n\n" + "="*80 + "\n\n")
        f.write("2. Pathway Analysis:\n\n")
        
        for region in regions:
            region_data = effects_df[effects_df['Region'] == region]
            f.write(f"{region}:\n\n")
            
            # Temperature pathways to GPP
            t_gpp_data = region_data[
                (region_data['Driver'] == 'Temperature') & 
                (region_data['Target'] == 'GPP')
            ]
            if len(t_gpp_data) > 0:
                t_gpp_total = t_gpp_data['Total'].values[0]
                t_gpp_direct = t_gpp_data['Direct'].values[0]
                t_vpd_gpp = t_gpp_data['Via_VPD'].values[0]
                t_lai_gpp = t_gpp_data['Via_LAI'].values[0]
                
                f.write(f"  Temperature → GPP:\n")
                f.write(f"    Total effect: {t_gpp_total:+.3f}\n")
                f.write(f"    Direct: {t_gpp_direct:+.3f} ({abs(t_gpp_direct/t_gpp_total)*100:.1f}%)\n")
                f.write(f"    Via VPD: {t_vpd_gpp:+.3f} ({abs(t_vpd_gpp/t_gpp_total)*100:.1f}%)\n")
                f.write(f"    Via LAI: {t_lai_gpp:+.3f} ({abs(t_lai_gpp/t_gpp_total)*100:.1f}%)\n\n")
            
            # Soil Moisture pathways to GPP
            sm_gpp_data = region_data[
                (region_data['Driver'] == 'Soil_Moisture') & 
                (region_data['Target'] == 'GPP')
            ]
            if len(sm_gpp_data) > 0:
                sm_gpp_total = sm_gpp_data['Total'].values[0]
                sm_gpp_direct = sm_gpp_data['Direct'].values[0]
                sm_lai_gpp = sm_gpp_data['Via_LAI'].values[0]
                
                f.write(f"  Soil Moisture → GPP:\n")
                f.write(f"    Total effect: {sm_gpp_total:+.3f}\n")
                f.write(f"    Direct: {sm_gpp_direct:+.3f} ({abs(sm_gpp_direct/sm_gpp_total)*100:.1f}%)\n")
                f.write(f"    Via LAI: {sm_lai_gpp:+.3f} ({abs(sm_lai_gpp/sm_gpp_total)*100:.1f}%)\n\n")
            
            # VPD effect on GPP
            vpd_gpp_data = region_data[
                (region_data['Driver'] == 'VPD') & 
                (region_data['Target'] == 'GPP')
            ]
            if len(vpd_gpp_data) > 0:
                vpd_gpp = vpd_gpp_data['Direct'].values[0]
                f.write(f"  VPD → GPP (direct): {vpd_gpp:+.3f}\n\n")
            
            # CO2 effect on GPP
            co2_gpp_data = region_data[
                (region_data['Driver'] == 'CO2') & 
                (region_data['Target'] == 'GPP')
            ]
            if len(co2_gpp_data) > 0:
                co2_gpp = co2_gpp_data['Direct'].values[0]
                f.write(f"  CO2 → GPP (fertilization): {co2_gpp:+.3f}\n\n")
            
            f.write("\n")
        f.write("="*80 + "\n\n")
        f.write("3. Regional Comparison:\n\n")
        
        for target in ['GPP', 'TER']:
            f.write(f"{target} Effects:\n")
            target_data = effects_df[effects_df['Target'] == target]
            
            for driver in ['Temperature', 'Soil_Moisture', 'VPD', 'CO2']:
                driver_data = target_data[target_data['Driver'] == driver]
                if len(driver_data) == 0:
                    continue
                f.write(f"\n  {driver} → {target}:\n")
                
                for _, row in driver_data.iterrows():
                    region = row['Region']
                    direct = row['Direct']
                    indirect = row['Indirect']
                    total = row['Total']
                    f.write(f"    {region}: Direct={direct:+.3f}, Indirect={indirect:+.3f}, Total={total:+.3f}\n")
            f.write("\n")
        
        f.write("="*80 + "\n\n")
        f.write("4. Key Findings:\n\n")
        
        for target in ['GPP', 'TER']:
            target_data = effects_df[effects_df['Target'] == target]
            if len(target_data) == 0:
                continue
            strongest = target_data.loc[target_data['Total'].abs().idxmax()]
            
            f.write(f"{target} Strongest Driver:\n")
            f.write(f"  {strongest['Driver']} in {strongest['Region']}\n")
            f.write(f"  Total Effect: {strongest['Total']:+.3f}\n")
            f.write(f"  (Direct: {strongest['Direct']:+.3f}, Indirect: {strongest['Indirect']:+.3f})\n\n")
        
        f.write("VPD Pathway Importance:\n")
        for region in regions:
            region_data = effects_df[effects_df['Region'] == region]
            # Temperature effect via VPD
            t_gpp_data = region_data[
                (region_data['Driver'] == 'Temperature') & 
                (region_data['Target'] == 'GPP')
            ]
            if len(t_gpp_data) > 0:
                t_gpp_total = t_gpp_data['Total'].values[0]
                t_vpd_gpp = t_gpp_data['Via_VPD'].values[0]
                if t_gpp_total != 0:
                    vpd_contribution = (t_vpd_gpp / t_gpp_total) * 100
                    f.write(f"  {region}: VPD pathway = {abs(vpd_contribution):.1f}% of Temperature → GPP\n")
        
        f.write("\n" + "="*80 + "\n\n")
        f.write("5. Temperature vs Moisture Dominance:\n\n")
        
        for region in regions:
            region_data = effects_df[
                (effects_df['Region'] == region) & 
                (effects_df['Driver'].isin(['Temperature', 'Soil_Moisture']))
            ]
            temp_effects = region_data[region_data['Driver'] == 'Temperature']['Total'].abs().sum()
            moisture_effects = region_data[region_data['Driver'] == 'Soil_Moisture']['Total'].abs().sum()
            
            f.write(f"{region}:\n")
            f.write(f"  Temperature total strength: {temp_effects:.3f}\n")
            f.write(f"  Soil Moisture total strength: {moisture_effects:.3f}\n")
            if temp_effects > moisture_effects:
                ratio = temp_effects / moisture_effects if moisture_effects > 0 else np.inf
                f.write(f"  Dominant factor: Temperature ({ratio:.2f}x stronger)\n\n")
            else:
                ratio = moisture_effects / temp_effects if temp_effects > 0 else np.inf
                f.write(f"  Dominant factor: Soil Moisture ({ratio:.2f}x stronger)\n\n")
        
        f.write("="*80 + "\n\n")
        f.write("6. CO2 and VPD Effects Summary:\n\n")
        
        for region in regions:
            region_data = effects_df[effects_df['Region'] == region]
            
            f.write(f"{region}:\n")
            
            # CO2 effects
            co2_gpp = region_data[
                (region_data['Driver'] == 'CO2') & 
                (region_data['Target'] == 'GPP')]['Total'].values[0] if len(region_data[
                (region_data['Driver'] == 'CO2') & 
                (region_data['Target'] == 'GPP')
            ]) > 0 else 0
            
            co2_ter = region_data[
                (region_data['Driver'] == 'CO2') & 
                (region_data['Target'] == 'TER')
            ]['Total'].values[0] if len(region_data[
                (region_data['Driver'] == 'CO2') & 
                (region_data['Target'] == 'TER')
            ]) > 0 else 0
            
            f.write(f"  CO2 fertilization:\n")
            f.write(f"    → GPP: {co2_gpp:+.3f} (direct)\n")
            f.write(f"    → TER: {co2_ter:+.3f} (via GPP)\n")
            
            # VPD effects
            vpd_gpp = region_data[
                (region_data['Driver'] == 'VPD') & 
                (region_data['Target'] == 'GPP')
            ]['Total'].values[0] if len(region_data[
                (region_data['Driver'] == 'VPD') & 
                (region_data['Target'] == 'GPP')
            ]) > 0 else 0
            
            vpd_ter = region_data[
                (region_data['Driver'] == 'VPD') & 
                (region_data['Target'] == 'TER')
            ]['Total'].values[0] if len(region_data[
                (region_data['Driver'] == 'VPD') & 
                (region_data['Target'] == 'TER')
            ]) > 0 else 0
            
            f.write(f"  VPD effects:\n")
            f.write(f"    → GPP: {vpd_gpp:+.3f} (direct)\n")
            f.write(f"    → TER: {vpd_ter:+.3f} (via GPP)\n\n")
        
        f.write("="*80 + "\n\n")
        f.write("7. Model Fit Comparison:\n\n")
        f.write(fit_df.to_string(index=False, float_format="%.3f"))
        
        f.write("\n\n" + "="*80 + "\n\n")
        f.write("8. Indirect Effect Contributions:\n\n")
        
        for region in regions:
            region_data = effects_df[
                (effects_df['Region'] == region) & 
                (effects_df['Driver'].isin(['Temperature', 'Soil_Moisture']))
            ]
            
            f.write(f"{region}:\n")
            for driver in ['Temperature', 'Soil_Moisture']:
                driver_data = region_data[region_data['Driver'] == driver]
                for _, row in driver_data.iterrows():
                    target = row['Target']
                    direct = row['Direct']
                    indirect = row['Indirect']
                    total = row['Total']
                    
                    if total != 0:
                        indirect_pct = (indirect / total) * 100
                        f.write(f"  {driver} → {target}:\n")
                        f.write(f"    Total: {total:+.3f} (Direct: {abs(direct/total)*100:.1f}%, Indirect: {abs(indirect_pct):.1f}%)\n")
            f.write("\n")
        
        f.write("="*80 + "\n")
        f.write("Analysis completed: " + pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S") + "\n")
        f.write("="*80 + "\n")
    print(f"  Report: {report_path}")
    
    print(f"\n{'='*60}")
    print("✓ Analysis Complete!")
    print(f"  📊 Path diagrams: {OUTPUT_DIR}/path_diagram_*.pdf")
    print(f"  📈 Effects comparison: {OUTPUT_DIR}/total_effects_comparison_two_regions.pdf")
    print(f"  📋 Effects table: {effects_csv}")
    print(f"  📋 Fit indices: {fit_csv}")
    print(f"  📄 Report: {report_path}")
    print("="*60)
    
    print(f"\n{'='*60}")
    print("Key Results Summary:")
    print("="*60)
    
    for target in ['GPP', 'TER']:
        print(f"\n{target} Effects Ranking:")
        target_data = effects_df[effects_df['Target'] == target].copy()
        target_data['abs_total'] = target_data['Total'].abs()
        target_data = target_data.sort_values('abs_total', ascending=False)
        
        for i, (_, row) in enumerate(target_data.head(8).iterrows(), 1):
            print(f"  {i}. {row['Driver']} ({row['Region']}): {row['Total']:+.3f}")
    
    print(f"\n{'='*60}")
    print("VPD Pathway Contribution:")
    print("="*60)
    
    for region in regions:
        region_data = effects_df[effects_df['Region'] == region]
        
        t_gpp_data = region_data[
            (region_data['Driver'] == 'Temperature') & 
            (region_data['Target'] == 'GPP')
        ]
        if len(t_gpp_data) > 0:
            t_gpp_total = t_gpp_data['Total'].values[0]
            t_gpp_direct = t_gpp_data['Direct'].values[0]
            t_vpd_gpp = t_gpp_data['Via_VPD'].values[0]
            t_lai_gpp = t_gpp_data['Via_LAI'].values[0]
            
            print(f"\n{region} - Temperature → GPP:")
            print(f"  Total: {t_gpp_total:+.3f}")
            print(f"  Direct: {t_gpp_direct:+.3f} ({abs(t_gpp_direct/t_gpp_total)*100:.1f}%)")
            print(f"  Via VPD: {t_vpd_gpp:+.3f} ({abs(t_vpd_gpp/t_gpp_total)*100:.1f}%)")
            print(f"  Via LAI: {t_lai_gpp:+.3f} ({abs(t_lai_gpp/t_gpp_total)*100:.1f}%)")
    
    print(f"\n{'='*60}")
    print("CO2 Fertilization Effects:")
    print("="*60)
    
    co2_data = effects_df[effects_df['Driver'] == 'CO2']
    for _, row in co2_data.iterrows():
        region = row['Region']
        target = row['Target']
        total = row['Total']
        effect_type = "Direct" if target == 'GPP' else "Indirect (via GPP)"
        print(f"  {region} - CO2 → {target}: {total:+.3f} ({effect_type})")
    
    print(f"\n{'='*60}")
    print("Temperature vs Moisture Dominance:")
    print("="*60)
    
    for region in regions:
        region_data = effects_df[
            (effects_df['Region'] == region) & 
            (effects_df['Driver'].isin(['Temperature', 'Soil_Moisture']))
        ]
        
        temp_effects = region_data[region_data['Driver'] == 'Temperature']['Total'].abs().sum()
        moisture_effects = region_data[region_data['Driver'] == 'Soil_Moisture']['Total'].abs().sum()
        
        print(f"\n{region}:")
        print(f"  Temperature: {temp_effects:.3f}")
        print(f"  Soil Moisture: {moisture_effects:.3f}")
        
        if temp_effects > moisture_effects:
            ratio = temp_effects / moisture_effects if moisture_effects > 0 else np.inf
            print(f"  → Temperature dominant ({ratio:.2f}x)")
        else:
            ratio = moisture_effects / temp_effects if temp_effects > 0 else np.inf
            print(f"  → Moisture dominant ({ratio:.2f}x)")
    
    print(f"\n{'='*60}")
    print("Model Fit Summary:")
    print("="*60)
    print(fit_df.to_string(index=False))
    print("="*60)
    
    # Regional comparison table
    print(f"\n{'='*60}")
    print("Regional Comparison Table:")
    print("="*60)
    
    comparison_data = []
    for driver in ['Temperature', 'Soil_Moisture', 'VPD', 'CO2']:
        for target in ['GPP', 'TER']:
            row_data = {'Driver': driver, 'Target': target}
            for region in regions:
                region_data = effects_df[
                    (effects_df['Region'] == region) &
                    (effects_df['Driver'] == driver) &
                    (effects_df['Target'] == target)
                ]
                if len(region_data) > 0:
                    row_data[region] = region_data['Total'].values[0]
                else:
                    row_data[region] = np.nan
            comparison_data.append(row_data)
    
    comparison_df = pd.DataFrame(comparison_data)
    print(comparison_df.to_string(index=False, float_format="%.3f"))
    
    comparison_csv = Path(OUTPUT_DIR) / "regional_comparison_table.csv"
    comparison_df.to_csv(comparison_csv, index=False, float_format="%.4f")
    print(f"\n✓ Regional comparison table saved: {comparison_csv}")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    main()