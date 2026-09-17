import os
import glob
import numpy as np
import pandas as pd
import rasterio

'''
    'NBP': {
        'dir':     r"E:\000000------文章1\数据库\GCAS_NBP\annual",
        'pattern': "NBP_*.tif",
        'label':   'NBP',
    },
     'NEP': {
         'dir':     r"E:\000000------文章1\数据库\GCAS_NEP\annual",
         'pattern': "NEP_*.tif",
         'label':   'NEP',
     },
     'GPP': {
         'dir':     r"E:\000000------文章1\数据库\GPP\annual",
         'pattern': "GPP_*.tif",
         'label':   'GPP',
     },
     'TER': {
         'dir':     r"E:\000000------文章1\数据库\TER\annual",
         'pattern': "TER_*.tif",
         'label':   'TER',
     },
     'Fire': {
         'dir':     r"E:\000000------文章1\数据库\Fire\annual",
         'pattern': "Fire_*.tif",
         'label':   'Fire',
     },
    # 取消注释以启用更多变量：
     'FluxSat_GPP': {
         'dir':     r"E:\Dong_NC_GCAS\CarbonGPP\FluxSat\yearly",
         'pattern': "FluxSat_GPP_*.tif",
         'label':   'FluxSat_GPP',
     },
     'GOSIF_GPP': {
         'dir':     r"E:\Dong_NC_GCAS\CarbonGPP\GOSIF\yearly\1degree",
         'pattern': "GOSIF_GPP_*.tif",
         'label':   'GOSIF_GPP',
     },
    'NBP_CAMS-Satellite': {
        'dir':     r"E:\000000------文章1\数据库\GCB2025\models—indivi\individual_models\1",
        'pattern': "NBP_CAMS-Satellite_*.tif",
        'label':   'NBP_CAMS-Satellite',
    },
    'NBP_CMS-Flux': {
        'dir':     r"E:\000000------文章1\数据库\GCB2025\models—indivi\individual_models\2",
        'pattern': "NBP_CMS-Flux_*.tif",
        'label':   'NBP_CMS-Flux',
    },
    'NBP_COLA': {
        'dir':     r"E:\000000------文章1\数据库\GCB2025\models—indivi\individual_models\3",
        'pattern': "NBP_COLA_*.tif",
        'label':   'NBP_COLA',
    },
    'NBP_GCASv2': {
        'dir':     r"E:\000000------文章1\数据库\GCB2025\models—indivi\individual_models\4",
        'pattern': "NBP_GCASv2_*.tif",
        'label':   'NBP_GCASv2',
    },
    'NBP_GONGGA': {
        'dir':     r"E:\000000------文章1\数据库\GCB2025\models—indivi\individual_models\5",
        'pattern': "NBP_GONGGA_*.tif",
        'label':   'NBP_GONGGA',
    },
    'NBP': {
        'dir':     r"E:\000000------文章1\数据库\GCB2025\models—indivi\individual_models\6",
        'pattern': "NBP_NISMON-CO2_GOSAT_*.tif",
        'label':   'NBP_NISMON-CO2_GOSAT',
    },
    'NBP_NTFVAR': {
        'dir':     r"E:\000000------文章1\数据库\GCB2025\models—indivi\individual_models\7",
        'pattern': "NBP_NTFVAR_*.tif",
        'label':   'NBP_NTFVAR',
    },
    'NBP_THU': {
        'dir':     r"E:\000000------文章1\数据库\GCB2025\models—indivi\individual_models\8",
        'pattern': "NBP_THU_*.tif",
        'label':   'NBP_THU',
    },
     'GCB_TER': {
         'dir':     r"E:\000000------文章1\数据库\GCB2025\ter_annual",
         'pattern': "GCB_TER_*.tif",
         'label':   'GCB_TER',
    },
'''
VARIABLES = {
     'GCB_TER': {
         'dir':     r"E:\000000------文章1\数据库\GCB2025\ter_annual",
         'pattern': "GCB_TER_*.tif",
         'label':   'GCB_TER',
    },
}

LUCC_PATH      = r"D:\00---全球NBP\Modis_LUCC_10_2_2023.tif"
TARGET_VALUES  = [6, 7, 9, 10]  

OUTPUT_CSV     = r"D:\00---全球NBP\GrassShrub_CarbonFlux_AllVars_GCB_TER.csv"

REGIONS = {
    'Global':   [(-90, 90)],            
    'HighNorth': [(56, 90)],
    'MidLat':   [(-60, -15), (15, 56)],  
}

UNIT_FACTOR = 1e-9


def build_grid(resolution: float = 1.0):

    nlon = int(360 / resolution)
    nlat = int(180 / resolution)

    lons = np.linspace(-180 + resolution / 2,  180 - resolution / 2, nlon)
    lats = np.linspace( 90  - resolution / 2, -90  + resolution / 2, nlat)

    R        = 6371.0
    dlon_rad = np.deg2rad(resolution)
    lat_hi   = np.deg2rad(lats + resolution / 2)
    lat_lo   = np.deg2rad(lats - resolution / 2)
    row_area = R**2 * dlon_rad * (np.sin(lat_hi) - np.sin(lat_lo))  # (nlat,)
    area_km2 = np.repeat(row_area[:, None], nlon, axis=1)     # (nlat, nlon)

    return lats, area_km2, nlat, nlon


def build_lucc_mask(lucc_path: str, target_values: list,
                    nlat: int, nlon: int) -> np.ndarray:
    with rasterio.open(lucc_path) as src:
        lucc = src.read(1)

    if lucc.shape != (nlat, nlon):
        raise ValueError(
        )

    mask = np.isin(lucc, target_values)
    return mask


def build_region_masks(lats: np.ndarray, nlon: int, regions: dict) -> dict:
    
    lat_col = lats[:, None]   # (nlat, 1)
    nlat = len(lats)
    masks = {}
    for name, bounds in regions.items():
        if bounds is None:
            masks[name] = np.ones((nlat, nlon), dtype=bool)
            continue
        if isinstance(bounds, tuple):
            bounds = [bounds]
        mask = np.zeros((nlat, nlon), dtype=bool)
        
        for lat_min, lat_max in bounds:
            mask |= (lat_col >= lat_min) & (lat_col < lat_max)
        
        masks[name] = mask
    
    return masks

def extract_year(filename: str) -> int:
    stem = os.path.splitext(os.path.basename(filename))[0]
    for part in reversed(stem.split('_')):
        if part.isdigit() and len(part) == 4:
            return int(part)
    raise ValueError(f"error: {filename}")


def process_variable(var_key: str, var_cfg: dict,
                     lucc_mask: np.ndarray,
                     area_km2: np.ndarray,
                     region_masks: dict,
                     nlat: int, nlon: int) -> pd.DataFrame:

    search = os.path.join(var_cfg['dir'], var_cfg['pattern'])
    files  = sorted(glob.glob(search))

    if not files:
        print(f"  ⚠️  [{var_key}] nofile: {search}")
        return pd.DataFrame()

    region_names = list(region_masks.keys())
    col_names    = [f"{var_key}_{r}" for r in region_names]

    rows = []
    for fp in files:
        try:
            year = extract_year(fp)
        except ValueError as e:
            print(f"    pass: {e}")
            continue

        with rasterio.open(fp) as src:
            data = src.read(1).astype(float)

        if data.shape != (nlat, nlon):
            print(f"   {year}: {data.shape} wrong")
            continue

        valid_base = lucc_mask & np.isfinite(data)

        row = {'Year': year}
        for rname, rmask in region_masks.items():
            valid = valid_base & rmask
            total = float(np.sum(data[valid] * area_km2[valid])) * UNIT_FACTOR
            row[f"{var_key}_{rname}"] = round(total, 8)

        rows.append(row)

        vals = " | ".join(f"{r}={row[f'{var_key}_{r}']:+.4f}" for r in region_names)
        print(f"    {year}: {vals}")

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).sort_values('Year').reset_index(drop=True)



def main():

    lats, area_km2, nlat, nlon = build_grid(resolution=1.0)

    lucc_mask = build_lucc_mask(LUCC_PATH, TARGET_VALUES, nlat, nlon)

    region_masks = build_region_masks(lats, nlon, REGIONS)
    for rname, bounds in REGIONS.items():
        px = region_masks[rname].sum()
        print(f"  {rname:<12}: {str(bounds):<15}  像元数={px:,}")

    df_all = None

    for var_key, var_cfg in VARIABLES.items():
        df_var = process_variable(
            var_key, var_cfg,
            lucc_mask, area_km2, region_masks,
            nlat, nlon
        )
        if df_var.empty:
            print(f"  [{var_key}] pass")
            continue

        if df_all is None:
            df_all = df_var
        else:
            df_all = pd.merge(df_all, df_var, on='Year', how='outer')

    if df_all is None or df_all.empty:
        return

    df_all = df_all.sort_values('Year').reset_index(drop=True)


    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 200)
    pd.set_option('display.float_format', '{:+.6f}'.format)
    print(df_all.to_string(index=False))

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    df_all.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig',
                  float_format='%.8f')
    print(f"\n saved: {OUTPUT_CSV}")

    print("\n" + "=" * 70)
    print("（mean ± std，PgC yr⁻¹）")
    print("=" * 70)
    num_cols = [c for c in df_all.columns if c != 'Year']
    summary_rows = []
    for col in num_cols:
        s = df_all[col].dropna()
        summary_rows.append({
            'Column':  col,
            'N':       len(s),
            'Mean':    f"{s.mean():+.6f}",
            'Std':     f"{s.std():.6f}",
            'Min':     f"{s.min():+.6f}",
            'Max':     f"{s.max():+.6f}",
        })
    df_summary = pd.DataFrame(summary_rows)
    print(df_summary.to_string(index=False))

    summary_path = OUTPUT_CSV.replace('.csv', '_summary_GCB_NEP+NBP.csv')
    df_summary.to_csv(summary_path, index=False, encoding='utf-8-sig')

if __name__ == "__main__":
    main()