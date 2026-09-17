import os
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.crs import CRS

def calculate_yearly_lsu_with_nep_mask():

    base_dir = r"E:\000000------文章1\数据库\Density\Resampled_1deg"
    nep_ref_path = r"E:\000000------文章1\数据库\GCAS_NEP\annual\NEP_2015.tif"
    out_dir = r"E:\000000------文章1\数据库\Density\LSU_Final_1deg"
    
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    with rasterio.open(nep_ref_path) as src_nep:
        dst_shape = src_nep.shape          
        dst_transform = src_nep.transform  
        dst_crs = src_nep.crs              
        
        nep_data = src_nep.read(1)
        nep_nodata = src_nep.nodata if src_nep.nodata is not None else -9999.0
        
        nep_valid_mask = (nep_data != nep_nodata) & (~np.isnan(nep_data))

    dst_nodata = -9999.0
    
    dst_meta = {
        'driver': 'GTiff',
        'dtype': 'float32',
        'nodata': dst_nodata,
        'width': dst_shape[1],
        'height': dst_shape[0],
        'count': 1,
        'crs': dst_crs,
        'transform': dst_transform,
        'compress': 'lzw'
    }

    weights = {
        'cattle': 1.0,
        'buffalo': 1.0,
        'horse': 0.65,
        'sheep': 0.1,
        'goat': 0.1
    }

    years = range(2015, 2023)
    
    for year in years:

        lsu_total = np.zeros(dst_shape, dtype=np.float32)
        
        for animal, weight in weights.items():
            file_name = f"gpw_{animal}.density_rf_m_1km_s_{year}0101_{year}1231_go_esri.54052_v1.tif"
            file_path = os.path.join(base_dir, file_name)
            
            if not os.path.exists(file_path):
                print(f" nofile: {file_name}")
                continue
                
            with rasterio.open(file_path) as src:

                animal_1deg = np.full(dst_shape, dst_nodata, dtype=np.float32)
                src_crs = src.crs if src.crs else CRS.from_epsg(4326)
                src_nodata = src.nodata if src.nodata is not None else -9999.0
                reproject(
                    source=rasterio.band(src, 1),
                    destination=animal_1deg,
                    src_transform=src.transform,
                    src_crs=src_crs,
                    dst_transform=dst_transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.average,
                    src_nodata=src_nodata,
                    dst_nodata=dst_nodata
                )
                
                valid_pixels = (animal_1deg != dst_nodata) & (~np.isnan(animal_1deg))
                animal_1deg[~valid_pixels] = 0.0

                lsu_total += animal_1deg * weight

        lsu_total[~nep_valid_mask] = dst_nodata
        
        out_path = os.path.join(out_dir, f"LSU_density_1deg_{year}.tif")
        with rasterio.open(out_path, 'w', **dst_meta) as dst:
            dst.write(lsu_total, 1)

if __name__ == "__main__":
    calculate_yearly_lsu_with_nep_mask()