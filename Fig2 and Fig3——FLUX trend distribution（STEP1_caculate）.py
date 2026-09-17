
import numpy as np
import pymannkendall as mk
import os
from osgeo import gdal

gdal.UseExceptions()


def read_tif(filepath):

    ds = gdal.Open(filepath, gdal.GA_ReadOnly)
    if ds is None:
        raise RuntimeError(f"GDALerror: {filepath}")

    try:
        cols = ds.RasterXSize
        rows = ds.RasterYSize
        geotrans = ds.GetGeoTransform()
        proj = ds.GetProjection()
        band = ds.GetRasterBand(1)
        if band is None:
            raise RuntimeError(f"noBAND: {filepath}")

        data = band.ReadAsArray()
        if data is None:
            raise RuntimeError(f"nodata: {filepath}")

        data = data.astype(np.float32)

        nodata = band.GetNoDataValue()
        if nodata is not None:
            data[data == nodata] = np.nan
        else:
            data[data < -9000] = np.nan

        return cols, rows, geotrans, proj, data
    finally:
        ds = None


def save_tif(data, reference_file, output_file, nodata_value=-9999):

    ds_ref = gdal.Open(reference_file, gdal.GA_ReadOnly)
    if ds_ref is None:
        raise RuntimeError(f"unopen: {reference_file}")

    rows, cols = data.shape
    driver = gdal.GetDriverByName("GTiff")
    out_ds = driver.Create(output_file, cols, rows, 1, gdal.GDT_Float32,
                           options=["COMPRESS=LZW"])
    out_ds.SetGeoTransform(ds_ref.GetGeoTransform())
    out_ds.SetProjection(ds_ref.GetProjection())
    out_band = out_ds.GetRasterBand(1)

    write_arr = np.where(np.isnan(data), nodata_value, data)
    out_band.WriteArray(write_arr.astype(np.float32))
    out_band.SetNoDataValue(nodata_value)
    out_band.FlushCache()
    out_ds = None
    ds_ref = None


def get_tif_list(directory, pattern=""):

    files = [f for f in os.listdir(directory) if f.lower().endswith('.tif')]
    if pattern:
        files = [f for f in files if pattern in f]
    files.sort() 
    return [os.path.join(directory, f) for f in files]


def process_mk_trend(data_array):

    years, rows, cols = data_array.shape
    slope_data = np.full((rows, cols), np.nan, dtype=np.float32)
    tau_data = np.full((rows, cols), np.nan, dtype=np.float32)
    p_data = np.full((rows, cols), np.nan, dtype=np.float32)
    z_data = np.full((rows, cols), np.nan, dtype=np.float32)
    s_data = np.full((rows, cols), np.nan, dtype=np.float32)


    for i in range(rows):
        if i % 20 == 0:
            print(f"stage: {i}/{rows} ({i/rows*100:.1f}%)")
        for j in range(cols):
            ts = data_array[:, i, j]
            valid_mask = ~np.isnan(ts)
            if np.sum(valid_mask) >= 4:
                valid_data = ts[valid_mask]
                try:
                    res = mk.original_test(valid_data)
                    slope_data[i, j] = getattr(res, 'slope', np.nan)
                    tau_data[i, j] = getattr(res, 'Tau', getattr(res, 'tau', np.nan))
                    p_data[i, j] = getattr(res, 'p', np.nan)
                    z_data[i, j] = getattr(res, 'z', np.nan)
                    s_data[i, j] = getattr(res, 's', np.nan)
                except Exception as e:

                    continue

    return slope_data, tau_data, p_data, z_data, s_data


if __name__ == '__main__':
    base_directory = r"D:/00_GLB_NEE/0000000000000000---MK趋势检验"

    variables = {

        'NEP': 'nep',
        'GPP': 'gpp',
        'TER': 'ter',
    }

    output_base = os.path.join(base_directory, "MK_results")
    os.makedirs(output_base, exist_ok=True)

    for var_name, folder_name in variables.items():
        input_directory = os.path.join(base_directory, folder_name)
        if not os.path.exists(input_directory):
            continue

        file_list = get_tif_list(input_directory)
        if len(file_list) == 0:
            continue

        try:
            cols, rows, geotrans, proj, _ = read_tif(file_list[0])
        except Exception as e:
            continue

        years = len(file_list)

        data_array = np.full((years, rows, cols), np.nan, dtype=np.float32)

        valid_file_count = 0
        for idx, filepath in enumerate(file_list):
            try:
                _, _, _, _, data = read_tif(filepath)
                if data.shape != (rows, cols):
                    raise RuntimeError(f"error: {filepath}")
                data_array[idx, :, :] = data
                valid_file_count += 1

            except Exception as e:
                print(f"pass: {filepath} -> {e}")

        if valid_file_count < 4:
            print(f"no enough {valid_file_count})")
            continue

        slope_data, tau_data, p_data, z_data, s_data = process_mk_trend(data_array)

        output_var_dir = os.path.join(output_base, var_name)
        os.makedirs(output_var_dir, exist_ok=True)

        save_tif(slope_data, file_list[0], os.path.join(output_var_dir, f"{var_name}_Sen_Slope.tif"))
        save_tif(tau_data, file_list[0], os.path.join(output_var_dir, f"{var_name}_Tau.tif"))
        save_tif(p_data, file_list[0], os.path.join(output_var_dir, f"{var_name}_P_value.tif"))
        save_tif(z_data, file_list[0], os.path.join(output_var_dir, f"{var_name}_Z_value.tif"))
        save_tif(s_data, file_list[0], os.path.join(output_var_dir, f"{var_name}_S_statistic.tif"))
