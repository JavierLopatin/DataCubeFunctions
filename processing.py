import xarray as xr
import numpy as np

def calculate_percentiles(ds):
    """
    Calculate the 0.05, 0.15, 0.5, 0.75, and 0.95 percentiles of an xarray dataset.
    ds: xarray dataset
    """
    # Check if the 'time' dimension is chunked into multiple parts
    if len(ds.chunks['time']) > 1:
        # If so, rechunk the dataset so that the entire 'time' dimension is in a single chunk
        ds = ds.chunk({'time': -1})

    # Calculate the percentiles for each data variable separately
    percentile_data_vars = {var: ds[var].quantile([0.05, 0.15, 0.5, 0.75, 0.95], dim='time') for var in ds.data_vars}

    # Combine the results into a single xarray Dataset
    percentiles = xr.Dataset(percentile_data_vars)

    return percentiles


def Sentinel2_calculate_indices(img):
    """
    Calculate NDVI, EVI, kNDVI, CIG, and ANDWI from an xarray object.
    img: xarray object
        img needs to have data variables called B02, B03, B04, B08, B11, and B12.
    """
    img['ndvi'] = (img['B08'] - img['B04']) / (img['B08'] + img['B04'])
    img['evi'] = 2.5 * ((img['B08'] - img['B04']) / (img['B08'] + 6.0 * img['B04'] - 7.5 * img['B02'] + 1.0))
    img['cig'] = (img['B08'] / img['B03']) - 1.0
    img['andwi'] = (img['B02']+img['B03']+img['B04']-img['B08']-img['B11']-img['B12']) / (img['B02']+img['B03']+img['B04']+img['B08']+img['B11']+img['B12'])
    sigma = 0.5*(img['B08'] + img['B04'])
    knr = np.exp(-(img['B08'] - img['B04'])**2/(2*sigma**2))
    img['kndvi'] = (1-knr) / (1+knr)

    return img