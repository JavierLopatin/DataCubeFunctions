import rasterio
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

def Sentinel2_TasseledCap(img):
    """
    Calculate brightness, greenness, and wetness from an xarray object.
    img: xarray object
        img needs to have data variables called B02, B03, B04, B08, B11, and B12.
    """
    img['Brightness'] = 0.3510*img['B02'] + 0.3813*img['B03'] + 0.3437*img['B04'] + 0.7196*img['B08'] + 0.2396*img['B11'] + 0.1949*img['B12']
    img['Greeness'] = -0.3599*img['B02'] - 0.3533*img['B03'] - 0.4734*img['B04'] + 0.6633*img['B08'] + 0.0087*img['B11'] - 0.2856*img['B12']
    img['Wetness'] = 0.2578*img['B02'] + 0.2305*img['B03'] + 0.0883*img['B04'] + 0.1071*img['B08'] - 0.7611*img['B11'] - 0.5308*img['B12']

    return img

def rasterize_xarray(data_array: xr.DataArray, shapefile: gpd.GeoDataFrame, column: str) -> xr.DataArray:
    """
    Rasterizes a GeoDataFrame into an xarray DataArray using the same coordinates as the chosen DataArray.

    Parameters:
    -----------
    data_array : xr.DataArray
        The DataArray to use for the coordinates and dimensions of the output raster.
    shapefile : gpd.GeoDataFrame
        The GeoDataFrame to be rasterized.
    column : str
        The name of the column in the GeoDataFrame to use for the values of the raster.

    Returns:
    --------
    xr.DataArray
        The rasterized GeoDataFrame as an xarray DataArray.
    """
    # Convert the specified column of the GeoDataFrame to a list of shapes
    shapes = [(geom, value) for geom, value in zip(shapefile.geometry, shapefile[column])]

    # Rasterize the shapes using the same coordinates as the chosen DataArray
    code2_raster = rasterio.features.rasterize(shapes, out_shape=data_array.shape, transform=data_array.rio.transform(), fill=np.nan)

    # Convert the raster to an xarray DataArray
    return xr.DataArray(code2_raster, coords=data_array.coords, dims=data_array.dims)