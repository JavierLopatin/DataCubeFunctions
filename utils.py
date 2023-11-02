from datacube.utils import masking
import xarray as xr
import rasterio
import rasterio.features


def Sentinel2_cloudMask(ds, valid_threshold=0.7):
    """
    Create a cloud mask for Sentinel 2 data. 
    This function is used in the Sentinel2 cloud masking notebook.

    ds: xarray dataset
        ds needs to have a data variable called SCL
    valid_threshold: float
        The threshold for the proportion of good pixels. Default is 0.7.
    """

    # Check if the dataset contains the required data variable
    if 'SCL' not in ds.data_vars:
        raise ValueError("The dataset does not contain the required 'SCL' data variable.")

    # Create a cloud mask
    cloud_free_mask = (
        masking.make_mask(ds.SCL, qa="vegetation") | 
        masking.make_mask(ds.SCL, qa="bare soils") |
        masking.make_mask(ds.SCL, qa="water") |
        masking.make_mask(ds.SCL, qa="snow or ice")
    )

    # Calculate proportion of good pixels
    valid_pixel_proportion = cloud_free_mask.sum(dim=("x", "y"))/(cloud_free_mask.shape[1] * cloud_free_mask.shape[2])

    # Identify observations to keep based on the valid pixel proportion threshold
    observations_to_keep = (valid_pixel_proportion >= valid_threshold)

    # Mask the data
    ds_s2_valid = ds.where(cloud_free_mask)

    # Only keep observations above the good pixel proportion threshold
    # The .compute() step means the values will be loaded into memory. This step may take some time
    ds_s2_keep = ds_s2_valid.sel(time=observations_to_keep)#.compute()

    return ds_s2_keep.drop_vars('SCL')# Drop the 'SCL' data variable


def get_noneNaN_dates(img, dims=('x', 'y')):
    """Check if an xarray dataset contains NaN values.

    img: xarray dataset
        output is the dates of free cloud (no NaN values) observations
    dims: tuple of strings, optional
        dimensions to check for NaN values, default is ('x', 'y')
    """
    # Check for non-NaN values across all dimensions except 'time' and specified dimensions
    non_nan_mask = img.notnull().all(dim=tuple(set(img.dims) - {'time'} - set(dims)))
    # Get the indices of 'time' dimension where there are no NaN values
    return non_nan_mask.where(non_nan_mask).dropna(dim='time')['time'].values


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

# Convert the xarray Dataset 2D
def xr_dataVar_2flat(ds):
    """
    Stack the x and y dimensions of a xarray dataset and convert it to a 2D numpy array.
    
    Parameters:
    -----------
    ds : xarray.Dataset
        The input dataset to be flattened.
    
    Returns:
    --------
    img_flat : xarray.DataArray
        The flattened dataset as a 2D numpy array.
    """
    img_flat = ds.stack(z=('y','x'))
    img_flat = img_flat.to_array(dim='variable')
    return img_flat.transpose("z", "variable")


def collapse_dim_to_variable(ds, dim):
    """
    Collapse a dimension of an xarray Dataset into data variables.
    All variables are repeted and stored i Data Variables acording to the values of the dimension.
    For example, if the dimension is 'time' and the Dataset has 3 variables, the output will have 3*len(ds['time']) variables.

    ds: xarray Dataset
    dim: string
        The name of the dimension to collapse.
    """
    # Create a dictionary to hold the new data variables
    new_data_vars = {}

    # Loop over each data variable in the original Dataset
    for var in ds.data_vars:
        # Loop over each value of the specified dimension
        for value in ds[dim]:
            # Create a new data variable for each value of the specified dimension
            new_var_name = f'{dim}{int(value.values*100)}_{var}'
            new_data_vars[new_var_name] = ds[var].sel({dim: value}).drop(dim)

    # Create a new xarray Dataset with the new data variables
    new_ds = xr.Dataset(new_data_vars)
    crs= ds.rio.crs
    new_ds.rio.write_crs(crs, inplace=True)
    
    return new_ds
