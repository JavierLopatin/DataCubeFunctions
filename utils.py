from datacube.utils import masking
import xarray as xr

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