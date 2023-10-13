import numpy as np
import matplotlib.pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar
from matplotlib import colors

def plot_classes(class_map, n_clusters, col=None):
    """
    Plot a prediction array with specified number of clusters and color palette.
    
    Parameters:
    - prediction_clip (xarray.DataArray): The prediction data to be plotted.
    - n_clusters (int): The number of unique classes or clusters in the data.
    - col (list of str, optional): A list of color codes to be used as colormap.
                                   If not provided, the default Matplotlib colormap will be used.
    
    Returns:
    - None. It will display a figure with the plotted data.
    """
    
    if col:
        cmap = colors.ListedColormap(col)
    else:
        cmap = plt.get_cmap('viridis', n_clusters)
    
    # Adjust the boundaries to center each class value
    boundaries = [i + 0.5 for i in range(n_clusters+1)]
    norm = colors.BoundaryNorm(boundaries, cmap.N, clip=True)
     
    fig, ax = plt.subplots()
    # Extract x and y coordinates from the xarray DataArray
    x = class_map.coords['x'].values
    y = class_map.coords['y'].values[::-1]  # Reverse the y-coordinates
    X, Y = np.meshgrid(x, y)

    plt.pcolormesh(X, Y, class_map[::-1, :], cmap=cmap, norm=norm)  # Reverse the data in the y-dimension

    # change title
    plt.title('')

    # delete axis titles
    plt.xlabel('')
    plt.ylabel('')
    plt.colorbar(ticks=range(1, n_clusters+1))

    # Add scale bar to the bottom left
    scalebar = ScaleBar(1, location=3)
    ax.add_artist(scalebar)