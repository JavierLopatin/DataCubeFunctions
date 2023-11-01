import numpy as np
import matplotlib.pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar
from matplotlib import colors
import ipywidgets as widgets
from IPython.display import display


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
    
    
    
def interactive_RGB_Plot(image):
    band_options = list(image.data_vars.keys())

    r_band_selector = widgets.Dropdown(options=band_options, description="Red Band:")
    g_band_selector = widgets.Dropdown(options=band_options, description="Green Band:")
    b_band_selector = widgets.Dropdown(options=band_options, description="Blue Band:")

    # Setting default values
    if len(band_options) >= 3:
        r_band_selector.value = band_options[0]
        g_band_selector.value = band_options[1]
        b_band_selector.value = band_options[2]

    plot_button = widgets.Button(description="Plot")

    def visualize_image(button):
        plt.figure()
        image[[r_band_selector.value, g_band_selector.value, b_band_selector.value]].to_array().plot.imshow(robust=True)
        plt.title(f'RGB: {r_band_selector.value}, {g_band_selector.value}, {b_band_selector.value}')
        plt.show()

    plot_button.on_click(visualize_image)

    display(r_band_selector, g_band_selector, b_band_selector, plot_button)

def plot_normRGB(img, vmax=0.6):
    # normalizar entre 0 y 1
    img = (img - img.min()) / (img.max() - img.min())
    img.to_array().plot.imshow(robust=True, vmax=vmax)