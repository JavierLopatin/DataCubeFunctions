import numpy as np
import matplotlib.pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar
from matplotlib import colors
import ipywidgets as widgets
from IPython.display import display
import folium
from pyproj import Proj, transform
from odc.ui import image_aspect
import warnings


def display_map(x, y, crs='EPSG:4326', margin=-0.5, zoom_bias=0):
    """ 
    Given a set of x and y coordinates, this function generates an 
    interactive map with a bounded rectangle overlayed on Google Maps 
    imagery. Based on DEAfrica notebooks       
    
    Last modified: September 2019
    
    Modified from function written by Otto Wagner available here: 
    https://github.com/ceos-seo/data_cube_utilities/tree/master/data_cube_utilities
    
    Parameters
    ----------  
    x : (float, float)
        A tuple of x coordinates in (min, max) format. 
    y : (float, float)
        A tuple of y coordinates in (min, max) format.
    crs : string, optional
        A string giving the EPSG CRS code of the supplied coordinates. 
        The default is 'EPSG:4326'.
    margin : float
        A numeric value giving the number of degrees lat-long to pad 
        the edges of the rectangular overlay polygon. A larger value 
        results more space between the edge of the plot and the sides 
        of the polygon. Defaults to -0.5.
    zoom_bias : float or int
        A numeric value allowing you to increase or decrease the zoom 
        level by one step. Defaults to 0; set to greater than 0 to zoom 
        in, and less than 0 to zoom out.
        
    Returns
    -------
    folium.Map : A map centered on the supplied coordinate bounds. A 
    rectangle is drawn on this map detailing the perimeter of the x, y 
    bounds.  A zoom level is calculated such that the resulting 
    viewport is the closest it can possibly get to the centered 
    bounding rectangle without clipping it. 
    """
    
    def _degree_to_zoom_level(l1, l2, margin=0.0):
        """
        Helper function to set zoom level for `display_map`
        """
        degree = abs(l1 - l2) * (1 + margin)
        zoom_level_int = 0
        if degree != 0:
            zoom_level_float = math.log(360 / degree) / math.log(2)
            zoom_level_int = int(zoom_level_float)
        else:
            zoom_level_int = 18
        return zoom_level_int

    # Convert each corner coordinates to lat-lon
    all_x = (x[0], x[1], x[0], x[1])
    all_y = (y[0], y[0], y[1], y[1])
    all_longitude, all_latitude = transform(Proj(crs),
                                            Proj('EPSG:4326'),
                                            all_x, all_y)

    # Calculate zoom level based on coordinates
    lat_zoom_level = _degree_to_zoom_level(min(all_latitude),
                                           max(all_latitude),
                                           margin=margin) + zoom_bias
    lon_zoom_level = _degree_to_zoom_level(min(all_longitude),
                                           max(all_longitude),
                                           margin=margin) + zoom_bias
    zoom_level = min(lat_zoom_level, lon_zoom_level)

    # Identify centre point for plotting
    center = [np.mean(all_latitude), np.mean(all_longitude)]

    # Create map
    interactive_map = folium.Map(
        location=center,
        zoom_start=zoom_level,
        tiles="http://mt1.google.com/vt/lyrs=y&z={z}&x={x}&y={y}",
        attr="Google")

    # Create bounding box coordinates to overlay on map
    line_segments = [(all_latitude[0], all_longitude[0]),
                     (all_latitude[1], all_longitude[1]),
                     (all_latitude[3], all_longitude[3]),
                     (all_latitude[2], all_longitude[2]),
                     (all_latitude[0], all_longitude[0])]

    # Add bounding box as an overlay
    interactive_map.add_child(
        folium.features.PolyLine(locations=line_segments,
                                 color='red',
                                 opacity=0.8))

    # Add clickable lat-lon popup box
    interactive_map.add_child(folium.features.LatLngPopup())

    return interactive_map

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
    """
    This function creates an interactive RGB plot for the given image.
    The user can select the red, green, and blue bands to be plotted using dropdown menus.
    The plot is displayed when the user clicks the "Plot" button.

    Parameters:
    image (xarray.Dataset): The image to be plotted.

    Returns:
    None
    """

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