"""
@author: igor
"""
# Adapted from: https://unidata.github.io/MetPy/latest/examples/plots/Station_Plot.html

import matplotlib.pyplot as plt                                 # Plotting library
import cartopy, cartopy.crs as ccrs                             # Plot maps
import cartopy.io.shapereader as shpreader                      # Import shapefiles
import cartopy.feature as cfeature                              # Common drawing and filtering operations
# import os                                                       # Miscellaneous operating system interfaces
import numpy as np                                              # Scientific computing with Python
# import requests                                                 # HTTP library for Python
from datetime import timedelta, date, datetime                  # Basic Dates and time types
from metpy.calc import reduce_point_density                     # Provide tools for unit-aware, meteorological calculations    
from metpy.io import metar                                      # Parse METAR-formatted data
from metpy.plots import current_weather, sky_cover, StationPlot # Contains functionality for making meteorological plots
# import pygrib                                                   # Provides a high-level interface to the ECWMF ECCODES C library for reading GRIB files
import xarray as xr                   ######  importando xarray
from cartopy.feature import ShapelyFeature
from cartopy.io.shapereader import Reader

#-----------------------------------------------------------------------------------------------------------

def plot_maxmin_points(lon, lat, data, extrema, nsize, symbol, color='k',
                       plotValue=True, transform=None):
   
    from scipy.ndimage.filters import maximum_filter, minimum_filter

    if (extrema == 'max'):
        data_ext = maximum_filter(data, nsize, mode='nearest')
    elif (extrema == 'min'):
        data_ext = minimum_filter(data, nsize, mode='nearest')
    else:
        raise ValueError('Value for hilo must be either max or min')

    mxy, mxx = np.where(data_ext == data)

    for i in range(len(mxy)):
        txt1 = ax.annotate(symbol, xy=(lon[mxy[i], mxx[i]], lat[mxy[i], mxx[i]]), xycoords=ccrs.PlateCarree()._as_mpl_transform(ax), color=color, size=24,
                clip_on=True, annotation_clip=True, horizontalalignment='center', verticalalignment='center',
                transform=ccrs.PlateCarree()) 

        txt2 = ax.annotate('\n' + str(int(data[mxy[i], mxx[i]])), xy=(lon[mxy[i], mxx[i]], lat[mxy[i], mxx[i]]), xycoords=ccrs.PlateCarree()._as_mpl_transform(ax), 
                color=color, size=12, clip_on=True, annotation_clip=True, fontweight='bold', horizontalalignment='center', verticalalignment='top',
                transform=ccrs.PlateCarree()) 

#----------------------------------------------------------------------------------------------------------- 

# # Select the extent [min. lon, min. lat, max. lon, max. lat]
extent = [-93.0, -60.00, -25.00, 18.00]

# #-----------------------------------------------------------------------------------------------------------
ds = xr.open_dataset('/media/ladsin/IGOR06_64/ETAPA7/slp.mean.nc')
campo2D = ds.sel(time='2022-04-11 00:00:00')
lons,lats = np.meshgrid(ds.lon,ds.lat)
# Convert to hPa
prmls = campo2D['slp'] / 100

# url = 'https://thredds-test.unidata.ucar.edu/thredds/fileServer/noaaport/text/metar' 
# METAR File
# https://unidata.github.io/MetPy/latest/examples/plots/Station_Plot.html
data = metar.parse_metar_file('/media/ladsin/IGOR06_64/ETAPA7/metar_20220411_0000.txt')

# =============================================================================
# # Drop rows with missing winds
# =============================================================================
data = data.dropna(how='any', subset=['wind_direction', 'wind_speed'])

# =============================================================================
# # Choose the plot size (width x height, in inches)
# =============================================================================
plt.figure(figsize=(10,10))

# =============================================================================
# # Set up the map projection
# =============================================================================
proj = ccrs.PlateCarree()

# =============================================================================
# # Use the Geostationary projection in cartopy
# =============================================================================
ax = plt.axes(projection=proj)

# =============================================================================
# # Define the image extent
# =============================================================================
img_extent = [extent[0], extent[2], extent[1], extent[3]]
ax.set_extent([extent[0], extent[2], extent[1], extent[3]], ccrs.PlateCarree())

# Change the DPI of the resulting figure. Higher DPI drastically improves the
# =============================================================================
# # look of the text rendering.
# =============================================================================
plt.rcParams['savefig.dpi'] = 255

# Use the Cartopy map projection to transform station locations to the map and
# =============================================================================
# # then refine the number of stations plotted by setting a minimum radius
# =============================================================================
point_locs = proj.transform_points(ccrs.PlateCarree(), data['longitude'].values, data['latitude'].values)
data = data[reduce_point_density(point_locs, 3)]

# =============================================================================
# # Add some various map elements to the plot to make it recognizable.
# =============================================================================
ax.add_feature(cfeature.LAND)
ax.add_feature(cfeature.OCEAN)

# =============================================================================
# # Define de contour interval
# =============================================================================
data_min = 500
data_max = 1050
interval = 2
levels = np.arange(data_min,data_max,interval)

# =============================================================================
# # Plot the contours
# =============================================================================
img1 = ax.contour(lons, lats, prmls, colors='black', linewidths=0.7, levels=levels)
ax.clabel(img1, inline=1, inline_spacing=0, fontsize='10',fmt = '%1.0f', colors= 'black')

# =============================================================================
# # Use definition to plot H/L symbols
# =============================================================================
plot_maxmin_points(lons, lats, prmls, 'max', 25, symbol='A', color='b',  transform=ccrs.PlateCarree())
plot_maxmin_points(lons, lats, prmls, 'min', 18, symbol='B', color='r', transform=ccrs.PlateCarree())

# =============================================================================
# # Add a shapefile
# =============================================================================
mapa_amsul = ShapelyFeature(Reader('/media/ladsin/IGOR06_64/CURSO_PYTHON_RONALDO/amsulrp2.shp').geometries(),
                            ccrs.PlateCarree(),
                            edgecolor='black',
                            linewidth=0.8,
                            facecolor='None')
ax.add_feature(mapa_amsul)
# =============================================================================
# # Add coastlines, borders and gridlines
# =============================================================================
ax.coastlines(resolution='10m', color='black', linewidth=0.8)
ax.add_feature(cartopy.feature.BORDERS, edgecolor='black', linewidth=0.5)
gl = ax.gridlines(crs=ccrs.PlateCarree(), color='white', alpha=1.0, linestyle='--', linewidth=0.25, xlocs=np.arange(-180, 180, 5), ylocs=np.arange(-90, 90, 5), draw_labels=True)
gl.top_labels = False
gl.right_labels = False

#-----------------------------------------------------------------------------------------------------------
# =============================================================================
# # Station Plot 
# =============================================================================

# =============================================================================
# # Start the station plot by specifying the axes to draw on, as well as the lon/lat of the stations (with transform). We also the fontsize to 12 pt.
# =============================================================================
stationplot = StationPlot(ax, data['longitude'].values, data['latitude'].values,
                          clip_on=True, transform=ccrs.PlateCarree(), fontsize=9)

# =============================================================================
# # Plot the temperature and dew point to the upper and lower left, respectively, of the center point. Each one uses a different color.
# =============================================================================
stationplot.plot_parameter('NW', data['air_temperature'].values, color='red')
stationplot.plot_parameter('SW', data['dew_point_temperature'].values,color='darkgreen')

# A more complex example uses a custom formatter to control how the sea-level pressure
# values are plotted. This uses the standard trailing 3-digits of the pressure value
# =============================================================================
# # in tenths of millibars.
# =============================================================================
stationplot.plot_parameter('NE', data['air_pressure_at_sea_level'].values,
                           formatter=lambda v: format(5 * v, '.0f')[-3:])

# Plot the cloud cover symbols in the center location. This uses the codes made above and uses the `sky_cover` mapper to convert these values to font codes for the
# =============================================================================
# # weather symbol font.
# =============================================================================
stationplot.plot_symbol('C', data['cloud_coverage'].values, sky_cover)
# =============================================================================
# # Same this time, but plot current weather to the left of center, using the
# =============================================================================
# stationplot.plot_symbol('W', data['present_weather'].values, current_weather)
# =============================================================================
# # Add wind barbs
# =============================================================================
stationplot.plot_barb(data['eastward_wind'].values, data['northward_wind'].values)
# =============================================================================
# # Also plot the actual text of the station id. Instead of cardinal directions,
# =============================================================================
# stationplot.plot_text((2, 0), data['station_id'].values)
# =============================================================================
# #-----------------------------------------------------------------------------------------------------------
# =============================================================================
ax.set_title("METAR  | " + "2022-04-11" + " 00:00 UTC")
# Save the image
# plt.savefig(f'{output}/image_26.png', bbox_inches='tight', pad_inches=0, dpi=300)

plt.show()