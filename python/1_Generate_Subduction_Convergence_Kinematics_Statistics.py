# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.4.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# ## Step 1:  Generate Subduction Convergence Kinematics Statistics 
#
# In this notebook, we are going to show you how to generate subduction convergence kinematics statistics. The data generated in this step will be used in the subsequent steps of this spatial temporal exploration workflow.
#
# The script finds subduction zones using so-called "topological plate boundaries" in our global plate model. We sample the subduction zones and calculate subduction convergence kinematics statistics at each trench sampling point.
#
# The implementation details can be found in [convergence.py](convergence.py) which depends on the PlateTectonicTools package. You can find the package at [https://github.com/EarthByte/PlateTectonicTools.git](https://github.com/EarthByte/PlateTectonicTools.git).
#     
# The parameters being used to run this process can be found in [config.json](config.json).
#
# Relevant parameters:
# * plate_tectonic_tools_path -- the path to the PlateTectonicTools code
# * rotation_files -- location of the rotation files
# * topology_files -- location of the topology files
# * threshold_sampling_distance_degrees -- the default threshold sampling distance along trenches (subduction zones)
# * time.start -- start time
# * time.end -- end time
# * time.step -- time interval between steps
# * velocity_delta_time -- time interval for velocity calculation
# * anchor_plate_id - the anchor plate id
# * convergence_data_filename_prefix -- the prefix of the output files
# * convergence_data_filename_ext -- the extension name of the output files
# * convergence_data_dir -- the name of the folder in which the output files go
#
# You may modify the above parameters and re-run the script to see the differences. 
#
# Now, let's run the script and check out the output.

# +
#
# SEE HERE! 
# This cell must run first to setup the working environment
#

import glob, os
import convergence , Utils
import pandas as pd

#load the config file
Utils.load_config('config.json')
Utils.print_parameters()

# +
# %%capture --no-stdout
print('running convergence...')
print('this may take a while, be patient...')
print('')

#run the convergence script
#this will generate a bunch of Subduction Convergence Kinematics Statistics files
#by default the files are placed in ./convergence_data
convergence.run_it()

# now, let's list all the output files
files = sorted(glob.glob(Utils.get_convergence_dir() + '*'), key=os.path.getmtime)
print('The number of generated files: ', len(files))
print('The first 10 files:')
for i in range(10):
    print(files[i])
# -

# The above cell took a while to finish and created a number of csv files. Each file contains the subduction convergence kinematics statistics at certain time. For example, the file "subStats_230.00.csv" contains data at time 230Ma.
#
# Now, let's open one of the files and see what is inside.

time=0.
conv_dir = Utils.get_convergence_dir()
pd.read_csv(f"{conv_dir}{Utils.get_parameter('convergence_data_filename_prefix')}_\
{time:0.2f}.{Utils.get_parameter('convergence_data_filename_ext')}")


# There are 3029 rows in the csv file, which means there are 3029 trench sampling points. Each row contains data for each sampled point along trench. 
#
# There are 20 columns in the csv file. They are the subduction convergence kinematics statistics. The meaning of each column is listed below.
#
# * 0 longitude of sample point
# * 1 latitude of sample point
# * 2 subducting convergence (relative to trench) velocity magnitude (in cm/yr)
# * 3 subducting convergence velocity obliquity angle (angle between trench normal vector and convergence velocity vector)
# * 4 trench absolute (relative to anchor plate) velocity magnitude (in cm/yr)
# * 5 trench absolute velocity obliquity angle (angle between trench normal vector and trench absolute velocity vector)
# * 6 length of arc segment (in degrees) that current point is on
# * 7 trench normal azimuth angle (clockwise starting at North, ie, 0 to 360 degrees) at current point
# * 8 subducting plate ID
# * 9 trench plate ID
# * 10 distance (in degrees) along the trench line to the nearest trench edge
# * 11 the distance (in degrees) along the trench line from the start edge of the trench
# * 12 convergence velocity orthogonal component(in cm/yr)
# * 13 convergence velocity parallel component(in cm/yr) 
# * 14 the trench plate absolute velocity orthogonal component(in cm/yr)
# * 15 the trench plate absolute velocity parallel component(in cm/yr)
# * 16 the subducting plate absolute velocity magnitude (in cm/yr)
# * 17 the subducting plate absolute velocityobliquity angle (in degrees)
# * 18 the subducting plate absolute velocity orthogonal component       
# * 19 the subducting plate absolute velocity parallel component
#
# Now, let's draw some maps to visualize the data. The data visualization is important because it allows trends and patterns to be more easily seen.

# +
# %matplotlib inline

import requests, os, glob
import pygplates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cartopy.crs as ccrs
import matplotlib.ticker as mticker
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

trench_data = Utils.get_trench_points(0)

#get topological plates boundaries
time = 0
resolved_topologies = []
shared_boundary_sections = []

rotation_files = Utils.get_files(Utils.get_parameter('rotation_files'))
topology_files = Utils.get_files(Utils.get_parameter("topology_files"))
   
#use pygplates to resolve the topologies
pygplates.resolve_topologies(topology_files, rotation_files, resolved_topologies, time, 
                             shared_boundary_sections)

geoms = [t.get_resolved_boundary() for t in resolved_topologies]           

#now, plot the data in a global map    
fig = plt.figure(figsize=(16,12),dpi=150)
ax = plt.axes(projection=ccrs.PlateCarree())
ax.stock_img()
ax.set_extent([-180, 180, -90, 90])
for geom in geoms:
    lat, lon =zip(*(geom.to_lat_lon_list()))
    plt.plot(lon, lat,
         color='white', linewidth=.5, #the topological plates boundaries in white
         transform=ccrs.Geodetic(),
    )
    
gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                  linewidth=1, color='gray', alpha=0.5, linestyle='--')
gl.xlabels_top = False
gl.ylabels_right = False
gl.xlocator = mticker.FixedLocator(range(-180,180,30))
gl.ylocator = mticker.FixedLocator(range(-90,90,15))
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER
gl.xlabel_style = {'color': 'gray', 'weight': 'bold'}
gl.ylabel_style = {'color': 'gray', 'weight': 'bold'}

#the subduction sample points are colored by property value.
cb=ax.scatter(trench_data['trench_lon'], trench_data['trench_lat'], 30, marker='.', 
              c=trench_data['dist_nearest_edge']* 6371. * np.pi / 180, cmap=plt.cm.jet)
plt.title('Present-day Subduction Zones Coloured by Distance To Nearest Edge(km)')
fig.colorbar(cb, shrink=0.5, label='Distance To Nearest Edge(km)')
plt.show()
# -

# ##### Take a closer look at the region of interest

# +
# %matplotlib inline
import Utils

region_of_interest_polygon = Utils.get_region_of_interest_polygon()
#display(region_of_interest_polygon)

mesh_points = Utils.get_mesh_points(region_of_interest_polygon.values.flatten())

#now, plot the data in a regional map    
fig = plt.figure(figsize=(8,8),dpi=96)
ax = plt.axes(projection=ccrs.PlateCarree())
ax.stock_img()
ax.set_extent(Utils.get_region_of_interest_extent())

gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                  linewidth=1, color='gray', alpha=0.5, linestyle='--')
gl.xlabels_top = False
gl.ylabels_right = False
gl.xlocator = mticker.FixedLocator(range(-180,180,10))
gl.ylocator = mticker.FixedLocator(range(-90,90,10))
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER
gl.xlabel_style = {'color': 'gray', 'weight': 'bold', 'fontsize': '5'}
gl.ylabel_style = {'color': 'gray', 'weight': 'bold', 'fontsize': '5'}

for geom in geoms:
    lat, lon =zip(*(geom.to_lat_lon_list()))
    plt.plot(lon, lat,
         color='white', linewidth=.5, #the topological plates boundaries in white
         transform=ccrs.Geodetic(),
    )
#the subduction sample points are colored by property value. see "color_by" above
cb=ax.scatter(trench_data['trench_lon'], trench_data['trench_lat'], 30, marker='.', 
              c=trench_data['dist_nearest_edge']* 6371. * np.pi / 180, cmap=plt.cm.jet)
plt.plot(region_of_interest_polygon['lon'],region_of_interest_polygon['lat'], transform=ccrs.Geodetic())
ax.scatter(mesh_points['lon'], mesh_points['lat'], 10, marker='.',color='yellow')
plt.title('Present-day Subduction Zones Coloured by Distance To Nearest Edge(km)')
fig.colorbar(cb, shrink=0.5, label='Distance To Nearest Edge(km)')
plt.show()

# -

# #### Plot a reconstruction map
# This is a reconstructed map with a paleo-age grid, paleo-coastlines, plate boundaries and subduction teeth.

# +
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from matplotlib.colors import LinearSegmentedColormap
import cartopy.crs as ccrs
from netCDF4 import Dataset
from shapely.geometry.polygon import LinearRing

import Utils, pygplates

time = 0
draw_velocity_vectors = True

#change the extent to see specific area
#map_extent = [-85, -30, -55, 15]
map_extent = [-180, 180, -90, 90]

agegrid_file = Utils.download_agegrid(time)
print(agegrid_file)
agegrid_cmap = Utils.get_age_grid_color_map_from_cpt('agegrid.cpt')

#reconstruct coastlines and topology
print("reconstructing geometries...")

resolved_topologies = []
shared_boundary_sections = []
#use pygplates to resolve the topologies
pygplates.resolve_topologies(topology_files, rotation_files, resolved_topologies, time, 
                             shared_boundary_sections)

#coastlines
reconstructed_geometries = []
pygplates.reconstruct(
                Utils.get_parameter('coastlines'),  
                rotation_files, 
                reconstructed_geometries, 
                time, 0)

#subduction zones
subduction_geoms=[]
Utils.get_subduction_geometries(subduction_geoms, shared_boundary_sections)

#velocity vectors
x,y, u,v = Utils.get_velocity_x_y_u_v(time,pygplates.RotationModel(rotation_files),topology_files)
       
# plot the map
fig = plt.figure(figsize=(16,12),dpi=150)
ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=0))
ax.set_extent(map_extent)

if agegrid_file:
    img = Dataset(agegrid_file) #age grid
    cb=ax.imshow(img.variables['z'], origin='lower', transform=ccrs.PlateCarree(),
          extent=[-180, 180, -90, 90], cmap=agegrid_cmap)

#plot coastlines
for geom in reconstructed_geometries:
    lat, lon =zip(*(geom.get_reconstructed_geometry().to_lat_lon_list()))
    plt.plot(lon, lat,
         color='black', linewidth=.5, #the coastlines in black
         transform=ccrs.Geodetic(),
    )

#plot topological plates boundaries
for t in resolved_topologies:
    lat, lon =zip(*(t.get_resolved_boundary().to_lat_lon_list()))
    plt.plot(lon, lat,
         color='blue', linewidth=.5, #the topological plates boundaries in blue
         transform=ccrs.Geodetic(),
    )
 
#plot subduction zones
for geom, aspect in subduction_geoms:
    lat, lon =zip(*(geom.to_lat_lon_list()))
    plt.plot(lon, lat,
         color='blue', linewidth=3, #the subduction zones in blue
         transform=ccrs.Geodetic(),
    )
    teeth = Utils.get_subduction_teeth(lon, lat, triangle_aspect=aspect)
    for tooth in teeth:
        ring = LinearRing(tooth)
        ax.add_geometries([ring], ccrs.PlateCarree(), facecolor='b', edgecolor='black', alpha=1)

 
if draw_velocity_vectors:
    #draw the velocity vectors
    #Some arrows are long and some are very short. To make the plot clearer, we nomalize the velocity magnitude.
    #And use color to denote the different speed.
    u = np.array(u)
    v = np.array(v)
    mag = np.sqrt(u*u+v*v)
    u = u/mag
    v = v/mag
    ax.quiver(x, y, u, v, mag,transform=ccrs.PlateCarree(),cmap='jet')    

gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                  linewidth=1, color='gray', alpha=0.5, linestyle='--')
gl.xlabels_top = False
gl.ylabels_right = False
gl.xlocator = mticker.FixedLocator([-180, -150, -120, -90, -60, -30, 0, 30, 60, 90, 120, 150, 180])
gl.ylocator = mticker.FixedLocator([-90, -75, -60, -45, -30, -15, 0, 15, 30, 45, 60, 75, 90])
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER
gl.xlabel_style = {'color': 'gray', 'weight': 'bold'}
gl.ylabel_style = {'color': 'gray', 'weight': 'bold'}


if agegrid_file:
    plt.title(f'Reconstruction Map with Paleo-age Grid, Paleo-coastlines and Plate Boundaries at {time} Ma')
    fig.colorbar(cb, shrink=0.5, label='Age(Ma)')
else:
    plt.title(f'Reconstruction Map with Paleo-coastlines and Plate Boundaries at {time} Ma')

plt.show()   


# -

# ##### make a subduction zones animation

# +
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from shapely.geometry.polygon import LinearRing
import matplotlib.ticker as mticker
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

import Utils, pygplates

p_time = Utils.get_parameter('time')
start_time = p_time['start'] 
end_time = p_time['end']
time_step = p_time['step']

map_extent = [-180, 180, -90, 90]
 
rotation_files = Utils.get_files(Utils.get_parameter('rotation_files'))
coastlines_file = Utils.get_parameter('coastlines')

for time in range(start_time, end_time+1, 10):
    #coastlines
    reconstructed_geometries = []
    pygplates.reconstruct(
                    coastlines_file, 
                    rotation_files, 
                    reconstructed_geometries, 
                    time, 0)

    #subduction zones
    '''
    topology_files = Utils.get_files(Utils.get_parameter("topology_files"))
    resolved_topologies = []
    shared_boundary_sections = []
    #use pygplates to resolve the topologies
    pygplates.resolve_topologies(topology_files, rotation_files, resolved_topologies, time, 
                                 shared_boundary_sections)
    subduction_geoms=[]
    Utils.get_subduction_geometries(subduction_geoms, shared_boundary_sections)
    '''

    # plot the map
    fig = plt.figure(figsize=(16,12),dpi=144)
    ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=0))
    ax.set_extent(map_extent)

    #plot coastlines
    for geom in reconstructed_geometries:
        lat, lon =zip(*(geom.get_reconstructed_geometry().to_lat_lon_list()))
        plt.plot(lon, lat,
             color='black', linewidth=.5, #the coastlines in black
             transform=ccrs.Geodetic(),
        )

    #plot subduction zones
    '''
    for geom, aspect in subduction_geoms:
        lat, lon =zip(*(geom.to_lat_lon_list()))
        plt.plot(lon, lat,
             color='blue', linewidth=1, #the subduction zones in blue
             transform=ccrs.Geodetic(),
        )
        teeth = Utils.get_subduction_teeth(lon, lat, triangle_aspect=aspect)
        for tooth in teeth:
            ring = LinearRing(tooth)
            ax.add_geometries([ring], ccrs.PlateCarree(), facecolor='b', edgecolor='black', alpha=1)
    '''

    trench_data = Utils.get_trench_points(time)
    #the subduction sample points are colored by property value. see "color_by" above
    cb=ax.scatter(trench_data['trench_lon'], trench_data['trench_lat'], 50, marker='.', 
                  c=trench_data['dist_nearest_edge']* 6371. * np.pi / 180, cmap=plt.cm.jet, vmax=3000, vmin=0)

    fig.colorbar(cb, shrink=0.5, label='Distance To Nearest Edge(km)')

    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=1, color='gray', alpha=0.5, linestyle='--')
    gl.xlabels_top = False
    gl.ylabels_right = False
    gl.xlocator = mticker.FixedLocator([-180, -150, -120, -90, -60, -30, 0, 30, 60, 90, 120, 150, 180])
    gl.ylocator = mticker.FixedLocator([-90, -75, -60, -45, -30, -15, 0, 15, 30, 45, 60, 75, 90])
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'color': 'gray', 'weight': 'bold'}
    gl.ylabel_style = {'color': 'gray', 'weight': 'bold'}

    plt.title(f'Subduction Zones at {time} Ma')
    plt.savefig(Utils.get_tmp_dir() + f'subduction_zones_{time}_Ma.png',bbox_inches='tight',pad_inches=0)
    print(f'plotting {time}Ma')
    plt.close()
    #plt.show()   
   


# +
# %%capture --no-stdout

import moviepy.editor as mpy
import Utils

p_time = Utils.get_parameter('time')
start_time = p_time['start'] 
end_time = p_time['end']
time_step = p_time['step']

frame_list = [Utils.get_tmp_dir() + f'subduction_zones_{time}_Ma.png' for time in range(start_time, end_time+1, 10)]
clip = mpy.ImageSequenceClip(frame_list, fps=2)
clip.write_gif(Utils.get_tmp_dir() + "subduction_zones.gif")
clip.write_videofile(Utils.get_tmp_dir() + "subduction_zones.mp4")
print('done')

# +
import io, base64

from IPython.display import HTML

video = io.open(Utils.get_tmp_dir() + "subduction_zones.mp4", 'r+b').read()

encoded = base64.b64encode(video)

HTML(data='''<video width=960 alt="subduction zones animation" controls>

<source src="data:video/mp4;base64,{0}" type="video/mp4" /> </video>'''.format(encoded.decode('ascii')))

# -

# #### This is the end of step 1 and now open the step 2 notebook --  "2_Plot_and_Select_Mineral_Resources.ipynb"


