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

# ## Step 3: Coregistration
#
# In this step, we are going to connect the selected mineral deposits in Step 2 with the trench sample points we have generated in Step 1. The mineral deposits csv file contains only 5 columns -- index, longitude, latitude, age and plate id. These attributes are not enough for the machine learning analysis. In order to obtain features associated with the deposits, we need to connect these mineral deposits with the trench sample points. We call this process coregistration.
#
# The coregistration method is simple. For a given mineral deposit, the coregistration process will try to find the nearest trench point within a certain region. If found, the subduction convergence kinematics statistics of the trench point will be associated with the mineral deposit. The attributes retrieved from the trench sample points will be used as input data for machine learning models later.
#
# First, let's run the coregistration script and see what will happen. The coregistration script can be configurated via parameters.py, such as the input mineral deposits file, output file name and region of interest, etc.
#
# Use config.json file to override the default parameters in [parameters.py](parameters.py):
#
#

# +
import Utils

#
# SEE HERE! 
# This cell must run first to setup the working environment
#

#load the config file
Utils.load_config('config.json')
Utils.print_parameters()

# +
#let's print out some important parameters
#you can change the 'input_file' in config.json to use different mineral deposits. 
#Remember the files we have created in step 2?
print('The file name of the mineral deposits: ', Utils.get_parameter('coreg_input_files'))
print('The output folder: ', Utils.get_coreg_input_dir())
print('The region of interest(in degree): ', Utils.get_parameter('regions'))
print('The subduction convergence kinematics statistics file name template: ', Utils.get_parameter('vector_files'))
print('\n')

import coregistration
coregistration.run() #run the coregistration script
#some files should have been created at this point
#let's move to the next cell and check the results

# +
import pandas as pd

#read in the coregistration output file
data = pd.read_csv(Utils.get_coreg_output_dir() + "positive_deposits.csv") 
display(data.head())#let's print the first 5 rows

print(data.columns)
print('\nThe meaning of the columns: \n')
Utils.print_columns()

input_data = pd.read_csv(Utils.get_coreg_input_dir() + Utils.get_parameter('coreg_input_files')[0])
display(input_data)

#the input data and output data has the same length
print('The shape of the output data: ', data.shape)
print('The shape of the input data: ',input_data.shape)
# -

# We can see in above code cell that the input data and output data has the same length. It means, for each input mineral deposit, there is one corresponding data row in the output file. 
#
# The coregistration program takes the mineral deposit coordinates and uses age and plate id to reconstruct the deposits back in time. And then the program searches the nearby subduction trench, if found, copy the subduction convergence kinematics statistics.

# ##### Now we plot some maps to see the attributes acquired from coregistration.

# +
# %matplotlib inline

import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs
import matplotlib.ticker as mticker
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

def set_ax(ax):
    ax.stock_img()
    ax.set_extent(Utils.get_region_of_interest_extent())

    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=1, color='gray', alpha=0.5, linestyle='--')
    gl.xlabels_top = False
    gl.ylabels_right = False
    gl.xlocator = mticker.FixedLocator([-180, -150, -120, -90, -80, -70, -60,-50,-40,-30, 0, 180])
    gl.ylocator = mticker.FixedLocator([-90,-50,-40, -30, -20,-10, 0, 10, 20, 30, 40,50, 90])
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'color': 'gray', 'weight': 'bold', 'fontsize': '5'}
    gl.ylabel_style = {'color': 'gray', 'weight': 'bold', 'fontsize': '5'}

trench_file = Utils.get_convergence_dir() + 'subStats_0.00.csv'
trench_data= np.genfromtxt(trench_file)

fig, (ax1, ax2) = plt.subplots(1, 2, subplot_kw={'projection': ccrs.PlateCarree()},figsize=(12,12),dpi=150)
set_ax(ax1)
set_ax(ax2)

cb_1 = ax1.scatter(data['lon'], data['lat'], 50, marker='.',
                c=data['conv_rate'],  cmap=plt.cm.jet)
cb_2 = ax2.scatter(data['lon'], data['lat'], 50, marker='.',
                c=data['dist_nearest_edge']* 6371. * np.pi / 180,  cmap=plt.cm.jet)

ax1.title.set_text('Deposits Coloured By Convergence Rate')
ax2.title.set_text('Deposits Coloured By Distance to Nearest Edge')
cbar_1 = fig.colorbar(cb_1, shrink=0.5, ax=[ax1], orientation='horizontal', pad=0.05)
cbar_1.set_label('Convergence Rate (cm/yr)',size=10)
cbar_1.ax.tick_params(labelsize=7)
cbar_2 = fig.colorbar(cb_2, shrink=0.5, ax=[ax2], orientation='horizontal', pad=0.05)
cbar_2.set_label('Distance to Nearest Edge (km)',size=10)
cbar_2.ax.tick_params(labelsize=7)
plt.show()
# -

# ### The following cells will create an animation to show which parts of subduction zones are used in co-registration.

# +
import numpy as np
import pandas as pd
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

positive = pd.read_csv(Utils.get_coreg_output_dir() + "positive_deposits.csv") 
negative = pd.read_csv(Utils.get_coreg_output_dir() + "negative_deposits.csv") 
candidates = pd.read_csv(Utils.get_coreg_output_dir() + "deposit_candidates.csv") 
coreg_data =  pd.concat([positive, negative, candidates])

for time in range(start_time, end_time+1, 5):
#for time in range(0, 1, 10):
    #coastlines
    reconstructed_geometries = []
    pygplates.reconstruct(
                    coastlines_file, 
                    rotation_files, 
                    reconstructed_geometries, 
                    time, 0)

    #Terranes
    reconstructed_terranes = []
    terranes = Utils.get_parameter('terranes')
    if terranes:
        pygplates.reconstruct(
                    terranes, 
                    rotation_files, 
                    reconstructed_terranes, 
                    time, 0)
   
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
        
    #plot Terranes
    for geom in reconstructed_terranes:
        lat, lon =zip(*(geom.get_reconstructed_geometry().to_lat_lon_list()))
        plt.plot(lon, lat,
             color='black', linewidth=.5, #the Terranes in black
             transform=ccrs.Geodetic(),
        )


    trench_data = Utils.get_trench_points(time)
    sub_idx = coreg_data[coreg_data['age']==time]['sub_idx'].dropna().unique()
    candidates_ =  candidates[candidates['age']==time][['recon_lon', 'recon_lat']].drop_duplicates()
    positive_ =  positive[positive['age']==time][['recon_lon', 'recon_lat']].drop_duplicates()
    negative_ =  negative[negative['age']==time][['recon_lon', 'recon_lat']].drop_duplicates()
    #print(sub_idx)
    #the subduction sample points are colored by property value. see "color_by" above
    ax.scatter(trench_data['trench_lon'], trench_data['trench_lat'], 5, marker='.',color='blue')
    
    ax.scatter(trench_data.iloc[sub_idx]['trench_lon'], trench_data.iloc[sub_idx]['trench_lat'], 
                  10, marker='.', color='red', zorder=100)
    
    ax.scatter(candidates_['recon_lon'], candidates_['recon_lat'] ,
                  5, marker='.', color='yellow')
    
    ax.scatter(positive_['recon_lon'], positive_['recon_lat'] ,
                  5, marker='.', color='green')
    
    ax.scatter(negative_['recon_lon'], negative_['recon_lat'] ,
                  5, marker='.', color='cyan')
   
    #fig.colorbar(cb, shrink=0.5, label='Distance To Nearest Edge(km)')

    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=1, color='gray', alpha=0.5, linestyle='--')
    gl.xlabels_top = False
    gl.ylabels_right = False
    gl.xlocator = mticker.FixedLocator(range(-180,180,30))
    gl.ylocator = mticker.FixedLocator(range(-90, 90, 15))
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'color': 'gray', 'weight': 'bold'}
    gl.ylabel_style = {'color': 'gray', 'weight': 'bold'}

    plt.title(f'Subduction Zones at {time} Ma')
    plt.savefig(Utils.get_tmp_dir() + f'subduction_zones_in_use_{time}_Ma.png',bbox_inches='tight',pad_inches=0)
    print(f'plotting {time}Ma')
    plt.close()
    #plt.show()   


# +
# %%capture --no-stdout

import moviepy.editor as mpy

p_time = Utils.get_parameter('time')
start_time = p_time['start'] 
end_time = p_time['end']
time_step = p_time['step']

frame_list = [Utils.get_tmp_dir() + f'subduction_zones_in_use_{time}_Ma.png' for time in range(end_time, start_time-1, -5)]
clip = mpy.ImageSequenceClip(frame_list, fps=2)
clip.write_gif(Utils.get_tmp_dir() + "subduction_zones_in_use.gif")
clip.write_videofile(Utils.get_tmp_dir() + "subduction_zones_in_use.mp4")
print('done')

# +
import io, base64

from IPython.display import HTML

video = io.open(Utils.get_tmp_dir() + "subduction_zones_in_use.mp4", 'r+b').read()

encoded = base64.b64encode(video)

HTML(data='''<video width=960 alt="subduction zones animation" controls>

<source src="data:video/mp4;base64,{0}" type="video/mp4" /> </video>'''.format(encoded.decode('ascii')))

# -

# #### This is the end of step 3 and now open the step 4 notebook -- "4_Data_Wrangling.ipynb"


