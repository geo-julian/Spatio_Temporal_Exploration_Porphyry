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

# ##  Step 2 : Plot and Select Mineral Resources
#
# Machine learning algorithms require data to train a predictive model. In this notebook, we are going to select some mineral deposits which we are interested in. And then these selected deposits will be used as input in Step 3 to retrieve attributes from the subduction zone sample points. After that, in Step 4, we can format the data so that they are ready to be fed into the machine learning models in Step 5.
#
# Now let's select some interesting mineral deposits and prepare the data for the next step. Step 3 takes a csv file as input, which should contain five columns. 
#
# * index       -- unique number to identify the deposit
# * lon         -- the longitude of the deposit
# * lat         -- the latitude of the deposit
# * age         -- how old the deposit is
# * plate id    -- an ID for the tectonic plate in which the deposit resides, this ID is used in plate tectonic reconstruction
#
# The ultimate goal of this step is to produce the csv file with the above five columns. There are many ways to get data and prepare the csv file. We are going to show you some examples below. After you have gone through the examples, you should have learnt how to create the csv file. And you are encouraged to come up with your own novel ways to find and process data, for example hack into KGB database, ect. Remember, only the final csv file matters. Focus on the five columns inside the csv file.
#
# ---

# #### Example 1: Use ../data/CopperDeposits/XYBer14_t2_ANDES.shp data
#
# This is the simplest example. 
#
# We just load in data from a shape file and write out the five-column data to a csv file.
#
# Run the script to create an example coregistration input file for XYBer14_t2_ANDES.shp dataset.
#
# The coregistration input file can be used later in step 3: coregistration.
#
# ##### The implementation details are in create_coregistration_input_data_example.py.
# If you would like to know more magic behind the scene, open [create_coregistration_input_data_example.py](create_coregistration_input_data_example.py)

# +
import pandas as pd
from parameters import parameters
from create_coregistration_input_data_example import *
import pygplates
import glob

#get start time, end time and time step from parameters.py
start_time = parameters["time"]["start"]
end_time = parameters["time"]["end"]
time_step = parameters["time"]["step"]

#first, we process the real deposits from XYBer14_t2_ANDES.shp
data = process_real_deposits(start_time, end_time, time_step)

#then, we copy the real deposits and replace the age with random number
#these deposits with random ages will be labeled as non-deposit later
random_data = generate_random_deposits(data, start_time, end_time)

#save negative deposits
random_data=pd.DataFrame(random_data, columns=['index', 'lon','lat','age','plate_id'])
random_data = random_data.astype({"plate_id": int, "age":int}) 
random_data.to_csv(Utils.get_coreg_input_dir() + 'negative_deposits.csv', index=False)

#save positive deposits
data=pd.DataFrame(data, columns=['index', 'lon','lat','age','plate_id'])
data = data.astype({"plate_id": int, "age":int}) 
data.to_csv(Utils.get_coreg_input_dir() + 'positive_deposits.csv', index=False)

#save deposit candidates
deposit_candidates = Utils.get_deposit_candidates()
deposit_candidates.to_csv(Utils.get_coreg_input_dir() + 'deposit_candidates.csv', index_label = 'index')
# -

# Now, let's plot a map to see the trench and copper deposits in Andes.
#
# Since the deposits are all in Andes, draw the map with extent "-85, -30, -55, 15"(South America). The deposits are coloured by their ages.

# +
import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs
import matplotlib.ticker as mticker
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

def set_ax(ax):
    ax.stock_img()
    ax.set_extent([-85, -29, -55, 15])

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
trench_data= np.genfromtxt(trench_file, skip_header=1, delimiter=',')

fig, (ax1, ax2) = plt.subplots(1, 2, subplot_kw={'projection': ccrs.PlateCarree()},figsize=(12,12),dpi=150)
set_ax(ax1)
set_ax(ax2)

cb = ax1.scatter(data['lon'], data['lat'], 50, marker='.',c=data['age'],  cmap=plt.cm.jet)
cb = ax2.scatter(random_data['lon'], random_data['lat'], 50, marker='.',c=random_data['age'],  cmap=plt.cm.jet)
ax1.scatter(trench_data[:,0], trench_data[:,1], 2, marker='.', color='white')# draw the trench in white
ax2.scatter(trench_data[:,0], trench_data[:,1], 2, marker='.', color='white')# draw the trench in white
ax1.title.set_text('Andes Copper Deposits Coloured By Real Ages')
ax2.title.set_text('Andes Copper Deposits Coloured By Random Ages')
cbar = fig.colorbar(cb, shrink=0.5, ax=[ax1, ax2], orientation='horizontal', pad=0.05)
cbar.set_label('Age(Myr)',size=10)
cbar.ax.tick_params(labelsize=7)
plt.show()
# -

# #### Example 2: Extract Data From EarthChem Data
#
# In this part, we are going to show you how to extract data from EarthChem database and draw the deposits on a map.
#
# We will select deposits which are within a region(5 degrees) of any trench sample point. The data will also be filtered by mineral type.
#
# #### Some mineral symbols and their meaning
#
# * CU -- Copper
# * CO2 -- Carbon dioxide
# * ZN -- Zinc
# * AU -- Gold
# * GA -- Gallium
# * CS -- Caesium
# * LI -- Lithium
# * AG -- Sliver
#
# #### All the mineral symbols in EarthChem data
# * SIO2,U234_U238,TIO2,AL2O3,FE2O3,TH230_TH232,FE2O3T,TH232_TH230,FEO,FEOT,MGO,RA228_RA226,CAO,NA2O,K2O,
# * P2O5,MNO,U238_ACTIVITY,LOI,H2O_PLUS,TH230,H2O_MINUS,H2O,RA226,CR2O3,NIO,LA,CE,CACO3,PR,SM,EU,GD,TB,DY,
# * HO,ER,TM,YB,U234_U238_ACTIVITY,LU,LI,BE,B,C,CO2,F,CL,K,CA,MG,SC,TI,V,FE,CR,MN,CO,NI,CU,ZN,GA,ZR,GER,SR,
# * K40_AR36,BI,OS187_OS188,NB,TH232_U238,PB208_PB206,CD,PO210_TH230,U238_PB204,BA,AR40_AR36,W,AR37_AR39,AU,
# * XE129_XE132,LU176_HF177,HG,OS186_OS188,PB206_PB208,TA,PB210_U238,SB,SR87_SR86,SE,PB207_PB204,PB206_PB204,
# * PB208_PB204,SN,S,TH230_U238,ND143_ND144,U,RA226_TH230,I,P,Y,EPSILON_ND,MO,OS184_OS188,PD,RA226_TH228,TE,
# * TH232_PB204,HF,OS187_OS186,CL36_CL,RA228_TH232,PB206_PB207,PB,INDIUM,H,PB210_RA226,AR38_AR36,AR40_AR39,D18O,
# * AG,TH,U235_PB204,NE21_NE22,TL,NE20_NE22,AS,HF176_HF177,RB,AL,BE10_BE9,AR36_AR39,ND,CS,quartz

# +
import requests, os
import numpy as np
import pandas as pd
import extract_earth_chem
from parameters import parameters

earth_chem_file = 'EarthChem_all.csv'
polygon_points = Utils.get_region_of_interest_polygon().values.flatten()
deposit_points = extract_earth_chem.query(earth_chem_file, 'AU', polygon_points)
display(deposit_points)
# -

# Now, we have selected 44160 copper deposits from the EarthChem database.
#
# Do you rememer we need five columns?
#
# The EarthChem_CU.csv does not have "index" and "plate id" columns and it has a extra "CU" column.
#
# The "index" column can be easily generated and the extra "CU" column can be dropped. However, the "plate id" column is a bit tricky. Let me show you how to deal with it in the next code cell.

# +
#first, let's find plate id for those deposits
static_polygons = pygplates.FeatureCollection(parameters['static_polygons_file'])
rotation_model = pygplates.RotationModel(Utils.get_files(parameters['rotation_files']))
plate_ids = Utils.get_plate_id(deposit_points.LONGITUDE.tolist(), deposit_points.LATITUDE.tolist(), 
                               static_polygons, rotation_model)

deposit_points.rename(columns = {'LONGITUDE':'lon', 'AGE':'age', 'LATITUDE':'lat'}, inplace = True) 
deposit_points.age = np.round(deposit_points.age)
deposit_points = deposit_points.astype({"age": int}) 
deposit_points = deposit_points[['lon', 'lat', 'age']]
deposit_points['plate_id'] = plate_ids

start_time = parameters['time']['start'] 
end_time = parameters['time']['end']
time_step = parameters['time']['step']

deposit_points = deposit_points[deposit_points['age']>start_time]
deposit_points = deposit_points[deposit_points['age']<end_time]
deposit_points.drop_duplicates(inplace=True) 
deposit_points.reset_index(drop=True, inplace=True)

deposit_points.to_csv(Utils.get_coreg_input_dir() + 'positive_deposits.csv', index_label = 'index')
deposit_points.age = np.random.randint(start_time+1, end_time, size=len(deposit_points))
deposit_points.to_csv(Utils.get_coreg_input_dir() +  'negative_deposits.csv', index_label = 'index')

#save deposit candidates
deposit_candidates = Utils.get_deposit_candidates()
deposit_candidates.to_csv(Utils.get_coreg_input_dir() + 'deposit_candidates.csv', index_label = 'index')


# -

# Now, the data in EarthChem_CU_all.csv have all the five columns we need. 
#
# Next, let's plot the data in a map.

# +
# %matplotlib inline
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import matplotlib.ticker as mticker
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

trench_file = Utils.get_convergence_dir() + 'subStats_0.00.csv'
if os.path.isfile(trench_file):
    trench_data= np.genfromtxt(trench_file, skip_header=1, delimiter=',')
else:
    raise Exception(f'\nERROR: unable to open file {trench_file}. \nRun Step 1 Generate Subduction Convergence Kinematics Statistics first!')
    
mesh_points = Utils.get_mesh_points(polygon_points)

#plot the data    
fig = plt.figure(figsize=(12,8),dpi=96)
ax = plt.axes(projection=ccrs.PlateCarree())

gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                  linewidth=1, color='gray', alpha=0.5, linestyle='--')
gl.xlabels_top = False
gl.ylabels_right = False
gl.xlocator = mticker.FixedLocator(range(-180,180,10))
gl.ylocator = mticker.FixedLocator(range(-90,90,10))
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER
gl.xlabel_style = {'color': 'gray', 'weight': 'bold'}
gl.ylabel_style = {'color': 'gray', 'weight': 'bold'}

ax.stock_img()
#ax.set_extent([-180, 180, -90, 90])
ax.set_extent(Utils.get_region_of_interest_extent())
cb = ax.scatter(deposit_points['lon'], deposit_points['lat'], 50, marker='.',c=deposit_points['age'], vmin=1, vmax=100, cmap=plt.cm.jet)
ax.scatter(mesh_points['lon'], mesh_points['lat'], 10, marker='.',color='yellow')
ax.scatter(trench_data[:,0], trench_data[:,1], 20, marker='.', color='white')
plt.plot(polygon_points[0::2],polygon_points[1::2], transform=ccrs.Geodetic())
plt.title('Deposits in North America Coloured by Age(Myr)')
fig.colorbar(cb, shrink=0.5, label='Age(Myr)')
plt.show()
# -

# #### Example 3: PorCuEX2008.csv

# +
# %matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs
import matplotlib.ticker as mticker
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import pandas as pd
import Utils

#read in data
data = pd.read_csv("../data/PorCuEX2008.csv")
trench_data= np.genfromtxt(Utils.get_convergence_dir() + 'subStats_0.00.csv', skip_header=1, delimiter=',')
print(f'Data shape: {data.shape}')
print('The first 5 rows in the data:')
display(data.head())

print('All the columns in the data:')
print(data.columns)

#select deposits within 5 degrees of trench
indices = Utils.select_points_in_region(data['LongitudeDecimal'], data['LatitudeDecimal'], 
                              trench_data[:,0], trench_data[:,1], 5)#5 degrees
data_near_trench = data[indices]
print(f'Data near trench shape: {data_near_trench.shape}')

#select data within bounding box [-85, -30, -55, 15]
data_south_america = data_near_trench[data_near_trench['LongitudeDecimal']>-85]
data_south_america = data_south_america[data_near_trench['LongitudeDecimal']<-30]
data_south_america = data_south_america[data_near_trench['LatitudeDecimal']>-55]
data_south_america = data_south_america[data_near_trench['LatitudeDecimal']<5]
print(f'Data South America shape: {data_south_america.shape}\n')

#the data_big and data_small do not always have the same length
#for example, when there are more than half of the values are zeros, 
#they should all be considered small values
def divide_data(data, column_name):
    tmp = data.fillna(value={column_name:0})
    data_sorted = tmp.sort_values(by=column_name, ascending=False)
    #print(data_sorted[column_name])
    middle_value = data_sorted[column_name].values.tolist()[int(data_sorted.shape[0]/2)]
    #print(middle_value)
    data_big = data_sorted[data_sorted[column_name]>middle_value]
    data_small = data_sorted[data_sorted[column_name]<=middle_value]
    #print(f'big {column_name}: \n', data_big[column_name])
    #print(f'small {column_name}: \n', data_small[column_name])
    return data_big, data_small

#divide data by Tonnage
big_tonnage, small_tonnage = divide_data(data_near_trench, 'Tonnage')

#We can do the same to 'Copper grade', 'Molybdenum grade', 'Gold grade', 'Silver grade'
big_copper_percent, small_copper_percent = divide_data(data_near_trench, 'Copper grade')
big_molybdenum_percent, small_molybdenum_percent = divide_data(data_near_trench, 'Molybdenum grade')
big_gold_percent, small_gold_percent = divide_data(data_near_trench, 'Gold grade')
big_silver_percent, small_siler_percent = divide_data(data_near_trench, 'Silver grade')

# -

# ##### Plot the data in a global map

# +
#*******************************************

#LOOK HERE!!
#you may choose the data to plot
#only keep the one line which you want uncommented

#plot_data = data
plot_data = data_near_trench
#plot_data = data_south_america
#plot_data = big_tonnage
#plot_data = small_tonnage
#plot_data = big_copper_percent 
#plot_data = small_copper_percent
#plot_data = big_molybdenum_percent
#plot_data = small_molybdenum_percent
#plot_data = big_gold_percent
#plot_data = small_gold_percent
#plot_data = big_silver_percent
#plot_data = small_siler_percent 
#*******************************************


#plot the map
fig = plt.figure(figsize=(16,12),dpi=150)
ax = plt.axes(projection=ccrs.PlateCarree())

gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                  linewidth=1, color='gray', alpha=0.5, linestyle='--')
gl.xlabels_top = False
gl.ylabels_left = False
gl.xlocator = mticker.FixedLocator([-180, -150, -120, -90, -60, -30, 0, 30, 60, 90, 120, 150, 180])
gl.ylocator = mticker.FixedLocator([-90, -75, -60, -45, -30, -15, 0, 15, 30, 45, 60, 75, 90])
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER
gl.xlabel_style = {'color': 'gray', 'weight': 'bold'}
gl.ylabel_style = {'color': 'gray', 'weight': 'bold'}

ax.stock_img()
ax.set_extent([-180, 180, -90, 90])
#ax.set_extent([-85, -29, -55, 15])
cb = ax.scatter(plot_data['LongitudeDecimal'], plot_data['LatitudeDecimal'], 50, marker='.',c=plot_data['AgeMY'], 
                vmin=1, vmax=100, cmap=plt.cm.jet)
ax.scatter(trench_data[:,0], trench_data[:,1], 5, marker='.', color='white')
plt.title('Deposits Near the Subduction Zones Coloured by Age(Myr)')
fig.colorbar(cb, shrink=0.5, label='Age(Myr)')
plt.show()
# -

# ##### Plot a regional map

# +
#divide data by Tonnage
big_tonnage, small_tonnage = divide_data(data_south_america, 'Tonnage')

#We can do the same to 'Copper grade', 'Molybdenum grade', 'Gold grade', 'Silver grade'
big_copper_percent, small_copper_percent = divide_data(data_south_america, 'Copper grade')
big_molybdenum_percent, small_molybdenum_percent = divide_data(data_south_america, 'Molybdenum grade')
big_gold_percent, small_gold_percent = divide_data(data_south_america, 'Gold grade')
big_silver_percent, small_siler_percent = divide_data(data_south_america, 'Silver grade')

#*******************************************

#LOOK HERE!!
#you may choose the data to plot
#only keep the one line which you want uncommented

#plot_data = data_south_america
plot_data = big_tonnage
#plot_data = small_tonnage
#plot_data = big_copper_percent 
#plot_data = small_copper_percent
#plot_data = big_molybdenum_percent
#plot_data = small_molybdenum_percent
#plot_data = big_gold_percent
#plot_data = small_gold_percent
#plot_data = big_silver_percent
#plot_data = small_siler_percent 
#******************************************

#plot the map
fig = plt.figure(figsize=(6,6),dpi=150)
ax = plt.axes(projection=ccrs.PlateCarree())
ax.stock_img()
ax.set_extent([-85, -29, -55, 15])
#ax.set_extent([-180, 180, -90, 90])
data=data[:155]

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

cb = ax.scatter(plot_data['LongitudeDecimal'], plot_data['LatitudeDecimal'], 20, marker='.',
                c=plot_data['AgeMY'], vmin=1, vmax=100, cmap=plt.cm.jet)
ax.scatter(trench_data[:,0], trench_data[:,1], 2, marker='.', color='white')# draw the trench in white
plt.title('Deposits Near the Subduction Zones Coloured By Age',fontsize=7)
cbar = fig.colorbar(cb, shrink=0.5)
cbar.set_label('Age(Myr)',size=10)
cbar.ax.tick_params(labelsize=7)
plt.show()
# -

# Add random age deposits and trench points. The PorCuEX2008_south_america.csv is ready to be used in Step 3.

# +
import pygplates
#first, let's find plate id for those deposits

static_polygons = pygplates.FeatureCollection(parameters['static_polygons_file'])
rotation_model = pygplates.RotationModel(Utils.get_files(parameters['rotation_files']))
plate_ids = Utils.get_plate_id(data_south_america.LongitudeDecimal.tolist(), 
                               data_south_america.LatitudeDecimal.tolist(), 
                               static_polygons, 
                               rotation_model)

ages = np.round(data_south_america.AgeMY)
indices = list(range(len(ages)))

data=np.c_[indices,data_south_america.LongitudeDecimal,data_south_america.LatitudeDecimal, ages, plate_ids]
data=data[~np.isnan(data).any(axis=1)].tolist()
#then, we copy the real deposits and replace the age with random number
#these deposits with random ages will be labeled as non-deposit later
random_data = generate_random_deposits(data, start_time, end_time)

#save negative deposits
random_data=pd.DataFrame(random_data, columns=['index', 'lon','lat','age','plate_id'])
random_data = random_data.astype({"plate_id": int, "age":int}) 
random_data.to_csv(Utils.get_coreg_input_dir() + '/negative_deposits_PorCuEX2008.csv', index=False)

#save positive deposits
data=pd.DataFrame(data, columns=['index', 'lon','lat','age','plate_id'])
data = data.astype({"plate_id": int, "age":int}) 
data.to_csv(Utils.get_coreg_input_dir() + '/positive_deposits_PorCuEX2008.csv', index=False)

#save deposit candidates
deposit_candidates = Utils.get_deposit_candidates()
deposit_candidates.to_csv(Utils.get_coreg_input_dir() + 'deposit_candidates_PorCuEX2008.csv', index_label = 'index')

data
# -

# #### This is the end of step 2 and now open the step 3 notebook -- "3_Coregistration.ipynb"


