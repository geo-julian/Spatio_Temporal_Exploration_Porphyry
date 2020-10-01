import requests, os
import numpy as np
import pandas as pd
import extract_earth_chem
from parameters import parameters
import pygplates
import Utils

def run():
    if not os.path.isdir(Utils.get_coreg_input_dir()):
            os.mkdir(Utils.get_coreg_input_dir())
            
    earth_chem_file = 'EarthChem_all.csv'
    polygon_points = Utils.get_region_of_interest_polygon().values.flatten()
    deposit_points = extract_earth_chem.query(earth_chem_file, 'AU', polygon_points)
    mesh_points = Utils.get_mesh_points(polygon_points)

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
    deposit_points.to_csv(Utils.get_coreg_input_dir() + 'negative_deposits.csv', index_label = 'index')

    mesh_plate_ids = Utils.get_plate_id(mesh_points.lon.tolist(), mesh_points.lat.tolist(),
                                        static_polygons, rotation_model)
    mesh_points['plate_id'] =  mesh_plate_ids  
    deposit_candidates=[]
    for t in range(start_time, end_time+1, time_step):
        for index, p in mesh_points.iterrows():
            deposit_candidates.append([p['lon'], p['lat'], t, p['plate_id']]) 
    deposit_candidates=pd.DataFrame(deposit_candidates, columns=['lon','lat','age','plate_id'])
    deposit_candidates = deposit_candidates.astype({"plate_id": int}) 
    deposit_candidates.to_csv(Utils.get_coreg_input_dir() + 'deposit_candidates.csv', index_label = 'index')
