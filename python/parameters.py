# These default parameters can be overridden in config.json
import os
script_dir = os.path.dirname(__file__)

parameters = {
    #IMPORTANT: 
    #the convergence.py depends on the PlateTectonicTools, 
    #you need to tell the script where to find it
    #The PlateTectonicTools repository https://github.com/EarthByte/PlateTectonicTools.git is a submodule now
    #run the following command after you have cloned this spatio-temporal-exploration repository.
    #git submodule update --init --recursive
    "plate_tectonic_tools_path" : script_dir + "/../PlateTectonicTools/ptt/",
 
    'case_name' : 'case_AREPS',
    
    #the file which contains the seed points.
    #coregistration input file 
    'coreg_input_files' : ['positive_deposits.csv', 'negative_deposits.csv','deposit_candidates.csv'],
    
    #folder contains the coregistration output files.
    'coreg_output_dir' : 'coreg_output',
    
    #folder contains the coregistration input files.
    'coreg_input_dir' : 'coreg_input',
    
    #folder contains the machine learning output files.
    'ml_output_dir' : 'ml_output',
    
    #folder contains the machine learning input files.
    'ml_input_dir' : 'ml_input',
    
    "time" : {
        "start" : 0,
        "end"   : 235,
        "step"  : 1
    },

    'machine_learning_engine' : 'RFC', #'RFC' random forest ; 'SVC' support vector Classification 

#    'feature_names' : ['total_sediment_thick', 'seafloor_age', 'conv_rate', 'subducting_abs_rate', 'conv_ortho'], ### Exercise 3    
    
#    'feature_names' : ['total_sediment_thick', 'seafloor_age', 'conv_rate', 'dist_nearest_edge', 'subducting_abs_rate'], ### Exercise 2
    
#     'feature_names' : ['conv_rate', 'conv_angle', 'trench_abs_rate', 'trench_abs_angle', 'arc_len', 'dist_nearest_edge', 'conv_ortho', 'conv_paral', 'trench_abs_ortho', 'trench_abs_paral', 'subducting_abs_rate', 'subducting_abs_angle', 'subducting_abs_ortho', 'subducting_abs_paral', 'seafloor_age', 'carbonate_sediment_thickness',  'total_sediment_thick', 'ocean_crust_carb_percent'],
    
    'feature_names' : ['conv_rate', 'dist_nearest_edge', 'subduction_volume_km3y', 'carbonate_sediment_thickness', 'ocean_crust_carb_percent'],
    
    #the region of interest parameters are used in coregistration.py
    #given a seed point, the coregistration code looks for the nearest geomery within region[0] first
    #if not found, continue to search in region[1], region[2], etc
    #if still not found after having tried all regions, give up
    #the distance is in degrees. 
    "regions" : [5, 10],
    "region_of_interest_polygon_file" : script_dir + "/../data/polygon_sout_america_julian.csv",
    
    "rotation_files" : [ script_dir+"/../data/2019_v2_Clennett/*.rot"],
    "topology_files" : [ script_dir+"/../data/2019_v2_Clennett/*.gpml"], 
    "static_polygons_file" : script_dir+'/../data/Shapefiles/StaticPolygons/Global_EarthByte_GPlates_PresentDay_StaticPlatePolygons_2015_v1.shp',
    "agegrid_url" : "",
    "coastlines" :  script_dir+"/../data/2019_v2_Clennett/Clennett_2019_Coastlines.gpml",
    
    #the following two parameters are used by subduction_convergence
    #see https://github.com/EarthByte/PlateTectonicTools/blob/master/ptt/subduction_convergence.py
    "threshold_sampling_distance_degrees" : 0.2,
    "velocity_delta_time" : 1,
    
    #a list of file paths from which the coregistration scripts query data
    "vector_files" : [
        '{conv_dir}subStats_{time:.2f}.csv', #can be generated by convergence.py
        #'./convergence_data/subStats_{time:.2f}.csv',
        #more files below
    ],
    
    "anchor_plate_id" : 0, #see https://www.gplates.org/user-manual/MoreReconstructions.html
    
    #a list of grid files from which the coregistration scripts query data
    "grid_files" : [
        #["../data/AgeGrids/Muller_etal_2016_AREPS_v1.17_AgeGrid-{time:d}.nc", "seafloor_age"],
        #["../data/carbonate_sed_thickness/decompacted_sediment_thickness_0.5_{time:d}.nc", "carbonate_sediment_thickness"],
        #["../data/predicted_oceanic_sediment_thickness/sed_thick_0.2d_{time:d}.nc", "total_sediment_thick"],
        #["../data/ocean_crust_CO2_grids/ocean_crust_carb_percent_{time:d}.nc", "ocean_crust_carb_percent"]
    ],
    
    "convergence_data_filename_prefix" : "subStats",
    "convergence_data_filename_ext" : "csv",
    "convergence_data_dir" : "./convergence/",

    'overwrite_existing_convergence_data' : True, #if True, always generate the new convergence data
#    'map_extent' : [-180, -80, 0, 85], #North America
    'map_extent' : [-85, -29, -55, 15], #South America
    'topo_grid' : '../data/topo15_3600x1800.nc',
    "terranes" : ""
}
