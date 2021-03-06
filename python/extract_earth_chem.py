#!/usr/bin/env python

import pandas as pd
import shapefile, csv, math, argparse, os, requests
from shapely.geometry import Point
import scipy.spatial
import numpy as np

import pygplates
from parameters import parameters as param
import Utils

convergence_filename_template = './convergence_data/subStats_{time:.2f}.csv'

def get_data_file(earth_chem_file):
    #if EarthChem_all.csv does not exist, download it.
    if not os.path.isfile(earth_chem_file):
        if os.path.isfile('/data/EarthChem_all.csv'):
            earth_chem_file = '/data/EarthChem_all.csv'
        else:
            print(f'downloading {earth_chem_file}')
            myfile = requests.get('https://www.earthbyte.org/webdav/ftp/earthbyte/EarthChem/EarthChem_all.csv')
            open(earth_chem_file, 'wb').write(myfile.content)
    return earth_chem_file

# input: degrees between two points on sphere
# output: straight distance between the two points (assume the earth radius is 1)
# to get the kilometers, use the return value to multiply by the real earth radius
def degree_to_straight_distance(degree):
    return math.sin(math.radians(degree)) / math.sin(math.radians(90 - degree/2.))

def main(input_filename, output_filename_stem, variable_name, region, trench_points_filename=None):
    input_filename = get_data_file(input_filename)
    print('loading data...')
    data=pd.read_csv(input_filename, sep=',', skipinitialspace=True)
    c=variable_name
    #print(data.columns.values)
    a=data.columns.get_loc(c)
    df=data.iloc[:,[6,7,10,a]] 
    #6,7,10 are the default column numbers for Long, Lat and Age. Please check their position beforehand 
    df=df[(df!=0).all(1)]
    df=df.dropna()
    #print(len(df))

    # build the tree of the trench points
    t=0
    if not trench_points_filename:
        trench_points_filename = convergence_filename_template.format(time=t)
    data=np.loadtxt(trench_points_filename,skiprows=1, delimiter=',')
    #print(trench_points_filename)

    points_3d = [pygplates.PointOnSphere((row[1],row[0])).to_xyz() for row in data]
    points_tree = scipy.spatial.cKDTree(points_3d)

    data_index=[]
    index_count=-1
    candidates=[]
    for index, row in df.iterrows():
        index_count+=1
        try:
            candidates.append(pygplates.PointOnSphere((row[0], row[1])).to_xyz())#lat, lon
            data_index.append(index_count)
        except pygplates.InvalidLatLonError:
            #print('invalid lat or lon: ',row)
            continue

    print('query data...')
    dists, indices = points_tree.query(
                candidates, k=1, distance_upper_bound=degree_to_straight_distance(region))

    result_index = np.array(data_index)[indices<len(points_3d)]
    print(len(result_index))

    # create a point shapefile
    with shapefile.Writer(output_filename_stem) as sf:
        # for every record there must be a corresponding geometry.
        sf.autoBalance = 1

        # create the field names and data type for each.
        sf.field("AGE", "N")
        sf.field(c, "C")
        #"C": Characters, text.
        #"N": Numbers, with or without decimals.
        #"F": Floats (same as "N").
        #"L": Logical, for boolean True/False values.
        #"D": Dates.
        #"M": Memo, has no meaning within a GIS and is part of the xbase spec instead.
        print('saving file...')
        for idx in result_index:
            # create the point geometry
            sf.point(df.iloc[idx][1],df.iloc[idx][0]) #lon lat
            #print(df.iloc[idx][1],df.iloc[idx][0])
            # add attribute data
            sf.record(df.iloc[idx][2], df.iloc[idx][3])

    df.iloc[result_index].to_csv(output_filename_stem+".csv",index=False,float_format='%.4f')

    
def query(earth_chem_file, variable_name, poloygon_points):
    earth_chem_file = get_data_file(earth_chem_file)
    data=pd.read_csv(earth_chem_file, sep=',', skipinitialspace=True)
    df = data[['LATITUDE', 'LONGITUDE', 'AGE', variable_name]]
    df=df[(df!=0).all(1)]
    df=df.dropna()
    points_mask = Utils.check_points_within_polygon(df['LONGITUDE'], df['LATITUDE'], poloygon_points)
    return df[points_mask]
    
if __name__ == "__main__":
    __description__ = \
    """Extract data from EarthChem data. A shafefile and a csv file will be created for the extracted data. 
    
        Example: python extract_earth_chem.py EarthChem_all.csv output CU 5
    """
    
    # The command-line parser.
    parser = argparse.ArgumentParser(
        description = __description__, 
        formatter_class=argparse.RawDescriptionHelpFormatter)
    
    parser.add_argument('input_filename', type=str,
            help='the name of the EarthChem data file name')
    
    parser.add_argument('output_filename_stem', type=str,
            help='the name stem of the output file, {output_filename_stem}.shp and {output_filename_stem}.csv will be created.')

    parser.add_argument('variable_name', type=str,
            help='the name of the interesting variable')
    
    parser.add_argument('region', type=int,
            help='in degrees, the selected points will be inside this region.')

    parser.add_argument('-t', '--trench-points-filename', type=str, required=False,
            metavar='trench_points_filename',
            help='csv file contains the coordinates of trech points, same as convergence output at time 0')
    
    # Parse command-line options.
    args = parser.parse_args()
    
    
    main(args.input_filename, args.output_filename_stem, args.variable_name, args.region, args.trench_points_filename)

