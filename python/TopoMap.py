'''
    Assign colour to grid data
    Set georeferenced extends of the images
    gdalbuildvrt topo15.vrt *.tif
'''

import struct, math, gzip, os
import numpy
import matplotlib.pyplot as plt
import matplotlib
from pylab import *
from matplotlib.colors import LinearSegmentedColormap

def set_shade(a,intensity=None,cmap=plt.cm.jet,scale=10.0,azdeg=165.0,altdeg=45.0):
    ''' sets shading for data array based on intensity layer or the data's value itself.
    inputs:
        a - a 2-d array or masked array
        intensity - a 2-d array of same size as a (no chack on that)
                    representing the intensity layer. if none is given
                    the data itself is used after getting the hillshade values
                    see hillshade for more details.
        cmap - a colormap (e.g matplotlib.colors.LinearSegmentedColormap
              instance)
        scale,azdeg,altdeg - parameters for hilshade function see there for
              more details
    output:
      rgb - an rgb set of the Pegtop soft light composition of the data and 
           intensity can be used as input for imshow()
    based on ImageMagick's Pegtop_light:
    http://www.imagemagick.org/Usage/compose/#pegtoplight'''
    if intensity is None:
        # hilshading the data
        intensity = hillshade(a,scale=10.0,azdeg=165.0,altdeg=45.0)
    else:
        # or normalize the intensity
        intensity = (intensity - intensity.min())/(intensity.max() - intensity.min())
        
    # get rgb of normalized data based on cmap
    
    '''
    PAY ATTENTION HERE. UGLY HACK TO MAKE IT WORK WITH TOPOGRAPHY DATA
    '''
    #rgb = cmap((a-a.min())/float(a.max()-a.min()))[:,:,:3]
    rgb = cmap((a+10927.5)/float(8726+10927.5))[:,:,:3]
    
    # form an rgb eqvivalent of intensity
    d = intensity.repeat(3).reshape(rgb.shape)
    # simulate illumination based on pegtop algorithm.
    rgb = 2*d*rgb+(rgb**2)*(1-2*d)
    return rgb

def hillshade(data,scale=10.0,azdeg=165.0,altdeg=45.0):
    ''' convert data to hillshade based on matplotlib.colors.LightSource class.
    input: 
         data - a 2-d array of data
         scale - scaling value of the data. higher number = lower gradient 
         azdeg - where the light comes from: 0 south ; 90 east ; 180 north ;
                      270 west
         altdeg - where the light comes from: 0 horison ; 90 zenith
    output: a 2-d array of normalized hilshade 
    '''
    # convert alt, az to radians
    az = azdeg*pi/180.0
    alt = altdeg*pi/180.0
    # gradient in x and y directions
    dx, dy = gradient(data/float(scale))
    slope = 0.5*pi - arctan(hypot(dx, dy))
    aspect = arctan2(dx, dy)
    intensity = sin(alt)*sin(slope) + cos(alt)*cos(slope)*cos(-az - aspect - 0.5*pi)
    intensity = (intensity - intensity.min())/(intensity.max() - intensity.min())
    return intensity

def get_rgb(r):
    '''
    You must be careful with the Nan data in the grid.
    '''
    where_are_NaNs = numpy.isnan(r)
    r[where_are_NaNs] = 0

    colors=[
        [10,0,121],
        [26,0,137],
        [38,0,152],
        [27,3,166],
        [16,6,180],
        [5,9,193],
        [0,14,203],
        [0,22,210],
        [0,30,216],
        [0,39,223],
        [12,68,231],
        [26,102,240],
        [19,117,244],
        [14,133,249],
        [21,158,252],
        [30,178,255],
        [43,186,255],
        [55,193,255],
        [65,200,255],
        [79,210,255],
        [94,223,255],
        [138,227,255],
        [188,230,255],
        [51,102,0],
        [51,204,102],
        [187,228,146],
        [255,220,185],
        [243,202,137],
        [230,184,88],
        [217,166,39],
        [168,154,31],
        [164,144,25],
        [162,134,19],
        [159,123,13],
        [156,113,7],
        [153,102,0],
        [162,89,89],
        [178,118,118],
        [183,147,147],
        [194,176,176],
        [204,204,204],
        [229,229,229],
    ]

    steps = [ 
       -10927,-10500,-10000,-9500,-9000,-8500,-8000,-7500,-7000,-6500,-6000,-5500,-5000,-4500,-4000,-3500,-3000,-2500,-2000,-1500,
        -1000,-500,-0.001,-0.0005, 100,200,500,1000,1500,2000,2500,3000,3500,4000,4500,5000,5500,6000,6500,7000,7500,8726
    ]

    color_list = []
    #print range(len(steps))
    #print len(steps)
    for i in range(len(steps)):
        color_list.append((float(steps[i]-steps[0])/(steps[-1]-steps[0]), [x/255.0 for x in colors[i]]))

    #print color_list
    cmap = LinearSegmentedColormap.from_list('topo', color_list, N=1024)
    rgb = set_shade(r,cmap=cmap)
    return rgb
