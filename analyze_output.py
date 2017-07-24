
import numpy as np
import plotFunctions
import flopy
import flopy.utils.binaryfile as bf
import makeRiyadhGrid
import matplotlib as ml
import matplotlib.pyplot as plt
from flopy.utils.geometry import Polygon, LineString, Point
from flopy.utils.reference import SpatialReference
from flopy.export.shapefile_utils import recarray2shp, shp2recarray
from flopy.utils.modpathfile import PathlineFile, EndpointFile
from flopy.utils.reference import epsgRef
import shapely
import os
import utm




timeToOpen = '2017-07-19 10:44:41'

# Plot settings
plotContours = True
plotGrid = True
plotMaxDrawdown = False
numHydrograph = 0
modflowSilent = True
pumpingCosts = False

modData = np.load('output_' + timeToOpen + '.npz')

# Unpack modData
timeSeries = modData['time']
numWells = np.asscalar(modData['numWells'])
pump_rate = modData['pump_rate']
well_loc = modData['well_loc']
nper = np.asscalar(modData['nper'])
perlen =np.asscalar(modData['perlen'])
nstp = np.asscalar(modData['nstp'])
steady = modData['steady']
headData = modData['headData']
startingHead = np.asscalar(modData['startingHead'])
hk = modData['hk']
vka = modData['vka']
sy = modData['sy']
modflow_success = modData['modflow_success']

# Plot hydrographs for a certain number of samples
for i in range(numHydrograph):
    # Get parameter and head data
    headDataSample = headData[:, :, i]
    hkSample = hk[i]
    vkaSample = vka[i]
    sySample = sy[i]
    saveName = timeToOpen + '_' + str(i)
    # Call hydrograph function to make plot
    plotFunctions.hydrograph(headDataSample, timeSeries, hkSample, vkaSample, sySample, numWells, pump_rate, saveName, startingHead)


if plotContours:
    # Get head object from file
    head_object = []
    headobj = bf.HeadFile('headData' + timeToOpen + '.hds')
    head_object.append(headobj)
    # Make model to plot
    mf = flopy.modflow.Modflow('mod', exe_name='./mf2005dbl')
    [dis, _, _, _, _] = makeRiyadhGrid.build_dis_bas_files(mf, startingHead, perlen, nper, nstp, steady)
    [mf, sr] = makeRiyadhGrid.build_spatial_reference(mf)
    [wel, _, _, _] = makeRiyadhGrid.build_wel_file(mf, sr)
    # Plot well locations and Riyadh?
    plotwellriyadh = True
    # Call countour function to make plot
    saveName = timeToOpen
    plotFunctions.contour(headobj, timeSeries, mf, sr, wel, dis, plotwellriyadh, saveName)


# Plot model grid
# Create modflow option and dis file - these are the same across all samples, so okay to recreate here
if plotGrid:
    mf = flopy.modflow.Modflow('mod', exe_name='./mf2005dbl')
    [dis, _, _, _, _] = makeRiyadhGrid.build_dis_bas_files(mf, startingHead, perlen, nper, nstp, steady)
    [mf, sr] = makeRiyadhGrid.build_spatial_reference(mf)
    modelMap = flopy.plot.map.ModelMap(sr=sr, model=mf, dis=dis, layer=0, rotation=sr.rotation, length_multiplier=1.0, xul=sr.xul, yul=sr.yul)
    lineCollection = modelMap.plot_grid()
    lineCollection.zorder = 1

    # Plot a shapefile of well locations
    shp = 'wells'
    patch_collection_wells = modelMap.plot_shapefile(shp, radius=2000, facecolor='red', edgecolor='red')
    patch_collection_wells.zorder = 3

    # Plot Royadh location
    shp = 'Riyadh'
    patch_collection_riyadh = modelMap.plot_shapefile(shp, radius=5000, facecolor='green')
    patch_collection_riyadh.zorder =2

    plt.show()


    # epsg=4326

# Calculate maximum drawdown across all wells
if plotMaxDrawdown:
    endDrawdowns = headData[-1, :, :]   # last time period for all wells for all samples
    maxDrawdown = np.amin(endDrawdowns,0)
    maxDrawdownWell = np.argmin(endDrawdowns,0)
    plt.figure()
    plt.title('Distribution of Drawdown after 30 years by well')
    plt.xlabel('Well number')
    plt.ylabel('Drawdown after 30 yeras')
    plt.boxplot([endDrawdowns[i, :] for i in range(numWells)])
    plt.show()
#
mf.sr = SpatialReference(delr=mf.dis.delr, delc=mf.dis.delc, xul=603224.64, yul=2433164.43,
                         proj4_str='UTM38Q', rotation=22.3)
# chk = dis.check()
# chk.summary_array
# get_vertices = mf.sr.get_vertices  # function to get the referenced vertices for a model cell
# geoms = Polygon(get_vertices(5, 4))
# print(geoms.type)
# print(geoms.exterior)
# geoms.plot()




