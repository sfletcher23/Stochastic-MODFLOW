
import numpy as np
import plotFunctions
import flopy
import flopy.utils.binaryfile as bf
import makeRiyadhGrid
import matplotlib as ml
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from flopy.utils.geometry import Polygon, LineString, Point
from flopy.utils.reference import SpatialReference
from flopy.export.shapefile_utils import recarray2shp, shp2recarray
from flopy.utils.modpathfile import PathlineFile, EndpointFile
from flopy.utils.reference import epsgRef
import shapely
import os
import utm
import csv




timeToOpen = '2017-07-25 10:57:33'

# Plot settings
plotContours = False
plotGrid = False
plotMaxDrawdown = True
numHydrograph = 1
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


# Get well names
with open('inputWellData.csv', 'rt') as csvfile:
    reader = csv.reader(csvfile, delimiter=',', quotechar='|')
    next(reader, None)  # skip header
    name = []
    for row in reader:
        name.append(row[0])

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
    [mf, _, wel, _, dis, _, _, _, _, _, _, _, _, _, _, sr] = makeRiyadhGrid.buildModel(False)
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
    patch_collection_wells = modelMap.plot_shapefile(shp, radius=3000, edgecolor='black')
    patch_collection_wells.zorder = 3
    values = np.arange(0, 1, 1/13)
    cmap = ml.cm.get_cmap('gist_ncar')
    patch_collection_wells.set(array=values, cmap=cmap)
    empty_patches = []
    for i in range(len(name)):
        empty_patches.append(patches.Patch(facecolor=cmap(values[i]), label=name[i]))
    plt.legend(handles=empty_patches)

    # Plot Royadh location
    shp = 'Riyadh'
    patch_collection_riyadh = modelMap.plot_shapefile(shp, radius=7000, facecolor='black', edgecolor='black')
    patch_collection_riyadh.zorder =2

    plt.show()




# Calculate maximum drawdown across all wells
if plotMaxDrawdown:
    endDrawdowns = headData[-1, :, :]   # last time period for all wells for all samples
    maxDrawdown = np.amin(endDrawdowns,0)
    maxDrawdownWell = np.argmin(endDrawdowns,0)
    plt.figure()
    plt.title('Distribution of Head after 30 years by well')
    plt.xlabel('Well number')
    plt.ylabel('Head [meters above bottom of aquifer]')
    plt.boxplot([endDrawdowns[i, :] for i in range(numWells)])
    ax = plt.gca()
    ax.set_xticklabels(name)
    plt.show()




