
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




timeToOpen = '2017-07-26 20:43:03'

# Plot settings
plotContours = True
plotGrid = False
plotMaxDrawdown = False
numHydrograph = 0
modflowSilent = True
pumpingCosts = False

# Adjust heads using Theim?
adjustHead = False

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
with open('inputWellData_USGS.csv', 'rt') as csvfile:
    reader = csv.reader(csvfile, delimiter=',', quotechar='|')
    next(reader, None)  # skip header
    well_name = []
    well_number = []
    for row in reader:
        well_number.append(row[0])
        well_name.append(row[3])
numWells = len(well_number)


def adjustHeadTheim(headData, pump_rate, numWells, hk, nper, rw):
    #  Updates heads calculated by model to account for smaller well radius
    mf = flopy.modflow.Modflow('mod', exe_name='./mf2005dbl')
    [dis, _, _, _, _] = makeRiyadhGrid.build_dis_bas_files(mf, startingHead, perlen, nper, nstp, steady)
    re = 0.208 * dis.delc._array[0]
    headAdjusted = np.zeros_like(headData)  # time x numWells x model run
    numSamples = np.size(headData,2)
    for i in range(nper):
        for j in range(numWells):
            for k in range(numSamples):
                Q = -pump_rate[j]
                K = hk[k]
                h = headData[i, j, k]
                headAdjusted[i, j, k] = np.sqrt(h**2 - Q/(np.pi*K) * np.log(re/rw))
    return headAdjusted




# Plot hydrographs for a certain number of samples
if adjustHead:
    rw = 1000
    headData = adjustHeadTheim(headData, pump_rate, numWells, hk, nper, rw)

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

    # Build model and grid
    mf = flopy.modflow.Modflow('mod', exe_name='./mf2005dbl')
    [dis, bas, _, _, _] = makeRiyadhGrid.build_dis_bas_files(mf, startingHead, perlen, nper, nstp, steady)
    [mf, sr] = makeRiyadhGrid.build_spatial_reference(mf)

    # Plot grid
    fig = plt.figure(figsize=(14, 9.8))
    modelMap = flopy.plot.map.ModelMap(sr=sr, model=mf, dis=dis, layer=0, rotation=sr.rotation, length_multiplier=1.0, xul=sr.xul, yul=sr.yul)
    lineCollection = modelMap.plot_grid()
    lineCollection.zorder = 1

    # Plot a shapefile of well locations
    shp = 'wells'
    patch_collection_wells = modelMap.plot_shapefile(shp, radius=1000, edgecolor='black')
    patch_collection_wells.zorder = 3
    values = np.arange(0, 1, 1/numWells)
    cmap = ml.cm.get_cmap('spectral')
    patch_collection_wells.set(array=values, cmap=cmap)
    empty_patches = []
    for i in range(len(well_name)):
        empty_patches.append(patches.Patch(facecolor=cmap(values[i]), label=well_name[i]))
    plt.legend(handles=empty_patches, bbox_to_anchor=(1, 1), loc='upper left', ncol=1)


    # Plot outcrop
    shp = 'outcrop'
    patch_collection_outcrop= modelMap.plot_shapefile(shp, facecolor='brown', edgecolor='black', alpha=0.5)
    patch_collection_outcrop.zorder =2

    # Plot no flow cells (ibound)
    quadmesh = modelMap.plot_ibound(alpha=0.5, color_noflow='brown')

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
    fig = plt.figure()
    plt.title('Distribution of Head after 30 years by well')
    plt.xlabel('Well number')
    plt.ylabel('Head [meters above bottom of aquifer]')
    plt.boxplot([endDrawdowns[i, :] for i in range(numWells)])
    ax = plt.gca()
    ax.set_xticklabels(well_name)
    plt.show()
    fig.savefig('maxDrawdownDist' + saveName + '.pdf')



