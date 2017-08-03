#Import packages

import numpy as np
import scipy as sp
import matplotlib as ml
import os
import sys
import flopy
import flopy.utils.binaryfile as bf
import subprocess
import matplotlib.pyplot as plt
import shutil
from scipy.stats import uniform
import scipy.io as io
import datetime
import pickle
import makeRiyadhGrid
import plotFunctions
flopypth = os.path.join('..', '..', 'flopy')
if flopypth not in sys.path:
    sys.path.append(flopypth)
import time

start_time = time.time()


# Script Run paramters

# plot grid?
plotGrid = False

# Run MODFLOW?
runMODFLOW = True
modflowSilent = True

# Delete modflow files after use?
deleteFiles = False

# Save output?
saveOutput = True
numHeadFileSave = 1


# Build MODFLOW grid of Riyadh
[mf, pcg, wel, oc, dis, bas, nper, nstp, perlen, numWells, model_name, well_loc, pump_rate, steady, startingHead, _, rch] = makeRiyadhGrid.buildModel(plotGrid)


if runMODFLOW:

    # Sampling setup and options

    # Parameter ranges for sampling
    hk_min = 0.4 # 1.97e-3  # m/day
    hk_max = 2.5 # 8.34e-3
    hk_mean = 1.170
    vka_min = hk_min / 10
    vka_max = hk_max / 10
    sy_min = 0.02  # estimate .07
    sy_max = 3.e-1
    sy_mean = 0.13


    # Fixed input parameters vs. LHS sampled parameters vs. read parameters from file
    paramInput = False
    paramReadFile = False
    if paramInput and paramReadFile:
        raise NameError('Only one type of parameter input allowed')

    # Sample size (number of MODFLOW runs for different inputR parameters)
    sampleSize = 1
    if paramInput or paramReadFile:
        sampleSize = 1

    # Get/sample variable inputs
    if paramReadFile:
        with open('sampleDict.txt', 'rb') as handle:
            samples = pickle.loads(handle.read())
    elif paramInput:
        samples = {'hk': [hk_mean], 'sy': [sy_mean], 'vka': [hk_mean/10]}

    else:
        samples = makeRiyadhGrid.genParamSamples(sampleSize=sampleSize, hk=[hk_min, hk_max], vka=[vka_min, vka_max], sy=[sy_min, sy_max])
        samples['vka'] = samples['hk'] / 10

    # Write samples to text file
    with open('sampleDict.txt', 'wb') as handle:
        pickle.dump(samples, handle)


    # For each sample, generate LPF file and run MODFLOW

    # Get date and setup saving
    datetimeStr = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Initialize output dictionary and output parameters
    if saveOutput:
        param = {}
        output = {}
    modflow_success = []
    head_object = []
    headData = np.zeros((nstp*nper, numWells, sampleSize))
    timeSeries = np.zeros(nstp*nper)

    # Loop over each of the parameter samples
    for i in range(sampleSize):

        # Get hydraulic conductivty, specific yield samples for this iteration
        samplesThisRun = {k: v[i] for k, v in samples.items()}

        # Flopy LPF object
        [lpf, hk, vka, sy] = makeRiyadhGrid.build_lpf_file(mf, samples=samplesThisRun)

        # Write the model input files
        mf.write_input()

        # Run the model
        success, modflow_output = mf.run_model(silent=modflowSilent, pause=False, report=True)
        modflow_success.append(success)

        # Create MODFLOW output file if there was an error
        if not success:
            file = open('modflow_output' + str(i) + '.txt', 'w')
            for n in modflow_output:
                file.write(n + '\n')
            raise Warning('MODFLOW did not terminate normally.')

        # Create the headfile object
        headobj = bf.HeadFile(model_name+'.hds')
        head_object.append(headobj)

        # Save time series
        if i == 0:
            timeSeries = headobj.get_times()

        # Get hydrograph data
        for n in range(numWells):
            headTemp = headobj.get_ts(tuple(well_loc[n]))
            headData[:,n,i] = headTemp[:,1]     # time x numWells x model run

        # Delete MODFLOW input and output files
        saveTime = datetimeStr

        if deleteFiles:
            if i < numHeadFileSave:
                os.rename(model_name + '.hds', 'headData' + saveTime + '.hds')
            try:
                os.remove(model_name + '.hds')
            except OSError:
                pass
            os.remove(model_name + '.bas')
            os.remove(model_name + '.dis')
            os.remove(model_name + '.list')
            os.remove(model_name + '.lpf')
            os.remove(model_name + '.nam')
            os.remove(model_name + '.oc')
            os.remove(model_name + '.pcg')
            os.remove(model_name + '.wel')
            os.remove(model_name + '.rch')


    np.savez('output_' + saveTime, time=timeSeries, numWells=numWells, pump_rate=pump_rate, well_loc=well_loc, nper=nper, nstp=nstp, steady=steady,
             perlen=perlen, startingHead=startingHead, headData=headData, hk=samples['hk'],  vka=samples['vka'], sy=samples['sy'], modflow_success=modflow_success)

print("--- %s seconds ---" % (time.time() - start_time))



