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
deleteFiles = True

# Save output?
saveOutput = True
numHeadFileSave = 1


# Build MODFLOW grid of Riyadh
[mf, pcg, wel, oc, dis, bas, nper, nstp, perlen, numWells, model_name, well_loc, pump_rate, steady, startingHead, _, rch, well_num] = makeRiyadhGrid.buildModel(plotGrid)


if runMODFLOW:

    # Sampling setup and options

    # Parameter ranges for sampling
    hk_min = 0.9   # m/day
    hk_max = 40
    vka_min = hk_min / 10
    vka_max = hk_max / 10
    ss_min = 0.5e-6  # estimate .07
    ss_max = 2.5e-5
    hk_input = 3
    ss_input = ss_min*1.4


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
        samples = {'hk': [hk_input], 'ss': [ss_input], 'vka': [hk_input/10]}

    else:
        samples = makeRiyadhGrid.genParamSamples(sampleSize=sampleSize, hk=[hk_min, hk_max], vka=[vka_min, vka_max], ss=[ss_min, ss_max])
        samples['vka'] = samples['hk'] / 10

    # Write samples to text file
    with open('sampleDict.txt', 'wb') as handle:
        pickle.dump(samples, handle)


    # For each sample, generate LPF file and run MODFLOW

    # Get date and setup saving
    datetimeStr = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    saveTime = datetimeStr
    runningSlurm = False
    if os.environ.get('SLURM_JOB_ID') != None:
        outputName = 'simulation_data/output_' + saveTime + '_' + os.environ.get('SLURM_ARRAY_JOB_ID') + '_' + os.environ.get('SLURM_ARRAY_TASK_ID')
        runningSlurm = True
    else:
        outputName = 'simulation_data/output_' + saveTime

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
        print(samplesThisRun)

        # Flopy LPF object
        [lpf, hk, vka, ss] = makeRiyadhGrid.build_lpf_file(mf, samples=samplesThisRun)

        # Write the model input files
        mf.write_input()
        # if false runningSlurm:
        #     from shutil import copyfile
        #     copyfile('mod_correct.wel', model_name + '.wel')

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

    np.savez(outputName, time=timeSeries, numWells=numWells, pump_rate=pump_rate, well_loc=well_loc, nper=nper, nstp=nstp, steady=steady, well_num=well_num,
             perlen=perlen, startingHead=startingHead, headData=headData, hk=samples['hk'],  vka=samples['vka'], ss=samples['ss'], modflow_success=modflow_success)

print("--- %s seconds ---" % (time.time() - start_time))



