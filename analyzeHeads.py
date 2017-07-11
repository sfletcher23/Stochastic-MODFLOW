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
flopypth = os.path.join('..', '..', 'flopy')
if flopypth not in sys.path:
    sys.path.append(flopypth)


# Script Run paramters

# Plot settings
plotContours = False
plotHydrograph = True
modflowSilent = True
pumpingCosts = False

# Delete modflow files after use?
deleteFiles = True

# Save output?
saveOutput = True



#Sampling setup and options

# Parameter ranges for sampling
hk_min = 1.97e-3    # m/day
hk_max = 8.34e-3
vka_min = hk_min / 10
vka_max = hk_max / 10
sy_min = 1.e-1
sy_max = 3.e-1

# Fixed input parameters vs. LHS sampled parameters vs. read parameters from file
paramInput = False
paramReadFile = False
if paramInput and paramReadFile:
    raise NameError('Only one type of parameter input allowed')

# Sample size (number of MODFLOW runs for different input parameters)
sampleSize = 2
if paramInput or paramReadFile:
    sampleSize = 1

# Get/sample variable inputs
if paramReadFile:
    with open('sampleDict.txt', 'rb') as handle:
        samples = pickle.loads(handle.read())
elif paramInput:
    samples = {}
else:
    samples = makeRiyadhGrid.genParamSamples(sampleSize=sampleSize, hk=[hk_min, hk_max], vka=[vka_min, vka_max], sy=[sy_min, sy_max])

# Write samples to text file
with open('sampleDict.txt', 'wb') as handle:
  pickle.dump(samples, handle)



# Pumping cost paramters
electricityCost = 0.12  # $/kWh
pumpEfficiency = 0.80



# Time step parameters
perlen = 365
nper = 30    # number of stress periods
nstp = 365      # Number of time steps per stress period
steady = [False] * nper
pumpingOn = [1] * nper



# Well parameters
numWells = 3
pump_rate = [-10000, -5000, - 7000]
well_loc = [[0, 20-1, 37-1], [0, 37-1, 18-1], [0, 5-1, 10-1]]
startingHead = 200
# check that number of pumpnig rates = number of wells
if not len(pump_rate) == numWells:
    raise NameError('Number of pumping rates must equal number of wells')
# check that number of locations = number of wells
if not len(well_loc) == numWells:
    raise NameError('Number of well locations must equal number of wells')



# Generate static MODFLOW input files

# Name model
model_name = 'mod'

# Stress period dictionary and well, oc objects
well_sp_data = {}
for n in range(numWells):
    well_sp_data[n] = well_loc[n][:]
    well_sp_data[n].append(pump_rate[n])  # Remember to use zero-based layer, row, column indices

stress_period_data = {}
for t in range(nper):
    stress_period_data[t] = []
    for n in range(numWells):
        temp_sp_data = well_sp_data[n][:]
        temp_sp_data[-1] *= pumpingOn[t]
        stress_period_data[t].append(temp_sp_data)

# MODFLOW input files
mf = flopy.modflow.Modflow(model_name, exe_name='./mf2005dbl')
[dis, bas, nper, nstp, perlen] = makeRiyadhGrid.build_dis_bas_files(mf, startingHead, perlen, nper, nstp, steady)
pcg = makeRiyadhGrid.build_pcg_file(mf)
wel = flopy.modflow.ModflowWel(mf, stress_period_data=stress_period_data)
oc = flopy.modflow.ModflowOc(mf, stress_period_data={(0, 0): ['print head', 'save head']})  # output control



# For each sample, generate LPF file and run MODFLOW

# Get date and setup saving
datetimeStr = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# Initialize output dictionary and output parameters
if saveOutput:
    output = {}
modflow_success = []
head_object = []
timeSeries = np.zeros([sampleSize, nstp*nper])
    
# Loop over each of the parameter samples
for i in range(sampleSize):

    # Get hydraulic conductivty, specific yield samples for this iteration
    samplesThisRun = {k: v[i] for k, v in samples.items()}

    # Flopy LPF object
    lpf = makeRiyadhGrid.build_lpf_file(mf, samples=samplesThisRun)

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
    timeSeries[i, :] = headobj.get_times()

    # Get hydrograph data
    headData = {}
    for n in range(numWells):
        headData[n] = headobj.get_ts(tuple(well_loc[n]))

    # Calculate pumping costs
    if pumpingCosts:
        kWhperJ = 2.77778e-7  # Convert from Joules to kWh
        mgh = {}
        pumpCosts = {}
        for n in range(numWells):
            mgh = 1000 * 9.8 * (ztop - headData[n][:, 1])  # 1000 kg per m^3 water * g * drawdown --> Joules per m^3
            pumpCosts[n] = mgh * kWhperJ * 1/pumpEfficiency * electricityCost  #* -pump_rate_1
            pumpOnTimes = np.zeros_like(headData[n][:, 1])
            for index, item in enumerate(pumpingOn):
                if item == 1:
                    pumpOnTimes[index * nstp: index * nstp + nstp] = 1
            pumpCosts[n] *= pumpOnTimes



    # Plot hydrograph: head versus time
    if plotHydrograph:
        fig1 = plt.figure()
        ax1 = fig1.add_subplot(111)
        ttl = 'Hydrograph: K = {:.2e} , vK = {:.2e}, sy = {:.2e}'.format(samplesThisRun['hk'], samplesThisRun['vka'], samplesThisRun['sy'])
        ax1.set_title(ttl)
        ax1.set_xlabel('time [years]')
        ax1.set_ylabel('head')
        ax1.set_ylim(-1,1000)
        for n in range(numWells):
            ax1.plot(headData[n][:, 0]/365, headData[n][:, 1], label='Well {}: Pump rate = {} m^3/d'.format(n, pump_rate[n]))
        plt.legend()

        if pumpingCosts:
            ax2 = ax1.twinx()
            for n in range(numWells):
                ax2.plot(headData[n][:, 0]/365, pumpCosts[n], '--',  label='Well {} Pumping costs'.format(n))
            ax2.set_ylabel('Pumping costs [$/m^3]')

        plt.legend()
        plt.show()
        fig1.savefig('hydrograph' + str(i) + '.pdf')

    # Plot contour map
    if plotContours:

        # Setup contour parameters
        levels = np.arange(1, 1010, 20)
        extent = (delr/2., Lx - delr/2., delc/2., Ly - delc/2.)

        # Make the plots
        mytimes = timeSeries[i, -1]
        if not isinstance(timeSeries, list):
            head = headobj.get_data(totim=mytimes)
            iplot = 0
            t = mytimes
        else:
            for iplot, t in enumerate(mytimes):
                head = headobj.get_data(totim=t)

        #Create the plot
        fig2 = plt.figure()
        ax2 = fig2.add_subplot(1, 1, 1, aspect='equal')
        ax2.set_title('stress period ' + str(iplot + 1) + 't = ' + str(t))
        ax2.imshow(head[0, :, :], extent=extent, cmap='BrBG', vmin=0., vmax=10.)
        CS = plt.contour(np.flipud(head[0, :, :]), levels=levels, extent=extent,
                            zorder=10)
        plt.clabel(CS, inline=1, fontsize=10, fmt='%1.1f', zorder=11)
        plt.colorbar()
        plt.savefig('contour' + str(i) + '.pdf')
        plt.show()

    # Calculate maximum drawdown across all wells
    endDrawdowns = [x[-1][1] for ind, x in enumerate(headData.values())]
    maxDrawdownWell = np.argmin(endDrawdowns)
    maxDrawdown = min(endDrawdowns)

    if deleteFiles:
        os.remove(model_name + '.bas')
        os.remove(model_name + '.dis')
        os.remove(model_name + '.hds')
        os.remove(model_name + '.list')
        os.remove(model_name + '.lpf')
        os.remove(model_name + '.nam')
        os.remove(model_name + '.oc')
        os.remove(model_name + '.pcg')
        os.remove(model_name + '.wel')

    # Save output in .mat file
    if saveOutput:
        output[i] = dict(zip(['timeSeries', 'modflow_success', 'hk', 'vka', 'sy', 'maxDrawdown', 'maxDrawdownWell'],
                             [timeSeries, modflow_success, samplesThisRun[hk], samplesThisRun[vka], samplesThisRun[sy], maxDrawdown, maxDrawdownWell]))
        for n in range(numWells):
            output[i]['head_well_{}'.format(n)] = headData[n][:, 1]

# print maxDrawdown array
maxDrawdownAll = [v['maxDrawdown'] for k, v in output.items()]
maxDrawdownWellAll = [v['maxDrawdownWell'] for k, v in output.items()]
print([maxDrawdownAll, maxDrawdownWellAll])

# Write output to text file
with open('output_{}.txt'.format(datetimeStr), 'wb') as handle:
    pickle.dump(output, handle)

        # io.savemat('modflowData' + datetimeStr, output)




