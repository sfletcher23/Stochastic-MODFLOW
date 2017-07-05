
#Import packages
import numpy as np
import scipy as sp
import matplotlib as ml
import os
import sys
flopypth = os.path.join('..', '..', 'flopy')
if flopypth not in sys.path:
    sys.path.append(flopypth)
import flopy
import flopy.utils.binaryfile as bf
import subprocess
import matplotlib.pyplot as plt
import shutil
from scipy.stats import uniform
import scipy.io as io
import datetime
from scipy.stats import uniform
from pyDOE import *
import pickle


# Script Run paramters

# Plot settings
plotContours = True
plotHydrograph = True
modflowSilent = True
pumpingCosts = False

# Delete modflow files after use?
deleteFiles = True

# Save output?
saveOutput = False

# Fixed input parameters vs. LHS sampled parameters vs. read parameters from file
paramInput = True
paramReadFile = False
if paramInput and paramReadFile:
    raise NameError('Only one type of parameter input allowed')


# Sample size (number of MODFLOW runs for different input parameters)
sampleSize = 1
if paramInput or paramReadFile:
    sampleSize = 1


# Write function to generate range of inputs for hk, vka, pumping rate in each well using latin hypercube sampling
def genParamSamples(numWells, sampleSize, hk_min=hk, hk_max=hk, vka_min=vka, vka_max=vka, sy_min=sy, sy_max=sy):
    # Generate list of all paramters
    params_in_sample = ['hk', 'vka', 'time']
    for n in range(numWells):
        params_in_sample.append('pump_rate_' + str(n + 1))

    # Generate LHS samples
    numParam = np.size(params_in_sample)
    lhd = lhs(numParam, samples=sampleSize)

    # Generate arrays of hk and vka
    loc = hk_min
    scale = hk_max - hk_min
    hk = uniform(loc=loc, scale=scale).ppf(lhd[:, 0])
    loc = vka_min
    scale = vka_max - vka_min
    vka = uniform(loc=loc, scale=scale).ppf(lhd[:, 1])

    # Generate arrays of time (length of stress period)
    loc = time_min
    scale = time_max - time_min
    time = uniform(loc=loc, scale=scale).ppf(lhd[:, 2])

    # Generate arrays of pumping rate
    pump = np.zeros([numWells, sampleSize])
    loc = pump_min
    scale = pump_max - pump_min
    for n in range(numWells):
        pump[n, :] = uniform(loc=loc, scale=scale).ppf(lhd[:, 3 + n])

    # Combine to form paramSample
    pumpSplit = np.split(pump, numWells)
    param_sample = np.stack([hk, vka, time])
    for i in range(numWells):
        param_sample = np.append(param_sample, pumpSplit[i], axis=0)

    # Create dictionary with samples for each parameter
    params = dict(zip(params_in_sample, param_sample))

    return params


# Pumping cost paramters
electricityCost = 0.12  # $/kWh
pumpEfficiency = 0.80



# MODFLOW input parameters
numWells = 2

# Parameter sample ranges
hk = 1.7e-2
vka = hk/10
pump_rate_1 = -10000
pump_rate_2 = -5000
perlen = 365


# Parameter sample ranges
hk_min = 1.e-3
hk_max = 1.
vka_min = 1.e-3
vka_max = 1.e-2
pump_min = -50000.
pump_max = -3000.
perlen_max = 3000     # length of stress period
perlen_min = 10

# Get/sample variable inputs
if paramReadFile:
    with open('sampleDict.txt', 'rb') as handle:
        samples = pickle.loads(handle.read())
elif paramInput:
    samples = {'hk': [hk], 'vka': [vka], 'pump_rate_1': [pump_rate_1], 'pump_rate_2': [pump_rate_2], 'time': [perlen]}
else:
    samples = genParamSamples(hk_min=hk_min, hk_max=hk_max,vka_min=vka_min, vka_max=vka_max, numWells=numWells,
                          pump_min=pump_min, pump_max=-pump_max, time_min=perlen_min, time_max=perlen_max, sampleSize=sampleSize)

# Write samples to text file
with open('sampleDict.txt', 'wb') as handle:
  pickle.dump(samples, handle)

for keys,values in samples.items():
    print(keys)
    print(values)


# Fixed Parameter Definitions
# Model domain and grid definition
Lx = 5000.
Ly = 1000.
ztop = 1000.
zbot = 0.
nlay = 1
nrow = 50
ncol = 50
delr = Lx / ncol
delc = Ly / nrow
delv = (ztop - zbot) / nlay
botm = np.linspace(ztop, zbot, nlay + 1)
sy = .25
ss = 4.e-7*10
laytyp = 1  # 1 = unconfined, 0 = confined
hdry = 0    # dry cell head set to this number
mxiter = 300
hclose = 1e-1

# Variables for the BAS package
ibound = np.ones((nlay, nrow, ncol), dtype=np.int32)    # Not sure if GW uses 0 or 1
strt = 1000 * np.ones((nlay, nrow, ncol), dtype=np.float32)     # Starting head

# Time step parameters
nper = 5    # number of stress periods
nstp = 365      # Number of time steps per stress period
steady = [False, False, False, False, False]
pumpingOn = [True, True, True, False, False]

# Well locations
wpt1 = (0, 20-1, 37-1)
wpt2 = (0, 37-1, 18-1)

# Output control
spd = {(0, 0): ['print head', 'save head']}


# Define output parameters for each run
modflow_success = []
head_object = []
head_data1 = np.zeros([sampleSize, nstp*nper])
head_data2 = np.zeros([sampleSize, nstp*nper])
timeSeries = np.zeros([sampleSize, nstp*nper])

# Get date and setup saving
datetimeStr = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
# Run MODFLOW for each of the paramter samples
for i in range(sampleSize):

    # Name model this run
    model_name = 'model' + str(i)

    # Get pumping rate same and make stress period dictionary:
    pumping_rate1 = samples['pump_rate_1'][i]
    pumping_rate2 = samples['pump_rate_2'][i]
    wel_sp1 = [0, 19, 36, pumping_rate1] # Remember to use zero-based layer, row, column indices!
    wel_sp2 = [0, 36, 17, pumping_rate2]
    stress_period_data = {}
    for t in range(nper):
        tempWell1 = wel_sp1[:]
        tempWell2 = wel_sp2[:]
        if not pumpingOn[t]:
            tempWell1[-1] = 0
            tempWell2[-1] = 0
        stress_period_data[t] = [tempWell1, tempWell2]

    # Get hydraulic conductivty sample
    hk = samples['hk'][i]
    vka = samples['vka'][i]

    # Get perlen sample
    perlen = samples['time'][i]
    if nper > 1:
        perlen = [perlen] * nper

    # Flopy objects
    mf = flopy.modflow.Modflow(model_name, exe_name='./mf2005dbl')
    dis = flopy.modflow.ModflowDis(mf, nlay, nrow, ncol, delr=delr, delc=delc,
                                   top=ztop, botm=botm[1:],
                                   nper=nper, perlen=perlen, nstp=[nstp]*nper, steady=steady)
    bas = flopy.modflow.ModflowBas(mf, ibound=ibound, strt=strt)
    lpf = flopy.modflow.ModflowLpf(mf, hk=hk, vka=vka, sy=sy, ss=ss, laytyp=laytyp, hdry=hdry)
    pcg = flopy.modflow.ModflowPcg(mf, mxiter=mxiter, hclose=hclose)  # This has stuff like iterations for solver
    wel = flopy.modflow.ModflowWel(mf, stress_period_data=stress_period_data)
    oc = flopy.modflow.ModflowOc(mf, stress_period_data=spd)

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
    ts1 = headobj.get_ts(wpt1)
    ts2 = headobj.get_ts(wpt2)

    # Calculate pumping costs
    if pumpingCosts:
        mgh1 = 1000 * 9.8 * (ztop - ts1[:, 1])   # 1000 kg per m^3 water * g * drawdown --> Joules per m^3
        mgh2 = 1000 * 9.8 * (ztop - ts2[:, 1])
        kWhperJ = 2.77778e-7
        pumpCostsWell1 = mgh1 * kWhperJ * 1/pumpEfficiency * electricityCost #* -pumping_rate1
        pumpCostsWell2 = mgh2 * kWhperJ * 1/pumpEfficiency * electricityCost #* -pumping_rate2
        pumpOnTimes1 = np.zeros_like(ts1[:, 1])
        pumpOnTimes2 = np.zeros_like(ts2[:, 1])
        #for t in range(len(ts1[:,1])):
        for index, item in enumerate(pumpingOn):
            if item:
                pumpOnTimes1[index * nstp: index * nstp + nstp] = 1
                pumpOnTimes2[index * nstp: index * nstp + nstp] = 1
        pumpCostsWell1 = pumpCostsWell1 * pumpOnTimes1
        pumpCostsWell2 = pumpCostsWell2 * pumpOnTimes2


    # Plot hydrograph: head versus time
    if plotHydrograph:
        fig1 = plt.figure()
        ax1 = fig1.add_subplot(111)
        ttl = 'Hydrograph: K = {:.2e} , vK = {:.2e}, ss = {:.2e}, sy = {:.2e}'.format(hk,vka, ss, sy)
        ax1.set_title(ttl)
        ax1.set_xlabel('time [days]')
        ax1.set_ylabel('head')
        ax1.set_ylim(-1,1000)
        ax1.plot(ts1[:, 0], ts1[:, 1], label='Well 1: Pump rate = ' + str(round(pumping_rate1,3)) + ' cm/d')
        ax1.plot(ts2[:, 0], ts2[:, 1], label='Well 2: Pump rate = ' + str(round(pumping_rate2,3)) + ' cm/d')
        plt.legend()

        if pumpingCosts:
            ax2 = ax1.twinx()
            ax2.plot(ts1[:, 0], pumpCostsWell1, '--',  label='Well 1 Pumping costs')
            ax2.plot(ts2[:, 0], pumpCostsWell2, '--',  label='Well 2 Pumping costs')
            ax2.set_ylabel('Pumping costs [$/day]')

        plt.legend()
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



    # Write hydrograph data to array
    head_data1[i, :] = ts1[:, 1]
    head_data2[i, :] = ts2[:, 1]

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
        outputDic = dict(zip(['head_data1', 'head_data2', 'timeSeries', 'modflow_success'],
                             [head_data1, head_data2, timeSeries, modflow_success]))
        outputDic.update(samples)
        io.savemat('modflowData' + datetimeStr, outputDic)




