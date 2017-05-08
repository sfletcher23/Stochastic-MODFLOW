
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


# Write function to generate range of inputs for hk, vka, pumping rate in each well
def genParamSamples(hk_min, hk_max, vka_min, vka_max, numWells, pump_min, pump_max, sampleSize):
    # Generate list of all paramters
    params_in_sample = ['hk', 'vka']
    for n in range(numWells):
        params_in_sample.append('pump_rate_' + str(n + 1))

    # Generate arrays of hk and vka
    hk = np.linspace(hk_min, hk_max, num=sampleSize)
    vka = np.linspace(vka_min, vka_max, num=sampleSize)

    # Generate arrays of pumping rate
    pump = np.zeros([numWells, sampleSize])
    for n in range(numWells):
        pump[n, :] = np.linspace(pump_min, pump_max, num=sampleSize)

    # Combine to form paramSample
    pumpSplit = np.split(pump, numWells)
    param_sample = np.stack([hk, vka])
    for i in range(numWells):
        param_sample = np.append(param_sample, pumpSplit[i], axis=0)

    # Create dictionary with samples for each paramter
    params = dict(zip(params_in_sample, param_sample))

    return params


# Plot settings
plotContours = True
plotHydrograph = True


# Parameter sample inputs
hk_min = 1.e-3
hk_max = 1.
vka_min = 1.e-3
vka_max = 1.e-2
numWells = 2
pump_min = -50000.
pump_max = -3000.
sampleSize = 10

# Fixed Parameter Definitions
# Model domain and grid definition
Lx = 1000.
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
sy = 2.5e-1
ss = 4.e-7
laytyp = 1  # 1 = unconfined, 0 = confined
hdry = 0    # dry cell head set to this number
mxiter = 100
hclose = 1e-3

# Variables for the BAS package
ibound = np.ones((nlay, nrow, ncol), dtype=np.int32)    # Not sure if GW uses 0 or 1
strt = 1000 * np.ones((nlay, nrow, ncol), dtype=np.float32)     # Starting head

# Time step parameters
nper = 1
perlen = [3650]
nstp = 50
steady = [False]

# Well locations
wpt1 = (0, 20-1, 37-1)
wpt2 = (0, 37-1, 18-1)

# Output control
spd = {(0, 0): ['print head', 'save head', 'print budget', 'save budget', 'print drawdown', 'save drawdown']}


# Get variable inputs
samples = genParamSamples(hk_min=hk_min, hk_max=hk_max,vka_min=vka_min, vka_max=vka_max, numWells=numWells,
                          pump_min=pump_min, pump_max=-pump_max, sampleSize=sampleSize)


# Define output parameters for each run
model_name = []
for i in range(sampleSize):
    model_name.append('model' + str(i))
modflow_output = []
modflow_succes = []
head_object = []
head_data1 = np.zeros([sampleSize, nstp])
head_data2 = np.zeros([sampleSize, nstp])
    
# Run MODFLOW for each of the paramter samples
for i in range(sampleSize):

    # Get pumping rate same and make stress period dictionary
    pumping_rate1 = samples['pump_rate_1'][i]
    pumping_rate2 = samples['pump_rate_2'][i]
    wel_sp1 = [0, 19, 36, pumping_rate1] # Remember to use zero-based layer, row, column indices!
    wel_sp2 = [0, 36, 17, pumping_rate2]
    stress_period_data = {0: [wel_sp1, wel_sp2]}

    # Get hydraulic conductivty sample
    hk = samples['hk'][i]
    vka = samples['vka'][i]

    # Flopy objects
    mf = flopy.modflow.Modflow(model_name[i], exe_name='./mf2005dbl')
    dis = flopy.modflow.ModflowDis(mf, nlay, nrow, ncol, delr=delr, delc=delc,
                                   top=ztop, botm=botm[1:],
                                   nper=nper, perlen=perlen, nstp=nstp, steady=steady)
    bas = flopy.modflow.ModflowBas(mf, ibound=ibound, strt=strt)
    lpf = flopy.modflow.ModflowLpf(mf, hk=hk, vka=vka, sy=sy, ss=ss, laytyp=laytyp, hdry=hdry)
    pcg = flopy.modflow.ModflowPcg(mf, mxiter=mxiter, hclose=hclose)  # This has stuff like iterations for solver
    wel = flopy.modflow.ModflowWel(mf, stress_period_data=stress_period_data)
    oc = flopy.modflow.ModflowOc(mf, stress_period_data=spd)

    # Write the model input files
    mf.write_input()

    # Run the model
    success, mfoutput = mf.run_model(silent=True, pause=False, report=True)
    modflow_output.append(mfoutput)
    modflow_succes.append(success)
    # if not success:
    #     raise Exception('MODFLOW did not terminate normally.')


    # Create the headfile object
    headobj = bf.HeadFile(model_name[i]+'.hds')
    head_object.append(headobj)
    time = headobj.get_times()

    # Plot hydrograph: head versus time
    if plotHydrograph:
        ts1 = headobj.get_ts(wpt1)
        ts2 = headobj.get_ts(wpt2)
        fig1 = plt.figure()
        ax1 = fig1.add_subplot(111)
        ttl = 'Head at cell ({0},{1},{2})'.format(wpt1[0] + 1, wpt1[1] + 1, wpt1[2] + 1)
        ax1.set_title(ttl)
        ax1.set_xlabel('time')
        ax1.set_ylabel('head')
        ax1.plot(ts1[:, 0], ts1[:, 1])
        ax1.plot(ts2[:, 0], ts2[:, 1])
        fig1.savefig('hydrograph' + str(i) + '.pdf')

    # Plot contour map
    if plotContours:

        # Setup contour parameters
        levels = np.arange(1, 1010, 20)
        extent = (delr/2., Lx - delr/2., delc/2., Ly - delc/2.)

        # Make the plots
        mytimes = [time[-1]]
        for iplot, t in enumerate(mytimes):
            head = headobj.get_data(totim=t)

        #Create the plot
            fig2 = plt.figure()
            ax2 = fig2.add_subplot(1, 1, 1, aspect='equal')
            ax2.set_title('stress period ' + str(iplot + 1))
            ax2.imshow(head[0, :, :], extent=extent, cmap='BrBG', vmin=0., vmax=10.)
            CS = plt.contour(np.flipud(head[0, :, :]), levels=levels, extent=extent,
                                zorder=10)
            plt.clabel(CS, inline=1, fontsize=10, fmt='%1.1f', zorder=11)
            plt.colorbar()
            mfc = 'None'
            if (iplot+1) == len(mytimes):
                mfc='black'
            plt.savefig('contour' + str(i) + '.pdf')

    file = open('modflow_output' + str(i) + '.txt', 'w')

    # Create MODFLOW output file if there was an error
    for n in modflow_output[i]:
        if not success:
            file.write(n + '\n')

    # Write hydrograph data to array
    head_data1[i, :] = ts1[:, 1]
    head_data2[i, :] = ts2[:, 1]


# Save output in .mat file
outputDic = dict(zip(['head_data1', 'head_data2', 'time'], [head_data1, head_data2, time]))
outputDic.update(samples)
io.savemat('modflowData.mat', outputDic)



