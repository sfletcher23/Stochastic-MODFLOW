
#Import packages
import numpy as np
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

# Plot settings
plotContours = False
plotHydrograph = True

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
hk = 6.     # horizontal hydraulic conductivity
vka = 5.26e-3    # vertical hydraulic conductivity
sy = 2.5e-1
ss = 4.e-7
laytyp = 1  # 1 = unconfined, 0 = confined
hdry = 0    # dry cell head set to this number

# Variables for the BAS package
ibound = np.ones((nlay, nrow, ncol), dtype=np.int32)    # Not sure if GW uses 0 or 1
strt = 1000 * np.ones((nlay, nrow, ncol), dtype=np.float32)     # Starting head

# Time step parameters
nper = 1
perlen = [3650]
nstp = [40]
steady = [False]

# Well locations
wpt1 = (0, 20-1, 37-1)
wpt2 = (0, 37-1, 18-1)

# Flopy objects
modelname = 'TestFlopy'
mf = flopy.modflow.Modflow(modelname, exe_name='./mf2005dbl')
dis = flopy.modflow.ModflowDis(mf, nlay, nrow, ncol, delr=delr, delc=delc,
                               top=ztop, botm=botm[1:],
                               nper=nper, perlen=perlen, nstp=nstp, steady=steady)
bas = flopy.modflow.ModflowBas(mf, ibound=ibound, strt=strt)
lpf = flopy.modflow.ModflowLpf(mf, hk=hk, vka=vka, sy=sy, ss=ss, laytyp=laytyp, hdry=hdry)
pcg = flopy.modflow.ModflowPcg(mf)  # This has stuff like iterations for solver

# Create the well package
# Remember to use zero-based layer, row, column indices!
pumping_rate1 = -71691.
pumping_rate2 = -341114.
wel_sp1 = [0, 19, 36, pumping_rate1]
wel_sp2 = [0, 36, 17, pumping_rate2]
stress_period_data = {0: [wel_sp1, wel_sp2]}
wel = flopy.modflow.ModflowWel(mf, stress_period_data=stress_period_data)

# Output control
spd = {(0, 0): ['print head', 'save head', 'print budget', 'save budget']}
oc = flopy.modflow.ModflowOc(mf, stress_period_data=spd)

# Write the model input files
mf.write_input()

# Run the model
success, mfoutput = mf.run_model(silent=True, pause=False, report=True)
if not success:
    raise Exception('MODFLOW did not terminate normally.')


# Create the headfile object
headobj = bf.HeadFile(modelname+'.hds')
time = headobj.get_times()

# Plot hydrograph: head versus time
if plotHydrograph:
    ts1 = headobj.get_ts(wpt1)
    ts2 = headobj.get_ts(wpt2)
    plt.subplot(1, 1, 1)
    ttl = 'Head at cell ({0},{1},{2})'.format(wpt1[0] + 1, wpt1[1] + 1, wpt1[2] + 1)
    plt.title(ttl)
    plt.xlabel('time')
    plt.ylabel('head')
    plt.plot(ts1[:, 0], ts1[:, 1])
    plt.plot(ts2[:, 0], ts2[:, 1])
    plt.show()

# Plot contour map
if plotContours:

    # Setup contour parameters
    levels = np.arange(1, 1010, 20)
    extent = (delr/2., Lx - delr/2., delc/2., Ly - delc/2.)
    print('Levels: ', levels)
    print('Extent: ', extent)
    print('Times: ', time)

    # Make the plots
    mytimes = [time[1], time[30]]
    for iplot, t in enumerate(mytimes):
        print('*****Processing time: ', t)
        head = headobj.get_data(totim=t)
        #Print statistics
        print('Head statistics')
        print('  min: ', head.min())
        print('  max: ', head.max())
        print('  std: ', head.std())

    #Create the plot
        #plt.subplot(1, len(mytimes), iplot + 1, aspect='equal')
        plt.subplot(1, 1, 1, aspect='equal')
        plt.title('stress period ' + str(iplot + 1))
        plt.imshow(head[0, :, :], extent=extent, cmap='BrBG', vmin=0., vmax=10.)
        plt.colorbar()
        CS = plt.contour(np.flipud(head[0, :, :]), levels=levels, extent=extent,
                            zorder=10)
        plt.clabel(CS, inline=1, fontsize=10, fmt='%1.1f', zorder=11)
        mfc = 'None'
        if (iplot+1) == len(mytimes):
            mfc='black'
        plt.show()
    plt.show()


