
import numpy as np
import scipy as sp
import matplotlib as ml
import flopy
import flopy.utils.binaryfile as bf
import matplotlib.pyplot as plt


def hydrograph(headobj, timeSeries, hk, vka, sy, well_loc, numWells, pump_rate):

    # Get hydrograph data
    headData = {}
    for n in range(numWells):
        headData[n] = headobj.get_ts(tuple(well_loc[n]))


    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    ttl = 'Hydrograph: K = {:.2e} , vK = {:.2e}, sy = {:.2e}'.format(hk, vka, sy)
    ax1.set_title(ttl)
    ax1.set_xlabel('time [years]')
    ax1.set_ylabel('head')
    ax1.set_ylim(-1, 1000)
    for n in range(numWells):
        ax1.plot(headData[n][:, 0] / 365, headData[n][:, 1],
                 label='Well {}: Pump rate = {} m^3/d'.format(n, pump_rate[n]))
    plt.legend()

    # if pumpingCosts:
    #     ax2 = ax1.twinx()
    #     for n in range(numWells):
    #         ax2.plot(headData[n][:, 0] / 365, pumpCosts[n], '--', label='Well {} Pumping costs'.format(n))
    #     ax2.set_ylabel('Pumping costs [$/m^3]')

    plt.legend()
    plt.show()
    fig1.savefig('hydrograph' + '.pdf')

