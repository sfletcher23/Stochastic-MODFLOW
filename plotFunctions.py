
import numpy as np
import scipy as sp
import matplotlib as ml
import flopy
import flopy.utils.binaryfile as bf
import matplotlib.pyplot as plt


def grid_withBCs(mf, dis, sr, well):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1, aspect='equal')
    modelmap = flopy.plot.ModelMap(model=mf, sr=sr, dis=dis)
    quadmesh = modelmap.plot_ibound()
    quadmesh = modelmap.plot_bc('WEL')
    linecollection = modelmap.plot_grid()
    plt.show()





def hydrograph(headData, timeSeries, hk, vka, sy, numWells, pump_rate, saveName, startingHead):

    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    ttl = 'Hydrograph: K = {:.2e} , vK = {:.2e}, sy = {:.2e}'.format(hk, vka, sy)
    ax1.set_title(ttl)
    ax1.set_xlabel('time [years]')
    ax1.set_ylabel('head')
    ax1.set_ylim(-1, startingHead)
    for n in range(numWells):
        ax1.plot(timeSeries / 365, headData[:,n],
                 label='Well {}: Pump rate = {} m^3/d'.format(n, pump_rate[n]))
    plt.legend()
    plt.show()
    fig1.savefig('hydrograph' + saveName + '.pdf')


# Plot contour map
def contour(headobj, timeSeries, saveName):

    # Setup contour parameters
    levels = np.arange(1, 1010, 20)
    extent = (delr/2., Lx - delr/2., delc/2., Ly - delc/2.)

    # Make the plots
    mytimes = timeSeries[-1]
    if not isinstance(timeSeries, list):
        head = headobj.get_data(totim=mytimes)
        iplot = 0
        t = mytimes
    else:
        for iplot, t in enumerate(mytimes):
            head = headobj.get_data(totim=t)

    # Create the plot
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
    fig2.savefig('contour' + saveName + '.pdf')




    # if pumpingCosts:
    #     ax2 = ax1.twinx()
    #     for n in range(numWells):
    #         ax2.plot(headData[n][:, 0] / 365, pumpCosts[n], '--', label='Well {} Pumping costs'.format(n))
    #     ax2.set_ylabel('Pumping costs [$/m^3]')

    # # Calculate pumping costs

    #
    # # Pumping cost paramters
    # electricityCost = 0.12  # $/kWh
    # pumpEfficiency = 0.80
    # if pumpingCosts:
    #     kWhperJ = 2.77778e-7  # Convert from Joules to kWh
    #     mgh = {}
    #     pumpCosts = {}
    #     for n in range(numWells):
    #         mgh = 1000 * 9.8 * (ztop - headData[n][:, 1])  # 1000 kg per m^3 water * g * drawdown --> Joules per m^3
    #         pumpCosts[n] = mgh * kWhperJ * 1/pumpEfficiency * electricityCost  #* -pump_rate_1
    #         pumpOnTimes = np.zeros_like(headData[n][:, 1])
    #         for index, item in enumerate(pumpingOn):
    #             if item == 1:
    #                 pumpOnTimes[index * nstp: index * nstp + nstp] = 1
    #         pumpCosts[n] *= pumpOnTimes




