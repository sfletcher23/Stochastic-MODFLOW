
import numpy as np
import scipy as sp
import matplotlib as ml
import flopy
import flopy.utils.binaryfile as bf
import matplotlib.pyplot as plt
import csv


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
    ax1 = fig1.add_subplot(121)
    ttl = 'Hydrograph: K = {:.2e} , vK = {:.2e}, sy = {:.2e}'.format(hk, vka, sy)
    ax1.set_title(ttl)
    ax1.set_xlabel('time [years]')
    ax1.set_ylabel('head')
    ax1.set_ylim(-1, startingHead)

    with open('inputWellData.csv', 'rt') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='|')

        next(reader, None)  # skip header

        name = []
        latitude = []
        longitude = []
        for row in reader:
            name.append(row[0])
            latitude.append(row[1])
            longitude.append(row[2])

    for n in range(numWells):
        ax1.plot(timeSeries / 365, headData[:,n],
                 label= name[n] + ' : Pump rate = {0:.0f} m^3/d'.format(pump_rate[n]))
    plt.legend(loc='right', bbox_to_anchor=(2.5, .5))
    plt.show()
    fig1.savefig('hydrograph' + saveName + '.pdf')




def contour(headobj, timeSeries, mf, sr, wel, dis, plot_wells_riyadh, saveName):
    # Plot contour map

    # Setup contour parameters
    levels = np.arange(0, 200, 1)

    # Get head data
    head = headobj.get_data()

    # Initialize figure
    fig = plt.figure(figsize=(10, 10))

    # 1st subplot: array plot
    ax = fig.add_subplot(1, 2, 1, aspect='equal')
    modelmap = flopy.plot.map.ModelMap(sr=sr, model=mf, dis=dis, layer=0, rotation=sr.rotation, length_multiplier=1.0,
                                       xul=sr.xul, yul=sr.yul)
    ax.set_title('plot_array()')
    quadmesh = modelmap.plot_ibound()
    quadmesh = modelmap.plot_array(head, masked_values=[999.], alpha=0.5)
    # modelmap.plot_bc("WEL")
    linecollection = modelmap.plot_grid()
    plt.colorbar(quadmesh)
    if plot_wells_riyadh:
        # Plot a shapefile of well locations
        shp = 'wells'
        patch_collection_wells = modelmap.plot_shapefile(shp, radius=1000, facecolor='red', edgecolor='red')
        patch_collection_wells.zorder = 3

        # Plot Royadh location
        shp = 'Riyadh'
        patch_collection_riyadh = modelmap.plot_shapefile(shp, radius=3000, facecolor='green', alpha=0.5)
        patch_collection_riyadh.zorder = 2

    # 2nd subplot: contour plot
    ax = fig.add_subplot(1, 2, 2, aspect='equal')
    modelmap = flopy.plot.map.ModelMap(sr=sr, model=mf, dis=dis, layer=0, rotation=sr.rotation, length_multiplier=1.0,
                                       xul=sr.xul, yul=sr.yul)
    ax.set_title('contour_array()')
    quadmesh = modelmap.plot_ibound()
    modelmap.plot_bc("WEL", wel)
    contour_set = modelmap.contour_array(head, masked_values=[999.], levels=levels, zorder=3)
    linecollection = modelmap.plot_grid(zorder=1)
    plt.colorbar(contour_set)
    if plot_wells_riyadh:
        # Plot a shapefile of well locations
        shp = 'wells'
        patch_collection_wells = modelmap.plot_shapefile(shp, radius=1000, facecolor='red', edgecolor='red')
        patch_collection_wells.zorder = 3

        # Plot Royadh location
        shp = 'Riyadh'
        patch_collection_riyadh = modelmap.plot_shapefile(shp, radius=3000, facecolor='green', alpha=0.5)
        patch_collection_riyadh.zorder = 2

    # Show and save figure
    plt.show()
    fig.savefig('contour' + saveName + '.pdf')




    # # Create the plot
    # fig2 = plt.figure()
    # ax2 = fig2.add_subplot(1, 1, 1, aspect='equal')
    # ax2.set_title('stress period ' + str(iplot + 1) + 't = ' + str(t))
    # ax2.imshow(head[0, :, :], extent=None, cmap='BrBG', vmin=0., vmax=10.)
    # CS = plt.contour(np.flipud(head[0, :, :]), levels=levels, extent=None,
    #                     zorder=10)
    # plt.clabel(CS, inline=1, fontsize=10, fmt='%1.1f', zorder=11)
    # plt.colorbar()
    # plt.show()
    # fig2.savefig('contour' + saveName + '.pdf')














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




