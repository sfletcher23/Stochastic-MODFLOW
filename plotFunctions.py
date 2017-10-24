
import numpy as np
import scipy as sp
import matplotlib as ml
import flopy
import flopy.utils.binaryfile as bf
import matplotlib.pyplot as plt
import csv
import pickle


def grid_withBCs(mf, dis, sr, well, rch):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1, aspect='equal')
    modelmap = flopy.plot.ModelMap(model=mf, sr=sr, dis=dis)
    quadmesh = modelmap.plot_ibound()
    quadmesh = modelmap.plot_array(rch.rech.array[0,0,:,:], masked_values=[0] )
    # quadmesh = modelmap.plot_bc('WEL')
    linecollection = modelmap.plot_grid()
    plt.show()



def hydrograph(headData, timeSeries, hk, vka, ss, numWells, pump_rate, saveName, startingHead, multiplepanels=False, saveFig=True):

    with open('inputWellData_USGS.csv', 'rt') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='|')

        next(reader, None)  # skip header

        well_number = []
        latitude = []
        longitude = []
        for row in reader:
            well_number.append(row[0])
            latitude.append(row[1])
            longitude.append(row[2])

    values = np.arange(0, 1, 1 / numWells)
    cmap = ml.cm.get_cmap('spectral')


    if multiplepanels:

        # Define different well sets to highlight
        wellsIndex = [None] * 4
        wellsNames = [None] * 4
        wellsIndex[0] = np.arange(13, 34)
        wellsNames[0] = 'Salbukh Well Field'
        wellsIndex[1] = np.arange(40, 58)
        wellsNames[1] = 'Buwayb Well Field'
        wellsIndex[2] = np.asarray([80, 107])
        wellsNames[2] = 'Al hair and Shemesy'
        wellsIndex[3] = np.arange(92, 99)
        wellsNames[3] = 'Central Riyadh'

        fig1 = plt.figure()
        for i in range(4):
            ax1 = fig1.add_subplot(2, 2, i+1)
            ax1.set_xlabel('time [years]')
            ax1.set_ylabel('head')
            ax1.set_ylim(-1, startingHead)
            ax1.set_yticks(np.arange(0, startingHead, 100))
            ax1.set_title(wellsNames[i])
            for n in range(numWells):
                if int(well_number[n]) in wellsIndex[i]:
                    ax1.plot(timeSeries / 365, headData[:,n],
                             label= well_number[n] + ' : Pump rate = {0:.0f} m^3/d'.format(pump_rate[n]),
                             color=cmap(values[n]), zorder=3)
                else:
                    ax1.plot(timeSeries / 365, headData[:, n], color='black', zorder=1)
            # plt.legend(loc='right', bbox_to_anchor=(1.5, .5))
        plt.suptitle('Hydrographs: K = {:.2} , vK = {:.2}, ss = {:.2}'.format(hk, vka, ss))
        plt.tight_layout()
        plt.subplots_adjust(top=0.85)
        plt.show()
        path = 'hydrograph' + saveName + '.pkl'
        pickle.dump(ax1, open(path, 'wb'))
        fig1.savefig('hydrograph' + saveName + '.pdf')


    else:
        fig1 = plt.figure()
        ax1 = fig1.add_subplot(121)
        ttl = 'Hydrograph: K = {:.2e} , vK = {:.2e}, ss = {:.2e}'.format(hk, vka, ss)
        ax1.set_title(ttl)
        ax1.set_xlabel('time [years]')
        ax1.set_ylabel('head')
        ax1.set_ylim(-1, startingHead)

        for n in range(numWells):
            ax1.plot(timeSeries / 365, headData[:,n],
                     label= well_number[n] + ' : Pump rate = {0:.0f} m^3/d'.format(pump_rate[n]), color=cmap(values[n]))
        plt.legend(loc='right', bbox_to_anchor=(2.5, .5))
        plt.show()
        path = 'hydrograph' + saveName + '.pkl'
        pickle.dump(fig1, open(path, 'wb'))
        fig1.savefig('hydrograph' + saveName + '.pdf')





def contour(headobj, timeSeries, mf, sr, wel, dis, plot_wells_riyadh, saveName, saveFig=True):
    # Plot contour map

    # Setup contour parameters
    levels = np.arange(0, 1000, 5)

    # Get head data
    head = headobj.get_data()

    # # 1st plot: array plot
    # fig1 = plt.figure(figsize=(10, 10))
    # modelmap = flopy.plot.map.ModelMap(sr=sr, model=mf, dis=dis, layer=0, rotation=sr.rotation, length_multiplier=1.0,
    #                                    xul=sr.xul, yul=sr.yul)
    # ax = plt.gca()
    # ax.set_title('plot_array()')
    # quadmesh = modelmap.plot_ibound()
    # quadmesh = modelmap.plot_array(head, masked_values=[999.], alpha=0.5)
    # # modelmap.plot_bc("WEL")
    # linecollection = modelmap.plot_grid()
    # plt.colorbar(quadmesh)
    # if plot_wells_riyadh:
    #     # Plot a shapefile of well locations
    #     shp = 'wells'
    #     patch_collection_wells = modelmap.plot_shapefile(shp, radius=1000, facecolor='red', edgecolor='red')
    #     patch_collection_wells.zorder = 3
    #
    #     # Plot Royadh location
    #     shp = 'Riyadh'
    #     patch_collection_riyadh = modelmap.plot_shapefile(shp, radius=3000, facecolor='green', alpha=0.5)
    #     patch_collection_riyadh.zorder = 2
    # plt.show()


    # 2nd subplot: contour plot
    fig2 = plt.figure(figsize=(10, 10))
    modelmap = flopy.plot.map.ModelMap(sr=sr, model=mf, dis=dis, layer=0, rotation=sr.rotation, length_multiplier=1.0,
                                       xul=sr.xul, yul=sr.yul)
    ax = plt.gca()
    ax.set_title('Water table after 30 years [meters above bottom of aquifer]')
    quadmesh = modelmap.plot_ibound()
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

    # # Plot inset zoom
    # from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
    # from mpl_toolkits.axes_grid1.inset_locator import mark_inset
    # axins = zoomed_inset_axes(ax, 2.5, loc='upper right')
    # # axins.imshow(contour_set, interpolation="nearest",
    # #              origin="lower")
    # x1, x2, y1, y2 = 650000, 750000, 265000, 280000
    # axins.set_xlim(x1, x2)
    # axins.set_ylim(y1, y2)

    # Show and save figure
    # path = 'contour' + saveName + '.pkl'
    # pickle.dump(fig2, open(path, 'wb'))
    plt.show()
    # fig1.savefig('contour_array' + saveName + '.pdf')
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




