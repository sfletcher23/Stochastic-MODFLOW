import os
import numpy as np
import datetime
import scipy.io as io

# Save options
saveNumpy = True
saveMat = True

# Single well only
singleWell = True
wellIndex = 52

# Use a counter to test on only three files
counter = 0
countMax = 500

# Preallocate arrays
DIR = 'simulation_data'
#runs = len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))])
runs = countMax

ss = np.empty([runs])
hk = np.empty([runs])
vka = np.empty([runs])

for file in os.listdir("simulation_data"):
    if file.startswith("output_2017-11-09") & (counter < countMax):
        if counter == 0:
            data = np.load("simulation_data/" + file)
            numWells = data['numWells']
            nstp = data['nstp']
            if singleWell:
                headData = np.empty((1, nstp * 30, runs), float)
            else:
                headData = np.empty((numWells, nstp * 30, runs), float)
            timeSeries = np.empty([runs, nstp * 30])
        data = np.load("simulation_data/" + file)
        if singleWell:
            headData[:, :, counter] = np.squeeze(np.transpose(data['headData'], (1, 0, 2)))[wellIndex,:]
        else:
            headData[:, :, counter] = np.squeeze(np.transpose(data['headData'],(1, 0, 2)))
        ss[counter] = data['ss']
        hk[counter] = data['hk']
        vka[counter] = data['vka']
        timeSeries[counter] = data['time']
        counter += 1

datetimeStr =str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

if saveNumpy:
    outputName = 'simulation_data/combined_output' + datetimeStr
    np.savez(outputName, headData=headData, ss=ss, hk=hk, vka=vka)

if saveMat:

    divisions = 5
    timeToOpen = datetimeStr

    for i in range(divisions):
        range_min = int(i * runs / divisions)
        range_max = int((i + 1) * runs / divisions)
        headDataTemp = headData[:, :, range_min:range_max]
        dicHead = dict(zip(['headData'], [headDataTemp]))
        io.savemat('modflowData_headData' + str(i) + timeToOpen, dicHead)

    data.close()

    dicHk = dict(zip(['hk'], [hk]))
    dicSs = dict(zip(['ss'], [ss]))
    dicTime = dict(zip(['timeSeries'], [timeSeries]))
    dicNstp = dict(zip(['nstp'], [nstp]))

    io.savemat('modflowData_hk' + timeToOpen, dicHk)
    io.savemat('modflowData_ss' + timeToOpen, dicSs)
    io.savemat('modflowData_time' + timeToOpen, dicTime)
    io.savemat('modflowData_nstp' + timeToOpen, dicNstp)