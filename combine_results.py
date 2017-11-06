import os
import numpy as np
import datetime
import scipy.io as io

# Save options
saveNumpy = True
saveMat = True

# Use a counter to test on only three files
counter = 0
countMax = 5000

# Preallocate arrays
DIR = 'simulation_data'
runs = len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))])


ss = np.empty([runs])
hk = np.empty([runs])
vka = np.empty([runs])


for file in os.listdir("simulation_data"):
    if file.startswith("output_2017-11") & (counter < countMax):
        if counter == 0:
            data = np.load("simulation_data/" + file)
            numWells = data['numWells']
            headData = np.empty((numWells, 52 * 30, runs), float)
        data = np.load("simulation_data/" + file)
        headData[:, :, counter] = np.reshape(data['headData'],(numWells,52*30))
        ss[counter] = data['ss']
        hk[counter] = data['hk']
        vka[counter] = data['vka']
        counter += 1

datetimeStr =str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

if saveNumpy:
    outputName = 'combined_output' + datetimeStr
    np.savez(outputName, headData=headData, ss=ss, hk=hk, vka=vka)

if saveMat:
    outputDic = dict(zip(['headData', 'hk', 'ss'], [headData, hk, ss]))
    io.savemat('modflowData' + datetimeStr, outputDic)