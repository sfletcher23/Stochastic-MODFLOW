import os
import numpy as np
import datetime

# Use a counter to test on only three files
counter = 0
countMax = 3


for file in os.listdir("simulation_data"):
    if file.startswith("output_2017-08") & (counter < countMax):
        data = np.load("simulation_data/" + file)
        tempHeadData = np.transpose(data['headData'],[1, 0, 2])
        if counter == 0:
            headData = tempHeadData
            # Variable inputs
            sy = data['sy']
            hk = data['hk']
            vka = data['vka']
        else:
            headData = np.append(headData, tempHeadData, 2)
            sy = np.append(sy, data['sy'])
            hk = np.append(hk, data['hk'])
            vka = np.append(vka, data['vka'])
        counter += 1

datetimeStr =str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
outputName = 'combined_output' + datetimeStr
np.savez(outputName, headData=headData, sy=sy, hk=hk, vka=vka)
