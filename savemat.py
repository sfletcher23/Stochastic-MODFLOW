import numpy as np
import matplotlib as ml
import matplotlib.pyplot as plt
import scipy.io as io

# open combined data
timeToOpen = '2017-11-07 12:25:37'

data = np.load('simulation_data/combined_output'+timeToOpen+'.npz')

divisions = 20;
runs = 5000;

for i in range(divisions):
    range_min = int(i*runs/divisions)
    range_max = int((i+1)*runs/divisions)
    headData = data['headData'][:,:,range_min:range_max]
    dicHead = dict(zip(['headData'], [headData]))
    io.savemat('modflowData_headData'+ str(i) + timeToOpen, dicHead)


sy = data['sy']
hk = data['hk']
vka = data['vka']
data.close()

dicHk = dict(zip(['hk'], [hk]))
dicSy = dict(zip(['sy'], [sy]))

io.savemat('modflowData_hk' + timeToOpen, dicHk)
io.savemat('modflowData_sy' + timeToOpen, dicSy)


