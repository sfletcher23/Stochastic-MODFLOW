import numpy as np
import matplotlib as ml
import matplotlib.pyplot as plt
import scipy.io as io

# which analysis?
plotBoxplot = False
plotHydrographDist = False
sabukh = False
buwayb = False
saveToMat = True

# open combined data
timeToOpen = '2017-08-15 14:10:25'

data = np.load('simulation_data/combined_output'+timeToOpen+'.npz')
headData = data['headData']
ss = data['ss']
hk = data['hk']
vka = data['vka']
runs = np.size(headData[0,0,:])

# list of key wells
# Define different well sets to highlight
wellsIndex = [None] * 9
wellsNames = [None] * 9
wellsIndex[0] = 16
wellsNames[0] = 'Salbukh 7 (Most)'
wellsIndex[1] = 27
wellsNames[1] = 'Salbukh 4 (Least)'
wellsIndex[2] = 51
wellsNames[2] = 'Buwayb 15 (Least)'
wellsIndex[3] = 55
wellsNames[3] = 'Buwayb 13 (Most)'
wellsIndex[4] = 107
wellsNames[4] = 'Shemesy'
wellsIndex[5] = 68
wellsNames[5] = 'RR2 (Central Riyadh)'
wellsIndex[6] = 92
wellsNames[6] = 'Royal Garage (Central Riyadh)'
wellsIndex[7] = 93
wellsNames[7] = 'Nasiriyah 1 (Central Riyadh)'
wellsIndex[8] = 65
wellsNames[8] = 'Malez 2'


# Make boxplot of drawdown after 30 years for key wells
if plotBoxplot:
    wellsNames[0] = 'Salbukh 7 \n(Most)'
    wellsNames[1] = 'Salbukh 4 \n(Least)'
    wellsNames[2] = 'Buwayb 15 \n(Least)'
    wellsNames[3] = 'Buwayb 13 \n(Most)'
    wellsNames[4] = 'Shemesy'
    wellsNames[5] = 'RR2 \n(Central Riyadh)'
    wellsNames[6] = 'Royal Garage \n(Central Riyadh)'
    wellsNames[7] = 'Nasiriyah 1 \n(Central Riyadh)'
    wellsNames[8] = 'Malez 2'

    x = np.transpose(headData[wellsIndex,-1,:])
    fig1 = plt.figure(figsize=(8,5))
    plt.boxplot(x, labels=wellsNames)
    plt.title('Boxplot of drawdown after 30 years by well')
    plt.ylabel('head [m]')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()
    fig1.savefig('boxplot_drawdown_'+ timeToOpen)


# Some plot showing distribution of time series for a single well
if plotHydrographDist:
    fig2 = plt.figure(figsize=(10,8))
    count=0
    time = np.arange(0,365*30) / 365
    for i in wellsIndex:
        count += 1
        p1 = np.zeros(365*30)
        p25 = np.zeros(365 * 30)
        p50 = np.zeros(365 * 30)
        p75 = np.zeros(365 * 30)
        p99 = np.zeros(365 * 30)
        for t in range(365*30):
            x = headData[i,t,:]
            p1[t] = np.percentile(x, 1)
            p25[t] = np.percentile(x, 25)
            p50[t] = np.percentile(x, 50)
            p75[t] = np.percentile(x, 75)
            p99[t] = np.percentile(x, 99)
        ax2 = fig2.add_subplot(3, 3, count)
        ax2.fill_between(time, p1, p99, facecolor='PaleTurquoise', alpha=1, label='99% CI')
        ax2.fill_between(time, p25, p75, facecolor='LightSeaGreen', alpha=1, label='50% CI')
        ax2.plot(time, p50, color='black', label='median')
        ax2.set_title(wellsNames[count-1])
        ax2.set_ylim([0, 200])
        ax2.set_xlabel('years')
        ax2.set_ylabel('head [m]')
        if count == 1:
            h1 = ml.lines.Line2D([], [], color='PaleTurquoise', alpha=1, linewidth=10, label='99% CI')
            h2 = ml.lines.Line2D([], [], color='LightSeaGreen', alpha=1, linewidth=10, label='50% CI')
            h3 = ml.lines.Line2D([], [], color='black', linewidth=2, label='median')
            plt.legend([h1, h2, h3], ['99% CI', '50% CI', 'median'])
    plt.suptitle('Hydrograph distribution')
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    plt.show()
    fig2.savefig('hydrograph_dist_' + timeToOpen)

# look at sabukh dist
if sabukh:
    x = headData[11:32,-1,1]
    print(x)
    # [ 185.60986328  180.70594788  181.37161255  184.0705719   179.12763977
    #   177.41018677  181.29855347  184.00674438  182.44107056  177.83282471
    #   177.78129578  181.37063599  183.41209412  181.93707275  184.0295105
    #   186.83331299  193.21746826  192.42012024  188.86715698  187.18287659
    #   184.88816833]

# look at buwayb dist
if buwayb:
    x = headData[38:57,-1,1]
    print(x)
    # [187.42407227  184.006073    183.43885803  181.13848877  182.10525513
    #  183.13520813  184.9389801   186.38580322  181.6137085   181.80001831
    #  183.33131409  182.20431519  183.19859314  183.64807129  186.96044922
    #  185.93339539  187.60023499  188.01049805 161.24845886]


# save data to mat
if saveToMat:
    outputDic = dict(zip(['headData', 'hk', 'ss'], [headData, hk, ss]))
    io.savemat('modflowData' + timeToOpen, outputDic)
