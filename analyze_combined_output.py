import numpy as np
import matplotlib as ml
import matplotlib.pyplot as plt
import scipy.io as io
import csv

# which analysis?
plotBoxplot = True
plotHydrographDist = True
sabukh = False
buwayb = False
saveToMat = False

# open combined data
timeToOpen = '2017-11-08 12:24:51'

data = np.load('simulation_data/combined_output'+timeToOpen+'.npz')
headData = data['headData']
ss = data['ss']
hk = data['hk']
vka = data['vka']
runs = np.size(headData[0,0,:])

ss = ss[0:1500]
hk = hk[0:1500]
vka = vka[0:1500]
headData = headData[:,:,0:1500]

fig = plt.figure()
ax1 = fig.add_subplot(1, 2, 1)
plt.hist(ss, bins=np.linspace(0.1e-6, 10.e-5,100))
ax1.set_title('SS')
ax2 = fig.add_subplot(1, 2, 2)
plt.hist(hk, bins=np.logspace(-7, 1, 100))
plt.title('HK histogram')
plt.show()



with open('inputWellData.csv', 'rt') as csvfile:
    reader = csv.reader(csvfile, delimiter=',', quotechar='|')
    next(reader, None)  # skip header
    pump_rate = []
    well_num = []
    well_name = []
    for row in reader:
        pump_rate.append(float(row[4]))
        well_num.append(int(row[0]))
        well_name.append(row[3])


# list of key wells
# Define different well sets to highlight
plotWells = [26, 27, 55, 53, 118, 29, 30, 31, 32, 44]
plotWells = np.argpartition(headData[:,-1,2],6)[0:6]
plotWells = np.append(plotWells, [27, 29, 53, 55])
map(int, plotWells)
wellsIndex = []
wellsNames = []
for i in range(len(well_num)):
    if well_num[i] in plotWells:
        wellsIndex.append(i)
        wellsNames.append(well_name[i])



# Make boxplot of drawdown after 30 years for key wells
if plotBoxplot:

    x = np.transpose(headData[wellsIndex,-1,:])
    fig1 = plt.figure(figsize=(8,5))
    plt.boxplot(x, labels=wellsNames, whis= [5, 95])
    plt.title('Boxplot of head after 30 years by well')
    plt.ylabel('head [m]')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.ylim([-600, 650])
    plt.axhline(y=-588, color='grey', linestyle='dotted')
    plt.axhline(y=612, color='grey', linestyle='dotted')
    plt.show()
    fig1.savefig('boxplot_drawdown_'+ timeToOpen)


# Some plot showing distribution of time series for a single well
if plotHydrographDist:
    fig2 = plt.figure(figsize=(10,8))
    count=0
    time = np.arange(0,52*30) / 52
    for i in wellsIndex:
        count += 1
        p1 = np.zeros(52*30)
        p25 = np.zeros(52 * 30)
        p50 = np.zeros(52 * 30)
        p75 = np.zeros(52 * 30)
        p99 = np.zeros(52 * 30)
        for t in range(52*30):
            x = headData[i,t,:]
            p1[t] = np.percentile(x, 1)
            p25[t] = np.percentile(x, 25)
            p50[t] = np.percentile(x, 50)
            p75[t] = np.percentile(x, 75)
            p99[t] = np.percentile(x, 99)
        ax2 = fig2.add_subplot(3, 4, count)
        ax2.fill_between(time, p1, p99, facecolor='PaleTurquoise', alpha=1, label='99% CI')
        ax2.fill_between(time, p25, p75, facecolor='LightSeaGreen', alpha=1, label='50% CI')
        ax2.plot(time, p50, color='black', label='median')
        ax2.set_title(wellsNames[count-1])
        ax2.set_ylim([-600, 650])
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
