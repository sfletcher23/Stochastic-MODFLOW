from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt

map = Basemap(llcrnrlat=15, llcrnrlon=40, urcrnrlat=35, urcrnrlon=55)

map.drawcoastlines()
well_info = map.readshapefile('wells', 'wells')
print(well_info)

for info, well in zip(map.wells_info, map.wells):
    map.plot(well[0], well[1], marker='+', color='m', markersize=8, markeredgewidth=2)



plt.show()