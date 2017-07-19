import shapefile
import csv
import utm
import pyproj

def getWKT_PRJ (epsg_code):
    from urllib.request import urlopen
    # access projection information
    with urlopen('http://spatialreference.org/ref/epsg/{0}/prettywkt/'.format(epsg_code)) as wkt:
    # remove spaces between charachters
        remove_spaces = wkt.read().decode('utf-8').replace('' '','')
        # place all the text on one line
        output = remove_spaces.replace('\n', '')
    return output


# Make wells shapefile

wells_shp = shapefile.Writer(shapefile.POINT)
wells_shp.autoBalance = 1
wells_shp.field("Name", "C")
wells_shp.field("Withdrawals", "C")

counter = 1


with open('inputWellData.csv', 'rt') as csvfile:
    reader = csv.reader(csvfile, delimiter=',', quotechar='|')

    next(reader, None)

    zoneNumFixed = 38
    p1 = pyproj.Proj(proj='utm', zone=zoneNumFixed, ellps='WGS84')

    for row in reader:
        name = row[0]
        latitude = row[1]
        longitude = row[2]
        withdrawals = row[3]

        # Convert from Lat/Long to to UTM - In the future, make sure the zone is the same for all points
        [easting, northing, zoneNum, zoneLet] = utm.from_latlon(float(latitude), float(longitude))
        if not (zoneNum == zoneNumFixed ):
            p2 = pyproj.Proj(proj='utm', zone=zoneNum, ellps='WGS84')
            [easting, northing] = pyproj.transform(p2, p1, easting, northing)

        wells_shp.point(easting, northing)

        wells_shp.record(name, withdrawals)

        print("Feature " + str(counter) + " added to Shapefile.")
        counter = counter + 1

# save the Shapefile
wells_shp.save("wells")

# create a projection file
prj = open("wells", "w")
epsg = getWKT_PRJ("4326")
prj.write(epsg)
prj.close()



# Make shapefile with Riyadh location

Riyadh_shp = shapefile.Writer(shapefile.POINT)
Riyadh_shp.autoBalance = 1
Riyadh_shp.field("Name", "C")

counter = 1

zoneNumFixed = 38
p1 = pyproj.Proj(proj='utm', zone=zoneNumFixed, ellps='WGS84')

name = 'Riyadh'
latitude = 24.716199
longitude = 46.712633

# Convert from Lat/Long to to UTM - In the future, make sure the zone is the same for all points
[easting, northing, zoneNum, zoneLet] = utm.from_latlon(float(latitude), float(longitude))
if not (zoneNum == zoneNumFixed):
    p2 = pyproj.Proj(proj='utm', zone=zoneNum, ellps='WGS84')
    [easting, northing] = pyproj.transform(p2, p1, easting, northing)

# Write to shapefile
Riyadh_shp.point(easting, northing)
Riyadh_shp.record(name)

print("Feature " + str(counter) + " added to Shapefile.")
counter = counter + 1

# save the Shapefile
Riyadh_shp.save("Riyadh")

# create a projection file
prj = open("Riyadh", "w")
epsg = getWKT_PRJ("4326")
prj.write(epsg)
prj.close()

