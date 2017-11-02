
# Import packages
import numpy as np
import scipy as sp
import matplotlib as ml
import flopy
import pickle
from flopy.utils.reference import SpatialReference
from flopy.utils.geometry import Polygon, LineString, Point
from flopy.export.shapefile_utils import recarray2shp, shp2recarray
from flopy.utils.modpathfile import PathlineFile, EndpointFile
from flopy.utils.reference import epsgRef
import plotFunctions
import csv
import time



def build_dis_bas_files(mf, startingHead, perlen, nper, nstp, steady):

    # Make row spacings, units in meters
    delr = np.zeros(108)
    delr[0:7] = 15.E3  # 10 km
    delr[7:9] = 10.E3
    delr[9:10] = 8.E3
    delr[10:11] = 6.E3
    delr[11:12] = 5.E3
    delr[12:16] = 3.E3
    delr[16:19] = 2.E3
    delr[19:59] = 1.E3
    delr[59:94] = 1.5E3
    delr[95] = 3.E3
    delr[96] = 5.E3
    delr[97] = 9.E3
    delr[98] = 13.E3
    delr[99:102] = 16.E3
    delr[102:-1] = 18.E3
    delr = np.flip(delr,0)
    xlength = sum(delr)


    # # Make row spacings, units in meters: expand the high resolution area - didn't make a difference
    # delr = np.zeros(108+18)
    # delr[0:7] = 15.E3  # 10 km
    # delr[7:9] = 10.E3
    # delr[9:10] = 8.E3
    # delr[10:11] = 6.E3
    # delr[11:12] = 5.E3
    # delr[12:16] = 3.E3
    # delr[16:19] = 2.E3
    # delr[19:59] = 1.E3
    # delr[59:94] = 1.E3
    # delr[94:114] = 1.E3
    # delr[114] = 2.E3
    # delr[115] = 4.E3
    # delr[116] = 8.E3
    # delr[117] = 13.E3
    # delr[118:121] = 16.E3
    # delr[121:-1] = 18.E3
    # delr = np.flip(delr,0)
    # xlength = sum(delr)



    # Make column spacings, units in meters
    delc = np.zeros(76)
    delc[0:7] = 15.E3
    delc[7:9] = 11.E3
    delc[10] = 8.E3
    delc[11:12] = 6.E3
    delc[12:15] = 4.E3
    delc[15:17] = 3.E3
    delc[17:18] = 2.E3
    delc[18:65] = 1.E3
    delc[65:67] = 2.E3
    delc[67] = 3.E3
    delc[68] = 5.E3
    delc[69] = 8.E3
    delc[70] = 10.E3
    delc[70:-1] = 12.E3
    ylength = sum(delc)

    # Other grid paratmers
    ztop = -466.
    zbot = -588.
    nlay = 1
    nrow = np.size(delr)
    ncol = np.size(delc)
    delv = (ztop - zbot) / nlay
    botm = np.linspace(ztop, zbot, nlay + 1)

    # # Constant grid spacing
    # nrow += 0
    # ncol += 25
    # delc = 5.E3
    # delr = 5.E3

    # # Double grid spacing
    # nrow *= 2
    # ncol *= 2
    # delc *= 0.5
    # delr *= 0.5


    dis = flopy.modflow.ModflowDis(mf, nlay, nrow, ncol, delr=delc, delc=delr,
                                   top=ztop, botm=botm[1:],
                                   nper=nper, perlen=perlen, nstp=[nstp]*nper, steady=steady)

    # BAS file inputs

    # No flow boundaries for outcrop
    ibound = np.ones((nlay, nrow, ncol), dtype=np.int32)  # 1: head varies with time, -1 head constant with time, 0 no flow cell
    ibound[:, :, 0] = 0
    ibound[:, 6:-3, 1] = 0
    ibound[:, 10:-4, 2] = 0
    ibound[:, 30:-5, 3] = 0
    ibound[:, 65:-7, 4] = 0


    # Starting head
    strt = startingHead * np.ones((nlay, nrow, ncol), dtype=np.float32)  # Starting head
    cone = np.ones((nlay, nrow, ncol), dtype=np.float32)
    gradient = np.ones((nlay, nrow, ncol), dtype=np.float32)
    center_row = int(nrow / 2)
    center_col = int(ncol / 2)
    max_start_head = 250
    min_start_head = 200
    max_dist_to_center = np.sqrt(sum(delr[0:center_row]) **2 + sum(delc[0:center_col]) **2)
    max_dist_to_outcrop = sum(delc)
    for i in range(nrow):
        for j in range(ncol):
            if i < center_row:
                row_dist_to_center = sum(delr[i:center_row])
            else:
                row_dist_to_center = sum(delr[center_row:i])
            if j < center_col:
                col_dist_to_center = sum(delc[j:center_col])
            else:
                col_dist_to_center = sum(delc[center_col:j])
            dist_to_center = np.sqrt(row_dist_to_center ** 2 + col_dist_to_center ** 2)
            cone[0,i,j] = max_start_head - (max_start_head - min_start_head) * (dist_to_center/max_dist_to_center) ** 1.4
            row_dist_to_outcrop = sum(delc[j:-1])
            gradient[0,i,j] = 585 #+ 220 * row_dist_to_outcrop / max_dist_to_outcrop
    strt = gradient - cone

    bas = flopy.modflow.ModflowBas(mf, ibound=ibound, strt=strt)

    return dis, bas, nper, nstp, perlen, strt



def build_lpf_file(mf, samples = None):
# samples is a dictionary with a single sample for each i  nput paramter

    laytyp = 1  # 1 = unconfined, 0 = confined
    hdry = 0  # dry cell head set to this number

    # Default paramter inputs if none given
    hk = 1.e-3
    vka = hk / 10
    ss = 1.e-5 / 315

    for key, value in samples.items():
        if key == 'hk':
            hk = value
        elif key == 'vka':
            vka = value
        elif key == 'sy':
            sy = value
        elif key == 'ss':
            ss = value

    lpf = flopy.modflow.ModflowLpf(mf, hk=hk, vka=vka, ss=ss, laytyp=laytyp, hdry=hdry)

    return lpf, hk, vka, ss



def build_pcg_file(mf):

    mxiter = 300
    hclose = 1e-1

    pcg = flopy.modflow.ModflowPcg(mf, mxiter=mxiter, hclose=hclose)

    return pcg



def build_wel_file(mf, sr):
    # Time step parameters
    perlen = 365
    nper = 30  # number of stress periods
    nstp = 52  # Number of time steps per stress period
    steady = [False] * nper
    pumpingOn = [1] * nper

    # Read well location from shapefile
    shp = 'wells'
    well_loc = flopy.plot.plotutil.shapefile_get_vertices(shp)
    wellx = [x[0][0] for x in well_loc]
    welly = [x[0][1] for x in well_loc]

    [wellxrot, wellyrot] = sr.rotate(np.array(wellx) - sr.xll, np.array(welly) - sr.yll, theta=sr.rotation *-1, xorigin=0, yorigin=0)
    [well_loc_r, well_loc_c] = get_rc(sr, wellxrot, wellyrot)

    numWells = np.size(wellx)
    well_loc_array = np.transpose(np.vstack((np.zeros(numWells), well_loc_r, well_loc_c)))
    well_loc = well_loc_array.astype(int).tolist()

    # Get well names
    with open('inputWellData_USGS.csv', 'rt') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='|')
        next(reader, None)  # skip header
        pump_rate_low = []
        pump_rate_high = []
        for row in reader:
            pump_rate_low.append(row[4])
            pump_rate_high.append(row[5])

    pump_rate = pump_rate_low
    pump_rate = np.asarray([float(i) for i in pump_rate]) * -1.E6 / 365 / 2 # This makes total of ~287,000 cm/d, which is close to Adnan's estimate

    # check that number of pumpnig rates = number of wells
    if not len(pump_rate) == numWells:
        raise NameError('Number of pumping rates must equal number of wells')
    # check that number of locations = number of wells
    if not len(well_loc) == numWells:
        raise NameError('Number of well locations must equal number of wells')

    # Stress period dictionary and well, oc objects
    well_sp_data = {}
    for n in range(numWells):
        well_sp_data[n] = list(well_loc[n][:])
        well_sp_data[n].append(pump_rate[n])  # Remember to use zero-based layer, row, column indices

    stress_period_data = {}
    for t in range(nper):
        stress_period_data[t] = []
        for n in range(numWells):
            temp_sp_data = well_sp_data[n][:]
            temp_sp_data[-1] *= pumpingOn[t]
            stress_period_data[t].append(temp_sp_data)

    wel = flopy.modflow.ModflowWel(mf, stress_period_data=stress_period_data)

    return wel, numWells, well_loc, pump_rate


def build_rch_file(mf, dis):
    recharge = np.zeros( (mf.nrow, mf.ncol), dtype=float)
    recharge_rate = 5.E-3 / 365 # m/d
    recharge[0:6, 1] = recharge_rate
    recharge[ -3:-1, 1] = recharge_rate
    recharge[ 6:10, 2] = recharge_rate
    recharge[ -4:-2, 2] = recharge_rate
    recharge[ 10:30, 3] = recharge_rate
    recharge[ -5:-3, 3] = recharge_rate
    recharge[ 30:65, 4] = recharge_rate
    recharge[ -7:-4, 4] = recharge_rate
    recharge[ 65:-7, 5] = recharge_rate

    rch = flopy.modflow.mfrch.ModflowRch(mf, rech=recharge)

    delc = dis.delr.array
    delr = dis.delc.array
    total_recharge = 0.
    recharge_area = 0.
    for i in range(mf.nrow):
        for j in range(mf.ncol):
            recharge_area_cell = delr[i] * delc[j] * recharge[i,j] / recharge_rate #/ 1.e6 # km2
            total_recharge = (total_recharge + recharge_area_cell * recharge[i,j]) #/ 1.e9 * 365
            recharge_area = recharge_area + recharge_area_cell
            rate = total_recharge / recharge_area
    recharge_area = recharge_area / 1.e6
    total_recharge = total_recharge * 365 / 1.e6
    print('recharge area: {} [km2], total recharge: {} [MCM/y, recharge: {} [mm/y]'
          .format(recharge_area, total_recharge, total_recharge/recharge_area*1.e3))

    return rch


def get_rc(sr, x, y):
    """Return the row and column of a point or sequence of points
    in real-world coordinates.

    Parameters
    ----------
    x : scalar or sequence of x coordinates
    y : scalar or sequence of y coordinates

    Returns
    -------
    r : row or sequence of rows (zero-based)
    c : column or sequence of columns (zero-based)
    """

    import numpy.matlib

    if np.isscalar(x):
        c = (np.abs(sr.xcenter - x)).argmin()
        r = (np.abs(sr.ycenter - y)).argmin()
    else:
        xcp = np.array([sr.xcenter] * (len(x)))
        ycp = np.array([sr.ycenter] * (len(x)))
        [_, xlen] = np.shape(xcp)
        [_, ylen] = np.shape(ycp)
        c = np.abs(xcp - np.transpose(np.matlib.repmat(x,xlen,1))).argmin(axis=1)
        r = np.abs(ycp - np.transpose(np.matlib.repmat(y,ylen,1))).argmin(axis=1)
    return r, c



def genParamSamples(sampleSize, **kwargs):
    # Function to generate range of inputs for hk, vka, pumping rate in each well using latin hypercube sampling
    # kwargs contains parameters for LHS sampling. Each parameter is an array containing two
    # values: the min paramter value and the max parameter value

    from scipy.stats import uniform
    from scipy.stats import lognorm
    from scipy.stats import triang
    from pyDOE import lhs

    # Generate LHS samples
    numParam = len(kwargs)
    lhd = lhs(numParam, samples=sampleSize)

    # Loop through parameters, use LHS values to generate parameter samples
    params = {}
    i = 0
    for key, value in kwargs.items():
        loc = value[0]
        scale = value[1] - value[0]
        # if key == 'hk':
        #     s = 0.56
        #     sample = lognorm(s).ppf(lhd[:, i])
        # elif key == 'ss':
        #     loc = 0.02
        #     scale = 0.28
        #     c = 0.179
        #     sample = triang(c, loc=loc, scale=scale).ppf(lhd[:, i])
        # else:
        sample = uniform(loc=loc, scale=scale).ppf(lhd[:, i])
        params[key] = sample
        i += 1

    # params is a dictionary where keys are parameter names and values are arrays of samples
    return params



def build_spatial_reference(mf):
    from flopy.utils.reference import SpatialReference
    # Position grid at 22 degrees lat 46 degrees long using UTM 38
    rotation = 22.3
    # rotation = 0
    ll_lat = 22.5
    ll_long = 45.79
    import utm
    [easting, northing, zoneNum, zoneLet] = utm.from_latlon(ll_lat, ll_long)
    sr = SpatialReference(delr=mf.dis.delr, delc=mf.dis.delc, xll=easting, yll=northing, epsg='32638', rotation=rotation)
    return(mf, sr)



def buildModel(plotgrid):

    # Build grid and stationary MODFLOW files

    # Time step parameters
    perlen = 365
    nper = 30    # number of stress periods
    nstp = 52      # Number of time steps per stress period
    steady = [False] * nper
    startingHead = 375

    # Generate static MODFLOW input files

    # Name model
    import os
    if os.environ.get('SLURM_ARRAY_JOB_ID') != None:
        model_name = 'mod_' + os.environ.get('SLURM_ARRAY_JOB_ID') + '_' + os.environ.get('SLURM_ARRAY_TASK_ID')
    else:
        model_name = 'mod'

    # MODFLOW input files
    mf = flopy.modflow.Modflow(model_name, exe_name='./mf2005dbl')
    [dis, bas, nper, nstp, perlen, strt] = build_dis_bas_files(mf, startingHead, perlen, nper, nstp, steady)
    [mf, sr] = build_spatial_reference(mf)
    pcg = build_pcg_file(mf)
    rch = build_rch_file(mf, dis)
    [wel, numWells, well_loc, pump_rate] = build_wel_file(mf, sr)
    oc = flopy.modflow.ModflowOc(mf, stress_period_data={(0, 0): ['print head', 'save head']})  # output control

    # plot grid w/ boundary conditions
    if plotgrid:
        plotFunctions.grid_withBCs(mf, dis, sr, wel, rch, strt)

    return mf, pcg, wel, oc, dis, bas, nper, nstp, perlen, numWells, model_name, well_loc, pump_rate, steady, startingHead, sr, rch








































# def getGridCell_fromLatLong(lat, long, gridOffset):
#     from pyproj import Proj
#     import numpy as np
#     import visual as v
#
#     # Define projection for UTM coordinaates in zone 38 (near riyadh)
#     p = Proj(proj='utm', zone=38)
#
#     # Get UTM coordinates for lower left corner of grid
#     [gridx, gridy] = p(22, 46)
#
#     # Get UTM coorindates for input lat long
#     [x, y] = p(lat, long)
#
#     # Get distance between input point and grid corner in E-W, N-S axes
#     delx = x - gridx
#     dely = y - gridy
#
#     # Rotate distance to along grid axes
#     v1 = v.vector(delx, dely)
#     v2 = v.
#
#     # # Convert to distance along grid axes
#     # theta = np.arctan(dely / delx)
#     # alpha = np.radians(gridOffset)
#     # h = np.sqrt(delx ** 2 + dely ** 2)
#     # a = h * np.cos(theta - alpha)
#     # b = h * np.sin(theta - alpha)
#
#     return a, b

