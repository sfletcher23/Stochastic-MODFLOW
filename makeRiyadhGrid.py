
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



def build_dis_bas_files(mf, startingHead, perlen, nper, nstp, steady):

    # Make row spacings, units in meters
    delr = np.zeros(84)
    delr[0] = 15.E3
    delr[1:12] = 10.E3  # 10 km
    delr[12:15] = 7.E3
    delr[15:20] = 5.E3
    delr[20:25] = 3.E3
    delr[25:28] = 1.5E3
    delr[28:45] = 0.9E3
    delr[45:49] = 1.3E3
    delr[49:52] = 3.4E3
    delr[52:68] = 1.4E3
    delr[68:73] = 2.E3
    delr[68:77] = 3.5E3
    delr[73] = 5.E3
    delr[74] = 6.E3
    delr[75:77] = 8.E3
    delr[77:84] = 13.E3
    delr = np.flip(delr,0)
    xlength = sum(delr)

    # Make column spacings, units in meters
    delc = np.zeros(62)
    delc[0:10] = 10.E3
    delc[10:14] = 6.E3
    delc[14:17] = 4.E3
    delc[17] = 3.E3
    delc[18] = 2.E3
    delc[0:10] = 10.E3
    delc[10:14] = 6.E3
    delc[14:17] = 4.E3
    delc[17] = 3.E3
    delc[18] = 2.E3
    delc[19] = 1E3
    delc[20:31] = 0.7E3
    delc[31:50] = 1.E3
    delc[50] = 2.E3
    delc[51] = 4.E3
    delc[52] = 6.E3
    delc[53] = 8.E3
    delc[54:-1] = 10.E3
    delc[19] = 1E3
    delc[20:31] = 0.7E3
    delc[31:50] = 0.8E3
    delc[50] = 1.8E3
    delc[51] = 3.8E3
    delc[52] = 5.5E3
    delc[53] = 8.E3
    delc[54:-1] = 10.E3
    ylength = sum(delc)

    # Other grid paratmers
    ztop = 315.
    zbot = 0.
    nlay = 1
    nrow = np.size(delr)
    ncol = np.size(delc)
    delv = (ztop - zbot) / nlay
    botm = np.linspace(ztop, zbot, nlay + 1)

    # Constant grid spacing
    nrow += 0
    ncol += 25
    delc = 5.E3
    delr = 5.E3

    # Double grid spacing
    nrow *= 2
    ncol *= 2
    delc *= 0.5
    delr *= 0.5


    dis = flopy.modflow.ModflowDis(mf, nlay, nrow, ncol, delr=delc, delc=delr,
                                   top=ztop, botm=botm[1:],
                                   nper=nper, perlen=perlen, nstp=[nstp]*nper, steady=steady)

    # BAS file inputs
    ibound = np.ones((nlay, nrow, ncol), dtype=np.int32)  # Not sure if GW uses 0 or 1
    strt = startingHead * np.ones((nlay, nrow, ncol), dtype=np.float32)  # Starting head

    bas = flopy.modflow.ModflowBas(mf, ibound=ibound, strt=strt)

    return dis, bas, nper, nstp, perlen



def build_lpf_file(mf, samples = None):
# samples is a dictionary with a single sample for each i  nput paramter

    laytyp = 1  # 1 = unconfined, 0 = confined
    hdry = 0  # dry cell head set to this number

    # Default paramter inputs if none given
    hk = 1.e-3
    vka = hk / 10
    sy = .25
    ss = 4.e-7 * 10

    for key, value in samples.items():
        if key == 'hk':
            hk = value
        elif key == 'vka':
            vka = value
        elif key == 'sy':
            sy = value
        elif key == 'ss':
            ss = value

    lpf = flopy.modflow.ModflowLpf(mf, hk=hk, vka=vka, sy=sy, ss=ss, laytyp=laytyp, hdry=hdry)

    return lpf, hk, vka, sy



def build_pcg_file(mf):

    mxiter = 300
    hclose = 1e-1

    pcg = flopy.modflow.ModflowPcg(mf, mxiter=mxiter, hclose=hclose)

    return pcg



def build_wel_file(mf, sr):
    # Time step parameters
    perlen = 365
    nper = 30  # number of stress periods
    nstp = 365  # Number of time steps per stress period
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


    # Read well pumping rates from shapefile
    wellData= flopy.export.shapefile_utils.shp2recarray(shp)
    pumpRate_MCMy = [float(x[1]) for x in wellData]
    pump_rate = np.asarray(pumpRate_MCMy) * -1 * 1e6 /365   # m/d

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
    ll_lat = 22
    ll_long = 46
    import utm
    [easting, northing, zoneNum, zoneLet] = utm.from_latlon(ll_lat, ll_long)
    sr = SpatialReference(delr=mf.dis.delr, delc=mf.dis.delc, xll=easting, yll=northing, epsg='32638', rotation=rotation)
    return(mf, sr)



def buildModel(plotgrid):

    # Build grid and stationary MODFLOW files

    # Time step parameters
    perlen = 365
    nper = 30    # number of stress periods
    nstp = 365      # Number of time steps per stress period
    steady = [False] * nper
    startingHead = 200

    # Generate static MODFLOW input files

    # Name model
    model_name = 'mod'

    # MODFLOW input files
    mf = flopy.modflow.Modflow(model_name, exe_name='./mf2005dbl')
    [dis, bas, nper, nstp, perlen] = build_dis_bas_files(mf, startingHead, perlen, nper, nstp, steady)
    [mf, sr] = build_spatial_reference(mf)
    pcg = build_pcg_file(mf)
    [wel, numWells, well_loc, pump_rate] = build_wel_file(mf, sr)
    oc = flopy.modflow.ModflowOc(mf, stress_period_data={(0, 0): ['print head', 'save head']})  # output control

    # plot grid w/ boundary conditions
    if plotgrid:
        plotFunctions.grid_withBCs(mf, dis, sr, wel)

    return mf, pcg, wel, oc, dis, bas, nper, nstp, perlen, numWells, model_name, well_loc, pump_rate, steady, startingHead, sr








































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

