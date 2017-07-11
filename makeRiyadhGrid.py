
# Import packages
import numpy as np
import scipy as sp
import matplotlib as ml
import flopy

def build_dis_bas_files(mf, startingHead, perlen, nper, nstp, steady):

    # Make row spacings, units in meters
    delr = np.zeros(84)
    delr[0:13] = 10.E3  # 10 km
    delr[13:18] = 6.E3
    delr[18:25] = 3.5E3
    delr[25:28] = 1.8E3
    delr[28:45] = 0.8E3
    delr[45:68] = 1.E3
    delr[68:77] = 3.5E3
    delr[73] = 5.E3
    delr[74] = 6.E3
    delr[75:77] = 7.E3
    delr[77:84] = 10.E3
    xlength = sum(delr)

    # Make column spacings, units in meters
    delc = np.zeros(62)
    delc[0:10] = 10.E3
    delc[10:14] = 7.E3
    delc[14:19] = 5.E3
    delc[19:22] = 2.E3
    delc[22:50] = 1.E3
    delc[50] = 2.E3
    delc[51] = 4.E3
    delc[52] = 6.E3
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


    # # Time step parameters
    # perlen = 365
    # nper = 30  # number of stress periods
    # nstp = 365  # Number of time steps per stress period
    # steady = [False] * nper


    dis = flopy.modflow.ModflowDis(mf, nlay, nrow, ncol, delr=delc, delc=delr,
                                   top=ztop, botm=botm[1:],
                                   nper=nper, perlen=perlen, nstp=[nstp]*nper, steady=steady)

    # BAS file inputs
    ibound = np.ones((nlay, nrow, ncol), dtype=np.int32)  # Not sure if GW uses 0 or 1
    strt = startingHead * np.ones((nlay, nrow, ncol), dtype=np.float32)  # Starting head

    bas = flopy.modflow.ModflowBas(mf, ibound=ibound, strt=strt)

    return dis, bas, nper, nstp, perlen



def build_lpf_file(mf, **kwargs):

    laytyp = 1  # 1 = unconfined, 0 = confined
    hdry = 0  # dry cell head set to this number

    # Default paramter inputs if none given
    hk = 1.e-3
    vka = hk / 10
    sy = .25
    ss = 4.e-7 * 10

    for key, value in kwargs.items():
        if key == 'hk':
            hk = value
        elif key == 'vka':
            vka = value
        elif key == 'sy':
            sy = value
        elif key == 'ss':
            ss = value

    lpf = flopy.modflow.ModflowLpf(mf, hk=hk, vka=vka, sy=sy, ss=ss, laytyp=laytyp, hdry=hdry)

    return lpf

def build_pcg_file(mf):

    mxiter = 300
    hclose = 1e-1

    pcg = flopy.modflow.ModflowPcg(mf, mxiter=mxiter, hclose=hclose)

    return pcg




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





