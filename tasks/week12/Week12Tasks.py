import numpy as np
from astropy.io import fits
from astropy.table import Table, vstack
from astropy.coordinates import SkyCoord
import astropy.units as u
import os
from glob import glob
from tasks.week9.Week9Tasks import flux2mag
from sklearn import neighbors
from numpy import random
import matplotlib.pyplot as plt


def isGood(sweepdat, inds, flag = 'ALLMASK'):
    #general for one or multiple objects
    objs = sweepdat[inds]
    ii = (objs[flag] == 0)
    return ii
    
def magCuts(sweepdata, rmagcut):

    rmags = flux2mag(sweepdata['FLUX_R'])  
    magcuts = (rmags < rmagcut)
    return magcuts

def matchObj(objCoord,allsweepcoords,maxsep = 0.5*u.arcsec):
    seps = objCoord.separation(allsweepcoords)
    goodind = np.argmin(seps)
    if np.min(seps.to(u.arcsec).value) < maxsep.value:
        return goodind
    else:
        return None

def matchAllObj(objCoords,sweepcoords,maxsep = 0.5*u.arcsec):
    inds,seps,sep3d = objCoords.match_to_catalog_sky(sweepcoords)
    goodinds = inds[seps.to(u.arcsec).value < maxsep.value]
    return goodinds
    
def matchSurvey(objCoords,surveycen,maxsep = 3*u.deg):
    
    seps = surveycen.separation(objCoords)
    inds = seps.to(u.deg).value < maxsep.value
    return inds

def sweepCuts(sweepdata, cencoord, maxsep = 3*u.deg, minrmag = 20, colorcut = 0.5):
    
    print("Making Cuts")
    Allsweepcoords = SkyCoord(sweepdata['RA'],sweepdata['DEC'],unit=u.deg)
    isPSF = (sweepdata['TYPE'] == 'PSF')
    PSFs = sweepdata[isPSF]
    PSFcoords = Allsweepcoords[isPSF]
    
    inCircle = matchSurvey(cencoord,PSFcoords,maxsep=maxsep)
    #gets indices of coords in PSFcoords within 3deg of cencoord. maxsep must be in arcsec
    psfobjs = PSFs[inCircle]
    rmags = flux2mag(psfobjs['FLUX_R'])
    W1mags = flux2mag(psfobjs['FLUX_W1'])
    W2mags = flux2mag(psfobjs['FLUX_W2'])
    magcuts = magCuts(psfobjs, 20)
    brightpsfobjs = psfobjs[magcuts]
    
    return brightpsfobjs
    
def choose_subset(dat,length = 100):
    allinds = np.arange(0,len(dat))
    subinds = random.choice(allinds,size=length,replace=False)
    subdat = dat[subinds]
    nans = (np.isnan(subdat)) | (np.isinf(subdat))
    while nans.any():
        for i in range(len(nans)):
            if nans[i].any():
                subdat[i] = dat[random.choice(allinds,size=1,replace=False)]
        nans = (np.isnan(subdat)) | (np.isinf(subdat))
    return subdat
        
def gen_mock_dat(g_zdat,r_W1dat,n):
    ming_z = np.nanmin(g_zdat)
    maxg_z = np.nanmax(g_zdat)
    minr_W1 = np.nanmin(r_W1dat)
    maxr_W1 = np.nanmax(r_W1dat)
    
    mockg_z = np.random.choice(np.linspace(ming_z,maxg_z,n),size=n)
    mockr_W1 = np.random.choice(np.linspace(minr_W1,maxr_W1,n),size=n)
    mockdat = np.column_stack([mockg_z,mockr_W1])
    return mockdat
    


if __name__ == "__main__":
    
    #Lesson 21    
    sweepdir =os.path.join(os.getenv("ASTR5160"), 'data','legacysurvey','dr9','south','sweep','9.0')
    sweepfile = os.path.join(sweepdir,'sweep-180p020-190p025.fits')

    objcoord = SkyCoord(188.53667, 21.04572,unit=u.deg)
    sweepdat = Table(fits.open(sweepfile)[1].data)
    sweepcoords = SkyCoord(sweepdat['RA'],sweepdat['DEC'],unit=u.deg)
    objind = matchObj(objcoord,sweepcoords)
    print(f'TYPE = {sweepdat[objind]["TYPE"]}')
    #Profile type is "EXP", which is a type of galactic profile.
    
    goodbands = [isGood(sweepdat,objind,flag='ALLMASK_G'),isGood(sweepdat,objind,flag='ALLMASK_R'),
    isGood(sweepdat,objind,flag='ALLMASK_Z')]
    print(f'Good bands for g, r, and z: {goodbands}')
    #Looks like none of the bands are good here
    
    #Looking on the Legacy Survey Sky Viewer
    #The object looks more like a foreground star than a galaxy
    #It is indeed very saturated in the center
    #The viewer shows Gaia data giving it a parallax of 1.3mas +/- 0mas
    #leading me to believe that it is a star with a measurable parallax
    #It also says that SDSS defined its profile with a PSF (point source profile)
    
    #4 files that includes area 3deg from RA=180deg Dec = +30deg
    print('Reading Sweep files. This may take a few minutes...')
    sweepfiles = [os.path.join(sweepdir,'sweep-180p030-190p035.fits'),os.path.join(sweepdir,'sweep-170p030-180p035.fits'),
    os.path.join(sweepdir,'sweep-180p025-190p030.fits'),os.path.join(sweepdir,'sweep-170p025-180p030.fits')]
    Allsweepdats = vstack([Table(fits.open(sweepf)[1].data) for sweepf in sweepfiles])
    
    cencoord = SkyCoord(180,30,unit=u.deg)
    psfobjs = sweepCuts(Allsweepdats,cencoord)
    print(f'number of psf objects = {len(psfobjs)}')
    psfcoords = SkyCoord(psfobjs['RA'],psfobjs['DEC'],unit=u.deg)
    
    qsofile = os.path.join(os.getenv("ASTR5160"), 'week10','qsos-ra180-dec30-rad3.fits')
    qsodat = Table(fits.open(qsofile)[1].data)
    qsocoords = SkyCoord(qsodat['RA'],qsodat['DEC'],unit=u.deg)

    matchQSOs = matchAllObj(qsocoords,psfcoords)
    sweepqsos = psfobjs[matchQSOs]
    qsos = sweepqsos[magCuts(sweepqsos,20)]
    print(f'Number of matched QSOs = {len(qsos)}')
    
    #----------------------------------------------------
    #Part 3 of lesson 22
    #Other parts can be found in notebook
    
    psfg_z = flux2mag(psfobjs['FLUX_G']) - flux2mag(psfobjs['FLUX_Z'])
    qsog_z = flux2mag(qsos['FLUX_G']) - flux2mag(qsos['FLUX_Z'])

    psfr_W1 = flux2mag(psfobjs['FLUX_R']) - flux2mag(psfobjs['FLUX_W1'])
    qsor_W1 = flux2mag(qsos['FLUX_R']) - flux2mag(qsos['FLUX_W1'])

    psfdat = np.column_stack([psfg_z,psfr_W1])
    qsodat = np.column_stack([qsog_z,qsor_W1])

    #making small subset up psfdat that is same length as qsodat
    psfdat_sub = choose_subset(psfdat,len(qsodat))

    fulldat = np.concatenate([psfdat_sub,qsodat])
    dat_class = np.concatenate([np.zeros(len(psfdat_sub), dtype='i'), 
                                np.ones(len(qsodat), dtype='i')])

    #0s are stars. 1s are QSOs
    
    qsoknn = neighbors.KNeighborsClassifier(n_neighbors = 4)  #4 seems reasonable here
    qsoknn.fit(fulldat,dat_class)
    
    n = 100000
    combg_z = np.concatenate([psfg_z,qsog_z])
    combr_W1 = np.concatenate([psfr_W1,qsor_W1])
    combg_z = combg_z[~np.isinf(combg_z)]
    combr_W1 = combr_W1[~np.isinf(combr_W1)]
    mockdat = gen_mock_dat(combg_z,combr_W1,n)
    
    mock_class = qsoknn.predict(mockdat)
    
    numqso = np.sum(mock_class)  #0s are stars. 1s are QSOs
    fracqso = numqso/n
    print(f'Fraction of point-like objects classified as QSO is {fracqso*100:.2f}')
    
    fig, ax = plt.subplots(1, 1, figsize=(8,6))
    colors = ['r','b']
    labs = ['Stars','QSOs']
    for i in range(2):
        target_class = mock_class == i
        ax.scatter(mockdat[target_class, 0], mockdat[target_class, 1], s=10, c= colors[i], label=labs[i])
        ax.tick_params(labelsize=14)
        ax.set_xlabel("g - z", size=15)
        ax.set_ylabel("r - W1", size=15)
        ax.legend(prop={'size': 15})
    plt.grid()
    
    plt.show()
