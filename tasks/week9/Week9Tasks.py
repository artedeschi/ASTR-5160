import numpy as np
from astropy.io import fits
from astropy import units
from astropy.table import Table
from astropy.coordinates import SkyCoord
import matplotlib.pyplot as plt
import os


def Johnson2ugriz(U,B,V,R,I,isQSO = False):
    #Conversion based on Jester et al. (2005)
    if isQSO:
        g = V+0.74*(B-V)-0.07
        r = V-0.19*(B-V)-0.02
        u = 1.25*(U-B)+1.02+g
        i = r-(0.90*(R-I)-0.20)
        z = r-(1.20*(R-I)-0.20)
    
    elif U - B < 0:
        g  = V+0.64*(B-V)-0.13
        r  = V-0.46*(B-V)+0.11
        u  = 1.28*(U-B)+1.14+g
        i = r-(0.98*(R-I)-0.22)
        z = r-(1.69*(R-I) - 0.42)
             
    else:
        g = V+0.6*(B-V)-0.12
        r = V-0.42*(B-V)+0.11
        u = 1.28*(U-B)+1.13+g
        i = r-(0.91*(R-I)-0.20)
        z = r-(1.72*(R-I)-0.41)
        
    return u,g,r,i,z

def flux2mag(F):
    #F in nanomaggies
    mag = 22.5-2.5*np.log10(F)
    return mag
    
#def matchObj(objCoord,sweepcoord):
#    seps = objCoord.separation(sweepcoord)
#    if any(seps.to(units.arcsec).value < 0.5):
#        index = np.argmin(seps.value)
#    else:
#        index = np.nan
#    return index
    
def matchAllObj(objCoords,allsweepcoords,maxsep = 0.5):
    inds,seps,sep3d = objCoords.match_to_catalog_sky(allsweepcoords)
    if objCoords.ndim > 0: #If there's more than one obj coord
        inds[seps.to(units.arcsec).value > maxsep] = -1
    else:
        if seps.to(units.arcsec).value > maxsep:
            inds = -1
    return inds
    
    
def plot_rW1_gz(starmags,qsomags):
    sg,sr,sz,sW1,sW2 = starmags
    qg,qr,qz,qW1,qW2 = qsomags
    
    plt.figure()
    plt.scatter(sg-sz,sr-sW1,c='r',label='Stars')
    plt.scatter(qg-qz,qr-qW1,c='b',label='QSOs')
    
    #Cuts based made roughly by eye
    xline = np.arange(0.3,8,.1)
    ycut = xline-1
    plt.plot(xline,ycut,'k--',label='QSO cut')
    xline2 = np.arange(-2,0.5,.1)
    ycut2=-xline2*1.3-.1
    plt.plot(xline2,ycut2,'k--')
    plt.xlabel('g-z')
    plt.ylabel('r-W1')
    plt.grid()
    plt.legend()
    
    
def isQSO(mags):
    #Based on color cuts from above
    g,r,z,W1,W2 = mags
    
    x = g-z
    y = r-W1
    q = (y >= x-1) & (y > -x*1.3-0.1)
    return q

        


if __name__ == "__main__":
    #Lesson 17
    import warnings
    warnings.simplefilter("ignore") #ignore warnings to make output cleaner
    
    V = 15.256
    B = V+0.873
    U = B+0.320
    R = V-0.505
    I = R-0.511
    
    u,g,r,i,z = Johnson2ugriz(U,B,V,R,I)
    
    print('g from SDSS Navigator = 15.70')
    print(f'g from UBVRI observation = {g:.2f}')
    
    print('z from SDSS Navigator = 14.55')
    print(f'z from UBVRI observation = {z:.2f}')
    
    #They seem very close. They both overestimate
    #the brightness a bit, but it's still only off by 0.04 mag max.
    
    #RA = 248.85827 deg
    #Dec = 9.79809 deg
    #Sweepfile needed is: sweep-240p005-250p010.fits
    
    sweepdir =os.path.join(os.getenv("ASTR5160"), 'data','legacysurvey','dr9','south','sweep','9.0') 
    fname = os.path.join(sweepdir,'sweep-240p005-250p010.fits')
    starcoord = SkyCoord(248.85827,9.79809,unit=units.deg)
    sweepdat = fits.open(fname)[1].data
    sweepcoord = SkyCoord(sweepdat['RA'],sweepdat['Dec'],unit=units.deg)
    starind = matchAllObj(starcoord,sweepcoord)
    
    FluxG = sweepdat['FLUX_G'][starind]  #in nanomaggies
    FluxR = sweepdat['FLUX_R'][starind]
    FluxZ = sweepdat['FLUX_Z'][starind]
    
    sweepg = flux2mag(FluxG)
    sweepr = flux2mag(FluxR)
    sweepz = flux2mag(FluxZ)
    
    print('-----------------------------------------------')
    print('g from SDSS Navigator = 15.70')
    print(f'g from sweep observation = {sweepg:.2f}')
    print('r from SDSS Navigator = 15.19')
    print(f'r from sweep observation = {sweepr:.2f}')
    print('z from SDSS Navigator = 14.55')
    print(f'z from sweep observation = {sweepz:.2f}')
    
    FluxW1 = sweepdat['FLUX_W1'][starind]  #also in nanomaggies
    FluxW2 = sweepdat['FLUX_W2'][starind]
    FluxW3 = sweepdat['FLUX_W3'][starind]
    FluxW4 = sweepdat['FLUX_W4'][starind]    
    
    sweepW1 = flux2mag(FluxW1)
    sweepW2 = flux2mag(FluxW2)
    sweepW3 = flux2mag(FluxW3)
    sweepW4 = flux2mag(FluxW4)
    #print(FluxW4)
    #Flux for W4 is negative. This is an aphysical measurement.
    
    print(f'WISE infrared Magnitudes: W1 = {sweepW1:.2f}, W2 = {sweepW2:.2f}, W3 = {sweepW3:.2f}, W4 = {sweepW4:.2f}')
    
    #It does not seem like they captured flux for W4
    
    
    #------------------------------------------------------------------------------------
    #Lesson 18
    
    starfile =  os.path.join(os.getenv("ASTR5160"), 'week10','stars-ra180-dec30-rad3.fits')
    qsofile = os.path.join(os.getenv("ASTR5160"), 'week10','qsos-ra180-dec30-rad3.fits')
    
    stardat = fits.open(starfile)[1].data
    qsodat = fits.open(qsofile)[1].data
    
    #4 files: 
    #sweep-180p030-190p035.fits
    #sweep-170p030-180p035.fits
    #sweep-180p025-190p030.fits
    #sweep-170p025-180p030.fits
    print("Importing Sweeps Catalogs...")
    sweepfiles = [os.path.join(sweepdir,'sweep-180p030-190p035.fits'),os.path.join(sweepdir,'sweep-170p030-180p035.fits'),
    os.path.join(sweepdir,'sweep-180p025-190p030.fits'),os.path.join(sweepdir,'sweep-170p025-180p030.fits')]
    sweepdats = [fits.open(sweepf)[1].data for sweepf in sweepfiles]
    
    star_inds = [] #table index
    starf_inds = [] #file index
    qso_inds = []
    qsof_inds = []
        

    starcoords = SkyCoord(stardat['RA'],stardat['Dec'],unit=units.deg)
    AllSweeps = np.concatenate(sweepdats)
    #sweepscoords = [SkyCoord(sweepdat['RA'],sweepdat['DEC'],unit=units.deg) for sweepdat in sweepdats]
    AllSweepCoords = SkyCoord(AllSweeps['RA'],AllSweeps['DEC'],unit=units.deg)
    print("Matching Stars...")
    star_inds = matchAllObj(starcoords,AllSweepCoords)
    good_star_inds = star_inds[star_inds >= 0]
    print(f'Found {len(good_star_inds)} stars out of {len(star_inds)} total in sweep files')

    print("Matching QSOs...")
    qsocoords = SkyCoord(qsodat['RA'],qsodat['Dec'],unit=units.deg)
    qso_inds = matchAllObj(qsocoords,AllSweepCoords)
   
    print(f'Found {np.sum(qso_inds > 0)} QSOs out of {len(qsodat)} total in sweep files')

    starFGs = AllSweeps['FLUX_G'][good_star_inds]/AllSweeps['MW_TRANSMISSION_G'][good_star_inds]
    starFRs = AllSweeps['FLUX_R'][good_star_inds]/AllSweeps['MW_TRANSMISSION_R'][good_star_inds]    
    starFZs = AllSweeps['FLUX_Z'][good_star_inds]/AllSweeps['MW_TRANSMISSION_Z'][good_star_inds]    
    starFW1s = AllSweeps['FLUX_W1'][good_star_inds]/AllSweeps['MW_TRANSMISSION_W1'][good_star_inds]   
    starFW2s = AllSweeps['FLUX_W2'][good_star_inds]/AllSweeps['MW_TRANSMISSION_W2'][good_star_inds]
    qsoFGs = AllSweeps['FLUX_G'][qso_inds]/AllSweeps['MW_TRANSMISSION_G'][qso_inds]
    qsoFRs = AllSweeps['FLUX_R'][qso_inds]/AllSweeps['MW_TRANSMISSION_R'][qso_inds]    
    qsoFZs = AllSweeps['FLUX_Z'][qso_inds]/AllSweeps['MW_TRANSMISSION_Z'][qso_inds]    
    qsoFW1s = AllSweeps['FLUX_W1'][qso_inds]/AllSweeps['MW_TRANSMISSION_W1'][qso_inds]   
    qsoFW2s = AllSweeps['FLUX_W2'][qso_inds]/AllSweeps['MW_TRANSMISSION_W2'][qso_inds]
    
    
    star_gs = flux2mag(starFGs)
    star_rs = flux2mag(starFRs)
    star_zs = flux2mag(starFZs)
    star_W1s = flux2mag(starFW1s)
    star_W2s = flux2mag(starFW2s)
    star_mags = [star_gs,star_rs,star_zs,star_W1s,star_W2s]
    qso_gs = flux2mag(qsoFGs)
    qso_rs = flux2mag(qsoFRs)
    qso_zs = flux2mag(qsoFZs)
    qso_W1s = flux2mag(qsoFW1s)
    qso_W2s = flux2mag(qsoFW2s)
    qso_mags = [qso_gs,qso_rs,qso_zs,qso_W1s,qso_W2s]
    
    plot_rW1_gz(star_mags,qso_mags)
    
    print(f'Number of QSOs based on cuts = {np.sum(isQSO(qso_mags))+np.sum(isQSO(star_mags))}')
    
    plt.show()
