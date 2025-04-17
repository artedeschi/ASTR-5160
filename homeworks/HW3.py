"""
Code by Adam Tedeschi
For ASTR5160 at UWyo 2025
HW3.py
"""

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
#from astropy.table import Table, vstack
from astropy.coordinates import SkyCoord
import astropy.units as u
import argparse
import os
import subprocess
from glob import glob
from tasks.week9.Week9Tasks import flux2mag
from tasks.week8.Week8Tasks import Find_Sweeps#, AddQuery2File
import warnings



def matchSurvey(objCoords,surveycen,maxsep = 3*u.deg):
    """
    NAME: matchSurvey
 
    PURPOSE: Returns a truth array of objCoord indices that correspond to coordinates
     that lie within maxsep of surveycen.


    INPUTS:
    
     objCoords - astropy.coordinates.SkyCoord object containing a list of coordinates to match with the
     survey parameters
      
     surveycen - astropy.coordinates.SkyCoord object containing a single coordinate that represents the
     center of the survey area
     
     maxsep - OPTIONAL   astropy.quantity denoting max angular separation from
     surveycen that defines the survey area. | default = 3*u.deg
     
    OUTPUTS: 
    
     inds - truth array of indices that correspond to coordinates that
     lie within maxsep of surveycen.
    
    COMMENTS: Only survey it allows is a single circle around a central coordinate
    """
    
    
    seps = surveycen.separation(objCoords)
    inds = seps < maxsep
    return inds
    
def magCuts(sweepdata, rmagcut, colorcut):
    """
    NAME: magCuts
 
    PURPOSE: Returns a truth array of data indices that correspond targets that
     satisfy the magnitude critereon given.

    INPUTS:
    
     sweepdata - Structured Astropy Table contatining data form Legacy Survey sweep files
      
     colorcut - minimum color index for WISE W1 - W2 bands. 
     
    OUTPUTS: 
    
     magcuts - truth array of indices that correspond to targets in the Legacy Survey
     data that satify the magnitude critereon.
    
    COMMENTS: None
    """
    
    
    W1mags = flux2mag(sweepdata['FLUX_W1'])
    W2mags = flux2mag(sweepdata['FLUX_W2'])
    rmags = flux2mag(sweepdata['FLUX_R'])
    
    magcuts = (rmags < rmagcut) & (W1mags - W2mags > colorcut)
    return magcuts
    
#The only reason this is a separate function from matchAllObj is to avoid an extra 'if statement'.    
def matchObj(objCoord,surveycoords,maxsep=0.5*u.arcsec):
    """
    NAME: matchObj
 
    PURPOSE: Returns the index of surveycoords that 'matches' a signle objCoord
     object. The function finds the index of surveycoords coordinate that is closest
     to the objCoord coordinate, and considers it a 'match' if the separation between the 
     two is less than maxsep. Returns None if no match made.

    INPUTS:
    
     objCoord - astropy.coordinates.SkyCoord object of single coordinate
     
     surveycoords - astropy.coordinates.SkyCoord object containing a list of coordinates to match with the
     given objCoord parameter
     
     maxsep - OPTIONAL   astropy.quantity denoting max angular separation between
      objCoord and the closest surveycoord that defines a 'match'. | default = 0.5*u.arcsec
     
    OUTPUTS: 
    
     ind - index (int) of surveycoords that corresponds to the matching objCoord. Returns None
     if no match is made.
    
    COMMENTS: None
    """

    #sep is an astropy.quantity angle too
    ind,sep,sep3d = objCoord.match_to_catalog_sky(surveycoords)
    if sep > maxsep:
        ind = None
    return ind
    
def matchAllObj(objCoords,surveycoords,maxsep = 0.5*u.arcsec):
    """
    NAME: matchAllObj
 
    PURPOSE: Returns the indices of surveycoords that 'matches' the each objCoord.
     The function finds the indices of surveycoords coordinates that is closest
     to the each objCoord coordinate, and considers each a 'match' if the separation between the 
     two is less than maxsep.

    INPUTS:
    
     objCoords - aastropy.coordinates.SkyCoord object containing a list of coordinates to match with
     surveycoords
     
     surveycoords - astropy.coordinates.SkyCoord object containing a list of coordinates to match with the
     given objCoords
     
     maxsep - OPTIONAL   astropy.quantity denoting max angular separation between
      objCoords and its closest surveycoord that defines a 'match'. | default = 0.5*u.arcsec
     
    OUTPUTS: 
    
     uniqueinds - numpy.ndarray of indices of surveycoords that corresponds to the matching objCoord. 
    
    COMMENTS: None
    """
    
    
    inds,seps,sep3d = objCoords.match_to_catalog_sky(surveycoords)
    goodinds = inds[seps < maxsep]
    #filter out repeats
    uniqueinds = np.unique(goodinds)
    return uniqueinds

def mag2flux(mag):
    """
    NAME: mag2flux
 
    PURPOSE: Converts (a) magnitude(s) into flux(es) in nanomaggies

    INPUTS:
    
     mag - float or numpy.ndarray of floats containing magnitudes to convert
     
     
    OUTPUTS: 
    
     F - float or numpy array of floats containing converted fluxes in nanomaggies
    
    COMMENTS: None
    """


    #F in nanomaggies
    F = 10**((22.5-mag)/2.5)
    return F

def AddQuery2List(RA, Dec):
    """
    NAME: AddQuery2List
 
    PURPOSE: Returns output of sdssDR9query.py taking inputs RA and Dec 
     as a string to be added to a list of sdss queries.

    INPUTS:
    
     RA - float of right assension in degrees
     
     Dec - float of declination in degrees.
     
     
    OUTPUTS: 
    
     output - Output of sdssDR9query formated as a string.
    
    COMMENTS: None
    """

    qfile = os.path.join(os.getenv("ASTR5160"), 'week8', 'sdssDR9query.py')
    #Takes what is printed to terminal by sdssDR9query.py and turns 
    #it into a subprocess.CompletedProcess object, which can be easily converted
    #into a string via .stdout.strip()
    result = subprocess.run(
        ['python', qfile, str(RA), str(Dec)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    output = result.stdout.strip()
    return output
    
def CleanOutput(output):
    """
    NAME: CleanOutput
 
    PURPOSE: Converts 1D list of strings returned by AddQuery2List into a 2D numpy
     array of floats. Removes lines where 'No objects have been found' was returned

    INPUTS:
    
     output - list of strings returned by AddQuery2List
     
    OUTPUTS: 
    
     float_arr - 2D numpy.ndarray of cleaned up data in 'output' converted to floats
     
     mask - truth array denoting indices of output where no objects were found as False
    
    COMMENTS: None
    """


    output = np.array(output)
    mask = output != 'No objects have been found'
    clean_output = output[mask]
    split_arr = np.char.split(clean_output, ',')  # splits each string into a list of strings
    float_arr = np.array(split_arr.tolist(), dtype=float)  # convert to float array
    return float_arr,mask
    
    
def plot_Fluxes(fluxs, Flam = False):
    """
    NAME: CleanOutput
 
    PURPOSE: Plots fluxes measured in  measured in the u, g, r, i, z, W1, W2, W3, and W4 bands
     against wavelength in nm.

    INPUTS:
    
     fluxs - list of 9 floats representing fluxes in units of nanomaggies
     
     Flam - Booleon determining if to plot pure Flux, or Flux-Lambda (flux/wavelength)
     
    OUTPUTS: 
    
     None
    
    COMMENTS: Currently setup to plot in a log-log scale. Flam will compare better to 
     the SDSS spectra shown on the Skyserver site.
    """

    
    #Effective wavelengths found online
    uwav = 354 #in nm
    gwav = 475 
    rwav = 622
    iwav = 763
    zwav = 905
    w1wav= 3368
    w2wav= 4618
    w3wav= 12082
    w4wav= 22194
    
    wavs = [uwav,gwav,rwav,iwav,zwav,w1wav,w2wav,w3wav,w4wav]
    #filternames = ['u','g','r','i','z','W1','W2','W3','W4']
    plt.figure(figsize=(8,6))
    if Flam:
        plt.plot(wavs,np.array(fluxs)/np.array(wavs),'bo',linestyle='None')#,label=filternames)
        plt.ylabel(r'$F_\lambda$ (nanomaggies/nm)')
    else:
        plt.plot(wavs,fluxs,'bo',linestyle='None')
        plt.ylabel('Flux (nanomaggies)')
        
    plt.xlabel('Wavelength (nm)')
    #plt.legend()
    plt.loglog()
    plt.grid()


if __name__ == "__main__":


    warnings.filterwarnings("ignore") #supress wanings in output

    parser = argparse.ArgumentParser(description=
    """Cross match objects from FIRST, Legacy Survey, 
    and SDSS within a 3deg circle around (RA,Dec) = (163,50) and analyze data from
    the object with the brightest u-band magnitude""")
    parser.add_argument('-L', '--plot_flam', action=argparse.BooleanOptionalAction,
    help="Plots Flux/wavelength in stead of Flux when flag is active.")

    args = parser.parse_args()
    flam = args.plot_flam
    
    
    
    #Reading in FIRST data
    print('Reading in FIRST data...')
    FIRSTfile = os.path.join(os.getenv("ASTR5160"), 'data', 'first', 'first_08jul16.fits')
    FIRSTdat = fits.open(FIRSTfile)[1].data
    FIRSTcoords = SkyCoord(FIRSTdat['RA'],FIRSTdat['DEC'],unit=u.deg)
    
    surveycen = SkyCoord(163,50,unit=u.deg)
    #Filtering objects to be contained within 3-deg circle around (163,50)
    goodinds = matchSurvey(FIRSTcoords,surveycen,maxsep = 3*u.deg)
    goodFIRST = FIRSTdat[goodinds]
    goodFIRSTcoords = FIRSTcoords[goodinds]
    
    print('Reading in Sweep files...')
    #Finding valid sweep files and reading in data 
    sweepdir = os.path.join(os.getenv("ASTR5160"), 'data','legacysurvey','dr9','north','sweep','9.0')
    allsweeps = glob(sweepdir+'/*.fits') #get list of sweep file names
    sweepfiles = Find_Sweeps(goodFIRST, allsweeps)
    sweepdat = np.hstack([fits.open(sweepfile,memmap=True)[1].data for sweepfile in sweepfiles])

    
    #filtering objects to satisfy magnitude cuts
    magcuts = magCuts(sweepdat, 22, 0.5)
    cutsweepdat = sweepdat[magcuts]
    cutsweepcoords = SkyCoord(cutsweepdat['RA'],cutsweepdat['DEC'],unit=u.deg)

    #matching filtered FIRST and sweep data objects
    matchFIRSTinds = matchAllObj(cutsweepcoords, goodFIRSTcoords, maxsep = 1*u.arcsec)
    goodFIRSTdat = goodFIRST[matchFIRSTinds]

    print(f'Number of objects matched between FIRST and sweep files = {len(goodFIRSTdat)}')    
    
    #Matching SDSS data with FIRST data
    output = []
    print("Querying SDSS database...")
    for n in range(len(goodFIRSTdat)):
        #print(f'SDSS Query {n}')
        FirstRA = goodFIRSTdat['RA'][n]
        FirstDec = goodFIRSTdat['DEC'][n]
        output.append(AddQuery2List(FirstRA,FirstDec))
        
#    print("Querying SDSS database...")   
#    for n in range(len(goodFIRSTdat)):
#        FirstRA = goodFIRSTdat['RA'][n]
#        FirstDec = goodFIRSTdat['DEC'][n]
#        qfile = 'HW3.txt'
#        AddQuery2File('HW3.txt',FirstRA,FirstDec)
#    output = open(qfile,'r').readlines()

    #cleaning output to be more usable.
    
    clean_output,mask = CleanOutput(output)
    cleanFIRSTdat = goodFIRSTdat[mask]
    print(f'Number of objects matched between FIRST and SDSS files = {len(cleanFIRSTdat)}')
    
    #identifying ubrite1 as brightest object in u, and getting data for it in
    #each survey
    SDSSus = clean_output[:,2]
    SDSSis = clean_output[:,5]
    ubrite1_ind = np.argmin(SDSSus)
    FIRST_ubrite1 = cleanFIRSTdat[ubrite1_ind]
    ubrite1coord = SkyCoord(FIRST_ubrite1['RA'],FIRST_ubrite1['DEC'],unit=u.deg)
    sweep_ubrite1 = cutsweepdat[matchObj(ubrite1coord ,cutsweepcoords, maxsep = 1*u.arcsec)]
    
    ubrite1_Fu = mag2flux(SDSSus[ubrite1_ind])
    ubrite1_Fi = mag2flux(SDSSis[ubrite1_ind])

    ubrite1_Fg = sweep_ubrite1['FLUX_G']
    ubrite1_Fr = sweep_ubrite1['FLUX_R']
    ubrite1_Fz = sweep_ubrite1['FLUX_Z']
    ubrite1_FW1 = sweep_ubrite1['FLUX_W1']
    ubrite1_FW2 = sweep_ubrite1['FLUX_W2']
    ubrite1_FW3 = sweep_ubrite1['FLUX_W3']
    ubrite1_FW4 = sweep_ubrite1['FLUX_W4']
    ubrite1_fluxs = [
        ubrite1_Fu,ubrite1_Fg,ubrite1_Fr,ubrite1_Fi,ubrite1_Fz,
        ubrite1_FW1,ubrite1_FW2,ubrite1_FW3,ubrite1_FW4
    ]
    
    #plotting ubrite1 data    
    plot_Fluxes(ubrite1_fluxs,Flam=flam)
        
    
    print('The Galaxy on the SDSS SkyServer is a bright blue gaalxy at redshift z = 1.035136')
    print('It is classified as a QSO, which makes sense given the extremely broad emission lines')
    print('The metallic emission lines also rules out this being a distance B or O type star')
    print('The redshift is consistent with where we typically find QSOs')
    print("""Given QSO continuums are typically bright in the u-band (unless it's been redshifted into the Ly-alpha forest), 
the fact that we chose the brightest u-band object made it even more likely that the object we identified would be a QSO.""")  
    
    
    plt.show()
    
    
    
    
    
    
    
