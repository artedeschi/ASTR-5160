"""
Code by Adam Tedeschi
For ASTR5160 at UWyo 2025
HW3.py
"""

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.table import Table
from astropy.coordinates import SkyCoord
import astropy.units as u
import argparse
import os
import subprocess
from glob import glob
from tasks.week6.Week6Tasks import CalcArea
from tasks.week9.Week9Tasks import flux2mag
from tasks.week8.Week8Tasks import Find_Sweeps#, AddQuery2File
from tasks.week12.Week12Tasks import magCuts, isGood
from homeworks.HW3 import matchAllObj
import warnings


def isGood(sweepdat, flag = 'ALLMASK'):
    """
    NAME: isGood
 
    PURPOSE: Outputs truth array that can be used to mask out bad data
     according to the given 'flag'.
    

    INPUTS:
    
     sweepdata - astropy.table.Table object containing data from Legacy Survey
     Sweep files.
     flat -  valid masking flag for Legacy Survay Sweep file data
     
    OUTPUTS: 
    
     1D truth array that is the same length as 'sweepdata' that can be used
     to mask out flagged data.
    
    COMMENTS: None.
    """

    #general for one or multiple objects

    ii = (sweepdat[flag] == 0)
    return ii

def plot_template(sweepQSOdat,sweepNoQSOdat):
    """
    NAME: plot_template
 
    PURPOSE: Creates and shows plots of our template data used to 
     determine the best color cuts to use.
    

    INPUTS:
    
     sweepQSOdata - astropy.table.Table object containing data for previously identified
     QSOs in our template sweep data
     sweepNoQSOdata -  astropy.table.Table object containing data for previously identified
     as non-QSOs in our template sweep data
     
    OUTPUTS: 
    
     None
    
    COMMENTS: None.
    """
    #recover magnitudes for all bands available in sweep data
    fig, ax = plt.subplots(1, 1, figsize=(8,6))
    gmagQSO = flux2mag(sweepQSOdat['FLUX_G'])
    rmagQSO = flux2mag(sweepQSOdat['FLUX_R'])
    zmagQSO = flux2mag(sweepQSOdat['FLUX_Z'])
    W1magQSO = flux2mag(sweepQSOdat['FLUX_W1'])
    W2magQSO = flux2mag(sweepQSOdat['FLUX_W2'])
    W3magQSO = flux2mag(sweepQSOdat['FLUX_W3'])
    W4magQSO = flux2mag(sweepQSOdat['FLUX_W4'])    
    None
    gmagNoQSO = flux2mag(sweepNoQSOdat['FLUX_G'])
    rmagNoQSO = flux2mag(sweepNoQSOdat['FLUX_R'])
    zmagNoQSO = flux2mag(sweepNoQSOdat['FLUX_Z'])
    W1magNoQSO = flux2mag(sweepNoQSOdat['FLUX_W1'])
    W2magNoQSO = flux2mag(sweepNoQSOdat['FLUX_W2'])
    W3magNoQSO = flux2mag(sweepNoQSOdat['FLUX_W3'])
    W4magNoQSO = flux2mag(sweepNoQSOdat['FLUX_W4'])        
    
    
    QSOmags = [gmagQSO,zmagQSO,rmagQSO,W1magQSO,W2magQSO,W3magQSO,W4magQSO]
    NoQSOmags = [gmagNoQSO,zmagNoQSO,rmagNoQSO,W1magNoQSO,W2magNoQSO,W3magNoQSO,W4magNoQSO]
    
    
    goodQSOinds = splendid_function(sweepQSOdat)
    goodQSOmags = [qmags[goodQSOinds] for qmags in QSOmags]
    badNoQSOinds = splendid_function(sweepNoQSOdat)
    badNoQSOmags = [Noqmags[badNoQSOinds] for Noqmags in NoQSOmags]
    
    
    
    #data for lines that deliniate cuts
    xs = np.linspace(-0.5,2,100)
    ys = 2*xs-0.5
    
    #plot for first color cut
    ax.scatter(QSOmags[0]-QSOmags[2], QSOmags[1]-QSOmags[3], s=10, c= 'b', label='QSOs')
    ax.scatter(NoQSOmags[0]-NoQSOmags[2], NoQSOmags[1]-NoQSOmags[3], s=10, c= 'r', label='Not QSOs')
    ax.scatter(goodQSOmags[0]-goodQSOmags[2], goodQSOmags[1]-goodQSOmags[3], marker='o', s=20, edgecolor='y', facecolor='none', label='identified QSO')
    ax.scatter(badNoQSOmags[0]-badNoQSOmags[2], badNoQSOmags[1]-badNoQSOmags[3], marker='o', s=20, edgecolor='y', facecolor='none')
    ax.plot(xs,ys,'k--')
    ax.tick_params(labelsize=14)
    ax.set_xlabel("g - z", size=15)
    ax.set_ylabel("r - W1", size=15)
    ax.legend(prop={'size': 15})
    ax.grid()
    
    #plot for second color cut
    fig2, ax2 = plt.subplots(1, 1, figsize=(8,6))
    ax2.scatter(QSOmags[3]-QSOmags[4], QSOmags[5]-QSOmags[6], s=10, c= 'b', label='QSOs')
    ax2.scatter(NoQSOmags[3]-NoQSOmags[4], NoQSOmags[5]-NoQSOmags[6], s=10, c= 'r', label='Not QSOs')
    ax2.scatter(goodQSOmags[3]-goodQSOmags[4], goodQSOmags[5]-goodQSOmags[6], marker='o', s=20, edgecolor='y',facecolor='none', label='identified QSO')
    ax2.scatter(badNoQSOmags[3]-badNoQSOmags[4], badNoQSOmags[5]-badNoQSOmags[6], marker='o', s=20, edgecolor='y',facecolor='none')
    ax2.vlines(-.1,-5,5,color='k',linestyle='--')
    ax2.tick_params(labelsize=14)
    ax2.set_xlabel("W1 - W2", size=15)
    ax2.set_ylabel("W3 - W4", size=15)
    ax2.legend(prop={'size': 15})
    ax2.grid()
    
    #plot for PM cut
    fig2,ax3 = plt.subplots(1, 1, figsize=(8,6))
    ax3.scatter(QSOmags[3]-QSOmags[4], sweepQSOdat['PMDEC'], s=10, c= 'b', label='QSOs')
    ax3.scatter(NoQSOmags[3]-NoQSOmags[4], sweepNoQSOdat['PMDEC'], s=10, c= 'r', label='Not QSOs')
    ax3.scatter(goodQSOmags[3]-goodQSOmags[4], sweepQSOdat['PMDEC'][goodQSOinds], marker='o', s=20, edgecolor='y',facecolor='none', label='identified QSO')
    ax3.scatter(badNoQSOmags[3]-badNoQSOmags[4], sweepNoQSOdat['PMDEC'][badNoQSOinds], marker='o', s=20, edgecolor='y',facecolor='none')  
    ax3.hlines([-2,2],-10,10, linestyle='--', color='k')
    ax3.set_ylim(-12,12)
    plt.show()
    
    
def splendid_function(data):
    """
    NAME: splendid_function
 
    PURPOSE: Uses color magnitude cuts and data availability to identify 
    QSOs in an astropy table containing Legacy Survey Sweep file-like data.

    INPUTS:
    
     data - astropy.table.Table or structured numpy.ndarray containing at least
     the same columns used in a Legacy Survey Sweep table.
     
    OUTPUTS: 
    
     QSOinds - 1D truth array that is the same length as 'data'. 'True' corresponds
     to a positive identification of a QSO for that index in 'data'.
    
    COMMENTS: None.
    """
    
    magcuts = magCuts(data, 19)
    cutindat = data[magcuts]
    goodinds = (
    isGood(cutindat,flag='ALLMASK_G') & isGood(cutindat,flag='ALLMASK_R') & isGood(cutindat,flag='ALLMASK_Z')
    & isGood(cutindat,flag='WISEMASK_W1') & isGood(cutindat,flag='WISEMASK_W2')
    )
    gooddata = cutindat[goodinds]
    #Getting SDSS and WISE band magnitudes from fluxes in nanomaggies
    gmag = flux2mag(gooddata['FLUX_G'])
    rmag = flux2mag(gooddata['FLUX_R'])
    zmag = flux2mag(gooddata['FLUX_Z'])
    W1mag = flux2mag(gooddata['FLUX_W1'])
    W2mag = flux2mag(gooddata['FLUX_W2'])
 
    #number of observations in each filter band
    ngobs = gooddata['NOBS_G']
    nrobs = gooddata['NOBS_R']
    nzobs = gooddata['NOBS_Z']
    nobs = np.array([ngobs,nrobs,nzobs])
    
    nw3obs = gooddata['NOBS_W3']
    nw4obs = gooddata['NOBS_W4']
    w3w4obs = np.array([nw3obs,nw4obs])

    pmra = gooddata['PMRA']
    pmdec = gooddata['PMDEC']
    decs = gooddata['DEC']
    corr_pmra = np.cos(np.radians(decs))*pmra

    mags = [gmag,zmag,rmag,W1mag,W2mag]
    
    xs = mags[0]-mags[2]  # g-z
    ys = mags[1]-mags[3]  # r-W1
    xs2 = mags[3]-mags[4] # W1-W2
    
    #ART 2 color cuts are used. r-W1 > 2*(g-z)-0.5 and W1-W2 > -0.1
    #Since there are many targets missing grz data, I found that I capture more quasars if I
    #only use the second cut (W1-W2 > -0.1) if grz data is not present (returns false of no W1 data at all)
    #I also found that I can eliminate most false positives in the W1-W2 cut if I stipulate that W3 and W4
    #data is present
    
    cond1 = ((ys > 2*xs - 0.5) & (xs2 > -0.1))
    cond2 = ((xs2 > -0.1) & np.any(nobs == 0, axis=0) & np.any(w3w4obs > 0, axis=0))
    cond3 = (-2 < pmdec) & (pmdec < 2) & (-2 < corr_pmra) & (corr_pmra  < 2)
    
    QSOinds = ((cond1 | cond2) & cond3)
    #QSOinds = ((ys > 2*xs - 0.5) & (xs2 > -0.1)) | (xs2 > -0.1)
    #Captures a few false positives (seems unavoidable), but leaves even less false negatives
    
    return QSOinds


if __name__ == "__main__":

    warnings.filterwarnings("ignore") #supress wanings in output

    parser = argparse.ArgumentParser(description=
    """Identifies QSOs in some given Legacy Survey Sweep file-like data based on a template 
    of FIRST data matched with Legacy Survey Sweep data.
    
    Use looks like 'python HW4.py [path-to-input-file]'""")
    parser.add_argument("infile", nargs="?", default='/d/scratch/ASTR5160/data/legacysurvey/dr9/south/sweep/9.0/sweep-190p020-200p025.fits', help="Path to input file containing sweep-like data to identify QSOs in. No input defaults to '/d/scratch/ASTR5160/data/legacysurvey/dr9/south/sweep/9.0/sweep-190p020-200p025.fits'")
    parser.add_argument('-t', '--plot_template', action=argparse.BooleanOptionalAction,
    help="Plots template targets in color-color space to show how QSO identification cuts were determined.")
    #other argument that I tested, but the plots looked ugly so I took it out.
#    parser.add_argument('-p', '--plot_indata', action=argparse.BooleanOptionalAction,
#    help="Plots 'indata' targets in color-color space to show how QSO identification cuts were made.")

    args = parser.parse_args()
    infile = args.infile
    plot_temp = args.plot_template
#    plot_indata = args.plot_indata
    
    
    warnings.filterwarnings("ignore") #supress wanings in output
    
    print('Reading in datafile...')
    
    try:
        indata = fits.open(infile,memmap=True)[1].data
    except (OSError, FileNotFoundError):
        raise OSError(f'Invalid input. No valid datafile found at \"{infile}\"')
        
    #Making magnitude cuts and masking flagged data
    
    print('Identifying QSOs...')
    #Identifying QSOs
    QSOinds = splendid_function(indata)
    print(f'Number of QSOs identified = {np.sum(QSOinds)}')
    
#    if plot_indata:
#       QSOdat = goodindat[QSOinds]
#       NoQSOdat = goodindat[~QSOinds]
#        
#       plot_template(QSOdat,NoQSOdat)

    numQSOs = np.sum(QSOinds)
    sweep_area = CalcArea(190*u.deg,200*u.deg,20*u.deg,25*u.deg)
    QSOdensity = (numQSOs/(sweep_area*u.sr)).to(1/u.deg**2)
    print(QSOdensity)
    
    
    
    
    if plot_temp:
    
        QSOf = os.path.join(os.getenv("ASTR5160"), 'week10', 'qsos-ra180-dec30-rad3.fits')
        QSOdat = Table(fits.open(QSOf)[1].data)
        QSOcoords = SkyCoord(QSOdat['RA'],QSOdat['DEC'],unit=u.deg)
        
        print('Reading in Sweep files...')
        #Finding valid sweep files and reading in data 
        sweepdir = os.path.join(os.getenv("ASTR5160"), 'data','legacysurvey','dr9','north','sweep','9.0')
        allsweeps = glob(sweepdir+'/*.fits') #get list of sweep file names
        sweepfiles = Find_Sweeps(QSOdat, allsweeps)
        sweepdat = np.hstack([fits.open(sweepfile,memmap=True)[1].data for sweepfile in sweepfiles])
        magcuts = magCuts(sweepdat, 19)
        cutsweepdat = sweepdat[magcuts]
        
        goodinds = (
        isGood(cutsweepdat,flag='ALLMASK_G') & isGood(cutsweepdat,flag='ALLMASK_R') & isGood(cutsweepdat,flag='ALLMASK_Z')
        & isGood(cutsweepdat,flag='WISEMASK_W1') & isGood(cutsweepdat,flag='WISEMASK_W2')
        )
        
        goodsweepdat = cutsweepdat[goodinds]
        goodsweepcoords = SkyCoord(goodsweepdat['RA'],goodsweepdat['DEC'],unit=u.deg)
        print('Matching QSOs with sweep file objects...')


        matchQSOinds = matchAllObj(QSOcoords, goodsweepcoords, maxsep = 1*u.arcsec)
        sweepQSOdat = goodsweepdat[matchQSOinds]
        sweepNoQSOdat = goodsweepdat[~matchQSOinds]
        
        plot_template(sweepQSOdat,sweepNoQSOdat)
        
        
    
    
