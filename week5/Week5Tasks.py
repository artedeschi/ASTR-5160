import numpy as np
from numpy.random import random
import astropy.units as u
from astropy.coordinates import SkyCoord
import healpy as hp

def GenData(N=1000000):
    ras = random(N)*360.
    decs = (180./np.pi)*np.arcsin(1.-random(N)*2.)
    
    return ras,decs
    
def GetPixDist(RAs,Decs):
    pixs = hp.ang2pix(1,RAs,Decs,lonlat=True)
    pix_num,num_in_pix = np.unique(pixs,return_counts=True)
    for n in range(len(pix_num)):
        print(f'{num_in_pix[n]} points in HEALpix # {pix_num[n]}')
        
def RA_Cap(RA):

    Dec = 0*u.deg
    h = 1.
    norm_coord = SkyCoord(RA+90*u.deg,Dec)
    xyz = np.array(norm_coord.cartesian.xyz.value,dtype='f2')
    four_vec = np.append(xyz,h)
    return four_vec

def Dec_Cap(Dec):
    
    RA = 0*u.deg
    point_dec = 90*u.deg
    norm_coord = SkyCoord(RA,point_dec)
    xyz = np.array(norm_coord.cartesian.xyz.value,dtype='f2')
    h = 1.-np.sin(Dec).value
    four_vec = np.append(xyz,h)
    return four_vec

def Coord_Cap(RA,Dec,theta):
    
    h = 1.-np.cos(theta).value
    norm_coord = SkyCoord(RA,Dec)
    xyz = np.array(norm_coord.cartesian.xyz.value,dtype='f2')
    four_vec = np.append(xyz,h)
    return four_vec
    

if __name__ == "__main__":
    #Lesson 9
    
    RAs,Decs = GenData()
    pixarea = hp.nside2pixarea(1)  #in Sr
    print(f'HEALpix area for nside=1 is {pixarea:.4f} Sr')
    #This is 1/12 of 4pi
    GetPixDist(RAs,Decs)
    #The number of points in each pixel are all about the same,
    #which is consistent if each HEALpix are were equal.
    
    #-------------------------------------------------------
    #Lesson 10
    
    RA=5*u.hourangle
    ra_4vec = RA_Cap(RA)
    Dec = 36*u.deg
    theta = 1*u.deg
    print('4-vector for great circle passing through RA=5hr is '+str(ra_4vec))
    dec_4vec = Dec_Cap(Dec)
    print('4-vector for dec-bounded cap passing through Dec=36deg is '+str(dec_4vec))
    coord_4vec = Coord_Cap(RA,Dec,theta)
    print('4-vector for a cap at RA = 5hr and Dec = 36deg bounded by 1deg is '+str(coord_4vec))
    
    
