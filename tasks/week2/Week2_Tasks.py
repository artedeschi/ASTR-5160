import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy.time import Time

def read_dat(fname):
    objs = Table.read(fname)
    return objs.as_array()
    
def init_fig():
    plt.figure()
    plt.grid()
    plt.gca().invert_xaxis()
    plt.xticks(fontsize=14)
    plt.xticks(fontsize=14)
    plt.xlabel('RA (deg)',fontsize=14)
    plt.ylabel('Dec (deg)',fontsize=14)
    
def Convert_Dec(dec):

    d,m,s = dec.split(':')
    sign = 1
    if "-" in d or "+" in d:
        if "-" in d:
             sign = -1
        d = d[1:]
    
    return sign*(float(d) + float(m)/60 + float(s)/3600)
    
def Convert_RA(ra):
    h,m,s = ra.split(':')
    
    return 15*(float(h)+float(m)/60+float(s)/3600)
    
    
def plot_coords(obj_dat, thresh = 0, col = 'b'):
    qs = obj_dat['EXTINCTION'][:,0] >= thresh 
    
    RAs = obj_dat['RA'][qs]
    Decs = obj_dat['DEC'][qs]
    
    
    plt.scatter(RAs,Decs,color=col,marker='*',label='E > '+str(thresh))
    plt.grid()
    plt.gca().invert_xaxis()
    
    plt.legend(fontsize=14)
    
        
if __name__ == "__main__":
    #Lesson 3
    fname = "/d/scratch/ASTR5160/week2/struc.fits"
    obj_dat = read_dat(fname)
    init_fig()
    plot_coords(obj_dat)
    plot_coords(obj_dat,thresh = 0.22, col = 'r')
    
    #Lesson 4
    #------------------------------------------
    
    RA = "19:58:21.676"
    Dec ="35:12:05.78"
    
    Coord = SkyCoord(RA,Dec,unit=('hourangle','deg'))
    
    print("Using SkyCoord module")
    print(f"Dec = {Dec} --->  {Coord.dec.deg:.6f} deg")
    print(f"RA = {RA} --->  {Coord.ra.deg:.6f} deg")
    
    print("Using exact formulae:")
    RAdeg = Convert_RA(RA)
    Decdeg = Convert_Dec(Dec)
    
    print(f"Dec = {Dec} --->  {Decdeg:.6f} deg")
    print(f"RA = {RA} --->  {RAdeg:.6f} deg")
    
    time = Time.now()
    jdtime = time.jd
    mjdtime = time.mjd
    
    print(f"Current Time = JD {jdtime:.5f} = MJD {mjdtime:.5f}")
    
    mjdconv = jdtime - 2400000.5
    
    print(f"Current MJD using JD - 2400000.5: {mjdconv:.5f}")
    
    adj_mjd = np.arange(-10,11,1)+round(mjdtime)
    
    print("Some adjacent MJD dates:")
    print(adj_mjd)
    
    plt.show()
