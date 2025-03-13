import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.table import Table
import os
from glob import glob

def PlotData(data, title, varsize = False):
    fig = plt.figure(title)
    if varsize:
        size = (0.5*((max(data['gMag'])+1)-data['gMag']))**2
        #larger size corresponds to smaller gMag
        #Numbers like the 0.5 and +1 were arbitrarily chosen based on what looked nicest
    else:
        size= 10
    plt.scatter(data['RA'],data['Dec'],s=size,marker='o',color='b')
    plt.xlabel('RA (deg)')
    plt.ylabel('Dec (deg)')
    plt.gca().invert_xaxis()
    plt.title(title)
    plt.grid()

    return fig
    
def AddQuery2File(fname,RA,Dec):
    os.system(f'python sdssDR9query.py {RA} {Dec} >> {fname}')
    

def decode_sweep_name(sweepname):
    #ADAPTED FROM CODE PROVIDED ON WyoCourses
    sweepname = os.path.basename(sweepname)

    # ADM the RA/Dec edges.
    ramin, ramax = float(sweepname[6:9]), float(sweepname[14:17])
    decmin, decmax = float(sweepname[10:13]), float(sweepname[18:21])

    # ADM flip the signs on the DECs, if needed.
    if sweepname[9] == 'm':
        decmin *= -1
    if sweepname[17] == 'm':
        decmax *= -1

    return [ramin, ramax, decmin, decmax]
    
def is_in_box(objs, radecbox):
    #ADAPTED FROM CODE PROVIDED ON WyoCourses
    ramin, ramax, decmin, decmax = radecbox
    
    if decmin < -90. or decmax > 90. or decmax <= decmin or ramax <= ramin:
        msg = "Strange input: [ramin, ramax, decmin, decmax] = {}".format(radecbox)
        print(msg)
        raise ValueError(msg)

    inbox_array = ((objs["RA"] >= ramin) & (objs["RA"] < ramax)
          & (objs["DEC"] >= decmin) & (objs["DEC"] < decmax))

    return inbox_array
    
def Find_Sweeps(data,sweepfiles):
    good_sweeps = []
    for f in sweepfiles:
        radecbox = decode_sweep_name(f)
        if any(is_in_box(data,radecbox)):
            good_sweeps.append(f)
    return good_sweeps
    



if __name__ == "__main__":

    #Lesson 15
    #Query used:
    #SELECT p.ra, p.dec, p.modelMag_g, p.type
    #FROM PhotoObjAll p, dbo.fGetNearbyObjEq(300, -1, 2) n
    #WHERE p.objID = n.objID
    
    dtype = [('RA',float),('Dec',float),('gMag',float),('type',int)]
    SDSSdat = np.genfromtxt('SDSS_Query.csv',dtype=dtype,delimiter = ',',skip_header=1)
    fig1=PlotData(SDSSdat,'SDSS Obj Locs')
    fig2=PlotData(SDSSdat,'SDSS Obj Locs with Marker Size Proportional to Magnitude',varsize=True)
    #Make a key press exit the plots and continue to the next lesson
    print("Press any key to close plots and continue...")
    def on_key(event):
        plt.close(fig1)
        plt.close(fig2)
    fig1.canvas.mpl_connect('key_press_event', on_key)
    fig2.canvas.mpl_connect('key_press_event', on_key)
    plt.show()
    #Plots look like the SDSS Navigator Tool View
    
    #---------------------------------------------------------------------------------------
    #Lesson 16
    fname = 'SDSS_Query2.txt'
    f = open(fname,'w') #creating file to write queries to
    try:
        #env "ASTR5160" is a path variable set in my .bashrc with the value "/d/scratch/ASTR5160/"
        firstdata = os.path.join(os.getenv("ASTR5160"), "data", "first", "first_08jul16.fits")
        sweepdir = os.path.join(os.getenv("ASTR5160"), "data", "legacysurvey","dr9","north","sweep","9.0")
    except TypeError:
        print("Set the ASTR5160 environment variable to point to the main " +
        "ASTR5160 data directory")
        raise OSError
    FirstTable = Table(fits.open(firstdata)[1].data[:100])

    for n in range(len(FirstTable)):
        # I would avoid a for loop but your query code only seems to take in one coord
        FirstRA = FirstTable['RA'][n]
        FirstDec = FirstTable['DEC'][n]
        #This print slows things down, but makes the code look like it's doing something and not siting there for 
        #a minute and a half.
        print(f'Adding SDSSn data for {FirstRA}, {FirstDec}...')
        AddQuery2File(fname,FirstRA,FirstDec)
        

    sweepfiles = glob(sweepdir+'/*.fits') #get list of sweep file names
    good_sweeps = Find_Sweeps(FirstTable, sweepfiles)
   
    print(f'Number of Sweep files containing FIRST data: {len(good_sweeps)}')
    print(good_sweeps)
    f.close()
    
    #I indeed got 11 sweep files
    
