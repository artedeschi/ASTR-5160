"""
Code by Adam Tedeschi
For ASTR5160 at UWyo 2025
HW1.py
"""

import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
from astropy.time import Time
from datetime import datetime
import argparse
import os

def ConvertCoord(coord):
    """
    Converts a coordinate string from hhmmss.ss+/-ddmmss.s to ##h##m##.##s +/-##d##m##.#s   
    to work with SkyCoord easier
    Inputs: coord (str in format hhmmss.ss+/-ddmmss.s)
    Outputs: convcoord (str in format ##h##m##.##s +/-##d##m##.#s)
    """


    convcoord = coord[:2]+'h'+coord[2:4]+'m'+coord[4:9] + 's ' +coord[9:12]+'d'+coord[12:14]+'m'+coord[14:]+'s'
    return convcoord

def MakeTable(month,coordlist,loc='kpno',shift=7):
    """
    Creates table of coordinates for the quasar with the lowest airmass at the location at 11PM local time
    Inputs: month (float from 1-12 representing January-December); coordlist (1D array of N strings containing 
    coordinates in "hhmmss.ss+/-ddmmss.ss" format; loc (string of observatory ID) default corresponds to Kitt Peak 
    National Observatory; shift (float timezone shift relative to UTC coorresponding to desired location) Default 
    corresponds to +7 (MT), which is the local time at KPNO
    Outputs: 5xN Structured Array of best quasar coordinate and its airmass to observe at the location for each 
    day of the month. 'N' is number of days in the month The 5 fields are 'obstimes' (local time), 'coords' (hms+/-dms), 
    'RAs' (deg), 'Decs' (deg), and 'Airmass'. This output is made to be directly input into the PrintTable() function
    """

    #ART Initializing and setting list of Dates in UTC as Time objects
    year = Time.now().datetime.year
    day = 1
    hour = 23
    minute = 0
    sec = 0
    date = Time(datetime(year,month,day,hour,minute,sec))
    utcdates = []
    while date.datetime.month == month:		  #ART While the date is still within the month
	     utcdates.append(date+shift*u.hour)   #ART add shift to convert from local to UTC (will decovert later)
	     date += 1*u.day
     
    coords = np.array([ConvertCoord(c) for c in coordlist]) #ART Converting cordinate strings into format that is easier for SkyCoord to Parse
    scoords = SkyCoord(coords,unit=(u.hourangle, u.deg)) #Creating Skycoord objecAltAz(location=LOC, obstime=utcdates)ts
    LOC = EarthLocation.of_site(loc)
    aa = AltAz(location=LOC, obstime=utcdates)	#ART Setting alt-az object with location and 28-31 dates
    exp_scoords = scoords[:,np.newaxis]         #ART Must add new axis to make frame tranformation work
    transcoords = exp_scoords.transform_to(aa)
    
    not_vis = ([transcoords.secz <= 0]*np.ones(transcoords.shape)*1e99)[0]   #ART Add very large number to negatives to find true min
    bestinds = np.argmin(transcoords.secz+not_vis,axis=0)
    
    obstimes = []
    formcoords = []
    RAs = []
    Decs = []
    airmasss = []
    for d in range(len(bestinds)):        #ART Formatting fields for table
        ind = bestinds[d]   #ART d means day here
        obstimes.append((aa[d].obstime-shift*u.hr).isot)    #ART Shfiting time back to local time
        formcoords.append(coordlist[ind])   #ART coords formatted like you originally had them
        RAs.append(scoords[ind].icrs.ra.deg)
        Decs.append(scoords[ind].icrs.dec.deg)
        airmasss.append(transcoords[ind,d].secz)
        
    dtype=[('obstimes','U23'),('coords','U18'),('RAs','f8'),('Decs','f8'),('airmass','f8')]	#ART Building table
    table = np.zeros(len(airmasss),dtype = dtype)
    table['obstimes'] = obstimes
    table['coords'] = formcoords
    table['RAs'] = RAs
    table['Decs'] = Decs
    table['airmass'] = airmasss


    return table


def PrintTable(table):
    print('--------------------------------------------------------------------------------------')
    print('Date | Quasar Coordinates (hms.ss ◦ ′ ′′) | RA (◦) | Dec (◦) | Airmass')
    for line in table:
        print(line[0] + ' | ' + line[1] + ' | ' + str(round(line[2],5)) + ' | ' + str(round(line[3],5)) + ' | ' + str(round(line[4],5)))
    print('--------------------------------------------------------------------------------------')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    prog='HW1',
    description='Prints table of the best Quasars to observe from Kitt Peak for every day of a desired month',
    epilog='Please input a number between 1-12 after the program name to run. 1 = January, 2 = February.... 12 = December. Example: to find best oboservations for April, type \"python HW1.py -m 4\"'
    parser.add_argument('-m', '--month', type=int, help='single int between 1-12 representing the 12 months')
    
    try:
        month = parser.parse_args().month
        assert(month <=12)
    except:
        print("WARNING: No month given or month out of acceptable range")
        print("Defaulting to month=1 (January)")
        month = 1
    
    try:
        fname = os.path.join(os.getenv("ASTR5160"), "week4", "HW1quasarfile.txt")
        coordlist = np.genfromtxt(fname,dtype=str)
    except TypeError:
        print("Set the ASTR5160 environment variable to point to the main " +
        "ASTR5160 data directory")
        raise OSError
    print('Generating Table...')        
    table = MakeTable(month,coordlist)
    PrintTable(table)
    


