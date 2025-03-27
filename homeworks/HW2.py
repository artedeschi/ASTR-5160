"""
Code by Adam Tedeschi
For ASTR5160 at UWyo 2025
HW2.py
"""
import numpy as np
from numpy.random import random
import astropy.units as u
from astropy.coordinates import SkyCoord
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import argparse
import os

def CalcRectArea(Rect):
    """
    NAME: CalcRectArea
 
    PURPOSE: function that takes in a 'Rect' array and returns the area of
    that defined rectangle on a sphere


    INPUTS:
    
      Rect - array consisting of four floats. These floats coorespond
      to the 4 bounds in degrees that make a rectangle on a sphere.
      These bounds are (in order) RAmin, RAmax, Decmin, and Decmax.
      The range of these bounds are: 
      -180 <= RA <= 180
      -90 <= Dec <= 90
      RAmin and Decmin must be less than RAmax and Decmax respectively.
      The whole "Rect" array should have an astropy quantity applied to it
      such that it has the units of an angle (deg, rad, etc.)

    OUTPUTS: 
    
    area -  Astropy quantity of the calculated area in degrees squared
    
    COMMENTS: This is adapted from a class lesson by Adam Myers
    """
    RAmin,RAmax,Decmin,Decmax = Rect
    
    S = (RAmax.to(u.rad)-RAmin.to(u.rad))*(np.sin(Decmax)-np.sin(Decmin))*u.rad
    area = S.to(u.deg**2)
    return area
    
def GetCenterpoint(Rect):
    """
    NAME: GetCenterpoint
 
    PURPOSE: function that takes in a 'Rect' array and returns the centerpoint
    of that rectangle on a spherical surface


    INPUTS:
    
      Rect - array consisting of four floats. These floats coorespond
      to the 4 bounds in degrees that make a rectangle on a sphere.
      These bounds are (in order) RAmin, RAmax, Decmin, and Decmax.
      The range of these bounds are: 
      -180 <= RA <= 180
      -90 <= Dec <= 90
      RAmin and Decmin must be less than RAmax and Decmax respectively.
      The whole "Rect" array should have an astropy quantity applied to it
      such that it has the units of an angle (deg, rad, etc.)

    OUTPUTS: 
    
    RAcen -  central RA value of the Rect in radians
    Deccen - central Dec value of the Rect in radians
    
    COMMENTS: This function is meant to be used in PlotBounds() in order
    to position text and arrows easily on when
    """
    RAmin,RAmax,Decmin,Decmax = Rect
    RAcen = ((RAmax+RAmin)/2).to(u.rad).value
    Deccen = ((Decmax+Decmin)/2).to(u.rad).value
    
    return RAcen,Deccen
    
def PlotBounds(Rects, figname = 'Rect Bounds',points=None):
    """
    NAME: PlotBounds
 
    PURPOSE: function that takes in an array of 'Rect' arrays and
    plots those rectangles on an Aitoff spherical projection. Additional 
    options to rename figure and plots a scatter of points on top.


    INPUTS:
    
      Rects - 2D array of 'n' 'Rect' arrays. Each 'Rect' object is an 
      array consisting of four floats. These floats coorespond
      to the 4 bounds in degrees that make a rectangle on a sphere.
      These bounds are (in order) RAmin, RAmax, Decmin, and Decmax.
      The range of these bounds are: 
      -180 <= RA <= 180
      -90 <= Dec <= 90
      RAmin and Decmin must be less than RAmax and Decmax respectively.
      Here, the 'Rect' array should be a normal 1D array of floats
      
      figname - OPTIONAL string to create new fig objects given multiple
      calls of this function.
      
      points - OPTIONAL 2xN array of floats representing N random points
      to plot onto the figure. points[0] are the RAs of the points in radians, 
      and points[1] are the Decs of the points in radians.

    OUTPUTS: 
    
      areas -  1D array of 'n' astropy quantities of the calculated areas
      of each 'Rect' in degrees squared.
    
    COMMENTS: To overplot onto the same fig object, use the same figname in each call.
    'Rects' can have a length of up to 9 in the currect state of the function
    """
    
    areas = []
    #ART Setup figure
    fig = plt.figure(figname)
    ax = fig.add_subplot(111,projection='aitoff')
    #ART give each rectangle a different color. We only need 4 for the assignment,
    #but I made it able to take up to 9
    colors = ['r','g','b','y','c','m','orange','lime','gray']
    #ART Loop through each rectangle to plot and get area
    for n in range(len(Rects)):
        RAmin,RAmax,Decmin,Decmax = np.deg2rad(Rects[n]) 
        area = CalcRectArea(Rects[n]*u.deg)
        RAcen,Deccen = GetCenterpoint(Rects[n]*u.deg)
        
        #ART Plotting rectangle using vlines and hlines
        ax.vlines([RAmin,RAmax],Decmin,Decmax,linestyle='-',color=colors[n])
        ax.hlines([Decmin,Decmax],RAmin,RAmax,linestyle='-',color=colors[n])
        #ART creating extra labels and arrows to denote areas of each rectangle
        arrow = dict(arrowstyle='-') 
        ax.annotate('Area = '+str(round(area.value,4))+r' deg$^2$',(RAcen,Deccen),xytext=(RAcen-180,Deccen+40),textcoords='offset points',arrowprops=arrow)
        areas.append(area)
    #ART Plot random points if given
    if points is not None:
        ax.scatter(points[0],points[1], c='b',s=1,alpha=.3)
    #ART lines at RA=0 and Dec=0 boldened to make plot easier to read
    ax.hlines([0],-np.pi,np.pi,color='k',linestyle='-',linewidth=0.5)
    ax.vlines([0],-np.pi/2,np.pi/2,color='k',linestyle='-',linewidth=0.5)
    ax.grid()
    return areas


def GenPoints(num = 10000):
    """
    NAME: GenPoints
 
    PURPOSE: function that generates 'num' amount of random and uniformly
    distributed points on a sphere in radians


    INPUTS:
    
      num - OPTIONAL  This is the number of points to generate | default = 10,000

    OUTPUTS: 
    
      RAs - 1D array of floats representing the Right Ascensions of the generated 
      points in radians with a length = num
      Decs- 1D array of floats representing the Declinations of the generated points 
      in radians with a length = num
    
    COMMENTS: This is adapted from a class lesson by Adam Myers
    """
    RAs = 2*np.pi*(random(num)-0.5)
    Decs = np.arcsin(1.-random(num)*2.)
    
    return RAs,Decs
    
    
def PointsinRect(points,Rect):
    """
    NAME: Points in Rect
 
    PURPOSE: function that gives the number of coordinates in 'points' that lie within the 
    coordinate bounds of the given 'Rect' object


    INPUTS:
    
      points - OPTIONAL 2xN array of floats representing N random points
      to plot onto the figure. points[0] are the RAs of the points in radians, 
      and points[1] are the Decs of the points in radians.
      
      Rect - Rect - array consisting of four floats. These floats coorespond
      to the 4 bounds in degrees that make a rectangle on a sphere.
      These bounds are (in order) RAmin, RAmax, Decmin, and Decmax.
      The range of these bounds are: 
      -180 <= RA <= 180
      -90 <= Dec <= 90
      RAmin and Decmin must be less than RAmax and Decmax respectively.
      Here, the 'Rect' array should be a normal 1D array of floats

    OUTPUTS: 
    
      numP - int representing number of coordinates in 'points' that lie within 'Rect'
    
    COMMENTS: This is adapted from a class lesson by Adam Myers
    """
    RAmin,RAmax,Decmin,Decmax = np.deg2rad(Rect)
    numP = np.sum((points[0] > RAmin) & (points[0] < RAmax) & (points[1] > Decmin) & (points[1] < Decmax))
    return numP


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Generate and save plot of 4 rectangular fields and save them in the specified directory.")
    
    #ART commandline argument to save plot to specified directory. Default is current directory
    parser.add_argument("plot_dir", nargs="?", default='./', help="Directory to save plots (default: './')")

    args = parser.parse_args()
    plot_dir = args.plot_dir
    
    #vertices defining a half-sphere (2pi rad or 20626 deg^2)
    testRAmin = 0
    testRAmax = 360
    testDecmin = 0
    testDecmax = 90
    #ART CalcRectArea takes in a "rect" in units of degrees
    testarea = CalcRectArea([testRAmin,testRAmax,testDecmin,testDecmax]*u.deg)
    
    print(testarea)
    #ART Is indeed correct area
    
    #ART defining borders for the 4 rectangles
    RAmin = 40  #Same for all 4
    RAmax = 80   #Same for all 4
    Decmin1 = -80
    Decmax1 = -30
    Decmin2 = -30
    Decmax2 = 10
    Decmin3 = -10
    Decmax3 = 50
    Decmin4 = 50
    Decmax4 = 85
    
    #ART Putting all rectangles into one array.
    Rects = [[RAmin,RAmax,Decmin1,Decmax1],[RAmin,RAmax,Decmin2,Decmax2],[RAmin,RAmax,Decmin3,Decmax3],[RAmin,RAmax,Decmin4,Decmax4]]
    
    #area = CalcRectArea(RAmin,RAmax,Decmin,Decmax)
    
    PlotBounds(Rects)
    
    plt.savefig(os.path.join(plot_dir, "HW2.png"))
    
    #Part 2. -----------------------------------------------------------------------
    
    N=100000
    Rect = [-105,-65,-20,60]
    points = GenPoints(num=N)
    #PlotBounds([Rect],figname='Rect with Points',points=points)  #ART Not necessary, but was helpful for debugging
    numP = PointsinRect(points,Rect)
    area2 = CalcRectArea(Rect*u.deg)
    #ART numP/N = area2/(4pi sr)
    print(f'Ratio of Points = {numP/N:.4f}')
    print(f'Ratio of Areas = {(area2.to(u.sr)/(4*np.pi*u.sr)).value:.4f}')
        
    plt.show()
   
    
    
