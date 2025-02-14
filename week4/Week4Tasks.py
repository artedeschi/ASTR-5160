import numpy as np
from numpy.random import random
import astropy.units as u
from astropy.coordinates import SkyCoord, EarthLocation, AltAz, search_around_sky
from astropy.time import Time
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as clrs
import os

def GenerateData(num = 10000):
    RAs = 2*np.pi*(random(10000)-0.5)
    Decs = np.arcsin(1.-random(10000)*2.)
    
    return RAs,Decs
    
def PlotData(dat, proj = None):
    RAs, Decs = dat
    fig = plt.figure(proj)
    ax = fig.add_subplot(111,projection=proj)  #Default cartesian if proj is None
    ax.scatter(RAs,Decs,s=0.25,c='y')
    if proj == None:
        xlab = [r'-180$^\circ$',r'120$^\circ$',r'-60$^\circ$',r'0$^\circ$',r'60$^\circ$',r'120$^\circ$',r'180$^\circ$']
        xticks = np.linspace(-np.pi,np.pi,7)
        ylab = [r'-90$^\circ$',r'-60$^\circ$',r'-30$^\circ$',r'0$^\circ$',r'30$^\circ$',r'60$^\circ$',r'90$^\circ$']
        yticks = np.linspace(-np.pi/2,np.pi/2,7)
    else:
        xlab = ['14h','16h','18h','20h','22h','0h','2h','4h','6h','8h','10h']
        xticks = np.linspace(-np.pi,np.pi,11)
        ylab = [r'-90$^\circ$',r'-75$^\circ$',r'-60$^\circ$',r'-45$^\circ$',r'-30$^\circ$',r'-15$^\circ$',r'0$^\circ$',r'15$^\circ$',r'30$^\circ$',r'45$^\circ$',r'60$^\circ$',r'75$^\circ$',r'90$^\circ$']
        yticks = np.linspace(-np.pi/2,np.pi/2,13)
    
    
    
    if proj == None:
        ax.set_xticks(xticks,labels=xlab)
        ax.set_yticks(yticks,labels=ylab)
        ax.set_xlabel('RA (deg)',fontsize=14)
        ax.set_ylabel('Dec (deg)',fontsize=14)
        ax.set_xlim(-np.pi,np.pi)
        ax.set_ylim(-np.pi/2,np.pi/2)
        plt.grid(color='k',linewidth=1)
        
    elif proj == 'aitoff':

        ax.set_xticks(xticks,labels=xlab, weight=800)
        ax.set_yticks(yticks,labels=ylab)
        ax.set_xlabel('RA (Hour)',fontsize=14,labelpad=20)
        ax.set_ylabel('Dec (deg)',fontsize=14)
        plt.grid(color='b', linestyle='--',linewidth=2)
        
    else:  #basically just lambert
        ax.set_xticks(xticks,labels=xlab, weight=800)
        ax.set_yticks(yticks,labels=[' ']*len(yticks))
        plt.grid(color='b', linestyle='--',linewidth=2)
        
    ax.invert_xaxis()
        
def FindAngle(coord1,coord2):
     x1,y1,z1 = coord1.x,coord1.y,coord1.z
     x2,y2,z2 = coord2.x,coord2.y,coord2.z
     
     dotprod = x1*x2+y1*y2+z1*z2
     
     abs1 = np.sqrt(x1**2+y1**2+z1**2)
     abs2 = np.sqrt(x2**2+y2**2+z2**2)
     
     cosz = dotprod/(abs1*abs2)
     
     z = np.arccos(cosz)
     
     return z
     
def PopulateSky(n=100,RAmin=2,RAmax=3,Decmin=-2,Decmax=2):

    RAs1 = np.random.uniform(RAmin,RAmax,n)
    Decs1 = np.random.uniform(Decmin,Decmax,n)
    RAs2 = np.random.uniform(RAmin,RAmax,n)
    Decs2 = np.random.uniform(Decmin,Decmax,n)
    
    pop1 = SkyCoord(RAs1,Decs1,unit=(u.hourangle,u.deg))
    pop2 = SkyCoord(RAs2,Decs2,unit=(u.hourangle,u.deg))
    
    return pop1, pop2
    
def PlotData2(ax,data,label,firstplot = False, color='b',marker='o',size=20):
    RAs = data.ra
    Decs = data.dec
    ax.scatter(RAs,Decs,c=color,marker=marker,s=size,label=label)
    if firstplot:
        ax.invert_xaxis()
        ax.set_xlabel('RA (deg)')
        ax.set_ylabel('Dec (deg)')
        ax.grid()
    ax.legend(fontsize=14,bbox_to_anchor=(1, 1))
    
def FindCloseCoords(pop1,pop2):
    ind1,ind2,angs,dists = search_around_sky(pop1,pop2,10*u.arcmin)
    closepop1 = pop1[ind1]
    closepop2 = pop2[ind2]
    return closepop1, closepop2
    
    
if __name__ == "__main__":
    #Lesson 7
    data = GenerateData()
    PlotData(data)
    
    #Looks like there's more points near the equator
    
    PlotData(data,proj='aitoff')
    PlotData(data,proj='lambert')
    
    coord1 = SkyCoord(263.75,-17.9,unit='deg')
    coord2 = SkyCoord('20h24m59.9s +10d06m00.0s')
    cartcoord1 = coord1.cartesian
    cartcoord2 = coord2.cartesian
    z = FindAngle(cartcoord1,cartcoord2).to(u.deg)
    print('Using manual calculation:')
    print(f'Zenith Angle = {z:.4}')
    print('Using SkyCoord.separation():')
    z2 = coord1.separation(coord2)
    print(f'Zenith Angle = {z2:.4}')
    
    #----------------------------------------------
    #Lesson 8
    
    pop1,pop2 = PopulateSky()
    
    fig = plt.figure('2 populations')
    ax = fig.add_subplot(111)
    PlotData2(ax,pop1,'Pop 1',firstplot=True)
    PlotData2(ax,pop2,'Pop 2',color='g',marker='*')
    
    closepop1,closepop2 = FindCloseCoords(pop1,pop2)
    PlotData2(ax,closepop1, 'Close Pop1', color='r', marker = 'o',size=40)
    PlotData2(ax,closepop2, 'Close Pop2', color='r', marker = '*',size=40)
    fig.tight_layout()
    
    plt.show()
