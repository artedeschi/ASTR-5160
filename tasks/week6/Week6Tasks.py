import numpy as np
from numpy.random import random
import astropy.units as u
from astropy.coordinates import SkyCoord
import pymangle
import matplotlib.pyplot as plt

#from week 5 tasks
def CoordCap(RA,Dec,theta):
    
    h = 1.-np.cos(theta).value
    norm_coord = SkyCoord(RA,Dec)
    xyz = np.array(norm_coord.cartesian.xyz.value,dtype='f2')
    four_vec = np.append(xyz,h)
    return four_vec
    
def Gen_ply_file(Npoly, caps, fname, Rev = False, sangle = None):
    #caps must have shape Ncaps x Npoly
    #Ncaps is numper of caps per polygon
    #sangle must have length Npoly
    print('Generating '+fname + '...')  
    plyfile = open(fname,'w')
    if Npoly ==1:
        if Rev:
            caps[0][-1] = -caps[0][-1]   #Flip sign of last part of the cap
        plyfile.write('1 polygons \n')
        if sangle is not None:
            plyfile.write(f'polygon 1 ( {len(caps)} caps, 1 weight, 0 pixel, {sangle:.4f} str):\n')
        else:
            plyfile.write(f'polygon 1 ( {len(caps)} caps, 1 weight, 0 pixel, 0 str):\n')
        for cap in caps:
            plyfile.write(str(cap).strip('[ ]')+'\n')
    else:
        plyfile.write(f'{len(caps)} polygons\n')
        for m in range(len(caps)):
            if sangle is not None:
                plyfile.write(f'polygon {m+1} ( {len(caps[m])} caps, 1 weight, 0 pixel, {sangle[m]:.4f} str):\n')
            else:
                plyfile.write(f'polygon {m+1} ( {len(caps[m])} caps, 1 weight, 0 pixel, 0 str):\n')
            for n in range(len(caps[m])):
                plyfile.write(str(caps[m][n]).strip('[ ]')+'\n')
    plyfile.close()
    
    
def PlotMaskPoints(ax,mask,N,label,color='red'):
    RAs, Decs = mask.genrand(N)
    ax.scatter(RAs,Decs,c=color,s=1,alpha = 0.3,label = label)
    ax.set_xlabel('RA (deg)')
    ax.set_ylabel('Dec (deg)')
    ax.legend(markerscale=5,loc='upper left', bbox_to_anchor=(1, 1))
    #ax.tight_layout()    
    
#From week 5 tasks    
def RA_Cap(RA):
    Dec = 0*u.deg
    h = 1.
    norm_coord = SkyCoord(RA+90*u.deg,Dec)
    xyz = np.array(norm_coord.cartesian.xyz.value,dtype='f2')
    four_vec = np.append(xyz,h)
    return four_vec

#from week 5 tasks
def Dec_Cap(Dec):
    RA = 0*u.deg
    point_dec = 90*u.deg
    norm_coord = SkyCoord(RA,point_dec)
    xyz = np.array(norm_coord.cartesian.xyz.value,dtype='f2')
    h = 1.-np.sin(Dec).value
    four_vec = np.append(xyz,h)
    return four_vec
    
def CalcArea(RA1,RA2,Dec1,Dec2):
    S = (RA2.to(u.rad)-RA1.to(u.rad))*(np.sin(Dec2)-np.sin(Dec1))
    return S.value


#From Week 4 Tasks    
def GenerateData(num = 10000):
    RAs = 2*np.pi*(random(10000)-0.5)*u.rad
    Decs = np.arcsin(1.-random(10000)*2.)*u.rad
    
    return RAs,Decs
    
if __name__ == "__main__":
    
    #Lesson 11
    
    cap1 = CoordCap(76*u.deg,36*u.deg,5*u.deg)
    cap2 = CoordCap(75*u.deg,35*u.deg,5*u.deg)
    
    #print(cap1)
    #print(cap2)
    
    Gen_ply_file(1,[cap1,cap2],'intersection.ply')
    Gen_ply_file(2,[[cap1],[cap2]],'bothcaps.ply')
    
    minter = pymangle.Mangle('intersection.ply')
    mboth = pymangle.Mangle('bothcaps.ply')
    fig,ax1 = plt.subplots(figsize=(10,6))
    PlotMaskPoints(ax1,minter,10000,'Intersection of 2 Polygons')
    PlotMaskPoints(ax1,mboth,10000,'2 Caps in 1 Polygon',color='blue')
    ax1.grid()
    ax1.invert_xaxis()
    fig.tight_layout()
    
    # It is clear from this plot that the intersection (1 polygon mask) acts like an "AND" logic operator,
    # while the bothcaps (2 polygon mask) acts like an "AND-OR" logic operator
    # So each polygon is a mask of what is included in ALL caps for that polygon,
    # But if each cap has their own polygon, then the total mask is what every cap includes.
    
    Gen_ply_file(1,[cap1,cap2],'flip1intersection.ply',Rev=True)
    mflip1 = pymangle.Mangle('flip1intersection.ply')
    fig2,ax2 = plt.subplots(figsize=(10,6))
    PlotMaskPoints(ax2,minter,10000,'Intersection of 2 Polygons')
    PlotMaskPoints(ax2,mflip1,10000,'Cap1 Flipped',color='green')
    ax2.grid()
    ax2.invert_xaxis()
    fig2.tight_layout()
    
    
    #-----------------------------------------------------------------------------
    #Lesson 12
    RA1 = 5*u.hourangle
    RA2 = 6*u.hourangle
    Dec1 = 30*u.deg
    Dec2 = 40*u.deg
    rectcap1 = RA_Cap(RA1)
    rectcap2 = RA_Cap(RA2)
    rectcap3 = Dec_Cap(Dec1)
    rectcap4 = Dec_Cap(Dec2)
    
    area1 = CalcArea(RA1,RA2,Dec1,Dec2)
    caps1 = [rectcap1,rectcap2,rectcap3,rectcap4]
    
    RA3 = 11*u.hourangle
    RA4 = 12*u.hourangle
    Dec3 = 60*u.deg
    Dec4 = 70*u.deg
    rect2cap1 = RA_Cap(RA3)
    rect2cap2 = RA_Cap(RA4)
    rect2cap3 = Dec_Cap(Dec3)
    rect2cap4 = Dec_Cap(Dec4)
    
    area2 = CalcArea(RA3,RA4,Dec3,Dec4)
    caps2 = [rect2cap1,rect2cap2,rect2cap3,rect2cap4]
    
    Gen_ply_file(2, [caps1,caps2], 'rectpoly.ply', Rev = False, sangle = [area1,area2])
    rectmask = pymangle.Mangle('rectpoly.ply')
    
    RAs, Decs = GenerateData(num=1e6)
    coords = SkyCoord(RAs,Decs,unit=u.rad)
    goodcoords = rectmask.contains(coords.ra.deg,coords.dec.deg)
    
    plt.figure(figsize=(10,6))
    plt.scatter(coords.ra.deg,coords.dec.deg,c='r',s=1,alpha = 0.3,label='all ponits')
    plt.scatter(coords.ra.deg[goodcoords],coords.dec.deg[goodcoords],s=1,alpha = 0.3,c='b',label='points in masks')
    plt.xlabel('RAs (deg)')
    plt.ylabel('Decs (deg)')
    plt.grid()
    plt.gca().invert_xaxis()
    plt.legend(markerscale=5,loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    
    plt.show()
