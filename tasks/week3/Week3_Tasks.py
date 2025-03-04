import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
from astropy.time import Time
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as clrs
import os

def RA_Dec2Cart(RA,Dec):
    coord = SkyCoord(RA,Dec,unit=('hourangle','deg'))
    coord.representation_type = 'cartesian'
    return coord
    
def RA_Dec2Cart2(RA,Dec):
    coord = SkyCoord(RA,Dec,unit=('hourangle','deg'))
    x = np.cos(coord.ra.rad)*np.cos(coord.dec.rad)
    y = np.sin(coord.ra.rad)*np.cos(coord.dec.rad)
    z = np.sin(coord.dec.rad)
    
    return x,y,z
    
def Gal2RA_Dec(l,b):
    galcoord = SkyCoord(l,b,unit='deg',frame='galactic')
    return galcoord.icrs
    
def Trace_Zenith(loc,offset=7):
    lon,lat = loc
    wirocoord = EarthLocation(lon,lat,height=2943) #Get coords for WIRO
    
    starttime = Time('2025-01-01')
    times = np.arange(0,365,1)*u.day+starttime+offset*u.hr   #List of times for midnight at WIRO at every day of the year
    
    l_bs = []
    
    for time in times:
        zenith_coord = SkyCoord(alt=90,az=0,unit='deg',frame=AltAz(obstime=time,location=wirocoord))   #get coord for the zenith at the locaiton and time
        #l_bs.append(zenith_coord.galactic)
        galcoord = zenith_coord.galactic   #convert to galactic coordinates
        l_bs.append((galcoord.l.deg,galcoord.b.deg))
    
    l_bs = np.array(l_bs,dtype=[('l','f8'),('b','f8')])
    
    color_grad = np.linspace(0,1,len(times))
    colormap = cm.get_cmap('hsv')
    colors = colormap(color_grad)
    norm = clrs.Normalize(vmin=0, vmax=365)
    
    plt.figure('l,b over the year')
    plt.scatter(l_bs['l'],l_bs['b'],marker='o',color=colors)
    plt.grid()
    plt.xlabel(r'Galactic longitude $\ell$ (deg)',fontsize=14)
    plt.ylabel(r'Galactic latitude $b$ (deg)',fontsize=14)
    plt.xlim(0,360)
    plt.ylim(-90,90)
    plt.colorbar(plt.cm.ScalarMappable(cmap=colormap,norm=norm), label="Days through year")
    plt.title('Zenith Galactic Coords Through the Year at \n'+str(wirocoord.lon) + '  ' + str(wirocoord.lat) + '\n and local midnight (UTC 7:00)')
    
def Compare_Quasars(qc1, qc2, mags1, mags2, sfd):

    g1,r1,i1 = mags1
    g2,r2,i2 = mags2
    
    gmr1 = g1-r1
    gmr2 = g2-r2
    rmi1 = r1-i1
    rmi2 = r2-i2
    
    gri = np.array([3.303 , 2.285 , 1.698])
    ebv1 = sfd(qc1)
    ebv2 = sfd(qc2)
    
    Ag1,Ar1,Ai1 = ebv1*gri
    Ag2,Ar2,Ai2 = ebv2*gri
    
    g1corr = g1-Ag1
    r1corr = r1-Ar1
    i1corr = i1-Ai1
    
    g2corr = g2-Ag2
    r2corr = r2-Ar2
    i2corr = i2-Ai2
    
    gmr1corr = g1corr-r1corr
    gmr2corr = g2corr-r2corr
    rmi1corr = r1corr-i1corr
    rmi2corr = r2corr-i2corr

    plt.figure('Quasar Comparison')
    plt.plot(rmi1,gmr1,'bo',markersize = 15, label = 'Quasar 1 Raw')
    plt.plot(rmi2,gmr2,'go',markersize = 15, label = 'Quasar 2 Raw')
    plt.plot(rmi1corr,gmr1corr,'b*',markersize = 15,label='Quasar 1 Corrected')
    plt.plot(rmi2corr,gmr2corr,'g*',markersize = 15,label='Quasar 2 Corrected')
    #plt.scatter([rmi1,rmi2],[gmr1,gmr2],marker = 'o', color = ['b','g'],label=['Quasar 1 Raw', 'Quasar 2 Raw'])
    #plt.scatter([rmi1corr,rmi2corr],[gmr1corr,gmr2corr],marker = '*', color = ['b','g'],label=['Quasar 1 Corrected', 'Quasar 2 Corrected'])
    plt.grid()
    plt.ylabel('g - r',fontsize=14)
    plt.xlabel('r - i',fontsize=14)
    plt.xlim(-0.15,0.15)
    plt.ylim(-.05,.4)
    plt.legend(fontsize=12)
    
def Map_Dust(cenra,cendec,sfd, dra=0.1,ddec=0.1):
    RAs = np.linspace(cenra - 50*dra, cenra + 50*dra, 100)
    Decs = np.linspace(cendec - 50*ddec, cendec + 50*ddec, 100) 
    RAgrid, Decgrid = np.meshgrid(RAs, Decs)
    
    skygrid = SkyCoord(ra=RAgrid, dec=Decgrid, unit=(u.deg, u.deg))
 
    dustgrid = sfd(skygrid)
    
    plt.figure(str(cenra)+' '+str(cendec))
    plt.imshow(dustgrid, origin='lower', extent=[RAs.min(), RAs.max(), Decs.min(), Decs.max()])
    plt.xticks(np.linspace(RAs.min(), RAs.max(), 5))
    plt.yticks(np.linspace(Decs.min(), Decs.max(), 5))
    plt.xlabel('RA (deg)',fontsize=14)
    plt.ylabel('Dec (deg)', fontsize=14)
    plt.gca().invert_xaxis()
    plt.title('Dust around '+str(cenra) + ' ' + str(cendec))
    
if __name__ == "__main__":

    import warnings
    warnings.filterwarnings("ignore") #supress wanings in output
    
    #---------------------------- Lesson 5
    
    test_RA = "06:45:08.91728"
    test_Dec ="-16:42:58.0171"
    cartcoord = RA_Dec2Cart(test_RA,test_Dec)
    print('Coordinate Transform with Astropy')
    print(test_RA + " " + test_Dec + f" ---> ({cartcoord.x.value:4f}, {cartcoord.y.value:4f}, {cartcoord.z.value:4f})")
    
    cartcoords2 = RA_Dec2Cart2(test_RA,test_Dec)
    print('Coordinate Transform with Astropy')
    print(test_RA + " " + test_Dec + f" ---> ({cartcoords2[0]:4f}, {cartcoords2[1]:4f}, {cartcoords2[2]:4f})")
    
    gal_cen = Gal2RA_Dec(0.0,0.0)
    print("Galactic Center at " + f"{gal_cen.ra.deg:.4f} {gal_cen.dec.deg:.4f}")
    #This is in Sagittarius
    
    wiro_lon = -105.977
    wiro_lat = 41.097
    Trace_Zenith([wiro_lon,wiro_lat])
    
    #--------------------------- Lesson 6
    
    from dustmaps.config import config
    from dustmaps.sfd import SFDQuery
    
    try:
        dustdir = os.path.join(os.getenv("ASTR5160"), "data", "dust", "v0_1", "maps")    #env "ASTR5160" is a path variable set in my .bashrc with the value "/d/scratch/ASTR5160/"
    except TypeError:
        print("Set the ASTR5160 environment variable to point to the main " +
        "ASTR5160 data directory")
    #dustdir = "/d/scratch/ASTR5160/data/dust/v0_1/maps"
    config["data_dir"] = dustdir
    sfd = SFDQuery()
    qc1 = SkyCoord(246.933, 40.795, unit='deg')
    qc2 = SkyCoord(236.562, 2.440, unit='deg')

    g1=18.81
    r1=18.74
    i1=18.81
    g2=19.10
    r2=18.79
    i2=18.72
    #Data copied from online SDSS Navigator Tool
    
    mags1 = [g1,r1,i1]
    mags2 = [g2,r2,i2]
    
    Compare_Quasars(qc1,qc2,mags1,mags2,sfd)
    #I'm not sure if quasars at a similar redshift are supposed to have the same color, but these two don't have similar colors before correction.
    #After extinction/reddening correction, the colors are definitely closer.
    
    cenra1 = 236.6
    cendec1 =2.4
    Map_Dust(cenra1,cendec1,sfd)
    
    cenra2 = 246.9
    cendec2 = 40.795
    dra2 = 0.13
    ddec2 = 0.1
    Map_Dust(cenra2,cendec2,sfd)
    
    
    
    plt.show()
