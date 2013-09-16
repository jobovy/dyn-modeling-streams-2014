import os, os.path
import pickle
import numpy
from galpy import potential
from galpy.orbit import Orbit
from galpy.util import bovy_coords, save_pickles
from galpy.df_src.streamdf import streamdf
_REFV0= 230.
_REFR0= 16.
_DATADIR=os.path.join(os.getenv('DATADIR'),
                      'bovy',
                      'gaia-challenge')
_DATAFILE= os.path.join(_DATADIR,'pal5.all_particles.txt')
def readPal5Data(R0=16.,V0=230.,subsample=None,leading=False,trailing=False):
    """
    NAME:
       readPal5Data
    PURPOSE:
       Read the Pal 5 data
    INPUT:
       R0= (default 16kpc) value of R0 to apply when normalizing
       V0= (default 230kpc) value of V0 to apply when normalizing
       subsample= (None) if set, subsample to this # of stars
       leading= (False) if True, only select stars in the leading arm
       trailing= (False) if True, only select stars in the trailing arm
    OUTPUT:
       R,vR,vT,z,vz,phi
    HISTORY:
       2013-09-16 - Written - Bovy (IAS)
    """
    data= numpy.loadtxt(_DATAFILE,comments='#')
    #Cut out stars near/in the progenitor
    px,py,pz,pvx,pvy,pvz= progenitorOrbit(R0=1.,V0=1.,xyz=True)
    pindx= ((px-data[:,0]/1000.)**2.+(py-data[:,1]/1000.)**2.+(pz-data[:,2]/1000.)**2.
            +(pvx-data[:,3])**2.+(pvy-data[:,4])**2.+(pvz-data[:,5])**2.) < 500.
    data= data[True-pindx,:]
    if leading:
        data= data[data[:,5] < pvz]
    if trailing:
        data= data[data[:,5] > pvz]
    if not subsample is None:
        indx= numpy.random.permutation(subsample)
        data= data[indx,:]
    R,phi,z= bovy_coords.rect_to_cyl(data[:,0]/1000.,
                                     data[:,1]/1000.,
                                     data[:,2]/1000.)
    vR, vT, vz= bovy_coords.rect_to_cyl_vec(data[:,3],
                                            data[:,4],
                                            data[:,5],
                                            R,phi,z,cyl=True)
    R/= R0
    z/= R0
    vR/= V0
    vT/= V0
    vz/= V0
    return (R,vR,vT,z,vz,phi)

def progenitorOrbit(R0=16.,V0=230.,xyz=False):
    """
    NAME:
       progenitorOrbit
    PURPOSE:
       return the phase-space point corresponding to the (given) progenitor
    INPUT:
       R0= (default 16kpc) value of R0 to apply when normalizing
       V0= (default 230kpc) value of V0 to apply when normalizing
       xyz= (False) if True, return rectangular coordinates
    OUTPUT:
       Orbit instance with progenitor orbit
    HISTORY:
       2013-09-16 - Written - Bovy (IAS)
    """
    x= 7816.082584/1000.
    y= 240.023507/1000.
    z= 16640.055966/1000.
    vx= -37.456858
    vy=-151.794112
    vz=-21.609662
    if xyz:
        return (x,y,z,vx,vy,vz)
    R,phi,z= bovy_coords.rect_to_cyl(x,y,z)
    vR,vT,vz= bovy_coords.rect_to_cyl_vec(vx,vy,vz,R,phi,z,cyl=True)
    R/= R0
    z/= R0
    vR/= V0
    vT/= V0
    vz/= V0
    return Orbit([R,vR,vT,z,vz,phi])

def fitDirect():
    savefile='pal5GC_100.sav'
    #Load data
    savefile= open(savefile,'rb')
    data= pickle.load(savefile)
    sR,svR,svT,sz,svz,sphi= data
    savefile.close()
    #Setup potential and DF grids
    vos= numpy.linspace(160.,280.,21)/_REFV0
    qs= numpy.linspace(0.6,1.4,21)
    sigvs= numpy.exp(numpy.linspace(numpy.log(1./_REFV0),
                                    numpy.log(1.),21))
    sigxs= numpy.exp(numpy.linspace(numpy.log(0.1/_REFR0),
                                    numpy.log(1.),21))
    #Load progenitor orbit
    progo= progenitorOrbit(V0=_REFV0,R0=_REFR0)
    ts= numpy.linspace(0.,300.,10000)
    outfilename= 'pal5GC_100_direct.sav'
    if not os.path.exists(outfilename):
        out= numpy.zeros((len(vos),len(qs),len(sigvs),len(sigxs)))
        ii,jj= 12,0
    else:
        outfile= open(outfilename,'rb')
        out= pickle.load(outfile)
        ii= pickle.load(outfile)
        jj= pickle.load(outfile)
        outfile.close()
    while ii < len(vos):
        progo= progenitorOrbit(V0=_REFV0*vos[ii],R0=_REFR0)
        while jj < len(qs):
            print ii, jj
            #Setup some df
            if hasattr(progo,'orbit'): delattr(progo,'orbit')
            sdf= streamdf(sigvs[0],sigxs[0],
                          pot=potential.LogarithmicHaloPotential(normalize=1.,
                                                                 q=qs[jj]),
                          progenitor=progo,ts=ts)
            sX,sY,sZ,svX,svY,svZ= sdf.prepData4Direct(sR,svR/vos[ii],svT/vos[ii],
                                                      sz,svz/vos[ii],sphi)
            for kk in range(len(sigvs)):
                for ll in range(len(sigxs)):
                    sdf= streamdf(sigvs[kk],sigxs[ll],
                                  pot=potential.LogarithmicHaloPotential(normalize=1.,
                                                                         q=qs[jj]),
                                  progenitor=progo,ts=ts)
                    out[ii,jj,kk,ll]= sdf(sX,sY,sZ,svX,svY,svZ,
                                          rect=True,log=True)
            jj+= 1
            if jj == len(qs):
                jj= 0
                ii+= 1
            save_pickles(outfilename,out,ii,jj)
    save_pickles(outfilename,out,ii,jj)
    return None
