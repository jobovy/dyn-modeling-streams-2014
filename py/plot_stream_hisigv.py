import sys
import os, os.path
import copy
import numpy
from scipy import optimize, special
from galpy.util import bovy_plot, bovy_coords, bovy_conversion
from galpy import potential
from galpy.orbit import Orbit
from galpy.actionAngle_src.actionAngleIsochroneApprox\
    import actionAngleIsochroneApprox
from galpy.df_src.streamdf import streamdf
from matplotlib import pyplot
from matplotlib.ticker import NullFormatter
_STREAMSNAPDIR= '../sim/snaps-hisigv'
_STREAMSNAPAADIR= '../sim/snaps_aai'
_NTRACKCHUNKS= 64
_SIGV=0.365*10. #x 10 because stream is 10x hotter
def plot_stream_xz(plotfilename):
    #Read stream
    data= numpy.loadtxt(os.path.join(_STREAMSNAPDIR,'gd1-hisigv_evol_00041.dat'),
                        delimiter=',')
    includeorbit= True
    if includeorbit:
        npts= 201
        pot= potential.LogarithmicHaloPotential(normalize=1.,q=0.9)
        pts= numpy.linspace(0.,17.,npts)
        #Calculate progenitor orbit around this point
        pox= numpy.median(data[:,1])
        poy= numpy.median(data[:,3])
        poz= numpy.median(data[:,2])
        povx= numpy.median(data[:,4])
        povy= numpy.median(data[:,6])
        povz= numpy.median(data[:,5])
        pR,pphi,pZ= bovy_coords.rect_to_cyl(pox,poy,poz)
        pvR,pvT,pvZ= bovy_coords.rect_to_cyl_vec(povx,povy,povz,pR,
                                                 pphi,pZ,cyl=True)
        ppo= Orbit([pR/8.,pvR/220.,pvT/220.,pZ/8.,pvZ/220.,pphi])
        pno= Orbit([pR/8.,-pvR/220.,-pvT/220.,pZ/8.,-pvZ/220.,pphi])
        ppo.integrate(pts,pot)
        pno.integrate(pts,pot)
        pvec= numpy.zeros((3,npts*2-1))
        pvec[0,:npts-1]= pno.x(pts)[::-1][:-1]
        pvec[1,:npts-1]= pno.z(pts)[::-1][:-1]
        pvec[2,:npts-1]= pno.y(pts)[::-1][:-1]
        pvec[0,npts-1:]= ppo.x(pts)
        pvec[1,npts-1:]= ppo.z(pts)
        pvec[2,npts-1:]= ppo.y(pts)
        pvec*= 8.
    includetrack= True
    if includetrack:
        #Setup stream model
        lp= potential.LogarithmicHaloPotential(q=0.9,normalize=1.)
        aAI= actionAngleIsochroneApprox(b=0.8,pot=lp)
        obs= numpy.array([1.56148083,0.35081535,-1.15481504,
                          0.88719443,-0.47713334,0.12019596])
        sdf= streamdf(_SIGV/220.,progenitor=Orbit(obs),pot=lp,aA=aAI,
                      leading=True,nTrackChunks=_NTRACKCHUNKS,
                      tdisrupt=4.5/bovy_conversion.time_in_Gyr(220.,8.),
                      deltaAngleTrack=13.5,multi=_NTRACKCHUNKS)
        sdft= streamdf(_SIGV/220.,progenitor=Orbit(obs),pot=lp,aA=aAI,
                       leading=False,nTrackChunks=_NTRACKCHUNKS,
                       tdisrupt=4.5/bovy_conversion.time_in_Gyr(220.,8.),
                       deltaAngleTrack=13.5,multi=_NTRACKCHUNKS)
    #Plot
    bovy_plot.bovy_print()
    bovy_plot.bovy_plot(data[:,1],data[:,2],'k,',
                        xlabel=r'$X\,(\mathrm{kpc})$',
                        ylabel=r'$Z\,(\mathrm{kpc})$',
                        xrange=[-30.,30.],
                        yrange=[-20.,20])
    if includeorbit:
        bovy_plot.bovy_plot(pox,poz,'o',color='0.5',mec='none',overplot=True,ms=8)
        bovy_plot.bovy_plot(pvec[0,:],pvec[1,:],'k--',overplot=True,lw=1.)
    if includetrack:
        d1= 'x'
        d2= 'z'
        sdf.plotTrack(d1=d1,d2=d2,interp=True,color='k',spread=0,
                      overplot=True,lw=1.,scaleToPhysical=True)
        sdft.plotTrack(d1=d1,d2=d2,interp=True,color='k',spread=0,
                       overplot=True,lw=1.,scaleToPhysical=True)
    bovy_plot.bovy_text(r'$M^p = 2\times 10^7\,M_\odot$'+'\n'+
                        r'$\sigma_v^p = 14\,\mathrm{km\,s}^{-1}$',
                        top_left=True,size=16.)
    bovy_plot.bovy_end_print(plotfilename)

if __name__ == '__main__':
    if 'xz' in sys.argv[1]:
        plot_stream_xz(sys.argv[1])
