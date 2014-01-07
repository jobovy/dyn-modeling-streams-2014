import sys
import copy
import os, os.path
import numpy
from galpy.util import bovy_plot, bovy_coords, bovy_conversion
from galpy import potential
from galpy.orbit import Orbit
from galpy.actionAngle_src.actionAngleIsochroneApprox\
    import actionAngleIsochroneApprox
from galpy.df_src.streamdf import streamdf
from matplotlib import pyplot
from matplotlib.ticker import NullFormatter
_STREAMSNAPDIR= '../sim/snaps'
_STREAMSNAPAADIR= '../sim/snaps_aai'
_NTRACKCHUNKS= 11
_SIGV=0.365
def plot_stream_xz(plotfilename):
    #Read stream
    data= numpy.loadtxt(os.path.join(_STREAMSNAPDIR,'gd1_evol_hitres_00800.dat'),
                        delimiter=',')
    aadata= numpy.loadtxt(os.path.join(_STREAMSNAPAADIR,
                                       'gd1_evol_hitres_aa_00800.dat'),
                        delimiter=',')
    thetar= aadata[:,6]
    thetar= (numpy.pi+(thetar-numpy.median(thetar))) % (2.*numpy.pi)
    if 'sim' in plotfilename:
        sindx= numpy.fabs(thetar-numpy.pi) > (4.*numpy.median(numpy.fabs(thetar-numpy.median(thetar))))
    else:
        sindx= numpy.fabs(thetar-numpy.pi) > (1.5*numpy.median(numpy.fabs(thetar-numpy.median(thetar))))
    includeorbit= True
    if includeorbit:
        npts= 201
        pot= potential.LogarithmicHaloPotential(normalize=1.,q=0.9)
        pts= numpy.linspace(0.,4.,npts)
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
        #Obs is at time 1312, need to go back 2 Gyr to time 800
        obs[1]*= -1.
        obs[2]*= -1.
        obs[4]*= -1.
        o= Orbit(obs)
        ts= numpy.linspace(0.,2.*977.7922212082034/1000./bovy_conversion.time_in_Gyr(220.,8.),1001)
        o.integrate(ts,lp)
        obs= o(ts[-1])._orb.vxvv
        obs[1]*= -1.
        obs[2]*= -1.
        obs[4]*= -1.       
        tdisrupt= 4.5-2.*977.7922212082034/1000.
        sdf= streamdf(_SIGV/220.,progenitor=Orbit(obs),pot=lp,aA=aAI,
                      leading=True,nTrackChunks=_NTRACKCHUNKS,
                      tdisrupt=tdisrupt/bovy_conversion.time_in_Gyr(220.,8.),
                      deltaAngleTrack=1.5*3./5.,multi=_NTRACKCHUNKS)
        sdft= streamdf(_SIGV/220.,progenitor=Orbit(obs),pot=lp,aA=aAI,
                       leading=False,nTrackChunks=_NTRACKCHUNKS,
                       tdisrupt=tdisrupt/bovy_conversion.time_in_Gyr(220.,8.),
                       deltaAngleTrack=1.5*3./5.,multi=_NTRACKCHUNKS)
    if 'sim' in plotfilename:
        #Replace data with simulated data
        forwardXY= sdf.sample(int(round(numpy.sum(sindx)/2.)),
                              xy=True)
        backwardXY= sdft.sample(int(round(numpy.sum(sindx)/2.)),
                                xy=True)
        data= numpy.empty((forwardXY.shape[1]+backwardXY.shape[1],7))
        data[:forwardXY.shape[1],1]= forwardXY[0]*8.
        data[:forwardXY.shape[1],2]= forwardXY[2]*8.
        data[:forwardXY.shape[1],3]= forwardXY[1]*8.
        data[:forwardXY.shape[1],4]= forwardXY[3]*220.
        data[:forwardXY.shape[1],5]= forwardXY[5]*220.
        data[:forwardXY.shape[1],6]= forwardXY[4]*220.
        data[forwardXY.shape[1]:,1]= backwardXY[0]*8.
        data[forwardXY.shape[1]:,2]= backwardXY[2]*8.
        data[forwardXY.shape[1]:,3]= backwardXY[1]*8.
        data[forwardXY.shape[1]:,4]= backwardXY[3]*220.
        data[forwardXY.shape[1]:,5]= backwardXY[5]*220.
        data[forwardXY.shape[1]:,6]= backwardXY[4]*220.
        sindx= numpy.ones(data.shape[0],dtype='bool')      
    #Plot
    bovy_plot.bovy_print()
    bovy_plot.bovy_plot(data[sindx,1],data[sindx,2],'k,',ms=2.,
                        xlabel=r'$X\,(\mathrm{kpc})$',
                        ylabel=r'$Z\,(\mathrm{kpc})$',
                        xrange=[-12.5,-3.],
                        yrange=[-12.5,-7.])
    if numpy.sum(True-sindx) > 0:
        #Also plot progenitor
        pindx= copy.copy(True-sindx)
        pindx[0:9900]= False #subsample
        bovy_plot.bovy_plot(data[pindx,1],data[pindx,2],
                            'k,',overplot=True)
    if includeorbit:
        bovy_plot.bovy_plot(pox,poz,'o',color='0.5',mec='none',overplot=True,ms=8)
        bovy_plot.bovy_plot(pvec[0,:],pvec[1,:],'k--',overplot=True,lw=1.)
    if 'sim' in plotfilename:
        bovy_plot.bovy_text(r'$\mathrm{mock\ stream}$',
                            bottom_left=True,size=16.)
    else:
        bovy_plot.bovy_text(r'$N\!\!-\!\!\mathrm{body\ stream}$',
                            bottom_left=True,size=16.)
    if includetrack:
        d1= 'x'
        d2= 'z'
        sdf.plotTrack(d1=d1,d2=d2,interp=True,color='k',spread=0,
                      overplot=True,lw=1.,scaleToPhysical=True)
        sdft.plotTrack(d1=d1,d2=d2,interp=True,color='k',spread=0,
                       overplot=True,lw=1.,scaleToPhysical=True)
        #Also create inset
        pyplot.plot([-9.,-9.],[-11.75,-10.3],'k-')
        pyplot.plot([-6.,-6.],[-11.75,-10.3],'k-')
        pyplot.plot([-9.,-6.],[-11.75,-11.75],'k-')
        pyplot.plot([-9.,-6.],[-10.3,-10.3],'k-')
        pyplot.plot([-6.,-3.4],[-10.3,-10.],'k:')
        pyplot.plot([-9.,-8.85],[-10.3,-10.],'k:')
        insetAxes= pyplot.axes([0.42,0.47,0.45,0.42])
        pyplot.sca(insetAxes)
        bovy_plot.bovy_plot(data[sindx,1],data[sindx,2],'k.',ms=2.,zorder=0.,
                            overplot=True)
        if numpy.sum(True-sindx) > 0:
            pindx= copy.copy(True-sindx)
            pindx[0:9700]= False #subsample
            bovy_plot.bovy_plot(data[pindx,1],data[pindx,2],'k,',
                                zorder=0.,
                                overplot=True)
        bovy_plot.bovy_plot(pvec[0,:],pvec[1,:],'k--',overplot=True,lw=1.,
                            zorder=1)
        sdf.plotTrack(d1=d1,d2=d2,interp=True,color='k',spread=0,
                      overplot=True,lw=1.,scaleToPhysical=True,zorder=2)
        nullfmt   = NullFormatter()         # no labels
        insetAxes.xaxis.set_major_formatter(nullfmt)
        insetAxes.yaxis.set_major_formatter(nullfmt)
        insetAxes.set_xlim(-9.,-6.)
        insetAxes.set_ylim(-11.75,-10.3)
        pyplot.tick_params(\
            axis='both',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom='off',      # ticks along the bottom edge are off
            top='off',         # ticks along the top edge are off
            left='off',      # ticks along the bottom edge are off
            right='off')         # ticks along the top edge are off
    bovy_plot.bovy_end_print(plotfilename)

if __name__ == '__main__':
    plot_stream_xz(sys.argv[1])
