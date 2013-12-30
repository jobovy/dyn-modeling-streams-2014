import sys
import numpy
from scipy.misc import logsumexp
from galpy.util import bovy_plot, bovy_conversion
from galpy import potential
from galpy.orbit import Orbit
from galpy.actionAngle_src.actionAngleIsochroneApprox\
    import actionAngleIsochroneApprox
from galpy.df_src.streamdf import streamdf
from matplotlib import pyplot
_STREAMSNAPDIR= '../sim/snaps'
_STREAMSNAPAADIR= '../sim/snaps_aai'
_NTRACKCHUNKS= 11
_SIGV=0.365
_NZS= 101
_NGL= 5
def plot_pdfs_xz(plotfilename1,plotfilename2,plotfilename3):
    lp= potential.LogarithmicHaloPotential(q=0.9,normalize=1.)
    aAI= actionAngleIsochroneApprox(b=0.8,pot=lp)
    obs= numpy.array([1.56148083,0.35081535,-1.15481504,
                      0.88719443,-0.47713334,0.12019596])
    sdf= streamdf(_SIGV/220.,progenitor=Orbit(obs),pot=lp,aA=aAI,
                  leading=True,nTrackChunks=_NTRACKCHUNKS,
                  tdisrupt=4.5/bovy_conversion.time_in_Gyr(220.,8.))
    bovy_plot.bovy_print()
    pyplot.figure()
    sdf.plotTrack(d1='x',d2='z',interp=True,color='k',spread=2,
                  overplot=True,lw=1.5,scaleToPhysical=True)
    sdf.plotProgenitor(d1='x',d2='z',color='k',
                       overplot=True,lw=1.5,scaleToPhysical=True,
                       ls='--')
    bovy_plot._add_axislabels(r'$X\,(\mathrm{kpc})$',
                               r'$Z\,(\mathrm{kpc})$')
    pyplot.xlim(12.,14.6)
    pyplot.ylim(-3.,7.5)
    x1= 13.5
    x2= 14.2
    pyplot.plot([x1,x1],[4.5,6.2],'k-',lw=3.)
    pyplot.plot([x2,x2],[-1.,5.],'k-',lw=3.)
    bovy_plot._add_ticks()
    bovy_plot.bovy_end_print(plotfilename1)
    if True:
        #First PDF
        cindx= sdf._find_closest_trackpoint(x1/8.,None,5.5/8.,None,None,None,
                                            xy=True,interp=True,usev=True)
        m,c= sdf.gaussApprox([x1/8.,None,None,None,None,None],
                         interp=True,cindx=cindx)
        zs= numpy.linspace(m[1]-5.*numpy.sqrt(c[1,1]),
                           m[1]+5.*numpy.sqrt(c[1,1]),
                           _NZS)
        logps= numpy.array([sdf.callMarg([x1/8.,None,z,None,None,None],
                                         interp=True,ngl=_NGL,
                                         nsigma=3) for z in zs])
        ps= numpy.exp(logps-logsumexp(logps))
        ps/= numpy.nansum(ps)*(zs[1]-zs[0])*8.
        bovy_plot.bovy_print()
        bovy_plot.bovy_plot(zs*8.,ps,'k-',lw=1.5,
                            xlabel=r'$Z\,(\mathrm{kpc})$',
                            ylabel=r'$p(Z|X)$',
                            xrange=[(m[1]-5.*numpy.sqrt(c[1,1]))*8.,
                                    (m[1]+5.*numpy.sqrt(c[1,1]))*8.],
                            yrange=[0.,1.2*numpy.nanmax(ps)])
        bovy_plot.bovy_plot(zs*8.,
                            1./numpy.sqrt(2.*numpy.pi*c[1,1])/8.\
                                *numpy.exp(-0.5*(zs-m[1])**2./c[1,1]),
                            'k-.',lw=1.5,overplot=True)
        bovy_plot.bovy_end_print(plotfilename2)
    #Second PDF
    cindx= sdf._find_closest_trackpoint(x2/8.,None,3./8.,None,None,None,
                                        xy=True,interp=True,usev=True)
    m,c= sdf.gaussApprox([x2/8.,None,None,None,None,None],
                         interp=True,cindx=cindx)
    zs= numpy.linspace(0.,4.12,_NZS)/8.
    logps= [sdf.callMarg([x2/8.,None,z,None,None,None],
                         cindx=cindx,interp=True,ngl=_NGL,
                         nsigma=3) for z in zs if z >= .25]
    cindx= sdf._find_closest_trackpoint(x2/8.,None,0.5/8.,None,None,None,
                                        xy=True,interp=True,usev=True)
    logps1= [sdf.callMarg([x2/8.,None,z,None,None,None],
                          cindx=cindx,interp=True,ngl=_NGL,
                          nsigma=3) for z in zs if z < .25]
    logps1.extend(logps)
    logps= numpy.array(logps1)
    logps[numpy.isnan(logps)]= -100000000000000000.
    ps= numpy.exp(logps-logsumexp(logps))
    ps/= numpy.nansum(ps)*(zs[1]-zs[0])*8.
    bovy_plot.bovy_print()
    bovy_plot.bovy_plot(zs*8.,ps,'k-',lw=1.5,
                        xlabel=r'$Z\,(\mathrm{kpc})$',
                        ylabel=r'$p(Z|X)$',
                        xrange=[0.,4.12],
                        yrange=[0.,1.1*numpy.nanmax(ps)])
    bovy_plot.bovy_plot(zs*8.,
                        1./numpy.sqrt(2.*numpy.pi*c[1,1])/8.\
                            *numpy.exp(-0.5*(zs-m[1])**2./c[1,1]),
                        'k-.',lw=1.5,overplot=True)
    bovy_plot.bovy_end_print(plotfilename3)
    return None

if __name__ == '__main__':
    plot_pdfs_xz(*sys.argv[1:])
