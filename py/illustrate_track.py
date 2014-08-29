import sys
import numpy
from galpy.util import bovy_plot, bovy_conversion
from galpy import potential
from galpy.orbit import Orbit
from galpy.actionAngle_src.actionAngleIsochroneApprox\
    import actionAngleIsochroneApprox
from galpy.df_src.streamdf import streamdf
from matplotlib import pyplot
_NTRACKCHUNKS= 4
_SIGV=0.365
def illustrate_track(plotfilename1,plotfilename2,plotfilename3):
    #Setup stream model
    lp= potential.LogarithmicHaloPotential(q=0.9,normalize=1.)
    aAI= actionAngleIsochroneApprox(b=0.8,pot=lp)
    obs= numpy.array([1.56148083,0.35081535,-1.15481504,
                      0.88719443,-0.47713334,0.12019596])
    sdf= streamdf(_SIGV/220.,progenitor=Orbit(obs),pot=lp,aA=aAI,
                  leading=True,nTrackChunks=_NTRACKCHUNKS,
                  tdisrupt=4.5/bovy_conversion.time_in_Gyr(220.,8.))
    #First calculate meanOmega and sigOmega
    mOs= numpy.array([sdf.meanOmega(t,oned=True) for t in sdf._thetasTrack])
    sOs= numpy.array([sdf.sigOmega(t) for t in sdf._thetasTrack])
    mOs-= sdf._progenitor_Omega_along_dOmega
    mOs*= -bovy_conversion.freq_in_Gyr(220.,8.)
    sOs*= bovy_conversion.freq_in_Gyr(220.,8.)
    progAngle= numpy.dot(sdf._progenitor_angle,sdf._dsigomeanProgDirection)
    bovy_plot.bovy_print(fig_width=8.25,fig_height=3.5)
    bovy_plot.bovy_plot(sdf._thetasTrack+progAngle,mOs,'ko',ms=8.,
                        xlabel=r'$\theta_\parallel$',
                        ylabel=r'$\Omega_\parallel\,(\mathrm{Gyr}^{-1})$',
                        xrange=[-0.2-1.14,1.6-1.14],
                        yrange=[22.05,22.55])
    bovy_plot.bovy_plot(sdf._thetasTrack+progAngle,mOs,'k-',lw=1.5,overplot=True)
    bovy_plot.bovy_plot(sdf._thetasTrack+progAngle,
                        mOs[0]*numpy.ones(len(sdf._thetasTrack))+0.03,
                        'ko',ls='--',dashes=(20,10),lw=1.5,overplot=True,
                        ms=6.)
    bovy_plot.bovy_plot(sdf._thetasTrack+progAngle,mOs+2*sOs,'ko',ms=6.,mfc='none',
                        zorder=1,overplot=True)
    bovy_plot.bovy_plot(sdf._thetasTrack+progAngle,mOs-2*sOs,'ko',ms=6.,mfc='none',
                        zorder=1,overplot=True)
    bovy_plot.bovy_plot(sdf._thetasTrack+progAngle,mOs+2*sOs,'k-.',lw=1.5,
                        zorder=0,overplot=True)
    bovy_plot.bovy_plot(sdf._thetasTrack+progAngle,mOs-2*sOs,'k-.',lw=1.5,
                        zorder=0,overplot=True)
    bovy_plot.bovy_plot(sdf._thetasTrack+progAngle,sdf._progenitor_Omega_along_dOmega*bovy_conversion.freq_in_Gyr(220.,8.)*numpy.ones(len(sdf._thetasTrack)),
                        'k--',lw=1.5,overplot=True)
    bovy_plot.bovy_plot((sdf._thetasTrack+progAngle)[0],(sdf._progenitor_Omega_along_dOmega*bovy_conversion.freq_in_Gyr(220.,8.)*numpy.ones(len(sdf._thetasTrack)))[0],
                        'ko',ms=6.,overplot=True)
    bovy_plot.bovy_text(1.05+progAngle,22.475,r'$\mathrm{progenitor\ orbit}$',size=16.)
    bovy_plot.bovy_text(progAngle+0.05,22.50,r'$\mathrm{current\ progenitor\ position}$',size=16.)
    bovy_plot.bovy_plot([progAngle+0.05,progAngle],[22.50,sdf._progenitor_Omega_along_dOmega*bovy_conversion.freq_in_Gyr(220.,8.)],'k:',overplot=True)
    bovy_plot.bovy_text(-1.2,22.35,r"$\mathrm{At\ the\ progenitor's}\ \theta_{\parallel}, \mathrm{we\ calculate\ an\ auxiliary\ orbit\ through}$"+'\n'+r"$(\mathbf{x}_a,\mathbf{v}_a) = (\mathbf{\Omega}_p+\Delta \mathbf{\Omega}^m,\boldsymbol{\theta}_p)\ \mathrm{using\ a\ linearized}\ (\mathbf{\Omega},\boldsymbol{\theta})\ \mathrm{to}\ (\mathbf{x},\mathbf{v}).$",size=16.)
    yarcs= numpy.linspace(22.30,22.39,101)
    bovy_plot.bovy_plot(sdf._thetasTrack[0]+progAngle-0.1*numpy.sqrt(1.-(yarcs-22.35)**2./0.05**2.),yarcs,'k:',
                        overplot=True)
    bovy_plot.bovy_text(-1.3,22.07,r'$\mathrm{At\ a\ small\ number\ of\ points, we\ calculate}$'+'\n'+r'$\partial(\mathbf{\Omega},\boldsymbol{\theta})/\partial (\mathbf{x},\mathbf{v}), \mathrm{the\ mean\ stream\ track\ in}\ (\mathbf{\Omega},\boldsymbol{\theta})^\dagger,$'+'\n'+r'$\mathrm{and\ estimate\ the\ spread\ around\ the\ track}.$',size=16.)
    bovy_plot.bovy_plot([-0.9,sdf._thetasTrack[1]+progAngle],
                        [22.185,mOs[1]+0.03],
                        'k:',overplot=True)
    bovy_plot.bovy_plot([-0.9,progAngle+sdf._thetasTrack[1]],
                        [22.185,mOs[1]],
                        'k:',overplot=True)
    bovy_plot.bovy_text(-0.18,22.265,r'$\mathrm{stream\ track\ +\ spread}$',
                         size=16.,
                        rotation=-20.)
    bovy_plot.bovy_end_print(plotfilename1)
    #Now plot Z,X
    bovy_plot.bovy_print(fig_width=8.25,fig_height=3.5)
    pyplot.figure()
    sdf.plotTrack(d1='z',d2='x',interp=True,
                  color='k',spread=2,overplot=True,lw=1.5,
                  scaleToPhysical=True)
    sdf.plotTrack(d1='z',d2='x',interp=False,marker='o',ms=8.,color='k',
                  overplot=True,ls='none',
                  scaleToPhysical=True)
    sdf.plotProgenitor(d1='z',d2='x',color='k',
                       overplot=True,ls='--',lw=1.5,dashes=(20,10),
                       scaleToPhysical=True)
    pyplot.plot(sdf._progenitor.z(sdf._trackts)*8.,
                sdf._progenitor.x(sdf._trackts)*8.,marker='o',ms=6.,
                ls='none',
                color='k')
    pyplot.xlim(8.,-3.)
    pyplot.ylim(12.,15.5)
    bovy_plot._add_ticks()
    bovy_plot._add_axislabels(r'$Z\,(\mathrm{kpc})$',r'$X\,(\mathrm{kpc})$')
    bovy_plot.bovy_text(0.,14.25,r'$\mathrm{auxiliary\ orbit}$',
                        size=16.,rotation=-20.)
    bovy_plot.bovy_text(1.,13.78,r'$\mathrm{stream\ track\ +\ spread}$',
                        size=16.,rotation=-25.)
    bovy_plot.bovy_text(7.5,14.2,r"$\mathrm{At\ these\ points, we\ calculate\ the\ stream\ position\ in}\ (\mathbf{x},\mathbf{v})\ \mathrm{from}$"+
                         '\n'+r"$\mathrm{the\ auxiliary's}\ (\mathbf{x}_a,\mathbf{v}_a) = (\mathbf{\Omega}_a,\boldsymbol{\theta}_a), \mathrm{the\ mean\ offset} (\Delta \mathbf{\Omega},\Delta \boldsymbol{\theta}),$"+'\n'+
                         r"$\mathrm{and}\ \left(\frac{\partial(\mathbf{\Omega},\boldsymbol{\theta})}{\partial (\mathbf{x},\mathbf{v})}\right)^{-1 \, \dagger}.$",
                        size=16.)
    bovy_plot.bovy_plot([sdf._progenitor.z(sdf._trackts[1])*8.,4.5],
                        [sdf._progenitor.x(sdf._trackts[1])*8.,14.8],
                        'k:',overplot=True)
    bovy_plot.bovy_text(5.6,12.4,r"$\mathrm{We\ interpolate\ the\ track\ between\ the}$"+'\n'+r"$\mathrm{calculated\ points\ and\ use\ slerp\ to}$"+'\n'+r"$\mathrm{interpolate\ the\ estimated\ 6D\ spread.}$",
                         size=16.)
    bovy_plot.bovy_plot([3.,sdf._interpolatedObsTrackXY[500,2]*8.],
                        [13.3,sdf._interpolatedObsTrackXY[500,0]*8.],
                        'k:',overplot=True)
    bovy_plot.bovy_end_print(plotfilename2)
    #Finally plot l vs. d
    bovy_plot.bovy_print(fig_width=8.25,fig_height=3.5)
    pyplot.figure()
    sdf.plotTrack(d1='ll',d2='dist',interp=True,
                  color='k',spread=2,overplot=True,lw=1.5)
    sdf.plotTrack(d1='ll',d2='dist',interp=False,marker='o',ms=8.,color='k',
                  overplot=True,ls='none')
    sdf.plotProgenitor(d1='ll',d2='dist',color='k',dashes=(20,10),
                       overplot=True,ls='--',lw=1.5)
    pyplot.plot(sdf._progenitor.ll(sdf._trackts,
                                        obs=[sdf._R0,0.,sdf._Zsun],ro=sdf._Rnorm),
                sdf._progenitor.dist(sdf._trackts,
                                        obs=[sdf._R0,0.,sdf._Zsun],ro=sdf._Rnorm),
                marker='o',ms=6.,
                ls='none',
                color='k')
    pyplot.xlim(157.,260.)
    pyplot.ylim(7.4,15.5)
    bovy_plot._add_ticks()
    bovy_plot._add_axislabels(r'$\mathrm{Galactic\ longitude\, (deg)}$',
                              r'$\mathrm{distance\, (kpc)}$')
    bovy_plot.bovy_text(165.,13.5,r"$\mathrm{Finally, the\ interpolated\ track\ in}\ (\mathbf{x},\mathbf{v})\ \mathrm{is}$"+'\n'+r"$\mathrm{converted\ to\ observable\ quantities\ (here}, l\ \mathrm{and}\ D).$",
                        size=16.)
    bovy_plot.bovy_plot([230.,sdf._interpolatedObsTrackLB[850,0]],
                        [13.25,sdf._interpolatedObsTrackLB[850,2]],
                        'k:',overplot=True)
    bovy_plot.bovy_text(170.,9.4,r"$\mathrm{The\ estimated\ spread\ is\ propagated}$"+'\n'+r"$\mathrm{at\ the\ points\ directly\ from}\ (\mathbf{\Omega},\boldsymbol{\theta})\ \mathrm{to}$"+'\n'+r"$(l,b,D,\ldots)\ \mathrm{and\ interpolated}$"+'\n'+r"$\mathrm{using\ slerp}.$",
                        size=16.)
    bovy_plot.bovy_plot([195.,sdf._ObsTrackLB[1,0]],
                        [9.7,sdf._ObsTrackLB[1,2]],
                        'k:',overplot=True)
    bovy_plot.bovy_end_print(plotfilename3)
    return None

if __name__ == '__main__':
    illustrate_track(sys.argv[1],sys.argv[2],sys.argv[3])
