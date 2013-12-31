import sys
import os, os.path
import numpy
from scipy import interpolate
from scipy.misc import logsumexp
from galpy.util import bovy_plot, bovy_conversion, multi, bovy_coords
import multiprocessing
from galpy import potential
from galpy.orbit import Orbit
from galpy.actionAngle_src.actionAngleIsochroneApprox\
    import actionAngleIsochroneApprox
from galpy.df_src.streamdf import streamdf
_STREAMSNAPDIR= '../sim/snaps'
_STREAMSNAPAADIR= '../sim/snaps_aai'
_NTRACKCHUNKS= 11
_SIGV=0.365
_NLS= 1001
def plot_pdfs_l(plotfilename):
    lp= potential.LogarithmicHaloPotential(q=0.9,normalize=1.)
    aAI= actionAngleIsochroneApprox(b=0.8,pot=lp)
    obs= numpy.array([1.56148083,0.35081535,-1.15481504,
                      0.88719443,-0.47713334,0.12019596])
    sdf= streamdf(_SIGV/220.,progenitor=Orbit(obs),pot=lp,aA=aAI,
                  leading=True,nTrackChunks=_NTRACKCHUNKS,
                  vsun=[0.,30.24*8.,0.],
                  tdisrupt=4.5/bovy_conversion.time_in_Gyr(220.,8.),
                  multi=_NTRACKCHUNKS)
    sdft= streamdf(_SIGV/220.,progenitor=Orbit(obs),pot=lp,aA=aAI,
                   leading=False,nTrackChunks=_NTRACKCHUNKS,
                   vsun=[0.,30.24*8.,0.],
                   tdisrupt=4.5/bovy_conversion.time_in_Gyr(220.,8.),
                   multi=_NTRACKCHUNKS)
    #Calculate the density as a function of l, p(l)
    #Sample from sdf
    llbd= sdf.sample(n=40000,lb=True)
    tlbd= sdft.sample(n=50000,lb=True)
    b,e= numpy.histogram(llbd[0],bins=101,normed=True)
    t= ((numpy.roll(e,1)-e)/2.+e)[1:]
    lspl= interpolate.UnivariateSpline(t,numpy.log(b),k=3,s=1.)
    lls= numpy.linspace(t[0],t[-1],_NLS)
    lps= numpy.exp(lspl(lls))
    lps/= numpy.sum(lps)*(lls[1]-lls[0])*2.
    b,e= numpy.histogram(tlbd[0],bins=101,normed=True)
    t= ((numpy.roll(e,1)-e)/2.+e)[1:]
    tspl= interpolate.UnivariateSpline(t,numpy.log(b),k=3,s=0.5)
    tls= numpy.linspace(t[0],t[-1],_NLS)
    tps= numpy.exp(tspl(tls))
    tps/= numpy.sum(tps)*(tls[1]-tls[0])*2.
    bovy_plot.bovy_print(fig_width=8.25,fig_height=3.5)
    bovy_plot.bovy_plot(lls,lps,'k-',lw=1.5,
                        xlabel=r'$\mathrm{Galactic\ longitude}\,(\mathrm{deg})$',
                        ylabel=r'$p(l)$',
                        xrange=[65.,250.],
                        yrange=[0.,1.2*numpy.nanmax(numpy.hstack((lps,tps)))])
    bovy_plot.bovy_plot(tls,tps,'k-',lw=1.5,overplot=True)
    #Also plot the stream histogram
    #Read stream
    data= numpy.loadtxt(os.path.join(_STREAMSNAPDIR,'gd1_evol_hitres_01312.dat'),
                        delimiter=',')
    #Transform to (l,b)
    XYZ= bovy_coords.galcenrect_to_XYZ(data[:,1],data[:,3],data[:,2],Xsun=8.)
    lbd= bovy_coords.XYZ_to_lbd(XYZ[0],XYZ[1],XYZ[2],degree=True)
    aadata= numpy.loadtxt(os.path.join(_STREAMSNAPAADIR,
                                       'gd1_evol_hitres_aa_01312.dat'),
                          delimiter=',')
    thetar= aadata[:,6]
    thetar= (numpy.pi+(thetar-numpy.median(thetar))) % (2.*numpy.pi)
    indx= numpy.fabs(thetar-numpy.pi) > (5.*numpy.median(numpy.fabs(thetar-numpy.median(thetar))))
    lbd= lbd[indx,:]
    bovy_plot.bovy_hist(lbd[:,0],bins=40,range=[65.,250.],
                        histtype='step',normed=True,
                        overplot=True,
                        lw=1.5,color='k')
    bovy_plot.bovy_end_print(plotfilename)

if __name__ == '__main__':
    numpy.random.seed(1)
    plot_pdfs_l(sys.argv[1])
