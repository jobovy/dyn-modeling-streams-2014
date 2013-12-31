import sys
import os, os.path
import numpy
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
_NLS= 301
_NGL= 5
def plot_pdfs_x(plotfilename):
    lp= potential.LogarithmicHaloPotential(q=0.9,normalize=1.)
    aAI= actionAngleIsochroneApprox(b=0.8,pot=lp)
    obs= numpy.array([1.56148083,0.35081535,-1.15481504,
                      0.88719443,-0.47713334,0.12019596])
    sdft= streamdf(_SIGV/220.,progenitor=Orbit(obs),pot=lp,aA=aAI,
                   leading=False,nTrackChunks=_NTRACKCHUNKS,
                   vsun=[0.,30.24*8.,0.],
                   tdisrupt=4.5/bovy_conversion.time_in_Gyr(220.,8.),
                   multi=_NTRACKCHUNKS)
    #Calculate the density as a function of l, p(l)
    txs= numpy.linspace(3.,12.4,_NLS)
    tlogps= multi.parallel_map((lambda x: sdft.callMarg([txs[x]/8.,None,None,None,None,None],
                                                        interp=True,ngl=_NGL,
                                                        nsigma=3)),
                              range(_NLS),
                              numcores=numpy.amin([_NLS,
                                                   multiprocessing.cpu_count()]))
    tlogps= numpy.array(tlogps)
    tlogps[numpy.isnan(tlogps)]= -100000000000000000.
    tps= numpy.exp(tlogps-logsumexp(tlogps))
    tps/= numpy.nansum(tps)*(txs[1]-txs[0])
    bovy_plot.bovy_print()
    bovy_plot.bovy_plot(txs,tps,'k-',lw=1.5,
                        xlabel=r'$X\,(\mathrm{kpc})$',
                        ylabel=r'$p(X)$',
                        xrange=[3.,12.4],
                        yrange=[0.,1.2*numpy.nanmax(tps)])
    bovy_plot.bovy_plot(txs,tps,'k-',lw=1.5,overplot=True)
    #Also plot the stream histogram
    #Read stream
    data= numpy.loadtxt(os.path.join(_STREAMSNAPDIR,'gd1_evol_hitres_01312.dat'),
                        delimiter=',')
    aadata= numpy.loadtxt(os.path.join(_STREAMSNAPAADIR,
                                       'gd1_evol_hitres_aa_01312.dat'),
                          delimiter=',')
    thetar= aadata[:,6]
    thetar= (numpy.pi+(thetar-numpy.median(thetar))) % (2.*numpy.pi)
    indx= thetar-numpy.pi < -(5.*numpy.median(numpy.fabs(thetar-numpy.median(thetar))))
    data= data[indx,:]
    bovy_plot.bovy_hist(data[:,1],bins=20,range=[3.,12.4],
                        histtype='step',normed=True,
                        overplot=True,
                        lw=1.5,color='k')
    bovy_plot.bovy_end_print(plotfilename)
    
if __name__ == '__main__':
    plot_pdfs_x(sys.argv[1])
