import sys
import os, os.path
import numpy
import subprocess
from galpy import potential
from galpy.orbit import Orbit
from galpy.util import bovy_plot
from galpy.actionAngle_src.actionAngleIsochroneApprox import actionAngleIsochroneApprox
def make_action_movie(aa=False):
    skip= 1
    if aa:
        savedir= '../movies/aa/'
        savedirpng= '../movies/aa/pngs/'
        moviefilename= os.path.join(savedir,'aa_orbit_skip%i.mpg' % skip)
        basefilename= 'aa_'
        xrange=[-0.7,7.]
        yrange=[-0.7,7.]
        xlabel=r'$\theta_R$'
        ylabel=r'$\theta_Z$'    
    else:
        savedir= '../movies/actions/'
        savedirpng= '../movies/actions/pngs/'
        basefilename= 'real_'
        moviefilename= os.path.join(savedir,'real_orbit_skip%i.mpg' % skip)
        xrange=[0.,7.]
        yrange=[-4.25,4.25]
        xlabel=r'$R$'
        ylabel=r'$Z$'    
    pot= potential.MWPotential
    o= Orbit([1.,0.8,1.5,0.8,0.5,0.])
    ts= numpy.linspace(0.,300.,1000)
    o.integrate(ts,pot)
    Rs= numpy.zeros(len(ts)+23)
    zs= numpy.zeros(len(ts)+23)
    Rs[:23]= numpy.nan
    zs[:23]= numpy.nan
    if aa:
        aAIA= actionAngleIsochroneApprox(b=5.,pot=pot)
        acfs= aAIA.actionsFreqsAngles(o,maxn=3)
        Rs[23:]= (acfs[6]+acfs[3]*ts) % (2.*numpy.pi)
        zs[23:]= (acfs[8]+acfs[5]*ts) % (2.*numpy.pi)
    else:
        Rs[23:]= o.R(ts)
        zs[23:]= o.z(ts)
        print numpy.nanmax(Rs)
        print numpy.nanmax(numpy.fabs(zs))
    if True:
        for ii in range(len(ts)-1):       
            bovy_plot.bovy_print()
            bovy_plot.bovy_plot(Rs[ii:ii+24:skip],zs[ii:ii+24:skip],
                                scatter=True,color='k',
                                s=numpy.arange(24/skip)*skip,
                                xlabel=xlabel,
                                ylabel=ylabel,
                                xrange=xrange,
                                yrange=yrange)
            bovy_plot.bovy_end_print(os.path.join(savedirpng,basefilename+'%s.png' % str(ii).zfill(5)))
    #Turn into movie
    framerate= 25
    bitrate= 1000000
    try:
        subprocess.check_call(['ffmpeg',
                               '-i',
                               os.path.join(savedirpng,basefilename+'%05d.png'),
                               '-y',
                               '-r',str(framerate),
                               '-b', str(bitrate),
                               moviefilename])
    except subprocess.CalledProcessError:
        print "'ffmpeg' failed"
    return None

if __name__ == '__main__':
    make_action_movie(aa=len(sys.argv) > 1)
