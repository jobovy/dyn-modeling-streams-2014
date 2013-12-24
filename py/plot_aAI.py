#Make plots showing how the actionAngleIsochroneApprox works
import sys
import numpy
from galpy.util import bovy_plot
from galpy import potential
from galpy.actionAngle_src.actionAngleIsochroneApprox import actionAngleIsochroneApprox 
def plot_jr(plotfilename):
    lp= potential.LogarithmicHaloPotential(q=0.9,normalize=1.)
    aAI= actionAngleIsochroneApprox(b=0.8,pot=lp,tintJ=200,ntintJ=20000)
    obs= numpy.array([1.56148083,0.35081535,-1.15481504,0.88719443,
                      -0.47713334,0.12019596])
    bovy_plot.bovy_print(fig_width=6.)
    aAI.plot(*obs,type='jr',downsample=True)
    bovy_plot.bovy_end_print(plotfilename)

def plot_araz(plotfilename):
    lp= potential.LogarithmicHaloPotential(q=0.9,normalize=1.)
    aAI= actionAngleIsochroneApprox(b=0.8,pot=lp,tintJ=50)
    obs= numpy.array([1.56148083,0.35081535,-1.15481504,0.88719443,
                      -0.47713334,0.12019596])
    bovy_plot.bovy_print(fig_width=6.)
    aAI.plot(*obs,type='araz',downsample=True,deperiod=True)
    bovy_plot.bovy_end_print(plotfilename)

if __name__ == '__main__':
    plotfilename= sys.argv[1]
    if 'jr' in plotfilename:
        plot_jr(plotfilename)
    elif 'araz' in plotfilename:
        plot_araz(plotfilename)
