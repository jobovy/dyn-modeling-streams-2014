import sys
import csv
import numpy
from galpy.util import bovy_coords
from galpy import potential
from galpy.orbit import Orbit
from galpy.actionAngle import actionAngleStaeckel
from galpy.actionAngle_src.actionAngleIsochroneApprox import actionAngleIsochroneApprox
def calc_progenitor_actions(savefilename):
    #Setup potential
    lp= potential.LogarithmicHaloPotential(normalize=1.,q=0.9)
    #Setup orbit
    x,z,y,vx,vz,vy= -11.63337239,-10.631736273934635,-20.76235661,-128.8281653,79.172383882274971,42.88727925
    R,phi,z= bovy_coords.rect_to_cyl(x,y,z)
    vR,vT,vz= bovy_coords.rect_to_cyl_vec(vx,vy,vz,R,phi,z,cyl=True)
    R/= 8.
    z/= 8.
    vR/= 220.
    vT/= 220.
    vz/= 220.
    o= Orbit([R,vR,vT,z,vz,phi])
    ts= numpy.linspace(0.,5.125*220./8.,1313) #times of the snapshots
    o.integrate(ts,lp,method='dopr54_c')
    if 'aas' in savefilename:
        aA= actionAngleStaeckel(pot=lp,delta=1.20,c=True)
    else:
        aA= actionAngleIsochroneApprox(pot=lp,b=0.8)
    #Now calculate actions, frequencies, and angles for all positions
    Rs= o.R(ts)
    vRs= o.vR(ts)
    vTs= o.vT(ts)
    zs= o.z(ts)
    vzs= o.vz(ts)
    phis= o.phi(ts)
    csvfile= open(savefilename,'wb')
    writer= csv.writer(csvfile,delimiter=',')
    for ii in range(len(ts)):
        acfs= aA.actionsFreqsAngles(Rs[ii],vRs[ii],vTs[ii],zs[ii],vzs[ii],
                                    phis[ii])
        writer.writerow([acfs[0][0],acfs[1][0],acfs[2][0],
                         acfs[3][0],acfs[4][0],acfs[5][0],
                         acfs[6][0],acfs[7][0],acfs[8][0]])
        csvfile.flush()
    csvfile.close()
    return None

if __name__ == '__main__':
    calc_progenitor_actions(sys.argv[1])
