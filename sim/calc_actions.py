import sys
import os, os.path
import csv
import glob
import multiprocessing
import numpy
from galpy import potential
from galpy.util import bovy_coords, multi
from galpy.actionAngle import actionAngleStaeckel
from galpy.actionAngle_src.actionAngleIsochroneApprox import actionAngleIsochroneApprox
def calc_actions(snapfile=None):
    #Directories
    snapdir= 'snaps/'
    basefilename= snapfile.split('.')[0]
    nsnap= len(glob.glob(os.path.join(snapdir,basefilename+'_*.dat')))
    print "Processing %i snapshots ..." % nsnap
    #Setup potential
    lp= potential.LogarithmicHaloPotential(normalize=1.,q=0.9)
    if False:
        aA= actionAngleStaeckel(pot=lp,delta=1.20,c=True)
        snapaadir= 'snaps_aas/'
    else:
        aA= actionAngleIsochroneApprox(pot=lp,b=0.8)
        snapaadir= 'snaps_aai/'
    #Run each snapshot
    args= (aA,snapdir,basefilename,snapaadir)
    dummy= multi.parallel_map((lambda x: indiv_calc_actions(x,
                                                            *args)),
                                 range(nsnap),
                                 numcores=numpy.amin([nsnap,
                                                      multiprocessing.cpu_count()]))
    return None

def indiv_calc_actions(ii,aA,snapdir,basefilename,snapaadir):
    print "Working on aa %i ..." % ii
    #Read data
    data= numpy.loadtxt(os.path.join(snapdir,
                                     basefilename+'_%s.dat' % str(ii).zfill(5)),
                        delimiter=',')
    R,phi,Z= bovy_coords.rect_to_cyl(data[:,1],data[:,3],data[:,2])
    vR,vT,vZ= bovy_coords.rect_to_cyl_vec(data[:,4],data[:,6],data[:,5],
                                          R,phi,Z,cyl=True)
    R/= 8.
    Z/= 8.
    vR/= 220.
    vT/= 220.
    vZ/= 220.
    #calculation actions, frequencies, and angles
    acfs= aA.actionsFreqsAngles(R,vR,vT,Z,vZ,phi)
    print len(acfs), len(acfs[0])
    csvfile= open(os.path.join(snapaadir,basefilename+'_aa_%s.dat' % str(ii).zfill(5)),'wb')
    writer= csv.writer(csvfile,delimiter=',')
    for jj in range(len(acfs[0])):
        writer.writerow([acfs[0][jj],acfs[1][jj],acfs[2][jj],
                         acfs[3][jj],acfs[4][jj],acfs[5][jj],
                         acfs[6][jj],acfs[7][jj],acfs[8][jj]])
    csvfile.close()
    return 1

if __name__ == '__main__':
    calc_actions(snapfile=sys.argv[1])
