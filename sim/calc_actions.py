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
                                 numcores=numpy.amin([64,nsnap,
                                                      multiprocessing.cpu_count()]))
    return None

def indiv_calc_actions(x,aA,snapdir,basefilename,snapaadir):
    print "Working on aa %i ..." % x
    #Read data
    data= numpy.loadtxt(os.path.join(snapdir,
                                     basefilename+'_%s.dat' % str(x).zfill(5)),
                        delimiter=',')
    R,phi,Z= bovy_coords.rect_to_cyl(data[:,1],data[:,3],data[:,2])
    vR,vT,vZ= bovy_coords.rect_to_cyl_vec(data[:,4],data[:,6],data[:,5],
                                          R,phi,Z,cyl=True)
    R/= 8.
    Z/= 8.
    vR/= 220.
    vT/= 220.
    vZ/= 220.
    if False: #Used for testing
        R= R[0:100]
        vR= vR[0:100]
        vT= vT[0:100]
        Z= Z[0:100]
        vZ= vZ[0:100]
        phi= phi[0:100]
    nx= len(R)
    #calculation actions, frequencies, and angles
    if isinstance(aA,actionAngleIsochroneApprox):
        #Processes in batches to not run out of memory
        jr,lz,jz,Or,Op,Oz,ar,ap,az= [],[],[],[],[],[],[],[],[]
        for ii in range(nx/20):
            tR= R[ii*20:numpy.amin([(ii+1)*20,nx])]
            tvR= vR[ii*20:numpy.amin([(ii+1)*20,nx])]
            tvT= vT[ii*20:numpy.amin([(ii+1)*20,nx])]
            tZ= Z[ii*20:numpy.amin([(ii+1)*20,nx])]
            tvZ= vZ[ii*20:numpy.amin([(ii+1)*20,nx])]
            tphi= phi[ii*20:numpy.amin([(ii+1)*20,nx])]
            tacfs= aA.actionsFreqsAngles(tR,tvR,tvT,tZ,tvZ,tphi)
            jr.extend(list(tacfs[0]))
            lz.extend(list(tacfs[1]))
            jz.extend(list(tacfs[2]))
            Or.extend(list(tacfs[3]))
            Op.extend(list(tacfs[4]))
            Oz.extend(list(tacfs[5]))
            ar.extend(list(tacfs[6]))
            ap.extend(list(tacfs[7]))
            az.extend(list(tacfs[8]))
        acfs= (jr,lz,jz,Or,Op,Oz,ar,ap,az)
    else:
        acfs= aA.actionsFreqsAngles(R,vR,vT,Z,vZ,phi)
    csvfile= open(os.path.join(snapaadir,basefilename+'_aa_%s.dat' % str(x).zfill(5)),'wb')
    writer= csv.writer(csvfile,delimiter=',')
    for jj in range(len(acfs[0])):
        writer.writerow([acfs[0][jj],acfs[1][jj],acfs[2][jj],
                         acfs[3][jj],acfs[4][jj],acfs[5][jj],
                         acfs[6][jj],acfs[7][jj],acfs[8][jj]])
    csvfile.close()
    print "Done with aa %i" % x
    return 1

if __name__ == '__main__':
    calc_actions(snapfile=sys.argv[1])