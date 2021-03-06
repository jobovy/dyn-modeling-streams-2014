import sys
import os, os.path
import csv
import glob
import multiprocessing
import subprocess
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
    if True:
        calcThese= []
        for ii in range(nsnap):
            csvfilename= os.path.join(snapaadir,basefilename+'_aa_%s.dat' % str(ii).zfill(5))
            if os.path.exists(csvfilename):
                #Don't recalculate those that have already been calculated
                nstart= int(subprocess.check_output(['wc','-l',csvfilename]).split(' ')[0])
                if nstart < 10000:
                    calcThese.append(ii)
            else:
                calcThese.append(ii)
        nsnap= len(calcThese)
    if len(calcThese) == 0:
        print "All done with everything ..."
        return None
    args= (aA,snapdir,basefilename,snapaadir)
    print "Using %i cpus ..." % (numpy.amin([64,nsnap,
                                             multiprocessing.cpu_count()]))
    dummy= multi.parallel_map((lambda x: indiv_calc_actions(x,
                                                            *args)),
                              calcThese,
#                              range(nsnap),
                              numcores=numpy.amin([64,nsnap,
                                                      multiprocessing.cpu_count()]))
    return None

def indiv_calc_actions(x,aA,snapdir,basefilename,snapaadir):
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
        csvfilename= os.path.join(snapaadir,basefilename+'_aa_%s.dat' % str(x).zfill(5))
        if os.path.exists(csvfilename):
            #Don't recalculate those that have already been calculated
            nstart= int(subprocess.check_output(['wc','-l',csvfilename]).split(' ')[0])
            csvfile= open(csvfilename,'ab')
        else:
            csvfile= open(csvfilename,'wb')
            nstart= 0
        if nstart >= nx: return 1 #Done already
        print "Working on aa %i ..." % x
        print "Starting from %i ..." % nstart
        nx-= nstart
        writer= csv.writer(csvfile,delimiter=',')
        nbatch= 20
        for ii in range(nx/nbatch):
            tR= R[nstart+ii*nbatch:numpy.amin([nstart+(ii+1)*nbatch,nstart+nx])]
            tvR= vR[nstart+ii*nbatch:numpy.amin([nstart+(ii+1)*nbatch,nstart+nx])]
            tvT= vT[nstart+ii*nbatch:numpy.amin([nstart+(ii+1)*nbatch,nstart+nx])]
            tZ= Z[nstart+ii*nbatch:numpy.amin([nstart+(ii+1)*nbatch,nstart+nx])]
            tvZ= vZ[nstart+ii*nbatch:numpy.amin([nstart+(ii+1)*nbatch,nstart+nx])]
            tphi= phi[nstart+ii*nbatch:numpy.amin([nstart+(ii+1)*nbatch,nstart+nx])]
            try:
                tacfs= aA.actionsFreqsAngles(tR,tvR,tvT,tZ,tvZ,tphi)
            except numpy.linalg.linalg.LinAlgError:
                print x,tR,tvR,tvT,tZ,tvZ,tphi
                raise
            for jj in range(len(tacfs[0])):
                writer.writerow([tacfs[0][jj],tacfs[1][jj],tacfs[2][jj],
                                 tacfs[3][jj],tacfs[4][jj],tacfs[5][jj],
                                 tacfs[6][jj],tacfs[7][jj],tacfs[8][jj]])
                csvfile.flush()
        csvfile.close()
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
