import sys
import os, os.path
import csv
import numpy
def split_snaps(snapfile=None):
    #Directories
    snapdir= 'snaps/'
    basefilename= snapfile.split('.')[0]
    #Read data
    print "Reading data ..."
    data= numpy.loadtxt(snapfile,comments='#')
    print "Done reading data"
    nx= 10000
    nt= data.shape[0]/nx
    #Split snapshots and save to separate files
    for ii in range(nt):
        tdata= data[ii*nx:(ii+1)*nx,:]
        csvfile= open(os.path.join(snapdir,basefilename+'_%s.dat' % str(ii).zfill(5)),'wb')
        writer= csv.writer(csvfile,
                           delimiter=',')
        for jj in range(nx):
            writer.writerow([tdata[jj,0],tdata[jj,1],tdata[jj,2],tdata[jj,3],
                             tdata[jj,4],tdata[jj,5],tdata[jj,6]])
        csvfile.close()
    return None

if __name__ == '__main__':
    split_snaps(snapfile=sys.argv[1])
