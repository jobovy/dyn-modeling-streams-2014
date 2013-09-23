import sys
import os, os.path
import subprocess
import numpy
from numpy import linalg
from galpy.util import bovy_plot, bovy_coords
from galpy import potential
from galpy.orbit import Orbit
def make_sim_movie(proj='xz',comov=False,skippng=False,
                   includeorbit=True):
    #Directories
    savedirpng= './movies/gd1/pngs/'
    basefilename= 'gd1_evol_'
    moviefilename= 'gd1_evol'
    #Read data
    datafile= 'gd1_evol_hitres.dat'
    #datafile= 'gd1_evol.dat'
    print "Reading data ..."
    data= numpy.loadtxt(datafile,comments='#')
    print "Done reading data"
    if proj.lower() == 'xz':
        includeorbit= False #just to be sure
        basefilename+= 'xz_'
        moviefilename+= '_xz'
        x= data[:,1]
        y= data[:,2]
        if comov:
            basefilename+= 'comov_'
            moviefilename+= '_comov'
            xrange=[-12.,12.]
            yrange=[-10.,10.]           
        else:
            xrange=[-30.,30.]
            yrange=[-15.,15.]           
        xlabel=r'$X\,(\mathrm{kpc})$'
        ylabel=r'$Z\,(\mathrm{kpc})$'
    elif proj.lower() == 'yz':
        basefilename+= 'yz_'
        moviefilename+= '_yz'
        x= data[:,3]
        y= data[:,2]
        if comov:
            basefilename+= 'comov_'
            moviefilename+= '_comov'
            xrange=[-12.,12.]
            yrange=[-10.,10.]           
        else:
            xrange=[-30.,30.]
            yrange=[-15.,15.]           
        xlabel=r'$Y\,(\mathrm{kpc})$'
        ylabel=r'$Z\,(\mathrm{kpc})$'
    elif proj.lower() == 'orbplane':
        basefilename+= 'orbplane_'
        moviefilename+= '_orbplane'
        x= numpy.zeros_like(data[:,1])
        y= numpy.zeros_like(data[:,2])
        nx= 10000
        nt= len(x)/nx
        diff= numpy.empty(nt)
        if includeorbit:
            npts= 201
            pot= potential.LogarithmicHaloPotential(normalize=1.,q=0.9)
            pts= numpy.linspace(0.,4.,npts)
            px= numpy.zeros(nt*(2*npts-1))
            py= numpy.zeros(nt*(2*npts-1))
        ii= 0
        #Calculate median angular momentum at t=0, use this to always go to the orbital plane
        Lx= numpy.median(data[ii*nx:(ii+1)*nx,2]*data[ii*nx:(ii+1)*nx,6]\
                             -data[ii*nx:(ii+1)*nx,3]*data[ii*nx:(ii+1)*nx,5])
        Ly= numpy.median(data[ii*nx:(ii+1)*nx,3]*data[ii*nx:(ii+1)*nx,4]\
                             -data[ii*nx:(ii+1)*nx,1]*data[ii*nx:(ii+1)*nx,6])
        Lz= numpy.median(data[ii*nx:(ii+1)*nx,1]*data[ii*nx:(ii+1)*nx,5]\
                             -data[ii*nx:(ii+1)*nx,2]*data[ii*nx:(ii+1)*nx,4])
        L= numpy.sqrt(Lx**2.+Ly**2.+Lz**2.)
        Lx/= L
        Ly/= L
        Lz/= L
        Txz= numpy.zeros((3,3))
        Tz= numpy.zeros((3,3))
        Txz[0,0]= Lx/numpy.sqrt(Lx**2.+Ly**2.)
        Txz[1,1]= Lx/numpy.sqrt(Lx**2.+Ly**2.)
        Txz[1,0]= Ly/numpy.sqrt(Lx**2.+Ly**2.)
        Txz[0,1]= -Ly/numpy.sqrt(Lx**2.+Ly**2.)
        Txz[2,2]= 1.
        Tz[0,0]= Lz
        Tz[1,1]= 1.
        Tz[2,2]= Lz
        Tz[2,0]= -numpy.sqrt(Lx**2.+Ly**2.)
        Tz[0,2]= numpy.sqrt(Lx**2.+Ly**2.)
        TL= numpy.dot(Tz,Txz)
        for ii in range(nt):
            if includeorbit:
                #Calculate progenitor orbit around this point
                pox= numpy.median(data[ii*nx:(ii+1)*nx,1])
                poy= numpy.median(data[ii*nx:(ii+1)*nx,3])
                poz= numpy.median(data[ii*nx:(ii+1)*nx,2])
                povx= numpy.median(data[ii*nx:(ii+1)*nx,4])
                povy= numpy.median(data[ii*nx:(ii+1)*nx,6])
                povz= numpy.median(data[ii*nx:(ii+1)*nx,5])
                pR,pphi,pZ= bovy_coords.rect_to_cyl(pox,poy,poz)
                pvR,pvT,pvZ= bovy_coords.rect_to_cyl_vec(povx,povy,povz,pR,
                                                         pphi,pZ,cyl=True)
                ppo= Orbit([pR/8.,pvR/220.,pvT/220.,pZ/8.,pvZ/220.,pphi])
                pno= Orbit([pR/8.,-pvR/220.,-pvT/220.,pZ/8.,-pvZ/220.,pphi])
                ppo.integrate(pts,pot)
                pno.integrate(pts,pot)
                pvec= numpy.zeros((3,npts*2-1))
                pvec[0,:npts-1]= pno.x(pts)[::-1][:-1]
                pvec[1,:npts-1]= pno.z(pts)[::-1][:-1]
                pvec[2,:npts-1]= pno.y(pts)[::-1][:-1]
                pvec[0,npts-1:]= ppo.x(pts)
                pvec[1,npts-1:]= ppo.z(pts)
                pvec[2,npts-1:]= ppo.y(pts)
                pvec*= 8.
            tvec= numpy.empty((3,nx))
            tvec[0,:]= data[ii*nx:(ii+1)*nx,1]
            tvec[1,:]= data[ii*nx:(ii+1)*nx,2]
            tvec[2,:]= data[ii*nx:(ii+1)*nx,3]
            x[ii*nx:(ii+1)*nx]= numpy.dot(TL,tvec)[0]
            y[ii*nx:(ii+1)*nx]= numpy.dot(TL,tvec)[1]
            if includeorbit:
                #Also rotate orbit
                px[ii*(2*npts-1):(ii+1)*(2*npts-1)]= numpy.dot(TL,pvec)[0] 
                py[ii*(2*npts-1):(ii+1)*(2*npts-1)]= numpy.dot(TL,pvec)[1] 
            #print numpy.std(numpy.dot(T,tvec)[2])
            #Rotate further to lign up between frames
            if not includeorbit:
                dist= numpy.sqrt((x[ii*nx:(ii+1)*nx]-numpy.median(x[ii*nx:(ii+1)*nx]))**2.\
                                     +(y[ii*nx:(ii+1)*nx]-numpy.median(y[ii*nx:(ii+1)*nx]))**2.)
                largedist= dist > (5.*numpy.std(dist))
                A= numpy.ones((numpy.sum(largedist),2))
                A[:,1]= x[ii*nx:(ii+1)*nx][largedist]
                m= numpy.dot(linalg.inv(numpy.dot(A.T,A)),
                             numpy.dot(A.T,y[ii*nx:(ii+1)*nx][largedist]))[1]
                sint= m/numpy.sqrt(1.+m**2.)
                cost= 1./numpy.sqrt(1.+m**2.)
            else:
                if False:
                    A= numpy.ones((51,2))
                    A[:25,1]= px[ii*(2*npts-1)+npts-50:ii*(2*npts-1)+npts-25]
                    A[25:,1]= px[ii*(2*npts-1)+npts+25:ii*(2*npts-1)+npts+51]
                    m= numpy.dot(linalg.inv(numpy.dot(A.T,A)),
                                 numpy.dot(A.T,py[ii*(2*npts-1)+npts-25:ii*(2*npts-1)+npts+26]))[1]
                    sint= m/numpy.sqrt(1.+m**2.)
                    cost= 1./numpy.sqrt(1.+m**2.)
                else:
                    sint= py[ii*(2*npts-1)+npts+1]/numpy.sqrt(px[ii*(2*npts-1)+npts+1]**2.+py[ii*(2*npts-1)+npts+1]**2.)
                    cost= px[ii*(2*npts-1)+npts+1]/numpy.sqrt(px[ii*(2*npts-1)+npts+1]**2.+py[ii*(2*npts-1)+npts+1]**2.)
            tvec= numpy.empty((2,nx))
            tvec[0,:]= x[ii*nx:(ii+1)*nx]-numpy.median(x[ii*nx:(ii+1)*nx])
            tvec[1,:]= y[ii*nx:(ii+1)*nx]-numpy.median(y[ii*nx:(ii+1)*nx])
            T= numpy.zeros((2,2))
            T[0,0]= cost
            T[1,1]= cost
            T[1,0]= -sint
            T[0,1]= sint
            tvec= numpy.dot(T,tvec)
            if includeorbit:
                #Also rotate orbit
                pvec2= numpy.empty((2,(2*npts-1)))
                pvec2[0,:]= px[ii*(2*npts-1):(ii+1)*(2*npts-1)]-numpy.median(x[ii*nx:(ii+1)*nx])
                pvec2[1,:]= py[ii*(2*npts-1):(ii+1)*(2*npts-1)]-numpy.median(y[ii*nx:(ii+1)*nx])
                pvec2= numpy.dot(T,pvec2)
            T[0,0]= numpy.cos(45./180.*numpy.pi)
            T[1,1]= numpy.cos(45./180.*numpy.pi)
            T[1,0]= numpy.sin(45./180.*numpy.pi)
            T[0,1]= -numpy.sin(45./180.*numpy.pi)
            tvec= numpy.dot(T,tvec)
            x[ii*nx:(ii+1)*nx]= tvec[0,:]
            y[ii*nx:(ii+1)*nx]= tvec[1,:]
            if includeorbit:
                pvec2= numpy.dot(T,pvec2)
                px[ii*(2*npts-1):(ii+1)*(2*npts-1)]= pvec2[0,:]
                py[ii*(2*npts-1):(ii+1)*(2*npts-1)]= pvec2[1,:]
        if comov:
            basefilename+= 'comov_'
            moviefilename+= '_comov'
            xrange=[-13.,13.]
            yrange=[-13.,13.]           
        else:
            xrange=[-30.,30.]
            yrange=[-15.,15.]           
        xlabel=r'$X_{\mathrm{orb}}\,(\mathrm{kpc})$'
        ylabel=r'$Y_{\mathrm{orb}}\,(\mathrm{kpc})$'
    if not skippng:
        nx= 10000
        nt= len(x)/nx
        for ii in range(nt):
            plotx= x[ii*nx:(ii+1)*nx]
            ploty= y[ii*nx:(ii+1)*nx]
            if comov:
                plotx-= numpy.median(plotx)
                ploty-= numpy.median(ploty)
            bovy_plot.bovy_print()
            bovy_plot.bovy_plot(plotx,ploty,'k.',ms=2.,
                                xlabel=xlabel,
                                ylabel=ylabel,
                                xrange=xrange,
                                yrange=yrange)
            if not comov:
                bovy_plot.bovy_plot(numpy.median(plotx),numpy.median(ploty),
                                    'bo',mec='none',overplot=True)
            if includeorbit:
                bovy_plot.bovy_plot(px[ii*(2*npts-1):(ii+1)*(2*npts-1)],
                                    py[ii*(2*npts-1):(ii+1)*(2*npts-1)],
                                    'b-',
                                    overplot=True)
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
                               moviefilename+'.mpg'])
    except subprocess.CalledProcessError:
        print "'ffmpeg' failed"
    return None                           

if __name__ == '__main__':
    make_sim_movie(proj=sys.argv[1],comov=len(sys.argv) > 2)
