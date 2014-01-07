import sys
import os, os.path
import copy
import glob
import subprocess
import numpy
from numpy import linalg
from galpy.util import bovy_plot, bovy_coords, bovy_conversion
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

def make_sim_movie_aa(proj='aaarazap',comov=False,
                      debris=False,
                      skippng=False,
                      includeorbit=True,aas=False):
    #Directories
    if aas:
        snapaadir= 'snaps_aas/'
        aastr= 'aas'
        progfile= 'gd1_evol_hitres_progaas.dat'
    else:
        snapaadir= 'snaps_aai/'
        aastr= 'aai'
        progfile= 'gd1_evol_hitres_progaai.dat'
    savedirpng= './movies/gd1_aa/pngs/'
    basefilename= 'gd1_evol_%s_' % aastr
    moviefilename= 'gd1_evol_%s' % aastr
    if includeorbit: #Load actions etc. for progenitor
        progaa= numpy.loadtxt(progfile,delimiter=',')
    if proj.lower() == 'aaarazap':
        basefilename+= 'arazap_'
        moviefilename+= '_arazap'
        xlabel=r'$\theta_R$'
        ylabel=r'$\theta_Z$'
        if comov:
            xrange=[numpy.pi-1.,numpy.pi+1.]
            yrange=[numpy.pi-1.,numpy.pi+1.]
            zrange=[numpy.pi-.5,numpy.pi+.5]
        else:
            xrange=[-0.5,2.*numpy.pi+0.5]
            yrange=[-0.5,2.*numpy.pi+0.5]
            zrange=[-0.5,2.*numpy.pi+0.5]           
    elif proj.lower() == 'aajrjzlz':
        basefilename+= 'jrjzlz_'
        moviefilename+= '_jrjzlz'
        xlabel=r'$J_R\,(220\,\mathrm{km\,s}^{-1}\,\mathrm{kpc})$'
        ylabel=r'$J_Z\,(220\,\mathrm{km\,s}^{-1}\,\mathrm{kpc})$'
        if 'aai' in snapaadir:
            xrange=[1.1,1.5]
            yrange=[3.85,4.3]
        elif 'aas' in snapaadir:
            xrange=[1.1,1.5]
            yrange=[3.85,4.3]
    if debris:
        basefilename+= 'debris_'
        moviefilename+= '_debris'
    if not skippng:
        nx= 10000
        nt= len(glob.glob(os.path.join(snapaadir,
                                       'gd1_evol_hitres_aa_*.dat')))
        if debris:
            #Load final snapshot first, determine which stars are debris
            data= numpy.loadtxt(os.path.join(snapaadir,
                                             'gd1_evol_hitres_aa_%s.dat' % str(1312).zfill(5)),
                                delimiter=',')
            thetar= data[:,6]
            thetar= (numpy.pi+(thetar-numpy.median(thetar))) % (2.*numpy.pi)
            debrisIndx= numpy.fabs(thetar-numpy.pi) > (5.*numpy.median(numpy.fabs(thetar-numpy.median(thetar))))
        for ii in range(nt):
            #Read data
            data= numpy.loadtxt(os.path.join(snapaadir,
                                       'gd1_evol_hitres_aa_%s.dat' % str(ii).zfill(5)),
                                delimiter=',')
            if debris:
                thetar= data[:,6]
                thetar= (numpy.pi+(thetar-numpy.median(thetar))) % (2.*numpy.pi)
                indx= numpy.fabs(thetar-numpy.pi) > (5.*numpy.median(numpy.fabs(thetar-numpy.median(thetar))))
                indx*= debrisIndx
                data= data[indx,:]
                if numpy.sum(indx) == 0:
                    data= -1000.+numpy.random.uniform(size=(2,9))*50.
            if proj.lower() == 'aaarazap':
                plotx= data[:,6]
                ploty= data[:,8]
                plotz= data[:,7]
                if comov:
                    plotx= (numpy.pi+(plotx-numpy.median(plotx))) % (2.*numpy.pi)
                    ploty= (numpy.pi+(ploty-numpy.median(ploty))) % (2.*numpy.pi)
                    plotz= (numpy.pi+(plotz-numpy.median(plotz))) % (2.*numpy.pi)
                else:
                    plotx= plotx % (2.*numpy.pi)
                    ploty= ploty % (2.*numpy.pi)
                    plotz= plotz % (2.*numpy.pi)
            elif proj.lower() == 'aajrjzlz':
                plotx= data[:,0]*8.
                ploty= data[:,2]*8.
                plotz= data[:,1]*8.
                if (numpy.amax(plotz)-numpy.amin(plotz)) <= 0.1:
                    zrange=[numpy.amin(plotz),numpy.amax(plotz)]
                else:
                    zrange=[numpy.amin(plotz)+0.05,numpy.amax(plotz)-0.05]
            bovy_plot.bovy_print()
            bovy_plot.bovy_plot(plotx,ploty,c=plotz,scatter=True,
                                edgecolor='none',s=2.,
                                xlabel=xlabel,
                                ylabel=ylabel,
                                xrange=xrange,
                                yrange=yrange,
                                crange=zrange,
                                vmin=zrange[0],vmax=zrange[1],zorder=2)
            if includeorbit:
                if proj.lower() == 'aaarazap':
                    px= progaa[ii,6]
                    py= progaa[ii,8]
                    if comov:
                       #plot frequency line
                        xs= numpy.array(xrange)
                        ys= (xs-numpy.pi)*progaa[ii,5]/progaa[ii,3]+numpy.pi
                        bovy_plot.bovy_plot(xs,ys,'k--',overplot=True,
                                            zorder=0)
                    else:
                        px = px % (2.*numpy.pi)
                        py = py % (2.*numpy.pi)
                        bovy_plot.bovy_plot(px,py,'ko',overplot=True)
                elif proj.lower() == 'aajrjzlz':
                    bovy_plot.bovy_plot(8.*progaa[ii,0],8.*progaa[ii,2],'ko',
                                        overplot=True,zorder=3)
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

def make_sim_movie_oparapar(proj='aaarazap',
                            skippng=True,#False,
                            aas=False):
    #Directories
    if aas:
        snapaadir= 'snaps_aas/'
        aastr= 'aas'
    else:
        snapaadir= 'snaps_aai/'
        aastr= 'aai'
    savedirpng= './movies/gd1_aa/pngs/'
    basefilename= 'gd1_evol_%s_' % aastr
    moviefilename= 'gd1_evol_%s' % aastr
    if True:
        basefilename+= 'oparapar_'
        moviefilename+= '_oparapar'
        xlabel=r'$|\theta_\parallel|$'
        ylabel=r'$|\Omega_\parallel|$'
        xrange=[0.,1.3]
        yrange=[0.1,0.3]
        zrange=[0.,4.5]
    if not skippng:
        nt= len(glob.glob(os.path.join(snapaadir,
                                       'gd1_evol_hitres_aa_*.dat')))
        #Load final snapshot first, determine which stars are debris and when they were stripped
        data= numpy.loadtxt(os.path.join(snapaadir,
                                         'gd1_evol_hitres_aa_%s.dat' % str(1312).zfill(5)),
                            delimiter=',')
        thetar= data[:,6]
        thetar= (numpy.pi+(thetar-numpy.median(thetar))) % (2.*numpy.pi)
        debrisIndx= numpy.fabs(thetar-numpy.pi) > (5.*numpy.median(numpy.fabs(thetar-numpy.median(thetar))))
        thetar= thetar[debrisIndx]
        #Calculate times at which stars were stripped, angles
        thetap= data[:,7]
        thetap= (numpy.pi+(thetap-numpy.median(thetap))) % (2.*numpy.pi)
        thetap= thetap[debrisIndx]
        thetaz= data[:,8]
        thetaz= (numpy.pi+(thetaz-numpy.median(thetaz))) % (2.*numpy.pi)
        thetaz= thetaz[debrisIndx]
        #center around 0 (instead of pi)
        thetar-= numpy.pi
        thetap-= numpy.pi
        thetaz-= numpy.pi
        #Frequencies
        Or= data[:,3]
        Op= data[:,4]
        Oz= data[:,5]
        dOr= Or[debrisIndx]-numpy.median(Or)
        dOp= Op[debrisIndx]-numpy.median(Op)
        dOz= Oz[debrisIndx]-numpy.median(Oz)
        #Times
        dangle= numpy.vstack((thetar,thetap,thetaz))
        dO= numpy.vstack((dOr,dOp,dOz))*bovy_conversion.freq_in_Gyr(220.,8.)
        dts= numpy.empty(data.shape[0])
        dts[debrisIndx]= numpy.sum(dO*dangle,axis=0)/numpy.sum(dO**2.,axis=0)
        #Direction in which the stream spreads
        dO4dir= copy.copy(dO)
        try:
            dO4dir[:,dO4dir[:,0] < 0.]*= -1.
        except IndexError:
            pass
        dOdir= numpy.median(dO4dir,axis=1)
        dOdir/= numpy.sqrt(numpy.sum(dOdir**2.))
        for ii in range(115,nt):#Skip the first 100, because nothing happens anyway
            #Read data
            data= numpy.loadtxt(os.path.join(snapaadir,
                                       'gd1_evol_hitres_aa_%s.dat' % str(ii).zfill(5)),
                                delimiter=',')
            if True:
                thetar= data[:,6]
                thetar= (numpy.pi+(thetar-numpy.median(thetar))) % (2.*numpy.pi)
                indx= numpy.fabs(thetar-numpy.pi) > (5.*numpy.median(numpy.fabs(thetar-numpy.median(thetar))))
                indx*= debrisIndx
                tdts= dts[indx]
                if numpy.sum(indx) == 0:
                    data= -1000.+numpy.random.uniform(size=(2,9))*50.
            if True:
                thetar= data[:,6]
                thetar= (numpy.pi+(thetar-numpy.median(thetar))) % (2.*numpy.pi)
                thetar= thetar[indx]
                thetap= data[:,7]
                thetap= (numpy.pi+(thetap-numpy.median(thetap))) % (2.*numpy.pi)
                thetap= thetap[indx]
                thetaz= data[:,8]
                thetaz= (numpy.pi+(thetaz-numpy.median(thetaz))) % (2.*numpy.pi)
                thetaz= thetaz[indx]
                #center around 0 (instead of pi)
                thetar-= numpy.pi
                thetap-= numpy.pi
                thetaz-= numpy.pi
                #Frequencies
                Or= data[:,3]
                Op= data[:,4]
                Oz= data[:,5]
                dOr= Or[indx]-numpy.median(Or)
                dOp= Op[indx]-numpy.median(Op)
                dOz= Oz[indx]-numpy.median(Oz)
                dO= numpy.vstack((dOr,dOp,dOz))*bovy_conversion.freq_in_Gyr(220.,8.)
                #Direction in which the stream spreads, taken from final snapshot
                #Times
                dangle= numpy.vstack((thetar,thetap,thetaz))
                plotx= numpy.fabs(numpy.dot(dangle.T,dOdir))
                ploty= numpy.fabs(numpy.dot(dO.T,dOdir))
                plotz= tdts
            bovy_plot.bovy_print()
            bovy_plot.bovy_plot(plotx,ploty,c=plotz,scatter=True,
                                edgecolor='none',s=9.5,
                                xlabel=xlabel,
                                ylabel=ylabel,
                                xrange=xrange,
                                yrange=yrange,
                                crange=zrange,
                                vmin=zrange[0],vmax=zrange[1],zorder=2)
            bovy_plot.bovy_end_print(os.path.join(savedirpng,basefilename+'%s.png' % str(ii-115).zfill(5)))
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
    if 'oparapar' in sys.argv[1].lower():
        make_sim_movie_oparapar(proj=sys.argv[1])
    elif 'aa' in sys.argv[1].lower():
        make_sim_movie_aa(proj=sys.argv[1],comov=len(sys.argv) > 2,
                          debris=len(sys.argv) > 3)
    else:
        make_sim_movie(proj=sys.argv[1],comov=len(sys.argv) > 2)
